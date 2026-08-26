from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SD3RewardBackbone


def _normalize_layer_indices(
    layer_indices: Iterable[int],
    *,
    num_layers: int,
    index_base: int = 0,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw_index in layer_indices:
        index = int(raw_index) - int(index_base)
        if index < 0 or index >= num_layers:
            raise ValueError(
                f"Layer index {raw_index} with index_base={index_base} resolves to {index}, "
                f"outside [0, {num_layers})"
            )
        normalized.append(index)
    if not normalized:
        raise ValueError("At least one feature layer is required")
    return tuple(dict.fromkeys(normalized))


class _FiLMLayerAdapter(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_proj_in: bool = False,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim) if use_proj_in or in_dim != hidden_dim else nn.Identity()
        self.layer_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        gamma, beta = self.cond_mlp(temb).chunk(2, dim=-1)
        hidden_states = hidden_states * (1.0 + gamma[:, None]) + beta[:, None]
        hidden_states = hidden_states + self.layer_embed
        return self.proj(hidden_states)


class _CrossAttentionBlockVGated(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 8,
        use_text: bool = True,
        dropout: float = 0.0,
        use_v_gating: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.use_text = bool(use_text)
        self.use_v_gating = bool(use_v_gating)

        self.norm_q = nn.RMSNorm(dim)
        self.norm_v = nn.RMSNorm(dim)
        self.norm_t = nn.RMSNorm(dim) if self.use_text else None

        self.to_q = nn.Linear(dim, dim)
        self.to_k_vis = nn.Linear(dim, dim)
        self.to_v_vis = nn.Linear(dim, dim)
        self.to_k_text = nn.Linear(dim, dim) if self.use_text else None
        self.to_v_text = nn.Linear(dim, dim) if self.use_text else None
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

        self.gate_vis = None
        self.gate_text = None
        if self.use_v_gating:
            self.gate_vis = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.gate_vis[-1].weight)
            nn.init.zeros_(self.gate_vis[-1].bias)
            if self.use_text:
                self.gate_text = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
                nn.init.zeros_(self.gate_text[-1].weight)
                nn.init.zeros_(self.gate_text[-1].bias)

    def _shape(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _unshape(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            self._shape(q),
            self._shape(k),
            self._shape(v),
            dropout_p=0.0,
            is_causal=False,
        )
        return self._unshape(output)

    def forward(
        self,
        queries: torch.Tensor,
        context_visual: torch.Tensor,
        context_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = queries
        queries = self.norm_q(queries)
        context_visual = self.norm_v(context_visual)
        if self.norm_t is not None and context_text is not None:
            context_text = self.norm_t(context_text)

        q = self.to_q(queries)
        k_vis = self.to_k_vis(context_visual)
        v_vis = self.to_v_vis(context_visual)
        if self.gate_vis is not None:
            v_vis = v_vis * torch.sigmoid(self.gate_vis(context_visual))
        hidden_states = self._attend(q, k_vis, v_vis)

        if self.use_text and context_text is not None and self.to_k_text is not None and self.to_v_text is not None:
            k_text = self.to_k_text(context_text)
            v_text = self.to_v_text(context_text)
            if self.gate_text is not None:
                v_text = v_text * torch.sigmoid(self.gate_text(context_text))
            hidden_states = hidden_states + self._attend(q, k_text, v_text)

        return residual + self.to_out(hidden_states)


class RTDinaRewardHead(nn.Module):
    """DiNa-style reward head with reward/query tokens over frozen SD3 states."""

    def __init__(
        self,
        *,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_visual_layers: int,
        n_text_layers: int,
        num_reward_tokens: int = 4,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
    ):
        super().__init__()
        if width == -1:
            width = token_dim
        if width % 4 != 0:
            raise ValueError(f"Reward-head width must be divisible by 4, got {width}")
        if n_visual_layers <= 0:
            raise ValueError("At least one visual layer is required")
        if num_reward_tokens <= 0:
            raise ValueError("`num_reward_tokens` must be positive")

        feature_out_dim = width // 4
        self.layer_adapters_visual = nn.ModuleList(
            [
                _FiLMLayerAdapter(
                    in_dim=token_dim,
                    emb_dim=token_dim,
                    hidden_dim=width,
                    output_dim=feature_out_dim,
                    use_proj_in=use_proj_in,
                )
                for _ in range(n_visual_layers)
            ]
        )
        self.layer_adapters_text = nn.ModuleList(
            [
                _FiLMLayerAdapter(
                    in_dim=token_dim,
                    emb_dim=token_dim,
                    hidden_dim=width,
                    output_dim=feature_out_dim,
                    use_proj_in=use_proj_in,
                )
                for _ in range(n_text_layers)
            ]
        )
        self.agg_visual = nn.Linear(n_visual_layers * feature_out_dim, width)
        self.agg_text = nn.Linear(n_text_layers * feature_out_dim, width) if n_text_layers > 0 else None
        self.reward_tokens = nn.Parameter(torch.randn(1, num_reward_tokens, width) * 0.02)

        self.attn1 = _CrossAttentionBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_text=n_text_layers > 0,
            dropout=dropout,
            use_v_gating=True,
        )
        self.attn2 = _CrossAttentionBlockVGated(
            dim=width,
            num_heads=num_attn_heads,
            use_text=False,
            dropout=dropout,
            use_v_gating=False,
        )
        self.norm_ff = nn.RMSNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(width, out_dim)

    def _build_context_tokens(
        self,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if len(visual_features) != len(self.layer_adapters_visual):
            raise ValueError(
                f"Expected {len(self.layer_adapters_visual)} visual features, got {len(visual_features)}"
            )
        visual_tokens = [
            adapter(feature, temb)
            for adapter, feature in zip(self.layer_adapters_visual, visual_features)
        ]
        visual_out = self.agg_visual(torch.cat(visual_tokens, dim=-1))

        if not self.layer_adapters_text:
            return visual_out, None
        if text_features is None or len(text_features) != len(self.layer_adapters_text):
            raise ValueError(f"Expected {len(self.layer_adapters_text)} text features")
        text_tokens = [
            adapter(feature, temb)
            for adapter, feature in zip(self.layer_adapters_text, text_features)
        ]
        text_out = self.agg_text(torch.cat(text_tokens, dim=-1)) if self.agg_text is not None else None
        return visual_out, text_out

    def _score_context(self, visual_out: torch.Tensor, text_out: torch.Tensor | None) -> torch.Tensor:
        queries = self.reward_tokens.expand(visual_out.shape[0], -1, -1)
        queries = self.attn1(queries, visual_out, text_out)
        queries = self.attn2(queries, visual_out, None)
        queries = queries + self.ff(self.norm_ff(queries))
        return self.head(queries).mean(dim=1).squeeze(-1)

    def forward(
        self,
        *,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        visual_out, text_out = self._build_context_tokens(visual_features, text_features, temb)
        return self._score_context(visual_out, text_out)

    def forward_ensemble(
        self,
        *,
        visual_features_per_noise: list[list[torch.Tensor]],
        text_features_per_noise: list[list[torch.Tensor]] | None,
        temb_per_noise: torch.Tensor,
    ) -> torch.Tensor:
        scores: list[torch.Tensor] = []
        for noise_index, visual_features in enumerate(visual_features_per_noise):
            text_features = text_features_per_noise[noise_index] if text_features_per_noise is not None else None
            scores.append(
                self.forward(
                    visual_features=visual_features,
                    text_features=text_features,
                    temb=temb_per_noise[:, noise_index],
                )
            )
        return torch.stack(scores, dim=0).mean(dim=0)


class SD3RewardTokenDinaHeadModel(nn.Module):
    """Frozen-DiT RT-DiNa++ scorer.

    The SD3 transformer is a read-only feature extractor. Trainable parameters
    are restricted to the DiNa-style reward readout head.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        width: int = -1,
        out_dim: int = 1,
        num_reward_tokens: int = 4,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze_all()

        self.visual_layers = _normalize_layer_indices(
            visual_layers,
            num_layers=self.backbone.num_layers,
            index_base=layer_index_base,
        )
        self.text_layers = _normalize_layer_indices(
            text_layers,
            num_layers=self.backbone.num_layers,
            index_base=layer_index_base,
        )
        self.stop_at_layer = max(max(self.visual_layers), max(self.text_layers)) + 1
        if num_transformer_layers is not None:
            requested_stop = int(num_transformer_layers)
            if requested_stop < self.stop_at_layer:
                raise ValueError(
                    f"num_transformer_layers={requested_stop} cannot cover selected layers; "
                    f"need at least {self.stop_at_layer}"
                )
            self.stop_at_layer = requested_stop

        self.layer_index_base = int(layer_index_base)
        self.reward_head = RTDinaRewardHead(
            token_dim=self.backbone.hidden_size,
            width=width,
            out_dim=out_dim,
            n_visual_layers=len(self.visual_layers),
            n_text_layers=len(self.text_layers),
            num_reward_tokens=num_reward_tokens,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
        )
        self.assert_trainable_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.transformer.eval()
        return self

    def assert_trainable_parameters(self) -> None:
        forbidden = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("reward_head.")
        ]
        if forbidden:
            raise RuntimeError(
                "RT-DiNa++ requires frozen SD3/DiT denoise parameters. "
                f"Unexpected trainable parameters: {forbidden[:20]}"
            )
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("RT-DiNa++ has no trainable reward-head parameters")

    def _extract_features(
        self,
        flat_latents: torch.Tensor,
        *,
        flat_prompt_embeds: torch.Tensor,
        flat_pooled_prompt_embeds: torch.Tensor,
        flat_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        device = flat_latents.device
        model_dtype = next(self.backbone.transformer.parameters()).dtype
        with torch.no_grad():
            hidden_states = self.backbone.transformer.pos_embed(flat_latents.to(dtype=model_dtype))
            temb = self.backbone.transformer.time_text_embed(
                flat_timesteps.to(device=device),
                flat_pooled_prompt_embeds.to(device=device, dtype=model_dtype),
            )
            encoder_hidden_states = self.backbone.transformer.context_embedder(
                flat_prompt_embeds.to(device=device, dtype=model_dtype)
            )

            visual_features_by_layer: dict[int, torch.Tensor] = {}
            text_features_by_layer: dict[int, torch.Tensor] = {}
            visual_set = set(self.visual_layers)
            text_set = set(self.text_layers)
            for index, block in enumerate(self.backbone.transformer.transformer_blocks):
                if index >= self.stop_at_layer:
                    break
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=None,
                )
                if index in visual_set:
                    visual_features_by_layer[index] = hidden_states
                if index in text_set:
                    text_features_by_layer[index] = encoder_hidden_states

        visual_features = [visual_features_by_layer[index] for index in self.visual_layers]
        text_features = [text_features_by_layer[index] for index in self.text_layers]
        return temb, visual_features, text_features

    def _flatten_inputs(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, group_size = latents.shape[:2]
        flat_latents = latents.reshape(batch_size * group_size, *latents.shape[2:])
        flat_prompt_embeds = (
            prompt_embeds.unsqueeze(1)
            .expand(-1, group_size, -1, -1)
            .reshape(batch_size * group_size, *prompt_embeds.shape[1:])
        )
        flat_pooled_prompt_embeds = (
            pooled_prompt_embeds.unsqueeze(1)
            .expand(-1, group_size, -1)
            .reshape(batch_size * group_size, pooled_prompt_embeds.shape[-1])
        )
        if timesteps.ndim == 1:
            flat_timesteps = timesteps.unsqueeze(1).expand(-1, group_size).reshape(-1)
        elif timesteps.ndim == 2:
            flat_timesteps = timesteps.reshape(-1)
        else:
            raise ValueError(f"Expected timesteps to have 1 or 2 dims, got {tuple(timesteps.shape)}")
        return batch_size, group_size, flat_latents, flat_prompt_embeds, flat_pooled_prompt_embeds, flat_timesteps

    def forward(
        self,
        latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            group_size,
            flat_latents,
            flat_prompt_embeds,
            flat_pooled_prompt_embeds,
            flat_timesteps,
        ) = self._flatten_inputs(latents, prompt_embeds, pooled_prompt_embeds, timesteps)
        temb, visual_features, text_features = self._extract_features(
            flat_latents,
            flat_prompt_embeds=flat_prompt_embeds,
            flat_pooled_prompt_embeds=flat_pooled_prompt_embeds,
            flat_timesteps=flat_timesteps,
        )
        scores = self.reward_head(
            visual_features=visual_features,
            text_features=text_features,
            temb=temb,
        )
        return scores.reshape(batch_size, group_size)

    def forward_ensemble(
        self,
        latents_per_noise: list[torch.Tensor],
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps_per_noise: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(latents_per_noise) != len(timesteps_per_noise):
            raise ValueError("latents_per_noise and timesteps_per_noise must have the same length")
        if not latents_per_noise:
            raise ValueError("At least one noisy latent tensor is required")

        batch_size = latents_per_noise[0].shape[0]
        group_size = latents_per_noise[0].shape[1]
        temb_list: list[torch.Tensor] = []
        visual_per_noise: list[list[torch.Tensor]] = []
        text_per_noise: list[list[torch.Tensor]] = []
        for latents, timesteps in zip(latents_per_noise, timesteps_per_noise):
            (
                cur_batch_size,
                cur_group_size,
                flat_latents,
                flat_prompt_embeds,
                flat_pooled_prompt_embeds,
                flat_timesteps,
            ) = self._flatten_inputs(latents, prompt_embeds, pooled_prompt_embeds, timesteps)
            if cur_batch_size != batch_size or cur_group_size != group_size:
                raise ValueError("All ensemble latent tensors must have matching batch/group shape")
            temb, visual_features, text_features = self._extract_features(
                flat_latents,
                flat_prompt_embeds=flat_prompt_embeds,
                flat_pooled_prompt_embeds=flat_pooled_prompt_embeds,
                flat_timesteps=flat_timesteps,
            )
            temb_list.append(temb)
            visual_per_noise.append(visual_features)
            text_per_noise.append(text_features)

        scores = self.reward_head.forward_ensemble(
            visual_features_per_noise=visual_per_noise,
            text_features_per_noise=text_per_noise,
            temb_per_noise=torch.stack(temb_list, dim=1),
        )
        return scores.reshape(batch_size, group_size)

    def parameter_groups(self, *, head_lr: float, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        del backbone_lr
        self.assert_trainable_parameters()
        return [{"params": list(self.reward_head.parameters()), "lr": head_lr}]

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "architecture": "rt_dina_head",
            "visual_layers": self.visual_layers,
            "text_layers": self.text_layers,
            "layer_index_base": self.layer_index_base,
            "stop_at_layer": self.stop_at_layer,
            "reward_head": self.reward_head.state_dict(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.reward_head.load_state_dict(state["reward_head"])
        self.assert_trainable_parameters()
