"""Z-Image single-stream reward register (2-head) — architecture port of exp11.

``ZImageLatentRewardGridPoolNoPEMultiHeadModel`` mirrors
``SD3LatentRewardGridPoolNoPEMultiHeadModel`` but rides on the Z-Image single-stream DiT
via ``ZImageRewardBackbone``. Everything above the trunk is reused verbatim from
``latent_reward_grid`` (``LatentRewardGridHead``, ``_SpatialPool2d``, its ``_FiLMLayerAdapter``,
``_JointAttentionBlock``, ``score_mlps`` construction). The load-bearing change is
``_run_frozen_zimage_block`` (RMSNorm + 4-chunk tanh modulation + single unified sequence
with RoPE) replacing SD3's ``_run_frozen_sd3_block``.

FiLM temb fix (report §6.4, option A): Z-Image's ``adaln_input`` is 256-dim while the head's
FiLM adapters expect ``emb_dim = hidden_size = 3840``; a tiny trainable ``temb_proj``
(Linear 256 -> 3840) bridges the gap, leaving ``LatentRewardGridHead`` untouched.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import zimage_common

from .latent_reward_grid import LatentRewardGridHead, _SpatialPool2d
from .reward_token_dina_head import _normalize_layer_indices
from .zimage_backbone import ZImageRewardBackbone


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Vendored from ``ZSingleStreamAttnProcessor.__call__`` (view-as-complex, fp32 upcast).

    ``x_in``: ``[B, S, heads, head_dim]``; ``freqs_cis``: ``[B_or_1, S, head_dim//2]`` complex.
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageLatentRewardGridPoolNoPEMultiHeadModel(nn.Module):
    """exp13z: exp11 trunk (32 position-free reward tokens, frozen Z-Image side-stream,
    pooled visual context + FiLM, joint self/cross-attention head) with per-reward MLP heads.

    ``forward`` returns ``dict[head_name -> [B, group_size]]`` consumed by ``multihead_loss``.
    """

    def __init__(
        self,
        backbone: ZImageRewardBackbone,
        *,
        head_names: Iterable[str],
        head_hidden_factor: int = 2,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 1,
        num_transformer_layers: int | None = None,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = True,
        skip_attn2: bool = True,
        pool_factor: int = 4,
        num_reward_tokens: int = 32,
        disable_side_stream: bool = False,
        side_stream_ffn: bool = False,
        freeze_q_proj: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze_all()

        self.disable_side_stream = bool(disable_side_stream)
        self.side_stream_ffn = bool(side_stream_ffn)
        self.freeze_q_proj = bool(freeze_q_proj)

        self.visual_layers = _normalize_layer_indices(
            visual_layers, num_layers=self.backbone.num_layers, index_base=layer_index_base
        )
        self.text_layers = _normalize_layer_indices(
            text_layers, num_layers=self.backbone.num_layers, index_base=layer_index_base
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
        self.vis_h = int(vis_h)
        self.vis_w = int(vis_w)
        self.pool_factor = int(pool_factor)
        self.num_reward_tokens = int(num_reward_tokens)
        hidden_size = self.backbone.hidden_size  # 3840

        # Position-free learnable reward-token set (NoPE): no reward_pos_embed.
        self.reward_pos_embed = None
        self.reward_tokens = nn.Parameter(torch.empty(self.num_reward_tokens, hidden_size))
        self.reward_q_proj = nn.Linear(hidden_size, hidden_size)
        if self.freeze_q_proj or self.disable_side_stream:
            for param in self.reward_q_proj.parameters():
                param.requires_grad = False

        self.visual_pool = _SpatialPool2d(self.pool_factor)

        self.reward_head = LatentRewardGridHead(
            token_dim=hidden_size,
            query_dim=hidden_size,
            width=width,
            out_dim=1,
            n_visual_layers=len(self.visual_layers),
            n_text_layers=len(self.text_layers),
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
            skip_attn2=skip_attn2,
        )

        # FiLM temb bridge: adaLN input is min(dim, ADALN_EMBED_DIM) = 256-dim, while the head
        # FiLM adapters expect emb_dim = hidden_size. A tiny trainable Linear bridges the gap.
        self.adaln_embed_dim = min(hidden_size, 256)
        self.temb_proj = nn.Linear(self.adaln_embed_dim, hidden_size)

        self.head_names = list(head_names)
        self.num_heads = len(self.head_names)
        if self.num_heads == 0:
            raise ValueError("head_names must be non-empty for the multi-head model")
        self.head_hidden_factor = int(head_hidden_factor)

        trunk_width = self.reward_head.head.in_features
        hidden = max(1, trunk_width // self.head_hidden_factor)
        self.score_mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(trunk_width, hidden), nn.GELU(), nn.Linear(hidden, 1))
                for _ in self.head_names
            ]
        )
        # Drop the inherited single-scalar readout (unused here) so DDP sees no dead params.
        self.reward_head.head = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.reward_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.reward_q_proj.weight)
        nn.init.zeros_(self.reward_q_proj.bias)
        nn.init.xavier_uniform_(self.temb_proj.weight)
        nn.init.zeros_(self.temb_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.transformer.eval()
        return self

    # ------------------------------------------------------------------
    # Frozen Z-Image block (single unified sequence, RMSNorm + 4-chunk tanh mod, RoPE)
    # ------------------------------------------------------------------
    def _run_frozen_zimage_block(self, block, x, freqs_cis, adaln_input, attn_mask=None):
        heads = block.attention.heads  # 30
        attn = block.attention
        with torch.no_grad():
            mod = block.adaLN_modulation(adaln_input)  # [B, 4*dim]
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            h = block.attention_norm1(x) * scale_msa
            q = attn.to_q(h).unflatten(-1, (heads, -1))
            k = attn.to_k(h).unflatten(-1, (heads, -1))
            v = attn.to_v(h).unflatten(-1, (heads, -1))
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
            # attn_mask: bool [B, S], True = real token (native _prepare_sequence semantics).
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=None if attn_mask is None else attn_mask[:, None, None, :],
                dropout_p=0.0, is_causal=False,
            )
            o = o.transpose(1, 2).flatten(2, 3)
            o = attn.to_out[0](o)
            if len(attn.to_out) > 1:
                o = attn.to_out[1](o)
            x = x + gate_msa * block.attention_norm2(o)  # post-norm on attn output
            x = x + gate_mlp * block.ffn_norm2(block.feed_forward(block.ffn_norm1(x) * scale_mlp))
        return x, k, v  # k, v: [B, S, heads, hd] (RoPE already applied to k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _flatten_inputs(self, latents, prompt_embeds, timesteps):
        batch_size, group_size = latents.shape[:2]
        flat_latents = latents.reshape(batch_size * group_size, *latents.shape[2:])
        flat_prompt_embeds = (
            prompt_embeds.unsqueeze(1)
            .expand(-1, group_size, -1, -1)
            .reshape(batch_size * group_size, *prompt_embeds.shape[1:])
        )
        if timesteps.ndim == 1:
            flat_timesteps = timesteps.unsqueeze(1).expand(-1, group_size).reshape(-1)
        elif timesteps.ndim == 2:
            flat_timesteps = timesteps.reshape(-1)
        else:
            raise ValueError(f"Expected timesteps to have 1 or 2 dims, got {tuple(timesteps.shape)}")
        return batch_size, group_size, flat_latents, flat_prompt_embeds, flat_timesteps

    def forward(self, latents, *, prompt_embeds, pooled_prompt_embeds=None, timesteps, prompts=None):
        batch_size, group_size, flat_latents, flat_prompt_embeds, flat_timesteps = self._flatten_inputs(
            latents, prompt_embeds, timesteps
        )
        device = flat_latents.device
        model_dtype = next(self.backbone.transformer.parameters()).dtype

        # Continuous flow level u in [0,1] (guard for the eval constant path that passes discrete t).
        u = flat_timesteps.to(device=device, dtype=torch.float32)
        if u.numel() > 0 and float(u.max()) > 1.0:
            u = u / self.backbone.num_train_timesteps

        # fixv2: trim the cached full-padded [512, 2560] caption embeds to their real token
        # length (native pipeline behavior — encode_prompt trims by attention mask). The
        # frozen trunk then applies its own cap_pad_token / batch pad mask machinery.
        # prompts is one string per group; fall back to full length when absent.
        if prompts is not None:
            if len(prompts) != batch_size:
                raise ValueError(f"prompts has {len(prompts)} entries, expected batch_size={batch_size}")
            cap_lens = [
                min(zimage_common.qwen3_cap_len(p), flat_prompt_embeds.shape[1]) for p in prompts
            ]
            flat_cap_lens = [L for L in cap_lens for _ in range(group_size)]
        else:
            flat_cap_lens = [flat_prompt_embeds.shape[1]] * flat_prompt_embeds.shape[0]

        image_list = [flat_latents[i].to(dtype=model_dtype).unsqueeze(1) for i in range(flat_latents.shape[0])]
        cap_list = [
            flat_prompt_embeds[i, : flat_cap_lens[i]].to(dtype=model_dtype)
            for i in range(flat_prompt_embeds.shape[0])
        ]

        unified, unified_freqs, unified_mask, adaln_input, x_seqlens, cap_seqlens = self.backbone.build_unified_inputs(
            image_list, cap_list, u
        )
        n_img = int(x_seqlens[0])
        if any(int(s) != n_img for s in x_seqlens):
            raise ValueError(f"Expected uniform image seqlens, got {x_seqlens}")
        cap_seqlens = [int(c) for c in cap_seqlens]  # rounded to SEQ_MULTI_OF per sample
        n_cap_max = max(cap_seqlens)

        # Valid-caption mask for the head's text features (True where the unified row holds
        # a real/rounded cap token rather than batch padding).
        if unified_mask is not None:
            cap_valid = torch.zeros((unified.shape[0], n_cap_max), dtype=unified.dtype, device=device)
            for i, c_len in enumerate(cap_seqlens):
                cap_valid[i, :c_len] = 1.0
            cap_valid = cap_valid.unsqueeze(-1)
        else:
            cap_valid = None

        reward_states = self.reward_tokens.unsqueeze(0).expand(flat_latents.shape[0], -1, -1).to(dtype=model_dtype)
        reward_freqs = self.backbone.build_register_freqs(self.num_reward_tokens, cap_seqlens, device)

        # Side-stream key mask: reward tokens (always valid) + unified sequence.
        if unified_mask is not None:
            side_mask = torch.cat(
                [
                    torch.ones(
                        (unified.shape[0], self.num_reward_tokens), dtype=torch.bool, device=device
                    ),
                    unified_mask,
                ],
                dim=1,
            )[:, None, None, :]
        else:
            side_mask = None

        visual_set = set(self.visual_layers)
        text_set = set(self.text_layers)
        visual_by_layer: dict[int, torch.Tensor] = {}
        text_by_layer: dict[int, torch.Tensor] = {}

        for index, block in enumerate(self.backbone.transformer.layers):
            if index >= self.stop_at_layer:
                break
            unified, k_u, v_u = self._run_frozen_zimage_block(
                block, unified, unified_freqs, adaln_input, attn_mask=unified_mask
            )

            if index in visual_set:
                visual_by_layer[index] = unified[:, 0:n_img, :]
            if index in text_set:
                text_feat = unified[:, n_img : n_img + n_cap_max, :]
                if cap_valid is not None:
                    text_feat = text_feat * cap_valid
                text_by_layer[index] = text_feat

            if not self.disable_side_stream:
                heads = block.attention.heads
                attn = block.attention
                mod = block.adaLN_modulation(adaln_input)
                s_msa, g_msa, s_mlp, g_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                g_msa, g_mlp = g_msa.tanh(), g_mlp.tanh()
                s_msa, s_mlp = 1.0 + s_msa, 1.0 + s_mlp

                hr = block.attention_norm1(reward_states) * s_msa
                q_r = self.reward_q_proj(hr).unflatten(-1, (heads, -1))
                k_r = attn.to_k(hr).unflatten(-1, (heads, -1))
                v_r = attn.to_v(hr).unflatten(-1, (heads, -1))
                if attn.norm_q is not None:
                    q_r = attn.norm_q(q_r)
                if attn.norm_k is not None:
                    k_r = attn.norm_k(k_r)
                q_r = apply_rotary_emb(q_r, reward_freqs)
                k_r = apply_rotary_emb(k_r, reward_freqs)

                key = torch.cat([k_r, k_u], dim=1)  # [B, R+S, heads, hd]
                value = torch.cat([v_r, v_u], dim=1)
                o = F.scaled_dot_product_attention(
                    q_r.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                    attn_mask=side_mask, dropout_p=0.0, is_causal=False,
                )
                o = o.transpose(1, 2).flatten(2, 3)
                o = attn.to_out[0](o)
                if len(attn.to_out) > 1:
                    o = attn.to_out[1](o)
                reward_states = reward_states + g_msa * block.attention_norm2(o)  # gated post-norm

                if self.side_stream_ffn:
                    reward_states = reward_states + g_mlp * block.ffn_norm2(
                        block.feed_forward(block.ffn_norm1(reward_states) * s_mlp)
                    )

        visual_features = [visual_by_layer[i] for i in self.visual_layers]
        text_features = [text_by_layer[i] for i in self.text_layers]
        visual_features = [self.visual_pool(feat, self.vis_h, self.vis_w) for feat in visual_features]

        temb = self.temb_proj(adaln_input.to(dtype=self.temb_proj.weight.dtype))
        return self._compute_outputs(reward_states, visual_features, text_features, temb, batch_size, group_size)

    def _compute_outputs(self, reward_states, visual_features, text_features, temb, batch_size, group_size):
        feats = self.reward_head.features(reward_states, visual_features, text_features, temb)
        return {
            name: self.score_mlps[k](feats).mean(dim=1).squeeze(-1).reshape(batch_size, group_size)
            for k, name in enumerate(self.head_names)
        }

    # ------------------------------------------------------------------
    # Parameter groups / checkpoint
    # ------------------------------------------------------------------
    def parameter_groups(self, *, head_lr: float, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        del backbone_lr
        params = [self.reward_tokens]
        if not self.freeze_q_proj and not self.disable_side_stream:
            params += list(self.reward_q_proj.parameters())
        params += list(self.reward_head.parameters())
        params += list(self.temb_proj.parameters())
        params += list(self.score_mlps.parameters())
        return [{"params": params, "lr": head_lr}]

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "architecture": "zimage_latent_reward_grid_pool_nope_multihead",
            "visual_layers": self.visual_layers,
            "text_layers": self.text_layers,
            "layer_index_base": self.layer_index_base,
            "stop_at_layer": self.stop_at_layer,
            "num_reward_tokens": self.num_reward_tokens,
            "vis_h": self.vis_h,
            "vis_w": self.vis_w,
            "pool_factor": self.pool_factor,
            "disable_side_stream": self.disable_side_stream,
            "side_stream_ffn": self.side_stream_ffn,
            "freeze_q_proj": self.freeze_q_proj,
            "head_names": list(self.head_names),
            "head_hidden_factor": self.head_hidden_factor,
            "reward_tokens": self.reward_tokens.detach().cpu(),
            "reward_q_proj": self.reward_q_proj.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "temb_proj": self.temb_proj.state_dict(),
            "score_mlps": self.score_mlps.state_dict(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.reward_tokens.data.copy_(
            state["reward_tokens"].to(device=self.reward_tokens.device, dtype=self.reward_tokens.dtype)
        )
        self.reward_q_proj.load_state_dict(state["reward_q_proj"])
        self.reward_head.load_state_dict(state["reward_head"])
        if "temb_proj" in state:
            self.temb_proj.load_state_dict(state["temb_proj"])
        if "score_mlps" in state:
            self.score_mlps.load_state_dict(state["score_mlps"])

