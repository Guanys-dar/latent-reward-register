"""FLUX.1-dev reward register (2-head) — architecture port of exp11.

``FluxLatentRewardGridPoolNoPEMultiHeadModel`` rides on the FLUX MMDiT via
``FluxRewardBackbone``. Everything above the trunk is reused from
the SD3 register (``LatentRewardGridHead``, ``_SpatialPool2d``, ``score_mlps``).
The FLUX-specific details are:

  * Frozen trunk = 19 double-stream blocks (separate img/txt streams, joint attention)
    then single-stream blocks (concat ``[txt, img]``), run by
    ``_run_frozen_flux_double_block`` / ``_run_frozen_flux_single_block`` — faithful
    re-implementations of ``FluxTransformerBlock.forward`` /
    ``FluxSingleTransformerBlock.forward`` (vendored diffusers) that additionally
    return the post-norm post-rotary K/V for the register side-stream.
    ``flux_parity_check.py`` asserts bit-level agreement with the stock forward.
  * Register NoPE: FLUX applies RoPE inside attention; rotation at position 0 is the
    identity, so the 32 reward tokens simply get NO rotary on their q/k (equivalent to
    ids = zeros — they sit positionally where the text tokens sit).
  * Timestep: u = sigma passed straight through (NO ``1 - u``; see flux_backbone.py).
  * temb is 3072-dim (= hidden_size): ``temb_proj`` is a same-dim trainable bridge
    into the head's FiLM adapters.
  * No caption masking / trimming: stock Flux attends T5 padding; the full 512-token
    text stream is the faithful text feature.
  * ``side_stream_ffn`` applies to double blocks only — in single blocks the MLP is
    fused with attention through ``proj_out`` (inseparable), so the register update
    there always includes it.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import apply_rotary_emb

from .flux_backbone import FluxRewardBackbone
from .sd3 import LatentRewardGridHead, _SpatialPool2d
from .attention import _normalize_layer_indices
from .gradmode import frozen_trunk_context


def run_flux_double_block(block, img, txt, temb, rotary):
    """``FluxTransformerBlock.forward`` re-implementation returning
    ``(txt, img, k_full, v_full)``; ``k_full``/``v_full`` are the post-qk-norm,
    post-rotary K/V over the ``[txt, img]`` sequence — exactly the tensors the frozen
    attention consumes. Grad-agnostic: the training model wraps this in ``no_grad``;
    the guided-sampling path (``flux_register_head_scores_grad``) calls it with grad
    enabled so d(score)/d(latent) flows through the frozen weights."""
    attn = block.attn
    heads = attn.heads
    norm_img, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(img, emb=temb)
    norm_txt, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(txt, emb=temb)

    q = attn.norm_q(attn.to_q(norm_img).unflatten(-1, (heads, -1)))
    k = attn.norm_k(attn.to_k(norm_img).unflatten(-1, (heads, -1)))
    v = attn.to_v(norm_img).unflatten(-1, (heads, -1))
    eq = attn.norm_added_q(attn.add_q_proj(norm_txt).unflatten(-1, (heads, -1)))
    ek = attn.norm_added_k(attn.add_k_proj(norm_txt).unflatten(-1, (heads, -1)))
    ev = attn.add_v_proj(norm_txt).unflatten(-1, (heads, -1))

    q_full = torch.cat([eq, q], dim=1)  # [B, L+N, heads, hd] — txt first
    k_full = torch.cat([ek, k], dim=1)
    v_full = torch.cat([ev, v], dim=1)
    q_full = apply_rotary_emb(q_full, rotary, sequence_dim=1)
    k_full = apply_rotary_emb(k_full, rotary, sequence_dim=1)

    o = F.scaled_dot_product_attention(
        q_full.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    )
    o = o.transpose(1, 2).flatten(2, 3).to(q_full.dtype)
    txt_len = txt.shape[1]
    txt_attn, img_attn = o[:, :txt_len], o[:, txt_len:]
    img_attn = attn.to_out[1](attn.to_out[0](img_attn))
    txt_attn = attn.to_add_out(txt_attn)

    img = img + gate_msa.unsqueeze(1) * img_attn
    norm_img2 = block.norm2(img) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    img = img + gate_mlp.unsqueeze(1) * block.ff(norm_img2)

    txt = txt + c_gate_msa.unsqueeze(1) * txt_attn
    norm_txt2 = block.norm2_context(txt) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    txt = txt + c_gate_mlp.unsqueeze(1) * block.ff_context(norm_txt2)
    return txt, img, k_full, v_full


def run_flux_single_block(block, img, txt, temb, rotary):
    """``FluxSingleTransformerBlock.forward`` re-implementation returning
    ``(txt, img, k_full, v_full)`` (K/V post-norm/rotary over ``[txt, img]``).
    Grad-agnostic — see ``run_flux_double_block``."""
    attn = block.attn
    heads = attn.heads
    txt_len = txt.shape[1]
    h = torch.cat([txt, img], dim=1)
    residual = h
    norm_h, gate = block.norm(h, emb=temb)
    mlp = block.act_mlp(block.proj_mlp(norm_h))

    q = attn.norm_q(attn.to_q(norm_h).unflatten(-1, (heads, -1)))
    k_full = attn.norm_k(attn.to_k(norm_h).unflatten(-1, (heads, -1)))
    v_full = attn.to_v(norm_h).unflatten(-1, (heads, -1))
    q = apply_rotary_emb(q, rotary, sequence_dim=1)
    k_full = apply_rotary_emb(k_full, rotary, sequence_dim=1)

    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    )
    o = o.transpose(1, 2).flatten(2, 3).to(q.dtype)
    h = residual + gate.unsqueeze(1) * block.proj_out(torch.cat([o, mlp], dim=2))
    txt, img = h[:, :txt_len], h[:, txt_len:]
    return txt, img, k_full, v_full


class FluxLatentRewardGridPoolNoPEMultiHeadModel(nn.Module):
    """exp18f: exp11 trunk (32 position-free reward tokens, frozen FLUX side-stream,
    pooled visual context + FiLM, joint self/cross-attention head) with per-reward MLP heads.

    ``forward`` returns ``dict[head_name -> [B, group_size]]`` consumed by ``multihead_loss``.
    """

    def __init__(
        self,
        backbone: FluxRewardBackbone,
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
        hidden_size = self.backbone.hidden_size  # 3072

        # Position-free learnable reward-token set (NoPE): no reward_pos_embed, and no
        # rotary applied to the register q/k (identity rotation at position 0).
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

        # FiLM temb bridge: FLUX temb is already hidden_size-dim (3072); keep a trainable
        # Same-dim linear bridge into the head's FiLM interface.
        self.adaln_embed_dim = hidden_size
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

        # Parity/debug: when True, forward stashes tap snapshots in self.last_debug.
        self.debug_capture = False
        self.last_debug: dict[str, Any] | None = None

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
    # Frozen FLUX blocks (faithful re-implementations that also expose K/V)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_frozen_flux_double_block(block, img, txt, temb, rotary):
        """Frozen-trunk wrapper over ``run_flux_double_block``.

        Grad-recording is off unless the caller opts in via
        ``latent_gradient_enabled()``; trunk weights stay frozen either way.
        """
        with frozen_trunk_context():
            return run_flux_double_block(block, img, txt, temb, rotary)

    @staticmethod
    def _run_frozen_flux_single_block(block, img, txt, temb, rotary):
        """Frozen-trunk wrapper over ``run_flux_single_block``. See the double-block note."""
        with frozen_trunk_context():
            return run_flux_single_block(block, img, txt, temb, rotary)

    # ------------------------------------------------------------------
    # Register side-stream (one-way: register queries frozen K/V; frozen stream
    # never sees the register, matching the exp11 design)
    # ------------------------------------------------------------------
    def _side_stream_double(self, block, reward_states, temb, k_full, v_full):
        attn = block.attn
        heads = attn.heads
        norm_r, r_gate_msa, r_shift_mlp, r_scale_mlp, r_gate_mlp = block.norm1(reward_states, emb=temb)
        q_r = attn.norm_q(self.reward_q_proj(norm_r).unflatten(-1, (heads, -1)))
        k_r = attn.norm_k(attn.to_k(norm_r).unflatten(-1, (heads, -1)))
        v_r = attn.to_v(norm_r).unflatten(-1, (heads, -1))
        # NoPE: no rotary on register q/k (identity rotation at position 0).
        key = torch.cat([k_r, k_full], dim=1)
        value = torch.cat([v_r, v_full], dim=1)
        o = F.scaled_dot_product_attention(
            q_r.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        )
        o = o.transpose(1, 2).flatten(2, 3).to(q_r.dtype)
        o = attn.to_out[1](attn.to_out[0](o))
        reward_states = reward_states + r_gate_msa.unsqueeze(1) * o
        if self.side_stream_ffn:
            norm_r2 = block.norm2(reward_states) * (1 + r_scale_mlp[:, None]) + r_shift_mlp[:, None]
            reward_states = reward_states + r_gate_mlp.unsqueeze(1) * block.ff(norm_r2)
        return reward_states

    def _side_stream_single(self, block, reward_states, temb, k_full, v_full):
        attn = block.attn
        heads = attn.heads
        norm_r, r_gate = block.norm(reward_states, emb=temb)
        mlp_r = block.act_mlp(block.proj_mlp(norm_r))
        q_r = attn.norm_q(self.reward_q_proj(norm_r).unflatten(-1, (heads, -1)))
        k_r = attn.norm_k(attn.to_k(norm_r).unflatten(-1, (heads, -1)))
        v_r = attn.to_v(norm_r).unflatten(-1, (heads, -1))
        key = torch.cat([k_r, k_full], dim=1)
        value = torch.cat([v_r, v_full], dim=1)
        o = F.scaled_dot_product_attention(
            q_r.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        )
        o = o.transpose(1, 2).flatten(2, 3).to(q_r.dtype)
        # Single-block update is attn+MLP fused through proj_out (inseparable).
        return reward_states + r_gate.unsqueeze(1) * block.proj_out(torch.cat([o, mlp_r], dim=2))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _flatten_inputs(self, latents, prompt_embeds, pooled_prompt_embeds, timesteps):
        batch_size, group_size = latents.shape[:2]
        flat_latents = latents.reshape(batch_size * group_size, *latents.shape[2:])
        flat_prompt_embeds = (
            prompt_embeds.unsqueeze(1)
            .expand(-1, group_size, -1, -1)
            .reshape(batch_size * group_size, *prompt_embeds.shape[1:])
        )
        flat_pooled = (
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
        return batch_size, group_size, flat_latents, flat_prompt_embeds, flat_pooled, flat_timesteps

    def forward(self, latents, *, prompt_embeds, pooled_prompt_embeds, timesteps):
        if pooled_prompt_embeds is None or pooled_prompt_embeds.ndim != 2:
            raise ValueError(
                "FLUX requires the real CLIP pooled embeds [B, 768]; "
                "re-point the manifest to the FLUX cache"
            )
        (
            batch_size,
            group_size,
            flat_latents,
            flat_prompt_embeds,
            flat_pooled,
            flat_timesteps,
        ) = self._flatten_inputs(latents, prompt_embeds, pooled_prompt_embeds, timesteps)
        device = flat_latents.device
        model_dtype = next(self.backbone.transformer.parameters()).dtype

        # Continuous flow level u = sigma in [0,1] (guard for the eval constant path that
        # passes the discrete scheduler timestep; FlowMatch timesteps are sigma*1000).
        u = flat_timesteps.to(device=device, dtype=torch.float32)
        if u.numel() > 0 and float(u.max()) > 1.0:
            u = u / self.backbone.num_train_timesteps

        img, txt, temb, rotary, txt_len, n_img = self.backbone.build_flux_inputs(
            flat_latents.to(dtype=model_dtype),
            flat_prompt_embeds.to(dtype=model_dtype),
            flat_pooled.to(dtype=model_dtype),
            u,
        )

        reward_states = self.reward_tokens.unsqueeze(0).expand(flat_latents.shape[0], -1, -1).to(dtype=model_dtype)

        visual_set = set(self.visual_layers)
        text_set = set(self.text_layers)
        visual_by_layer: dict[int, torch.Tensor] = {}
        text_by_layer: dict[int, torch.Tensor] = {}

        for index in range(self.stop_at_layer):
            block = self.backbone.get_block(index)
            if self.backbone.is_double_block(index):
                txt, img, k_full, v_full = self._run_frozen_flux_double_block(block, img, txt, temb, rotary)
            else:
                txt, img, k_full, v_full = self._run_frozen_flux_single_block(block, img, txt, temb, rotary)

            if index in visual_set:
                visual_by_layer[index] = img
            if index in text_set:
                text_by_layer[index] = txt

            if not self.disable_side_stream:
                if self.backbone.is_double_block(index):
                    reward_states = self._side_stream_double(block, reward_states, temb, k_full, v_full)
                else:
                    reward_states = self._side_stream_single(block, reward_states, temb, k_full, v_full)

        if self.debug_capture:
            self.last_debug = {
                "visual_by_layer": {i: t.detach() for i, t in visual_by_layer.items()},
                "text_by_layer": {i: t.detach() for i, t in text_by_layer.items()},
                "temb": temb.detach(),
                "u": u.detach(),
            }

        visual_features = [visual_by_layer[i] for i in self.visual_layers]
        text_features = [text_by_layer[i] for i in self.text_layers]
        visual_features = [self.visual_pool(feat, self.vis_h, self.vis_w) for feat in visual_features]

        temb_head = self.temb_proj(temb.to(dtype=self.temb_proj.weight.dtype))
        return self._compute_outputs(reward_states, visual_features, text_features, temb_head, batch_size, group_size)

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
            "architecture": "flux_latent_reward_grid_pool_nope_multihead",
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
