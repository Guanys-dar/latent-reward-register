"""DiNa-style reward head over frozen backbone states.

Ported from the research workspace. Kept
checkpoint-faithful: the architecture must stay byte-compatible with the
published checkpoints, so prefer provenance notes over refactoring here.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

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

