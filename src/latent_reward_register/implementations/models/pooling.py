from __future__ import annotations

import torch
import torch.nn as nn


def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, dim], got {tuple(hidden_states.shape)}")
    return hidden_states.mean(dim=1)


class QueryAttentionPooling(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.input_norm = nn.LayerNorm(feature_dim) if layer_norm else None
        self.output_norm = nn.LayerNorm(feature_dim) if layer_norm else None
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.queries = nn.Parameter(torch.empty(num_queries, feature_dim))
        nn.init.xavier_uniform_(self.queries)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, dim], got {tuple(hidden_states.shape)}")

        inputs = hidden_states if self.input_norm is None else self.input_norm(hidden_states)
        queries = self.queries.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        attended, _ = self.attention(queries, inputs, inputs, need_weights=False)
        pooled = (attended + queries).mean(dim=1)
        if self.output_norm is not None:
            pooled = self.output_norm(pooled)
        return pooled

