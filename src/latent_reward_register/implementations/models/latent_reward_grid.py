"""SD3 reward register (exp11 release baseline) and its ablation variants.

Ported from the research workspace. Kept
checkpoint-faithful: the architecture must stay byte-compatible with the
published checkpoints, so prefer provenance notes over refactoring here.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SD3RewardBackbone
from .pooling import QueryAttentionPooling, mean_pool
from .reward_token_dina_head import (
    _CrossAttentionBlockVGated,
    _FiLMLayerAdapter,
    _normalize_layer_indices,
)
from ..gradmode import frozen_trunk_context


def _reshape_heads(hidden_states: torch.Tensor, heads: int) -> torch.Tensor:
    batch_size, sequence_length, dim = hidden_states.shape
    head_dim = dim // heads
    return hidden_states.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)


def _merge_heads(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, heads, sequence_length, head_dim = hidden_states.shape
    return hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, heads * head_dim)


class _SpatialPool2d(nn.Module):
    """2D average-pool over a flattened token grid.

    (B, H*W, C) -> reshape (B, C, H, W) -> AvgPool2d(pool_factor) -> (B, H'*W', C).
    Parameter-free; mirrors DiNa-LRM's SpatialDownsample so a 64x64 visual snapshot
    pools 4x4 -> 16x16 = 256 tokens.
    """

    def __init__(self, pool_factor: int = 4):
        super().__init__()
        self.pool_factor = int(pool_factor)
        self.pool = nn.AvgPool2d(kernel_size=self.pool_factor, stride=self.pool_factor)

    def forward(self, x: torch.Tensor, h_tokens: int, w_tokens: int) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        if num_tokens != h_tokens * w_tokens:
            raise ValueError(
                f"_SpatialPool2d expected {h_tokens * w_tokens} tokens (={h_tokens}x{w_tokens}), "
                f"got {num_tokens}"
            )
        x = x.transpose(1, 2).reshape(batch_size, channels, h_tokens, w_tokens)
        x = self.pool(x)
        return x.flatten(2).transpose(1, 2)


class _JointAttentionBlock(nn.Module):
    """Reward tokens attend jointly over [reward, visual, text] in a single softmax.

    Query = the reward tokens; Key/Value = concat(reward, visual, text).  This fuses
    self-attention (reward<->reward) and cross-attention (reward->visual/text) into one
    attention operation (a single softmax over the concatenated context), with a residual
    connection on the reward tokens.  Same forward signature as _CrossAttentionBlockVGated,
    so it can be dropped in as the head's first attention stage.
    """

    def __init__(self, *, dim: int, num_heads: int = 8, use_text: bool = True, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.use_text = use_text
        self.dropout = dropout

        self.norm_q = nn.RMSNorm(dim)
        self.norm_v = nn.RMSNorm(dim)
        self.norm_t = nn.RMSNorm(dim) if use_text else None

        self.to_q = nn.Linear(dim, dim)
        # Reward tokens contribute to K/V too -> self-attention term.
        self.to_k_reward = nn.Linear(dim, dim)
        self.to_v_reward = nn.Linear(dim, dim)
        self.to_k_vis = nn.Linear(dim, dim)
        self.to_v_vis = nn.Linear(dim, dim)
        self.to_k_text = nn.Linear(dim, dim) if use_text else None
        self.to_v_text = nn.Linear(dim, dim) if use_text else None
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        queries: torch.Tensor,
        context_visual: torch.Tensor,
        context_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = queries
        q_in = self.norm_q(queries)
        vis = self.norm_v(context_visual)

        # Keys/values = concat(reward, visual, text) along the token axis.
        keys = [self.to_k_reward(q_in), self.to_k_vis(vis)]
        values = [self.to_v_reward(q_in), self.to_v_vis(vis)]
        if self.use_text and context_text is not None and self.to_k_text is not None:
            txt = self.norm_t(context_text)
            keys.append(self.to_k_text(txt))
            values.append(self.to_v_text(txt))

        q = _reshape_heads(self.to_q(q_in), self.num_heads)
        k = _reshape_heads(torch.cat(keys, dim=1), self.num_heads)
        v = _reshape_heads(torch.cat(values, dim=1), self.num_heads)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        attended = _merge_heads(attended)
        return residual + self.to_out(attended)


class LatentRewardGridHead(nn.Module):
    """DiNA-style reward head that accepts external queries (reward_states from side-stream)."""

    def __init__(
        self,
        *,
        token_dim: int,
        query_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_visual_layers: int,
        n_text_layers: int,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = False,
        skip_attn2: bool = False,
    ):
        super().__init__()
        if width == -1:
            width = token_dim
        if width % 4 != 0:
            raise ValueError(f"Reward-head width must be divisible by 4, got {width}")
        if n_visual_layers <= 0:
            raise ValueError("At least one visual layer is required")

        self.skip_attn2 = bool(skip_attn2)
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

        self.query_proj = nn.Linear(query_dim, width)

        # First attention stage. When use_self_attn is set, the reward tokens attend jointly
        # over [reward, visual, text] in a single softmax (self-attention + cross-attention
        # together). Otherwise it is the exp4 cross-attention to visual+text only. Both
        # share the same (queries, visual, text) forward signature.
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.attn1 = _JointAttentionBlock(
                dim=width,
                num_heads=num_attn_heads,
                use_text=n_text_layers > 0,
                dropout=dropout,
            )
        else:
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

    def features(
        self,
        queries: torch.Tensor,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Shared trunk: per-token query features before the scalar readout.

        Returns ``[B_flat, num_queries, width]``.  Exposed so multi-head subclasses
        can attach their own per-head readouts to the same trunk.
        """
        visual_out, text_out = self._build_context_tokens(visual_features, text_features, temb)
        q = self.query_proj(queries)
        # attn1 is joint (K/V = reward+visual+text) when use_self_attn, else cross-only.
        q = self.attn1(q, visual_out, text_out)
        if not self.skip_attn2:
            q = self.attn2(q, visual_out, None)
        q = q + self.ff(self.norm_ff(q))
        return q

    def forward(
        self,
        queries: torch.Tensor,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        q = self.features(queries, visual_features, text_features, temb)
        return self.head(q).mean(dim=1).squeeze(-1)


class LatentRewardGridMLPHead(nn.Module):
    """Simple MLP readout over reward-grid snapshots (no cross-attention).

    Consumes the 256-token side-stream reward grid captured at each feature layer
    (3x256 tokens for layers [4, 8, 12]).  Each snapshot is timestep-FiLM-adapted,
    the per-layer snapshots are fused on the channel dim, mean-pooled over the 256
    grid tokens, and an MLP predicts the scalar reward.  Unlike LatentRewardGridHead
    this uses neither learnable queries nor the image/text feature branches.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_grid_layers: int,
        dropout: float = 0.0,
        use_proj_in: bool = False,
    ):
        super().__init__()
        if width == -1:
            width = token_dim
        if width % 4 != 0:
            raise ValueError(f"Reward-head width must be divisible by 4, got {width}")
        if n_grid_layers <= 0:
            raise ValueError("At least one grid-snapshot layer is required")

        feature_out_dim = width // 4
        self.layer_adapters = nn.ModuleList(
            [
                _FiLMLayerAdapter(
                    in_dim=token_dim,
                    emb_dim=token_dim,
                    hidden_dim=width,
                    output_dim=feature_out_dim,
                    use_proj_in=use_proj_in,
                )
                for _ in range(n_grid_layers)
            ]
        )
        self.agg = nn.Linear(n_grid_layers * feature_out_dim, width)
        self.norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, out_dim),
        )

    def forward(
        self,
        grid_snapshots: list[torch.Tensor],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        if len(grid_snapshots) != len(self.layer_adapters):
            raise ValueError(
                f"Expected {len(self.layer_adapters)} grid snapshots, got {len(grid_snapshots)}"
            )
        tokens = [
            adapter(snapshot, temb)
            for adapter, snapshot in zip(self.layer_adapters, grid_snapshots)
        ]  # each (B, 256, width // 4)
        fused = self.agg(torch.cat(tokens, dim=-1))  # (B, 256, width)
        pooled = self.norm(fused.mean(dim=1))  # (B, width)
        return self.mlp(pooled).squeeze(-1)  # (B,)


class LatentRewardGridSetHead(nn.Module):
    """exp6: minimal token-set readout over reward-grid snapshots.

    Treats the per-layer grid snapshots ([4, 8, 12] -> 3 x 256 = 768 tokens) as a single
    token set.  Each snapshot is tagged with a learned per-layer embedding, the whole set
    is timestep-FiLM-adapted by ONE shared adapter, pooled to a vector, and an MLP predicts
    the scalar reward.  Unlike LatentRewardGridMLPHead this keeps all 768 tokens (token-axis
    concat rather than channel-fusion back to 256) and uses a single FiLM adapter instead of
    one per layer, so the head is much lighter.  The reward depends only on the reward-grid
    tokens (no image/text branch, no cross-attention).
    """

    def __init__(
        self,
        *,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_grid_layers: int,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        pool: str = "attn",
        num_pool_queries: int = 1,
        num_pool_heads: int = 8,
    ):
        super().__init__()
        if width == -1:
            width = token_dim
        if width % 4 != 0:
            raise ValueError(f"Reward-head width must be divisible by 4, got {width}")
        if n_grid_layers <= 0:
            raise ValueError("At least one grid-snapshot layer is required")
        if pool not in {"attn", "mean"}:
            raise ValueError(f"pool must be 'attn' or 'mean', got {pool!r}")

        self.n_grid_layers = int(n_grid_layers)
        self.pool_kind = pool

        # One learned embedding per source layer, broadcast over that layer's 256 tokens so
        # the head can tell which depth each token came from after the token-axis concat.
        self.layer_embed = nn.Parameter(torch.randn(n_grid_layers, 1, token_dim) * 0.02)

        # Single shared timestep-FiLM over the full 768-token set -> width.
        self.film = _FiLMLayerAdapter(
            in_dim=token_dim,
            emb_dim=token_dim,
            hidden_dim=width,
            output_dim=width,
            use_proj_in=use_proj_in,
        )

        if pool == "attn":
            if width % num_pool_heads != 0:
                raise ValueError(
                    f"width={width} must be divisible by num_pool_heads={num_pool_heads}"
                )
            self.pool = QueryAttentionPooling(
                width,
                num_queries=num_pool_queries,
                num_heads=num_pool_heads,
                dropout=dropout,
            )
        else:
            self.pool = None

        self.norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, out_dim),
        )

    def forward(
        self,
        grid_snapshots: list[torch.Tensor],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        if len(grid_snapshots) != self.n_grid_layers:
            raise ValueError(
                f"Expected {self.n_grid_layers} grid snapshots, got {len(grid_snapshots)}"
            )
        # Tag each snapshot with its layer embedding, then concat on the token axis -> 768.
        tagged = [
            snapshot + self.layer_embed[i] for i, snapshot in enumerate(grid_snapshots)
        ]  # each (B, 256, token_dim)
        tokens = torch.cat(tagged, dim=1)  # (B, n_grid_layers * 256, token_dim)
        tokens = self.film(tokens, temb)  # (B, n_grid_layers * 256, width)
        pooled = mean_pool(tokens) if self.pool is None else self.pool(tokens)  # (B, width)
        pooled = self.norm(pooled)
        return self.mlp(pooled).squeeze(-1)  # (B,)


class LatentRewardGridQueryHead(nn.Module):
    """exp9: DiNA cross-attention readout over reward-grid snapshots with learnable queries.

    Reward-grid-only (no image/text context, unlike exp4): a small set of learnable query
    tokens cross-attend (DiNA-style, V-gated) to the per-layer reward-grid snapshots
    ([4, 8, 12] -> 3 x 256 tokens), which serve as the head's "visual" context.  This is the
    faithful DiNa-LRM RewardHead shape (learnable queries over feature tokens), reusing
    LatentRewardGridHead with its text branch disabled (n_text_layers=0).  Shares the
    (grid_snapshots, temb) forward signature with LatentRewardGridMLPHead so the exp5
    side-stream forward is reused unchanged.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        width: int = -1,
        out_dim: int = 1,
        n_grid_layers: int,
        num_queries: int = 16,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = False,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        # Learnable query tokens (in token_dim; the core's query_proj maps token_dim -> width).
        self.queries = nn.Parameter(torch.randn(num_queries, token_dim) * 0.02)
        self.core = LatentRewardGridHead(
            token_dim=token_dim,
            query_dim=token_dim,
            width=width,
            out_dim=out_dim,
            n_visual_layers=n_grid_layers,  # grid snapshots play the role of the visual context
            n_text_layers=0,                # reward-grid-only: no text branch
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
        )

    def forward(
        self,
        grid_snapshots: list[torch.Tensor],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = grid_snapshots[0].shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        # text_features=None -> LatentRewardGridHead skips the (absent) text branch.
        return self.core(queries, grid_snapshots, None, temb)


class SD3LatentRewardGridModel(nn.Module):
    """Reward-token side-stream (256 tokens, 16x16 grid) with DiNA-style head.

    Combines the side-stream attention from SD3RewardTokenModel with
    multi-layer feature extraction and a DiNA reward head.  Backbone is
    fully frozen; trainable parameters are reward_tokens, reward_pos_embed,
    reward_q_proj, and reward_head.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = False,
        skip_attn2: bool = False,
        num_reward_tokens: int | None = None,
        use_pos_embed: bool = True,
        disable_side_stream: bool = False,
        side_stream_ffn: bool = False,
        freeze_q_proj: bool = False,
        head_input_tokens: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze_all()

        # Ablation knobs (all default to the canonical Reward Register behavior):
        #   disable_side_stream: skip the per-block reward side-stream entirely so the
        #     reward tokens stay as static learned queries into the head (register OFF).
        #   side_stream_ffn: additionally pass the reward states through the frozen
        #     block FFN in the side-stream (not_skip_ffn); default is gated-attention only.
        #   freeze_q_proj: keep reward_q_proj fixed at init instead of training it.
        self.disable_side_stream = bool(disable_side_stream)
        self.side_stream_ffn = bool(side_stream_ffn)
        self.freeze_q_proj = bool(freeze_q_proj)
        # exp13: feed the reward head the layer-0 DiT INPUT tokens (pos_embed(latents) /
        # context_embedder(text), FiLM-adapted with temb) instead of the layer-4/8/12 block
        # snapshots. The register side-stream still runs the full backbone depth
        # (num_transformer_layers); only the head's visual/text context source changes.
        # This makes reward guidance cheap: dR/dlatents flows through pos_embed + head only,
        # never a backward through the DiT blocks (the snapshots/register reads stay detached).
        self.head_input_tokens = bool(head_input_tokens)

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
        self.reward_grid_h = int(reward_grid_h)
        self.reward_grid_w = int(reward_grid_w)
        self.vis_h = int(vis_h)
        self.vis_w = int(vis_w)
        self.use_pos_embed = bool(use_pos_embed)
        grid_tokens = self.reward_grid_h * self.reward_grid_w
        self.num_reward_tokens = int(num_reward_tokens) if num_reward_tokens is not None else grid_tokens
        hidden_size = self.backbone.hidden_size

        # Optional spatial pooling applied to visual snapshots before the head.
        # None -> exp4 behavior (full unpooled visual tokens). Subclasses (exp7) set it.
        self.visual_pool: _SpatialPool2d | None = None

        self.reward_tokens = nn.Parameter(torch.empty(self.num_reward_tokens, hidden_size))
        self.reward_q_proj = nn.Linear(hidden_size, hidden_size)
        # reward_q_proj carries no gradient when explicitly frozen (knob 3) or when the
        # side-stream is disabled (knob 1) -- in the latter case it is never used in the
        # forward pass, so leaving it trainable would make DDP abort on an unused parameter.
        if self.freeze_q_proj or self.disable_side_stream:
            for param in self.reward_q_proj.parameters():
                param.requires_grad = False

        # Position embedding for the reward tokens (spatially interpolated from the backbone
        # grid). exp8 drops it (use_pos_embed=False) and uses a non-grid token count, so the
        # reward tokens are a position-free learnable set.
        if self.use_pos_embed:
            if self.num_reward_tokens != grid_tokens:
                raise ValueError(
                    f"reward_pos_embed requires num_reward_tokens (={self.num_reward_tokens}) to equal "
                    f"reward_grid_h*reward_grid_w (={grid_tokens}); set use_pos_embed=False for a "
                    "non-grid token count"
                )
            self.reward_pos_embed = nn.Parameter(self._init_pos_embed(vis_h, vis_w, hidden_size))
        else:
            self.reward_pos_embed = None

        self.reward_head = LatentRewardGridHead(
            token_dim=hidden_size,
            query_dim=hidden_size,
            width=width,
            out_dim=out_dim,
            n_visual_layers=1 if self.head_input_tokens else len(self.visual_layers),
            n_text_layers=1 if self.head_input_tokens else len(self.text_layers),
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
            skip_attn2=skip_attn2,
        )

        self._init_weights()

    def _init_pos_embed(self, vis_h: int, vis_w: int, hidden_size: int) -> torch.Tensor:
        pe_mod = self.backbone.transformer.pos_embed
        if pe_mod.pos_embed_max_size is None:
            return torch.randn(1, self.reward_grid_h * self.reward_grid_w, hidden_size) * 0.02

        # Slicing a constant positional-embedding table: never on the latent path.
        with torch.no_grad():
            max_size = pe_mod.pos_embed_max_size
            full_pe = pe_mod.pos_embed.float().reshape(1, max_size, max_size, -1)
            top = (max_size - vis_h) // 2
            left = (max_size - vis_w) // 2
            vis_pe = full_pe[:, top : top + vis_h, left : left + vis_w, :]
            vis_pe = vis_pe.permute(0, 3, 1, 2)
            rew_pe = F.interpolate(
                vis_pe,
                size=(self.reward_grid_h, self.reward_grid_w),
                mode="bilinear",
                align_corners=False,
            )
            rew_pe = rew_pe.permute(0, 2, 3, 1).reshape(
                1, self.reward_grid_h * self.reward_grid_w, -1
            )
        return rew_pe

    def _init_weights(self) -> None:
        nn.init.normal_(self.reward_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.reward_q_proj.weight)
        nn.init.zeros_(self.reward_q_proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.transformer.eval()
        return self

    # ------------------------------------------------------------------
    # Frozen SD3 block (identical to reward_token.py)
    # ------------------------------------------------------------------

    def _run_frozen_sd3_block(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if block.context_pre_only:
            raise ValueError("Latent-reward-grid model only supports non-final joint blocks")

        heads = block.attn.heads

        with frozen_trunk_context():
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                hidden_states, emb=temb
            )
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                block.norm1_context(encoder_hidden_states, emb=temb)
            )

            q_img = _reshape_heads(block.attn.to_q(norm_hidden_states), heads)
            k_img = _reshape_heads(block.attn.to_k(norm_hidden_states), heads)
            v_img = _reshape_heads(block.attn.to_v(norm_hidden_states), heads)
            q_txt = _reshape_heads(block.attn.add_q_proj(norm_encoder_hidden_states), heads)
            k_txt = _reshape_heads(block.attn.add_k_proj(norm_encoder_hidden_states), heads)
            v_txt = _reshape_heads(block.attn.add_v_proj(norm_encoder_hidden_states), heads)

            if block.attn.norm_q is not None:
                q_img = block.attn.norm_q(q_img)
            if block.attn.norm_k is not None:
                k_img = block.attn.norm_k(k_img)
            if block.attn.norm_added_q is not None:
                q_txt = block.attn.norm_added_q(q_txt)
            if block.attn.norm_added_k is not None:
                k_txt = block.attn.norm_added_k(k_txt)

            joint_query = torch.cat([q_img, q_txt], dim=2)
            joint_key = torch.cat([k_img, k_txt], dim=2)
            joint_value = torch.cat([v_img, v_txt], dim=2)
            joint_output = F.scaled_dot_product_attention(
                joint_query, joint_key, joint_value, dropout_p=0.0, is_causal=False,
            )
            joint_output = _merge_heads(joint_output)
            attn_output, context_attn_output = joint_output.split(
                (hidden_states.shape[1], encoder_hidden_states.shape[1]), dim=1,
            )

            attn_output = block.attn.to_out[0](attn_output)
            attn_output = block.attn.to_out[1](attn_output)
            context_attn_output = block.attn.to_add_out(context_attn_output)

            hidden_states = hidden_states + gate_msa[:, None] * attn_output
            encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attn_output

            norm_hidden_states_ff = block.norm2(hidden_states)
            norm_hidden_states_ff = norm_hidden_states_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            hidden_states = hidden_states + gate_mlp[:, None] * block.ff(norm_hidden_states_ff)

            norm_encoder_hidden_states_ff = block.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states_ff = (
                norm_encoder_hidden_states_ff * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp[:, None] * block.ff_context(norm_encoder_hidden_states_ff)
            )

        return hidden_states, encoder_hidden_states, k_img, v_img, k_txt, v_txt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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

        device = flat_latents.device
        model_dtype = next(self.backbone.transformer.parameters()).dtype
        hidden_states = self.backbone.transformer.pos_embed(flat_latents.to(dtype=model_dtype))
        temb = self.backbone.transformer.time_text_embed(
            flat_timesteps.to(device=device),
            flat_pooled_prompt_embeds.to(device=device, dtype=model_dtype),
        )
        encoder_hidden_states = self.backbone.transformer.context_embedder(
            flat_prompt_embeds.to(device=device, dtype=model_dtype)
        )

        # exp13: snapshot the layer-0 DiT inputs (before any transformer block) so the head
        # can read them instead of mid-trunk block outputs. These stay differentiable w.r.t.
        # latents through pos_embed only (the blocks below run under no_grad), giving a cheap
        # guidance gradient. The register side-stream still consumes the full backbone depth.
        input_visual = hidden_states
        input_text = encoder_hidden_states

        reward_states = (
            self.reward_tokens.unsqueeze(0).expand(flat_latents.shape[0], -1, -1).to(dtype=model_dtype)
        )
        if self.reward_pos_embed is not None:
            reward_states = reward_states + self.reward_pos_embed.to(dtype=model_dtype)

        visual_set = set(self.visual_layers)
        text_set = set(self.text_layers)
        visual_features_by_layer: dict[int, torch.Tensor] = {}
        text_features_by_layer: dict[int, torch.Tensor] = {}

        for index, block in enumerate(self.backbone.transformer.transformer_blocks):
            if index >= self.stop_at_layer:
                break

            (
                hidden_states,
                encoder_hidden_states,
                k_img,
                v_img,
                k_txt,
                v_txt,
            ) = self._run_frozen_sd3_block(block, hidden_states, encoder_hidden_states, temb)

            if index in visual_set:
                visual_features_by_layer[index] = hidden_states
            if index in text_set:
                text_features_by_layer[index] = encoder_hidden_states

            # Reward side-stream attention (matching reward_token.py). The whole
            # side-stream is skipped when disable_side_stream is set (register OFF):
            # the reward tokens then remain the static learned set fed to the head.
            if not self.disable_side_stream:
                heads = block.attn.heads
                norm_reward_states, gate_r_msa, shift_r_mlp, scale_r_mlp, gate_r_mlp = block.norm1(
                    reward_states, emb=temb
                )
                q_r = _reshape_heads(self.reward_q_proj(norm_reward_states), heads)
                k_r = _reshape_heads(block.attn.to_k(norm_reward_states), heads)
                v_r = _reshape_heads(block.attn.to_v(norm_reward_states), heads)

                if block.attn.norm_q is not None:
                    q_r = block.attn.norm_q(q_r)
                if block.attn.norm_k is not None:
                    k_r = block.attn.norm_k(k_r)

                reward_key = torch.cat([k_r, k_img, k_txt], dim=2)
                reward_value = torch.cat([v_r, v_img, v_txt], dim=2)
                reward_attn_output = F.scaled_dot_product_attention(
                    q_r, reward_key, reward_value, dropout_p=0.0, is_causal=False,
                )
                reward_attn_output = _merge_heads(reward_attn_output)
                reward_attn_output = block.attn.to_out[0](reward_attn_output)
                reward_attn_output = block.attn.to_out[1](reward_attn_output)

                reward_states = reward_states + gate_r_msa[:, None] * reward_attn_output

                # not_skip_ffn: additionally route the reward states through the frozen
                # block FFN (mirrors reward_token.py's not_skip_ffn variant). Default
                # (side_stream_ffn=False) is gated-attention only.
                if self.side_stream_ffn:
                    norm_reward_states_ff = block.norm2(reward_states)
                    norm_reward_states_ff = (
                        norm_reward_states_ff * (1 + scale_r_mlp[:, None]) + shift_r_mlp[:, None]
                    )
                    reward_states = reward_states + gate_r_mlp[:, None] * block.ff(norm_reward_states_ff)

        if self.head_input_tokens:
            # exp13: head reads the layer-0 DiT inputs, not the mid-trunk block snapshots.
            visual_features = [input_visual]
            text_features = [input_text]
        else:
            visual_features = [visual_features_by_layer[i] for i in self.visual_layers]
            text_features = [text_features_by_layer[i] for i in self.text_layers]

        # exp7: spatially pool the visual context (e.g. 64x64 -> 16x16 = 256) before the
        # head. Text is left unpooled. No-op for exp4 (self.visual_pool is None).
        if self.visual_pool is not None:
            visual_features = [
                self.visual_pool(feature, self.vis_h, self.vis_w) for feature in visual_features
            ]

        return self._compute_outputs(
            reward_states, visual_features, text_features, temb, batch_size, group_size
        )

    def _compute_outputs(
        self,
        reward_states: torch.Tensor,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
        batch_size: int,
        group_size: int,
    ):
        """Map trunk reward states to per-image outputs.

        Base behavior: a single scalar reward, reshaped to ``[B, group_size]``.
        Multi-head subclasses override this to return a ``dict[str, [B, group_size]]``.
        """
        scores = self.reward_head(
            queries=reward_states,
            visual_features=visual_features,
            text_features=text_features,
            temb=temb,
        )
        return scores.reshape(batch_size, group_size)

    # ------------------------------------------------------------------
    # Parameter groups / checkpoint
    # ------------------------------------------------------------------

    def parameter_groups(self, *, head_lr: float, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        del backbone_lr
        params = [self.reward_tokens]
        if self.reward_pos_embed is not None:
            params.append(self.reward_pos_embed)
        if not self.freeze_q_proj and not self.disable_side_stream:
            params += list(self.reward_q_proj.parameters())
        params += list(self.reward_head.parameters())
        return [{"params": params, "lr": head_lr}]

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "architecture": "latent_reward_grid",
            "visual_layers": self.visual_layers,
            "text_layers": self.text_layers,
            "layer_index_base": self.layer_index_base,
            "stop_at_layer": self.stop_at_layer,
            "reward_grid_h": self.reward_grid_h,
            "reward_grid_w": self.reward_grid_w,
            "num_reward_tokens": self.num_reward_tokens,
            "use_pos_embed": self.use_pos_embed,
            "disable_side_stream": self.disable_side_stream,
            "side_stream_ffn": self.side_stream_ffn,
            "freeze_q_proj": self.freeze_q_proj,
            "head_input_tokens": self.head_input_tokens,
            "reward_tokens": self.reward_tokens.detach().cpu(),
            "reward_q_proj": self.reward_q_proj.state_dict(),
            "reward_head": self.reward_head.state_dict(),
        }
        if self.reward_pos_embed is not None:
            state["reward_pos_embed"] = self.reward_pos_embed.detach().cpu()
        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.reward_tokens.data.copy_(
            state["reward_tokens"].to(device=self.reward_tokens.device, dtype=self.reward_tokens.dtype)
        )
        if self.reward_pos_embed is not None and state.get("reward_pos_embed") is not None:
            self.reward_pos_embed.data.copy_(
                state["reward_pos_embed"].to(device=self.reward_pos_embed.device, dtype=self.reward_pos_embed.dtype)
            )
        self.reward_q_proj.load_state_dict(state["reward_q_proj"])
        self.reward_head.load_state_dict(state["reward_head"])


class SD3LatentRewardGridMLPModel(SD3LatentRewardGridModel):
    """exp5: reward-grid side-stream + simple MLP readout (no cross-attention).

    Identical side-stream as SD3LatentRewardGridModel: a 256-token reward grid
    (16x16) flows through the frozen SD3 blocks.  The grid state is snapshotted
    after each feature layer ([4, 8, 12]) and a LatentRewardGridMLPHead predicts
    the reward directly from those 3x256 tokens via timestep-FiLM, channel-fusion
    of the layers, mean-pooling, and an MLP.  The image-feature branch, text-feature
    branch, and the Q-former cross-attention of exp4 are removed.  Backbone is fully
    frozen; trainable parameters are reward_tokens, reward_pos_embed, reward_q_proj,
    and reward_head.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int] | None = None,
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        dropout: float = 0.0,
        use_proj_in: bool = False,
    ):
        # Intentionally bypass SD3LatentRewardGridModel.__init__ (which builds the
        # cross-attention head); reuse only nn.Module init + this class's setup.
        nn.Module.__init__(self)
        self.backbone = backbone
        self.backbone.freeze_all()

        # Layers at which to snapshot the reward grid (the exp4 "feature layers").
        self.grid_layers = _normalize_layer_indices(
            visual_layers,
            num_layers=self.backbone.num_layers,
            index_base=layer_index_base,
        )
        # Kept for compatibility with parent helpers/attributes; unused by the head.
        self.visual_layers = self.grid_layers
        self.text_layers = self.grid_layers

        self.stop_at_layer = max(self.grid_layers) + 1
        if num_transformer_layers is not None:
            requested_stop = int(num_transformer_layers)
            if requested_stop < self.stop_at_layer:
                raise ValueError(
                    f"num_transformer_layers={requested_stop} cannot cover selected layers; "
                    f"need at least {self.stop_at_layer}"
                )
            self.stop_at_layer = requested_stop

        self.layer_index_base = int(layer_index_base)
        self.reward_grid_h = int(reward_grid_h)
        self.reward_grid_w = int(reward_grid_w)
        num_reward_tokens = self.reward_grid_h * self.reward_grid_w
        hidden_size = self.backbone.hidden_size

        self.reward_tokens = nn.Parameter(torch.empty(num_reward_tokens, hidden_size))
        self.reward_q_proj = nn.Linear(hidden_size, hidden_size)
        self.reward_pos_embed = nn.Parameter(
            self._init_pos_embed(vis_h, vis_w, hidden_size)
        )

        self.reward_head = LatentRewardGridMLPHead(
            token_dim=hidden_size,
            width=width,
            out_dim=out_dim,
            n_grid_layers=len(self.grid_layers),
            dropout=dropout,
            use_proj_in=use_proj_in,
        )

        self._init_weights()

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

        device = flat_latents.device
        model_dtype = next(self.backbone.transformer.parameters()).dtype
        hidden_states = self.backbone.transformer.pos_embed(flat_latents.to(dtype=model_dtype))
        temb = self.backbone.transformer.time_text_embed(
            flat_timesteps.to(device=device),
            flat_pooled_prompt_embeds.to(device=device, dtype=model_dtype),
        )
        encoder_hidden_states = self.backbone.transformer.context_embedder(
            flat_prompt_embeds.to(device=device, dtype=model_dtype)
        )

        reward_states = (
            self.reward_tokens.unsqueeze(0).expand(flat_latents.shape[0], -1, -1).to(dtype=model_dtype)
            + self.reward_pos_embed.to(dtype=model_dtype)
        )

        grid_set = set(self.grid_layers)
        grid_by_layer: dict[int, torch.Tensor] = {}

        for index, block in enumerate(self.backbone.transformer.transformer_blocks):
            if index >= self.stop_at_layer:
                break

            (
                hidden_states,
                encoder_hidden_states,
                k_img,
                v_img,
                k_txt,
                v_txt,
            ) = self._run_frozen_sd3_block(block, hidden_states, encoder_hidden_states, temb)

            # Reward side-stream attention (identical to SD3LatentRewardGridModel).
            heads = block.attn.heads
            norm_reward_states, gate_r_msa, _shift_r_mlp, _scale_r_mlp, _gate_r_mlp = block.norm1(
                reward_states, emb=temb
            )
            q_r = _reshape_heads(self.reward_q_proj(norm_reward_states), heads)
            k_r = _reshape_heads(block.attn.to_k(norm_reward_states), heads)
            v_r = _reshape_heads(block.attn.to_v(norm_reward_states), heads)

            if block.attn.norm_q is not None:
                q_r = block.attn.norm_q(q_r)
            if block.attn.norm_k is not None:
                k_r = block.attn.norm_k(k_r)

            reward_key = torch.cat([k_r, k_img, k_txt], dim=2)
            reward_value = torch.cat([v_r, v_img, v_txt], dim=2)
            reward_attn_output = F.scaled_dot_product_attention(
                q_r, reward_key, reward_value, dropout_p=0.0, is_causal=False,
            )
            reward_attn_output = _merge_heads(reward_attn_output)
            reward_attn_output = block.attn.to_out[0](reward_attn_output)
            reward_attn_output = block.attn.to_out[1](reward_attn_output)

            reward_states = reward_states + gate_r_msa[:, None] * reward_attn_output

            if index in grid_set:
                grid_by_layer[index] = reward_states

        grid_snapshots = [grid_by_layer[i] for i in self.grid_layers]

        scores = self.reward_head(grid_snapshots=grid_snapshots, temb=temb)
        return scores.reshape(batch_size, group_size)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "architecture": "latent_reward_grid_mlp",
            "grid_layers": self.grid_layers,
            "layer_index_base": self.layer_index_base,
            "stop_at_layer": self.stop_at_layer,
            "reward_grid_h": self.reward_grid_h,
            "reward_grid_w": self.reward_grid_w,
            "reward_tokens": self.reward_tokens.detach().cpu(),
            "reward_pos_embed": self.reward_pos_embed.detach().cpu(),
            "reward_q_proj": self.reward_q_proj.state_dict(),
            "reward_head": self.reward_head.state_dict(),
        }


class SD3LatentRewardGridQueryModel(SD3LatentRewardGridMLPModel):
    """exp9: exp5's reward-grid side-stream + DiNA learnable-query cross-attention head.

    Identical reward-grid side-stream as exp5 (SD3LatentRewardGridMLPModel): a 256-token
    reward grid (16x16, position-bound via reward_pos_embed) flows through the frozen SD3
    blocks and is snapshotted after each feature layer ([4, 8, 12]).  The ONLY change vs
    exp5 is the readout head: instead of the mean-pool MLP, a LatentRewardGridQueryHead lets
    a small set of learnable query tokens cross-attend (DiNA-style) to the 3x256 grid
    snapshots.  Still reward-grid-only (no image/text context) -- distinct from exp4.  The
    side-stream forward, parameter_groups, and load_checkpoint_state are inherited unchanged.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int] | None = None,
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        num_queries: int = 16,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = False,
    ):
        # Mirror exp5's __init__ (bypass exp4's cross-attn init); only the head differs.
        nn.Module.__init__(self)
        self.backbone = backbone
        self.backbone.freeze_all()

        self.grid_layers = _normalize_layer_indices(
            visual_layers,
            num_layers=self.backbone.num_layers,
            index_base=layer_index_base,
        )
        # Kept for compatibility with parent helpers/attributes; unused by the head.
        self.visual_layers = self.grid_layers
        self.text_layers = self.grid_layers

        self.stop_at_layer = max(self.grid_layers) + 1
        if num_transformer_layers is not None:
            requested_stop = int(num_transformer_layers)
            if requested_stop < self.stop_at_layer:
                raise ValueError(
                    f"num_transformer_layers={requested_stop} cannot cover selected layers; "
                    f"need at least {self.stop_at_layer}"
                )
            self.stop_at_layer = requested_stop

        self.layer_index_base = int(layer_index_base)
        self.reward_grid_h = int(reward_grid_h)
        self.reward_grid_w = int(reward_grid_w)
        num_reward_tokens = self.reward_grid_h * self.reward_grid_w
        hidden_size = self.backbone.hidden_size

        self.reward_tokens = nn.Parameter(torch.empty(num_reward_tokens, hidden_size))
        self.reward_q_proj = nn.Linear(hidden_size, hidden_size)
        self.reward_pos_embed = nn.Parameter(
            self._init_pos_embed(vis_h, vis_w, hidden_size)
        )

        self.num_head_queries = int(num_queries)
        self.num_attn_heads = int(num_attn_heads)
        self.head_use_self_attn = bool(use_self_attn)
        self.reward_head = LatentRewardGridQueryHead(
            token_dim=hidden_size,
            width=width,
            out_dim=out_dim,
            n_grid_layers=len(self.grid_layers),
            num_queries=self.num_head_queries,
            num_attn_heads=self.num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=self.head_use_self_attn,
        )

        self._init_weights()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "architecture": "latent_reward_grid_xattn",
            "grid_layers": self.grid_layers,
            "layer_index_base": self.layer_index_base,
            "stop_at_layer": self.stop_at_layer,
            "reward_grid_h": self.reward_grid_h,
            "reward_grid_w": self.reward_grid_w,
            "num_head_queries": self.num_head_queries,
            "num_attn_heads": self.num_attn_heads,
            "use_self_attn": self.head_use_self_attn,
            "reward_tokens": self.reward_tokens.detach().cpu(),
            "reward_pos_embed": self.reward_pos_embed.detach().cpu(),
            "reward_q_proj": self.reward_q_proj.state_dict(),
            "reward_head": self.reward_head.state_dict(),
        }


class SD3LatentRewardGridSetModel(SD3LatentRewardGridMLPModel):
    """exp6: exp5's reward-grid input + a minimal token-set head.

    The side-stream and snapshotting are identical to SD3LatentRewardGridMLPModel (one
    256-token grid extracted at [4, 8, 12]); ``forward`` is inherited unchanged.  Only the
    readout differs: a LatentRewardGridSetHead reads the 3 x 256 = 768 snapshot tokens as a
    single layer-tagged token set (one shared FiLM, pool, MLP) instead of exp5's per-layer
    channel-fusion.  Trainable params are reward_tokens, reward_pos_embed, reward_q_proj,
    and reward_head; the backbone is fully frozen.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int] | None = None,
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        pool: str = "attn",
        num_pool_queries: int = 1,
    ):
        super().__init__(
            backbone,
            visual_layers=visual_layers,
            text_layers=text_layers,
            layer_index_base=layer_index_base,
            num_transformer_layers=num_transformer_layers,
            reward_grid_h=reward_grid_h,
            reward_grid_w=reward_grid_w,
            vis_h=vis_h,
            vis_w=vis_w,
            width=width,
            out_dim=out_dim,
            dropout=dropout,
            use_proj_in=use_proj_in,
        )
        # Replace exp5's channel-fusion MLP head with the lean token-set head.  The parent's
        # _init_weights only touches reward_tokens / reward_q_proj, so swapping the head here
        # is safe and the discarded MLP head is never used.
        self.reward_head = LatentRewardGridSetHead(
            token_dim=self.backbone.hidden_size,
            width=width,
            out_dim=out_dim,
            n_grid_layers=len(self.grid_layers),
            dropout=dropout,
            use_proj_in=use_proj_in,
            pool=pool,
            num_pool_queries=num_pool_queries,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["architecture"] = "latent_reward_grid_set"
        return state


class SD3LatentRewardGridPoolModel(SD3LatentRewardGridModel):
    """exp7: exp4 with spatially pooled visual context (reward registers + pooled-visual + text).

    Identical to SD3LatentRewardGridModel (exp4): a 256-token side-stream reward grid flows
    through the frozen SD3 DiT and serves as the queries of the DiNA-style
    ``LatentRewardGridHead``.  Two changes vs exp4: (1) the visual snapshots at [4, 8, 12]
    are spatially average-pooled (e.g. 64x64 -> 16x16 = 256 tokens) before entering the head
    so the cross-attention context sequence is short (text snapshots stay unpooled); (2) the
    head's first attention stage is a joint attention where the reward tokens (queries)
    attend over concat(reward, visual, text) as K/V in a single softmax -- self-attention
    (reward<->reward) and cross-attention (reward->visual/text) together -- followed by the
    exp4 visual-only refinement and FFN.  The pool is parameter-free.  Trainable params are
    reward_tokens, reward_pos_embed, reward_q_proj, reward_head; the backbone is fully
    frozen.  ``forward``, ``parameter_groups``, and ``load_checkpoint_state`` are inherited
    unchanged.
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = True,
        skip_attn2: bool = False,
        pool_factor: int = 4,
        num_reward_tokens: int | None = None,
        use_pos_embed: bool = True,
        disable_side_stream: bool = False,
        side_stream_ffn: bool = False,
        freeze_q_proj: bool = False,
        head_input_tokens: bool = False,
    ):
        super().__init__(
            backbone,
            visual_layers=visual_layers,
            text_layers=text_layers,
            layer_index_base=layer_index_base,
            num_transformer_layers=num_transformer_layers,
            reward_grid_h=reward_grid_h,
            reward_grid_w=reward_grid_w,
            vis_h=vis_h,
            vis_w=vis_w,
            width=width,
            out_dim=out_dim,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
            skip_attn2=skip_attn2,
            num_reward_tokens=num_reward_tokens,
            use_pos_embed=use_pos_embed,
            disable_side_stream=disable_side_stream,
            side_stream_ffn=side_stream_ffn,
            freeze_q_proj=freeze_q_proj,
            head_input_tokens=head_input_tokens,
        )
        self.pool_factor = int(pool_factor)
        self.visual_pool = _SpatialPool2d(self.pool_factor)

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["architecture"] = "latent_reward_grid_pool"
        state["pool_factor"] = self.pool_factor
        return state


class SD3LatentRewardGridPoolNoPEModel(SD3LatentRewardGridPoolModel):
    """exp8: exp7 with a small position-free reward-token set.

    Identical to SD3LatentRewardGridPoolModel (exp7) -- pooled visual context + joint
    self/cross attention head -- with two changes: the reward tokens are a flat learnable
    set of ``num_reward_tokens`` (default 32, vs exp7's 256), and the spatially interpolated
    ``reward_pos_embed`` is dropped (``use_pos_embed=False``).  Everything else (side-stream,
    snapshots at [4, 8, 12], head, training) is aligned with exp7.  ``forward``,
    ``parameter_groups``, and ``load_checkpoint_state`` are inherited (the parent gates the
    now-absent position embedding).
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        out_dim: int = 1,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = True,
        skip_attn2: bool = False,
        pool_factor: int = 4,
        num_reward_tokens: int = 32,
        disable_side_stream: bool = False,
        side_stream_ffn: bool = False,
        freeze_q_proj: bool = False,
        head_input_tokens: bool = False,
    ):
        super().__init__(
            backbone,
            visual_layers=visual_layers,
            text_layers=text_layers,
            layer_index_base=layer_index_base,
            num_transformer_layers=num_transformer_layers,
            reward_grid_h=reward_grid_h,
            reward_grid_w=reward_grid_w,
            vis_h=vis_h,
            vis_w=vis_w,
            width=width,
            out_dim=out_dim,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
            skip_attn2=skip_attn2,
            pool_factor=pool_factor,
            num_reward_tokens=num_reward_tokens,
            use_pos_embed=False,
            disable_side_stream=disable_side_stream,
            side_stream_ffn=side_stream_ffn,
            freeze_q_proj=freeze_q_proj,
            head_input_tokens=head_input_tokens,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["architecture"] = "latent_reward_grid_pool_nope"
        return state


class SD3LatentRewardGridPoolNoPEMultiHeadModel(SD3LatentRewardGridPoolNoPEModel):
    """exp10/exp11: exp8 trunk with multiple reward targets (HPS / PickScore / ImageReward).

    Shares the entire exp8 trunk -- 32 position-free reward tokens, frozen SD3 side-stream,
    pooled visual context + FiLM at layers [4, 8, 12], joint self/cross-attention head and FFN.
    Only the final readout is per-reward: each head owns a small MLP
    (``Linear(W -> W/head_hidden_factor) -> GELU -> Linear(-> 1)``) applied to the shared
    per-token features, then mean-pooled over the reward tokens to one scalar per image.
    ``forward`` returns a ``dict[head_name -> [B, group_size]]`` consumed by ``multihead_loss``.

    The inherited single-scalar ``reward_head.head`` (Linear(W -> 1)) is unused here and is
    replaced with ``nn.Identity`` in ``__init__`` so it carries no trainable parameters
    (otherwise DDP aborts on a parameter that never participates in the loss).
    """

    def __init__(
        self,
        backbone: SD3RewardBackbone,
        *,
        head_names: Iterable[str],
        head_hidden_factor: int = 2,
        visual_layers: Iterable[int],
        text_layers: Iterable[int],
        layer_index_base: int = 0,
        num_transformer_layers: int | None = None,
        reward_grid_h: int = 16,
        reward_grid_w: int = 16,
        vis_h: int = 64,
        vis_w: int = 64,
        width: int = -1,
        num_attn_heads: int = 8,
        dropout: float = 0.0,
        use_proj_in: bool = False,
        use_self_attn: bool = True,
        skip_attn2: bool = False,
        pool_factor: int = 4,
        num_reward_tokens: int = 32,
        disable_side_stream: bool = False,
        side_stream_ffn: bool = False,
        freeze_q_proj: bool = False,
        head_input_tokens: bool = False,
    ):
        super().__init__(
            backbone,
            visual_layers=visual_layers,
            text_layers=text_layers,
            layer_index_base=layer_index_base,
            num_transformer_layers=num_transformer_layers,
            reward_grid_h=reward_grid_h,
            reward_grid_w=reward_grid_w,
            vis_h=vis_h,
            vis_w=vis_w,
            width=width,
            out_dim=1,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            use_proj_in=use_proj_in,
            use_self_attn=use_self_attn,
            skip_attn2=skip_attn2,
            pool_factor=pool_factor,
            num_reward_tokens=num_reward_tokens,
            disable_side_stream=disable_side_stream,
            side_stream_ffn=side_stream_ffn,
            freeze_q_proj=freeze_q_proj,
            head_input_tokens=head_input_tokens,
        )

        self.head_names = list(head_names)
        self.num_heads = len(self.head_names)
        if self.num_heads == 0:
            raise ValueError("head_names must be non-empty for the multi-head model")
        self.head_hidden_factor = int(head_hidden_factor)

        # The shared trunk emits per-token features of this width.
        trunk_width = self.reward_head.head.in_features
        hidden = max(1, trunk_width // self.head_hidden_factor)
        self.score_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(trunk_width, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                for _ in self.head_names
            ]
        )

        # Drop the inherited single-scalar readout: exp10 reads ``reward_head.features``
        # and routes through ``score_mlps`` instead, so ``reward_head.head`` never
        # participates in the loss. Leaving it as a trainable Linear makes DDP abort with
        # "parameters that were not used in producing loss". Replacing it with Identity
        # removes those params entirely (no find_unused_parameters needed).
        self.reward_head.head = nn.Identity()

    def _compute_outputs(
        self,
        reward_states: torch.Tensor,
        visual_features: list[torch.Tensor],
        text_features: list[torch.Tensor] | None,
        temb: torch.Tensor,
        batch_size: int,
        group_size: int,
    ) -> dict[str, torch.Tensor]:
        feats = self.reward_head.features(reward_states, visual_features, text_features, temb)
        return {
            name: self.score_mlps[k](feats).mean(dim=1).squeeze(-1).reshape(batch_size, group_size)
            for k, name in enumerate(self.head_names)
        }

    def parameter_groups(self, *, head_lr: float, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        groups = super().parameter_groups(head_lr=head_lr, backbone_lr=backbone_lr)
        groups[0]["params"] = list(groups[0]["params"]) + list(self.score_mlps.parameters())
        return groups

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["architecture"] = "latent_reward_grid_pool_nope_multihead"
        state["head_names"] = list(self.head_names)
        state["head_hidden_factor"] = self.head_hidden_factor
        state["score_mlps"] = self.score_mlps.state_dict()
        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        super().load_checkpoint_state(state)
        if "score_mlps" in state:
            self.score_mlps.load_state_dict(state["score_mlps"])
