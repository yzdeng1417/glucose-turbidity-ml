"""Multi-tower classifier with block-aware encoders and a choice of fusion.

Blocks are encoded independently to expose block-level structure to the rest of
the pipeline (training, attribution, reporting). Two fusion strategies are
available: a plain concat-MLP, and a block-attention fusion that produces
per-sample softmax weights over blocks. These weights are internal model
quantities, not causal explanations or validated feature importance. Both share the same prediction head, so swapping fusion never
changes the loss / training loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn


FusionKind = Literal["concat", "attention"]


@dataclass
class BlockEncoderConfig:
    out_dim: int = 16
    dropout: float = 0.1
    # For timeseries blocks
    hidden_channels: int = 16
    kernel_size: int = 3
    # Stack depth of the conv block in the curve encoder.
    curve_conv_depth: int = 2
    # Optional BatchNorm1d after each convolution; disabled by default.
    curve_batch_norm: bool = False
    # Concatenate first-order temporal differences as a second curve channel
    # as an optional inductive bias; disabled by default.
    curve_use_diff_channel: bool = False


@dataclass
class FusionConfig:
    kind: FusionKind = "attention"
    hidden_dim: int = 64
    dropout: float = 0.2
    # Multi-head attention only used when kind == "attention".
    n_heads: int = 2


@dataclass
class MultiTowerConfig:
    block_dims: dict[str, int] = field(default_factory=dict)
    timeseries_blocks: tuple[str, ...] = ("glu", "ntu")
    tabular_blocks: tuple[str, ...] = ("species", "antibiotic", "concentration")
    encoder: BlockEncoderConfig = field(default_factory=BlockEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


class _TabularTower(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _CurveEncoder(nn.Module):
    """1D conv stack over time + global pool to a fixed embedding.

    Input shape: ``[B, T]`` with T = number of time points (e.g. 8). The raw
    curve can optionally be augmented with its first-order temporal difference as a second
    input channel (zero-padded at t=0) so the conv stack can model both
    instantaneous levels and local rate-of-change without needing a deeper
    receptive field — important when T is small.

    Pooling concatenates mean + max so the head sees both trend (mean) and
    extremum (max) information.
    """

    def __init__(
        self,
        t_len: int,
        hidden_channels: int,
        out_dim: int,
        dropout: float,
        kernel_size: int = 3,
        depth: int = 2,
        batch_norm: bool = False,
        use_diff_channel: bool = True,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        in_ch = 2 if use_diff_channel else 1
        layers: list[nn.Module] = []
        prev = in_ch
        for _ in range(max(1, depth)):
            layers.append(nn.Conv1d(prev, hidden_channels, kernel_size=kernel_size, padding=pad))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.GELU())
            prev = hidden_channels
        self.conv = nn.Sequential(*layers)
        self.use_diff_channel = use_diff_channel
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_diff_channel:
            diff = torch.zeros_like(x)
            diff[:, 1:] = x[:, 1:] - x[:, :-1]
            xt = torch.stack([x, diff], dim=1)  # [B, 2, T]
        else:
            xt = x.unsqueeze(1)
        h = self.conv(xt)  # [B, C, T]
        mean = h.mean(dim=-1)
        mx, _ = h.max(dim=-1)
        feat = torch.cat([mean, mx], dim=-1)
        return self.proj(feat)


class _BlockAttention(nn.Module):
    """Multi-head soft attention over [B, K, D] block embeddings.

    Returns the pooled vector and per-block attention weights averaged across
    heads (so the report still shows one weight per block). Multiple heads give
    the model freedom to weight blocks differently per latent subspace, but
    these weights do not establish causal or biological contribution.
    """

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embedding dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, blocks: Tensor) -> tuple[Tensor, Tensor]:
        # blocks: [B, K, D]
        B, K, _ = blocks.shape
        k = self.key(blocks).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, K, d]
        v = self.value(blocks).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.query.view(1, self.n_heads, 1, self.head_dim).expand(B, -1, -1, -1)  # [B, H, 1, d]
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2) * self._scale  # [B, H, K]
        attn = torch.softmax(scores, dim=-1)  # [B, H, K]
        pooled = torch.matmul(attn.unsqueeze(-2), v).squeeze(-2)  # [B, H, d]
        pooled = pooled.reshape(B, self.dim)
        pooled = self.out_proj(pooled)
        attn_mean = attn.mean(dim=1)  # average across heads -> [B, K]
        return pooled, attn_mean


class MultiTowerClassifier(nn.Module):
    """Per-block encoders -> fusion -> single logit (binary classification).

    The forward returns logits of shape ``[B]``. Setting ``return_attention=True``
    returns a tuple ``(logits, attn)`` where ``attn`` is either the softmax
    weights of the block attention (shape ``[B, K]``) when fusion is attention,
    or ``None`` otherwise.
    """

    def __init__(self, config: MultiTowerConfig) -> None:
        super().__init__()
        if not config.block_dims:
            raise ValueError("MultiTowerConfig.block_dims must be provided")

        self.config = config
        self.block_order: tuple[str, ...] = tuple(config.block_dims.keys())

        emb_dim = config.encoder.out_dim
        self.encoders = nn.ModuleDict()
        for name, dim in config.block_dims.items():
            if name in config.timeseries_blocks:
                self.encoders[name] = _CurveEncoder(
                    t_len=dim,
                    hidden_channels=config.encoder.hidden_channels,
                    out_dim=emb_dim,
                    dropout=config.encoder.dropout,
                    kernel_size=config.encoder.kernel_size,
                    depth=config.encoder.curve_conv_depth,
                    batch_norm=config.encoder.curve_batch_norm,
                    use_diff_channel=config.encoder.curve_use_diff_channel,
                )
            else:
                self.encoders[name] = _TabularTower(dim, emb_dim, config.encoder.dropout)

        if config.fusion.kind == "concat":
            self.attn = None
            fused_in = emb_dim * len(self.block_order)
        elif config.fusion.kind == "attention":
            self.attn = _BlockAttention(emb_dim, n_heads=config.fusion.n_heads)
            fused_in = emb_dim
        else:
            raise ValueError(f"Unknown fusion kind: {config.fusion.kind}")

        self.head = nn.Sequential(
            nn.Linear(fused_in, config.fusion.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
            nn.Linear(config.fusion.hidden_dim, 1),
        )

    def encode_blocks(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode each block and stack: returns [B, K, D] in declared order."""
        embeds = [self.encoders[name](batch[name]) for name in self.block_order]
        return torch.stack(embeds, dim=1)

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        embeds = self.encode_blocks(batch)  # [B, K, D]
        if self.attn is not None:
            pooled, attn = self.attn(embeds)
        else:
            pooled = embeds.reshape(embeds.size(0), -1)
            attn = None
        logits = self.head(pooled).squeeze(-1)
        if return_attention:
            return logits, attn
        return logits
