"""Time-series scaling fitted on training observations only."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from cmp.dataset import CMPArrays

@dataclass
class FittedStats:
    """Per-block standardization stats. Only time-series blocks are standardized;
    one-hot blocks pass through unchanged but are listed for completeness."""
    means: dict[str, np.ndarray] = field(default_factory=dict)
    stds: dict[str, np.ndarray] = field(default_factory=dict)
    standardize: dict[str, bool] = field(default_factory=dict)



def _standardize(
    arr: CMPArrays,
    stats: FittedStats,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, mat in arr.blocks.items():
        if stats.standardize.get(name, False):
            mean = stats.means[name]
            std = stats.stds[name]
            out[name] = ((mat - mean) / std).astype(np.float32, copy=False)
        else:
            out[name] = mat.astype(np.float32, copy=False)
    return out



def _fit_stats(arr: CMPArrays, ts_blocks: Sequence[str]) -> FittedStats:
    stats = FittedStats()
    for name, mat in arr.blocks.items():
        if name in ts_blocks:
            mean = mat.mean(axis=0, dtype=np.float64)
            std = mat.std(axis=0, dtype=np.float64)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            stats.means[name] = mean.astype(np.float32)
            stats.stds[name] = std
            stats.standardize[name] = True
        else:
            stats.standardize[name] = False
    return stats
