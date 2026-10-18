"""Tensor conversion and deterministic random seeds."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

class _BlockDataset(Dataset):
    """Wrap arrays of per-block features + targets into a torch Dataset."""

    def __init__(self, blocks: dict[str, np.ndarray], y: np.ndarray | None) -> None:
        any_b = next(iter(blocks.values()))
        self._n = int(any_b.shape[0])
        self._blocks = {k: torch.from_numpy(v.astype(np.float32, copy=False)) for k, v in blocks.items()}
        self._y = torch.from_numpy(y.astype(np.float32, copy=False)) if y is not None else None

    def __len__(self) -> int:  # noqa: D401
        return self._n

    def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], Tensor]:
        item = {k: v[idx] for k, v in self._blocks.items()}
        target = self._y[idx] if self._y is not None else torch.tensor(0.0)
        return item, target



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
