"""Within-row centering and finite differences, adapted from research code."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cmp.dataset import load_cmp_tsv

REPRESENTATIONS = ("raw", "centered", "first_derivative", "second_derivative")


def transform_channel(values, representation, time_step=1.0):
    """Return float32 features; differences use a declared uniform time step."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite rows with at least three time points")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    if representation == "raw":
        result = x
    elif representation == "centered":
        result = x - x[:, :1]
    elif representation == "first_derivative":
        result = np.diff(x, axis=1) / time_step
    elif representation == "second_derivative":
        result = np.diff(x, n=2, axis=1) / time_step**2
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return result.astype(np.float32)


def load_representation(path, representation, time_step=1.0):
    arrays = load_cmp_tsv(path, has_label=True)
    native = pd.read_csv(path, sep="\t")
    for channel in ("glu", "ntu"):
        arrays.blocks[channel] = transform_channel(
            native[arrays.columns[channel]].to_numpy(), representation, time_step
        )
        arrays.columns[channel] = [
            f"{channel}_{representation}_{i + 1}"
            for i in range(arrays.blocks[channel].shape[1])
        ]
    arrays.feature_names = [column for columns in arrays.columns.values() for column in columns]
    return arrays
