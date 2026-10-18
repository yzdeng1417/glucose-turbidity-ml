"""TSV → per-block tensors.

``CMPArrays`` is the lightweight, framework-agnostic representation used by
training/evaluation/explain. Conversion to a torch ``Dataset`` happens in
:mod:`cmp.neural` to keep this module importable without torch.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cmp.schema import FeatureSchema, load_schema, validate_dataframe_columns


@dataclass
class CMPArrays:
    """Per-block float matrices plus integer labels.

    Attributes:
        blocks: ordered dict ``{block_name: np.ndarray of shape [n, d_block]}``.
        columns: ordered dict ``{block_name: list of column names}``.
        y: int64 array of shape [n] (label values; absent when ``has_label=False``).
        feature_names: flat list of all feature columns in block order.
    """
    blocks: dict[str, np.ndarray]
    columns: dict[str, list[str]]
    y: np.ndarray | None
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        any_block = next(iter(self.blocks.values()))
        return int(any_block.shape[0])

    @property
    def block_dims(self) -> dict[str, int]:
        return {k: int(v.shape[1]) for k, v in self.blocks.items()}

    def stack_dense(self) -> np.ndarray:
        """Concatenate all block matrices in schema order — useful for sklearn baselines."""
        return np.concatenate([self.blocks[name] for name in self.blocks], axis=1)


def load_cmp_tsv(
    path: str | Path,
    schema: FeatureSchema | None = None,
    *,
    has_label: bool | None = None,
) -> CMPArrays:
    """Load a TSV and split columns into semantic blocks.

    By default the schema is the bundled :mod:`cmp.feature_schema`. Pass an
    alternative for experiments. ``has_label`` is auto-detected from the
    presence of ``schema.label`` in the columns when not given.
    """
    schema = schema or load_schema()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate TSV column names")
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError("Input must contain at least one row")
    cols = list(df.columns)

    label_present = schema.label in cols
    if has_label is None:
        has_label = label_present
    elif has_label and not label_present:
        raise ValueError(f"Label column '{schema.label}' is missing from {path}")

    mapping = validate_dataframe_columns(schema, cols, require_label=has_label)

    blocks: dict[str, np.ndarray] = {}
    columns: dict[str, list[str]] = {}
    feature_names: list[str] = []
    for block in schema.blocks:
        names = mapping[block.name]
        if block.kind == "timeseries":
            suffixes = [name.removeprefix(block.prefix) for name in names]
            if not all(s.isdigit() for s in suffixes):
                raise ValueError(f"Time columns must end in an integer: {block.name}")
            numbers = sorted(int(s) for s in suffixes)
            if numbers != list(range(1, len(names) + 1)) or len(names) < 3:
                raise ValueError(f"Expected contiguous time1..timeN, N>=3: {block.name}")
            names = sorted(names, key=lambda name: int(name.removeprefix(block.prefix)))
        columns[block.name] = names
        values = df[names].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {block.name}")
        if block.kind == "onehot":
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"Expected binary indicators in {block.name}")
            totals = values.sum(axis=1)
            valid = totals == 1 if block.mutually_exclusive else totals >= 1
            if not valid.all():
                raise ValueError(f"Invalid categorical encoding in {block.name}")
        blocks[block.name] = values
        feature_names.extend(names)

    time_lengths = [blocks[b.name].shape[1] for b in schema.blocks if b.kind == "timeseries"]
    if len(set(time_lengths)) > 1:
        raise ValueError("Paired time-series blocks must have equal lengths")
    if has_label and not df[schema.label].isin([0, 1]).all():
        raise ValueError("Growth labels must be exactly 0 or 1")
    y = df[schema.label].to_numpy(dtype=np.int64) if has_label else None
    return CMPArrays(blocks=blocks, columns=columns, y=y, feature_names=feature_names)


def iter_blocks(arr: CMPArrays) -> Iterable[tuple[str, np.ndarray, list[str]]]:
    for name in arr.blocks:
        yield name, arr.blocks[name], arr.columns[name]
