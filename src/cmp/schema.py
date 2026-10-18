"""Feature schema: a single source of truth mapping TSV columns to semantic blocks.

A schema describes the label column and an ordered list of blocks. Each block is
either a categorical one-hot group or a fixed-length time series. The schema is
data-driven (YAML), but the loaded :class:`FeatureSchema` is what the rest of
``cmp`` uses for column resolution, validation, and hashing for reports.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import yaml

BlockKind = Literal["onehot", "timeseries"]
_DEFAULT_SCHEMA = Path(__file__).with_name("feature_schema.yaml")


@dataclass(frozen=True)
class BlockSpec:
    name: str
    kind: BlockKind
    prefix: str
    exclude_prefix: str | None = None
    mutually_exclusive: bool = False
    time_axis: str | None = None
    ordinal_hint: bool = False

    def matches(self, column: str) -> bool:
        if not column.startswith(self.prefix):
            return False
        if self.exclude_prefix and column.startswith(self.exclude_prefix):
            return False
        return True


@dataclass(frozen=True)
class FeatureSchema:
    label: str
    blocks: tuple[BlockSpec, ...]
    version: int = 1

    def block_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.blocks)

    def get_block(self, name: str) -> BlockSpec:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"Unknown block: {name!r}")

    def resolve_columns(self, columns: Sequence[str]) -> dict[str, list[str]]:
        """Map each block to the subset of ``columns`` that belong to it.

        Columns are kept in the order they appear in ``columns`` to make
        downstream tensor slicing deterministic.
        """
        result: dict[str, list[str]] = {b.name: [] for b in self.blocks}
        for col in columns:
            for block in self.blocks:
                if block.matches(col):
                    result[block.name].append(col)
                    break
        return result

    def hash(self) -> str:
        payload = json.dumps(
            {"label": self.label, "version": self.version,
             "blocks": [asdict(b) for b in self.blocks]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_schema(path: str | Path | None = None) -> FeatureSchema:
    p = Path(path) if path else _DEFAULT_SCHEMA
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    blocks = tuple(BlockSpec(**b) for b in raw["blocks"])
    return FeatureSchema(label=raw["label"], blocks=blocks, version=int(raw.get("version", 1)))


def validate_dataframe_columns(
    schema: FeatureSchema,
    columns: Sequence[str],
    *,
    require_label: bool = True,
) -> dict[str, list[str]]:
    """Validate that ``columns`` cover the label and at least one column per block.

    Returns a dict mapping block name -> ordered list of columns. Raises
    :class:`ValueError` if validation fails. Extra columns that match no block
    (and are not the label) are also reported.
    """
    cols = list(columns)
    if require_label and schema.label not in cols:
        raise ValueError(f"Label column '{schema.label}' missing from data")

    mapping = schema.resolve_columns([c for c in cols if c != schema.label])

    empty = [name for name, c in mapping.items() if not c]
    if empty:
        raise ValueError(
            f"Schema blocks have no matching columns in data: {empty}. "
            "Check column prefixes in your TSV against the schema."
        )

    assigned = {c for cols_ in mapping.values() for c in cols_}
    unknown = [c for c in cols if c != schema.label and c not in assigned]
    if unknown:
        raise ValueError(
            f"Found columns not covered by any block: {unknown}. "
            "Update the schema or remove the columns."
        )
    return mapping


def block_iterator(schema: FeatureSchema, mapping: dict[str, list[str]]) -> Iterable[tuple[BlockSpec, list[str]]]:
    for block in schema.blocks:
        yield block, mapping[block.name]
