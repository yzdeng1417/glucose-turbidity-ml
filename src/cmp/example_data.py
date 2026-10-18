"""Packaged, fixed synthetic example; no private inputs or calibration required."""
from __future__ import annotations

import hashlib
from importlib.resources import files
from io import BytesIO
import json

import pandas as pd

from cmp.synthetic import make_synthetic

DATASETS = ("calibrated", "toy")


def load_demo_data(dataset="calibrated", seed=2026):
    """Return synthetic features, metadata and public provenance.

    The calibrated example is frozen: seed affects downstream training, not
    these rows or their split. The optional toy example is generated afresh.
    """
    if dataset == "toy":
        frame, metadata = make_synthetic(seed=seed)
        return frame, metadata, {
            "dataset_id": "equation-toy", "data_kind": "synthetic",
            "generator": "cmp.synthetic.make_synthetic", "generation_seed": seed,
            "time_points": list(range(1, 9)), "time_unit": "toy index", "time_step": 1.0,
            "signal_unit": "arbitrary signal units",
        }
    if dataset != "calibrated":
        raise ValueError(f"Unknown dataset: {dataset}")
    directory = files("cmp").joinpath("data", "calibrated-v1")
    manifest = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
    content = {}
    for name in ("synthetic.tsv", "synthetic_metadata.tsv"):
        data = directory.joinpath(name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Packaged synthetic data checksum mismatch: {name}")
        content[name] = pd.read_csv(BytesIO(data), sep="\t")
    frame = content["synthetic.tsv"]
    metadata = content["synthetic_metadata.tsv"].rename(columns={"synthetic_split": "split"})
    if len(frame) != len(metadata) or len(frame) != manifest["rows"]:
        raise ValueError("Packaged features and metadata must have matching row counts")
    if metadata.isna().any().any() or not metadata.row_id.is_unique:
        raise ValueError("Synthetic metadata must have unique row IDs and no missing values")
    if not metadata.row_id.str.fullmatch(r"SIM_R\d{5}").all() or not metadata.group_id.str.fullmatch(r"SIM_G\d{3}").all():
        raise ValueError("Expected exclusively synthetic row and group IDs")
    if not metadata.label_origin.eq("simulated_endpoint_label").all():
        raise ValueError("The packaged example must use simulated endpoint labels")
    if not metadata.split.isin(["development", "heldout"]).all():
        raise ValueError("Invalid synthetic split membership")
    if not metadata.groupby("group_id").split.nunique().eq(1).all():
        raise ValueError("Packaged development / holdout groups overlap")
    if metadata.group_id.nunique() != manifest["groups"]:
        raise ValueError("Packaged group count disagrees with the manifest")
    for split in ("development", "heldout"):
        selected = metadata.split.eq(split)
        if selected.sum() != manifest[f"{split}_rows"] or metadata.loc[selected, "group_id"].nunique() != manifest[f"{split}_groups"]:
            raise ValueError("Packaged split counts disagree with the manifest")
    return frame, metadata, manifest
