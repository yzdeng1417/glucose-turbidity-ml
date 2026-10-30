"""Verified real paper inputs and frozen, anonymized row roles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cmp.dataset import load_cmp_tsv
from cmp.representations import load_representation

DATA_DIR = Path(__file__).with_name("data") / "paper-v1"
SEEDS = (42, 43, 44, 45, 46)
SCORE_COLUMNS = ["score", *(f"member_score_{seed}" for seed in SEEDS)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip", keep_default_na=False)


def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["dataset_id"] != "paper-centered-real-v1" or manifest["synthetic"] is not False:
        raise ValueError("Expected the frozen real paper dataset")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(data_dir / name) != expected:
            raise ValueError(f"Packaged data checksum mismatch: {name}")
    protocol = json.loads((data_dir / "protocol.json").read_text())
    policy = json.loads((data_dir / "reconciliation_policy.json").read_text())
    if protocol["threshold"]["value"] != 0.5 or protocol["ensemble"]["member_seeds"] != list(SEEDS):
        raise ValueError("Frozen model protocol changed")
    cohorts = {}
    for name, n_rows, n_groups, positives in (("development", 1605, 43, 961), ("balf", 168, 8, 21)):
        path = data_dir / f"{name}.tsv"
        raw = load_cmp_tsv(path, has_label=True)
        # The original training reader uses pandas' default float parser here.
        # Center in float64 before casting to float32; preserve the zero T1.
        arrays = load_representation(path, "centered", time_step=15.0)
        metadata = pd.read_csv(data_dir / f"{name}_metadata.tsv", sep="\t", dtype=str, keep_default_na=False)
        for column in ("row_index0", "concentration_level", "replicate_index"):
            metadata[column] = metadata[column].astype(int)
        metadata["concentration_ug_ml"] = metadata.concentration_ug_ml.astype(float)
        saved = read_table(data_dir / f"{name}_saved_predictions.tsv")
        if (raw.n_samples, metadata.group_id.nunique(), int(raw.y.sum())) != (n_rows, n_groups, positives):
            raise ValueError(f"Cohort counts changed: {name}")
        for frame in (metadata, saved):
            if (len(frame) != n_rows or frame.row_index0.tolist() != list(range(n_rows))
                    or frame.row_id.duplicated().any()):
                raise ValueError(f"Row coverage changed: {name}")
        if not np.array_equal(metadata[["row_id", "group_id"]], saved[["row_id", "group_id"]]):
            raise ValueError(f"Prediction identities differ: {name}")
        if not np.array_equal(raw.y, saved.growth):
            raise ValueError(f"Manual labels differ: {name}")
        if (not np.isfinite(saved[SCORE_COLUMNS].to_numpy()).all()
                or not saved[SCORE_COLUMNS].ge(0).all().all() or not saved[SCORE_COLUMNS].le(1).all().all()
                or not saved.threshold.eq(.5).all()
                or not np.array_equal(saved.prediction, (saved.score >= .5).astype(int))):
            raise ValueError(f"Invalid frozen predictions: {name}")
        levels = np.argmax(raw.blocks["concentration"], axis=1)
        if not np.array_equal(levels, metadata.concentration_level):
            raise ValueError(f"Concentration context differs: {name}")
        for channel in ("glu", "ntu"):
            if arrays.blocks[channel].shape != (n_rows, 8) or np.any(arrays.blocks[channel][:, 0] != 0):
                raise ValueError(f"Invalid centered time support: {name}")
        cohorts[name] = {"arrays": arrays, "metadata": metadata, "saved": saved}
    if set(cohorts["development"]["metadata"].group_id) & set(cohorts["balf"]["metadata"].group_id):
        raise ValueError("Development/BALF group overlap")
    splits = read_table(data_dir / "split_manifest.tsv")
    plan = frozen_plan(splits, cohorts["development"]["metadata"], cohorts["development"]["arrays"].y)
    return {"manifest": manifest, "protocol": protocol, "policy": policy, "cohorts": cohorts,
            "splits": splits, "plan": plan, "directory": data_dir,
            "comparators": read_table(data_dir / "balf_comparators.tsv")}


def frozen_plan(splits, metadata, labels):
    """Consume row roles verbatim; group names must never drive re-splitting."""
    expected = {(fold, seed) for fold in range(6) for seed in SEEDS}
    if (len(splits) != len(metadata) * 30 or splits.duplicated(["outer_fold", "member_seed", "row_index0"]).any()
            or set(zip(splits.outer_fold, splits.member_seed)) != expected):
        raise ValueError("Incomplete or duplicate frozen split roles")
    plan, outer_membership = [], {}
    for fold in (1, 2, 3, 4, 5, 0):
        for seed in SEEDS:
            frame = splits.loc[splits.outer_fold.eq(fold) & splits.member_seed.eq(seed)].sort_values("row_index0")
            roles = {"optimizer", "early_stop", "outer_validation"} if fold else {"optimizer", "early_stop"}
            if (frame.row_index0.tolist() != list(range(len(metadata))) or set(frame.role) != roles
                    or not np.array_equal(frame.group_id, metadata.group_id)
                    or frame.groupby("group_id").role.nunique().max() != 1):
                raise ValueError(f"Invalid grouped role assignment: fold {fold}, seed {seed}")
            indices = {role: frame.loc[frame.role.eq(role), "row_index0"].to_numpy(dtype=int) for role in roles}
            if any(len(np.unique(labels[idx])) != 2 for idx in indices.values()):
                raise ValueError("Each optimizer, stopping and validation partition requires both classes")
            evaluation = indices.get("outer_validation")
            if fold:
                if seed == SEEDS[0]:
                    outer_membership[fold] = evaluation
                elif not np.array_equal(evaluation, outer_membership[fold]):
                    raise ValueError("Outer validation changed across members")
            plan.append((fold, seed, indices["optimizer"], indices["early_stop"], evaluation))
    if sorted(np.concatenate(list(outer_membership.values())).tolist()) != list(range(len(metadata))):
        raise ValueError("Every development row must be evaluated in exactly one fold")
    return plan
