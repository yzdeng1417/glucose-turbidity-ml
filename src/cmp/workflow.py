"""Group-aware evaluation with a development-only operating threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmp.dataset import CMPArrays
from cmp.evaluation import select_threshold


def subset(arrays, indices):
    return CMPArrays(
        {name: values[indices] for name, values in arrays.blocks.items()},
        arrays.columns, None if arrays.y is None else arrays.y[indices], arrays.feature_names,
    )


def make_baseline():
    # The pipeline is fitted inside each training fold, including its scaler.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


def grouped_splits(y, groups, n_splits=3, seed=2026):
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or pd.isna(groups).any():
        raise ValueError("Every row must have a non-missing group ID")
    if n_splits < 2 or n_splits > len(np.unique(groups)):
        raise ValueError("Fold count must be between 2 and the number of groups")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, validation in splits:
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Group overlap between training and validation")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Each training and validation fold needs both classes")
    return splits


def neural_protocol(epochs):
    # Deliberately small public-demo settings, not a frozen research protocol.
    return {
        "model": {"encoder_out_dim": 16, "encoder_dropout": 0.1, "hidden_channels": 16,
                  "kernel_size": 3, "curve_conv_depth": 2, "curve_batch_norm": False,
                  "curve_use_diff_channel": False, "fusion": "attention", "attention_heads": 2,
                  "head_hidden_dim": 32, "head_dropout": 0.2},
        "training": {"epochs": epochs, "batch_size": 64, "lr": 0.002,
                     "weight_decay": 0.0001, "grad_clip_norm": 5.0},
        "early_stopping": {"minimum_improvement": 0.0, "patience": 5},
    }


def fit_model(arrays, groups, backend, seed, epochs, checkpoint):
    if backend == "baseline":
        model = make_baseline().fit(arrays.stack_dense(), arrays.y)
        return model, None, {"optimizer": np.arange(arrays.n_samples)}
    if backend != "neural":
        raise ValueError(f"Unknown backend: {backend}")
    from cmp.neural import fit_member, stopping_split
    fit, stop, _ = stopping_split(arrays.y, groups, seed, n_splits=4)
    model, stats, _ = fit_member(arrays, fit, stop, seed, neural_protocol(epochs), checkpoint)
    return model, stats, {"optimizer": fit, "early_stop": stop}


def score_model(model, stats, arrays, backend):
    if backend == "baseline":
        return model.predict_proba(arrays.stack_dense())[:, 1]
    from cmp.neural import predict
    return predict(model, stats, arrays, batch_size=64)


def develop(arrays, groups, backend, seed, n_splits, epochs, output):
    """Accept development data only; returns OOF scores and the final model.

    OOF operating metrics at the selected threshold are selection diagnostics.
    The separate synthetic holdout is evaluated only after this returns.
    """
    splits = grouped_splits(arrays.y, groups, n_splits, seed)
    scores = np.full(arrays.n_samples, np.nan)
    folds = np.zeros(arrays.n_samples, dtype=int)
    audit = []
    for fold, (train, validation) in enumerate(splits, 1):
        model, stats, roles = fit_model(
            subset(arrays, train), groups[train], backend, seed + fold, epochs,
            output / f"fold_{fold}.pt",
        )
        scores[validation] = score_model(model, stats, subset(arrays, validation), backend)
        folds[validation] = fold
        for role, local_indices in roles.items():
            audit.extend((fold, int(train[i]), groups[train[i]], role) for i in local_indices)
        audit.extend((fold, int(i), groups[i], "outer_validation") for i in validation)
    if not np.isfinite(scores).all() or (folds == 0).any():
        raise RuntimeError("Incomplete out-of-fold predictions")
    threshold, selection_ba = select_threshold(arrays.y, scores)
    model, stats, roles = fit_model(arrays, groups, backend, seed, epochs, output / "final.pt")
    for role, indices in roles.items():
        audit.extend((0, int(i), groups[i], role) for i in indices)
    audit = pd.DataFrame(audit, columns=["fold", "development_row_index", "group_id", "role"])
    return scores, folds, threshold, selection_ba, model, stats, audit
