import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cmp.dataset import load_cmp_tsv
from cmp.demo import run
from cmp.evaluation import select_threshold
from cmp.scaling import _fit_stats, _standardize
from cmp.synthetic import make_synthetic
from cmp.workflow import grouped_splits, make_baseline, subset


def test_each_group_is_evaluated_once_without_overlap():
    frame, metadata = make_synthetic()
    groups = metadata.group_id.to_numpy()
    folds = grouped_splits(frame.growth.to_numpy(), groups)
    seen = []
    for train, validation in folds:
        assert not set(groups[train]) & set(groups[validation])
        seen.extend(validation)
    assert sorted(seen) == list(range(len(frame)))


def test_baseline_scaling_cannot_fit_validation_extremes():
    x = np.array([[1, 2], [2, 3], [4, 2], [3, 1]], dtype=float)
    pipeline = make_baseline().fit(x, [0, 0, 1, 1])
    means = pipeline.named_steps["standardscaler"].mean_.copy()
    pipeline.predict_proba(np.full((2, 2), 1e6))
    np.testing.assert_array_equal(means, x.mean(axis=0))
    np.testing.assert_array_equal(pipeline.named_steps["standardscaler"].mean_, means)


def test_neural_scaling_only_uses_optimizer_rows(tmp_path):
    frame, _ = make_synthetic()
    path = tmp_path / "data.tsv"
    frame.to_csv(path, sep="\t", index=False)
    arrays = load_cmp_tsv(path)
    arrays.blocks["glu"][16:] = 1e6
    fit = subset(arrays, np.arange(16))
    stats = _fit_stats(fit, ("glu", "ntu"))
    np.testing.assert_allclose(stats.means["glu"], fit.blocks["glu"].mean(axis=0, dtype=np.float64), rtol=1e-6)
    _standardize(subset(arrays, np.arange(16, 24)), stats)
    assert np.max(np.abs(stats.means["glu"])) < 10
    assert not stats.standardize["species"]


def test_threshold_enumeration_ties_and_invalid_scores():
    threshold, ba = select_threshold(np.array([0, 0, 1, 1]), np.array([0.2, 0.2, 0.8, 0.8]))
    assert threshold == 0.5 and ba == 1
    with pytest.raises(ValueError):
        select_threshold(np.array([0, 1]), np.array([np.nan, 0.5]))


def test_complete_demo_hashes_split_audit_and_overwrite_guard(tmp_path):
    out = tmp_path / "demo"
    result = run(out)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert result["data_kind"] == "synthetic"
    assert result["threshold"]["heldout_used"] is False
    assert manifest["counts"] == {"development_rows": 288, "heldout_rows": 96, "development_groups": 36, "heldout_groups": 12}
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    audit = pd.read_csv(out / "split_audit.tsv", sep="\t")
    heldout = pd.read_csv(out / "synthetic_holdout.tsv", sep="\t")
    assert not set(audit.group_id) & set(heldout.group_id)
    assert audit.groupby(["fold", "group_id"]).role.nunique().eq(1).all()
    assert (audit.role == "outer_validation").sum() == 288
    with pytest.raises(FileExistsError):
        run(out)


def test_heldout_labels_cannot_change_model_scores_or_selected_threshold(tmp_path, monkeypatch):
    original = make_synthetic
    first = tmp_path / "original"
    result = run(first)

    def flip_heldout_labels(**kwargs):
        frame, metadata = original(**kwargs)
        rows = metadata.split.eq("heldout")
        frame.loc[rows, "growth"] = 1 - frame.loc[rows, "growth"]
        return frame, metadata

    monkeypatch.setattr("cmp.demo.make_synthetic", flip_heldout_labels)
    changed = tmp_path / "changed_labels"
    changed_result = run(changed)
    assert result["threshold"] == changed_result["threshold"]
    for name in ("development_oof.tsv", "synthetic_holdout.tsv"):
        a = pd.read_csv(first / name, sep="\t")
        b = pd.read_csv(changed / name, sep="\t")
        np.testing.assert_array_equal(a.score, b.score)
