import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cmp.demo import run  # noqa: E402
from cmp.models import MultiTowerClassifier, MultiTowerConfig  # noqa: E402
from cmp.neural import stopping_split  # noqa: E402


def test_internal_stop_groups_do_not_overlap():
    groups = np.repeat(np.arange(20), 4)
    y = np.tile([0, 1, 1, 0], 20)
    fit, stop, _ = stopping_split(y, groups, seed=2026)
    assert not set(groups[fit]) & set(groups[stop])


def test_model_returns_logits_and_normalized_block_weights():
    torch.manual_seed(1)
    dimensions = {"species": 3, "antibiotic": 2, "concentration": 4, "glu": 8, "ntu": 8}
    model = MultiTowerClassifier(MultiTowerConfig(block_dims=dimensions)).eval()
    batch = {name: torch.zeros(4, width) for name, width in dimensions.items()}
    logits, attention = model(batch, return_attention=True)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    assert attention.shape == (4, 5)
    torch.testing.assert_close(attention.sum(axis=1), torch.ones(4))


def test_neural_demo_trains_with_three_disjoint_roles(tmp_path):
    out = tmp_path / "neural"
    result = run(out, model="neural", epochs=2)
    audit = pd.read_csv(out / "split_audit.tsv", sep="\t")
    assert set(audit.role) == {"optimizer", "early_stop", "outer_validation"}
    assert audit.groupby(["fold", "group_id"]).role.nunique().eq(1).all()
    assert len(list(out.glob("*.pt"))) == 4
    assert np.isfinite(result["synthetic_holdout"]["roc_auc"])
