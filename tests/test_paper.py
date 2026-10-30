"""Scientific boundaries of the real paper reproduction entry point."""
import json
import shutil

import numpy as np
import pytest

from cmp.paper_data import DATA_DIR, frozen_plan, load_data
from cmp.paper_mic import compare_offscale, endpoint
from cmp.reproduce import run


@pytest.fixture(scope="module")
def bundle():
    return load_data()


def test_real_cohorts_and_negative_controls(bundle):
    for name, rows, groups, positives in (("development", 1605, 43, 961), ("balf", 168, 8, 21)):
        c = bundle["cohorts"][name]
        assert c["arrays"].n_samples == rows
        assert c["metadata"].group_id.nunique() == groups
        assert c["arrays"].y.sum() == positives
        assert not c["arrays"].blocks["glu"][:, 0].any()
        assert not c["arrays"].blocks["ntu"][:, 0].any()
    balf = bundle["cohorts"]["balf"]
    controls = balf["metadata"].concentration_level.eq(0).to_numpy()
    assert controls.sum() == 24
    assert (balf["arrays"].y[controls] == 0).sum() == 12


def test_anonymous_renaming_cannot_change_frozen_roles(bundle):
    metadata = bundle["cohorts"]["development"]["metadata"].copy()
    splits = bundle["splits"].copy()
    mapping = {name: f"renamed-{i}" for i, name in enumerate(reversed(sorted(metadata.group_id.unique())))}
    metadata["group_id"] = metadata.group_id.map(mapping)
    splits["group_id"] = splits.group_id.map(mapping)
    plan = frozen_plan(splits, metadata, bundle["cohorts"]["development"]["arrays"].y)
    for old, new in zip(bundle["plan"], plan):
        assert old[:2] == new[:2]
        for a, b in zip(old[2:], new[2:]):
            np.testing.assert_array_equal(a, b)
    splits.loc[0, "role"] = "early_stop"
    with pytest.raises(ValueError, match="Invalid grouped"):
        frozen_plan(splits, metadata, bundle["cohorts"]["development"]["arrays"].y)


def test_corrupt_packaged_input_is_rejected(tmp_path):
    copy = tmp_path / "data"
    shutil.copytree(DATA_DIR, copy)
    with (copy / "balf.tsv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_data(copy)


@pytest.mark.parametrize("calls,control,display", [
    ([1, 1, 0, 1, 0], 1, "16"),
    ([1, 1, 0, 0, 1], 1, "NR"),
    ([0, 0, 0, 0, 0], 1, "<=1"),
    ([1, 1, 1, 1, 1], 1, ">16"),
    ([0, 0, 0, 0, 0], 0, "NR"),
    ([1, None, 0, 0, 0], 1, "NR"),
])
def test_mic_bounds_controls_and_skipped_concentrations(calls, control, display):
    assert endpoint([1, 2, 4, 8, 16], calls, control)["display"] == display


@pytest.mark.parametrize("tv,ts,rv,rs,expected", [
    (8, ">", 32, ">", (0, True)),
    (32, "=", 32, ">", (-1, True)),
    (32, ">=", 32, ">", (0, True)),
    (1, "=", 2, "=", (-1, True)),
    (1, "=", 2, "<=", (0, True)),
    (.5, "<=", 1, "<=", (0, True)),
])
def test_offscale_examples_and_paper_pairs(tv, ts, rv, rs, expected):
    assert compare_offscale(tv, ts, rv, rs) == expected


def test_saved_recomputes_complete_paper_results(tmp_path):
    output = tmp_path / "saved"
    result = run("saved", output)
    assert result["models_trained"] == 0
    assert result["mic"]["development"] == {
        "series": 228, "reportable": 194, "status_counts": {"=": 82, "<=": 51, ">": 61, "NR": 34},
        "numeric_pairs": 82, "exact_n": 51, "ea_n": 72}
    assert result["mic"]["balf"] == {
        "samples": 8, "participant_groups": 6, "reportable": 4, "comparable": 4,
        "comparable_participant_groups": 3, "exact_after_offscale_n": 3, "ea_n": 4}
    assert json.loads((output / "reconciliation.json").read_text())["passed"]
    assert not (output / "models").exists()
    with pytest.raises(FileExistsError):
        run("saved", output)
