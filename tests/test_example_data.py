from importlib.resources import files
import shutil

import numpy as np
import pytest

from cmp.example_data import load_demo_data


def test_frozen_example_seed_does_not_change_rows_or_split():
    frame, meta, info = load_demo_data()
    again, again_meta, again_info = load_demo_data(seed=9001)
    assert frame.equals(again) and meta.equals(again_meta) and info == again_info
    assert len(frame) == 2253 and meta.group_id.nunique() == 60
    assert np.isfinite(frame.to_numpy()).all()
    assert set(meta.label_origin) == {"simulated_endpoint_label"}
    assert meta.groupby("group_id").split.nunique().eq(1).all()
    for column, prefix in (("species", "species_"), ("level", "antibiotic_concentration_level")):
        assert all(frame.loc[i, prefix + str(value)] == 1 for i, value in enumerate(meta[column]))
    for context in ("amp_sul", "amk_lev"):
        selected = meta.context.eq(context)
        assert selected.any()
        for drug in context.split("_"):
            assert frame.loc[selected, "antibiotic_" + drug].eq(1).all()


def test_modified_packaged_data_is_rejected(tmp_path, monkeypatch):
    source = files("cmp").joinpath("data", "calibrated-v1")
    target = tmp_path / "data/calibrated-v1"
    shutil.copytree(source, target)
    with (target / "synthetic.tsv").open("ab") as stream:
        stream.write(b"\n")
    monkeypatch.setattr("cmp.example_data.files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_demo_data()


def test_toy_generator_remains_explicit_and_seeded():
    frame, meta, info = load_demo_data("toy")
    changed, _, _ = load_demo_data("toy", seed=9001)
    assert len(frame) == 384 and meta.group_id.nunique() == 48
    assert info["time_step"] == 1.0
    assert not frame.equals(changed)
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_demo_data("unknown")
