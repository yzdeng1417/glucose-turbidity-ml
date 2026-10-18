import numpy as np
import pytest

from cmp.dataset import load_cmp_tsv
from cmp.representations import transform_channel
from cmp.synthetic import make_synthetic


def write_fixture(tmp_path, frame):
    path = tmp_path / "toy.tsv"
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_generator_is_reproducible_and_separates_groups():
    data, meta = make_synthetic()
    again, again_meta = make_synthetic()
    assert data.equals(again) and meta.equals(again_meta)
    assert len(data) == 384
    assert meta.groupby("group_id").split.nunique().eq(1).all()
    assert not data.equals(make_synthetic(seed=12)[0])


def test_loader_keeps_ids_out_of_features_and_sorts_times_numerically(tmp_path):
    data, _ = make_synthetic()
    original = load_cmp_tsv(write_fixture(tmp_path, data))
    reordered = data[[c for c in data if not c.startswith("glu_time")] + [f"glu_time{i}" for i in range(8, 0, -1)]]
    arrays = load_cmp_tsv(write_fixture(tmp_path, reordered))
    np.testing.assert_array_equal(original.blocks["glu"], arrays.blocks["glu"])
    data["group_id"] = "SIM_GROUP"
    with pytest.raises(ValueError, match="not covered"):
        load_cmp_tsv(write_fixture(tmp_path, data))


@pytest.mark.parametrize("column,value", [("growth", 0.5), ("growth", np.nan), ("glu_time1", np.inf), ("species_A", 0.4)])
def test_loader_rejects_invalid_values(tmp_path, column, value):
    frame, _ = make_synthetic()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        load_cmp_tsv(write_fixture(tmp_path, frame))


def test_loader_rejects_duplicate_and_missing_time_columns(tmp_path):
    frame, _ = make_synthetic()
    path = write_fixture(tmp_path, frame)
    text = path.read_text().replace("glu_time2", "glu_time1", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="Duplicate"):
        load_cmp_tsv(path)
    with pytest.raises(ValueError, match="contiguous"):
        load_cmp_tsv(write_fixture(tmp_path, frame.drop(columns="glu_time4")))


def test_combination_indicators_are_allowed_but_species_must_be_one_hot(tmp_path):
    frame, _ = make_synthetic()
    frame.loc[0, ["antibiotic_A", "antibiotic_B"]] = 1
    load_cmp_tsv(write_fixture(tmp_path, frame))
    frame.loc[0, ["species_A", "species_B", "species_C"]] = 1
    with pytest.raises(ValueError, match="categorical"):
        load_cmp_tsv(write_fixture(tmp_path, frame))


def test_known_finite_differences_and_centering():
    x = np.array([[2, 6, 14, 26]], dtype=float)
    np.testing.assert_array_equal(transform_channel(x, "centered"), [[0, 4, 12, 24]])
    np.testing.assert_array_equal(transform_channel(x, "first_derivative", 2), [[2, 4, 6]])
    np.testing.assert_array_equal(transform_channel(x, "second_derivative", 2), [[1, 1]])
    for representation in ("centered", "first_derivative", "second_derivative"):
        np.testing.assert_array_equal(transform_channel(x, representation), transform_channel(x + 100, representation))
    with pytest.raises(ValueError, match="positive"):
        transform_channel(x, "first_derivative", 0)
