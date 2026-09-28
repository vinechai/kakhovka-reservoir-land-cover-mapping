# point sampling + label storage

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.baseline.index_classifier import BARE, DENSE_VEG, WATER
from src.labeling.sampling import DESIGNS, allocate, make_strata, sample_points, stratum_sizes
from src.labeling.store import load_labels, save_label

TRANSFORM = from_origin(600_000, 5_260_000, 10, 10)


@pytest.fixture
def class_map():
    # columns: water | bare (ndvi 0.1) | mid (ndvi 0.35) | high (ndvi 0.8)
    classes = np.full((200, 200), DENSE_VEG, dtype="uint8")
    ndvi = np.full((200, 200), 0.8, dtype="float32")
    classes[:, :40] = WATER
    ndvi[:, :40] = -0.2
    classes[:, 40:80] = BARE
    ndvi[:, 40:80] = 0.1
    classes[:, 80:120] = BARE   # phase 1 says bare (ndvi < 0.3), stratum says mid
    ndvi[:, 80:120] = 0.35
    region = np.zeros_like(classes, dtype=bool)
    region[20:180, 20:180] = True
    return make_strata(classes, ndvi), region


def test_strata_follow_ndvi_and_water(class_map):
    strata, region = class_map
    assert stratum_sizes(strata, region) == {"water": 160 * 20, "bare": 160 * 40,
                                             "mid": 160 * 40, "high": 160 * 60}


@pytest.mark.parametrize("design", list(DESIGNS))
def test_allocate_adds_up(design):
    for n in (30, 150, 300, 7):
        assert sum(allocate(n, DESIGNS[design]).values()) == n


def test_sampling_counts_region_and_spacing(class_map):
    strata, region = class_map
    want = {"water": 5, "bare": 6, "mid": 7, "high": 12}
    pts = sample_points(strata, region, TRANSFORM, "EPSG:32636", want, min_dist_m=100, seed=1)

    assert pts.stratum.value_counts().to_dict() == {"water": 5, "bare": 6, "mid": 7, "high": 12}
    assert region[pts.row, pts.col].all()                       # all inside the region
    codes = {"water": 1, "bare": 2, "mid": 3, "high": 4}
    assert (strata[pts.row, pts.col] == pts.stratum.map(codes)).all()  # right stratum
    xy = pts[["x", "y"]].to_numpy()
    dists = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    assert dists[np.triu_indices(len(xy), 1)].min() >= 100     # spaced out


def test_sampling_keeps_away_from_other_sets(class_map):
    strata, region = class_map
    first = sample_points(strata, region, TRANSFORM, "EPSG:32636", {"high": 10}, 100, seed=1)
    second = sample_points(strata, region, TRANSFORM, "EPSG:32636", {"high": 10}, 100,
                           seed=2, exclude_xy=first[["x", "y"]].to_numpy())
    a, b = first[["x", "y"]].to_numpy(), second[["x", "y"]].to_numpy()
    assert np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)).min() >= 100


def test_sampling_is_reproducible(class_map):
    strata, region = class_map
    args = (strata, region, TRANSFORM, "EPSG:32636", {"high": 8}, 100, 42)
    assert sample_points(*args).equals(sample_points(*args))


def test_relabelling_replaces_the_row(tmp_path):
    save_label("t", 3, "dense_veg", True, "", "2024-08", "2024-08-23", "me", root=tmp_path)
    save_label("t", 5, "water", True, "", "2024-08", "2024-08-23", "me", root=tmp_path)
    save_label("t", 3, "sparse_veg", False, "changed my mind", "2024-08", "2024-08-23", "me",
               root=tmp_path)
    df = load_labels("t", root=tmp_path)
    assert len(df) == 2
    assert df.set_index("point_id").loc[3, "label"] == "sparse_veg"


def test_unknown_label_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_label("t", 1, "forest", True, "", "2024-08", "", "me", root=tmp_path)
