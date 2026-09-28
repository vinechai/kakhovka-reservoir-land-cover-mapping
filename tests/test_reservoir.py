# region labelling on a synthetic footprint: one big "reservoir", one separate lake, one pond.

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from src.data.reservoir import OTHER_WATER, OUTSIDE, RESERVOIR_BED, build_regions

X0, Y0 = 600_000, 5_260_000  # utm 36N, roughly the test area


@pytest.fixture
def mask_tif(tmp_path):
    mask = np.zeros((100, 100), dtype="uint8")
    mask[5:60, 5:95] = 1     # big: the reservoir
    mask[70:90, 60:85] = 1   # separate lake
    mask[80:83, 10:13] = 1   # small pond
    tif = tmp_path / "mask.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=100, height=100, count=1,
                       dtype="uint8", crs="EPSG:32636",
                       transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(mask, 1)
    return tif


def lonlat(row, col):
    xs, ys = transform("EPSG:32636", "EPSG:4326", [X0 + col * 10 + 5], [Y0 - row * 10 - 5])
    return xs[0], ys[0]


def test_regions_get_the_right_codes(mask_tif):
    regions, polys, _ = build_regions(mask_tif, {"lake": lonlat(80, 70)})
    assert regions[30, 50] == RESERVOIR_BED
    assert regions[80, 70] == 10          # first separate water body
    assert regions[81, 11] == OTHER_WATER
    assert regions[65, 50] == OUTSIDE
    assert set(polys.name) == {"reservoir_bed", "lake", "other_water"}


def test_point_outside_any_water_fails(mask_tif):
    with pytest.raises(ValueError, match="not inside"):
        build_regions(mask_tif, {"lake": lonlat(65, 50)})


def test_point_in_the_main_reservoir_fails(mask_tif):
    with pytest.raises(ValueError, match="main reservoir"):
        build_regions(mask_tif, {"lake": lonlat(30, 50)})
