# phase 1 rules on hand-made pixels with known values. phase 2 labels come from these rules,
# so a silent change here would quietly change the training data.

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.baseline.index_classifier import (
    BARE, DENSE_VEG, NODATA, SNOW_ICE, SPARSE_VEG, WATER, classify, load_bands, mndwi, ndvi,
)


def pixel(blue=0.05, green=0.06, red=0.05, nir=0.05, swir1=0.05, swir2=0.04):
    """one-pixel 'image' with the given reflectances (0-1)."""
    vals = {"B2": blue, "B3": green, "B4": red, "B8": nir, "B11": swir1, "B12": swir2}
    return {b: np.array([[v]], dtype="float32") for b, v in vals.items()}


VALID = np.array([[True]])


def test_index_formulas():
    assert ndvi(pixel(red=0.1, nir=0.5))[0, 0] == pytest.approx(0.4 / 0.6)
    assert mndwi(pixel(green=0.08, swir1=0.02))[0, 0] == pytest.approx(0.06 / 0.10)


def test_clear_water():
    px = pixel(green=0.06, red=0.03, nir=0.02, swir1=0.01)
    assert classify(px, VALID, "2024-07")[0, 0] == WATER


def test_dense_vegetation():
    px = pixel(green=0.07, red=0.03, nir=0.40, swir1=0.20)
    assert classify(px, VALID, "2024-07")[0, 0] == DENSE_VEG


def test_bare_sand():
    px = pixel(green=0.22, red=0.25, nir=0.30, swir1=0.38)
    assert classify(px, VALID, "2024-07")[0, 0] == BARE


def test_algae_water_is_still_water():
    # green-ish water: ndvi ~0.38 would say "vegetation", but mndwi > 0 wins.
    # this is the whole reason for using mndwi over ndwi.
    px = pixel(green=0.08, red=0.04, nir=0.09, swir1=0.01)
    assert ndvi(px)[0, 0] > 0.3
    assert classify(px, VALID, "2025-08")[0, 0] == WATER


@pytest.mark.parametrize("nir, expected", [
    (0.180, BARE),        # ndvi 0.29
    (0.192, SPARSE_VEG),  # ndvi 0.31
    (0.290, SPARSE_VEG),  # ndvi 0.49
    (0.310, DENSE_VEG),   # ndvi 0.51
])
def test_vegetation_thresholds(nir, expected):
    # not testing "exactly 0.3": in float32 that comes out as 0.29999998 and lands below the
    # cut, which is meaningless at the precision the data has anyway
    px = pixel(red=0.1, nir=nir, swir1=0.3)
    assert classify(px, VALID, "2024-07")[0, 0] == expected


@pytest.mark.parametrize("month, expected", [
    ("2026-01", SNOW_ICE),  # winter: bright + "wet" = snow
    ("2025-12", SNOW_ICE),
    ("2025-08", BARE),      # summer: same signature = pale wet shoreline
    ("2024-04", BARE),
])
def test_snow_only_in_winter(month, expected):
    px = pixel(green=0.60, red=0.58, nir=0.55, swir1=0.08)
    assert classify(px, VALID, month)[0, 0] == expected


def test_no_clear_view_is_nodata():
    px = pixel(green=0.06, nir=0.02, swir1=0.01)
    assert classify(px, np.array([[False]]), "2024-07")[0, 0] == NODATA


def test_load_bands_reads_by_name_not_position(tmp_path):
    # file with bands in an unusual order: values must land under the right names
    order = ["B11", "B3", "clear_count", "B4", "B8", "B2", "B12"]
    data = np.stack([np.full((2, 2), (i + 1) * 100, dtype="uint16") for i in range(len(order))])
    tif = tmp_path / "s2.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=2, height=2, count=len(order),
                       dtype="uint16", crs="EPSG:32636",
                       transform=from_origin(0, 20, 10, 10)) as dst:
        dst.write(data)
        dst.descriptions = tuple(order)

    bands, valid, _ = load_bands(tif)
    assert bands["B11"][0, 0] == pytest.approx(0.01)  # 100 / 1e4
    assert bands["B3"][0, 0] == pytest.approx(0.02)
    assert bands["B12"][0, 0] == pytest.approx(0.07)
    assert "clear_count" not in bands
    assert valid.all()
