# phase 1: land cover from index thresholds, no learning.
# water = mndwi > 0, dense = ndvi >= 0.5, sparse = ndvi 0.3-0.5, bare = the rest.
# bright and wet pixels are snow/ice in nov-mar, bare otherwise (pale wet shorelines).
# mndwi instead of ndwi because algae-rich water reflects nir and fools ndwi.

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from src.data.sentinel2 import BANDS

WATER_MNDWI = 0.0
VEG_NDVI = 0.3
DENSE_NDVI = 0.5
SNOW_NIR = 0.11
SNOW_GREEN = 0.10
WINTER_MONTHS = {11, 12, 1, 2, 3}  # months where snow/ice is possible on the lower dnipro

# codes as stored in the class rasters (4 stays snow/ice so older outputs keep their meaning)
NODATA, WATER, BARE, SPARSE_VEG, SNOW_ICE, DENSE_VEG = 0, 1, 2, 3, 4, 5
CLASS_NAMES = {WATER: "water", BARE: "bare", SPARSE_VEG: "sparse_veg", DENSE_VEG: "dense_veg",
               SNOW_ICE: "snow_ice"}


def load_bands(tif: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """reflectance (0-1) per band, a valid mask (at least one clear look), and the profile."""
    with rasterio.open(tif) as src:
        arr = src.read().astype("float32")
        profile = src.profile
        names = src.descriptions
    # bands are looked up by name, not position, so files pulled with a different band list
    # (e.g. before the red-edge bands were added) still read correctly
    bands = {b: arr[i] / 1e4 for i, b in enumerate(names) if b in BANDS}
    valid = arr[names.index("clear_count")] > 0
    return bands, valid, profile


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, 1e-4)


def mndwi(bands) -> np.ndarray:
    return _nd(bands["B3"], bands["B11"])


def ndvi(bands) -> np.ndarray:
    return _nd(bands["B8"], bands["B4"])


def classify(bands: dict[str, np.ndarray], valid: np.ndarray, month: str) -> np.ndarray:
    w, v = mndwi(bands), ndvi(bands)
    wet = w > WATER_MNDWI
    bright_wet = wet & (bands["B8"] > SNOW_NIR) & (bands["B3"] > SNOW_GREEN)

    out = np.full(valid.shape, BARE, dtype="uint8")
    out[v >= VEG_NDVI] = SPARSE_VEG
    out[v >= DENSE_NDVI] = DENSE_VEG
    out[wet] = WATER
    # bright + "wet" is snow/ice in winter. in summer the same signature shows up on pale wet
    # shorelines (the drying rim of bilozerskyi lyman, wet sand bars), so there it's sediment.
    out[bright_wet] = SNOW_ICE if int(month[5:7]) in WINTER_MONTHS else BARE
    out[~valid] = NODATA
    return out
