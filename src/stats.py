# km2 of each class per region per month, from a folder of class rasters. shared by phase 1
# (run_baseline.py) and the cnn (predict_cnn.py), so both go through exactly the same counting.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.baseline.index_classifier import CLASS_NAMES, NODATA
from src.config import COLLAPSE_DATE, SCALE

PX_KM2 = SCALE * SCALE / 1e6
MAX_HIDDEN_FRACTION = 0.2  # above this share of snow/ice + no clear view, a month isn't usable


def region_names(area_path: Path) -> dict[int, str]:
    gj = json.loads((area_path / "regions.geojson").read_text())
    return {f["properties"]["code"]: f["properties"]["name"] for f in gj["features"]}


def class_areas(classes: np.ndarray, regions: np.ndarray, names: dict[int, str],
                month: str) -> list[dict]:
    rows = []
    for code, name in names.items():
        inside = regions == code
        if not inside.any():
            continue
        row = {"month": month, "region": name, "area_km2": inside.sum() * PX_KM2,
               "nodata_km2": (inside & (classes == NODATA)).sum() * PX_KM2}
        for c, cname in CLASS_NAMES.items():
            row[f"{cname}_km2"] = (inside & (classes == c)).sum() * PX_KM2
        rows.append(row)
    return rows


def finish_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["post_collapse"] = df.month >= COLLAPSE_DATE[:7]
    # a month whose region is mostly snow or unseen says nothing about land cover
    hidden = (df.snow_ice_km2 + df.nodata_km2) / df.area_km2
    df["usable"] = hidden < MAX_HIDDEN_FRACTION
    return df.round(3)


def tabulate(class_dir: Path, area_path: Path) -> pd.DataFrame:
    """table for every class_<month>.tif in class_dir."""
    with rasterio.open(area_path / "regions.tif") as src:
        regions = src.read(1)
    names = region_names(area_path)
    rows = []
    for tif in sorted(class_dir.glob("class_*.tif")):
        with rasterio.open(tif) as src:
            rows += class_areas(src.read(1), regions, names, tif.stem.removeprefix("class_"))
    return finish_table(rows)


def speckle_rate(classes: np.ndarray, mask: np.ndarray) -> float:
    """share of pixels (inside mask) whose class differs from the most common class among their
    8 neighbours. high = salt-and-pepper map, low = coherent patches. descriptive only: real
    landscapes do have small patches, so lower isn't automatically more correct."""
    h, w = classes.shape
    padded = np.pad(classes, 1, mode="edge")
    neigh = np.stack([padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])
    counts = np.stack([(neigh == c).sum(0) for c in range(int(classes.max()) + 1)])
    majority = counts.argmax(0)
    return float((classes != majority)[mask].mean())
