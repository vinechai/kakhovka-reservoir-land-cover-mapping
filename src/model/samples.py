# training examples for the cnn: weak labels (pixels where the index rules are very sure) and
# hand labels (the only source for sparse vegetation). no weak label within GUARD_M of a test
# or held-out point, so the model is never trained on a spot it's graded on.

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from src.baseline.index_classifier import WINTER_MONTHS
from src.config import ALL_MONTHS, area_dir
from src.labeling.store import list_point_sets, load_labels, load_points
from src.model.features import month_layers

# classes the cnn predicts (index = network output). codes match the phase 1 rasters.
CLASSES = ["water", "bare", "sparse_veg", "dense_veg", "snow_ice"]
CLASS_CODES = {"water": 1, "bare": 2, "sparse_veg": 3, "dense_veg": 5, "snow_ice": 4}

GUARD_M = 300
HAND_SETS = ("pilot", "train")


def weak_rules(spec: np.ndarray, month: str, weak_sparse: bool = False) -> dict[str, np.ndarray]:
    """confident masks per class from one month's spectral layers. weak_sparse also takes
    phase 1's sparse band (ndvi 0.32-0.48) as noisy sparse labels."""
    b3, b8, nd, mn = spec[1], spec[3], spec[10], spec[11]
    winter = int(month[5:7]) in WINTER_MONTHS
    bright = (b8 > 0.2) & (b3 > 0.2)
    out = {
        "water": (mn > 0.25) & (b8 < 0.11),
        "bare": (nd < 0.15) & (mn < -0.15),
        "dense_veg": (nd > 0.65) & (mn < 0),
    }
    if winter:
        out["snow_ice"] = (mn > 0.2) & bright
    if weak_sparse:
        out["sparse_veg"] = (nd >= 0.32) & (nd <= 0.48) & (mn < -0.1)
    return out


def hand_labels(area: str, seed: int = 0, val_frac: float = 0.25) -> pd.DataFrame:
    """pilot + train hand labels with a stratified train / val split (column `split`)."""
    rows = []
    for s in HAND_SETS:
        if s not in list_point_sets():
            continue
        d = load_points(s).merge(load_labels(s)[["point_id", "label"]], on="point_id")
        d["set"] = s
        rows.append(d[d.area == area])
    d = pd.concat(rows, ignore_index=True)
    d = d[d.label.isin(CLASSES)].copy()
    rng = np.random.default_rng(seed)
    d["split"] = "train"
    for _, grp in d.groupby("label"):
        n_val = int(round(len(grp) * val_frac))
        d.loc[rng.choice(grp.index, n_val, replace=False), "split"] = "val"
    return d.rename(columns={"target_month": "month"})


def _far_from(rows, cols, transform, pts_xy: np.ndarray, dist: float) -> np.ndarray:
    if len(pts_xy) == 0:
        return np.ones(len(rows), bool)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    keep = np.ones(len(rows), bool)
    for px, py in pts_xy:  # few hundred points, fine to loop
        keep &= (xs - px) ** 2 + (ys - py) ** 2 >= dist ** 2
    return keep


def weak_labels(area: str, per_class: int, guard_xy: np.ndarray, seed: int = 0,
                months: list[str] = ALL_MONTHS, weak_sparse: bool = False) -> pd.DataFrame:
    """up to per_class confident pixels per class per month, inside the jrc footprint,
    none within GUARD_M of guard_xy."""
    d = area_dir(area)
    with rasterio.open(d / "regions.tif") as src:
        footprint = src.read(1) > 0
        transform = src.transform
    rng = np.random.default_rng(seed)
    out = []
    for month in months:
        spec, valid = month_layers(str(d / "s2"), month)
        for label, mask in weak_rules(spec, month, weak_sparse).items():
            rr, cc = np.nonzero(mask & valid & footprint)
            if len(rr) == 0:
                continue
            pick = rng.choice(len(rr), min(len(rr), per_class * 3), replace=False)
            rr, cc = rr[pick], cc[pick]
            keep = _far_from(rr, cc, transform, guard_xy, GUARD_M)
            rr, cc = rr[keep][:per_class], cc[keep][:per_class]
            out.append(pd.DataFrame({"month": month, "row": rr, "col": cc, "label": label}))
    return pd.concat(out, ignore_index=True)


def extract_patches(samples: pd.DataFrame, stack_fn, half: int) -> np.ndarray:
    """[N, C, 2*half+1, 2*half+1] float16 patches around each sample, month by month.
    stack_fn(month) -> (features [C, H, W], valid). edges are zero-padded."""
    size = 2 * half + 1
    first = stack_fn(samples.month.iloc[0])[0]
    out = np.zeros((len(samples), first.shape[0], size, size), dtype="float16")
    for month, grp in samples.groupby("month"):
        feats, _ = stack_fn(month)
        padded = np.pad(feats, ((0, 0), (half, half), (half, half)))
        for i, r, c in zip(grp.index, grp.row, grp.col):
            out[samples.index.get_loc(i)] = padded[:, r:r + size, c:c + size]
    return out
