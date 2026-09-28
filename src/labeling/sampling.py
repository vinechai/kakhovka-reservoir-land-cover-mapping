# picks random points on the reservoir bed for hand labelling. stratified by the reference
# month's map (water / bare / mid ndvi / high ndvi) so rare cover gets enough points, and
# kept a minimum distance apart.

from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from src.baseline.index_classifier import WATER

STRATA = {1: "water", 2: "bare", 3: "mid", 4: "high"}
NDVI_BARE, NDVI_HIGH = 0.2, 0.5

# share of points per stratum. training leans harder on "mid": that's where the formulas
# can't give free labels, so hand labels matter most there.
DESIGNS = {
    "test": {"water": 0.15, "bare": 0.20, "mid": 0.35, "high": 0.30},
    "train": {"water": 0.10, "bare": 0.15, "mid": 0.50, "high": 0.25},
}


def make_strata(classes: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """stratum code per pixel (see STRATA), 0 where there's no data."""
    s = np.zeros(classes.shape, dtype="uint8")
    s[ndvi < NDVI_BARE] = 2
    s[(ndvi >= NDVI_BARE) & (ndvi < NDVI_HIGH)] = 3
    s[ndvi >= NDVI_HIGH] = 4
    s[classes == WATER] = 1
    s[classes == 0] = 0
    return s


def allocate(n_total: int, shares: dict[str, float]) -> dict[str, int]:
    counts = {k: int(round(n_total * v)) for k, v in shares.items()}
    biggest = max(shares, key=shares.get)
    counts[biggest] += n_total - sum(counts.values())  # rounding leftovers
    return counts


def sample_points(strata: np.ndarray, region_mask: np.ndarray, transform, crs: str,
                  per_stratum: dict[str, int], min_dist_m: float, seed: int,
                  exclude_xy: np.ndarray | None = None,
                  names: dict[int, str] = STRATA) -> pd.DataFrame:
    """random points inside region_mask, `per_stratum[name]` from each stratum, no two closer
    than min_dist_m (also not closer than that to any exclude_xy point)."""
    rng = np.random.default_rng(seed)
    codes = {v: k for k, v in names.items()}
    taken = [] if exclude_xy is None else [tuple(p) for p in exclude_xy]
    rows = []
    for name, n in per_stratum.items():
        rr, cc = np.nonzero(region_mask & (strata == codes[name]))
        got = 0
        for i in rng.permutation(len(rr)):
            x, y = xy(transform, rr[i], cc[i])  # pixel centre
            if taken and np.min(np.hypot(*(np.array(taken) - (x, y)).T)) < min_dist_m:
                continue
            taken.append((x, y))
            rows.append({"row": int(rr[i]), "col": int(cc[i]), "x": x, "y": y, "stratum": name})
            got += 1
            if got == n:
                break
        if got < n:
            print(f"  warning: only found {got} of {n} points for stratum '{name}'")
    df = pd.DataFrame(rows)
    if len(df):
        lon, lat = warp_transform(crs, "EPSG:4326", df.x.tolist(), df.y.tolist())
        df["lon"], df["lat"] = np.round(lon, 6), np.round(lat, 6)
        # shuffle so the labeller doesn't see all water points first, then all bare, ...
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def stratum_sizes(strata: np.ndarray, region_mask: np.ndarray,
                  names: dict[int, str] = STRATA) -> dict[str, int]:
    """pixels per stratum inside the region, needed to weight accuracy later."""
    return {n: int((region_mask & (strata == c)).sum()) for c, n in names.items()}
