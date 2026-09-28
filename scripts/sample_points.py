# samples points for hand labelling. each point is labelled for the month esri's high-res
# photo of that spot was taken, points with winter photos are dropped.
#
# usage: python scripts/sample_points.py --set test --n 300 --design test

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio

from src.config import ALL_MONTHS, AREAS, area_dir
from src.data.reservoir import RESERVOIR_BED
from src.labeling.imagery import esri_capture_date
from src.baseline.index_classifier import WINTER_MONTHS, load_bands, ndvi
from src.labeling.sampling import DESIGNS, allocate, make_strata, sample_points, stratum_sizes
from src.labeling.store import LABELS_DIR, list_point_sets, load_points, points_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    parser.add_argument("--set", required=True, help="name, e.g. pilot / test")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--design", default="test", choices=list(DESIGNS))
    parser.add_argument("--ref-month", default="2024-08",
                        help="month whose phase 1 map + ndvi define the strata; also the "
                             "fallback target month")
    parser.add_argument("--min-dist", type=float, default=300, help="metres between points")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = points_path(args.set)
    if out.exists():
        print(f"{out} already exists, pick another --set name (or delete it on purpose)")
        return

    d = area_dir(args.area)
    with rasterio.open(d / "regions.tif") as src:
        bed = src.read(1) == RESERVOIR_BED
    with rasterio.open(d / "baseline" / f"class_{args.ref_month}.tif") as src:
        classes, transform, crs = src.read(1), src.transform, src.crs.to_string()
    bands, _, _ = load_bands(d / "s2" / f"s2_{args.ref_month}.tif")
    strata = make_strata(classes, ndvi(bands))
    sizes = stratum_sizes(strata, bed)
    print(f"stratum sizes on the bed (km2): { {k: round(v / 1e4, 1) for k, v in sizes.items()} }")

    others = [load_points(s)[["x", "y"]].to_numpy() for s in list_point_sets()]
    exclude = np.vstack(others) if others else None

    counts = allocate(args.n, DESIGNS[args.design])
    # draw ~15% extra: points whose high-res photo is from winter get dropped below
    extra = {k: int(np.ceil(v * 1.15)) for k, v in counts.items()}
    pts = sample_points(strata, bed, transform, crs, extra, args.min_dist, args.seed, exclude)

    print("looking up high-res imagery dates...")
    # one web lookup per point, run 8 at a time (one by one takes ~10 min for 300 points)
    with ThreadPoolExecutor(8) as pool:
        pts["imagery_date"] = list(pool.map(esri_capture_date, pts.lon, pts.lat))
    months = pts.imagery_date.str[:7]
    pts["target_month"] = months.where(months.isin(ALL_MONTHS), args.ref_month)

    # winter photos (nov-mar): plants leafless or under snow, cover can't be judged, and the
    # matching sentinel-2 composites are often mostly snow. drop those points, then keep the
    # wanted number per stratum (pts is already shuffled, so "first n" is still random).
    winter = pts.target_month.str[5:7].astype(int).isin(WINTER_MONTHS)
    print(f"dropping {winter.sum()} points with winter high-res imagery")
    pts = pts[~winter]
    pts = pd.concat([pts[pts.stratum == k].head(n) for k, n in counts.items()])
    pts = pts.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    short = {k: n - (pts.stratum == k).sum() for k, n in counts.items() if (pts.stratum == k).sum() < n}
    if short:
        print(f"  warning: strata short of points: {short}")
    print(f"kept {len(pts)} points: {pts.stratum.value_counts().to_dict()}")
    print(f"target months: {pts.target_month.value_counts().to_dict()}")

    pts.insert(0, "point_id", range(len(pts)))
    pts["area"] = args.area
    LABELS_DIR.mkdir(exist_ok=True)
    pts.to_csv(out, index=False)
    (LABELS_DIR / f"strata_{args.set}.json").write_text(json.dumps(
        {"ref_month": args.ref_month, "design": args.design, "pixels": sizes}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
