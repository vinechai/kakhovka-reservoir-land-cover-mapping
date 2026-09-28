# phase 1 for every month: a class map per month and km2 per class, region and month.
#
# usage: python scripts/run_baseline.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio

from src.baseline.index_classifier import NODATA, classify, load_bands
from src.config import AREAS, ROOT, area_dir
from src.stats import class_areas, finish_table, region_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    args = parser.parse_args()

    d = area_dir(args.area)
    with rasterio.open(d / "regions.tif") as src:
        regions = src.read(1)
    names = region_names(d)

    out_dir = d / "baseline"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for tif in sorted((d / "s2").glob("s2_*.tif")):
        month = tif.stem.removeprefix("s2_")
        bands, valid, profile = load_bands(tif)
        classes = classify(bands, valid, month)

        profile.update(count=1, dtype="uint8", nodata=NODATA)
        with rasterio.open(out_dir / f"class_{month}.tif", "w", **profile) as dst:
            dst.write(classes, 1)

        month_rows = class_areas(classes, regions, names, month)
        rows += month_rows
        bed = next(r for r in month_rows if r["region"] == "reservoir_bed")
        print(f"{month}: bed water {bed['water_km2']:6.1f}  bare {bed['bare_km2']:6.1f}  "
              f"sparse {bed['sparse_veg_km2']:6.1f}  dense {bed['dense_veg_km2']:6.1f}  "
              f"snow/ice {bed['snow_ice_km2']:5.1f}  nodata {bed['nodata_km2']:5.1f} km2")

    out = ROOT / "outputs" / "tables" / f"baseline_{args.area}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    finish_table(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
