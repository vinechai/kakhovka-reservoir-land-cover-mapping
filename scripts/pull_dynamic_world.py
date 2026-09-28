# pulls google dynamic world (most common class per pixel per month) as an outside reference.
#
# usage: python scripts/pull_dynamic_world.py --months 2024-08 2025-08 2026-08

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee

from src.config import AREAS, GEE_PROJECT, area_dir
from src.data import gee
from src.data.sentinel2 import month_range

DW = "GOOGLE/DYNAMICWORLD/V1"
# dynamic world class index -> name (0-8); 255 = no data
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub",
              "built", "bare", "snow_and_ice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    ap.add_argument("--months", nargs="+", default=["2024-08", "2025-08", "2026-08"])
    ap.add_argument("--project", default=GEE_PROJECT)
    args = ap.parse_args()

    gee.init(args.project)
    area = AREAS[args.area]
    aoi = gee.area_geometry(area.bbox)
    for month in args.months:
        start, end = month_range(month)
        col = ee.ImageCollection(DW).filterBounds(aoi).filterDate(start, end).select("label")
        print(f"{month}: {col.size().getInfo()} dynamic world scenes")
        img = col.mode().unmask(255).toUint8().rename("label")
        out = area_dir(area.name) / "dynamic_world" / f"dw_{month}.tif"
        gee.download_image(img, ["label"], area.bbox, out, dtype="uint8", nodata=255)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
