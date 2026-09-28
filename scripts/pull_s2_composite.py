# pulls monthly sentinel-2 composites, months already on disk with all bands are skipped.
#
# usage: python scripts/pull_s2_composite.py --all

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from src.config import ALL_MONTHS, AREAS, GEE_PROJECT, area_dir
from src.data import gee
from src.data.sentinel2 import BANDS, monthly_composite, scene_counts


def has_all_bands(tif: Path) -> bool:
    """true if the file exists and already holds every band in BANDS. files from before a
    band was added get re-pulled."""
    if not tif.exists():
        return False
    with rasterio.open(tif) as src:
        return set(BANDS) <= set(src.descriptions)


def pull_month(area, month: str, cloud_mask: str, out_dir: Path) -> dict:
    aoi = gee.area_geometry(area.bbox)
    counts = scene_counts(aoi, month)
    print(f"{area.name} {month}: {counts['scenes']} scenes, "
          f"{counts['with_s2cloudless']} with s2cloudless")
    info = {"month": month, **counts}
    if counts["scenes"] == 0:
        print("  no usable scenes this month, nothing to pull")
        return info

    method = cloud_mask
    if method == "auto":
        method = "s2cloudless" if counts["with_s2cloudless"] == counts["scenes"] else "csplus"
    print(f"  cloud mask: {method}")

    tif = out_dir / f"s2_{month}.tif"
    image = monthly_composite(aoi, month, method)
    gee.download_image(image, BANDS + ["clear_count"], area.bbox, tif, dtype="uint16", nodata=0)

    with rasterio.open(tif) as src:
        clear = src.read(len(BANDS) + 1)
    info.update(cloud_mask=method, clear_median=float(np.median(clear)),
                never_clear_pct=float((clear == 0).mean() * 100))
    print(f"  clear looks per pixel: median {info['clear_median']:.0f}, "
          f"{info['never_clear_pct']:.1f}% of pixels never clear")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--months", nargs="+", help="YYYY-MM ...")
    group.add_argument("--all", action="store_true", help="every month in config.ALL_MONTHS")
    parser.add_argument("--project", default=GEE_PROJECT)
    parser.add_argument("--cloud-mask", default="auto", choices=["auto", "s2cloudless", "csplus"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    gee.init(args.project)
    area = AREAS[args.area]
    out_dir = area_dir(area.name) / "s2"
    out_dir.mkdir(exist_ok=True)

    # per-month pull info (scene counts, which cloud mask) is kept next to the rasters
    log_path = out_dir / "pull_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    months = ALL_MONTHS if args.all else args.months
    failed = []
    for month in months:
        if has_all_bands(out_dir / f"s2_{month}.tif") and not args.force:
            continue
        t0 = time.time()
        try:
            log[month] = pull_month(area, month, args.cloud_mask, out_dir)
        except Exception as e:  # keep the batch going, rerun picks it up
            print(f"  {month} FAILED: {e}")
            failed.append(month)
            continue
        log_path.write_text(json.dumps(log, indent=1, sort_keys=True))
        print(f"  took {time.time() - t0:.0f}s")

    print(f"done. failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
