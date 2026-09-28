# pulls the jrc reservoir outline and splits it into regions (bed, lyman, small ponds).
#
# usage: python scripts/pull_reservoir_mask.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio

from src.config import AREAS, GEE_PROJECT, SCALE, SEPARATE_WATER_BODIES, area_dir
from src.data import gee
from src.data.reservoir import build_regions, footprint_image, mask_area_km2, occurrence_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    parser.add_argument("--project", default=GEE_PROJECT)
    parser.add_argument("--occurrence-min", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="re-download even if it exists")
    args = parser.parse_args()

    area = AREAS[args.area]
    out_dir = area_dir(area.name)
    tif = out_dir / "reservoir_mask.tif"

    if args.force or not tif.exists():
        gee.init(args.project)
        image = footprint_image(args.occurrence_min).addBands(occurrence_image())
        print(f"{area.name}: pulling jrc footprint (occurrence >= {args.occurrence_min}%)...")
        gee.download_image(image, ["reservoir", "occurrence"], area.bbox, tif,
                           dtype="uint8", nodata=255)

    with rasterio.open(tif) as src:
        mask = src.read(1)
    total_km2 = mask.size * SCALE * SCALE / 1e6
    print(f"reservoir footprint: {mask_area_km2(mask, SCALE):.1f} km2 of {total_km2:.1f} km2 area")

    regions, polys, profile = build_regions(tif, SEPARATE_WATER_BODIES.get(area.name, {}))
    profile.update(count=1, nodata=None)
    with rasterio.open(out_dir / "regions.tif", "w", **profile) as dst:
        dst.write(regions, 1)
    polys.to_crs("EPSG:4326").to_file(out_dir / "regions.geojson", driver="GeoJSON")

    for name, grp in polys.groupby("name", sort=False):
        print(f"  {name:20s} {grp.area_km2.sum():7.2f} km2  ({len(grp)} polygon(s))")
    print(f"-> {out_dir / 'regions.tif'}")


if __name__ == "__main__":
    main()
