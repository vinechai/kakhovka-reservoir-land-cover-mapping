# copies the class maps the explorer needs into explorer_data/, which is tracked in git.
#
# usage: python scripts/export_explorer_data.py

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio

from src.config import AREAS, DATA_DIR, ROOT

OUT = ROOT / "explorer_data"


def recompress(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        profile = src.profile
        profile.update(compress="deflate", zlevel=9, tiled=True)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(src.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    args = ap.parse_args()

    src, dst = DATA_DIR / args.area, OUT / args.area
    recompress(src / "regions.tif", dst / "regions.tif")
    shutil.copy(src / "regions.geojson", dst / "regions.geojson")
    for method in ("baseline", "cnn"):
        files = sorted((src / method).glob("class_*.tif"))
        for f in files:
            recompress(f, dst / method / f.name)
        print(f"{method}: {len(files)} months")
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"-> {dst} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
