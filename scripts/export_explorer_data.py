# copies what the explorer needs into explorer_data/, which is tracked in git: class maps of
# both methods, and a true colour image per month (20 m, lat/lon, jpeg) as the map background.
#
# usage: python scripts/export_explorer_data.py

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.baseline.index_classifier import load_bands

from src.config import AREAS, DATA_DIR, ROOT

OUT = ROOT / "explorer_data"


def recompress(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        profile = src.profile
        profile.update(compress="deflate", zlevel=9, tiled=True)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(src.read())


def export_rgb(s2_tif: Path, out_jpg: Path, factor: int = 2) -> list[list[float]]:
    """true colour of one monthly composite, reprojected to lat/lon at `factor` x coarser
    resolution. returns the image bounds [[south, west], [north, east]]."""
    bands, valid, profile = load_bands(s2_tif)
    rgb = np.stack([bands["B4"], bands["B3"], bands["B2"]])
    rgb = (np.clip(rgb / 0.25, 0, 1) ** (1 / 1.4) * 255).astype("uint8")
    rgb[:, ~valid] = 255
    with rasterio.open(s2_tif) as src:
        tr, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height,
                                               *src.bounds)
        crs, transform = src.crs, src.transform
    tr = tr * tr.scale(factor)
    w, h = w // factor, h // factor
    out = np.zeros((3, h, w), dtype="uint8")
    for i in range(3):
        reproject(rgb[i], out[i], src_transform=transform, src_crs=crs, dst_transform=tr,
                  dst_crs="EPSG:4326", resampling=Resampling.average)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.transpose(1, 2, 0)).save(out_jpg, quality=85)
    return [[tr.f + h * tr.e, tr.c], [tr.f, tr.c + w * tr.a]]


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
    bounds = None
    tifs = sorted((src / "s2").glob("s2_*.tif"))
    for t in tifs:
        bounds = export_rgb(t, dst / "rgb" / f"rgb_{t.stem.removeprefix('s2_')}.jpg")
    (dst / "rgb" / "bounds.json").write_text(json.dumps(bounds))
    print(f"rgb: {len(tifs)} months")
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"-> {dst} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
