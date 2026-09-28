# first pipeline check: true colour and ndwi for two months with the reservoir outline.
#
# usage: python scripts/quicklook.py --months 2021-07 2026-08

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.config import AREAS, ROOT, area_dir
from src.baseline.index_classifier import load_bands


def load(tif: Path) -> tuple[dict, np.ndarray]:
    bands, valid, _ = load_bands(tif)
    return bands, valid


def rgb(bands: dict, valid: np.ndarray) -> np.ndarray:
    img = np.dstack([bands["B4"], bands["B3"], bands["B2"]])
    img = np.clip(img / 0.25, 0, 1) ** (1 / 1.4)  # simple stretch + gamma for display
    img[~valid] = 1
    return img


def ndwi(bands: dict) -> np.ndarray:
    # mcfeeters ndwi: green vs nir, water > 0
    g, n = bands["B3"], bands["B8"]
    return (g - n) / np.maximum(g + n, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    parser.add_argument("--months", nargs="+", required=True)
    args = parser.parse_args()

    d = area_dir(args.area)
    with rasterio.open(d / "reservoir_mask.tif") as src:
        footprint = src.read(1) == 1

    fig, axes = plt.subplots(2, len(args.months), figsize=(6.5 * len(args.months), 11),
                             squeeze=False)
    for col, month in enumerate(args.months):
        bands, valid = load(d / "s2" / f"s2_{month}.tif")
        w = ndwi(bands)
        inside = footprint & valid
        water_pct = (w[inside] > 0).mean() * 100 if inside.any() else float("nan")
        print(f"{month}: {valid.mean() * 100:.1f}% valid pixels, "
              f"ndwi>0 over {water_pct:.1f}% of the old reservoir footprint")

        axes[0, col].imshow(rgb(bands, valid))
        axes[0, col].set_title(f"{month}  true colour")
        im = axes[1, col].imshow(np.where(valid, w, np.nan), cmap="RdBu", vmin=-0.6, vmax=0.6)
        axes[1, col].set_title(f"{month}  NDWI  (water > 0: {water_pct:.0f}% of footprint)")
        for ax in axes[:, col]:
            ax.contour(footprint, levels=[0.5], colors="gold", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label="NDWI")
    fig.suptitle(f"{AREAS[args.area].description}\ngold = JRC pre-collapse reservoir footprint")

    out = ROOT / "outputs" / "figures" / f"quicklook_{args.area}_{'_vs_'.join(args.months)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
