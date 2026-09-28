# phase 1 vs cnn figures (time series, maps with a zoomed crop) and the speckle table.
#
# usage: python scripts/compare_methods.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch, Rectangle

from scripts.plot_landcover import (
    CLASS_CMAP, COLORS, GRID, INK, INK_2, LABELS, NODATA_COLOR, SURFACE, VMAX, _rgb, _style,
)
from src.config import AREAS, COLLAPSE_DATE, ROOT, area_dir
from src.stats import speckle_rate

FIG_DIR = ROOT / "outputs" / "figures"
METHOD_STYLE = {"baseline": ("Phase 1 (index rules)", "--"), "cnn": ("CNN", "-")}


def load_tables(area: str) -> dict[str, pd.DataFrame]:
    out = {}
    for m in METHOD_STYLE:
        df = pd.read_csv(ROOT / "outputs" / "tables" / f"{m}_{area}.csv", dtype={"month": str})
        df = df[(df.region == "reservoir_bed") & df.usable].copy()
        df["date"] = pd.to_datetime(df.month) + pd.Timedelta(days=14)
        out[m] = df.sort_values("date")
    return out


def plot_timeseries(area: str):
    tables = load_tables(area)
    classes = ["water", "bare", "sparse_veg", "dense_veg"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, c in zip(axes.flat, classes):
        for m, (label, ls) in METHOD_STYLE.items():
            t = tables[m][tables[m].post_collapse | (tables[m].month >= "2023-01")]
            ax.plot(t.date, t[f"{c}_km2"], ls, color=COLORS[c], lw=2 if m == "cnn" else 1.4,
                    marker="o" if m == "cnn" else None, ms=3, label=label)
        ax.axvline(pd.Timestamp(COLLAPSE_DATE), color=INK_2, lw=0.8, ls=":")
        ax.set_title(LABELS[c], loc="left", fontsize=11, color=INK)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("km² of reservoir bed", color=INK_2)
    axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Phase 1 (dashed) vs CNN (solid): land cover on the former reservoir bed, "
                 "usable months from 2023", fontsize=12, color=INK)
    fig.tight_layout()
    out = FIG_DIR / f"compare_timeseries_{area}.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


def _classes(d: Path, method: str, month: str) -> np.ndarray:
    with rasterio.open(d / method / f"class_{month}.tif") as src:
        return src.read(1)


def plot_maps(area: str, months: list[str], crop: tuple[int, int, int]):
    d = area_dir(area)
    with rasterio.open(d / "regions.tif") as src:
        bed = src.read(1) == 1
    r0, c0, size = crop
    fig, axes = plt.subplots(len(months), 6, figsize=(21, 3.6 * len(months)), facecolor=SURFACE)
    titles = ["True colour", "Phase 1", "CNN", "True colour (zoom)", "Phase 1 (zoom)", "CNN (zoom)"]
    for i, month in enumerate(months):
        rgb = _rgb(d / "s2" / f"s2_{month}.tif")
        imgs = [rgb, _classes(d, "baseline", month), _classes(d, "cnn", month)]
        for j, img in enumerate(imgs + [im[r0:r0 + size, c0:c0 + size] for im in imgs]):
            ax = axes[i, j]
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=CLASS_CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
            if j < 3:
                ax.contour(bed, levels=[0.5], colors=INK, linewidths=0.4)
                ax.add_patch(Rectangle((c0, r0), size, size, fill=False, ec="#ffffff", lw=1.2))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=11, color=INK)
        axes[i, 0].set_ylabel(month, fontsize=12, color=INK)
    handles = [Patch(color=COLORS[c], label=LABELS[c]) for c in COLORS] + \
              [Patch(color=NODATA_COLOR, label="No clear view")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10)
    fig.suptitle("Phase 1 vs CNN class maps. Black line = reservoir bed; white square = zoomed "
                 f"area ({size * 10 / 1000:.1f} km across)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = FIG_DIR / f"compare_maps_{area}.png"
    fig.savefig(out, dpi=90, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


def speckle_table(area: str, months: list[str]):
    """speckle rate (see src/stats.py) per method and month on the reservoir bed."""
    d = area_dir(area)
    with rasterio.open(d / "regions.tif") as src:
        bed = src.read(1) == 1
    rows = {m: {} for m in METHOD_STYLE}
    for method in METHOD_STYLE:
        for month in months:
            rows[method][month] = round(speckle_rate(_classes(d, method, month), bed), 3)
    t = pd.DataFrame(rows).T
    out = ROOT / "outputs" / "tables" / f"speckle_{area}.md"
    out.write_text("Share of reservoir-bed pixels whose class differs from most of their 8 "
                   "neighbours (lower = more coherent map)\n\n" + t.to_markdown(), encoding="utf-8")
    print(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    ap.add_argument("--months", nargs="+", default=["2023-07", "2024-08", "2025-04", "2026-08"])
    ap.add_argument("--crop", type=int, nargs=3, default=[900, 700, 300],
                    help="row, col, size (pixels) of the zoomed crop")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeseries(args.area)
    plot_maps(args.area, args.months, tuple(args.crop))
    speckle_table(args.area, args.months)


if __name__ == "__main__":
    main()
