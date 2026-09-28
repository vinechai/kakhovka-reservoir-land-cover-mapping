# land cover figures for one method: monthly chart, map strip and animation.
#
# usage: python scripts/plot_landcover.py [--method cnn]

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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.baseline.index_classifier import load_bands
from src.config import AREAS, COLLAPSE_DATE, ROOT, area_dir

# class colours: conventional land-cover meaning, validated as a set (all-pairs, cvd-safe)
COLORS = {"water": "#2a78d6", "bare": "#eda100", "sparse_veg": "#1baf7a", "dense_veg": "#008300",
          "snow_ice": "#4a3aa7"}
LABELS = {"water": "Water", "bare": "Bare sediment / sand", "sparse_veg": "Sparse vegetation",
          "dense_veg": "Dense vegetation", "snow_ice": "Snow / ice"}
NODATA_COLOR = "#d9d9d6"
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# class raster codes 0..5 -> colours (0 = nodata, see index_classifier)
CLASS_CMAP = ListedColormap([NODATA_COLOR, COLORS["water"], COLORS["bare"],
                             COLORS["sparse_veg"], COLORS["snow_ice"], COLORS["dense_veg"]])
VMAX = 5

FIG_DIR = ROOT / "outputs" / "figures"

# which class maps to draw: data/<area>/<METHOD>/ and outputs/tables/<METHOD>_<area>.csv
METHOD = "baseline"
METHOD_TITLE = {"baseline": "index baseline", "cnn": "CNN"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_timeseries(df: pd.DataFrame, area: str):
    bed = df[df.region == "reservoir_bed"].sort_values("month").copy()
    bed["date"] = pd.to_datetime(bed.month) + pd.Timedelta(days=14)
    classes = ["water", "bare", "sparse_veg", "dense_veg", "snow_ice"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    # months mostly under snow / unseen are drawn faded: they say nothing about land cover
    alpha = np.where(bed.usable, 1.0, 0.3)
    bottom = np.zeros(len(bed))
    for c in classes:
        bars = ax.bar(bed.date, bed[f"{c}_km2"], bottom=bottom, width=24, color=COLORS[c],
                      edgecolor=SURFACE, linewidth=1, label=LABELS[c])
        for b, a in zip(bars, alpha):
            b.set_alpha(a)
        bottom += bed[f"{c}_km2"].to_numpy()
    bars = ax.bar(bed.date, bed.nodata_km2, bottom=bottom, width=24, color=NODATA_COLOR,
                  edgecolor=SURFACE, linewidth=1, label="No clear view")
    for b, a in zip(bars, alpha):
        b.set_alpha(a)
    ax.scatter(bed.date[~bed.usable], np.full((~bed.usable).sum(), bed.area_km2.iloc[0] + 8),
               marker="x", s=18, color=INK_2, linewidths=1,
               label="Month mostly snow/cloud (faded)")

    # direct labels on the last month for the three main classes
    last = bed.iloc[-1]
    y = 0.0
    for c in ["water", "bare", "sparse_veg", "dense_veg"]:
        v = last[f"{c}_km2"]
        if v > 8:
            ax.annotate(f"{LABELS[c].split(' /')[0]} {v:.0f} km²", (last.date, y + v / 2),
                        xytext=(16, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        y += v

    collapse = pd.Timestamp(COLLAPSE_DATE)
    for a in (ax, ax2):
        a.axvline(collapse, color=INK_2, linewidth=1, linestyle="--")
        _style(a)
    ax.text(collapse, bed.area_km2.iloc[0] * 1.03, "  dam destroyed 6 Jun 2023",
            fontsize=9, color=INK_2, va="bottom")
    ax.set_ylabel("km² of reservoir bed", color=INK_2)
    # headroom above the bars holds the legend
    ax.set_ylim(0, bed.area_km2.iloc[0] * 1.28)
    ax.set_title("Former Kakhovka reservoir bed, Velykyi Luh test area: land cover by month "
                 f"({METHOD_TITLE[METHOD]})", loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9, handlelength=1)

    lym = df[df.region == "bilozerskyi_lyman"].sort_values("month").copy()
    if len(lym):
        lym["date"] = pd.to_datetime(lym.month) + pd.Timedelta(days=14)
        # frozen lake is still lake: count ice as water surface here (not on the bed, where
        # snow sits on land just as much as on water)
        surface = (lym.water_km2 + lym.snow_ice_km2).where(lym.nodata_km2 / lym.area_km2 < 0.2)
        ax2.plot(lym.date, surface, color=COLORS["water"], linewidth=2,
                 marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
        ax2.axhline(lym.area_km2.iloc[0], color=INK_2, linewidth=0.8, linestyle=":")
        ax2.text(lym.date.iloc[0], lym.area_km2.iloc[0], "full outline", fontsize=8,
                 color=INK_2, va="bottom")
        ax2.set_ylim(0, lym.area_km2.iloc[0] * 1.25)
        ax2.set_ylabel("water km²", color=INK_2)
        ax2.set_title("Bilozerskyi Lyman (separately dammed lake, not in the stats above): "
                      "water surface, open or frozen", loc="left", fontsize=10, color=INK)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = FIG_DIR / f"{METHOD}_timeseries_{area}.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


def _rgb(tif: Path) -> np.ndarray:
    bands, valid, _ = load_bands(tif)
    img = np.dstack([bands["B4"], bands["B3"], bands["B2"]])
    img = np.clip(img / 0.25, 0, 1) ** (1 / 1.4)
    img[~valid] = 1
    return img


def _classes(d: Path, month: str) -> np.ndarray:
    with rasterio.open(d / METHOD / f"class_{month}.tif") as src:
        return src.read(1)


def _outline(ax, regions):
    ax.contour(regions == 1, levels=[0.5], colors=INK, linewidths=0.5)


def _legend_handles():
    return [Patch(color=COLORS[c], label=LABELS[c]) for c in COLORS] + \
           [Patch(color=NODATA_COLOR, label="No clear view")]


def plot_strip(d: Path, area: str, months: list[str], regions):
    months = [m for m in months if (d / METHOD / f"class_{m}.tif").exists()]
    fig, axes = plt.subplots(2, len(months), figsize=(3.3 * len(months), 6.6), facecolor=SURFACE)
    for i, m in enumerate(months):
        axes[0, i].imshow(_rgb(d / "s2" / f"s2_{m}.tif"))
        axes[1, i].imshow(_classes(d, m), cmap=CLASS_CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
        _outline(axes[1, i], regions)
        axes[0, i].set_title(m, fontsize=11, color=INK)
        for a in axes[:, i]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"True colour (top) and {METHOD_TITLE[METHOD]} classes (bottom); black line = "
                 "pre-collapse reservoir bed", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = FIG_DIR / f"{METHOD}_strip_{area}.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out}")


def make_gif(d: Path, area: str, df: pd.DataFrame, regions, step: int = 2):
    months = sorted(df.month.unique())
    bed = df[df.region == "reservoir_bed"].set_index("month")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE)
    for a in (a1, a2):
        a.set_xticks([])
        a.set_yticks([])
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    # downsample for a reasonable gif size
    im1 = a1.imshow(_rgb(d / "s2" / f"s2_{months[0]}.tif")[::step, ::step])
    im2 = a2.imshow(_classes(d, months[0])[::step, ::step], cmap=CLASS_CMAP, vmin=0, vmax=VMAX,
                    interpolation="nearest")
    a2.contour(regions[::step, ::step] == 1, levels=[0.5], colors=INK, linewidths=0.5)
    title = fig.suptitle("", fontsize=12, color=INK)

    def frame(i):
        m = months[i]
        im1.set_data(_rgb(d / "s2" / f"s2_{m}.tif")[::step, ::step])
        im2.set_data(_classes(d, m)[::step, ::step])
        r = bed.loc[m]
        tag = "before collapse" if m < COLLAPSE_DATE[:7] else "after collapse"
        title.set_text(f"{m} ({tag})    reservoir bed: water {r.water_km2:.0f} km², "
                       f"bare {r.bare_km2:.0f}, sparse {r.sparse_veg_km2:.0f}, dense {r.dense_veg_km2:.0f} km²")
        return im1, im2, title

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    anim = FuncAnimation(fig, frame, frames=len(months), blit=False)
    out = FIG_DIR / f"{METHOD}_animation_{area}.gif"
    anim.save(out, writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    parser.add_argument("--strip-months", nargs="+",
                        default=["2021-07", "2023-05", "2023-07", "2024-07", "2025-07", "2026-08"])
    parser.add_argument("--method", default="baseline", choices=list(METHOD_TITLE))
    args = parser.parse_args()
    global METHOD
    METHOD = args.method

    d = area_dir(args.area)
    df = pd.read_csv(ROOT / "outputs" / "tables" / f"{METHOD}_{args.area}.csv", dtype={"month": str})
    with rasterio.open(d / "regions.tif") as src:
        regions = src.read(1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_timeseries(df, args.area)
    plot_strip(d, args.area, args.strip_months, regions)
    make_gif(d, args.area, df, regions)


if __name__ == "__main__":
    main()
