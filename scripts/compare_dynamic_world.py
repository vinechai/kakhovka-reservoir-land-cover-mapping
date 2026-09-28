# dynamic world vs our maps and hand labels, written to outputs/tables/dynamic_world_<area>.md
#
# usage: python scripts/compare_dynamic_world.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # windows consoles default to cp1252

import pandas as pd
import rasterio

from scripts.pull_dynamic_world import DW_CLASSES
from src.config import AREAS, ROOT, area_dir
from src.model.samples import CLASS_CODES, hand_labels

DW_NAMES = {**dict(enumerate(DW_CLASSES)), 255: "nodata"}
OUR_NAMES = {**{v: k for k, v in CLASS_CODES.items()}, 0: "nodata"}


def read(path: Path):
    with rasterio.open(path) as src:
        return src.read(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    ap.add_argument("--months", nargs="+", default=["2024-08", "2025-08", "2026-08"])
    args = ap.parse_args()

    d = area_dir(args.area)
    bed = read(d / "regions.tif") == 1
    lines = [f"# Dynamic World vs our maps, `{args.area}` reservoir bed", ""]

    comp = {}
    for m in args.months:
        dw = pd.Series(read(d / "dynamic_world" / f"dw_{m}.tif")[bed]).map(DW_NAMES)
        comp[m] = (dw.value_counts() / 1e4).round(1)
    lines += ["## Dynamic World classes on the bed (km²)", "",
              pd.DataFrame(comp).fillna(0).to_markdown(), ""]

    for method in ("baseline", "cnn"):
        if not (d / method).exists():
            continue
        for m in args.months:
            ours = pd.Series(read(d / method / f"class_{m}.tif")[bed]).map(OUR_NAMES)
            dw = pd.Series(read(d / "dynamic_world" / f"dw_{m}.tif")[bed]).map(DW_NAMES)
            t = pd.crosstab(ours.rename(method), dw.rename("dynamic world"), normalize="index")
            lines += [f"## {m}: what Dynamic World calls each `{method}` class (row shares)", "",
                      (t * 100).round(0).astype(int).to_markdown(), ""]

    h = hand_labels(args.area, val_frac=0)
    h = h[h.month.isin(args.months)]
    if len(h):
        h["dw"] = [DW_NAMES[int(read(d / "dynamic_world" / f"dw_{m}.tif")[r, c])]
                   for m, r, c in zip(h.month, h.row, h.col)]
        lines += [f"## Hand labels vs Dynamic World ({len(h)} points)", "",
                  pd.crosstab(h.label.rename("hand label"), h.dw.rename("dynamic world")).to_markdown(), ""]

    out = ROOT / "outputs" / "tables" / f"dynamic_world_{args.area}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
