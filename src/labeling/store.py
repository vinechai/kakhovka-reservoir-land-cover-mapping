# hand labels are csv files in labels/, tracked in git: points_<set>.csv (sampled points) and
# labels_<set>.csv (one row per labelled point, labelling again replaces the row).

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT

LABELS_DIR = ROOT / "labels"

# classes to choose from, judged inside the 10 m pixel on the high-res map. vegetation is split
# by cover, not type: in the pilot (labels_pilot_v1.csv) willow vs grass vs reeds couldn't be
# told apart from above.
LABEL_CLASSES = {
    "water": "Water (open water, incl. shallow water over sand)",
    "bare": "Bare (plants fill less than 1 of the 4 quarters)",
    "sparse_veg": "Sparse vegetation (in between: plants and ground both clearly there)",
    "dense_veg": "Dense vegetation (plants fill ~all 4 quarters, ground in at most ~1)",
    "unsure": "Can't tell (e.g. brown plants vs brown ground)",
}

LABELLING_RULES = """
Judge only what's inside the small yellow square (10 m) on the high-res map.
All fractions are about **plants**. Mentally split the square into **4 quarters**:
- **Bare**: plants fill less than 1 quarter's worth (mostly sand / mud).
- **Dense**: plants fill nearly all 4 quarters; ground shows in at most ~1 quarter's worth.
  Lumpy or smooth, dark or light green doesn't matter.
- **Sparse**: anything in between: plants and sand / mud both clearly there.
- **Water**: you can see a water surface (smooth even sheet, reflections / glare, ripples, a
  waterline), even if sand shows through underneath.
- **Wet sand** (grey, damp, sand texture, no visible water surface) counts as **bare**.
- **Can't tell**: e.g. wet sand vs a few cm of water, or brownish plants vs brown ground.
  Add a short note saying which.
- For water vs wet sand, decide from the high-res map, **not** the MNDWI line in the chart:
  that line is what Phase 1 uses, and these labels are meant to check it.
- Tick **tree crowns** only if you see distinct crowns casting shadows.
- Don't decide the species; cover is what we record.
"""

LABEL_COLUMNS = ["point_id", "label", "trees_visible", "confident", "note", "target_month",
                 "imagery_date", "labeller", "labelled_at"]


def points_path(point_set: str, root: Path = LABELS_DIR) -> Path:
    return root / f"points_{point_set}.csv"


def labels_path(point_set: str, root: Path = LABELS_DIR) -> Path:
    return root / f"labels_{point_set}.csv"


def list_point_sets(root: Path = LABELS_DIR) -> list[str]:
    return sorted(p.stem.removeprefix("points_") for p in root.glob("points_*.csv"))


def load_points(point_set: str, root: Path = LABELS_DIR) -> pd.DataFrame:
    return pd.read_csv(points_path(point_set, root))


def load_labels(point_set: str, root: Path = LABELS_DIR) -> pd.DataFrame:
    p = labels_path(point_set, root)
    if not p.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(p, dtype={"target_month": str, "imagery_date": str, "note": str})
    if "trees_visible" not in df:
        df["trees_visible"] = False
    return df


def save_label(point_set: str, point_id: int, label: str, confident: bool, note: str,
               target_month: str, imagery_date: str, labeller: str,
               trees_visible: bool = False, root: Path = LABELS_DIR) -> pd.DataFrame:
    if label not in LABEL_CLASSES:
        raise ValueError(f"unknown label: {label}")
    df = load_labels(point_set, root)
    df = df[df.point_id != point_id]
    row = {"point_id": point_id, "label": label, "trees_visible": bool(trees_visible),
           "confident": confident, "note": note or "",
           "target_month": target_month, "imagery_date": imagery_date or "",
           "labeller": labeller, "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("point_id")
    root.mkdir(parents=True, exist_ok=True)
    df[LABEL_COLUMNS].to_csv(labels_path(point_set, root), index=False)
    return df
