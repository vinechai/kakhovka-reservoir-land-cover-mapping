# phase 1 scored like the cnn: its ndvi cut-offs are re-tuned inside each cross-validation fold.
#
# usage: python scripts/cv_phase1.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ROOT, area_dir
from src.labeling.imagery import _read_window
from src.model.metrics import scores

GRID_BARE = np.round(np.arange(0.15, 0.46, 0.01), 2)
GRID_DENSE = np.round(np.arange(0.35, 0.81, 0.01), 2)


def rule(ndvi, mndwi, t_bare, t_dense):
    return np.where(mndwi > 0, "water", np.where(ndvi < t_bare, "bare",
                    np.where(ndvi < t_dense, "sparse_veg", "dense_veg")))


def tune(ndvi, mndwi, y):
    best = (-1, None, None)
    for tb in GRID_BARE:
        for td in GRID_DENSE[GRID_DENSE > tb]:
            f = scores(y, rule(ndvi, mndwi, tb, td), sorted(set(y)))["macro_f1"]
            if f > best[0]:
                best = (f, tb, td)
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh")
    ap.add_argument("--cv-file", default=str(ROOT / "outputs" / "tables" / "cv_predictions_fcn.csv"))
    args = ap.parse_args()

    cv = pd.read_csv(args.cv_file, dtype={"month": str})
    s2 = area_dir(args.area) / "s2"
    nd, mn = [], []
    for _, p in cv.iterrows():
        b = _read_window(s2 / f"s2_{p.month}.tif", ["B3", "B4", "B8", "B11"], int(p.row), int(p.col), 0)
        nd.append(float(((b["B8"] - b["B4"]) / (b["B8"] + b["B4"]))[0, 0]))
        mn.append(float(((b["B3"] - b["B11"]) / (b["B3"] + b["B11"]))[0, 0]))
    cv["ndvi"], cv["mndwi"] = nd, mn

    cv["phase1_cv"] = None
    for k in sorted(cv.fold.unique()):
        tr, va = cv.fold != k, cv.fold == k
        tb, td = tune(cv.ndvi[tr].to_numpy(), cv.mndwi[tr].to_numpy(), cv.label[tr].to_numpy())
        cv.loc[va, "phase1_cv"] = rule(cv.ndvi[va], cv.mndwi[va], tb, td)
        print(f"fold {k}: tuned bare<{tb:.2f}, dense>={td:.2f}")

    y = cv.label.tolist()
    for col, name in [("phase1", "phase 1, tuned on all labels (optimistic)"),
                      ("phase1_cv", "phase 1, tuned per fold (fair)"), ("cnn", "cnn")]:
        s = scores(y, cv[col].tolist(), sorted(set(y)))
        print(f"{name:45s} acc {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}  "
              f"{ {c: round(f, 2) for c, f in s['per_class_f1'].items()} }")
    cv.to_csv(args.cv_file, index=False)


if __name__ == "__main__":
    main()
