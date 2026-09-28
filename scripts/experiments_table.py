# all cnn runs from mlflow side by side, plus phase 1 on the same cross-validation points.
#
# usage: python scripts/experiments_table.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # windows consoles default to cp1252
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from src.config import ROOT
from scripts.evaluate import bootstrap_diff
from src.model.metrics import scores

DESCRIPTIONS = {
    "fcn": "base: 23 px view, weak labels for water / bare / dense / snow + hand labels",
    "fcn_bal": "fewer weak labels, hand labels x100, fully balanced class weights",
    "fcn_ws": "base + phase 1's sparse band as noisy weak labels",
    "fcn_ws_m1_m2": "fcn_ws + the two previous months as extra input",
    "fcn_nohand": "base without any hand labels (control)",
    "fcn_rf11": "base with a smaller view: 11 px (110 m) instead of 23 px",
    "fcn_m1_m2": "base + the two previous months as extra input",
    "fcn_rf11_m1_m2": "11 px view + the two previous months as extra input",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh")
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    exp = mlflow.get_experiment_by_name(f"cnn_{args.area}")
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    rows = []
    for r in sorted(runs, key=lambda r: r.info.start_time):
        m = r.data.metrics
        if "cv_macro_f1" not in m:
            continue
        rows.append({"run": r.info.run_name, "what": DESCRIPTIONS.get(r.info.run_name, ""),
                     "accuracy": m["cv_acc"], "macro-F1": m["cv_macro_f1"],
                     **{f"F1 {c}": m.get(f"cv_f1_{c}") for c in
                        ["water", "bare", "sparse_veg", "dense_veg"]}})

    # phase 1 on the same points, cut-offs re-tuned per fold (added to the cv file by cv_phase1.py)
    cv_file = ROOT / "outputs" / "tables" / "cv_predictions_fcn.csv"
    cv = pd.read_csv(cv_file)
    # paired bootstrap: each cnn run vs phase 1 on the very same points (same folds, same seed)
    for row in rows:
        f = ROOT / "outputs" / "tables" / f"cv_predictions_{row['run']}.csv"
        if f.exists() and "phase1_cv" in cv:
            other = pd.read_csv(f).set_index(["set", "point_id"]).loc[
                cv.set_index(["set", "point_id"]).index]
            lo, hi = bootstrap_diff(cv.label, cv.phase1_cv, other.cnn.to_numpy())["macro_f1_diff_95"]
            row["vs phase 1 (95%)"] = f"{lo:+.2f} to {hi:+.2f}"
    if "phase1_cv" in cv:
        s = scores(cv.label.tolist(), cv.phase1_cv.tolist())
        rows.append({"run": "phase 1 (per fold)", "what": "index rules, cut-offs tuned per fold",
                     "accuracy": s["accuracy"], "macro-F1": s["macro_f1"],
                     **{f"F1 {c}": s["per_class_f1"].get(c) for c in
                        ["water", "bare", "sparse_veg", "dense_veg"]}})

    t = pd.DataFrame(rows).round(3)
    out = ROOT / "outputs" / "tables" / f"experiments_{args.area}.md"
    out.write_text(t.to_markdown(index=False), encoding="utf-8")
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
