# grades phase 1 and the cnn on hand labels: accuracy, macro f1, bootstrap interval for the
# difference, area-weighted accuracy, lyman noise check and class areas estimated from the labels.
#
# usage: python scripts/evaluate.py --set test

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # windows consoles default to cp1252

import numpy as np
import pandas as pd
import rasterio

from src.config import AREAS, ROOT, area_dir
from src.labeling.store import LABELS_DIR, load_labels, load_points
from src.model.metrics import confusion, scores, stratified_area_estimate, weighted_accuracy
from src.stats import PX_KM2
from src.model.samples import CLASS_CODES

CODE_TO_NAME = {v: k for k, v in CLASS_CODES.items()}
METHODS = {"phase1": "baseline", "cnn": "cnn"}


def predicted_at(area: str, method_dir: str, pts: pd.DataFrame) -> list[str]:
    out = []
    for _, p in pts.iterrows():
        with rasterio.open(area_dir(area) / method_dir / f"class_{p.target_month}.tif") as src:
            code = int(src.read(1, window=((p.row, p.row + 1), (p.col, p.col + 1)))[0, 0])
        out.append(CODE_TO_NAME.get(code, "nodata"))
    return out


def bootstrap_diff(y, a, b, n=5000, seed=0) -> dict:
    """95% interval of macro-F1(b) - macro-F1(a) and accuracy(b) - accuracy(a)."""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    d_f1, d_acc = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        classes = sorted(set(y))
        d_f1.append(scores(y[i], b[i], classes)["macro_f1"] - scores(y[i], a[i], classes)["macro_f1"])
        d_acc.append((b[i] == y[i]).mean() - (a[i] == y[i]).mean())
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"macro_f1_diff_95": q(d_f1), "accuracy_diff_95": q(d_acc)}


def lyman_noise(area: str, method_dir: str) -> float | None:
    t = ROOT / "outputs" / "tables" / f"{method_dir}_{area}.csv"
    if not t.exists():
        return None
    df = pd.read_csv(t, dtype={"month": str})
    lym = df[(df.region == "bilozerskyi_lyman") & df.usable & df.post_collapse].sort_values("month")
    surface = lym.water_km2 + lym.snow_ice_km2
    return float(surface.diff().abs().median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    ap.add_argument("--set", default="test")
    args = ap.parse_args()

    pts = load_points(args.set).merge(load_labels(args.set)[["point_id", "label"]], on="point_id")
    pts = pts[(pts.area == args.area) & (pts.label != "unsure")].reset_index(drop=True)
    if pts.empty:
        print(f"no labels in labels/labels_{args.set}.csv yet")
        return
    n_all = len(load_points(args.set))
    print(f"{args.set}: {len(pts)} labelled points used (of {n_all} sampled)")

    y = pts.label.tolist()
    preds = {m: predicted_at(args.area, d, pts) for m, d in METHODS.items()}
    strata_file = LABELS_DIR / f"strata_{args.set}.json"
    share = None
    if strata_file.exists():
        px = json.loads(strata_file.read_text())["pixels"]
        share = {k: v / sum(px.values()) for k, v in px.items()}

    report = {"set": args.set, "n": len(pts), "methods": {}}
    lines = [f"# Evaluation on `{args.set}` ({len(pts)} hand-labelled points)", ""]
    lines += ["| | accuracy | macro-F1 | area-weighted accuracy | Lyman month-to-month change (km², median) |",
              "|---|---|---|---|---|"]
    for m, p in preds.items():
        s = scores(y, p, sorted(set(y)))
        w = weighted_accuracy(y, p, pts.stratum, share) if share else None
        noise = lyman_noise(args.area, METHODS[m])
        report["methods"][m] = {**s, "area_weighted_accuracy": w, "lyman_noise_km2": noise}
        lines.append(f"| {m} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                     f"{'' if w is None else f'{w:.3f}'} | {'' if noise is None else f'{noise:.2f}'} |")
    diff = bootstrap_diff(y, preds["phase1"], preds["cnn"])
    report["cnn_minus_phase1"] = diff
    lo, hi = diff["macro_f1_diff_95"]
    lines += ["", f"cnn minus phase 1, 95% bootstrap interval: macro-F1 [{lo:+.3f}, {hi:+.3f}], "
              f"accuracy [{diff['accuracy_diff_95'][0]:+.3f}, {diff['accuracy_diff_95'][1]:+.3f}]",
              "", "Per-class F1:", "", "| class | phase1 | cnn |", "|---|---|---|"]
    for c in sorted(set(y)):
        lines.append(f"| {c} | {report['methods']['phase1']['per_class_f1'][c]:.2f} | "
                     f"{report['methods']['cnn']['per_class_f1'][c]:.2f} |")
    for m, p in preds.items():
        lines += ["", f"Confusion, {m} (rows = hand label):", "", confusion(y, p).to_markdown()]

    if strata_file.exists():
        meta = json.loads(strata_file.read_text())
        ref = pts[pts.target_month == meta["ref_month"]]
        if len(ref) >= 30:
            classes = ["water", "bare", "sparse_veg", "dense_veg"]
            est = stratified_area_estimate(ref.label, ref.stratum, meta["pixels"], PX_KM2, classes)
            for m, d in METHODS.items():
                t = ROOT / "outputs" / "tables" / f"{d}_{args.area}.csv"
                if t.exists():
                    tab = pd.read_csv(t, dtype={"month": str})
                    row = tab[(tab.month == meta["ref_month"]) & (tab.region == "reservoir_bed")].iloc[0]
                    est[f"{m} map"] = [row[f"{c}_km2"] for c in classes]
            report["area_estimate"] = {"month": meta["ref_month"], "n": len(ref),
                                       "table": est.round(1).to_dict("records")}
            lines += ["", f"Area on the reservoir bed, {meta['ref_month']}: estimated from the "
                      f"{len(ref)} points labelled for that month (95% interval), vs the maps (km²):",
                      "", est.round(1).to_markdown(index=False)]

    out = ROOT / "outputs" / "tables" / f"evaluation_{args.set}"
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    pts.assign(phase1=preds["phase1"], cnn=preds["cnn"]).to_csv(
        out.parent / f"evaluation_{args.set}_points.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
