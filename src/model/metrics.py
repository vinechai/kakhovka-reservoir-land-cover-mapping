# scores: accuracy, macro f1 (every class counts equally), confusion tables, area-weighted
# accuracy and class areas estimated from a stratified sample.

from __future__ import annotations

import numpy as np
import pandas as pd


def scores(y_true: list[str], y_pred: list[str], classes: list[str] | None = None) -> dict:
    y, p = np.asarray(y_true), np.asarray(y_pred)
    classes = classes or sorted(set(y))  # only classes that actually occur in the labels
    f1 = {}
    for c in classes:
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"accuracy": float((y == p).mean()), "macro_f1": float(np.mean(list(f1.values()))),
            "per_class_f1": f1, "n": int(len(y))}


def confusion(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    return pd.crosstab(pd.Series(y_true, name="label"), pd.Series(y_pred, name="predicted"))


def weighted_accuracy(y_true, y_pred, strata, stratum_share: dict[str, float]) -> float:
    """accuracy re-weighted to the real area share of each stratum. the test points
    over-sample rare strata on purpose; this undoes that for an 'over the whole bed' figure."""
    y, p, s = map(np.asarray, (y_true, y_pred, strata))
    acc = {k: float((y[s == k] == p[s == k]).mean()) for k in set(s)}
    total = sum(stratum_share[k] for k in acc)
    return sum(acc[k] * stratum_share[k] for k in acc) / total


def stratified_area_estimate(labels, strata, stratum_pixels: dict[str, int], px_km2: float,
                             classes: list[str]) -> pd.DataFrame:
    """class areas with 95% intervals from a stratified random sample of labels, independent
    of any map (stratified estimator as in olofsson et al. 2014)."""
    labels, strata = np.asarray(labels), np.asarray(strata)
    total = sum(stratum_pixels[h] for h in set(strata))
    rows = []
    for k in classes:
        share, var = 0.0, 0.0
        for h in set(strata):
            in_h = strata == h
            n_h = in_h.sum()
            w_h = stratum_pixels[h] / total
            p_hk = (labels[in_h] == k).mean()
            share += w_h * p_hk
            if n_h > 1:
                var += w_h ** 2 * p_hk * (1 - p_hk) / (n_h - 1)
        km2 = share * total * px_km2
        half = 1.96 * np.sqrt(var) * total * px_km2
        rows.append({"class": k, "km2": km2, "ci95_low": km2 - half, "ci95_high": km2 + half})
    return pd.DataFrame(rows)
