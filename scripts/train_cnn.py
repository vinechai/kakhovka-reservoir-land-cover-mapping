# trains the pixel cnn on weak + hand labels, scores it with 4-fold cross-validation on the
# hand labels, then trains a final model on all of them. every run is logged to mlflow.
#
# usage: python scripts/train_cnn.py --dilations 1 1 1 1 1 --context -1 -2 --name fcn_rf11_m1_m2
#        mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn

from src.config import ROOT, area_dir
from src.labeling.store import load_points
from src.model.features import channel_names, feature_stack
from src.model.metrics import confusion, scores
from src.model.net import DILATIONS, PixelFCN, receptive_field
from src.model.samples import CLASSES, CLASS_CODES, extract_patches, hand_labels, weak_labels

COLS = ["month", "row", "col", "label"]
HALF = receptive_field() // 2  # set from --dilations in main()
CODE_TO_NAME = {v: k for k, v in CLASS_CODES.items()}


def phase1_at(area: str, pts: pd.DataFrame) -> list[str]:
    out = []
    for _, p in pts.iterrows():
        with rasterio.open(area_dir(area) / "baseline" / f"class_{p.month}.tif") as src:
            code = int(src.read(1, window=((p.row, p.row + 1), (p.col, p.col + 1)))[0, 0])
        out.append(CODE_TO_NAME.get(code, "nodata"))
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    """random 90-degree rotation + flip per batch: land cover has no 'up'."""
    x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(1) < 0.5 else x


def train_model(args, x_tr: np.ndarray, y_tr: np.ndarray, is_hand: np.ndarray, dev: str,
                log_prefix: str = "") -> PixelFCN:
    # hand labels are few: repeat them so they carry real weight in every epoch
    idx = np.arange(len(y_tr))
    if is_hand.any():
        idx = np.concatenate([idx] + [idx[is_hand]] * (args.hand_repeat - 1))
    print(f"  {log_prefix}{len(y_tr)} examples ({is_hand.sum()} hand), {len(idx)} per epoch")

    # class weights: rare classes (sparse!) count more. "sqrt": 1/sqrt(frequency), a mild boost;
    # "balanced": 1/frequency, every class weighs the same in total
    counts = np.bincount(y_tr[idx], minlength=len(CLASSES)).astype(float)
    power = 0.5 if args.class_weight == "sqrt" else 1.0
    weights = np.where(counts > 0, 1 / np.maximum(counts, 1) ** power, 0)
    weights = weights / weights[counts > 0].mean()

    model = PixelFCN(x_tr.shape[1], len(CLASSES), width=args.width,
                     dilations=args.dilations).to(dev)
    centre = x_tr[:, :, HALF, HALF].astype("float32")
    model.mean.copy_(torch.tensor(centre.mean(0)))
    model.std.copy_(torch.tensor(centre.std(0) + 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev),
                                  label_smoothing=0.05)

    # keep all examples on the gpu (float16) when they fit: copying every batch from main
    # memory was the bottleneck (~45 s per epoch vs a few seconds)
    xs = torch.from_numpy(x_tr)
    if dev == "cuda":
        try:
            xs = xs.to(dev)
        except torch.OutOfMemoryError:  # too big for the card: stay in main memory (slower)
            torch.cuda.empty_cache()
            print(f"  {log_prefix}examples don't fit on the gpu, batching from main memory")
    ys = torch.from_numpy(y_tr).to(dev)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(idx)
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = torch.from_numpy(order[i:i + args.batch])
            xb = augment(xs[b.to(xs.device)].to(dev).float())
            yb = ys[b.to(dev)]
            loss = loss_fn(model(xb)[:, :, 0, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {log_prefix}epoch {epoch + 1:2d} loss {total / len(order):.3f}")
    model.eval()
    del xs, ys
    torch.cuda.empty_cache()
    return model


def predict_points(model, x_np: np.ndarray, dev: str) -> list[str]:
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = model(x)[:, :, 0, 0].argmax(1).cpu().numpy()
    return [CLASSES[i] for i in pred]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh")
    ap.add_argument("--context", type=int, nargs="*", default=[],
                    help="month offsets to add as extra input, e.g. -1 -2")
    ap.add_argument("--per-class", type=int, default=300, help="weak labels per class per month")
    ap.add_argument("--hand-repeat", type=int, default=40,
                    help="how many times each hand label appears per epoch")
    ap.add_argument("--no-hand", action="store_true", help="ablation: weak labels only")
    ap.add_argument("--class-weight", default="sqrt", choices=["sqrt", "balanced"])
    ap.add_argument("--weak-sparse", action="store_true",
                    help="also use phase 1's sparse band as noisy weak labels")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--no-final", action="store_true", help="skip training the final model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--dilations", type=int, nargs="+", default=list(DILATIONS),
                    help="per-layer dilations; sets how far each pixel sees (receptive field). "
                         "default 1 1 2 4 2 1 -> 23 px; 1 1 1 1 1 -> 11 px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    context = tuple(args.context)
    name = args.name or ("fcn" + "".join(f"_m{-o}" for o in context)
                         + ("_nohand" if args.no_hand else ""))
    s2_dir = area_dir(args.area) / "s2"
    stack = lambda m: feature_stack(s2_dir, m, context)
    global HALF
    HALF = receptive_field(args.dilations) // 2  # patch = exactly what the network sees


    hand = hand_labels(args.area, seed=args.seed, val_frac=0)
    rng = np.random.default_rng(args.seed)
    # stratified folds: each fold gets its share of every class
    hand["fold"] = -1
    for _, grp in hand.groupby("label"):
        hand.loc[grp.index, "fold"] = rng.permutation(len(grp)) % args.folds
    test_xy = load_points("test")[["x", "y"]].to_numpy()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    exp_name = f"cnn_{args.area}"
    if mlflow.get_experiment_by_name(exp_name) is None:
        # experimentKind: show it as a classic model-training experiment (runs, metrics,
        # charts), not the llm-app layout (traces) mlflow 3 opens by default
        mlflow.create_experiment(exp_name, artifact_location=(ROOT / "mlartifacts").as_uri(),
                                 tags={"mlflow.experimentKind": "custom_model_development"})
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=name):
        chans = channel_names(context)
        mlflow.log_params({**vars(args), "context": list(context), "channels": len(chans),
                           "receptive_field_px": receptive_field(args.dilations),
                           "patch_px": 2 * HALF + 1,
                           "n_hand": len(hand), "device": dev})

        # --- examples, prepared once and reused by every fold and the final model.
        # weak labels keep GUARD_M away from every test point and every hand label, so
        # whichever fold a hand label is held out in, no weak label sits on top of it.
        t0 = time.time()
        guard = np.vstack([test_xy, hand[["x", "y"]].to_numpy()])
        weak = weak_labels(args.area, args.per_class, guard, seed=args.seed,
                           weak_sparse=args.weak_sparse)
        x_weak = extract_patches(weak[COLS], stack, HALF)
        y_weak = weak.label.map(CLASSES.index).to_numpy()
        x_hand = extract_patches(hand[COLS].reset_index(drop=True), stack, HALF)
        y_hand = hand.label.map(CLASSES.index).to_numpy()
        print(f"weak {weak.label.value_counts().to_dict()}, hand {len(hand)}, "
              f"channels {x_weak.shape[1]}  ({time.time() - t0:.0f}s)")
        mlflow.log_params({"n_weak": len(weak)})

        def fit(hand_mask):
            use = hand_mask & (not args.no_hand)
            x = np.concatenate([x_weak, x_hand[use]])
            y = np.concatenate([y_weak, y_hand[use]])
            is_hand = np.r_[np.zeros(len(y_weak), bool), np.ones(use.sum(), bool)]
            return x, y, is_hand

        # --- cross-validation
        hand["cnn"] = None
        folds = hand.fold.to_numpy()
        for k in range(args.folds):
            model = train_model(args, *fit(folds != k), dev, log_prefix=f"fold {k}: ")
            val_idx = hand.index[folds == k]
            hand.loc[val_idx, "cnn"] = predict_points(model, x_hand[folds == k], dev)
            s = scores(hand.loc[val_idx, "label"].tolist(), hand.loc[val_idx, "cnn"].tolist())
            mlflow.log_metrics({"fold_acc": s["accuracy"], "fold_macro_f1": s["macro_f1"]}, step=k)
            print(f"fold {k}: acc {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}  "
                  f"({time.time() - t0:.0f}s)")

        hand["phase1"] = phase1_at(args.area, hand)
        cv = scores(hand.label.tolist(), hand.cnn.tolist())
        p1 = scores(hand.label.tolist(), hand.phase1.tolist())
        mlflow.log_metrics({"cv_acc": cv["accuracy"], "cv_macro_f1": cv["macro_f1"],
                            "phase1_acc": p1["accuracy"], "phase1_macro_f1": p1["macro_f1"],
                            **{f"cv_f1_{c}": f for c, f in cv["per_class_f1"].items()},
                            **{f"phase1_f1_{c}": f for c, f in p1["per_class_f1"].items()}})
        print(f"\nCV over {len(hand)} hand labels:  CNN acc {cv['accuracy']:.3f} "
              f"macro-F1 {cv['macro_f1']:.3f}  |  phase 1 acc {p1['accuracy']:.3f} "
              f"macro-F1 {p1['macro_f1']:.3f}")
        print("per-class F1  CNN:", {c: round(f, 2) for c, f in cv["per_class_f1"].items()})
        print("per-class F1  P1: ", {c: round(f, 2) for c, f in p1["per_class_f1"].items()})
        print(confusion(hand.label.tolist(), hand.cnn.tolist()))

        cv_path = ROOT / "outputs" / "tables" / f"cv_predictions_{name}.csv"
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        hand[["set", "point_id", "month", "row", "col", "stratum", "label", "fold", "cnn",
              "phase1"]].to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path))

        # --- final model on all hand labels
        if not args.no_final:
            model = train_model(args, *fit(np.ones(len(hand), bool)), dev, log_prefix="final: ")
            out = ROOT / "data" / "models" / f"{name}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "channels": chans, "context": context,
                        "classes": CLASSES, "width": args.width, "half": HALF,
                        "dilations": list(args.dilations)}, out)
            mlflow.log_artifact(str(out))
            out.with_suffix(".json").write_text(json.dumps(
                {"run_id": mlflow.active_run().info.run_id, "cv": cv, "phase1": p1}, indent=1))
            print(f"saved {out}")


if __name__ == "__main__":
    main()
