# runs a trained cnn over every monthly composite: a class map per month and the km2 table.
#
# usage: python scripts/predict_cnn.py [--model fcn]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import torch

from src.baseline.index_classifier import NODATA, WINTER_MONTHS
from src.config import ALL_MONTHS, AREAS, ROOT, area_dir
from src.model.features import feature_stack
from src.model.net import DILATIONS, PixelFCN
from src.model.samples import CLASS_CODES
from src.stats import tabulate

TILE = 512


def load_model(path: Path, dev: str) -> tuple[PixelFCN, dict]:
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = PixelFCN(len(ck["channels"]), len(ck["classes"]), width=ck["width"],
                     dilations=ck.get("dilations", DILATIONS)).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


@torch.no_grad()
def predict_image(model, feats: np.ndarray, dev: str) -> np.ndarray:
    """class probabilities [K, H, W] for a whole feature stack, tile by tile. the image is
    zero-padded by the network's reach (it uses no padding itself), so every pixel gets an
    output and tiles join without seams."""
    margin = model.rf // 2
    c, h, w = feats.shape
    padded = np.pad(feats, ((0, 0), (margin, margin), (margin, margin)))
    out = None
    for r in range(0, h, TILE):
        for col in range(0, w, TILE):
            th, tw = min(TILE, h - r), min(TILE, w - col)
            x = padded[:, r:r + th + 2 * margin, col:col + tw + 2 * margin]
            p = torch.softmax(model(torch.from_numpy(np.ascontiguousarray(x[None])).to(dev)), 1)
            p = p[0].cpu().numpy()
            if out is None:
                out = np.zeros((p.shape[0], h, w), dtype="float32")
            out[:, r:r + th, col:col + tw] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="velykyi_luh", choices=list(AREAS))
    ap.add_argument("--model", default="fcn_rf11_m1_m2",
                    help="name in data/models/ (default: the chosen final model)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = area_dir(args.area)
    model, ck = load_model(ROOT / "data" / "models" / f"{args.model}.pt", dev)
    codes = np.array([CLASS_CODES[c] for c in ck["classes"]], dtype="uint8")
    snow = ck["classes"].index("snow_ice")
    out_dir = d / "cnn"
    out_dir.mkdir(exist_ok=True)

    with rasterio.open(d / "s2" / f"s2_{ALL_MONTHS[0]}.tif") as src:
        profile = src.profile
    profile.update(count=1, dtype="uint8", nodata=None)

    for month in ALL_MONTHS:
        feats, valid = feature_stack(d / "s2", month, tuple(ck["context"]))
        prob = predict_image(model, feats, dev)
        if int(month[5:7]) not in WINTER_MONTHS:
            prob[snow] = 0
        classes = codes[prob.argmax(0)]
        classes[~valid] = NODATA
        conf = (prob.max(0) * 100).round().astype("uint8")
        with rasterio.open(out_dir / f"class_{month}.tif", "w", **profile) as dst:
            dst.write(classes, 1)
        with rasterio.open(out_dir / f"conf_{month}.tif", "w", **profile) as dst:
            dst.write(conf, 1)
        print(f"{month} done")

    table = tabulate(out_dir, d)
    out = ROOT / "outputs" / "tables" / f"cnn_{args.area}.csv"
    table.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
