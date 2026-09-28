# cnn input for one month: 10 bands, ndvi, mndwi, month as sin/cos, and optionally the same
# layers from earlier months (context).

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.baseline.index_classifier import load_bands, mndwi, ndvi
from src.config import ALL_MONTHS
from src.data.sentinel2 import BANDS

SPECTRAL = BANDS + ["ndvi", "mndwi"]      # per-month layers
SEASON = ["month_sin", "month_cos"]


def channel_names(context: tuple[int, ...] = ()) -> list[str]:
    names = SPECTRAL + SEASON
    for off in context:
        names += [f"{c}_m{off}" for c in SPECTRAL]
    return names


@lru_cache(maxsize=6)
def month_layers(s2_dir: str, month: str) -> tuple[np.ndarray, np.ndarray]:
    """(spectral layers float32 [12, H, W], valid mask [H, W]) for one month."""
    bands, valid, _ = load_bands(Path(s2_dir) / f"s2_{month}.tif")
    layers = [np.clip(bands[b], 0, 1.5) for b in BANDS] + [ndvi(bands), mndwi(bands)]
    return np.stack(layers).astype("float32"), valid


def shift_month(month: str, offset: int) -> str:
    y, m = map(int, month.split("-"))
    idx = y * 12 + (m - 1) + offset
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def feature_stack(s2_dir: Path, month: str, context: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """all channels (see channel_names) for `month`, plus its valid mask. a context month that
    wasn't pulled (gaps before the collapse) falls back to the current month's layers."""
    spec, valid = month_layers(str(s2_dir), month)
    m = int(month[5:7])
    h, w = valid.shape
    season = np.stack([np.full((h, w), np.sin(2 * np.pi * m / 12), "float32"),
                       np.full((h, w), np.cos(2 * np.pi * m / 12), "float32")])
    parts = [spec, season]
    for off in context:
        other = shift_month(month, off)
        parts.append(month_layers(str(s2_dir), other)[0] if other in ALL_MONTHS else spec)
    return np.concatenate(parts), valid
