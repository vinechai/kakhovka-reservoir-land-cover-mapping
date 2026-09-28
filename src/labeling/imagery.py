# what the labelling app shows for a point: sentinel-2 true / false colour cut-outs, ndvi and
# mndwi over time, and the capture date of esri's high-res imagery at that spot.

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from rasterio.windows import Window

WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
ESRI_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
              "tile/{z}/{y}/{x}")


def _read_window(tif: Path, names: list[str], row: int, col: int, half: int) -> dict:
    with rasterio.open(tif) as src:
        idx = {d: i + 1 for i, d in enumerate(src.descriptions)}
        win = Window(col - half, row - half, 2 * half + 1, 2 * half + 1)
        return {n: src.read(idx[n], window=win, boundless=True, fill_value=0).astype("float32") / 1e4
                for n in names}


def _stretch(img: np.ndarray, top: float) -> np.ndarray:
    return np.clip(img / top, 0, 1) ** (1 / 1.4)


def chips(tif: Path, row: int, col: int, half: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(true colour, false colour) rgb arrays of (2*half+1)^2 pixels centred on the point."""
    b = _read_window(tif, ["B2", "B3", "B4", "B8"], row, col, half)
    true = _stretch(np.dstack([b["B4"], b["B3"], b["B2"]]), 0.25)
    false = _stretch(np.dstack([b["B8"], b["B4"], b["B3"]]), 0.45)
    return true, false


def index_series(s2_dir: Path, months: list[str], row: int, col: int) -> list[dict]:
    """median ndvi / mndwi of the 3x3 pixels around the point, per month."""
    out = []
    for m in months:
        tif = s2_dir / f"s2_{m}.tif"
        if not tif.exists():
            continue
        try:
            b = _read_window(tif, ["B3", "B4", "B8", "B11"], row, col, 1)
        except (KeyError, rasterio.errors.RasterioError):
            continue  # file being replaced by a download right now, skip this month
        valid = b["B4"] > 0
        if not valid.any():
            continue
        nd = lambda a, c: (a - c) / np.maximum(a + c, 1e-4)
        out.append({"month": m,
                    "ndvi": float(np.median(nd(b["B8"], b["B4"])[valid])),
                    "mndwi": float(np.median(nd(b["B3"], b["B11"])[valid]))})
    return out


@lru_cache(maxsize=1)
def _current_metadata_url() -> str:
    cfg = json.load(urllib.request.urlopen(WAYBACK_CONFIG, timeout=20))
    latest = max(cfg.values(), key=lambda v: v["itemTitle"])
    return latest["metadataLayerUrl"]


@lru_cache(maxsize=4096)
def esri_capture_date(lon: float, lat: float) -> str | None:
    """capture date (YYYY-MM-DD) of the sharpest esri world imagery at this point, or None.
    the metadata service has one layer per zoom band; lower layer numbers = sharper imagery."""
    try:
        url = _current_metadata_url()
        for layer in range(0, 12):
            q = (f"{url}/{layer}/query?geometry={lon},{lat}&geometryType=esriGeometryPoint"
                 f"&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=SRC_DATE,SRC_RES"
                 f"&returnGeometry=false&f=json")
            feats = json.load(urllib.request.urlopen(q, timeout=20)).get("features", [])
            if feats and feats[0]["attributes"].get("SRC_DATE"):
                d = str(feats[0]["attributes"]["SRC_DATE"])
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    except Exception:
        return None
    return None
