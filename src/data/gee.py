# earth engine helpers: init, and tiled downloads onto our fixed 10 m grid as geotiff.

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from src.config import CRS, GEE_PROJECT, SCALE


def init(project: str = GEE_PROJECT):
    ee.Initialize(project=project)


def area_grid(bbox: tuple[float, float, float, float], crs: str = CRS, scale: int = SCALE):
    """turn a lon/lat box into a pixel grid on the utm crs, snapped to whole pixels.
    returns (x_min, y_max, width, height) in crs units / pixels."""
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox)
    x_min = math.floor(west / scale) * scale
    y_max = math.ceil(north / scale) * scale
    width = math.ceil((east - x_min) / scale)
    height = math.ceil((y_max - south) / scale)
    return x_min, y_max, width, height


def area_geometry(bbox: tuple[float, float, float, float]) -> ee.Geometry:
    return ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)


def _fetch_tile(image: ee.Image, bands: list[str], x0: float, y0: float, w: int, h: int,
                crs: str, scale: int, retries: int = 4) -> np.ndarray:
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": bands,
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": scale, "shearX": 0, "translateX": x0,
                "shearY": 0, "scaleY": -scale, "translateY": y0,
            },
            "crsCode": crs,
        },
    }
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels(request)
            # structured array (h, w) with one field per band -> (bands, h, w)
            return np.stack([arr[b] for b in bands])
        except ee.EEException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  tile failed ({e}), retrying in {wait}s")
            time.sleep(wait)


def download_image(image: ee.Image, bands: list[str], bbox, out_path: Path, dtype: str,
                   nodata=0, crs: str = CRS, scale: int = SCALE, tile: int = 1024) -> Path:
    """pull `bands` of `image` over `bbox` to a geotiff on the shared grid.
    masked pixels come back as `nodata`, the caller should unmask() to that value first."""
    x_min, y_max, width, height = area_grid(bbox, crs, scale)
    out = np.full((len(bands), height, width), nodata, dtype=dtype)

    tiles = [(r, c) for r in range(0, height, tile) for c in range(0, width, tile)]
    for i, (r, c) in enumerate(tiles, 1):
        h, w = min(tile, height - r), min(tile, width - c)
        print(f"  tile {i}/{len(tiles)} ({w}x{h})")
        out[:, r:r + h, c:c + w] = _fetch_tile(
            image, bands, x_min + c * scale, y_max - r * scale, w, h, crs, scale
        ).astype(dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": len(bands),
        "dtype": dtype, "crs": crs, "transform": from_origin(x_min, y_max, scale, scale),
        "nodata": nodata, "compress": "deflate", "tiled": True,
    }
    # write to a temp file, then swap it in: anyone reading out_path (the labelling app, a
    # rerun) sees either the complete old file or the complete new one, never half a file
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(out)
        dst.descriptions = tuple(bands)
    os.replace(tmp, out_path)
    return out_path
