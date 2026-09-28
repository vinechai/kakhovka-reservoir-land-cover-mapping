# grid + tile stitching. the real download talks to earth engine, so here the per-tile fetch is
# swapped for a fake that returns known numbers, and we check they end up in the right place.

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from src.data import gee

BBOX = (34.30, 47.45, 34.302, 47.4515)  # tiny box, ~15 x 17 pixels


def test_area_grid_is_snapped_and_covers_the_box():
    x_min, y_max, w, h = gee.area_grid(BBOX, "EPSG:32636", 10)
    assert x_min % 10 == 0 and y_max % 10 == 0
    west, south, east, north = transform_bounds("EPSG:4326", "EPSG:32636", *BBOX)
    assert x_min <= west and y_max >= north
    assert x_min + w * 10 >= east and y_max - h * 10 <= south


def test_tiles_are_stitched_in_the_right_place(tmp_path, monkeypatch):
    x_min, y_max, width, height = gee.area_grid(BBOX, "EPSG:32636", 10)

    def fake_fetch(image, bands, x0, y0, w, h, crs, scale):
        # every pixel's value encodes its global (row, col) and band
        r0, c0 = round((y_max - y0) / scale), round((x0 - x_min) / scale)
        rows, cols = np.mgrid[r0:r0 + h, c0:c0 + w]
        return np.stack([rows * 1000 + cols + b * 1_000_000 for b in range(len(bands))])

    monkeypatch.setattr(gee, "_fetch_tile", fake_fetch)
    # tile=6 doesn't divide the grid evenly, so edge tiles are smaller: the tricky case
    out = gee.download_image(None, ["a", "b"], BBOX, tmp_path / "x.tif", dtype="int32",
                             crs="EPSG:32636", scale=10, tile=6)

    with rasterio.open(out) as src:
        arr = src.read()
        assert src.descriptions == ("a", "b")
        assert src.transform.c == x_min and src.transform.f == y_max
    rows, cols = np.mgrid[0:height, 0:width]
    assert arr.shape == (2, height, width)
    np.testing.assert_array_equal(arr[0], rows * 1000 + cols)
    np.testing.assert_array_equal(arr[1], rows * 1000 + cols + 1_000_000)
