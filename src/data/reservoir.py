# pre-collapse reservoir outline from jrc global surface water: pixels that were water at least
# 50% of the time in 1984-2021, small ponds dropped, split into named regions.

from __future__ import annotations

from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import shape

JRC_GSW = "JRC/GSW1_4/GlobalSurfaceWater"


def footprint_image(occurrence_min: int = 50, min_pixels: int = 500) -> ee.Image:
    """1 where the pixel was water >= occurrence_min % of the time 1984-2021, else 0.
    min_pixels is in jrc 30 m pixels (500 px ~ 0.45 km2)."""
    occurrence = ee.Image(JRC_GSW).select("occurrence")
    water = occurrence.gte(occurrence_min).selfMask()
    # connectedPixelCount caps at maxSize, which is all we need to tell big from small
    size = water.connectedPixelCount(maxSize=min_pixels, eightConnected=True)
    footprint = water.updateMask(size.gte(min_pixels))
    return footprint.unmask(0).rename("reservoir").toUint8()


def occurrence_image() -> ee.Image:
    """raw occurrence (0-100 %), kept alongside the mask so the threshold can be revisited."""
    return ee.Image(JRC_GSW).select("occurrence").unmask(0).toUint8()


def footprint_to_polygons(mask_tif: Path) -> gpd.GeoDataFrame:
    """vectorise the downloaded mask into polygons (area in km2), biggest first."""
    with rasterio.open(mask_tif) as src:
        mask = src.read(1)
        geoms = [
            shape(g) for g, v in shapes(mask, mask=mask == 1, transform=src.transform) if v == 1
        ]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
    gdf["area_km2"] = gdf.area / 1e6
    return gdf.sort_values("area_km2", ascending=False, ignore_index=True)


def mask_area_km2(mask: np.ndarray, scale: int) -> float:
    return float((mask == 1).sum()) * scale * scale / 1e6


# region codes in regions.tif
OUTSIDE, RESERVOIR_BED, OTHER_WATER = 0, 1, 2
# separate water bodies get codes from 10 up, in the order they're listed in config


def build_regions(mask_tif: Path, separate: dict[str, tuple[float, float]]):
    """split the footprint into named regions:
    - reservoir_bed: the biggest footprint polygon (the reservoir proper)
    - each separate water body from config (the polygon containing its point)
    - other_water: leftover small ponds/bays, kept out of everything but the maps
    returns (regions array, gdf of polygons with name + code, rasterio profile of the mask)."""
    polys = footprint_to_polygons(mask_tif)
    polys["name"] = "other_water"
    polys["code"] = OTHER_WATER
    polys.loc[0, ["name", "code"]] = ["reservoir_bed", RESERVOIR_BED]

    pts = gpd.GeoSeries(
        gpd.points_from_xy(*zip(*separate.values())), crs="EPSG:4326"
    ).to_crs(polys.crs) if separate else []
    for i, (name, pt) in enumerate(zip(separate, pts)):
        hit = polys.index[polys.contains(pt)]
        if len(hit) == 0:
            raise ValueError(f"{name}: point not inside any footprint polygon")
        if hit[0] == 0:
            raise ValueError(f"{name}: point falls in the main reservoir polygon")
        polys.loc[hit[0], ["name", "code"]] = [name, 10 + i]

    with rasterio.open(mask_tif) as src:
        regions = rasterize(
            ((g, int(c)) for g, c in zip(polys.geometry, polys.code)),
            out_shape=(src.height, src.width), transform=src.transform,
            fill=OUTSIDE, dtype="uint8",
        )
        profile = src.profile
    return regions, polys, profile
