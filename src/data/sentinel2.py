# monthly cloud-masked sentinel-2 median composites. clouds from the s2cloudless probability
# collection, shadows by projecting clouds along the sun direction, cloud score+ as fallback.

from __future__ import annotations

import ee

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

# 10 m: blue, green, red, nir. 20 m: swir1, swir2 and red-edge (B5-B7, B8A), resampled to 10 m.
# red-edge is there for vegetation density.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "B5", "B6", "B7", "B8A"]

SCENE_CLOUD_MAX = 70     # skip scenes that are almost entirely cloud before doing any work
CLD_PRB_THRESH = 50      # s2cloudless probability (%) above which a pixel is cloud
NIR_DRK_THRESH = 0.15    # nir reflectance below which a pixel could be shadow
CLD_PRJ_DIST = 1         # how far (km) to project clouds when looking for their shadows
BUFFER = 50              # dilate the final cloud+shadow mask by this many metres
CS_PLUS_THRESH = 0.60    # cloud score+ cs_cdf above which a pixel counts as clear


def month_range(month: str) -> tuple[str, str]:
    """'2021-07' -> ('2021-07-01', '2021-08-01')"""
    y, m = map(int, month.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-01", f"{ny:04d}-{nm:02d}-01"


def _s2(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_SR)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUD_MAX))
    )


def _with_s2cloudless(aoi, start, end) -> ee.ImageCollection:
    probs = ee.ImageCollection(S2_CLOUD_PROB).filterBounds(aoi).filterDate(start, end)
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=_s2(aoi, start, end),
        secondary=probs,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def _mask_s2cloudless(img: ee.Image) -> ee.Image:
    prob = ee.Image(img.get("s2cloudless")).select("probability")
    is_cloud = prob.gt(CLD_PRB_THRESH)

    not_water = img.select("SCL").neq(6)
    dark = img.select("B8").lt(NIR_DRK_THRESH * 1e4).multiply(not_water)
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    cloud_proj = (
        is_cloud.directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(crs=img.select(0).projection(), scale=100)
        .select("distance")
        .mask()
    )
    is_shadow = cloud_proj.multiply(dark)

    bad = (
        is_cloud.add(is_shadow).gt(0)
        .focalMin(2).focalMax(BUFFER * 2 / 20)
        .reproject(crs=img.select(0).projection(), scale=20)
    )
    return img.select(BANDS).updateMask(bad.Not())


def _mask_csplus(img: ee.Image) -> ee.Image:
    return img.select(BANDS).updateMask(img.select("cs_cdf").gte(CS_PLUS_THRESH))


def _with_csplus(aoi, start, end) -> ee.ImageCollection:
    # linkCollection adds the matching cloud score+ band onto each s2 scene
    return _s2(aoi, start, end).linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS), ["cs_cdf"])


def scene_counts(aoi: ee.Geometry, month: str) -> dict:
    """how many scenes are there, and how many have a matching s2cloudless image."""
    start, end = month_range(month)
    return {
        "scenes": _s2(aoi, start, end).size().getInfo(),
        "with_s2cloudless": _with_s2cloudless(aoi, start, end).size().getInfo(),
    }


def monthly_composite(aoi: ee.Geometry, month: str, method: str = "s2cloudless") -> ee.Image:
    """median of cloud-masked scenes in `month`. returns BANDS as uint16 reflectance x1e4
    (0 = no clear observation) plus a `clear_count` band: how many clear looks each pixel got."""
    start, end = month_range(month)
    if method == "s2cloudless":
        col = _with_s2cloudless(aoi, start, end).map(_mask_s2cloudless)
    elif method == "csplus":
        col = _with_csplus(aoi, start, end).map(_mask_csplus)
    else:
        raise ValueError(f"unknown cloud mask method: {method}")

    composite = col.median().select(BANDS).unmask(0).toUint16()
    clear_count = col.select("B4").count().unmask(0).toUint16().rename("clear_count")
    return composite.addBands(clear_count).set({"month": month, "cloud_mask": method})
