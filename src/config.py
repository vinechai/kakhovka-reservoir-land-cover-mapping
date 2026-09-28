# shared settings: earth engine project, test area, 10 m utm grid and the months we use.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# same cloud project as the building-damage work
GEE_PROJECT = "buildings-damage"

# dnipro at ~34E sits in utm zone 36N
CRS = "EPSG:32636"
SCALE = 10  # metres, native res of the s2 10 m bands

COLLAPSE_DATE = "2023-06-06"


def _month_span(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# pre-collapse: a few summers as the "normal full lake" baseline, plus the months leading up
# to the collapse (the level was drawn down over winter 2022/23, then refilled by may 2023)
PRE_COLLAPSE_MONTHS = [
    "2019-07", "2020-07", "2021-07", "2021-08", "2022-07",
    "2022-10", "2023-02", "2023-04", "2023-05",
]
# post-collapse: every month from the collapse to the last full month.
# june 2023 mixes 5 days of full lake with 25 days after the dam went, the median mostly
# shows the post-collapse state.
POST_COLLAPSE_MONTHS = _month_span("2023-06", "2026-08")
ALL_MONTHS = PRE_COLLAPSE_MONTHS + POST_COLLAPSE_MONTHS


@dataclass(frozen=True)
class Area:
    name: str
    description: str
    # lon/lat box: west, south, east, north
    bbox: tuple[float, float, float, float]


AREAS = {
    # first-pass test area: the core of the old velykyi luh floodplain, the widest part of
    # the reservoir between nikopol (north bank) and kamianka-dniprovska (south bank).
    # ~23 x 20 km, a bit under 500 km2 of which most was open water before the collapse.
    "velykyi_luh": Area(
        name="velykyi_luh",
        description="Velykyi Luh core, between Nikopol and Kamianka-Dniprovska",
        bbox=(34.15, 47.40, 34.45, 47.58),
    ),
}


# lakes inside the jrc outline with their own dam, found by one (lon, lat) point inside each.
# kept as their own region: on maps and in training, but not in the reservoir stats.
SEPARATE_WATER_BODIES = {
    "velykyi_luh": {
        # dammed floodplain lake south of kamianka-dniprovska, ~15.6 km2, level was held by
        # a pumping station, didn't drain with the reservoir
        "bilozerskyi_lyman": (34.39, 47.46),
    },
}


def area_dir(area: str) -> Path:
    d = DATA_DIR / area
    d.mkdir(parents=True, exist_ok=True)
    return d
