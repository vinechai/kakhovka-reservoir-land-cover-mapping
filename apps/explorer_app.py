# explorer: phase 1 vs cnn maps for any month, over the sentinel-2 image of that same month,
# with km2 stats. uses data/ if it's there, else the small copy in explorer_data/ (streamlit cloud).
#
# run: streamlit run apps/explorer_app.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium

from src.palette import COLORS, LABELS
from src.config import ALL_MONTHS, AREAS, COLLAPSE_DATE, DATA_DIR, ROOT
from src.labeling.imagery import ESRI_TILES

st.set_page_config(page_title="Kakhovka explorer", layout="wide")

METHODS = {"baseline": "Phase 1 (index rules)", "cnn": "CNN"}
CODE_COLORS = {1: COLORS["water"], 2: COLORS["bare"], 3: COLORS["sparse_veg"],
               4: COLORS["snow_ice"], 5: COLORS["dense_veg"]}
CLASS_KEYS = ["water", "bare", "sparse_veg", "dense_veg", "snow_ice"]


def maps_dir(area: str) -> Path:
    """full local data if it's there (your machine), else the slim copy tracked in git
    (streamlit cloud; made by scripts/export_explorer_data.py)."""
    local = DATA_DIR / area
    return local if (local / "cnn").exists() else ROOT / "explorer_data" / area


def hex_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


@st.cache_data(show_spinner=False)
def overlay(area: str, method: str, month: str, bed_only: bool):
    """rgba image in lat/lon + its bounds [[south, west], [north, east]]."""
    d = maps_dir(area)
    with rasterio.open(d / method / f"class_{month}.tif") as src:
        cls = src.read(1)
        if bed_only:
            with rasterio.open(d / "regions.tif") as reg:
                cls = np.where(reg.read(1) == 1, cls, 0)
        tr, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height,
                                               *src.bounds)
        out = np.zeros((h, w), dtype="uint8")
        reproject(cls, out, src_transform=src.transform, src_crs=src.crs, dst_transform=tr,
                  dst_crs="EPSG:4326", resampling=Resampling.nearest)
    rgba = np.zeros((h, w, 4), dtype="uint8")
    for code, col in CODE_COLORS.items():
        rgba[out == code] = (*hex_rgb(col), 200)
    west, north = tr.c, tr.f
    east, south = west + w * tr.a, north + h * tr.e
    return rgba, [[south, west], [north, east]]


@st.cache_data(show_spinner=False)
def month_image(area: str, month: str):
    """true colour sentinel-2 image of the month as rgba (edges from the reprojection made
    transparent) + its bounds. made by scripts/export_explorer_data.py."""
    d = ROOT / "explorer_data" / area / "rgb"
    rgb = np.asarray(Image.open(d / f"rgb_{month}.jpg"))
    empty = rgb.sum(2) < 15
    for _ in range(2):  # widen by 2 px so jpeg speckles along the edge disappear too
        empty = empty | np.roll(empty, 1, 0) | np.roll(empty, -1, 0) | np.roll(empty, 1, 1) | np.roll(empty, -1, 1)
    alpha = np.where(empty, 0, 255).astype("uint8")
    return np.dstack([rgb, alpha]), json.loads((d / "bounds.json").read_text())


@st.cache_data
def tables(area: str) -> dict[str, pd.DataFrame]:
    out = {}
    for m in METHODS:
        p = ROOT / "outputs" / "tables" / f"{m}_{area}.csv"
        if p.exists():
            out[m] = pd.read_csv(p, dtype={"month": str})
    return out


area = st.sidebar.selectbox("Area", list(AREAS))
show_classes = st.sidebar.checkbox("Show classes", value=True)
bed_only = st.sidebar.checkbox("Only the old reservoir bed", value=True)
opacity = st.sidebar.slider("Class opacity", 0.0, 1.0, 0.6, 0.05)
tabs = tables(area)
methods = [m for m in METHODS if m in tabs and (maps_dir(area) / m).exists()]

st.title("Kakhovka reservoir bed, month by month")
start = st.query_params.get("month", "2026-08")  # e.g. ...?month=2021-07 opens that month
month = st.select_slider("Month", options=ALL_MONTHS,
                         value=start if start in ALL_MONTHS else "2026-08")
tag = "before the collapse" if month < COLLAPSE_DATE[:7] else "after the collapse"
st.caption(f"{month}, {tag}. Dam destroyed 6 June 2023. Background inside the study area: "
           f"the Sentinel-2 image of {month}. Turn off the classes in the sidebar to see it alone.")

legend = " ".join(f"<span style='background:{CODE_COLORS[c]};padding:2px 8px;border-radius:3px;"
                  f"color:white;margin-right:6px'>{LABELS[k]}</span>"
                  for c, k in zip([1, 2, 3, 5, 4], ["water", "bare", "sparse_veg", "dense_veg", "snow_ice"]))
st.markdown(legend, unsafe_allow_html=True)

cols = st.columns(len(methods))
for col, m in zip(cols, methods):
    with col:
        st.subheader(METHODS[m])
        rgba, bounds = overlay(area, m, month, bed_only)
        centre = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        fmap = folium.Map(location=centre, zoom_start=11, tiles=None)
        fmap.fit_bounds(bounds)  # open showing the whole area, whatever the window size
        # plain map outside the study area. esri's photos are a fixed mosaic from mixed years
        # (older ones when zoomed out), so they're only an optional layer
        folium.TileLayer("OpenStreetMap", name="Map").add_to(fmap)
        folium.TileLayer(ESRI_TILES, attr="Esri World Imagery", max_zoom=19,
                         name="Esri photos (fixed, mixed dates)").add_to(fmap)
        img, img_bounds = month_image(area, month)
        folium.raster_layers.ImageOverlay(img, bounds=img_bounds, mercator_project=True,
                                          name=f"Sentinel-2, {month}").add_to(fmap)
        if show_classes:
            folium.raster_layers.ImageOverlay(rgba, bounds=bounds, opacity=opacity,
                                              mercator_project=True, name="Classes").add_to(fmap)
        folium.LayerControl(collapsed=True).add_to(fmap)
        st_folium(fmap, height=480, use_container_width=True, returned_objects=[],
                  key=f"map_{m}_{month}_{bed_only}_{show_classes}_{opacity}")
        row = tabs[m][(tabs[m].month == month) & (tabs[m].region == "reservoir_bed")].iloc[0]
        st.dataframe(pd.DataFrame({"km²": [row[f"{k}_km2"] for k in CLASS_KEYS] + [row.nodata_km2]},
                                  index=[LABELS[k] for k in CLASS_KEYS] + ["No clear view"]).round(1),
                     width="stretch")
        if not row.usable:
            st.warning("This month is mostly snow or cloud; its land-cover numbers aren't meaningful.")

st.subheader("Reservoir bed through time (usable months)")
series_class = st.radio("Class", CLASS_KEYS[:4], horizontal=True, format_func=lambda k: LABELS[k])
chart = pd.DataFrame({METHODS[m]: tabs[m][(tabs[m].region == "reservoir_bed") & tabs[m].usable]
                      .set_index("month")[f"{series_class}_km2"] for m in methods})
st.line_chart(chart, y_label="km²")
