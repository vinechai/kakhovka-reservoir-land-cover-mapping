# hand labelling app: sentinel-2 cut-outs, esri high-res map with the pixel outlined and the
# point's ndvi / mndwi over time. the phase 1 class is never shown (blind labelling).
#
# run: streamlit run apps/labeling_app.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rasterio.warp import transform as warp_transform
from streamlit_folium import st_folium

from src.config import ALL_MONTHS, COLLAPSE_DATE, CRS, SCALE, area_dir
from src.labeling.imagery import ESRI_TILES, chips, index_series
from src.labeling.store import (
    LABEL_CLASSES, LABELLING_RULES, list_point_sets, load_labels, load_points, save_label,
)

st.set_page_config(page_title="Kakhovka labelling", layout="wide")

MARK = np.array([1.0, 0.85, 0.0])  # yellow outline for the labelled pixel


# ---------- helpers ----------

@st.cache_data
def cached_series(area: str, row: int, col: int) -> pd.DataFrame:
    return pd.DataFrame(index_series(area_dir(area) / "s2", ALL_MONTHS, row, col))


@st.cache_data
def cached_chips(area: str, month: str, row: int, col: int):
    return chips(area_dir(area) / "s2" / f"s2_{month}.tif", row, col, half=32)


def enlarge_and_mark(img: np.ndarray, k: int = 5) -> np.ndarray:
    """blow each pixel up to k x k (so it's visible) and outline the centre pixel."""
    big = np.kron(img, np.ones((k, k, 1)))
    c = img.shape[0] // 2
    r0, r1 = c * k - 1, (c + 1) * k
    big[r0, r0:r1 + 1] = big[r1, r0:r1 + 1] = MARK
    big[r0:r1 + 1, r0] = big[r0:r1 + 1, r1] = MARK
    return big


def pixel_square(x: float, y: float, half: float) -> list[list[float]]:
    """lat/lon corners of a square around a utm point (for drawing on the map)."""
    xs = [x - half, x + half, x + half, x - half]
    ys = [y - half, y - half, y + half, y + half]
    lon, lat = warp_transform(CRS, "EPSG:4326", xs, ys)
    return [[la, lo] for lo, la in zip(lon, lat)]


def series_chart(df: pd.DataFrame, target: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    dates = pd.to_datetime(df.month) + pd.Timedelta(days=14)
    ax.plot(dates, df.ndvi, color="#008300", marker="o", ms=3, lw=1.5, label="NDVI (green plants)")
    ax.plot(dates, df.mndwi, color="#2a78d6", marker="o", ms=3, lw=1.5, label="MNDWI (water)")
    ax.axhline(0.3, color="#008300", lw=0.7, ls=":")
    ax.axhline(0.0, color="#2a78d6", lw=0.7, ls=":")
    ax.axvline(pd.Timestamp(COLLAPSE_DATE), color="#52514e", lw=0.8, ls="--")
    ax.axvline(pd.Timestamp(target + "-15"), color="#eda100", lw=6, alpha=0.3)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ---------- sidebar: which set, who, progress ----------

sets = list_point_sets()
if not sets:
    st.error("no point sets yet, run: python scripts/sample_points.py --set pilot --n 30")
    st.stop()

with st.sidebar:
    st.header("Labelling")
    # training labels are needed first (phase 2 waits on them), so open on that set
    point_set = st.selectbox("Point set", sets, index=sets.index("train") if "train" in sets else 0)
    labeller = st.text_input("Your name", value=st.session_state.get("labeller", ""))
    st.session_state.labeller = labeller

points = load_points(point_set)
labels = load_labels(point_set)
done = set(labels.point_id)

if "idx" not in st.session_state or st.session_state.get("set") != point_set:
    st.session_state.set = point_set
    todo = [i for i in points.point_id if i not in done]
    st.session_state.idx = todo[0] if todo else 0

with st.sidebar:
    st.progress(len(done) / len(points), text=f"{len(done)} / {len(points)} labelled")
    if len(labels):
        st.caption("Labels so far")
        st.dataframe(labels.label.value_counts(), width="stretch")
    with st.expander("Extra hints"):
        st.markdown(
            "In false colour, dense plants are bright red and sparse ones pinkish over grey "
            "ground. In the chart, water keeps MNDWI (blue line) above 0. The high-res map is "
            "the main evidence; the rest is support.")

# ---------- current point ----------

p = points.set_index("point_id").loc[st.session_state.idx]
pid = int(st.session_state.idx)
target = p.target_month

nav = st.columns([1, 1, 1, 4])
if nav[0].button("◀ Previous", disabled=pid == 0):
    st.session_state.idx = pid - 1
    st.rerun()
if nav[1].button("Next ▶", disabled=pid == len(points) - 1):
    st.session_state.idx = pid + 1
    st.rerun()
if nav[2].button("Next unlabelled"):
    todo = [i for i in points.point_id if i not in done]
    if todo:
        st.session_state.idx = next((i for i in todo if i > pid), todo[0])
        st.rerun()
nav[3].markdown(f"### Point {pid}  ·  label for **{target}**  "
                f"·  high-res photo from {p.imagery_date if isinstance(p.imagery_date, str) else 'unknown date'}")

left, mid, right = st.columns([1, 1.4, 1])

with left:
    month = st.selectbox("Sentinel-2 month", ALL_MONTHS, index=ALL_MONTHS.index(target),
                         key=f"month_{pid}")
    true, false = cached_chips(p.area, month, int(p.row), int(p.col))
    st.image(enlarge_and_mark(true), caption="True colour (640 m across, yellow = the pixel)",
             width="stretch")
    st.image(enlarge_and_mark(false), caption="False colour (plants = red)",
             width="stretch")

with mid:
    m = folium.Map(location=[p.lat, p.lon], zoom_start=18, max_zoom=19, tiles=None)
    folium.TileLayer(ESRI_TILES, attr="Esri World Imagery", name="Esri", max_zoom=19,
                     max_native_zoom=19).add_to(m)
    folium.Polygon(pixel_square(p.x, p.y, SCALE / 2), color="#ffd900", weight=2,
                   fill=False, tooltip="the 10 m pixel").add_to(m)
    folium.Polygon(pixel_square(p.x, p.y, SCALE * 1.5), color="#ffd900", weight=1,
                   dash_array="4", fill=False, tooltip="3 x 3 pixels").add_to(m)
    st_folium(m, height=520, use_container_width=True, returned_objects=[], key=f"map_{pid}")
    st.caption("Esri high-res imagery. Scroll to zoom out for context.")

with right:
    series = cached_series(p.area, int(p.row), int(p.col))
    st.pyplot(series_chart(series, target), width="stretch")
    st.caption("This point through time. Shaded = target month; dashed = dam collapse; "
               "dotted = Phase 1 cut-offs.")

    existing = labels[labels.point_id == pid]
    prev = existing.iloc[0] if len(existing) else None
    keys = list(LABEL_CLASSES)
    with st.expander("Labelling rules", expanded=True):
        st.markdown(LABELLING_RULES)
    with st.form(f"label_{pid}"):
        choice = st.radio("What is at the yellow pixel in " + target + "?", keys,
                          index=keys.index(prev.label)
                          if prev is not None and prev.label in keys else None,
                          format_func=lambda k: LABEL_CLASSES[k])
        trees = st.checkbox("Clear tree crowns with shadows visible",
                            value=False if prev is None else bool(prev.trees_visible))
        confident = st.checkbox("I'm confident", value=True if prev is None else bool(prev.confident))
        note = st.text_input("Note (optional)",
                             value="" if prev is None or pd.isna(prev.note) else prev.note)
        if st.form_submit_button("Save & next", type="primary"):
            if not labeller:
                st.error("Enter your name in the sidebar first.")
            elif choice is None:
                st.error("Pick a class.")
            else:
                save_label(point_set, pid, choice, confident, note, target,
                           p.imagery_date if isinstance(p.imagery_date, str) else "", labeller,
                           trees_visible=trees)
                todo = [i for i in points.point_id if i not in done | {pid}]
                st.session_state.idx = next((i for i in todo if i > pid), todo[0] if todo else pid)
                st.rerun()
    if prev is not None:
        st.caption(f"Already labelled: **{prev.label}**. Saving again replaces it.")
