# file: app.py
"""
Interactive Streamlit map for Boeing plane crashes by country:
- 0 crashes: explicit neutral color (does not blend with 1).
- 1+ crashes: saturated gradient starting at 1 (strong contrast).
- Hover tooltips, zoom/pan, configurable palette, optional log scale.

Run:
    pip install streamlit geopandas country_converter pydeck shapely matplotlib
    streamlit run app.py
"""

import json
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Boeing Crashes – Interactive Map", layout="wide")
st.title("Boeing Plane Crashes by Country (Interactive)")

with st.sidebar:
    st.header("Controls")
    csv_path = st.text_input("Crash CSV path", value="BA_Crash.csv")
    country_col = st.text_input("Country column in CSV", value="country")

    shp_path = st.text_input(
        "World shapefile (.shp) path",
        value="/Users/luke/Documents/Master of Finance/ISOM_Data_Viz/Assignment 3/shape_files/ne_110m_admin_0_countries.shp",
    )

    palette = st.selectbox(
        "Palette for ≥1",
        ["OrRd", "Blues", "YlOrRd", "Greens", "Purples", "Viridis"],
        index=0,
    )
    trunc_low = st.slider("Saturate low end (avoid pale 1s)", 0.0, 0.6, 0.35, 0.05)
    trunc_high = st.slider("Truncate high end", 0.8, 1.0, 0.95, 0.01)
    zero_color = st.color_picker("Color for 0 crashes", value="#F2F2F2")
    edge_color = st.color_picker("Border color", value="#C8C8C8")

    use_log = st.checkbox("Log scale for ≥1 (helps when a few countries dominate)", value=False)

# ---------------------------- Helpers ----------------------------
def truncated_cmap(name: str, low: float, high: float, n: int = 256) -> LinearSegmentedColormap:
    """Why: ensures value=1 is visibly saturated by removing pale tail."""
    base = mpl.cm.get_cmap(name, n)
    idx = np.linspace(int(low * (n - 1)), int(high * (n - 1)), n - 1 - int(low * (n - 1)), dtype=int)
    return LinearSegmentedColormap.from_list(f"{name}_trunc", base(idx), N=n)

def hex_to_rgb255(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b)

@st.cache_data(show_spinner=False)
def load_counts(csv_file: str, country_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    counts = (
        df[country_col_name].astype(str).str.strip()
        .value_counts()
        .rename_axis("country")
        .reset_index(name="crashes")
    )
    return counts

@st.cache_data(show_spinner=False)
def to_iso3(df_counts: pd.DataFrame) -> pd.DataFrame:
    import country_converter as coco  # local import keeps cache stable if lib missing
    cc = coco.CountryConverter()
    df_counts = df_counts.copy()
    df_counts["iso3"] = cc.convert(df_counts["country"], to="ISO3")
    return df_counts[df_counts["iso3"] != "not found"]

@st.cache_data(show_spinner=False)
def load_world(shp_file: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(shp_file)
    possible_iso3_cols = ["adm0_a3", "iso_a3", "ISO_A3", "ADM0_A3"]
    iso3_col = next((c for c in possible_iso3_cols if c in world.columns), None)
    if iso3_col is None:
        raise ValueError(f"Could not find ISO3 column in shape. Looked for: {possible_iso3_cols}")
    world = world.rename(columns={iso3_col: "iso3"})
    world["iso3"] = world["iso3"].astype(str).str.upper()
    if world.crs is None:
        # Most Natural Earth layers are already WGS84; you can set if known.
        world.set_crs(4326, inplace=True)  # assumes WGS84 lon/lat
    else:
        world = world.to_crs(4326)
    return world

def build_norm(vmin_one: float, vmax_val: float, use_log_scale: bool):
    if use_log_scale:
        # Why: avoids flattening when a few countries have very high counts.
        return LogNorm(vmin=max(1, vmin_one), vmax=max(1.001, vmax_val))
    return Normalize(vmin=max(1, vmin_one), vmax=max(1.001, vmax_val))

def compute_colors(series: pd.Series, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, zero_hex: str) -> np.ndarray:
    """Returns Nx3 uint8 RGB array for each value in `series`."""
    rgb_zeros = np.array(hex_to_rgb255(zero_hex), dtype=np.uint8)
    values = series.to_numpy()
    out = np.empty((len(values), 3), dtype=np.uint8)
    mask_zero = values <= 0
    out[mask_zero] = rgb_zeros
    if (~mask_zero).any():
        vals = values[~mask_zero]
        mapped = cmap(norm(vals))[:, :3] * 255.0
        out[~mask_zero] = mapped.astype(np.uint8)
    return out

def to_view_state(bounds: Tuple[float, float, float, float]):
    """Return pydeck-like initial view state (lon, lat, zoom)."""
    minx, miny, maxx, maxy = bounds
    lat = (miny + maxy) / 2
    lon = (minx + maxx) / 2
    # Heuristic zoom based on world bounds (~360 deg); tune for global fit
    lon_span = max(1e-6, maxx - minx)
    zoom = 1 if lon_span > 150 else 2 if lon_span > 60 else 3
    return lon, lat, zoom

# ---------------------------- Data ----------------------------
try:
    counts = load_counts(csv_path, country_col)
    counts_iso3 = to_iso3(counts)
    world = load_world(shp_path)

    merged = world.merge(counts_iso3[["iso3", "crashes"]], on="iso3", how="left")
    merged["crashes"] = merged["crashes"].fillna(0).astype(float)

    vmax = float(max(1.0, merged["crashes"].max()))
    cmap = truncated_cmap(palette, low=trunc_low, high=trunc_high)
    # set 'under' for zeros
    try:
        cmap = cmap.with_extremes(under=zero_color)
    except AttributeError:
        cmap.set_under(zero_color)  # for older Matplotlib

    norm = build_norm(1.0, vmax, use_log)
    colors = compute_colors(merged["crashes"], cmap, norm, zero_color)
    merged = merged.assign(fill_r=colors[:, 0], fill_g=colors[:, 1], fill_b=colors[:, 2])

    # Tooltip fields
    merged["name_label"] = merged.get("NAME_EN", merged.get("NAME_LONG", merged.get("name", merged.get("NAME", "Unknown"))))
    merged["crashes_int"] = merged["crashes"].astype(int)

    # GeoJSON for pydeck
    geojson = json.loads(merged.to_json())

    # ---------------------------- Map ----------------------------
    import pydeck as pdk  # imported late to keep startup lean

    lon, lat, zoom = to_view_state(merged.total_bounds)
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b]",
        get_line_color=list(hex_to_rgb255(edge_color)),
        line_width_min_pixels=0.5,
    )

    # Render
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,  # transparent background; switch to "mapbox://styles/mapbox/light-v9" if you want basemap
        tooltip={"text": "{name_label}\nCrashes: {crashes_int}"},
    )

    st.pydeck_chart(r, use_container_width=True)

    # ---------------------------- Legend ----------------------------
    # Minimal colorbar for ≥1 and a discrete swatch for 0
    col1, col2 = st.columns([3, 1])
    with col1:
        # Matplotlib scalar mappable colorbar image
        fig, ax = plt.subplots(figsize=(6, 0.45))
        fig.subplots_adjust(bottom=0.5)
        cb = mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            orientation="horizontal",
            label="Boeing crashes (≥1)",
        )
        st.pyplot(fig, use_container_width=True, clear_figure=True)
    with col2:
        r0, g0, b0 = hex_to_rgb255(zero_color)
        st.markdown("**0 crashes**")
        swatch = f'<div style="width:100%;height:18px;background:rgb({r0},{g0},{b0});border:1px solid #999;"></div>'
        st.markdown(swatch, unsafe_allow_html=True)

    # ---------------------------- Download ----------------------------
    with st.expander("Download merged GeoJSON"):
        st.download_button(
            "Download GeoJSON",
            data=json.dumps(geojson).encode("utf-8"),
            file_name="boeing_crashes_merged.geojson",
            mime="application/geo+json",
        )

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
