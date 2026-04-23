"""
Interactive World Map
======================
Creates a choropleth map where:
  - Color intensity = mean annual snowfall (white → deep blue)
  - Hover tooltip  = country name, snowfall, medals breakdown

Compatible with geopandas >= 1.0 (no deprecated datasets).
Downloads world borders shapefile on first run, caches locally.

Output:
  data/figures/snowfall_medals_map.html  ← open in any browser

Usage:
  pip install folium geopandas requests
  python visualise.py
"""

from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_DIR  = Path("data/clean")
FIG_DIR    = Path("data/figures")
MASTER     = CLEAN_DIR / "master.csv"
OUT_HTML   = FIG_DIR / "snowfall_medals_map.html"

# Cached shapefile location
SHAPE_DIR  = Path("data/naturalearth")
SHAPE_FILE = SHAPE_DIR / "ne_110m_admin_0_countries.shp"

# Natural Earth 110m countries — reliable, stable URL
SHAPEFILE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)
SHAPEFILE_URL_BACKUP = (
    "https://github.com/nvkelso/natural-earth-vector/raw/master/zips/"
    "ne_110m_admin_0_countries.zip"
)

# ---------------------------------------------------------------------------
# Download shapefile if not cached
# ---------------------------------------------------------------------------

def ensure_shapefile() -> Path:
    if SHAPE_FILE.exists():
        print(f"[MAP] Using cached shapefile: {SHAPE_FILE}")
        return SHAPE_FILE

    SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    print("[MAP] Downloading Natural Earth world borders (first run only)…")

    for url in [SHAPEFILE_URL, SHAPEFILE_URL_BACKUP]:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(SHAPE_DIR)
            print(f"[MAP] Saved to {SHAPE_DIR}/")
            return SHAPE_FILE
        except Exception as e:
            print(f"[MAP] Failed ({url}): {e} — trying backup…")

    raise RuntimeError(
        "Could not download shapefile. "
        "Please manually download ne_110m_admin_0_countries.zip from "
        "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/ "
        f"and extract it into {SHAPE_DIR}/"
    )

# ---------------------------------------------------------------------------
# Load + aggregate per country
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER)

    df = df.rename(columns={
        "snowfall_km3":           "snowfall_total",
        "snowfall_mm":            "snowfall_mean_gridcell",
        "gdp_usd":                "gdp",
        "population_total":       "population",
        "iso3":                   "country",
    }, errors="ignore")

    agg = df.groupby("country").agg(
        country_name        =("team_name",              "first"),
        total_medals        =("total_medals",           "sum"),
        gold                =("gold",                   "sum"),
        silver              =("silver",                 "sum"),
        bronze              =("bronze",                 "sum"),
        editions_attended   =("year",                   "nunique"),
        snowfall_mm_mean    =("snowfall_mean_gridcell", "mean"),
        snowfall_total_mean =("snowfall_total",         "mean"),
    ).reset_index()

    agg["country_name"]        = agg["country_name"].fillna(agg["country"])
    agg["snowfall_mm_mean"]    = agg["snowfall_mm_mean"].fillna(0).round(2)
    agg["snowfall_total_mean"] = agg["snowfall_total_mean"].fillna(0).round(2)

    print(f"[LOAD] {len(agg)} countries aggregated")
    return agg

# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------

def build_map(agg: pd.DataFrame) -> folium.Map:
    shp = ensure_shapefile()
    world = gpd.read_file(shp)

    # Natural Earth uses ADM0_A3 or ISO_A3 — pick whichever is present
    iso_col = None
    for candidate in ["ISO_A3", "ADM0_A3", "SOV_A3"]:
        if candidate in world.columns:
            iso_col = candidate
            break
    if iso_col is None:
        raise RuntimeError(f"No ISO column found. Available: {list(world.columns)}")
    print(f"[MAP] Using ISO column: {iso_col}")

    # Normalise to uppercase stripped string
    world[iso_col] = world[iso_col].astype(str).str.upper().str.strip()

    # Merge our data onto world
    world = world.merge(
        agg.rename(columns={"country": iso_col}),
        on=iso_col,
        how="left",
    )

    world["snowfall_mm_mean"]    = world["snowfall_mm_mean"].fillna(0)
    world["snowfall_total_mean"] = world["snowfall_total_mean"].fillna(0)
    world["total_medals"]        = world["total_medals"].fillna(0).astype(int)
    world["gold"]                = world["gold"].fillna(0).astype(int)
    world["silver"]              = world["silver"].fillna(0).astype(int)
    world["bronze"]              = world["bronze"].fillna(0).astype(int)
    world["editions_attended"]   = world["editions_attended"].fillna(0).astype(int)
    world["country_name"]        = world["country_name"].fillna(world["NAME"].fillna(world[iso_col]))

    # Convert to GeoJSON
    geojson = world.__geo_interface__

    # ── Base map ──
    m = folium.Map(
        location=[30, 10],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    # ── Choropleth (snowfall → blue) ──
    folium.Choropleth(
        geo_data=geojson,
        data=world[[iso_col, "snowfall_mm_mean"]],
        columns=[iso_col, "snowfall_mm_mean"],
        key_on=f"feature.properties.{iso_col}",
        fill_color="Blues",
        fill_opacity=0.75,
        line_opacity=0.25,
        line_color="white",
        nan_fill_color="#eeeeee",
        nan_fill_opacity=0.4,
        legend_name="Mean annual snowfall depth (mm water equivalent per Olympic edition)",
        name="Snowfall intensity",
    ).add_to(m)

    # ── Tooltip overlay ──
    folium.GeoJson(
        geojson,
        style_function=lambda _: {
            "fillColor":   "transparent",
            "color":       "#aaa",
            "weight":      0.3,
            "fillOpacity": 0,
        },
        highlight_function=lambda _: {
            "fillColor":   "#ffdd44",
            "color":       "#333",
            "weight":      2,
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "country_name",
                "snowfall_mm_mean",
                "snowfall_total_mean",
                "total_medals",
                "gold",
                "silver",
                "bronze",
                "editions_attended",
            ],
            aliases=[
                "Country",
                "Avg snowfall depth (mm / edition)",
                "Avg snowfall volume (km³ / edition)",
                "Total medals",
                "Gold",
                "Silver",
                "Bronze",
                "Editions attended",
            ],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color: white;"
                "color: #222;"
                "font-family: sans-serif;"
                "font-size: 13px;"
                "padding: 10px 14px;"
                "border-radius: 6px;"
                "border: 1px solid #ccc;"
                "box-shadow: 2px 2px 8px rgba(0,0,0,0.15);"
            ),
        ),
        name="Country details",
    ).add_to(m)

    # ── Title ──
    m.get_root().html.add_child(folium.Element("""
    <div style="
        position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
        z-index: 9999; background: white; padding: 10px 22px;
        border-radius: 8px; border: 1px solid #ddd;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.12);
        font-family: sans-serif; font-size: 15px; font-weight: 600;
        color: #222; pointer-events: none; white-space: nowrap;
    ">
        Winter Olympics &mdash; Snowfall &amp; Medal Performance
        <span style="font-weight: 400; font-size: 12px; color: #888; margin-left: 10px;">
            hover over a country
        </span>
    </div>
    """))

    folium.LayerControl().add_to(m)
    return m

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg = load()
    m   = build_map(agg)
    m.save(str(OUT_HTML))
    print(f"\n✓ Map saved → {OUT_HTML}")
    print("  Double-click the HTML file to open it in your browser.")

if __name__ == "__main__":
    main()