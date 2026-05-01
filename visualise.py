"""
Interactive World Map — Winter Olympics Snowfall & Medals
==========================================================
Creates a choropleth world map where:
  - Colour intensity = total snowfall volume across all editions attended (white → deep blue)
  - Hover tooltip    = country name, snowfall, medals breakdown, GDP, population, medals per million


Column names used here match master.csv exactly as produced by clean.py:
  country               — ISO3 code (e.g. NOR, DEU) — used to join the shapefile
  noc_code              — NOC Olympic code (e.g. GER, NED) — shown in tooltip
  team_name             — readable country name for tooltip display
  total_medals          — sum of all medals across editions
  gold / silver / bronze
  snowfall_mean_gridcell — mean mm depth per ERA5 grid cell (size-normalised)
  snowfall_total         — total snowfall volume in km³
  gdp                   — total GDP in current USD (NOT per capita; see below)
  population             — total population
  medals_per_million    — total_medals / population * 1e6 (size-normalised outcome)
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

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"
OUT_HTML  = FIG_DIR / "snowfall_medals_map.html"

# Cached shapefile — downloaded once and reused on subsequent runs.
SHAPE_DIR  = Path("data/naturalearth")
SHAPE_FILE = SHAPE_DIR / "ne_110m_admin_0_countries.shp"

# Natural Earth 110m country polygons.  Primary URL + GitHub backup.
SHAPEFILE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)
SHAPEFILE_URL_BACKUP = (
    "https://github.com/nvkelso/natural-earth-vector/raw/master/zips/"
    "ne_110m_admin_0_countries.zip"
)

# ---------------------------------------------------------------------------
# Download shapefile if not already cached
# ---------------------------------------------------------------------------

def ensure_shapefile() -> Path:
    """
    Return the path to the Natural Earth shapefile, downloading it if needed.

    The shapefile provides country polygon geometries for the choropleth map.
    We use Natural Earth 110m resolution — lower resolution but fast enough
    to render smoothly in a browser.

    The file is cached in data/naturalearth/ so subsequent runs skip the
    download entirely.
    """
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
# Load and aggregate per country
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    """
    Read master.csv and collapse from country-year rows to one row per country.

    master.csv uses the exact column names produced by clean.py — no renaming
    is needed here. The key columns and how they are aggregated:

      Medals (gold, silver, bronze, total_medals)
        → sum across all editions.  A country that competed 9 times accumulates
          medals from all 9 Games.

      Snowfall (snowfall_mean_gridcell)
        → mean across editions.  Climate is broadly stable over the ~30-year
          window, so the average depth is a good representative value.

      Snowfall (snowfall_total)
        → sum across editions.  Accumulates total snowfall volume over all
          Games attended; used as the choropleth colour variable.

      Economic / demographic (gdp, population)
        → mean across editions.  These change year to year; the average over
          all attended editions is the most neutral summary.

      medals_per_million
        → mean across editions.  It is already a ratio (medals per million
          people), so averaging is appropriate rather than re-computing from
          summed totals.

      editions_attended
        → count of distinct years.

    Two display-friendly columns are derived here:
      gdp_billions        = avg gdp / 1e9   (e.g. 412.3 B USD)
      population_millions = avg pop / 1e6   (e.g. 17.2 M)
    These avoid showing raw 12-digit numbers in the tooltip.
    """
    df = pd.read_csv(MASTER)

    agg = (
        df.groupby("country")
        .agg(
            # Readable name and reference code for tooltip
            country_name        = ("team_name",              "first"),
            noc_code            = ("noc_code",               "first"),
            # Medal tallies: accumulated across all editions attended
            total_medals        = ("total_medals",           "sum"),
            gold                = ("gold",                   "sum"),
            silver              = ("silver",                 "sum"),
            bronze              = ("bronze",                 "sum"),
            # Number of distinct Olympic Games attended
            editions_attended   = ("year",                   "nunique"),
            # Snowfall: mean depth averaged across editions (stable climate variable);
            # total volume summed across all editions attended.
            snowfall_mm_mean  = ("snowfall_mean_gridcell", "mean"),
            snowfall_total_sum = ("snowfall_total",        "sum"),
            # Total GDP (not per capita): averaged across editions.
            # Used to represent national economic capacity, consistent
            # with the main regression models in analyse.py.
            gdp_mean            = ("gdp",                    "mean"),
            # Population: averaged across editions
            population_mean     = ("population",             "mean"),
            # medals_per_million: averaged across editions (already a ratio)
            medals_per_million  = ("medals_per_million",     "mean"),
        )
        .reset_index()
    )

    # Countries with no snowfall data (e.g. tiny island nations not covered
    # by ERA5) are treated as zero-snowfall for choropleth colouring.
    agg["country_name"]        = agg["country_name"].fillna(agg["country"])
    agg["snowfall_mm_mean"]   = agg["snowfall_mm_mean"].fillna(0).round(2)
    agg["snowfall_total_sum"] = agg["snowfall_total_sum"].fillna(0).round(2)
    agg["medals_per_million"]  = agg["medals_per_million"].fillna(0).round(3)

    # Convert to human-readable units for tooltip display.
    # Raw GDP in USD can be 12 digits (e.g. 412,300,000,000) — converting to
    # billions keeps the tooltip readable without losing meaning.
    agg["gdp_billions"]        = (agg["gdp_mean"] / 1e9).round(1)
    agg["population_millions"] = (agg["population_mean"] / 1e6).round(2)

    print(f"[LOAD] {len(agg)} countries aggregated from master.csv")
    return agg

# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------
def build_map(agg: pd.DataFrame) -> folium.Map:
    """
    Build the interactive folium choropleth map.

    Architecture: two stacked layers
    ---------------------------------
    1. Choropleth layer — draws coloured country polygons.
       Folium's Choropleth class handles the colour encoding and legend
       efficiently, but does not support rich hover tooltips on its own.

    2. Transparent GeoJson overlay — carries the tooltip data.
       This layer sits on top of the choropleth with no visible fill or border.
       On hover, the country highlights in yellow and a styled tooltip appears.

    The join between our data and the shapefile uses ISO3 codes because:
      - Natural Earth uses ISO_A3 / ADM0_A3 (ISO3-like codes)
      - Our master.csv uses 'country' which is ISO3 (as set by clean.py)
      - NOC codes differ from ISO3 for many countries (e.g. GER vs DEU),
        so we join on ISO3 and only display the NOC code in the tooltip.

    Known Natural Earth data issue — ISO_A3 = '-99':
      Natural Earth uses '-99' as a sentinel in the ISO_A3 column for
      countries it considers disputed or that lack an obvious ISO code.
      In practice this incorrectly affects real countries like Norway and
      France.  We patch these using ADM0_A3 (always populated) as a
      fallback.  See the build_map body for the full fix.
    """
    shp   = ensure_shapefile()
    world = gpd.read_file(shp)

    # ── Build a clean ISO3 join key from the shapefile ────────────────────
    #
    # Natural Earth has a well-known data quality issue: ISO_A3 is set to
    # '-99' for several real countries, including Norway, France, Kosovo,
    # and a handful of others.  '-99' is Natural Earth's sentinel value
    # meaning "no ISO code assigned", but these countries DO have ISO codes —
    # the shapefile just omits them in the ISO_A3 column.
    #
    # Strategy: build a synthetic 'iso3' column that:
    #   1. Prefers ISO_A3_EH if present ('EH' = 'Except Holes' — a patched
    #      version of ISO_A3 that fixes most -99 cases, available in newer
    #      Natural Earth releases).
    #   2. Falls back to ISO_A3 when it is not '-99'.
    #   3. Patches any remaining '-99' values with ADM0_A3, which is always
    #      populated and uses the same ISO3 codes for all real countries.
    #
    # This means Norway (ISO_A3='-99', ADM0_A3='NOR') will correctly get
    # 'NOR' as its join key and match our master.csv data.

    # Step 1: pick the best available ISO source column
    if "ISO_A3_EH" in world.columns:
        # Newest Natural Earth releases include this pre-patched column
        base_col = "ISO_A3_EH"
    elif "ISO_A3" in world.columns:
        base_col = "ISO_A3"
    elif "ADM0_A3" in world.columns:
        base_col = "ADM0_A3"
    else:
        raise RuntimeError(
            f"No ISO column found in shapefile. Available: {list(world.columns)}"
        )
    print(f"[MAP] Base ISO column: {base_col}")

    # Step 2: normalise to uppercase stripped string
    world["iso3"] = world[base_col].astype(str).str.upper().str.strip()

    # Step 3: patch any remaining '-99' sentinels with ADM0_A3
    # ADM0_A3 is always a valid 3-letter code in Natural Earth
    if "ADM0_A3" in world.columns:
        bad_mask = world["iso3"].isin(["-99", "-1", ""])
        n_bad = bad_mask.sum()
        if n_bad > 0:
            world.loc[bad_mask, "iso3"] = (
                world.loc[bad_mask, "ADM0_A3"]
                .astype(str).str.upper().str.strip()
            )
            patched = world.loc[bad_mask, ["NAME", "iso3"]].values.tolist()
            print(f"[MAP] Patched {n_bad} rows with ADM0_A3 fallback:")
            for name, code in patched:
                print(f"       {name} → {code}")

    iso_col = "iso3"   # all subsequent joins and Choropleth keys use this column

    # ── Merge our aggregated data onto the world polygons ─────────────────
    # left join: every world polygon is kept; countries not in our dataset
    # will have NaN for all our columns (filled below).
    world = world.merge(
        agg.rename(columns={"country": iso_col}),
        on=iso_col,
        how="left",
    )

    # Fill missing values for countries not in our Olympic dataset.
    world["snowfall_mm_mean"]   = world["snowfall_mm_mean"].fillna(0)
    world["snowfall_total_sum"] = world["snowfall_total_sum"].fillna(0)
    world["total_medals"]        = world["total_medals"].fillna(0).astype(int)
    world["gold"]                = world["gold"].fillna(0).astype(int)
    world["silver"]              = world["silver"].fillna(0).astype(int)
    world["bronze"]              = world["bronze"].fillna(0).astype(int)
    world["editions_attended"]   = world["editions_attended"].fillna(0).astype(int)
    world["medals_per_million"]  = world["medals_per_million"].fillna(0).round(3)
    world["gdp_billions"]        = world["gdp_billions"].fillna(0).round(1)
    world["population_millions"] = world["population_millions"].fillna(0).round(2)
    world["noc_code"]            = world["noc_code"].fillna("—")
    # Use our team_name; fall back to the shapefile's NAME field, then ISO code.
    world["country_name"] = world["country_name"].fillna(
        world.get("NAME", world[iso_col])
    )

    # Convert to GeoJSON — the format folium requires for both layers.
    geojson = world.__geo_interface__

    # ── Base map ──────────────────────────────────────────────────────────
    # CartoDB positron: clean, light, low-contrast basemap.
    # Starting at zoom_start=2 shows the full world on load.
    m = folium.Map(
        location=[30, 10],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    # ── Layer 1: Choropleth (snowfall → blue colour ramp) ─────────────────
    # Log-scale transform: total snowfall is heavily right-skewed — Russia's
    # continental landmass produces values so large that a linear scale renders
    # almost every other country white.  log1p (log(1 + x)) compresses the
    # upper tail while mapping zero exactly to zero, giving meaningful colour
    # differentiation across the full range of countries.
    world["snowfall_log"] = np.log1p(world["snowfall_total_sum"])

    # Use quantile bins so colour steps are spread evenly across countries
    # rather than being dominated by the raw value range.
    nonzero = world.loc[world["snowfall_log"] > 0, "snowfall_log"]
    # 8 quantile breakpoints across the full log range gives a smooth gradient.
    # Using the global min/max (not just nonzero) as outer edges ensures every
    # value — including zeros — falls within the bin range.
    q_breaks = list(nonzero.quantile([0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92]))
    bins = (
        [float(world["snowfall_log"].min())]
        + [round(b, 6) for b in q_breaks]
        + [float(world["snowfall_log"].max()) + 1e-6]
    )
    bins = sorted(set(bins))  # deduplicate if any quantiles collapse

    # fill_color="Blues" is a Colorbrewer sequential palette: white (low)
    # to deep blue (high).
    folium.Choropleth(
        geo_data=geojson,
        data=world[[iso_col, "snowfall_log"]],
        columns=[iso_col, "snowfall_log"],
        key_on=f"feature.properties.{iso_col}",
        fill_color="Blues",
        fill_opacity=0.75,
        line_opacity=0.25,
        line_color="white",
        nan_fill_color="#eeeeee",    # light grey for countries with no snowfall data
        nan_fill_opacity=0.4,
        bins=bins,
        legend_name=(
            "Total snowfall volume — log scale (km³, summed across all Olympic editions attended)"
        ),
        name="Snowfall intensity",
    ).add_to(m)

    # ── Layer 1b: Black fill for non-participating countries ─────────────────
    # Countries that have never attended a Winter Games are filled black so
    # they are immediately distinguishable from participating nations.
    non_participating_geojson = {
        "type": "FeatureCollection",
        "features": [
            f for f in geojson["features"]
            if f["properties"].get("editions_attended", 0) == 0
        ],
    }
    folium.GeoJson(
        non_participating_geojson,
        style_function=lambda _: {
            "fillColor":   "#111111",
            "color":       "#333333",
            "weight":      0.5,
            "fillOpacity": 0.7,
        },
        name="Non-participating countries",
        show=True,
    ).add_to(m)

    # ── Layer 2: Transparent GeoJson overlay (tooltip) ────────────────────
    # This layer is invisible (transparent fill and near-invisible border).
    # Its only job is to supply the hover tooltip with all the project-relevant
    # data fields.  On hover, the country flashes yellow to confirm selection.
    folium.GeoJson(
        geojson,
        style_function=lambda _: {
            "fillColor":   "transparent",
            "color":       "#aaa",
            "weight":      0.3,
            "fillOpacity": 0,
        },
        highlight_function=lambda _: {
            "fillColor":   "#ffdd44",   # yellow highlight on hover
            "color":       "#333",
            "weight":      2,
            "fillOpacity": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "country_name",
                "noc_code",
                "snowfall_mm_mean",
                "snowfall_total_sum",
                "total_medals",
                "gold",
                "silver",
                "bronze",
                "medals_per_million",
                "editions_attended",
                "gdp_billions",
                "population_millions",
            ],
            aliases=[
                "Country",
                "NOC code",
                "Avg snowfall depth (mm / edition)",
                "Total snowfall volume (km³, all editions)",
                "Total medals (all editions)",
                "Gold 🥇",
                "Silver 🥈",
                "Bronze 🥉",
                "Medals per million inhabitants",
                "Editions attended",
                "Avg total GDP (billion USD)",
                "Avg population (millions)",
            ],
            localize=True,   # use browser locale for number formatting
            sticky=True,     # tooltip stays visible while mouse is over country
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

    # ── Title banner ──────────────────────────────────────────────────────
    # Fixed-position HTML injected into the map page header.
    # pointer-events: none ensures the banner does not block map interaction.
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
            hover over a country for details
        </span>
    </div>
    """))

    # Layer control: lets the user toggle the choropleth layer on/off.
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
    print("  Open the HTML file in any browser to view the interactive map.")
    print("  Hover over any country to see:")
    print("    • Snowfall depth (mm) and volume (km³)")
    print("    • Total medals broken down by gold / silver / bronze")
    print("    • Medals per million inhabitants")
    print("    • Avg total GDP (billion USD) and population (millions)")
    print("    • Number of Winter Games editions attended")

if __name__ == "__main__":
    main()