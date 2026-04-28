"""
Collection Pipeline — Raw Data
=====================================
Fetches raw data from all sources. No filtering, no cleaning, no merging. ALl refinements happens in clean.py.

Data sources used:
  1. World Bank API       — GDP and population figures for all countries
  2. ERA5 NetCDF file     — Monthly snowfall raster data (local file, pre-downloaded)
  3. Kaggle CSV           — Historical Winter Olympics athlete data (1992–2014)
  4. Wikipedia (scraped)  — Winter Olympics medals + participants (2018–2022)

Output files in data/raw/:
  worldbank_gdp.csv               — GDP + GDP per capita, ALL countries, 1992–2024
  worldbank_population.csv        — Population total, ALL countries, 1992–2024
  snowfall_raw.csv                — Snowfall, ALL countries, 1992–2026
  olympics_participants.csv       — All Winter Olympic participants from 1992-2014
  olympics_medals.csv             — All NOC x Year combinations, medals = 0 if none won, from 1992-2014
  olympics_participants_wikipedia — Webscraped Wikipedia of all winter Olympics participants 2018–2022
  olympics_medals_wikipedia       — Webscraped Wikipedia of all winter Olympics medal winners 2018–2022

Usage - inside terminal :
  python fetch.py               # fetch all
  python fetch.py --refresh     # force re-fetch (ignore cache)
  python fetch.py --skip-snow   # skip the slow ERA5 step
  python fetch.py --skip-wiki   # skip Wikipedia scraping
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import regionmask   # Provides country polygon masks aligned to grid coordinates
import requests
import xarray as xr   # Used to open and process the ERA5 NetCDF climate dataset
from bs4 import BeautifulSoup  # HTML parser for Wikipedia scraping
from Country_mapping import ENGLISH_NAME_TO_NOC
import pycountry

START_YEAR    = 1992 # Earliest year of data we want across all data sources
RAW_DIR       = Path("data/raw") # Where all raw output CSVs are written
NC_FILE       = Path('Data/data_stream-moda.nc') # Local ERA5 NetCDF file — contains the snowfall data
OLYMPICS_FILE = Path("data/olympicDataset/athlete_events.csv") # Kaggle Winter Olympics dataset — one row per athlete per event per year.

# Used to compute the surface area of each 0.1° x 0.1° grid cell (for snowfall volume)
EARTH_RADIUS_KM = 6371.0 # Mean radius of the Earth in kilometres
CELL_DEG        = 0.1 # Grid resolution in degrees (ERA5 dataset is 0.1° x 0.1°)

GDP_INDICATORS = { # World Bank API indicator codes for the two GDP metrics we want.These are the official codes used in the World Bank's data API v2.
    "gdp_usd":            "NY.GDP.MKTP.CD", # Total GDP in current USD
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD", # GDP per capita in current USD
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def to_iso3(name: str) -> str:
    """
    Convert a regionmask country name to an ISO 3166-1 alpha-3 code.
    Uses pycountry fuzzy search, with a small manual table for the
    abbreviated names that regionmask uses which pycountry can't resolve.
    """
    MANUAL = {
        "Bosnia and Herz.": "BIH",
        "Turkey":           "TUR",  # pycountry knows it as Türkiye
        "Kosovo": "XKX",  # not ISO recognised; pycountry wrongly returns SRB
    }
    if name in MANUAL:
        return MANUAL[name]
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except Exception:
        return None

def compute_cell_area_km2(latitudes: np.ndarray) -> xr.DataArray:
    """
    Compute the surface area (in km²) of each 0.1° x 0.1° ERA5 grid cell.

    Grid cells are square in degrees, but their actual physical size shrinks towards the poles because lines of longitude converge.
    Specifically, cell width = R * cos(lat) * Δlon, so area = R² * Δlat * Δlon * cos(lat).

    This area array is used to convert snowfall depth (metres) to volume (km³) by multiplying: volume = depth * cell_area / 1000.
    """
    delta_rad = np.deg2rad(CELL_DEG)
    lat_rad   = np.deg2rad(latitudes)
    area_km2  = (EARTH_RADIUS_KM ** 2) * (delta_rad ** 2) * np.cos(lat_rad)
    return xr.DataArray(area_km2, coords={"latitude": latitudes}, dims=["latitude"])

def fetch_wb_indicator(code: str) -> pd.DataFrame:
    """"
    Fetch a single World Bank indicator for ALL countries from the World Bank API v2.

    The World Bank API is paginated. We request the maximum allowed page size (32767 rows) to minimise the number of HTTP round trips needed.

    Retries up to 3 times with increasing wait times on timeout or connection errors.
    This is important because the World Bank API can be slow and occasionally drops requests.

    Args:
        code: World Bank indicator code (e.g. "NY.GDP.MKTP.CD" for total GDP)

    Returns:
        DataFrame with columns: iso3, country_name, year, value
        One row per (country, year) combination. Value is None if data is missing.
    """
    url    = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "per_page": 32767, "date": f"{START_YEAR}:2024"}
    rows, page, total_pages = [], 1, None

    def _get_page(p: int) -> list:
        for attempt in range(3): # Fetch a single page from the API, retrying up to 3 times on failure.
            try:
                resp = requests.get(url, params={**params, "page": p}, timeout=90)
                resp.raise_for_status() # Raise an error for HTTP 4xx/5xx responses
                data = resp.json() # The API returns a 2-element list: [metadata_dict, records_list]
                if not isinstance(data, list) or len(data) < 2:
                    raise ValueError(f"Unexpected response: {data}")
                return data
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    ValueError) as e:
                if attempt == 2:
                    raise # Give up after 3 failed attempts
                wait = 5 * (attempt + 1)
                print(f"\n      [{code}] page {p} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    while True: # Paginate through all pages until we've collected every record
        data = _get_page(page)
        meta, records = data[0], data[1] or [] # data[0] = metadata, data[1] = records
        if total_pages is None: # On the first page, read the total page count from the metadata
            total_pages = meta["pages"]
            print(f"    {code}: {meta['total']} records, {total_pages} page(s)")
        if total_pages == 0:
            break # No data at all for this indicator

        for r in records:
            if r: # Skip null entries that the API occasionally returns
                rows.append({
                    "iso3":         r.get("countryiso3code") or r.get("country", {}).get("id", ""),
                    "country_name": r.get("country", {}).get("value", ""),
                    "year":         int(r["date"]),
                    "value":        r.get("value"), # None if data not available for that year
                })

        print(f"      page {page}/{total_pages}", end="\r")
        if page >= total_pages:
            break
        page += 1

    print() # Move to a new line after the in-place page counter
    return pd.DataFrame(rows)

def save(df: pd.DataFrame, path: Path) -> None:
    """
       Write a DataFrame to CSV and print a confirmation message.

       index=False prevents pandas from writing the row index as an extra column.

       Args:
           df:   DataFrame to save
           path: Destination file path (should be inside RAW_DIR)
       """
    df.to_csv(path, index=False)
    print(f"  [saved] {len(df):,} rows → {path}")

# ---------------------------------------------------------------------------
# 1. World Bank GDP  (all countries, gdp + gdp per capita side by side)
# ---------------------------------------------------------------------------
def fetch_gdp(refresh: bool = False) -> None: # Fetch total GDP and GDP per capita for all countries (1992–2024) from the World Bank API.
    out_path = RAW_DIR / "worldbank_gdp.csv"
    if out_path.exists() and not refresh:
        print("  [cache] worldbank_gdp.csv")
        return

    print("\n[WORLD BANK — GDP] Fetching all countries (parallel)...")
    frames = {}

    # Fetch GDP and GDP-per-capita simultaneously using two worker threads
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(fetch_wb_indicator, code): col_name
                   for col_name, code in GDP_INDICATORS.items()}
        for future in as_completed(futures):
            col_name = futures[future]
            # Rename the generic "value" column to the specific metric name
            frames[col_name] = future.result().rename(columns={"value": col_name})

    # Outer merge so we keep rows even if one indicator is missing for a country/year
    df = frames["gdp_usd"].merge(
        frames["gdp_per_capita_usd"][["iso3", "year", "gdp_per_capita_usd"]],
        on=["iso3", "year"],
        how="outer",
    )
    save(df, out_path)

# ---------------------------------------------------------------------------
# 2. World Bank Population  (all countries, total only)
# ---------------------------------------------------------------------------

def fetch_population(refresh: bool = False) -> None: # Fetch total population for all countries (1992–2024) from the World Bank API.
    out_path = RAW_DIR / "worldbank_population.csv"
    if out_path.exists() and not refresh:
        print("  [cache] worldbank_population.csv")
        return

    print("\n[WORLD BANK — Population] Fetching all countries…")
    df = fetch_wb_indicator("SP.POP.TOTL").rename(columns={"value": "population_total"})
    save(df, out_path)

# ---------------------------------------------------------------------------
# 3. ERA5 Snowfall  (all countries)
# ---------------------------------------------------------------------------
def fetch_snowfall(refresh: bool = False) -> None:
    """
    Aggregate ERA5 monthly snowfall data into annual per-country totals.

    The ERA5 dataset is a global climate reanalysis product from the Copernicus Climate Data Store.
    The local NetCDF file (NC_FILE) contains the variable "sf" (snowfall, in metres of water equivalent) on a 0.1° x 0.1° global grid,
    with one timestep per month.

    Processing steps:
      1. Open the NetCDF and extract the "sf" variable
      2. Chunk the data with Dask so it can be processed piece by piece without
         loading the full dataset into RAM
      3. Resample monthly timesteps into annual sums
      4. Compute each cell's physical area in km² (cells shrink towards the poles)
      5. Multiply depth * area / 1000 to get snowfall volume in km³
      6. Compute total land area per country using the same cell areas (for depth calc)
      7. Apply a country mask (from regionmask) to assign each grid cell to a country
      8. Sum volumes per country per year
      9. Derive area-weighted snowfall depth: depth_mm = volume_km3 / country_area_km2 * 1_000_000
     10. Write one row per (country, year) to CSV

    Output: data/raw/snowfall_raw.csv
      Columns: iso3, country_name, year, snowfall_km3, snowfall_mm
        snowfall_km3 — total annual snowfall volume for the country
        snowfall_mm  — area-weighted average annual snowfall depth (physically correct)

    Note: This step is slow (can take 5–20 minutes depending on hardware).
    Use --skip-snow to skip it if you already have the cached CSV.
    """
    out_path = RAW_DIR / "snowfall_raw.csv"
    if out_path.exists() and not refresh:
        print("\n[SNOWFALL] Using cached data.")
        return

    print("\n[SNOWFALL] Processing ERA5 NetCDF — all countries…")
    t0 = time.time()

    print("  Opening dataset…")
    ds = xr.open_dataset(NC_FILE)
    sf = ds["sf"]   # "sf" = snowfall variable (metres of water equivalent per month)
    print(f"  {len(sf.valid_time)} timesteps · {len(sf.latitude)} lats · {len(sf.longitude)} lons")

    # Chunk the dataset so Dask can process it lazily without loading everything into RAM. valid_time=12 means one year of months per chunk; latitude/longitude=-1 = no chunking on those axes.
    sf = sf.chunk({"valid_time": 12, "latitude": -1, "longitude": -1})

    print("  Resampling monthly → annual…")
    sf_yearly = sf.resample(valid_time="YE").sum() # Sum monthly snowfall values into a single annual total per grid cell

    print("  Computing cell areas…")
    # compute_cell_area_km2 returns an xr.DataArray with a latitude coordinate.
    # When multiplied against sf_yearly (which has lat + lon + time dims), xarray automatically broadcasts the 1D latitude array
    # across longitude and time, so every cell gets the correct area for its latitude row.
    cell_area_da = compute_cell_area_km2(sf.latitude.values)

    sf_volume = sf_yearly * cell_area_da / 1000.0     # Convert depth (m water equivalent) × area (km²) / 1000 → volume in km³ water equivalent

    print("  Building country mask…")
    # regionmask assigns each (lat, lon) grid cell an integer country ID using Natural Earth country polygons at 1:110m resolution
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(sf.longitude, sf.latitude)   # Shape: (n_lat, n_lon), values = country index or NaN

    print("  Computing groupby (slow step)…")
    t_compute = time.time()
    try:
        # Use Dask's ProgressBar if available for a visual progress indicator during compute()
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            # Total annual snowfall volume per country per year (km³)
            result_sum = sf_volume.groupby(mask).sum().compute()

            # Total land area per country according to the mask (km²).
            # broadcast_like expands the 1D (latitude-only) cell_area_da into a full
            # 2D (lat × lon) grid matching a single timestep, so the groupby over the
            # 2D mask can correctly sum all cell areas belonging to each country.
            # This is the denominator used to convert volume → area-weighted depth.
            country_area = (
                cell_area_da
                .broadcast_like(sf_yearly.isel(valid_time=0))
                .groupby(mask)
                .sum()
                .compute()
            )
    except ImportError:
        result_sum = sf_volume.groupby(mask).sum().compute()
        country_area = (cell_area_da
            .broadcast_like(sf_yearly.isel(valid_time=0))
            .groupby(mask)
            .sum()
            .compute()
        )
    print(f"  Groupby done in {time.time() - t_compute:.0f}s")

    print("  Building DataFrame...")
    country_ids = result_sum.coords["mask"].values.astype(int)
    valid_ids   = country_ids[country_ids >= 0]    # Filter out -1 (ocean / unmasked cells)
    years       = result_sum.coords["valid_time"].dt.year.values

    records = []
    for idx in valid_ids:
        region = countries[int(idx)]
        iso3   = to_iso3(region.name)
        vols   = result_sum.sel(mask=idx).values         # Annual km³ values across all years
        area   = float(country_area.sel(mask=idx).values)  # Total masked land area in km²

        for year, vol in zip(years, vols):
            if not np.isnan(vol) and area > 0:
                snowfall_km3 = float(vol)
                # Area-weighted depth: convert km³ / km² → km, then × 1,000,000 → mm
                snowfall_mm  = snowfall_km3 / area * 1_000_000
            else:
                snowfall_km3 = None
                snowfall_mm  = None

            records.append((iso3, region.name, int(year), snowfall_km3, snowfall_mm, ))

    df_snow = pd.DataFrame(records, columns=[
        "iso3", "country_name", "year", "snowfall_km3", "snowfall_mm"])

    save(df_snow, out_path)
    print(f"  Total time: {time.time() - t0:.0f}s")

# ---------------------------------------------------------------------------
# 4. Olympics — participants + complete medals table (0 for non-winners)
# ---------------------------------------------------------------------------
def fetch_olympics(refresh: bool = False) -> None:
    """
    Build a unified Winter Olympics table from the Kaggle athlete dataset.

    The raw Kaggle file has one row per athlete per event. We aggregate into
    one row per (NOC, year) containing team_name, n_athletes, and medal counts.
    Countries that won zero medals are explicitly included with medal counts of 0
    — a missing row is ambiguous, a 0 is an explicit fact.

    Non-country NOC codes (IOA, AHO) are dropped here before aggregation:
      IOA = Individual Olympic Athletes (stateless, no ISO3 successor)
      AHO = Netherlands Antilles (dissolved 2010, no meaningful successor)

    Data coverage: 1992–2014. Wikipedia scraping (fetch_wikipedia) covers 2018+.

    Output: data/raw/olympics.csv
      Columns: noc_code, year, team_name, n_athletes,
               gold, silver, bronze, total_medals
    """
    out_path = RAW_DIR / "olympics.csv"
    if out_path.exists() and not refresh:
        print("\n[OLYMPICS] Using cached data.")
        return

    DROP_NOC = {"IOA", "AHO"}

    print("\n[OLYMPICS] Processing athlete_events.csv…")
    df = pd.read_csv(OLYMPICS_FILE)

    # Filter to Winter Games in scope, and drop non-country entities
    df = df[
        (df["Season"] == "Winter") &
        (df["Year"] >= START_YEAR) &
        (df["Year"] <= 2022) &
        (~df["NOC"].isin(DROP_NOC))
    ].copy()

    # ── Per-country aggregates ──────────────────────────────────────────────
    base = (
        df.groupby(["NOC", "Year"])
        .agg(
            team_name  = ("Team", lambda x: x.value_counts().index[0]),
            n_athletes = ("Name", "nunique"),
        )
        .reset_index()
        .rename(columns={"NOC": "noc_code", "Year": "year"})
    )

    # ── Medal counts ────────────────────────────────────────────────────────
    medal_df = df[df["Medal"].notna()].copy()
    medal_df["gold"]   = (medal_df["Medal"] == "Gold").astype(int)
    medal_df["silver"] = (medal_df["Medal"] == "Silver").astype(int)
    medal_df["bronze"] = (medal_df["Medal"] == "Bronze").astype(int)

    medal_counts = (
        medal_df.groupby(["NOC", "Year"])
        .agg(gold=("gold", "sum"), silver=("silver", "sum"), bronze=("bronze", "sum"))
        .reset_index()
        .rename(columns={"NOC": "noc_code", "Year": "year"})
    )

    # ── Merge medals onto base — non-winners get 0 ─────────────────────────
    olympics = base.merge(medal_counts, on=["noc_code", "year"], how="left")
    olympics[["gold", "silver", "bronze"]] = (
        olympics[["gold", "silver", "bronze"]].fillna(0).astype(int)
    )
    olympics["total_medals"] = olympics["gold"] + olympics["silver"] + olympics["bronze"]

    save(olympics.sort_values(["year", "noc_code"]), out_path)
    print(f"  {olympics['noc_code'].nunique()} NOC codes · "
          f"years: {sorted(olympics['year'].unique())[0]}–"
          f"{sorted(olympics['year'].unique())[-1]}")

# ---------------------------------------------------------------------------
# 5. Wikipedia — medal tables + participants 2018, 2022, 2026
# ---------------------------------------------------------------------------
# Main Wikipedia page for each Winter Olympics (used to find the participant list)
GAMES_URLS = {
    2018: "https://en.wikipedia.org/wiki/2018_Winter_Olympics",
    2022: "https://en.wikipedia.org/wiki/2022_Winter_Olympics",
}

# Wikipedia medal table pages.
MEDAL_URLS = {
    2018: "https://en.wikipedia.org/wiki/2018_Winter_Olympics_medal_table",
    2022: "https://en.wikipedia.org/wiki/2022_Winter_Olympics_medal_table",
}

# Mimic a real browser's User-Agent header so Wikipedia doesn't block the request
WIKI_HEADERS = {"User-Agent": "olympics-snowfall-research-bot/1.0 (student project; contact: your@email.com)"}

def _fetch_page(url: str) -> BeautifulSoup:
    """
        Download a Wikipedia page and parse it into a BeautifulSoup object.

        Includes a 1-second sleep after each request to be a polite web scraper and avoid triggering Wikipedia's rate limiting.
    """
    resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(5)  # Polite delay — avoids hammering Wikipedia's servers
    return BeautifulSoup(resp.text, "html.parser")

def _scrape_medal_table(year: int, soup: BeautifulSoup) -> pd.DataFrame:
    """"
    Extract the medal table from a parsed Wikipedia page.

    Wikipedia medal tables are HTML <table> elements with class "wikitable".
    We identify the correct table by looking for one whose first row has headers containing "Gold", "Silver", "Bronze", and "Total".

    Each data row looks like:
      Rank | Country (hyperlink) | Gold | Silver | Bronze | Total

    The country name is extracted from the first hyperlink in the row (the flag image link or country name link).
    Medal counts are extracted by parsing all integer-valued cells, then reading them from the end of
    the row (to handle variable numbers of leading columns like rank).

    Rows are skipped if:
      - They have fewer than 4 cells (header/footer rows)
      - The country name can't be mapped to a known NOC code
      - total_medals > 200 (likely a "Totals" summary row, not a country)
    """
    table = None
    for t in soup.find_all("table", class_="wikitable"):     # Find the first wikitable whose header row contains all four medal columns
        first_row = t.find("tr")
        if not first_row:
            continue
        hdrs = [th.get_text(strip=True) for th in first_row.find_all("th")]
        if "Gold" in hdrs and "Silver" in hdrs and "Bronze" in hdrs and "Total" in hdrs:
            table = t
            break
    if table is None: # Page had no medal table — return empty result
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue # Page had no medal table — return empty result

        # Country name from the first anchor tag in the row
        country_name = next(
            (a.get_text(strip=True).replace("*","").strip()
             for cell in cells for a in [cell.find("a")] if a
             and len(a.get_text(strip=True)) > 2),
            None
        )
        if not country_name:
            continue

        noc = ENGLISH_NAME_TO_NOC.get(country_name)
        if not noc: # Skip if we can't map the name to a known NOC code
            continue

        numbers = []
        for cell in cells:
            try:
                numbers.append(int(cell.get_text(strip=True).replace("*","").replace("\xa0","")))
            except ValueError: # Skip non-numeric cells (country name, rank with letters, etc.)
                pass

        if len(numbers) < 3:
            continue # Need at least gold, silver, bronze

        # Read medal counts from the end of the numbers list. Wikipedia rows end with: [..., Gold, Silver, Bronze, Total]
        gold, silver, bronze = numbers[-4], numbers[-3], numbers[-2]
        total = numbers[-1] if len(numbers) >= 4 else gold + silver + bronze

        if total > 200:
            continue

        rows.append({"noc_code": noc, "country_name": country_name, "year": year, "gold":gold,
            "silver": silver, "bronze": bronze, "total_medals": total})

    return pd.DataFrame(rows)


def _scrape_participants(year: int, soup: BeautifulSoup, medal_nocs: set) -> pd.DataFrame:
    """"
    Parse the participant list from a Wikipedia Winter Olympics page.

    Wikipedia lists participating nations in a section headed something like
    "Participating National Olympic Committees". Each entry looks like:
      Albania (2), Andorra (5), Australia (40), ...
    where the number in parentheses is the count of athletes.

    Strategy:
      1. Find the "Participating" section heading (h2 or h3)
      2. Collect all <li> elements that follow it until the next heading
      3. Parse each item with a regex to extract country name and athlete count
      4. Always include medal-winning countries even if they're not in the list
         (as a safety net for scraping misses)
    """
    import re

    # Find the "Participating" section heading
    heading = next(
        (h for h in soup.find_all(["h2", "h3"])
         if "Participating" in h.get_text() or "National Olympic" in h.get_text()),
        None
    )

    found = {}  # noc_code -> {name, n_athletes}

    if heading:         # Walk forward through the DOM from the heading, collecting <li> items.
        # Stop when we hit the next section heading (h2 or h3).
        for el in heading.find_all_next(["li", "h2", "h3"]):
            if el.name in ["h2", "h3"]:
                break  # Reached the next section — stop collecting

            text = el.get_text(strip=True)
            m = re.match(r"^(.+?)\s*\((\d+)\)", text)  # Match pattern: "Country Name (NumberOfAthletes)" with optional trailing symbols
            if not m or not (1 <= int(m.group(2)) <= 500):
                continue

            # Strip footnote markers (*, †, [) that sometimes appear after the country name
            name = m.group(1).strip().rstrip("*†[").strip()
            n    = int(m.group(2))
            noc  = ENGLISH_NAME_TO_NOC.get(name)
            if noc:
                found[noc] = {"name": name, "n_athletes": n}

    # Safety net: ensure every medal-winning country appears in the participant list.
    # If Wikipedia's participant section is missing or malformed, we'd otherwise lose medal-winning countries from the output entirely.
    for noc in medal_nocs:
        if noc not in found:
            found[noc] = {"name": noc, "n_athletes": 0}

    return pd.DataFrame([{
        "noc_code":   noc,
        "year":       year,
        "team_name":  info["name"],
        "n_athletes": info["n_athletes"],
    } for noc, info in sorted(found.items())])


def fetch_wikipedia(refresh: bool = False) -> None:
    """
    Scrape Winter Olympics data from Wikipedia (2018–2022).

    Produces a single unified file per year — one row per (NOC, year) with
    team_name, n_athletes, and medal counts — matching the structure of
    olympics.csv produced by fetch_olympics().

    Non-medal countries are included with medal counts of 0.
    Non-country NOC codes (IOA, AHO) are dropped here before saving.

    Output: data/raw/olympics_wikipedia.csv
      Columns: noc_code, year, team_name, n_athletes,
               gold, silver, bronze, total_medals
    """
    DROP_NOC = {"IOA", "AHO"}

    out_path = RAW_DIR / "olympics_wikipedia.csv"
    if out_path.exists() and not refresh:
        print("\n[WIKIPEDIA] Using cached data.")
        return

    print("\n[WIKIPEDIA] Scraping 2018 / 2022")
    all_years = []

    for year in [2018, 2022]:

        # ── Medal table ──────────────────────────────────────────────────────
        print(f"\n  [{year}] Medal table ({MEDAL_URLS[year].split('/')[-1]})...")
        df_m = pd.DataFrame()
        try:
            soup_m = _fetch_page(MEDAL_URLS[year])
            df_m   = _scrape_medal_table(year, soup_m)
            if not df_m.empty:
                print(f"    -> {len(df_m)} countries with medals")
            else:
                print(f"    [WARNING] No medals found for {year}")
        except Exception as e:
            print(f"    [ERROR] medals: {e}")

        # ── Participants ─────────────────────────────────────────────────────
        print(f"  [{year}] Participants ({GAMES_URLS[year].split('/')[-1]})...")
        try:
            # For 2018 the games page == medal page — avoid double fetch
            soup_g = soup_m if GAMES_URLS[year] == MEDAL_URLS[year] else _fetch_page(GAMES_URLS[year])
            medal_nocs = set(df_m["noc_code"].tolist()) if not df_m.empty else set()
            df_p       = _scrape_participants(year, soup_g, medal_nocs)
            print(f"    -> {len(df_p)} countries")
        except Exception as e:
            print(f"    [ERROR] participants: {e}")
            df_p = pd.DataFrame()

        if df_p.empty:
            continue

        # ── Merge medals into participants — non-winners get 0 ───────────────
        combined = df_p.merge(df_m, on=["noc_code", "year"], how="left")
        combined[["gold", "silver", "bronze", "total_medals"]] = (
            combined[["gold", "silver", "bronze", "total_medals"]].fillna(0).astype(int)
        )

        # Drop non-country entities
        combined = combined[~combined["noc_code"].isin(DROP_NOC)]

        all_years.append(combined)

    if all_years:
        result = (
            pd.concat(all_years, ignore_index=True)
            .drop_duplicates(subset=["noc_code", "year"])
        )
        save(result[["noc_code", "year", "team_name", "n_athletes",
                      "gold", "silver", "bronze", "total_medals"]], out_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """
        Parse command-line arguments and run each data fetch step in order.

        Steps run in this order:
          1. GDP           (fast — World Bank API, ~30s)
          2. Population    (fast — World Bank API, ~15s)
          3. Olympics      (fast — local CSV, <5s)
          4. Wikipedia     (moderate — 3 web pages, ~10s)
          5. Snowfall      (slow — ERA5 NetCDF processing, 5–20 min)

        Snowfall is last because it's by far the slowest step.
        Each step is skippable independently via flags.
    """
    p = argparse.ArgumentParser(description="Fetch all raw data — no filtering")
    p.add_argument("--refresh",   action="store_true", help="Ignore cache, re-fetch everything")
    p.add_argument("--skip-snow", action="store_true", help="Skip the slow ERA5 snowfall step")
    p.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia scraping")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)     # Create the output directory if it doesn't already exist

    fetch_gdp(refresh=args.refresh)
    fetch_population(refresh=args.refresh)
    fetch_olympics(refresh=args.refresh)

    if not args.skip_wiki:
        fetch_wikipedia(refresh=args.refresh)
    else:
        print("\n[WIKIPEDIA] Skipped (--skip-wiki)")

    if not args.skip_snow:
        fetch_snowfall(refresh=args.refresh)
    else:
        print("\n[SNOWFALL] Skipped (--skip-snow)")

    print("\n✓ All raw data collected -> data/raw/")
    print("  worldbank_gdp.csv                  — gdp_usd + gdp_per_capita_usd")
    print("  worldbank_population.csv            — population_total")
    print("  olympics_participants.csv           — Kaggle: 1992-2014")
    print("  olympics_medals.csv                 — Kaggle: 1992-2014")
    print("  olympics_participants_wikipedia.csv — Wikipedia: 2018-2014")
    print("  olympics_medals_wikipedia.csv       — Wikipedia: 2018-2022")
    print("  snowfall_raw.csv                    — all countries, snowfall per year")

if __name__ == "__main__":
    main()