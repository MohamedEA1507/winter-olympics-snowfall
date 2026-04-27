"""
Collection Pipeline — Raw Data Only
=====================================
Fetches raw data from all sources. No filtering, no cleaning, no merging.
All refinement happens in clean.py.

Output files in data/raw/:
  worldbank_gdp.csv            — GDP + GDP per capita, ALL countries, 1992–2024
  worldbank_population.csv     — Population total, ALL countries, 1992–2024
  snowfall_raw.csv             — Snowfall, ALL countries, 1992–2025
  olympics_participants.csv    — All Winter Olympic participants (NOC x Year), no filtering
  olympics_medals.csv          — All NOC x Year combinations, medals = 0 if none won

Usage:
  python fetch.py               # fetch all
  python fetch.py --refresh     # force re-fetch (ignore cache)
  python fetch.py --skip-snow   # skip the slow ERA5 step
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import regionmask
import requests
import xarray as xr
from bs4 import BeautifulSoup

START_YEAR    = 1992
RAW_DIR       = Path("data/raw")
NC_FILE       = Path('Data/data_stream-moda.nc')
OLYMPICS_FILE = Path("data/olympicDataset/athlete_events.csv")

EARTH_RADIUS_KM = 6371.0
CELL_DEG        = 0.1

GDP_INDICATORS = {
    "gdp_usd":            "NY.GDP.MKTP.CD",
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
}

OLYMPIC_HOSTS = {
    1992: "FRA", 1994: "NOR", 1998: "JPN", 2002: "USA",
    2006: "ITA", 2010: "CAN", 2014: "RUS", 2018: "KOR",
    2022: "CHN", 2026: "ITA",
}

REGIONMASK_TO_ISO3 = {
    "A":   "AUT",  "N":   "NOR",  "S":   "SWE",  "J":   "JPN",
    "CH":  "CHE",  "IS":  "ISL",  "IL":  "ISR",  "CL":  "CHL",
    "CA":  "CAN",  "US":  "USA",  "AL":  "ALB",  "ME":  "MNE",
    "KO":  "XKX",  "SLO": "SVN",  "SK":  "SVK",  "NM":  "MKD",
    "BG":  "BGR",  "TR":  "TUR",  "GE":  "GEO",  "GL":  "GRL",
    "AF":  "AFG",  "NP":  "NPL",  "BT":  "BTN",  "TJ":  "TJK",
    "KG":  "KGZ",  "TF":  "ATF",  "CZ":  "CZE",  "BiH": "BIH",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def progress(step, total, label, start_time):
    pct     = step / total
    done    = int(40 * pct)
    bar     = "█" * done + "░" * (40 - done)
    elapsed = time.time() - start_time
    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
    print(f"\r[{bar}] {pct*100:5.1f}%  {label}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
          end="", flush=True)

def to_iso3(abbrev: str, name: str) -> str:
    if abbrev in REGIONMASK_TO_ISO3:
        return REGIONMASK_TO_ISO3[abbrev]
    if len(abbrev) == 3:
        return abbrev.upper()
    try:
        import pycountry
        match = pycountry.countries.search_fuzzy(name)
        if match:
            return match[0].alpha_3
    except Exception:
        pass
    return abbrev.upper()

def compute_cell_area_km2(latitudes: np.ndarray) -> xr.DataArray:
    delta_rad = np.deg2rad(CELL_DEG)
    lat_rad   = np.deg2rad(latitudes)
    area_km2  = (EARTH_RADIUS_KM ** 2) * (delta_rad ** 2) * np.cos(lat_rad)
    return xr.DataArray(area_km2, coords={"latitude": latitudes}, dims=["latitude"])

def fetch_wb_indicator(code: str) -> pd.DataFrame:
    """Fetch a single World Bank indicator for ALL countries, no filtering.

    Uses per_page=32767 (API max) to minimise round trips, and retries
    up to 3 times with exponential back-off on timeout or server errors.
    """
    url    = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "per_page": 32767, "date": f"{START_YEAR}:2024"}
    rows, page, total_pages = [], 1, None

    def _get_page(p: int) -> list:
        for attempt in range(3):
            try:
                resp = requests.get(url, params={**params, "page": p}, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list) or len(data) < 2:
                    raise ValueError(f"Unexpected response: {data}")
                return data
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    ValueError) as e:
                if attempt == 2:
                    raise
                wait = 5 * (attempt + 1)
                print(f"\n      [{code}] page {p} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    while True:
        data = _get_page(page)
        meta, records = data[0], data[1] or []
        if total_pages is None:
            total_pages = meta["pages"]
            print(f"    {code}: {meta['total']} records, {total_pages} page(s)")
        if total_pages == 0:
            break

        for r in records:
            if r:
                rows.append({
                    "iso3":         r.get("countryiso3code") or r.get("country", {}).get("id", ""),
                    "country_name": r.get("country", {}).get("value", ""),
                    "year":         int(r["date"]),
                    "value":        r.get("value"),
                })

        print(f"      page {page}/{total_pages}", end="\r")
        if page >= total_pages:
            break
        page += 1

    print()
    return pd.DataFrame(rows)

def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  [saved] {len(df):,} rows → {path}")

# ---------------------------------------------------------------------------
# 1. World Bank GDP  (all countries, gdp + gdp per capita side by side)
# ---------------------------------------------------------------------------

def fetch_gdp(refresh: bool = False) -> None:
    out_path = RAW_DIR / "worldbank_gdp.csv"
    if out_path.exists() and not refresh:
        print("  [cache] worldbank_gdp.csv")
        return

    print("\n[WORLD BANK — GDP] Fetching all countries (parallel)...")
    frames = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(fetch_wb_indicator, code): col_name
                   for col_name, code in GDP_INDICATORS.items()}
        for future in as_completed(futures):
            col_name = futures[future]
            frames[col_name] = future.result().rename(columns={"value": col_name})

    df = frames["gdp_usd"].merge(
        frames["gdp_per_capita_usd"][["iso3", "year", "gdp_per_capita_usd"]],
        on=["iso3", "year"],
        how="outer",
    )
    save(df, out_path)

# ---------------------------------------------------------------------------
# 2. World Bank Population  (all countries, total only)
# ---------------------------------------------------------------------------

def fetch_population(refresh: bool = False) -> None:
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
    out_path = RAW_DIR / "snowfall_raw.csv"
    if out_path.exists() and not refresh:
        print("\n[SNOWFALL] Using cached data.")
        return

    print("\n[SNOWFALL] Processing ERA5 NetCDF — all countries…")
    t0 = time.time()

    print("  Opening dataset…")
    ds = xr.open_dataset(NC_FILE)
    sf = ds["sf"]
    print(f"  {len(sf.valid_time)} timesteps · {len(sf.latitude)} lats · {len(sf.longitude)} lons")
    sf = sf.chunk({"valid_time": 12, "latitude": -1, "longitude": -1})

    print("  Resampling monthly → annual…")
    sf_yearly = sf.resample(valid_time="YE").sum()

    print("  Computing cell areas…")
    cell_area = compute_cell_area_km2(sf.latitude.values)
    sf_volume = sf_yearly * cell_area / 1000.0  # km³ w.e.

    print("  Building country mask…")
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(sf.longitude, sf.latitude)

    print("  Computing groupby (slow step)…")
    t_compute = time.time()
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            result_sum  = sf_volume.groupby(mask).sum().compute()
            result_mean = sf_yearly.groupby(mask).mean().compute()
    except ImportError:
        result_sum  = sf_volume.groupby(mask).sum().compute()
        result_mean = sf_yearly.groupby(mask).mean().compute()
    print(f"  Groupby done in {time.time() - t_compute:.0f}s")

    print("  Building DataFrame...")
    country_ids = result_sum.coords["mask"].values.astype(int)
    valid_ids   = country_ids[country_ids >= 0]
    years       = result_sum.coords["valid_time"].dt.year.values

    records = []
    for idx in valid_ids:
        region = countries[int(idx)]
        iso3   = to_iso3(region.abbrev, region.name)
        vols   = result_sum.sel(mask=idx).values
        means  = result_mean.sel(mask=idx).values
        for year, vol, mean in zip(years, vols, means):
            records.append((
                iso3,
                region.name,
                int(year),
                float(vol)  if not np.isnan(vol)  else None,
                float(mean) * 1000 if not np.isnan(mean) else None,
            ))

    df_snow = pd.DataFrame(records, columns=[
        "iso3", "country_name", "year", "snowfall_km3", "snowfall_mm"
    ])

    save(df_snow, out_path)
    print(f"  Total time: {time.time() - t0:.0f}s")

# ---------------------------------------------------------------------------
# 4. Olympics — participants + complete medals table (0 for non-winners)
# ---------------------------------------------------------------------------

def fetch_olympics(refresh: bool = False) -> None:
    out_participants = RAW_DIR / "olympics_participants.csv"
    out_medals       = RAW_DIR / "olympics_medals.csv"

    if out_participants.exists() and out_medals.exists() and not refresh:
        print("\n[OLYMPICS] Using cached data.")
        return

    print("\n[OLYMPICS] Processing athlete_events.csv…")
    df = pd.read_csv(OLYMPICS_FILE)

    # Filter winter only + year range — NO filtering on NOC, keep everything
    df = df[
        (df["Season"] == "Winter") &
        (df["Year"] >= START_YEAR) &
        (df["Year"] <= 2026)
    ].copy()

    # ── Participants: one row per (NOC, Year) — every country that showed up
    participants = (
        df.groupby(["NOC", "Year"])
        .agg(
            team_name=("Team",  lambda x: x.value_counts().index[0]),
            n_athletes=("Name", "nunique"),
        )
        .reset_index()
        .rename(columns={"NOC": "noc_code", "Year": "year"})
    )
    participants["host_flag"] = participants.apply(
        lambda r: 1 if OLYMPIC_HOSTS.get(r["year"]) == r["noc_code"] else 0, axis=1
    )

    save(participants.sort_values(["year", "noc_code"]), out_participants)
    print(f"  {participants['noc_code'].nunique()} NOC codes · "
          f"years: {sorted(participants['year'].unique()).pop(0)}–"
          f"{sorted(participants['year'].unique()).pop()}")

    # ── Medals: start from ALL (NOC, Year) combinations from participants
    #    then left-join actual medal counts → countries with no medals get 0
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

    # Merge onto full participant list so every country appears
    medals = participants[["noc_code", "year"]].merge(
        medal_counts,
        on=["noc_code", "year"],
        how="left",
    )
    medals[["gold", "silver", "bronze"]] = medals[["gold", "silver", "bronze"]].fillna(0).astype(int)
    medals["total_medals"] = medals["gold"] + medals["silver"] + medals["bronze"]

    save(medals.sort_values(["year", "noc_code"]), out_medals)


# ---------------------------------------------------------------------------
# 5. Wikipedia — medal tables + participants 2018, 2022, 2026
# ---------------------------------------------------------------------------

# English Wikipedia
GAMES_URLS = {
    2018: "https://en.wikipedia.org/wiki/2018_Winter_Olympics",
    2022: "https://en.wikipedia.org/wiki/2022_Winter_Olympics",
    2026: "https://en.wikipedia.org/wiki/2026_Winter_Olympics",
}
# 2018: medal table is on the main games page
# 2022/2026: medal table has its own dedicated URL
MEDAL_URLS = {
    2018: "https://en.wikipedia.org/wiki/2018_Winter_Olympics_medal_table",
    2022: "https://en.wikipedia.org/wiki/2022_Winter_Olympics_medal_table",
    2026: "https://en.wikipedia.org/wiki/2026_Winter_Olympics_medal_table",
}
WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
WIKI_NOC_TO_ISO3 = {
    "NOR": "NOR", "GER": "DEU", "CAN": "CAN", "USA": "USA", "NED": "NLD",
    "SWE": "SWE", "KOR": "KOR", "SUI": "CHE", "FRA": "FRA", "AUT": "AUT",
    "JPN": "JPN", "ITA": "ITA", "OAR": "RUS", "ROC": "RUS", "AIN": "RUS",
    "CZE": "CZE", "BLR": "BLR", "CHN": "CHN", "SVK": "SVK", "FIN": "FIN",
    "GBR": "GBR", "POL": "POL", "HUN": "HUN", "UKR": "UKR", "AUS": "AUS",
    "SLO": "SVN", "BEL": "BEL", "NZL": "NZL", "ESP": "ESP", "KAZ": "KAZ",
    "LAT": "LVA", "LIE": "LIE", "EST": "EST", "GEO": "GEO", "BUL": "BGR",
    "DEN": "DNK", "BRA": "BRA", "ARM": "ARM", "ROU": "ROU", "CRO": "HRV",
    "SRB": "SRB", "MEX": "MEX", "NIG": "NGA", "GRE": "GRC", "ISL": "ISL",
    "MKD": "MKD", "MNE": "MNE", "POR": "PRT", "RSA": "ZAF", "TUR": "TUR",
    "LTU": "LTU",
}


def _fetch_page(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(1)
    return BeautifulSoup(resp.text, "html.parser")



def _country_name_to_noc(name: str) -> str | None:
    """Map English country name to NOC code."""
    ENGLISH_NAME_TO_NOC = {
        "Albania": "ALB", "Andorra": "AND", "Argentina": "ARG",
        "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
        "Azerbaijan": "AZE", "Belarus": "BLR", "Belgium": "BEL",
        "Bermuda": "BER", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
        "Brazil": "BRA", "Bulgaria": "BUL", "Canada": "CAN",
        "Chile": "CHI", "China": "CHN", "Chinese Taipei": "TPE",
        "Colombia": "COL", "Croatia": "CRO", "Cyprus": "CYP",
        "Czech Republic": "CZE", "Czechia": "CZE", "Denmark": "DEN",
        "Ecuador": "ECU", "Eritrea": "ERI", "Estonia": "EST",
        "Finland": "FIN", "France": "FRA", "Georgia": "GEO",
        "Germany": "GER", "Ghana": "GHA", "Great Britain": "GBR",
        "Greece": "GRE", "Hong Kong": "HKG", "Hungary": "HUN",
        "Iceland": "ISL", "India": "IND", "Iran": "IRI",
        "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
        "Jamaica": "JAM", "Japan": "JPN", "Kazakhstan": "KAZ",
        "Kenya": "KEN", "Kosovo": "KOS", "Kyrgyzstan": "KGZ",
        "Latvia": "LAT", "Lebanon": "LIB", "Liechtenstein": "LIE",
        "Lithuania": "LTU", "Luxembourg": "LUX", "North Macedonia": "MKD",
        "Macedonia": "MKD", "Madagascar": "MAD", "Malaysia": "MAS",
        "Malta": "MLT", "Mexico": "MEX", "Moldova": "MDA",
        "Monaco": "MON", "Mongolia": "MGL", "Montenegro": "MNE",
        "Morocco": "MAR", "Netherlands": "NED", "New Zealand": "NZL",
        "Nigeria": "NGR", "North Korea": "PRK", "Norway": "NOR",
        "Olympic Athletes from Russia": "OAR", "ROC": "ROC",
        "Individual Neutral Athletes": "AIN", "Pakistan": "PAK",
        "Philippines": "PHI", "Poland": "POL", "Portugal": "POR",
        "Puerto Rico": "PUR", "Romania": "ROU", "Russia": "RUS",
        "San Marino": "SMR", "Serbia": "SRB", "Singapore": "SGP",
        "Slovakia": "SVK", "Slovenia": "SLO", "South Africa": "RSA",
        "South Korea": "KOR", "Korea": "KOR", "Spain": "ESP",
        "Sweden": "SWE", "Switzerland": "SUI", "Thailand": "THA",
        "Timor-Leste": "TLS", "Togo": "TOG", "Tonga": "TGA",
        "Turkey": "TUR", "Ukraine": "UKR", "United States": "USA",
        "Uzbekistan": "UZB", "Tajikistan": "TJK", "Peru": "PER",
        "American Samoa": "ASA", "Virgin Islands": "ISV",
        "United Arab Emirates": "UAE", "Uruguay": "URU",
        "Venezuela": "VEN", "Zimbabwe": "ZIM", "Nepal": "NEP",
        "Paraguay": "PAR", "British Virgin Islands": "IVB",
        "Cayman Islands": "CAY", "Dominica": "DMA", "Indonesia": "INA",
        "Guatemala": "GUA", "Haiti": "HAI", "El Salvador": "ESA",
        "Trinidad and Tobago": "TTO", "Saudi Arabia": "KSA",
        "Benin": "BEN", "Guinea-Bissau": "GBS", "Senegal": "SEN",
        "Tunisia": "TUN", "Ethiopia": "ETH", "Tanzania": "TAN",
        "Uganda": "UGA", "Namibia": "NAM", "Cameroon": "CMR",
        "Egypt": "EGY", "Algeria": "ALG", "Fiji": "FIJ",
        "Samoa": "SAM", "Guam": "GUM", "Kosovo": "KOS",
        "Israel": "ISR", "Liechtenstein": "LIE", "Jamaica": "JAM",
    }
    return ENGLISH_NAME_TO_NOC.get(name)

def _scrape_medal_table(year: int, soup: BeautifulSoup) -> pd.DataFrame:
    """
    Parse the medal table — first wikitable with Gold/Silver/Bronze headers.
    Rows: Rank | Country (link) | Gold | Silver | Bronze | Total
    """
    # Find the medal table: first row must have Rank/NOC + Gold/Silver/Bronze/Total
    table = None
    for t in soup.find_all("table", class_="wikitable"):
        first_row = t.find("tr")
        if not first_row:
            continue
        hdrs = [th.get_text(strip=True) for th in first_row.find_all("th")]
        if "Gold" in hdrs and "Silver" in hdrs and "Bronze" in hdrs and "Total" in hdrs:
            table = t
            break
    if table is None:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        # Country name from the first anchor tag in the row
        country_name = next(
            (a.get_text(strip=True).replace("*","").strip()
             for cell in cells for a in [cell.find("a")] if a
             and len(a.get_text(strip=True)) > 2),
            None
        )
        if not country_name:
            continue

        noc = _country_name_to_noc(country_name)
        if not noc:
            continue

        numbers = []
        for cell in cells:
            try:
                numbers.append(int(cell.get_text(strip=True).replace("*","").replace("\xa0","")))
            except ValueError:
                pass

        if len(numbers) < 3:
            continue

        gold, silver, bronze = numbers[-4], numbers[-3], numbers[-2]
        total = numbers[-1] if len(numbers) >= 4 else gold + silver + bronze

        if total > 200:
            continue

        rows.append({
            "noc_code":     noc,
            "country_name": country_name,
            "year":         year,
            "gold":         gold,
            "silver":       silver,
            "bronze":       bronze,
            "total_medals": total,
            "iso3":         WIKI_NOC_TO_ISO3.get(noc, noc),
            "host_flag":    1 if noc == OLYMPIC_HOSTS.get(year, "") else 0,
        })

    return pd.DataFrame(rows)


def _scrape_participants(year: int, soup: BeautifulSoup, medal_nocs: set) -> pd.DataFrame:
    """
    Parse the participant list from the "Participating National Olympic Committees"
    section. Entries look like: Albania (2), Andorra (5), ...
    We find the heading then collect all li items until the next heading.
    """
    import re

    # Find the "Participating" section heading
    heading = next(
        (h for h in soup.find_all(["h2", "h3"])
         if "Participating" in h.get_text() or "National Olympic" in h.get_text()),
        None
    )

    found = {}  # noc_code -> {name, n_athletes}

    if heading:
        for el in heading.find_all_next(["li", "h2", "h3"]):
            if el.name in ["h2", "h3"]:
                break
            text = el.get_text(strip=True)
            m = re.match(r"^(.+?)\s*\((\d+)\)", text)
            if not m or not (1 <= int(m.group(2)) <= 500):
                continue
            name = m.group(1).strip().rstrip("*†[").strip()
            n    = int(m.group(2))
            noc  = _country_name_to_noc(name)
            if noc:
                found[noc] = {"name": name, "n_athletes": n}

    # Always include medal countries
    for noc in medal_nocs:
        if noc not in found:
            found[noc] = {"name": noc, "n_athletes": 0}

    return pd.DataFrame([{
        "noc_code":   noc,
        "year":       year,
        "team_name":  info["name"],
        "n_athletes": info["n_athletes"],
        "iso3":       WIKI_NOC_TO_ISO3.get(noc, noc),
        "host_flag":  1 if noc == OLYMPIC_HOSTS.get(year, "") else 0,
    } for noc, info in sorted(found.items())])


def fetch_wikipedia(refresh: bool = False) -> None:
    out_medals = RAW_DIR / "olympics_medals_wikipedia.csv"
    out_parts  = RAW_DIR / "olympics_participants_wikipedia.csv"
    if out_medals.exists() and out_parts.exists() and not refresh:
        print("\n[WIKIPEDIA] Using cached data.")
        return
    print("\n[WIKIPEDIA] Scraping 2018 / 2022 / 2026...")
    all_medals, all_parts = [], []
    for year in [2018, 2022, 2026]:

        # ── Medal table ──
        print(f"\n  [{year}] Medal table ({MEDAL_URLS[year].split('/')[-1]})...")
        df_m = pd.DataFrame()
        try:
            soup_m = _fetch_page(MEDAL_URLS[year])
            df_m   = _scrape_medal_table(year, soup_m)
            if not df_m.empty:
                print(f"    -> {len(df_m)} countries with medals")
                all_medals.append(df_m)
            else:
                print(f"    [WARNING] No medals found for {year}")
        except Exception as e:
            print(f"    [ERROR] medals: {e}")

        # ── Participants (always from main games page) ──
        print(f"  [{year}] Participants ({GAMES_URLS[year].split('/')[-1]})...")
        try:
            # For 2018 the games page == medal page, avoid double fetch
            if GAMES_URLS[year] == MEDAL_URLS[year] and not df_m.empty:
                soup_g = soup_m
            else:
                soup_g = _fetch_page(GAMES_URLS[year])

            medal_nocs = set(df_m["noc_code"].tolist()) if not df_m.empty else set()
            df_p       = _scrape_participants(year, soup_g, medal_nocs)
            print(f"    -> {len(df_p)} countries")
            all_parts.append(df_p)
        except Exception as e:
            print(f"    [ERROR] participants: {e}")

    if all_parts:
        parts = pd.concat(all_parts, ignore_index=True).drop_duplicates(subset=["noc_code", "year"])
        save(parts, out_parts)
    else:
        parts = pd.DataFrame()

    if all_medals:
        medals = pd.concat(all_medals, ignore_index=True)

        # Expand to ALL participants — non-winners get 0 medals
        if not parts.empty:
            medals = parts[["noc_code", "year", "iso3"]].merge(
                medals.drop(columns=["iso3"], errors="ignore"),
                on=["noc_code", "year"],
                how="left",
            )
            medals[["gold", "silver", "bronze", "total_medals"]] = (
                medals[["gold", "silver", "bronze", "total_medals"]].fillna(0).astype(int)
            )
            medals["country_name"] = medals["country_name"].fillna(medals["noc_code"])
            medals["host_flag"]    = medals["host_flag"].fillna(0).astype(int)

        save(medals, out_medals)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fetch all raw data — no filtering")
    p.add_argument("--refresh",   action="store_true", help="Ignore cache, re-fetch everything")
    p.add_argument("--skip-snow", action="store_true", help="Skip the slow ERA5 snowfall step")
    p.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia scraping")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

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
    print("  olympics_participants_wikipedia.csv — Wikipedia: 2018-2026")
    print("  olympics_medals_wikipedia.csv       — Wikipedia: 2018-2026")
    print("  snowfall_raw.csv                    — all countries, snowfall per year")

if __name__ == "__main__":
    main()