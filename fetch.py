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
from pathlib import Path

import numpy as np
import pandas as pd
import regionmask
import requests
import xarray as xr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    """Fetch a single World Bank indicator for ALL countries, no filtering."""
    url    = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "per_page": 1000, "date": f"{START_YEAR}:2024"}
    rows, page, total_pages = [], 1, None

    while True:
        data = requests.get(url, params={**params, "page": page}, timeout=30).json()
        meta, records = data[0], data[1] or []
        if total_pages is None:
            total_pages = meta["pages"]
            print(f"    {code}: {meta['total']} records, {total_pages} pages")

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
        time.sleep(0.1)

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

    print("\n[WORLD BANK — GDP] Fetching all countries…")
    frames = {}
    for col_name, code in GDP_INDICATORS.items():
        df = fetch_wb_indicator(code).rename(columns={"value": col_name})
        frames[col_name] = df

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

    print("  Building DataFrame…")
    country_ids = result_sum.coords["mask"].values.astype(int)
    valid_ids   = country_ids[country_ids >= 0]
    years       = result_sum.coords["valid_time"].dt.year.values
    t_df        = time.time()

    rows = []
    for i, idx in enumerate(valid_ids):
        region = countries[int(idx)]
        iso3   = to_iso3(region.abbrev, region.name)
        progress(i + 1, len(valid_ids), f"{region.name[:25]:<25}", t_df)
        for year, vol, mean in zip(years,
                                   result_sum.sel(mask=idx).values,
                                   result_mean.sel(mask=idx).values):
            rows.append({
                "iso3":         iso3,
                "country_name": region.name,
                "year":         int(year),
                "snowfall_km3": float(vol)  if not np.isnan(vol)  else None,
                "snowfall_mm":  float(mean) * 1000 if not np.isnan(mean) else None,
            })
    print()

    save(pd.DataFrame(rows), out_path)
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
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fetch all raw data — no filtering")
    p.add_argument("--refresh",   action="store_true", help="Ignore cache, re-fetch everything")
    p.add_argument("--skip-snow", action="store_true", help="Skip the slow ERA5 snowfall step")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetch_gdp(refresh=args.refresh)
    fetch_population(refresh=args.refresh)
    fetch_olympics(refresh=args.refresh)

    if not args.skip_snow:
        fetch_snowfall(refresh=args.refresh)
    else:
        print("\n[SNOWFALL] Skipped (--skip-snow)")

    print("\n✓ All raw data collected → data/raw/")
    print("  worldbank_gdp.csv         — all countries, gdp_usd + gdp_per_capita_usd")
    print("  worldbank_population.csv  — all countries, population_total")
    print("  olympics_participants.csv — all Winter Olympic participants per edition")
    print("  olympics_medals.csv       — all participants, medals = 0 if none won")
    print("  snowfall_raw.csv          — all countries, snowfall per year")

if __name__ == "__main__":
    main()