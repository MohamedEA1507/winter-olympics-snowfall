"""
Snowfall by Country — Optimized Pipeline
==========================================
Extracts TOTAL annual snowfall per country from ERA5 monthly NetCDF data.
Total = sum of all grid cells weighted by cell area (cos latitude correction).

Output columns:
  snowfall_km3  — total volume of snowfall in km³ water equivalent
  snowfall_mm   — average depth in mm water equivalent (for reference)

Output:
  clean/snowfall_by_country.parquet
  clean/snowfall_by_country.csv

Usage:
  python snowfall_by_country.py
"""

import time
import pandas as pd
import numpy as np
import xarray as xr
import regionmask
import pycountry
from pathlib import Path

NC_FILE     = r"E:\ae0a85222114609e694b93e2ea4d1e41\data_stream-moda.nc"
CLEAN_DIR   = Path("clean")
OUT_PARQUET = CLEAN_DIR / "snowfall_by_country.parquet"
OUT_CSV     = CLEAN_DIR / "snowfall_by_country.csv"

EARTH_RADIUS_KM = 6371.0
CELL_DEG        = 0.1   # grid resolution in degrees

# ---------------------------------------------------------------------------
# ISO-3 mapping
# ---------------------------------------------------------------------------

REGIONMASK_TO_ISO3 = {
    "A":   "AUT",  "N":   "NOR",  "S":   "SWE",  "J":   "JPN",
    "CH":  "CHE",  "IS":  "ISL",  "IL":  "ISR",  "CL":  "CHL",
    "CA":  "CAN",  "US":  "USA",  "AL":  "ALB",  "ME":  "MNE",
    "KO":  "XKX",  "SLO": "SVN",  "SK":  "SVK",  "NM":  "MKD",
    "BG":  "BGR",  "TR":  "TUR",  "GE":  "GEO",  "GL":  "GRL",
    "AF":  "AFG",  "NP":  "NPL",  "BT":  "BTN",  "TJ":  "TJK",
    "KG":  "KGZ",  "TF":  "ATF",  "CZ":  "CZE",  "BiH": "BIH",
}

def to_iso3(abbrev: str, name: str) -> str:
    if abbrev in REGIONMASK_TO_ISO3:
        return REGIONMASK_TO_ISO3[abbrev]
    if len(abbrev) == 3:
        return abbrev.upper()
    try:
        match = pycountry.countries.search_fuzzy(name)
        if match:
            return match[0].alpha_3
    except Exception:
        pass
    return abbrev.upper()

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def progress(step, total, label, start_time):
    pct     = step / total
    done    = int(40 * pct)
    bar     = "█" * done + "░" * (40 - done)
    elapsed = time.time() - start_time
    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
    print(f"\r[{bar}] {pct*100:5.1f}%  {label}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
          end="", flush=True)

# ---------------------------------------------------------------------------
# Cell area weights
# ---------------------------------------------------------------------------

def compute_cell_area_km2(latitudes: np.ndarray) -> xr.DataArray:
    """
    Compute the area in km² for each grid cell.

    A cell at latitude φ with resolution Δ degrees has area:
      area = (R · Δlat_rad) × (R · cos(φ) · Δlon_rad)
           = R² · Δlat_rad · Δlon_rad · cos(φ)

    Since Δlat = Δlon = 0.1°, this simplifies to a 1D array over latitude.
    """
    delta_rad = np.deg2rad(CELL_DEG)
    lat_rad   = np.deg2rad(latitudes)
    area_km2  = (EARTH_RADIUS_KM ** 2) * (delta_rad ** 2) * np.cos(lat_rad)
    return xr.DataArray(area_km2, coords={"latitude": latitudes}, dims=["latitude"])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # ── 1. Open dataset and rechunk cleanly
    print("[1/5] Opening dataset…")
    ds = xr.open_dataset(NC_FILE)
    sf = ds["sf"]  # (valid_time, latitude, longitude) in metres water equivalent
    print(f"      {len(sf.valid_time)} timesteps · {len(sf.latitude)} lats · {len(sf.longitude)} lons")
    sf = sf.chunk({"valid_time": 12, "latitude": -1, "longitude": -1})

    # ── 2. Resample monthly → annual sum (lazy)
    print("[2/5] Resampling monthly → annual totals…")
    sf_yearly = sf.resample(valid_time="YE").sum()   # still in metres w.e.
    print(f"      {len(sf_yearly.valid_time)} yearly slices")

    # ── 3. Compute cell areas and apply
    print("[3/5] Computing cell areas…")
    cell_area = compute_cell_area_km2(sf.latitude.values)  # (latitude,) in km²

    # snowfall_vol [km³ w.e.] = snowfall [m] × area [km²] × (1km / 1000m)
    # = snowfall_m × area_km2 / 1000
    sf_volume = sf_yearly * cell_area / 1000.0  # (valid_time, lat, lon) in km³ w.e.
    print(f"      Cell area at equator:  {cell_area.values[900]:.2f} km²")
    print(f"      Cell area at 60°N:     {cell_area.values[300]:.2f} km²")
    print(f"      Cell area at 80°N:     {cell_area.values[100]:.2f} km²")

    # ── 4. Build country mask
    print("[4/5] Building country mask…")
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(sf.longitude, sf.latitude)
    print(f"      {len(countries)} countries")

    # ── 5. Single-pass groupby sum — total snowfall volume per country per year
    print("[5/5] Computing totals (this is the slow step)…")
    t_compute = time.time()
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            # sum gives total volume; mean gives average depth (keep both)
            result_sum  = sf_volume.groupby(mask).sum().compute()   # km³ w.e.
            result_mean = sf_yearly.groupby(mask).mean().compute()  # m w.e.
    except ImportError:
        result_sum  = sf_volume.groupby(mask).sum().compute()
        result_mean = sf_yearly.groupby(mask).mean().compute()
    print(f"      Done in {time.time() - t_compute:.0f}s")

    # ── 6. Build tidy DataFrame
    print("Building DataFrame…")
    country_ids = result_sum.coords["mask"].values.astype(int)
    valid_ids   = country_ids[country_ids >= 0]
    years       = result_sum.coords["valid_time"].dt.year.values
    t_df        = time.time()

    rows = []
    for i, idx in enumerate(valid_ids):
        region = countries[int(idx)]
        iso3   = to_iso3(region.abbrev, region.name)
        progress(i + 1, len(valid_ids), f"{region.name[:25]:<25}", t_df)

        vol_values  = result_sum.sel(mask=idx).values   # km³ per year
        mean_values = result_mean.sel(mask=idx).values  # m w.e. per year

        for year, vol, mean in zip(years, vol_values, mean_values):
            rows.append({
                "iso3":          iso3,
                "country_name":  region.name,
                "year":          int(year),
                "snowfall_km3":  float(vol)  if not np.isnan(vol)  else None,
                "snowfall_mm":   float(mean) * 1000 if not np.isnan(mean) else None,
            })
    print()

    df = pd.DataFrame(rows)
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nDone in {time.time() - t0:.0f}s — {len(df):,} rows · "
          f"{df['iso3'].nunique()} countries · {df['year'].min()}–{df['year'].max()}")
    print(f"Saved → {OUT_PARQUET}  and  {OUT_CSV}")


if __name__ == "__main__":
    main()