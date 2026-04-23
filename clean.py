"""
Cleaning Pipeline
==================
Reads raw data from data/raw/, cleans each dataset, handles missing values,
standardises numeric columns, and produces:

Separate clean files:
  data/clean/olympics.csv       — all participants + medals, iso3 mapped
  data/clean/gdp.csv            — GDP for olympic countries, 1992–2024
  data/clean/population.csv     — Population for olympic countries, 1992–2024
  data/clean/snowfall.csv       — Snowfall for olympic countries, 1992–2024

Master file:
  data/clean/master.csv         — all four datasets merged + standardised columns

Standardisation strategy:
  gdp_usd, population_total     → log10 transform (spans many orders of magnitude)
  gdp_per_capita_usd            → log10 + min-max (useful for cross-country comparison)
  snowfall_km3, snowfall_mm,
  n_athletes, total_medals      → min-max scaling (0–1)

Missing value strategy:
  IOA (Individual Olympic Athletes) → dropped (not a real country)
  Small territories / microstates   → kept as NULL (structurally missing,
                                       imputing would be misleading)

Usage:
  python clean.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")

OLYMPIC_YEARS = [1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]

AGGREGATE_PREFIXES = {
    "XJ","XL","XM","XN","XO","XP","XQ","XR","XS","XU",
    "XY","ZB","ZF","ZG","ZH","ZI","ZJ","ZQ","ZT","ZW","OE","EU",
    "V1","V2","V3","V4",
}
AGGREGATE_KEYWORDS = {
    "world","income","oecd","euro","region","blend","ida","ibrd","ifc","miga",
    "small states","fragile","heavily indebted","least developed","sub-saharan",
    "latin america","east asia","south asia","middle east","north africa",
    "pacific","caribbean","central europe","balkans","developing","emerging",
    "advanced economies",
}

NOC_TO_ISO3 = {
    "GER": "DEU",  "NED": "NLD",  "SUI": "CHE",  "DEN": "DNK",
    "POR": "PRT",  "SLO": "SVN",  "CRO": "HRV",  "LAT": "LVA",
    "MGL": "MNG",  "PHI": "PHL",  "TRI": "TTO",  "ZIM": "ZWE",
    "IRI": "IRN",  "MAS": "MYS",  "UAE": "ARE",  "VIE": "VNM",
    "TPE": "TWN",  "GRE": "GRC",  "BUL": "BGR",  "HAI": "HTI",
    "CHI": "CHL",  "RSA": "ZAF",  "ALG": "DZA",  "PAR": "PRY",
    "URU": "URY",  "CRC": "CRI",  "GUA": "GTM",  "HON": "HND",
    "ESA": "SLV",  "NCA": "NIC",  "BAR": "BRB",  "SKN": "KNA",
    "TGA": "TON",  "ANT": "ATG",  "ISV": "VIR",  "PUR": "PRI",
    "GUM": "GUM",  "ASA": "ASM",  "HKG": "HKG",  "ANG": "AGO",
    # Defunct → successor
    "URS": "RUS",  "EUN": "RUS",  "GDR": "DEU",  "FRG": "DEU",
    "TCH": "CZE",  "YUG": "SRB",  "SCG": "SRB",
}

# Rows to drop — not real countries
DROP_NOC = {"IOA"}  # Individual Olympic Athletes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_aggregate(iso3: str, name: str) -> bool:
    if not iso3 or len(iso3) != 3:
        return True
    if iso3[:2] in AGGREGATE_PREFIXES or iso3 in AGGREGATE_PREFIXES:
        return True
    return any(kw in str(name).lower() for kw in AGGREGATE_KEYWORDS)

def noc_to_iso3(noc: str) -> str:
    return NOC_TO_ISO3.get(noc, noc)

def minmax(s: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1], ignoring NaN."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)

def log10_transform(s: pd.Series) -> pd.Series:
    """Log10 transform — values <= 0 become NaN."""
    return np.log10(s.where(s > 0))

def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  [saved] {len(df):,} rows · {df.shape[1]} cols → {path}")

def missing_report(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  {label}: no missing values")
    else:
        for col, cnt in missing.items():
            print(f"  {label} — {col}: {cnt} NULL ({cnt/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 1 — Clean Olympics
# ---------------------------------------------------------------------------

def clean_olympics() -> pd.DataFrame:
    print("\n[1/5] Cleaning Olympics…")

    participants = pd.read_csv(RAW_DIR / "olympics_participants.csv")
    medals       = pd.read_csv(RAW_DIR / "olympics_medals.csv")

    # Drop non-countries
    participants = participants[~participants["noc_code"].isin(DROP_NOC)]
    medals       = medals[~medals["noc_code"].isin(DROP_NOC)]

    # Map NOC → ISO3
    participants["iso3"] = participants["noc_code"].map(noc_to_iso3)
    medals["iso3"]       = medals["noc_code"].map(noc_to_iso3)

    # Aggregate defunct NOCs that map to same ISO3 + year
    participants = (
        participants
        .groupby(["iso3", "year"], as_index=False)
        .agg(
            noc_code  =("noc_code",  "first"),
            team_name =("team_name", "first"),
            n_athletes=("n_athletes","sum"),
            host_flag =("host_flag", "max"),
        )
    )
    medals = (
        medals
        .groupby(["iso3", "year"], as_index=False)
        .agg(gold=("gold","sum"), silver=("silver","sum"),
             bronze=("bronze","sum"), total_medals=("total_medals","sum"))
    )

    df = participants.merge(medals, on=["iso3","year"], how="left")
    df[["gold","silver","bronze","total_medals"]] = (
        df[["gold","silver","bronze","total_medals"]].fillna(0).astype(int)
    )

    df = df.sort_values(["year","iso3"]).reset_index(drop=True)
    df = df[["iso3","noc_code","team_name","year",
             "n_athletes","host_flag","gold","silver","bronze","total_medals"]]

    missing_report(df, "olympics")
    save(df, CLEAN_DIR / "olympics.csv")
    print(f"  {df['iso3'].nunique()} countries · {df['year'].nunique()} editions")
    return df

# ---------------------------------------------------------------------------
# Step 2 — Clean GDP
# ---------------------------------------------------------------------------

def clean_gdp(olympic_iso3: set) -> pd.DataFrame:
    print("\n[2/5] Cleaning GDP…")

    df = pd.read_csv(RAW_DIR / "worldbank_gdp.csv")

    # Drop aggregates
    before = len(df)
    df = df[~df.apply(lambda r: is_aggregate(r["iso3"], r["country_name"]), axis=1)]
    print(f"  Dropped {before - len(df):,} aggregate rows")

    # Keep only olympic countries
    df = df[df["iso3"].isin(olympic_iso3)]

    # Types + year range
    df["year"]               = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gdp_usd"]            = pd.to_numeric(df["gdp_usd"], errors="coerce")
    df["gdp_per_capita_usd"] = pd.to_numeric(df["gdp_per_capita_usd"], errors="coerce")
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]
    df = df.drop_duplicates(subset=["iso3","year"], keep="last")
    df = df.sort_values(["iso3","year"]).reset_index(drop=True)
    df = df[["iso3","country_name","year","gdp_usd","gdp_per_capita_usd"]]

    missing_report(df, "gdp")
    # Note: NULLs are kept intentionally — structurally missing for small
    # territories (Monaco, Bermuda, Taiwan) and early transition years

    save(df, CLEAN_DIR / "gdp.csv")
    print(f"  {df['iso3'].nunique()} countries · {df['year'].min()}–{df['year'].max()}")
    return df

# ---------------------------------------------------------------------------
# Step 3 — Clean Population
# ---------------------------------------------------------------------------

def clean_population(olympic_iso3: set) -> pd.DataFrame:
    print("\n[3/5] Cleaning Population…")

    df = pd.read_csv(RAW_DIR / "worldbank_population.csv")

    before = len(df)
    df = df[~df.apply(lambda r: is_aggregate(r["iso3"], r["country_name"]), axis=1)]
    print(f"  Dropped {before - len(df):,} aggregate rows")

    df = df[df["iso3"].isin(olympic_iso3)]
    df["year"]             = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["population_total"] = pd.to_numeric(df["population_total"], errors="coerce").astype("Int64")
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]
    df = df.drop_duplicates(subset=["iso3","year"], keep="last")
    df = df.sort_values(["iso3","year"]).reset_index(drop=True)
    df = df[["iso3","country_name","year","population_total"]]

    missing_report(df, "population")
    save(df, CLEAN_DIR / "population.csv")
    print(f"  {df['iso3'].nunique()} countries · {df['year'].min()}–{df['year'].max()}")
    return df

# ---------------------------------------------------------------------------
# Step 4 — Clean Snowfall
# ---------------------------------------------------------------------------

def clean_snowfall(olympic_iso3: set) -> pd.DataFrame:
    print("\n[4/5] Cleaning Snowfall…")

    df = pd.read_csv(RAW_DIR / "snowfall_raw.csv")

    before = len(df)
    df = df[df["iso3"].isin(olympic_iso3)]
    print(f"  Kept {df['iso3'].nunique()} olympic countries ({before - len(df):,} rows dropped)")

    df["year"]         = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["snowfall_km3"] = pd.to_numeric(df["snowfall_km3"], errors="coerce")
    df["snowfall_mm"]  = pd.to_numeric(df["snowfall_mm"],  errors="coerce")
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]
    df = df.drop_duplicates(subset=["iso3","year"], keep="last")
    df = df.sort_values(["iso3","year"]).reset_index(drop=True)
    df = df[["iso3","country_name","year","snowfall_km3","snowfall_mm"]]

    missing_report(df, "snowfall")
    # Note: NULLs are kept intentionally — small island nations and microstates
    # fall between ERA5 grid cells (0.1° resolution ~11km), imputing would be invented data
    save(df, CLEAN_DIR / "snowfall.csv")
    print(f"  {df['iso3'].nunique()} countries · {df['year'].min()}–{df['year'].max()}")
    return df

# ---------------------------------------------------------------------------
# Step 5 — Build Master + Standardise
# ---------------------------------------------------------------------------

def build_master(olympics: pd.DataFrame, gdp: pd.DataFrame,
                 population: pd.DataFrame, snowfall: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/5] Building master dataset + engineered features…")

    # ── Merge ──
    master = olympics.copy()
    master = master.merge(gdp[["iso3","year","gdp_usd","gdp_per_capita_usd"]],
                          on=["iso3","year"], how="left")
    master = master.merge(population[["iso3","year","population_total"]],
                          on=["iso3","year"], how="left")
    master = master.merge(snowfall[["iso3","year","snowfall_km3","snowfall_mm"]],
                          on=["iso3","year"], how="left")

    # ── Rename to clean output names ──
    master = master.rename(columns={
        "iso3":            "country",
        "gdp_usd":         "gdp",
        "gdp_per_capita_usd": "gdp_per_capita",
        "population_total":   "population",
        "snowfall_km3":    "snowfall_total",
        "snowfall_mm":     "snowfall_mean_gridcell",
    })

    # ── Select and order core columns ──
    master = master[[
        "country", "noc_code", "team_name", "year",
        "n_athletes", "host_flag",
        "gold", "silver", "bronze", "total_medals",
        "snowfall_total", "snowfall_mean_gridcell",
        "gdp", "gdp_per_capita", "population",
    ]]

    master = master.sort_values(["year","country"]).reset_index(drop=True)

    # ── Missing value report ──
    print("\n  Missing values in master:")
    total = len(master)
    for col in ["gdp","gdp_per_capita","population","snowfall_total","snowfall_mean_gridcell"]:
        n = master[col].isna().sum()
        if n:
            countries = master[master[col].isna()]["country"].unique()
            print(f"    {col}: {n} NULL ({n/total*100:.1f}%) "
                  f"— {len(countries)} countries: {', '.join(sorted(countries))}")

    # ── Engineered features ──
    print("\n  Adding engineered features…")

    # Binary: did this country win any medal at this edition?
    master["won_any_medal"] = (master["total_medals"] > 0).astype(int)

    # Log(1 + total_medals) — handles 0s, compresses large values
    master["log_total_medals"] = np.log1p(master["total_medals"])

    # Log GDP and population — both span many orders of magnitude
    master["log_gdp"]        = np.log10(master["gdp"].where(master["gdp"] > 0))
    master["log_population"] = np.log10(master["population"].where(master["population"] > 0))

    # Medals per million inhabitants — normalises for country size
    master["medals_per_million"] = (
        master["total_medals"] / master["population"] * 1e6
    ).where(master["population"].notna())

    print("  won_any_medal      = 1 if total_medals > 0 else 0")
    print("  log_total_medals   = log(1 + total_medals)")
    print("  log_gdp            = log10(gdp)")
    print("  log_population     = log10(population)")
    print("  medals_per_million = total_medals / population * 1e6")

    save(master, CLEAN_DIR / "master.csv")
    print(f"\n  {master['country'].nunique()} countries · "
          f"{master['year'].nunique()} editions · "
          f"{len(master):,} rows total")
    return master

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    olympics     = clean_olympics()
    olympic_iso3 = set(olympics["iso3"].unique())
    print(f"\n  → {len(olympic_iso3)} unique ISO3 codes from olympics data")

    gdp        = clean_gdp(olympic_iso3)
    population = clean_population(olympic_iso3)
    snowfall   = clean_snowfall(olympic_iso3)

    build_master(olympics, gdp, population, snowfall)

    print("\n✓ Cleaning complete → data/clean/")
    print("  olympics.csv    — participants + medals per country per edition")
    print("  gdp.csv         — GDP + GDP per capita per country per year")
    print("  population.csv  — population per country per year")
    print("  snowfall.csv    — snowfall per country per year")
    print("  master.csv      — all merged + standardised columns")

if __name__ == "__main__":
    main()