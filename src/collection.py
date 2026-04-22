"""
World Bank Data Pipeline — GDP + Population
============================================
Fetches 4 indicators for all countries from 1992 onwards.

Indicators:
  NY.GDP.MKTP.CD   → gdp_usd
  NY.GDP.PCAP.CD   → gdp_per_capita_usd
  SP.POP.TOTL      → population_total
  SP.POP.GROW      → population_growth_pct

Output:
  raw/<indicator>.parquet        — cached API responses
  clean/worldbank_cleaned.parquet
  clean/worldbank_cleaned.csv

Usage:
  python worldbank.py                        # fetch + clean
  python worldbank.py --refresh              # force re-fetch
  python worldbank.py --country BEL --year 2022
  python worldbank.py --top 10 --year 2022 --sort gdp_usd
  python worldbank.py --query                # interactive REPL
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDICATORS: dict[str, str] = {
    "gdp_usd":                "NY.GDP.MKTP.CD",
    "gdp_per_capita_usd":     "NY.GDP.PCAP.CD",
    "population_total":       "SP.POP.TOTL",
    "population_growth_pct":  "SP.POP.GROW",
}
START_YEAR = 1992
RAW_DIR    = Path("raw")
CLEAN_DIR  = Path("clean")
CLEAN_FILE = CLEAN_DIR / "worldbank_cleaned.parquet"
CLEAN_CSV  = CLEAN_DIR / "worldbank_cleaned.csv"

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

SORT_CHOICES = list(INDICATORS.keys())

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_indicator(col_name: str, code: str) -> pd.DataFrame:
    """Fetch one indicator from the World Bank API, using cache if available."""
    raw_path = RAW_DIR / f"{code.replace('.', '_')}.parquet"
    if raw_path.exists():
        print(f"  [cache] {col_name}")
        return pd.read_parquet(raw_path).rename(columns={"value": col_name})

    url    = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "per_page": 1000, "date": f"{START_YEAR}:2024"}
    rows, page, total_pages = [], 1, None

    while True:
        data = requests.get(url, params={**params, "page": page}, timeout=30).json()
        meta, records = data[0], data[1] or []
        if total_pages is None:
            total_pages = meta["pages"]
            print(f"  [fetch] {col_name} ({code}): {meta['total']} records, {total_pages} pages")

        for r in records:
            if r:
                rows.append({
                    "iso3":         r.get("countryiso3code") or r.get("country", {}).get("id", ""),
                    "country_name": r.get("country", {}).get("value", ""),
                    "year":         int(r["date"]),
                    "value":        r.get("value"),
                })

        print(f"    page {page}/{total_pages}", end="\r")
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.1)

    print()
    df = pd.DataFrame(rows)
    RAW_DIR.mkdir(exist_ok=True)
    df.to_parquet(raw_path, index=False)
    return df.rename(columns={"value": col_name})

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def is_aggregate(iso3: str, name: str) -> bool:
    if not iso3 or len(iso3) != 3:
        return True
    if iso3[:2] in AGGREGATE_PREFIXES or iso3 in AGGREGATE_PREFIXES:
        return True
    return any(kw in name.lower() for kw in AGGREGATE_KEYWORDS)

def build_clean() -> pd.DataFrame:
    frames = [fetch_indicator(col, code) for col, code in INDICATORS.items()]

    # Merge all indicators on (iso3, country_name, year)
    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame.drop(columns="country_name"), on=["iso3", "year"], how="outer") \
               .merge(frame[["iso3", "year", "country_name"]], on=["iso3", "year"], how="left",
                      suffixes=("", "_r"))
        # keep first non-null country_name
        if "country_name_r" in df.columns:
            df["country_name"] = df["country_name"].combine_first(df.pop("country_name_r"))

    # Filter aggregates
    mask = df.apply(lambda r: is_aggregate(r["iso3"], str(r["country_name"])), axis=1)
    df   = df[~mask].copy()

    # Types
    df["year"]                 = df["year"].astype("Int64")
    df["population_total"]     = pd.to_numeric(df["population_total"],    errors="coerce").astype("Int64")
    df["population_growth_pct"]= pd.to_numeric(df["population_growth_pct"], errors="coerce")
    df["gdp_usd"]              = pd.to_numeric(df["gdp_usd"],              errors="coerce")
    df["gdp_per_capita_usd"]   = pd.to_numeric(df["gdp_per_capita_usd"],   errors="coerce")
    df["iso3"]                 = df["iso3"].str.upper().str.strip()
    df["country_name"]         = df["country_name"].str.strip()

    df = (df[df["year"] >= START_YEAR]
            .drop_duplicates(subset=["iso3", "year"], keep="last")
            .sort_values(["iso3", "year"])
            .reset_index(drop=True)
          [["iso3", "country_name", "year",
            "gdp_usd", "gdp_per_capita_usd",
            "population_total", "population_growth_pct"]])

    CLEAN_DIR.mkdir(exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[CLEAN] {len(df):,} rows · {df['iso3'].nunique()} countries · "
          f"{df['year'].min()}–{df['year'].max()} → {CLEAN_FILE}")
    return df

def load_clean() -> pd.DataFrame:
    if not CLEAN_FILE.exists():
        raise FileNotFoundError("Run without --query first to fetch data.")
    return pd.read_parquet(CLEAN_FILE)

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(df, country=None, year=None, top_n=None,
          sort_by="gdp_usd", ascending=False) -> pd.DataFrame:
    if country:
        df = df[df["iso3"].str.upper().eq(country.upper()) |
                df["country_name"].str.contains(country, case=False, na=False)]
    if year:
        df = df[df["year"] == year]
    df = df.sort_values(sort_by, ascending=ascending)
    return df.head(top_n).reset_index(drop=True) if top_n else df.reset_index(drop=True)

def interactive_query(df: pd.DataFrame) -> None:
    cols = ", ".join(SORT_CHOICES)
    print(f"\n=== World Bank Query REPL ===")
    print(f"Commands: country <ISO3|name> | year <YYYY> | top <N> | sort <{cols}> | show | reset | exit\n")
    filters: dict = {}
    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else None
        if   cmd == "exit":                      break
        elif cmd == "reset":                     filters.clear(); print("Filters cleared.")
        elif cmd == "show":
            r = query(df, **filters)
            print(r.to_string(index=False)); print(f"\n{len(r)} rows\n")
        elif cmd == "country" and arg:           filters["country"] = arg
        elif cmd == "year"    and arg:           filters["year"]    = int(arg)
        elif cmd == "top"     and arg:           filters["top_n"]   = int(arg)
        elif cmd == "sort"    and arg:           filters["sort_by"] = arg
        else:                                    print(f"Unknown: {raw}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="World Bank GDP + Population pipeline")
    p.add_argument("--query",   action="store_true")
    p.add_argument("--country", help="ISO-3 or name fragment")
    p.add_argument("--year",    type=int)
    p.add_argument("--top",     type=int)
    p.add_argument("--sort",    default="gdp_usd", choices=SORT_CHOICES)
    p.add_argument("--refresh", action="store_true", help="Clear cache and re-fetch")
    args = p.parse_args()

    if args.refresh:
        for f in RAW_DIR.glob("*.parquet"): f.unlink()
        print("[CACHE] Cleared.")

    df = build_clean() if not CLEAN_FILE.exists() or args.refresh else load_clean()
    if not args.refresh and not args.query:
        print(f"[LOAD] {len(df):,} rows from {CLEAN_FILE}")

    if args.country or args.year or args.top:
        r = query(df, country=args.country, year=args.year, top_n=args.top, sort_by=args.sort)
        print(r.to_string(index=False)); print(f"\n{len(r)} rows")

    if args.query:
        interactive_query(df)

if __name__ == "__main__":
    main()