"""
PURPOSE
-------
This script implements the CLEAN and INTEGRATE stages of the project's data pipeline.

In a data engineering pipeline, raw data should never be modified.
Instead, this script reads from data/raw/ and writes clean, validated, integrated data to data/clean/.
This separation is a core principle of data governance (Lecture 2): raw data is the immutable source of truth,
cleaned data is derived and always reproducible.

PIPELINE STAGE OVERVIEW
------------------------
    Ingest (fetch.py)
        ↓ data/raw/  ← raw, unmodified
    Clean (this file)
        ↓ data/clean/  ← validated, typed, joined
    Analyse (analyse.py)

DATA GOVERNANCE PRINCIPLES APPLIED (Lecture 2)
------------------------------------------------
1. Data lineage: every transformation is documented in comments
   so the journey from raw to clean is fully traceable.
2. Data quality: we check accuracy, completeness, and consistency
   at each step with explicit missing value reports.
3. Master data management: a single canonical ISO3 country code is
   used as the spine that joins every data source. The NOC→ISO3
   crosswalk creates one trusted "golden record" per country.
4. Data minimisation: we keep only the columns and rows needed for
   the analysis. Aggregate rows (e.g. "World", "High income") that
   the World Bank API returns are explicitly filtered out.

WHAT THIS FILE PRODUCES
------------------------
Individual cleaned files (one per source):
  data/clean/olympics.csv    — participants + medals per country per edition
  data/clean/gdp.csv         — GDP + GDP per capita, Olympic countries only
  data/clean/population.csv  — Population, Olympic countries only
  data/clean/snowfall.csv    — ERA5 snowfall, Olympic countries only

Master analytical table:
  data/clean/master.csv      — all four sources merged, with engineered
                                features ready for modelling in analyse.py
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")

# Olympic years covered by this project. Starting at 1992 avoids most Cold-War-era NOC complexity:
# no USSR (dissolved 1991), no East/West Germany (reunified 1990).
# The 1992 Unified Team (EUN) still needs mapping — handled below.
OLYMPIC_YEARS = [1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

# =============================================================================
# 2. REFERENCE DATA: COUNTRY IDENTIFIER CROSSWALKS
# =============================================================================

# --- World Bank aggregate row filter -------------------------------------------
# The World Bank API returns aggregate entries alongside country rows — things like "World", "High income", "OECD members", "Euro area".
# These are NOT countries and must be excluded before any analysis.
# We identify them by their ISO3 prefix (aggregates use codes starting
# with X or Z, which are not assigned to any real country by ISO 3166) and by keywords in their name.

# This is an example of data quality practice from Lecture 2:
# knowing where data physically comes from and what artefacts the source
# introduces. The World Bank API explicitly mixes country and aggregate
# entries with no flag to distinguish them — we must filter manually.

AGGREGATE_ISO3_PREFIXES = {
    # World Bank aggregate codes — not ISO 3166 country codes
    "XJ", "XL", "XM", "XN", "XO", "XP", "XQ", "XR", "XS", "XU",
    "XY", "ZB", "ZF", "ZG", "ZH", "ZI", "ZJ", "ZQ", "ZT", "ZW",
    # Other non-country codes used by World Bank
    "OE",  # OECD members
    "EU",  # European Union
    "V1", "V2", "V3", "V4",  # income-group aggregates
}

AGGREGATE_NAME_KEYWORDS = {
    # These keywords in a country_name field indicate an aggregate, not a country
    "world", "income", "oecd", "euro", "region", "blend",
    "ida", "ibrd", "ifc", "miga",           # World Bank lending groups
    "small states", "fragile", "heavily indebted",
    "least developed", "sub-saharan",
    "latin america", "east asia", "south asia",
    "middle east", "north africa",
    "pacific", "caribbean", "central europe",
    "balkans", "developing", "emerging",
    "advanced economies",
}

# --- NOC → ISO3 crosswalk -------------------------------------------------------
# Olympic datasets use IOC National Olympic Committee (NOC) codes, which differ from ISO 3166-1 alpha-3 codes used by the World Bank and ERA5.
# Entity resolution — mapping between identifier systems — is one of the classic hard problems in data engineering (blueprint Section 9).

# Two categories of mapping:
#   1. Simple substitutions: IOC chose a different 3-letter code than ISO
#      (e.g. GER→DEU, NED→NLD, SUI→CHE)
#   2. Defunct states: countries that no longer exist and must be mapped
#      to a modern successor for joining with current economic data.
#      Each mapping decision is a deliberate analytical choice that must
#      be documented (blueprint Section 9, "Country-name changes"):
#        URS (Soviet Union) → RUS (Russia):  Russia inherited most Soviet
#            Olympic infrastructure and is the recognized successor state
#            for international purposes.
#        EUN (Unified Team 1992) → RUS:  The 1992 Unified Team consisted
#            of former Soviet republics competing together one last time.
#            We assign to RUS to maintain continuity with URS records.
#        GDR (East Germany) → DEU:  East Germany merged into unified
#            Germany in 1990. All GDR athletes/medals attributed to DEU.
#        FRG (West Germany) → DEU:  Same reasoning as GDR.
#        TCH (Czechoslovakia) → CZE:  The Czech Republic is treated as
#            the primary successor for Winter Olympics purposes (Slovakia
#            competed separately from 1994 onward).
#        YUG (Yugoslavia) → SRB:  Serbia is the recognized successor
#            state to Yugoslavia and Serbia-Montenegro.
#        SCG (Serbia-Montenegro 2006) → SRB:  Montenegro declared
#            independence in 2006. We attribute to Serbia.

NOC_TO_ISO3 = {
    # --- Simple code differences (same country, different code system) ---
    "GER": "DEU",   # Germany: IOC uses GER, ISO uses DEU
    "NED": "NLD",   # Netherlands: IOC uses NED, ISO uses NLD
    "SUI": "CHE",   # Switzerland: IOC uses SUI, ISO uses CHE
    "DEN": "DNK",   # Denmark: IOC uses DEN, ISO uses DNK
    "POR": "PRT",   # Portugal
    "SLO": "SVN",   # Slovenia: IOC uses SLO, ISO uses SVN
    "CRO": "HRV",   # Croatia
    "LAT": "LVA",   # Latvia
    "MGL": "MNG",   # Mongolia
    "PHI": "PHL",   # Philippines
    "TRI": "TTO",   # Trinidad and Tobago
    "ZIM": "ZWE",   # Zimbabwe
    "IRI": "IRN",   # Iran
    "MAS": "MYS",   # Malaysia
    "UAE": "ARE",   # United Arab Emirates
    "VIE": "VNM",   # Vietnam
    "TPE": "TWN",   # Chinese Taipei (Taiwan) — politically sensitive;
                    # IOC uses TPE, ISO uses TWN
    "GRE": "GRC",   # Greece
    "BUL": "BGR",   # Bulgaria
    "HAI": "HTI",   # Haiti
    "CHI": "CHL",   # Chile: IOC uses CHI, ISO uses CHL
    "RSA": "ZAF",   # South Africa
    "ALG": "DZA",   # Algeria
    "PAR": "PRY",   # Paraguay
    "URU": "URY",   # Uruguay
    "CRC": "CRI",   # Costa Rica
    "GUA": "GTM",   # Guatemala
    "HON": "HND",   # Honduras
    "ESA": "SLV",   # El Salvador
    "NCA": "NIC",   # Nicaragua
    "BAR": "BRB",   # Barbados
    "SKN": "KNA",   # Saint Kitts and Nevis
    "TGA": "TON",   # Tonga
    "ANT": "ATG",   # Antigua and Barbuda
    "ISV": "VIR",   # US Virgin Islands
    "PUR": "PRI",   # Puerto Rico (US territory, competes separately)
    "GUM": "GUM",   # Guam (no ISO3; kept as-is)
    "ASA": "ASM",   # American Samoa
    "HKG": "HKG",   # Hong Kong (special administrative region)
    "ANG": "AGO",   # Angola
    # --- Defunct states → modern successors ---
    "URS": "RUS",   # Soviet Union → Russia (see note above)
    "EUN": "RUS",   # Unified Team 1992 → Russia
    "GDR": "DEU",   # East Germany → unified Germany
    "FRG": "DEU",   # West Germany → unified Germany
    "TCH": "CZE",   # Czechoslovakia → Czech Republic
    "YUG": "SRB",   # Yugoslavia → Serbia
    "SCG": "SRB",   # Serbia-Montenegro → Serbia
}

# NOC codes that represent non-country entities and should be dropped entirely.
# IOA = Individual Olympic Athletes (stateless athletes competing without
# a national team). They have no ISO3 code and cannot be joined to any
# economic or climate data.
DROP_NOC = {"IOA"}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def is_aggregate(iso3: str, name: str) -> bool:
    """
    Return True if this row represents a World Bank aggregate, not a country.

    The World Bank API mixes country rows with regional/income-group aggregates.
    We identify aggregates by:
      1. Non-standard ISO3 prefix (X* and Z* codes are not assigned to countries)
      2. Keywords in the country name that indicate a group, not a country

    This function is applied row-by-row during GDP and population cleaning.
    """
    if not iso3 or len(iso3) != 3:
        return True   # malformed code — treat as aggregate
    if iso3[:2] in AGGREGATE_ISO3_PREFIXES or iso3 in AGGREGATE_ISO3_PREFIXES:
        return True
    return any(kw in str(name).lower() for kw in AGGREGATE_NAME_KEYWORDS)


def noc_to_iso3(noc: str) -> str:
    """
    Map an IOC NOC code to an ISO 3166-1 alpha-3 code.

    If the NOC code is not in the crosswalk, return it unchanged.
    This handles NOC codes that already match ISO3 (e.g. USA, CAN, NOR).
    """
    return NOC_TO_ISO3.get(noc, noc)


def log10_transform(s: pd.Series) -> pd.Series:
    """
    Apply log base-10 transform to a numeric series.

    Values <= 0 become NaN (log is undefined for non-positive numbers).
    We use log10 rather than natural log because:
      - GDP in USD spans ~10 orders of magnitude (10^9 to 10^13)
      - log10 values have intuitive interpretations: log10(1e9) = 9
      - Coefficients in regression represent effects per order of magnitude

    Note: this is a feature engineering transform, not data cleaning.
    It is applied in build_master() after merging, not during source cleaning.
    Keeping raw columns intact preserves data lineage.
    """
    return np.log10(s.where(s > 0))


def save(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV and print a confirmation with row/column counts."""
    df.to_csv(path, index=False)
    print(f"  [saved] {len(df):,} rows · {df.shape[1]} cols → {path}")


def missing_report(df: pd.DataFrame, label: str) -> None:
    """
    Print a missing value report for a DataFrame.

    Transparency about missing data is a core data quality principle
    (Lecture 2). We report counts and percentages so the reader can
    judge whether missingness is structurally expected (e.g. ERA5 doesn't
    cover tiny island nations) or a data quality problem.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  {label}: no missing values ✓")
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"  {label} — {col}: {cnt:,} NULL ({pct:.1f}%)")


# =============================================================================
# 4. STEP 1 — CLEAN OLYMPICS DATA
# =============================================================================

def load_participants() -> pd.DataFrame:
    """
    Load and combine participant lists from two sources:
      - Kaggle athlete_events.csv (1992–2014, processed by fetch.py)
      - Wikipedia scraped data (2018–2026, scraped by fetch.py)

    Why two sources?
        The Kaggle dataset ("120 years of Olympic history") only covers
        up to the 2016 Games. Wikipedia provides more recent editions.
        This is a common pattern in data engineering: combining a reliable
        historical dataset with a more current but less structured source.

    Data lineage: raw files → this function → clean_olympics()
    """
    # ── Kaggle (1992–2014) ────────────────────────────────────────────────
    kaggle = pd.read_csv(RAW_DIR / "olympics_participants.csv")
    kaggle_years = sorted(kaggle["year"].unique())
    print(f"  Kaggle participants : {len(kaggle):,} rows | years {kaggle_years}")

    # ── Wikipedia (2018–2026) ─────────────────────────────────────────────
    wiki_path = RAW_DIR / "olympics_participants_wikipedia.csv"
    if wiki_path.exists():
        wiki = pd.read_csv(wiki_path)
        # Wikipedia data may not include athlete counts — fill with 0
        # (unknown, not zero — we acknowledge this is imputed)
        if "n_athletes" not in wiki.columns:
            wiki["n_athletes"] = 0
        if "team_name" not in wiki.columns:
            wiki["team_name"] = wiki["noc_code"]
        wiki_years = sorted(wiki["year"].unique())
        print(f"  Wikipedia participants: {len(wiki):,} rows | years {wiki_years}")
    else:
        print("  [WARNING] olympics_participants_wikipedia.csv not found")
        print("            Run fetch.py first to scrape Wikipedia data")
        wiki = pd.DataFrame(columns=kaggle.columns)

    # ── Combine ────────────────────────────────────────────────────────────
    # The two sources cover non-overlapping year ranges, so a simple
    # concatenation is safe. We drop_duplicates as a defensive measure
    # in case the year ranges overlap in future (idempotency principle).
    combined = pd.concat([kaggle, wiki], ignore_index=True)
    combined = combined.drop_duplicates(subset=["noc_code", "year"], keep="first")
    return combined


def load_medals() -> pd.DataFrame:
    """
    Load and combine medal tables from Kaggle (1992–2014) and Wikipedia (2018–2026).

    Medal data structure: one row per (noc_code, year) with gold/silver/bronze/total.
    Countries that participated but won zero medals are explicitly included with
    medal counts of 0 — this is critical for regression analysis where the
    zero-medal cases are the majority of the data.
    """
    # ── Kaggle (1992–2014) ────────────────────────────────────────────────
    kaggle = pd.read_csv(RAW_DIR / "olympics_medals.csv")
    print(f"  Kaggle medals       : {len(kaggle):,} rows | "
          f"years {sorted(kaggle['year'].unique())}")

    # ── Wikipedia (2018–2026) ─────────────────────────────────────────────
    wiki_path = RAW_DIR / "olympics_medals_wikipedia.csv"
    if wiki_path.exists():
        wiki = pd.read_csv(wiki_path)
        print(f"  Wikipedia medals    : {len(wiki):,} rows | "
              f"years {sorted(wiki['year'].unique())}")
    else:
        print("  [WARNING] olympics_medals_wikipedia.csv not found")
        wiki = pd.DataFrame(columns=kaggle.columns)

    combined = pd.concat([kaggle, wiki], ignore_index=True)
    combined = combined.drop_duplicates(subset=["noc_code", "year"], keep="first")
    return combined


def clean_olympics() -> pd.DataFrame:
    """
    Clean and integrate Olympic participants and medal data.

    CLEANING STEPS (in order):
    1. Load raw data from both sources
    2. Drop non-country entities (IOA)
    3. Map NOC codes → ISO3 (entity resolution)
    4. Aggregate rows that now share the same ISO3 + year
       (e.g. GDR and FRG both map to DEU — we sum their athletes/medals)
    5. Left-join medals onto participants
       (ensures every participating country has a row, even if medals = 0)
    6. Sort and select final columns

    Why left-join participants + medals (not inner join)?
        Inner join would silently drop countries that participated but
        won no medals. Since zero-medal countries are ~70% of our data
        and are essential for regression analysis, we must preserve them.
        The left join keeps all participants; non-winners get medal = 0.

    Output schema:
        iso3, noc_code, team_name, year, n_athletes,
        gold, silver, bronze, total_medals
    """
    print("\n" + "="*60)
    print("[STEP 1/5] Cleaning Olympics data")
    print("  Sources: Kaggle (1992–2014) + Wikipedia (2018–2026)")
    print("="*60)

    participants = load_participants()
    medals       = load_medals()

    # ── Step 2: Drop non-countries ─────────────────────────────────────────
    # IOA = Individual Olympic Athletes (stateless athletes).
    # They have no ISO3 code and cannot be matched to economic data.
    n_before = len(participants)
    participants = participants[~participants["noc_code"].isin(DROP_NOC)]
    medals       = medals[~medals["noc_code"].isin(DROP_NOC)]
    print(f"\n  Dropped {n_before - len(participants)} IOA rows")

    # ── Step 3: Map NOC → ISO3 (entity resolution) ─────────────────────────
    # After this mapping, defunct state NOCs (URS, EUN, GDR, etc.)
    # will share the same ISO3 as their successors. For example,
    # URS 1992 → RUS, EUN 1992 → RUS. We aggregate in step 4.
    participants["iso3"] = participants["noc_code"].map(noc_to_iso3)
    medals["iso3"]       = medals["noc_code"].map(noc_to_iso3)

    # ── Step 4: Aggregate defunct-state rows ────────────────────────────────
    # After NOC→ISO3 mapping, multiple rows may share the same (iso3, year).
    # Example: if both URS and EUN records existed for 1992, both now have
    # iso3=RUS for year=1992. We aggregate by summing athletes and medals.
    participants = (
        participants
        .groupby(["iso3", "year"], as_index=False)
        .agg(
            noc_code  = ("noc_code",   "first"),   # keep one representative NOC
            team_name = ("team_name",  "first"),
            n_athletes= ("n_athletes", "sum"),      # sum athletes across merged teams
        )
    )
    medals = (
        medals
        .groupby(["iso3", "year"], as_index=False)
        .agg(
            gold         = ("gold",         "sum"),
            silver       = ("silver",       "sum"),
            bronze       = ("bronze",       "sum"),
            total_medals = ("total_medals", "sum"),
        )
    )

    # ── Step 5: Join participants + medals ─────────────────────────────────
    # Left join: all participants kept, medals filled with 0 where missing.
    # Countries that participated but won nothing appear with medal columns = 0.
    df = participants.merge(medals, on=["iso3", "year"], how="left")
    df[["gold", "silver", "bronze", "total_medals"]] = (
        df[["gold", "silver", "bronze", "total_medals"]]
        .fillna(0)
        .astype(int)
    )

    # ── Step 6: Final sort and column selection ────────────────────────────
    df = df.sort_values(["year", "iso3"]).reset_index(drop=True)
    df = df[[
        "iso3", "noc_code", "team_name", "year",
        "n_athletes", "gold", "silver", "bronze", "total_medals",
    ]]

    missing_report(df, "olympics")
    save(df, CLEAN_DIR / "olympics.csv")
    print(f"  {df['iso3'].nunique()} unique countries | "
          f"{df['year'].nunique()} editions | "
          f"zero-medal rows: {(df['total_medals'] == 0).sum():,}")
    return df


# =============================================================================
# 5. STEP 2 — CLEAN GDP DATA
# =============================================================================

def clean_gdp(olympic_iso3: set) -> pd.DataFrame:
    """
    Clean World Bank GDP data.

    RAW DATA SOURCE:
        World Bank Open Data API, indicators:
          NY.GDP.MKTP.CD  → total GDP in current USD
          NY.GDP.PCAP.CD  → GDP per capita in current USD
        Fetched by fetch.py using pagination through the JSON API.

    DATA QUALITY ISSUES ADDRESSED:
    1. Aggregate rows: the World Bank API returns regional/income-group
       aggregates alongside country data. Filtered using is_aggregate().
    2. Scope filtering: we keep only countries that appear in our
       Olympics dataset (olympic_iso3). The World Bank covers ~215
       entities; we only need the ~120 that sent Olympic teams.
    3. Type coercion: the API returns all values as strings or mixed
       types. We cast explicitly to numeric with errors="coerce" so
       unparseable values become NaN rather than crashing.
    4. Deduplication: in case of duplicate (iso3, year) rows (e.g.
       from multiple API pages), we keep the last occurrence.

    MISSING VALUES STRATEGY:
        NULLs are kept intentionally. Small territories (Monaco, Bermuda,
        Taiwan) are not covered by the World Bank. Early transition years
        (1992–1995) are missing for some post-Soviet states.
        Imputing these with regional averages would be misleading because
        these are structurally absent cases, not random missingness.

    Output columns: iso3, country_name, year, gdp_usd, gdp_per_capita_usd
    """
    print("\n" + "="*60)
    print("[STEP 2/5] Cleaning GDP data (World Bank API)")
    print("="*60)

    df = pd.read_csv(RAW_DIR / "worldbank_gdp.csv")
    print(f"  Raw rows: {len(df):,}")

    # ── Filter out World Bank aggregate rows ───────────────────────────────
    before = len(df)
    df = df[~df.apply(
        lambda r: is_aggregate(r["iso3"], r["country_name"]), axis=1
    )]
    print(f"  Dropped {before - len(df):,} aggregate rows "
          f"(World Bank regional/income groups)")

    # ── Keep only Olympic participant countries ────────────────────────────
    # This reduces the dataset from ~215 World Bank entities to the ~120
    # countries that actually appear in our Olympics data.
    before = len(df)
    df = df[df["iso3"].isin(olympic_iso3)]
    print(f"  Filtered to Olympic countries: {before - len(df):,} rows dropped")

    # ── Type coercion ──────────────────────────────────────────────────────
    df["year"]               = pd.to_numeric(df["year"],               errors="coerce").astype("Int64")
    df["gdp_usd"]            = pd.to_numeric(df["gdp_usd"],            errors="coerce")
    df["gdp_per_capita_usd"] = pd.to_numeric(df["gdp_per_capita_usd"], errors="coerce")

    # ── Year range filter ─────────────────────────────────────────────────
    # Keep 1992–2024 to cover all Olympic editions in our scope.
    # The World Bank API was queried with this range in fetch.py,
    # but we filter explicitly for robustness.
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]

    # ── Deduplication ─────────────────────────────────────────────────────
    df = df.drop_duplicates(subset=["iso3", "year"], keep="last")

    # ── Sort and select final columns ─────────────────────────────────────
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df = df[["iso3", "country_name", "year", "gdp_usd", "gdp_per_capita_usd"]]

    missing_report(df, "gdp")
    save(df, CLEAN_DIR / "gdp.csv")
    print(f"  {df['iso3'].nunique()} countries | "
          f"{df['year'].min()}–{df['year'].max()}")
    return df


# =============================================================================
# 6. STEP 3 — CLEAN POPULATION DATA
# =============================================================================

def clean_population(olympic_iso3: set) -> pd.DataFrame:
    """
    Clean World Bank population data (indicator SP.POP.TOTL).

    WHY WE NEED POPULATION:
        Larger countries have a bigger athlete talent pool. Without
        controlling for population, we might mistake "Norway wins more
        medals than India" for a climate effect when it is partly a
        scale effect. We also use population to compute medals_per_million,
        which normalises medal counts for country size (hypothesis H3).

    CLEANING STEPS are identical to clean_gdp():
    1. Filter aggregate rows
    2. Filter to Olympic countries
    3. Cast types explicitly
    4. Filter year range
    5. Deduplicate
    6. Sort and select

    MISSING VALUES:
        Same strategy as GDP — NULLs are preserved. Small territories
        (Hong Kong, Bermuda) may be absent or merged into their
        parent country in the World Bank data.

    Output columns: iso3, country_name, year, population_total
    """
    print("\n" + "="*60)
    print("[STEP 3/5] Cleaning Population data (World Bank API)")
    print("="*60)

    df = pd.read_csv(RAW_DIR / "worldbank_population.csv")
    print(f"  Raw rows: {len(df):,}")

    # ── Filter aggregates ──────────────────────────────────────────────────
    before = len(df)
    df = df[~df.apply(
        lambda r: is_aggregate(r["iso3"], r["country_name"]), axis=1
    )]
    print(f"  Dropped {before - len(df):,} aggregate rows")

    # ── Filter to Olympic countries ────────────────────────────────────────
    df = df[df["iso3"].isin(olympic_iso3)]

    # ── Type coercion ──────────────────────────────────────────────────────
    df["year"]             = pd.to_numeric(df["year"],             errors="coerce").astype("Int64")
    df["population_total"] = pd.to_numeric(df["population_total"], errors="coerce").astype("Int64")

    # ── Year range, deduplication, sort ───────────────────────────────────
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]
    df = df.drop_duplicates(subset=["iso3", "year"], keep="last")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df = df[["iso3", "country_name", "year", "population_total"]]

    missing_report(df, "population")
    save(df, CLEAN_DIR / "population.csv")
    print(f"  {df['iso3'].nunique()} countries | "
          f"{df['year'].min()}–{df['year'].max()}")
    return df


# =============================================================================
# 7. STEP 4 — CLEAN SNOWFALL DATA
# =============================================================================

def clean_snowfall(olympic_iso3: set) -> pd.DataFrame:
    """
    Clean ERA5 snowfall data produced by fetch.py's snowfall pipeline.

    DATA SOURCE:
        ERA5 reanalysis from the Copernicus Climate Data Store.
        fetch.py processes a NetCDF file (ERA5 monthly data) and
        aggregates grid-cell-level snowfall to country level using
        the regionmask library and country polygon shapefiles.

    TWO SNOWFALL METRICS:
        snowfall_km3  — total snowfall volume in km³ water equivalent,
                        summed across all ERA5 grid cells within the
                        country boundary. Larger countries have larger
                        totals even if they are not very snowy.
        snowfall_mm   — mean snowfall depth in mm per ERA5 grid cell
                        within the country. This is the mean over
                        grid cells, so it is size-normalised and more
                        comparable across countries of different areas.
                        This is our primary climate variable in modelling.

    WHY snowfall_mm IS THE MAIN PREDICTOR:
        A country like Russia has enormous snowfall_km3 simply because
        it is huge, even in its least snowy regions. The mean gridcell
        depth (snowfall_mm) captures whether the country is generally
        snowy, not just whether it is large. This is the correct
        construct for our research question.

    MISSING VALUES:
        ERA5 uses a 0.1° grid (~11km resolution). Very small islands
        and microstates may fall between grid cells or be mostly ocean
        within their bounding box. Their snowfall values are NaN and
        we keep them as NaN — imputing snowfall for Malta or Bermuda
        would be inventing data, which would violate data quality
        principles from Lecture 2.

    Output columns: iso3, country_name, year, snowfall_km3, snowfall_mm
    """
    print("\n" + "="*60)
    print("[STEP 4/5] Cleaning Snowfall data (ERA5 reanalysis)")
    print("="*60)

    df = pd.read_csv(RAW_DIR / "snowfall_raw.csv")
    print(f"  Raw rows: {len(df):,} | {df['iso3'].nunique()} countries")

    # ── Filter to Olympic countries ────────────────────────────────────────
    # ERA5 covers all land areas globally. We keep only the countries
    # in our Olympic dataset — no need for the full global coverage.
    before = len(df)
    df = df[df["iso3"].isin(olympic_iso3)]
    print(f"  Filtered to Olympic countries: {before - len(df):,} rows dropped")
    print(f"  Kept {df['iso3'].nunique()} Olympic countries")

    # ── Type coercion ──────────────────────────────────────────────────────
    df["year"]         = pd.to_numeric(df["year"],         errors="coerce").astype("Int64")
    df["snowfall_km3"] = pd.to_numeric(df["snowfall_km3"], errors="coerce")
    df["snowfall_mm"]  = pd.to_numeric(df["snowfall_mm"],  errors="coerce")

    # ── Year range, deduplication, sort ───────────────────────────────────
    df = df[(df["year"] >= 1992) & (df["year"] <= 2024)]
    df = df.drop_duplicates(subset=["iso3", "year"], keep="last")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df = df[["iso3", "country_name", "year", "snowfall_km3", "snowfall_mm"]]

    # ── Missing value report ───────────────────────────────────────────────
    missing_report(df, "snowfall")
    print("  NOTE: NULLs are structurally expected for small island nations")
    print("        whose territory falls between ERA5 grid cells (0.1° ≈ 11km)")

    save(df, CLEAN_DIR / "snowfall.csv")
    print(f"  {df['iso3'].nunique()} countries | "
          f"{df['year'].min()}–{df['year'].max()}")
    return df


# =============================================================================
# 8. STEP 5 — BUILD MASTER ANALYTICAL TABLE
# =============================================================================

def build_master(
    olympics:   pd.DataFrame,
    gdp:        pd.DataFrame,
    population: pd.DataFrame,
    snowfall:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all four cleaned datasets into one wide analytical table.

    This implements the INTEGRATE stage of the pipeline:
        olympics (spine)
            LEFT JOIN gdp        ON iso3, year
            LEFT JOIN population ON iso3, year
            LEFT JOIN snowfall   ON iso3, year

    WHY LEFT JOINS (not inner joins)?
        Every country that ever participated in Winter Olympics from
        1992–2026 should appear in the master table, even if we lack
        GDP, population, or snowfall data for them. Using inner joins
        would silently discard these countries, biasing the analysis
        toward countries with complete data (typically wealthier,
        better-documented nations). Left joins preserve all participants
        and make missingness visible and documented.

    The Olympics table is the SPINE (the left side of all joins) because:
      - It defines the population of interest (Olympic participants)
      - It has the cleanest, most complete coverage
      - The key (iso3, year) is unique within it after deduplication

    WHY MATCH ON YEAR (not just ISO3)?
        GDP and population vary significantly across years. Norway's GDP
        in 1992 is very different from 2022. Matching on both iso3 AND
        year ensures each country-edition gets its contemporary economic
        context, not a time-averaged figure.

    ENGINEERED FEATURES ADDED:
        These are computed here (not in analyse.py) so they are available
        in the saved master.csv for exploration notebooks and other scripts.
        All transformations are documented below with their rationale.

    Output: one row per (country, Olympic year). Key = (iso3, year).

    COLUMN GLOSSARY:
        country              — ISO3 country code (the analytical key)
        noc_code             — IOC National Olympic Committee code
        team_name            — Team name as used in competition
        year                 — Olympic year (1992, 1994, ..., 2026)
        n_athletes           — number of athletes sent by this country
        gold/silver/bronze   — medal counts by type
        total_medals         — sum of gold + silver + bronze
        snowfall_total       — total snowfall volume in km³ (size-dependent)
        snowfall_mean_gridcell — mean mm per ERA5 grid cell (size-normalised)
        gdp                  — total GDP in current USD
        gdp_per_capita       — GDP per person in current USD
        population           — total population
        won_any_medal        — 1 if total_medals > 0, else 0
        log_total_medals     — log(1 + total_medals)
        log_gdp              — log10(gdp)
        log_population       — log10(population)
        medals_per_million   — total_medals / population * 1,000,000
    """
    print("\n" + "="*60)
    print("[STEP 5/5] Building master analytical table")
    print("="*60)

    # ── Merge all sources onto the Olympics spine ──────────────────────────
    master = olympics.copy()

    master = master.merge(
        gdp[["iso3", "year", "gdp_usd", "gdp_per_capita_usd"]],
        on=["iso3", "year"],
        how="left",    # keep all Olympic rows; NaN where World Bank data absent
    )
    master = master.merge(
        population[["iso3", "year", "population_total"]],
        on=["iso3", "year"],
        how="left",
    )
    master = master.merge(
        snowfall[["iso3", "year", "snowfall_km3", "snowfall_mm"]],
        on=["iso3", "year"],
        how="left",
    )

    # ── Rename columns to final clean names ────────────────────────────────
    # These names are used consistently in analyse.py and graph_analytics.py.
    # Renaming here (not in the source cleaners) maintains clarity about
    # what each column represents in the merged context.
    master = master.rename(columns={
        "iso3":               "country",
        "gdp_usd":            "gdp",
        "gdp_per_capita_usd": "gdp_per_capita",
        "population_total":   "population",
        "snowfall_km3":       "snowfall_total",
        "snowfall_mm":        "snowfall_mean_gridcell",
    })

    # ── Select and order core columns ──────────────────────────────────────
    core_cols = [
        "country", "noc_code", "team_name", "year",
        "n_athletes", "gold", "silver", "bronze", "total_medals",
        "snowfall_total", "snowfall_mean_gridcell",
        "gdp", "gdp_per_capita", "population",
    ]
    master = master[core_cols]
    master = master.sort_values(["year", "country"]).reset_index(drop=True)

    # ── Missing value report ───────────────────────────────────────────────
    # This is the most important data quality report in the pipeline.
    # It shows which countries are missing which data sources, and why.
    # This directly feeds into the "Missingness" section of the report.
    print("\n  Missing values in master table:")
    print(f"  Total rows: {len(master):,}")
    for col in ["gdp", "gdp_per_capita", "population",
                "snowfall_total", "snowfall_mean_gridcell"]:
        n = master[col].isna().sum()
        if n:
            countries = sorted(master[master[col].isna()]["country"].unique())
            print(f"\n  {col}:")
            print(f"    {n:,} NULL rows ({n/len(master)*100:.1f}%)")
            print(f"    Affected countries ({len(countries)}): "
                  f"{', '.join(countries[:10])}"
                  f"{'...' if len(countries) > 10 else ''}")

    # ── Feature engineering ────────────────────────────────────────────────
    # These are derived columns added on top of the raw joined data.
    # They are needed by analyse.py for modelling. Computing them here
    # rather than in analyse.py means they are available in the saved
    # CSV for exploration without running the full analysis pipeline.
    print("\n  Computing engineered features...")

    # won_any_medal
    # Binary flag: did this country win at least one medal at this edition?
    # Used as the dependent variable in Model G (logistic regression).
    master["won_any_medal"] = (master["total_medals"] > 0).astype(int)

    # log_total_medals
    # log(1 + total_medals). The +1 handles the zero-medal case gracefully
    # since log(0) is undefined. For large medal counts, log1p(x) ≈ log(x).
    # Used as the dependent variable in OLS Models A–D.
    master["log_total_medals"] = np.log1p(master["total_medals"])

    # log_gdp and log_population
    # Log base-10 transforms. GDP spans ~10 orders of magnitude (1e9 to 1e13)
    # and population spans ~4 (1e5 to 1e9). In log-space, a one-unit increase
    # represents a 10× increase in the original scale, making coefficients
    # interpretable as effects per order of magnitude.
    # Values ≤ 0 are set to NaN (log is undefined there).
    master["log_gdp"]        = log10_transform(master["gdp"])
    master["log_population"] = log10_transform(master["population"])

    # medals_per_million
    # Normalises medal counts by country population.
    # Tests hypothesis H3: does the snow effect survive normalisation for size?
    # Norway (5M people, 300+ medals) looks very different from USA (330M, 400+)
    # on this metric — it reveals which countries over-perform relative to size.
    master["medals_per_million"] = (
        master["total_medals"] / master["population"] * 1_000_000
    ).where(master["population"].notna())

    # ── Summary of engineered features ────────────────────────────────────
    print("  won_any_medal      = 1 if total_medals > 0 else 0")
    print("  log_total_medals   = log(1 + total_medals)  [handles zero-medal rows]")
    print("  log_gdp            = log10(gdp)              [compresses scale]")
    print("  log_population     = log10(population)       [compresses scale]")
    print("  medals_per_million = total_medals / population * 1e6  [size-normalised]")

    # ── Save ──────────────────────────────────────────────────────────────
    save(master, CLEAN_DIR / "master.csv")

    # ── Final summary ─────────────────────────────────────────────────────
    complete = master.dropna(subset=[
        "snowfall_mean_gridcell", "log_gdp", "log_population"
    ])
    print(f"\n  Master table summary:")
    print(f"  Total rows             : {len(master):,}")
    print(f"  Rows with complete data: {len(complete):,} "
          f"({len(complete)/len(master)*100:.1f}%)")
    print(f"  Unique countries       : {master['country'].nunique()}")
    print(f"  Olympic editions       : {master['year'].nunique()} "
          f"({master['year'].min()}–{master['year'].max()})")
    print(f"  Countries with medals  : "
          f"{master[master['total_medals']>0]['country'].nunique()}")
    return master


# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    """
    Run the full cleaning pipeline in sequence.

    Pipeline order matters:
    1. clean_olympics() first — it defines the set of Olympic countries
       (olympic_iso3) that all subsequent cleaners filter to.
    2. clean_gdp(), clean_population(), clean_snowfall() — all filtered
       to Olympic countries for efficiency and scope control.
    3. build_master() — joins all four cleaned datasets into the
       analytical table consumed by analyse.py and graph_analytics.py.

    This is an idempotent pipeline: running it multiple times produces
    the same output. Re-running after updating raw data refreshes all
    downstream clean files automatically.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  CLEANING PIPELINE")
    print("  Winter Olympics & Snowfall Project")
    print("="*60)

    # Step 1: Olympics — defines the scope (which countries and years)
    olympics     = clean_olympics()
    olympic_iso3 = set(olympics["iso3"].unique())
    print(f"\n  → {len(olympic_iso3)} unique ISO3 country codes "
          f"from Olympics data (used to filter all other sources)")

    # Steps 2–4: Economic and climate data, filtered to Olympic scope
    gdp        = clean_gdp(olympic_iso3)
    population = clean_population(olympic_iso3)
    snowfall   = clean_snowfall(olympic_iso3)

    # Step 5: Build the master analytical table
    build_master(olympics, gdp, population, snowfall)

    # ── Pipeline completion summary ────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✓ Cleaning pipeline complete")
    print("="*60)
    print(f"  Output directory: {CLEAN_DIR}/")
    print()
    print("  olympics.csv           — participants + medals per country per edition")
    print("    key: (iso3, year)    | source: Kaggle + Wikipedia")
    print()
    print("  gdp.csv                — GDP + GDP per capita, Olympic countries only")
    print("    key: (iso3, year)    | source: World Bank API")
    print()
    print("  population.csv         — population, Olympic countries only")
    print("    key: (iso3, year)    | source: World Bank API")
    print()
    print("  snowfall.csv           — ERA5 snowfall, Olympic countries only")
    print("    key: (iso3, year)    | source: ERA5 reanalysis (NetCDF)")
    print()
    print("  master.csv             — all four sources merged + engineered features")
    print("    key: (country, year) | consumed by: analyse.py, graph_analytics.py")
    print()
    print("  Next step: python analyse.py")


if __name__ == "__main__":
    main()