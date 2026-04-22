import pandas as pd

# ==============================
# CONFIG
# ==============================

INPUT_PATH = "data/raw/athlete_events.csv"
OUTPUT_PATH = "data/processed/preprocessedOlympics.csv"

START_YEAR = 1992
END_YEAR = 2026

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_PATH)

print(f"Original shape: {df.shape}")

# ==============================
# FILTER: Winter Olympics + years
# ==============================

df = df[df["Season"] == "Winter"]
df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)]

print(f"After filtering Winter + years: {df.shape}")

# ==============================
# KEEP ONLY MEDALS
# ==============================

df = df[df["Medal"].notna()]

print(f"After keeping medals only: {df.shape}")

# ==============================
# CREATE MEDAL COLUMNS
# ==============================

df["gold"] = (df["Medal"] == "Gold").astype(int)
df["silver"] = (df["Medal"] == "Silver").astype(int)
df["bronze"] = (df["Medal"] == "Bronze").astype(int)

# ==============================
# CREATE NOC → TEAM MAPPING
# ==============================

# neem meest voorkomende teamnaam per NOC
noc_team_map = (
    df.groupby("NOC")["Team"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"Team": "team"})
)

# ==============================
# AGGREGATE TO COUNTRY-YEAR
# ==============================

grouped = (
    df.groupby(["NOC", "Year"])
    .agg({
        "gold": "sum",
        "silver": "sum",
        "bronze": "sum"
    })
    .reset_index()
)

grouped["total_medals"] = (
    grouped["gold"] + grouped["silver"] + grouped["bronze"]
)

# ==============================
# RENAME COLUMNS
# ==============================

grouped = grouped.rename(columns={
    "NOC": "noc_code",
    "Year": "edition_year"
})

# ==============================
# ADD TEAM NAME BACK
# ==============================

grouped = grouped.merge(
    noc_team_map,
    left_on="noc_code",
    right_on="NOC",
    how="left"
).drop(columns=["NOC"])

# ==============================
# OPTIONAL: REMOVE HISTORICAL/PROBLEM NOCs
# ==============================

problem_nocs = ["URS", "EUN", "TCH", "YUG", "FRG", "GDR"]

grouped = grouped[~grouped["noc_code"].isin(problem_nocs)]

# ==============================
# ADD ISO3
# ==============================

grouped["iso3"] = grouped["noc_code"]

# ==============================
# ADD HOST FLAG
# ==============================

hosts = {
    1992: "FRA",
    1994: "NOR",
    1998: "JPN",
    2002: "USA",
    2006: "ITA",
    2010: "CAN",
    2014: "RUS",
    2018: "KOR",
    2022: "CHN",
    2026: "ITA"
}

grouped["host_flag"] = grouped.apply(
    lambda row: 1 if hosts.get(row["edition_year"]) == row["noc_code"] else 0,
    axis=1
)

# ==============================
# DATA QUALITY CHECKS
# ==============================

duplicates = grouped.duplicated(["iso3", "edition_year"]).sum()
assert duplicates == 0, f"Found {duplicates} duplicate rows!"

assert (
    grouped["total_medals"]
    == grouped["gold"] + grouped["silver"] + grouped["bronze"]
).all(), "Medal totals mismatch!"

assert grouped["iso3"].notna().all(), "Missing ISO3 values!"

print("Data quality checks passed.")

# ==============================
# SORT & SAVE
# ==============================

grouped = grouped.sort_values(["edition_year", "iso3"])

grouped.to_csv(OUTPUT_PATH, index=False)

print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
print(f"Final shape: {grouped.shape}")