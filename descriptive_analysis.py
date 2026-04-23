"""
Descriptive Analysis Pipeline
================================
Reads data/clean/master.csv and produces all descriptive plots + summary tables.

Plots saved to data/figures/:
  01_hist_total_medals.png
  02_hist_snowfall.png
  03_scatter_snowfall_vs_medals.png
  04_scatter_gdp_vs_medals.png
  05_scatter_population_vs_medals.png
  06_scatter_snowfall_km3_vs_medals.png
  07_correlation_heatmap.png
  08_top10_medals_vs_snow.png
  09_medals_by_snowfall_quartile.png

Usage:
  python analyse.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_DIR  = Path("data/clean")
FIG_DIR    = Path("data/figures")
MASTER     = CLEAN_DIR / "master.csv"

# Consistent style
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#BA7517"
GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

OUTLIER_LABELS = {"NOR", "CAN", "USA", "DEU", "FIN", "SWE", "AUT", "CHE",
                  "RUS", "ISL", "TJK", "KGZ", "ITA", "JPN"}

def label_outliers(ax, df, xcol, ycol, threshold_pct=0.85):
    """Label the top points by y-value or notable countries."""
    top = df.nlargest(6, ycol)
    notable = df[df["country"].isin(OUTLIER_LABELS)]
    to_label = pd.concat([top, notable]).drop_duplicates("country")
    for _, row in to_label.iterrows():
        ax.annotate(
            row["country"],
            xy=(row[xcol], row[ycol]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#555555",
        )

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER)
    # Engineered columns (in case running before clean.py adds them)
    if "won_any_medal" not in df.columns:
        df["won_any_medal"]      = (df["total_medals"] > 0).astype(int)
    if "log_total_medals" not in df.columns:
        df["log_total_medals"]   = np.log1p(df["total_medals"])
    if "log_gdp" not in df.columns:
        df["log_gdp"]            = np.log10(df["gdp"].where(df["gdp"] > 0))
    if "log_population" not in df.columns:
        df["log_population"]     = np.log10(df["population"].where(df["population"] > 0))
    if "medals_per_million" not in df.columns:
        df["medals_per_million"] = df["total_medals"] / df["population"] * 1e6
    # Rename snowfall columns if they still have old names
    df = df.rename(columns={
        "snowfall_km3":  "snowfall_total",
        "snowfall_mm":   "snowfall_mean_gridcell",
        "gdp_usd":       "gdp",
        "gdp_per_capita_usd": "gdp_per_capita",
        "population_total":   "population",
        "iso3":          "country",
    }, errors="ignore")
    print(f"[LOAD] {len(df):,} rows · {df['country'].nunique()} countries")
    return df

# ---------------------------------------------------------------------------
# 1 & 2. Histograms
# ---------------------------------------------------------------------------

def plot_hist_medals(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = list(range(0, 20)) + [20, 25, 30, 40, 50, 70, 100]
    ax.hist(df["total_medals"], bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Total medals")
    ax.set_ylabel("Country-editions")
    ax.set_title("Distribution of total medals — heavily right-skewed with 68% zeros")
    zero_pct = (df["total_medals"] == 0).mean() * 100
    ax.text(0.98, 0.95, f"{zero_pct:.0f}% of rows = 0",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=CORAL)
    save(fig, "01_hist_total_medals.png")

def plot_hist_snowfall(df):
    sf = df["snowfall_mean_gridcell"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sf, bins=25, color=TEAL, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean snowfall per grid cell (mm water equivalent)")
    ax.set_ylabel("Country-editions")
    ax.set_title("Distribution of snowfall — right-skewed, many tropical/island nations near zero")
    save(fig, "02_hist_snowfall.png")

# ---------------------------------------------------------------------------
# 3-6. Scatterplots
# ---------------------------------------------------------------------------

def _scatter(df, xcol, ycol, color, xlabel, title, log_x=False, figname=""):
    sub = df[[xcol, ycol, "country"]].dropna()
    if log_x:
        sub = sub[sub[xcol] > 0].copy()
        sub["_x"] = np.log10(sub[xcol])
        x_vals = sub["_x"]
        x_label = f"log₁₀({xlabel})"
    else:
        x_vals = sub[xcol]
        x_label = xlabel

    corr = sub[xcol].corr(sub[ycol])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, sub[ycol], color=color, alpha=0.55, s=35, linewidths=0)

    # Regression line
    m, b = np.polyfit(x_vals, sub[ycol], 1)
    xs = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax.plot(xs, m * xs + b, color=color, linewidth=1.5, alpha=0.7, linestyle="--")

    # Label notable points
    top = sub.nlargest(5, ycol)
    notable = sub[sub["country"].isin(OUTLIER_LABELS)]
    to_label = pd.concat([top, notable]).drop_duplicates("country")
    xplot = to_label["_x"] if log_x else to_label[xcol]
    for _, row in to_label.iterrows():
        xv = np.log10(row[xcol]) if log_x else row[xcol]
        ax.annotate(row["country"], xy=(xv, row[ycol]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color="#444444")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Total medals")
    ax.set_title(f"{title}  (r = {corr:.2f})")
    save(fig, figname)

def plot_scatters(df):
    _scatter(df, "snowfall_mean_gridcell", "total_medals", BLUE,
             "snowfall_mm", "Snowfall (mean depth) vs medals", figname="03_scatter_snowfall_vs_medals.png")
    _scatter(df, "gdp", "total_medals", CORAL,
             "GDP (USD)", "GDP vs medals", log_x=True, figname="04_scatter_gdp_vs_medals.png")
    _scatter(df, "population", "total_medals", TEAL,
             "Population", "Population vs medals", log_x=True, figname="05_scatter_population_vs_medals.png")
    _scatter(df, "snowfall_total", "total_medals", PURPLE,
             "snowfall_km3", "Snowfall (total volume km³) vs medals", figname="06_scatter_snowfall_km3_vs_medals.png")

# ---------------------------------------------------------------------------
# 7. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df):
    cols = {
        "total_medals":          "total_medals",
        "snowfall_total":        "snowfall_km3",
        "snowfall_mean_gridcell":"snowfall_mm",
        "gdp":                   "gdp",
        "gdp_per_capita":        "gdp_per_capita",
        "population":            "population",
        "n_athletes":            "n_athletes",
    }
    sub = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        square=True, ax=ax, annot_kws={"size": 10},
    )
    ax.set_title("Correlation matrix — numeric variables")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "07_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 8. Top 10 medals vs top 10 snowy
# ---------------------------------------------------------------------------

def plot_top10(df):
    top_medal = (df.groupby("country")["total_medals"]
                 .sum().sort_values(ascending=False).head(10))
    top_snow  = (df.groupby("country")["snowfall_mean_gridcell"]
                 .mean().sort_values(ascending=False).head(10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # medals bar
    ax1.barh(top_medal.index[::-1], top_medal.values[::-1], color=BLUE, height=0.6)
    ax1.set_xlabel("Total medals (all editions)")
    ax1.set_title("Top 10 by total medals")
    for i, (v, c) in enumerate(zip(top_medal.values[::-1], top_medal.index[::-1])):
        ax1.text(v + 5, i, str(int(v)), va="center", fontsize=9)

    # snow bar
    ax2.barh(top_snow.index[::-1], top_snow.values[::-1], color=TEAL, height=0.6)
    ax2.set_xlabel("Mean snowfall depth (mm)")
    ax2.set_title("Top 10 by mean snowfall")
    for i, (v, c) in enumerate(zip(top_snow.values[::-1], top_snow.index[::-1])):
        ax2.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=9)

    # Highlight overlap in both
    overlap = set(top_medal.index) & set(top_snow.index)
    for ax, countries in [(ax1, top_medal.index[::-1]), (ax2, top_snow.index[::-1])]:
        for tick, country in zip(ax.get_yticklabels(), countries):
            if country in overlap:
                tick.set_color(CORAL)
                tick.set_fontweight("bold")

    fig.suptitle("Top 10 medal nations vs top 10 snowy nations  (red = overlap)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "08_top10_medals_vs_snow.png")

# ---------------------------------------------------------------------------
# 9. Average medals by snowfall quartile
# ---------------------------------------------------------------------------

def plot_quartile(df):
    sub = df.dropna(subset=["snowfall_mean_gridcell"]).copy()
    sub["sf_quartile"] = pd.qcut(
        sub["snowfall_mean_gridcell"], q=4,
        labels=["Q1 low\n(0–0.95mm)", "Q2\n(0.95–2.6mm)",
                "Q3\n(2.6–5.6mm)", "Q4 high\n(5.6–27mm)"]
    )
    agg = sub.groupby("sf_quartile", observed=True).agg(
        avg_medals=("total_medals", "mean"),
        n=("total_medals", "count"),
        pct_winners=("won_any_medal", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    colors = [GRAY, BLUE, BLUE, CORAL]

    bars = ax1.bar(agg["sf_quartile"], agg["avg_medals"], color=colors, width=0.5)
    ax1.set_ylabel("Average medals per country-edition")
    ax1.set_title("Average medals by snowfall quartile")
    for bar, val in zip(bars, agg["avg_medals"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(agg["sf_quartile"], agg["pct_winners"] * 100, color=colors, width=0.5)
    ax2.set_ylabel("% of country-editions that won ≥1 medal")
    ax2.set_title("% winning any medal by snowfall quartile")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    for bar, val in zip(bars2, agg["pct_winners"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.5,
                 f"{val*100:.0f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Snowfall quartile analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_medals_by_snowfall_quartile.png")

# ---------------------------------------------------------------------------
# Summary tables (printed)
# ---------------------------------------------------------------------------

def print_summary(df):
    print("\n=== Basic stats ===")
    cols = ["total_medals", "snowfall_mean_gridcell", "snowfall_total",
            "gdp", "gdp_per_capita", "population"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe().round(2).to_string())

    print("\n=== Zero-medal share ===")
    print(f"  {(df['total_medals'] == 0).mean()*100:.1f}% of country-editions have 0 medals")

    print("\n=== Correlations with total_medals ===")
    num_cols = ["snowfall_total", "snowfall_mean_gridcell", "gdp",
                "gdp_per_capita", "population", "n_athletes"]
    num_cols = [c for c in num_cols if c in df.columns]
    corrs = df[num_cols + ["total_medals"]].corr()["total_medals"].drop("total_medals")
    for col, val in corrs.sort_values(ascending=False).items():
        print(f"  {col:<30} r = {val:.3f}")

    print("\n=== Top 10 by total medals ===")
    top = df.groupby("country")["total_medals"].sum().sort_values(ascending=False).head(10)
    print(top.to_string())

    print("\n=== Avg medals by snowfall quartile ===")
    sub = df.dropna(subset=["snowfall_mean_gridcell"]).copy()
    sub["sf_q"] = pd.qcut(sub["snowfall_mean_gridcell"], q=4,
                          labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    print(sub.groupby("sf_q", observed=True)["total_medals"]
          .agg(["mean", "count"]).round(2).to_string())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    print_summary(df)

    print("\n[PLOTS]")
    plot_hist_medals(df)
    plot_hist_snowfall(df)
    plot_scatters(df)
    plot_heatmap(df)
    plot_top10(df)
    plot_quartile(df)

    print(f"\n✓ All plots saved → {FIG_DIR}/")

if __name__ == "__main__":
    main()