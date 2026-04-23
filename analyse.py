"""
Regression Modelling Pipeline
================================
Runs all models from simple bivariate to full panel-data specification.

Models:
  Model A  — OLS: log_medals ~ snowfall  (bivariate baseline)
  Model B  — OLS: log_medals ~ snowfall + log_gdp + log_population
  Model C  — OLS: log_medals ~ snowfall + log_gdp + log_population + year FE
  Model D  — OLS: log_medals ~ snowfall + log_gdp + log_population + year FE + country FE

  Model E  — Poisson:          total_medals ~ snowfall + log_gdp + log_population + year FE
  Model F  — Negative Binomial: total_medals ~ snowfall + log_gdp + log_population + year FE
  Model G  — Logistic:          won_any_medal ~ snowfall + log_gdp + log_population + year FE

Outputs:
  Console: model summaries, coefficients, fit statistics
  data/figures/10_model_coefficients.png  — coefficient plot across OLS models
  data/figures/11_poisson_vs_actual.png   — predicted vs actual for count model
  data/figures/12_logistic_roc.png        — ROC curve for logistic model

Usage:
  python model.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, classification_report
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"

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
})

# ---------------------------------------------------------------------------
# Load + prepare
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER)

    # Standardise column names in case they differ
    df = df.rename(columns={
        "snowfall_km3":           "snowfall_total",
        "snowfall_mm":            "snowfall_mean_gridcell",
        "gdp_usd":                "gdp",
        "gdp_per_capita_usd":     "gdp_per_capita",
        "population_total":       "population",
        "iso3":                   "country",
    }, errors="ignore")

    # Engineered columns
    df["log_total_medals"]   = np.log1p(df["total_medals"])
    df["log_gdp"]            = np.log10(df["gdp"].where(df["gdp"] > 0))
    df["log_population"]     = np.log10(df["population"].where(df["population"] > 0))
    df["won_any_medal"]      = (df["total_medals"] > 0).astype(int)
    df["medals_per_million"] = df["total_medals"] / df["population"] * 1e6

    # Two snowfall proxies — use mean gridcell depth as the main one
    # (more directly comparable across countries of different sizes)
    df["snowfall"] = df["snowfall_mean_gridcell"]

    # Year as categorical for fixed effects
    df["year_fe"] = df["year"].astype(str)

    # Drop rows missing any model variable
    model_cols = ["total_medals", "log_total_medals", "snowfall",
                  "log_gdp", "log_population", "won_any_medal", "year_fe", "country"]
    df_model = df.dropna(subset=model_cols).copy()

    print(f"[LOAD] {len(df):,} total rows → {len(df_model):,} rows with complete data")
    print(f"       {df_model['country'].nunique()} countries · "
          f"{df_model['year'].nunique()} editions")

    # Overdispersion check
    mean_m = df_model["total_medals"].mean()
    var_m  = df_model["total_medals"].var()
    print(f"\n[OVERDISPERSION CHECK]")
    print(f"  mean(medals) = {mean_m:.2f}  var(medals) = {var_m:.2f}")
    print(f"  variance / mean = {var_m/mean_m:.2f}  "
          f"({'overdispersed → use Negative Binomial' if var_m/mean_m > 2 else 'mild → Poisson may work'})")

    return df_model

# ---------------------------------------------------------------------------
# Helper: print model summary compactly
# ---------------------------------------------------------------------------

def print_model(label, result, extra=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        print(f"  Observations : {int(result.nobs)}")
    except Exception:
        pass
    try:
        print(f"  R² / Pseudo-R²: {result.rsquared:.4f}" if hasattr(result, 'rsquared')
              else f"  Pseudo-R²: {result.prsquared:.4f}")
    except Exception:
        pass
    try:
        print(f"  AIC          : {result.aic:.1f}")
    except Exception:
        pass

    # Coefficient table — only non-FE terms
    try:
        coef_df = pd.DataFrame({
            "coef":   result.params,
            "se":     result.bse,
            "pvalue": result.pvalues,
        })
        # Filter out FE dummies for readability
        mask = ~coef_df.index.str.startswith("year_fe[") & \
               ~coef_df.index.str.startswith("country[") & \
               ~coef_df.index.str.startswith("C(")
        coef_df = coef_df[mask]
        coef_df["sig"] = coef_df["pvalue"].apply(
            lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        )
        print(f"\n  {'Variable':<30} {'Coef':>10} {'SE':>10} {'p-value':>10} {'':>5}")
        print(f"  {'-'*65}")
        for name, row in coef_df.iterrows():
            print(f"  {name:<30} {row['coef']:>10.4f} {row['se']:>10.4f} "
                  f"{row['pvalue']:>10.4f} {row['sig']:>5}")
    except Exception as e:
        print(f"  [could not extract coefficients: {e}]")

    if extra:
        print(f"\n  {extra}")

# ---------------------------------------------------------------------------
# OLS Models A–D
# ---------------------------------------------------------------------------

def run_ols(df):
    print("\n\n" + "█"*60)
    print("  OLS MODELS (dependent variable: log(1 + total_medals))")
    print("█"*60)

    results = {}

    # Model A — bivariate
    mA = smf.ols("log_total_medals ~ snowfall", data=df).fit()
    print_model("Model A — bivariate: log_medals ~ snowfall", mA)
    results["A"] = mA

    # Model B — add GDP + population
    mB = smf.ols("log_total_medals ~ snowfall + log_gdp + log_population", data=df).fit()
    print_model("Model B — controls: log_medals ~ snowfall + log_gdp + log_population", mB)
    results["B"] = mB

    # Model C — add year fixed effects
    mC = smf.ols("log_total_medals ~ snowfall + log_gdp + log_population + C(year_fe)",
                 data=df).fit()
    print_model("Model C — year FE: + year fixed effects", mC,
                extra="Year FE coefficients suppressed for readability")
    results["C"] = mC

    # Model D — add country fixed effects
    mD = smf.ols("log_total_medals ~ snowfall + log_gdp + log_population + C(year_fe) + C(country)",
                 data=df).fit()
    print_model("Model D — country + year FE: within-country estimation", mD,
                extra="Country FE suppressed. If snowfall coef ≈ 0 here, snowfall is mostly between-country.")
    results["D"] = mD

    # F-test: do year FEs jointly matter?
    try:
        ftest = mC.compare_f_test(mB)
        print(f"\n  F-test (year FE jointly = 0): F={ftest[0]:.2f}, p={ftest[1]:.4f}")
    except Exception:
        pass

    return results

# ---------------------------------------------------------------------------
# Count Models E–F
# ---------------------------------------------------------------------------

def run_count(df):
    print("\n\n" + "█"*60)
    print("  COUNT MODELS (dependent variable: total_medals)")
    print("█"*60)

    results = {}

    # Build design matrix (year dummies)
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X_base = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_base = sm.add_constant(X_base)
    y = df["total_medals"].reset_index(drop=True)

    # Model E — Poisson
    print("\n  Fitting Poisson regression…")
    try:
        mE = sm.GLM(y, X_base, family=sm.families.Poisson()).fit()
        print_model("Model E — Poisson: total_medals ~ snowfall + log_gdp + log_pop + year FE", mE)
        results["E"] = mE

        # Pearson χ² goodness of fit
        chi2_stat = mE.pearson_chi2
        df_resid  = mE.df_resid
        print(f"\n  Pearson χ²/df = {chi2_stat/df_resid:.2f} "
              f"({'overdispersed → prefer NB' if chi2_stat/df_resid > 2 else 'acceptable'})")
    except Exception as e:
        print(f"  [Poisson failed: {e}]")

    # Model F — Negative Binomial
    print("\n  Fitting Negative Binomial regression…")
    try:
        mF = sm.NegativeBinomial(y, X_base).fit(disp=False)
        print_model("Model F — Negative Binomial: total_medals ~ snowfall + log_gdp + log_pop + year FE", mF)
        results["F"] = mF

        alpha = mF.params.get("alpha", np.nan)
        print(f"\n  Dispersion parameter α = {alpha:.4f} "
              f"({'significant overdispersion' if alpha > 0.1 else 'mild overdispersion'})")
    except Exception as e:
        print(f"  [Negative Binomial failed: {e}]")

    return results

# ---------------------------------------------------------------------------
# Logistic Model G
# ---------------------------------------------------------------------------

def run_logistic(df):
    print("\n\n" + "█"*60)
    print("  LOGISTIC MODEL (dependent variable: won_any_medal = 0/1)")
    print("█"*60)

    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X = sm.add_constant(X)
    y = df["won_any_medal"].reset_index(drop=True)

    try:
        mG = sm.Logit(y, X).fit(disp=False)
        print_model("Model G — Logistic: won_any_medal ~ snowfall + log_gdp + log_pop + year FE", mG)

        # Odds ratios for interpretable output
        print("\n  Odds ratios (exp(coef)) for main predictors:")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mG.params:
                coef = mG.params[var]
                pval = mG.pvalues[var]
                sig  = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"    {var:<25} OR = {np.exp(coef):.3f}  (p={pval:.4f}) {sig}")

        return mG
    except Exception as e:
        print(f"  [Logistic failed: {e}]")
        return None

# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def print_interpretation(ols_results, count_results, logit_result):
    print("\n\n" + "█"*60)
    print("  INTERPRETATION SUMMARY")
    print("█"*60)

    snowfall_coefs = {}
    for label, m in ols_results.items():
        if "snowfall" in m.params:
            snowfall_coefs[f"OLS Model {label}"] = (m.params["snowfall"], m.pvalues["snowfall"])
    for label, m in count_results.items():
        if "snowfall" in m.params:
            snowfall_coefs[f"Count Model {label}"] = (m.params["snowfall"], m.pvalues["snowfall"])
    if logit_result and "snowfall" in logit_result.params:
        snowfall_coefs["Logistic Model G"] = (
            logit_result.params["snowfall"], logit_result.pvalues["snowfall"])

    print("\n  Snowfall coefficient across all models:")
    print(f"  {'Model':<25} {'Coef':>10} {'p-value':>10} {'Significant?':>14}")
    print(f"  {'-'*60}")
    for model, (coef, pval) in snowfall_coefs.items():
        sig = "YES ***" if pval < 0.001 else ("YES **" if pval < 0.01 else
              ("YES *" if pval < 0.05 else "no"))
        print(f"  {model:<25} {coef:>10.4f} {pval:>10.4f} {sig:>14}")

    print("""
  How to read these results:
  ─────────────────────────
  OLS Model A  (bivariate)
    → Raw relationship: does snowfall alone predict medals?
    → No controls — coefficient is confounded by GDP, geography

  OLS Model B  (+ GDP + population)
    → Does snowfall matter AFTER accounting for wealth and size?
    → If coef drops a lot from A→B, GDP was doing much of the work

  OLS Model C  (+ year FE)
    → Controls for global trends (more events, more nations)
    → Compares countries at the same point in Olympic history

  OLS Model D  (+ country FE)
    → Within-country variation only
    → If coef ≈ 0: snowfall barely changes within a country over time
    → This is EXPECTED — snowfall is a slow-moving climate variable
    → Important finding: the effect is a BETWEEN-country story

  Count Models (Poisson/NB)
    → More appropriate for non-negative integer counts
    → Negative Binomial preferred if overdispersion > 2
    → Coefficients are log rate ratios: exp(coef) = multiplicative effect

  Logistic (won_any_medal)
    → Odds ratio: how much more likely to win ≥1 medal per unit of snowfall?
    → Useful for binary framing: participation vs performance
    """)

# ---------------------------------------------------------------------------
# Plot: coefficient comparison across OLS models
# ---------------------------------------------------------------------------

def plot_coefficients(ols_results):
    vars_to_plot = ["snowfall", "log_gdp", "log_population"]
    model_labels = list(ols_results.keys())
    colors = [BLUE, CORAL, TEAL, PURPLE]

    fig, axes = plt.subplots(1, len(vars_to_plot), figsize=(13, 5))

    for ax, var in zip(axes, vars_to_plot):
        coefs = []
        cis   = []
        labels = []
        for label, m in ols_results.items():
            if var in m.params:
                coef = m.params[var]
                ci   = m.conf_int().loc[var]
                coefs.append(coef)
                cis.append((ci[0], ci[1]))
                labels.append(f"Model {label}")

        y_pos = np.arange(len(labels))
        for i, (coef, (lo, hi), color) in enumerate(zip(coefs, cis, colors)):
            ax.barh(y_pos[i], coef, xerr=[[coef - lo], [hi - coef]],
                    color=color, height=0.5, capsize=4, alpha=0.85,
                    error_kw={"linewidth": 1.2, "ecolor": "#333"})

        ax.axvline(0, color="#999", linewidth=1, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(var.replace("_", " "), fontsize=12)
        ax.set_xlabel("Coefficient (OLS)")

    fig.suptitle("OLS coefficients across model specifications\n(bars = 95% CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "10_model_coefficients.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Plot: predicted vs actual (Poisson / NB)
# ---------------------------------------------------------------------------

def plot_predicted_actual(count_results, df):
    best_key = "F" if "F" in count_results else ("E" if "E" in count_results else None)
    if best_key is None:
        return
    m = count_results[best_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pred = m.predict()
    actual = df["total_medals"].values[:len(pred)]

    ax1.scatter(pred, actual, alpha=0.4, s=30, color=BLUE, linewidths=0)
    lim = max(actual.max(), pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=CORAL, linewidth=1.5)
    ax1.set_xlabel("Predicted medals")
    ax1.set_ylabel("Actual medals")
    ax1.set_title(f"Model {best_key} — predicted vs actual")

    # Residual distribution
    resid = actual - pred
    ax2.hist(resid, bins=40, color=TEAL, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=CORAL, linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (actual − predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual distribution")

    fig.tight_layout()
    path = FIG_DIR / "11_poisson_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Plot: ROC curve (Logistic)
# ---------------------------------------------------------------------------

def plot_roc(logit_result, df):
    if logit_result is None:
        return

    y_true = df["won_any_medal"].values
    y_pred = logit_result.predict()[:len(y_true)]

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Classification report at 0.5 threshold
    y_class = (y_pred >= 0.5).astype(int)
    print("\n  Classification report (threshold = 0.5):")
    print(classification_report(y_true, y_class,
                                 target_names=["no medal", "any medal"]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Model G — ROC curve (won_any_medal)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "12_logistic_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load()

    ols_results   = run_ols(df)
    count_results = run_count(df)
    logit_result  = run_logistic(df)

    print_interpretation(ols_results, count_results, logit_result)

    print("\n[PLOTS]")
    plot_coefficients(ols_results)
    plot_predicted_actual(count_results, df)
    plot_roc(logit_result, df)

    print(f"\n✓ Modelling complete → {FIG_DIR}/")

if __name__ == "__main__":
    main()