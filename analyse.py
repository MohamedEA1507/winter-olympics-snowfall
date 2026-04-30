"""
Research question:

To what extent does a country's snowy climate relate to Winter Olympic medal
performance, and does that relationship remain after controlling for GDP
and population?

Hypotheses:

H1: Snowier countries win more Winter Olympic medals (positive bivariate
    correlation with snowfall; negative with temperature).
H2: The snow effect attenuates but does not disappear after controlling for
    GDP and population, wealth alone cannot create ski culture or
    infrastructure.

Analysis structure:

  Main analysis:
    Model A  — OLS: snowfall only (baseline relationship)
    Model B  — OLS: snowfall + GDP + population (main research question)
    Model C  — OLS: snowfall + GDP + population + year FE (main controlled model)
    Model F  — Negative Binomial: count model with year FE (robustness)

  Secondary:
    Model G  — Logistic regression: won_any_medal (classification check)

  Robustness only (not main output):
    Model D  — OLS + country FE: within-country estimation

Data Requirements:

Input: data/clean/master.csv

Outputs:
  Console: model summaries, coefficient tables, fit statistics
  data/figures/10_model_coefficients.png   — OLS coefficient comparison (A–C)
  data/figures/11_count_model_fit.png      — predicted vs actual (NB model F)
  data/figures/12_logistic_roc.png         — ROC curve for Model G

"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import (roc_curve, auc, classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"

# Set to True to also run Model D (country + year FE) as a robustness check.
# Disabled by default — mention it in the report rather than the main output.
RUN_COUNTRY_FE_ROBUSTNESS = False

# Colour palette for figures
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
GRAY   = "#888780"

# Matplotlib global style
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

# =============================================================================
# 2. DATA LOADING AND PREPARATION
# =============================================================================

def load() -> pd.DataFrame:
    """Load master.csv, engineer modelling variables, and report basic checks."""
    df = pd.read_csv(MASTER)

    df = df.rename(columns={"snowfall_mean_gridcell": "snowfall"}, errors="ignore")

    # log transform to reduce skew
    df["log_total_medals"] = np.log1p(df["total_medals"])

    # binary target for secondary classification model (Model G)
    df["won_any_medal"] = (df["total_medals"] > 0).astype(int)

    # categorical year for fixed effects
    df["year_fe"] = df["year"].astype(str)

    model_cols = [
        "total_medals", "log_total_medals", "snowfall",
        "log_gdp", "log_population", "won_any_medal", "year_fe", "country",
    ]
    n_before  = len(df)
    df_model  = df.dropna(subset=model_cols).copy()
    n_after   = len(df_model)
    n_lost    = n_before - n_after

    print(f"\n{'='*60}")
    print("  DATA LOADING")
    print(f"{'='*60}")
    print(f"  Raw rows                    : {n_before:,}")
    print(f"  Rows with complete data     : {n_after:,}")
    print(f"  Rows dropped (missing)      : {n_lost:,} ({n_lost/n_before*100:.1f}%)")
    print(f"  Unique countries            : {df_model['country'].nunique()}")
    print(f"  Olympic editions            : {sorted(int(y) for y in df_model['year'].unique())}")

    # Overdispersion check — justifies Negative Binomial over Poisson
    mean_m = df_model["total_medals"].mean()
    var_m  = df_model["total_medals"].var()
    ratio  = var_m / mean_m
    print(f"\n  OVERDISPERSION CHECK")
    print(f"  mean(medals) = {mean_m:.2f}  |  var(medals) = {var_m:.2f}")
    print(f"  var/mean     = {ratio:.2f}  "
          f"{'→ overdispersed: Negative Binomial preferred over Poisson' if ratio > 2 else '→ mild: Poisson may be adequate'}")

    return df_model


# =============================================================================
# 3. HELPER: COMPACT MODEL SUMMARY PRINTER
# =============================================================================

def print_model_summary(label: str, result, extra_note: str = "") -> None:
    """Print a compact regression summary, hiding fixed-effect dummies."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    try:
        print(f"  Observations : {int(result.nobs):,}")
    except Exception:
        pass

    try:
        if hasattr(result, "rsquared"):
            print(f"  R²           : {result.rsquared:.4f}  "
                  f"(adj. R²: {result.rsquared_adj:.4f})")
        elif hasattr(result, "prsquared"):
            print(f"  Pseudo-R²    : {result.prsquared:.4f}")
    except Exception:
        pass

    try:
        print(f"  AIC          : {result.aic:.1f}")
    except Exception:
        pass

    try:
        coef_df = pd.DataFrame({
            "coef":   result.params,
            "se":     result.bse,
            "pvalue": result.pvalues,
        })
        # hide year/country fixed-effect dummies for readability
        keep = ~(
            coef_df.index.str.startswith("year_fe[") |
            coef_df.index.str.startswith("country[") |
            coef_df.index.str.startswith("C(") |
            coef_df.index.str.startswith("yr_")
        )
        coef_df = coef_df[keep]

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

    if extra_note:
        print(f"\n  NOTE: {extra_note}")


# =============================================================================
# 4. MAIN OLS MODELS (Models A–C)
# =============================================================================

def run_ols(df: pd.DataFrame) -> dict:
    """
    Fit the three main OLS models that directly address the research question.

      Model A — snowfall only            : baseline / bivariate relationship
      Model B — + GDP + population       : main research question
      Model C — + year fixed effects     : controls for Olympic-edition trends

    Model D (country + year FE) is run separately as a robustness check only.
    """
    print(f"\n\n{'█'*60}")
    print("  OLS MODELS  (main analysis)")
    print("  Dependent variable: log(1 + total_medals)")
    print("  Coefficients are interpreted on the log(1 + medals) scale")
    print("  Standard errors: HC3 heteroskedasticity-robust")
    print(f"{'█'*60}")

    results = {}

    # ------------------------------------------------------------------
    # Model A: bivariate — is snowfall related to medals at all?
    # ------------------------------------------------------------------
    mA = smf.ols("log_total_medals ~ snowfall", data=df).fit(cov_type="HC3")
    print_model_summary(
        "Model A — Bivariate: log_medals ~ snowfall",
        mA,
        extra_note="No controls. Coefficient is confounded by GDP and geography.",
    )
    results["A"] = mA

    # ------------------------------------------------------------------
    # Model B: add economic controls — does snow matter beyond wealth/size?
    # ------------------------------------------------------------------
    mB = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population",
        data=df,
    ).fit(cov_type="HC3")
    print_model_summary(
        "Model B — Economic controls: log_medals ~ snowfall + log_gdp + log_pop",
        mB,
        extra_note="Does snowfall matter AFTER accounting for wealth and population size?",
    )
    results["B"] = mB

    # Regression diagnostics for Model B
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(mB.resid, mB.model.exog)
        print(f"\n  Breusch-Pagan heteroskedasticity test (Model B):")
        print(f"  LM stat = {bp_test[0]:.3f}  |  p = {bp_test[1]:.4f}  "
              f"({'heteroskedastic — robust HC3 standard errors used' if bp_test[1] < 0.05 else 'homoskedastic'})")
    except Exception:
        pass

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_vif = df[["snowfall", "log_gdp", "log_population"]].dropna()
        X_vif = sm.add_constant(X_vif)
        print(f"\n  Variance Inflation Factors (Model B predictors):")
        for i, col in enumerate(X_vif.columns[1:], start=1):
            vif = variance_inflation_factor(X_vif.values, i)
            print(f"  {col:<25} VIF = {vif:.2f} {'⚠ collinearity' if vif > 10 else 'OK'}")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Model C: add year fixed effects — does the effect survive era controls?
    # ------------------------------------------------------------------
    mC = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population + C(year_fe)",
        data=df,
    ).fit(cov_type="HC3")
    print_model_summary(
        "Model C — Year FE: adds Olympic-year fixed effects",
        mC,
        extra_note="Year FE dummies suppressed. Controls for era-level trends.",
    )
    results["C"] = mC
    return results


# =============================================================================
# 4b. ROBUSTNESS CHECK — Model D (country + year fixed effects)
# =============================================================================

def run_ols_robustness(df: pd.DataFrame) -> dict:
    """
    Model D: country + year fixed effects (within-country estimator).

    This is a robustness check only, NOT part of the main analysis.

    Interpretation: country FEs absorb all stable between-country differences,
    so the model asks whether *within-country changes* in snowfall predict
    medal changes over time. Because snowfall is a slow-moving climate variable,
    the snowfall coefficient is expected to weaken here. A near-zero result
    confirms that the main snowfall effect is a between-country story
    (stable climate advantage), which is exactly what H1 and H2 claim.
    """
    print(f"\n\n{'─'*60}")
    print("  ROBUSTNESS CHECK — Model D (country + year fixed effects)")
    print("  NOT a main model. Included to verify between-country interpretation.")
    print(f"{'─'*60}")

    mD = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population "
        "+ C(year_fe) + C(country)",
        data=df,
    ).fit(cov_type="HC3")
    print_model_summary(
        "Model D — Country + Year FE: within-country estimation",
        mD,
        extra_note=(
            "If snowfall coef ≈ 0 here, the effect is BETWEEN countries, "
            "not within. This is expected for a slow-moving climate variable."
        ),
    )
    return {"D": mD}


# =============================================================================
# 5. COUNT MODEL — Negative Binomial (Model F)
# =============================================================================

def run_count_model(df: pd.DataFrame) -> dict:
    """
    Fit a Negative Binomial regression on raw medal counts.

    Poisson is not used as a main model because medal counts are typically
    overdispersed (var >> mean, checked at load time). The Negative Binomial
    relaxes Poisson's mean=variance assumption via a dispersion parameter α.
    If α is large and significant, Negative Binomial is the correct choice.
    """
    print(f"\n\n{'█'*60}")
    print("  COUNT MODEL — Negative Binomial  (main robustness model)")
    print("  Dependent variable: total_medals (raw integer count)")
    print("  exp(coef) = multiplicative effect on expected medal count")
    print(f"{'█'*60}")

    results = {}

    # build design matrix with year dummies
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X_base = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_base = sm.add_constant(X_base)
    y = df["total_medals"].reset_index(drop=True)

    print("\n  Fitting Negative Binomial regression...")
    try:
        mF = sm.NegativeBinomial(y, X_base).fit(disp=False)
        print_model_summary(
            "Model F — Negative Binomial: total_medals ~ snowfall + log_gdp + log_pop + year FE",
            mF,
        )

        alpha   = mF.params.get("alpha", np.nan)
        alpha_p = mF.pvalues.get("alpha", np.nan)
        print(f"\n  Dispersion parameter α = {alpha:.4f}  (p={alpha_p:.4f})")
        if alpha_p < 0.05:
            print("  Significant α confirms overdispersion — Negative Binomial is preferred over Poisson.")
        else:
            print("  α is not significant — overdispersion evidence is weaker.")

        print(f"\n  Key exp(coef) — multiplicative effects on medal count:")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mF.params.index:
                print(f"  exp({var:<20}) = {np.exp(mF.params[var]):.3f}  "
                      f"(p={mF.pvalues[var]:.4f})")
        results["F"] = mF
    except Exception as e:
        print(f"  [Negative Binomial failed: {e}]")

    return results


# =============================================================================
# 6. SECONDARY CLASSIFICATION MODEL — Logistic Regression (Model G)
# =============================================================================

def run_logistic(df: pd.DataFrame):
    """
    Secondary analysis: logistic regression predicting won_any_medal.

    This addresses the many-zero structure of the data (several countries
    never win a medal) and demonstrates the course's classification workflow
    with precision/recall/F1 and AUC evaluation.
    """
    print(f"\n\n{'█'*60}")
    print("  SECONDARY — LOGISTIC REGRESSION  (classification model)")
    print("  Dependent variable: won_any_medal (0 = no medals, 1 = any medal)")
    print("  exp(coef) = odds ratio")
    print(f"{'█'*60}")

    X_simple = sm.add_constant(
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True)
    )
    y = df["won_any_medal"].reset_index(drop=True)

    try:
        mG = sm.Logit(y, X_simple).fit(disp=False)
        print_model_summary(
            "Model G — Logistic: won_any_medal ~ snowfall + log_gdp + log_pop",
            mG,
        )

        print(f"\n  Odds ratios exp(coef) for key predictors:")
        print(f"  {'Variable':<25} {'Odds Ratio':>12} {'p-value':>10} {'':>5}")
        print(f"  {'-'*55}")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mG.params.index:
                coef = mG.params[var]
                pval = mG.pvalues[var]
                sig  = ("***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "")))
                print(f"  {var:<25} {np.exp(coef):>12.3f} {pval:>10.4f} {sig:>5}")

        # train/test split for out-of-sample evaluation
        X_np = X_simple.values
        y_np = y.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.20, random_state=42, stratify=y_np,
        )
        mG_train = sm.Logit(y_train, X_train).fit(disp=False)
        y_prob   = mG_train.predict(X_test)
        y_pred   = (y_prob >= 0.5).astype(int)

        print(f"\n  Classification report (threshold = 0.5, test set):")
        print(classification_report(y_test, y_pred, target_names=["no medal", "any medal"]))

        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"  AUC (out-of-sample): {roc_auc:.3f}")
        print(f"  Interpretation: {'Good' if roc_auc > 0.8 else 'Moderate'} "
              f"discrimination. AUC = 0.5 is random, 1.0 is perfect.")

        return mG, (y_test, y_prob)

    except Exception as e:
        print(f"  [Logistic failed: {e}]")
        return None, (None, None)


# =============================================================================
# 7. INTERPRETATION SUMMARY
# =============================================================================

def print_interpretation(ols_results: dict, count_results: dict, logit_result) -> None:
    """
    Compare the snowfall coefficient across all models in one table.

    Reading guide:
      - If snowfall is significant in A but disappears in B:
          the bivariate effect was entirely driven by GDP/population.
      - If snowfall survives through B and C:
          the climate effect is robust to economic controls and era trends.
      - If snowfall weakens in D (robustness):
          the effect is a between-country story (stable climate advantage),
          which is the expected and interpretable result for H1 and H2.
    """
    print(f"\n\n{'█'*60}")
    print("  INTERPRETATION SUMMARY")
    print("  Snowfall coefficient across main and secondary models")
    print(f"{'█'*60}")
    print(f"\n  NOTE: Coefficients are not directly comparable in size across model types.")
    print(f"  OLS = change in log(1+medals) | NegBin = change in log expected count")
    print(f"  Logistic = change in log odds of winning any medal")
    print(f"  Compare sign and significance, not raw magnitude.")

    rows = []
    for label, m in ols_results.items():
        if "snowfall" in m.params:
            rows.append({
                "Model": f"OLS Model {label}",
                "Coef": m.params["snowfall"],
                "p-value": m.pvalues["snowfall"],
            })
    for label, m in count_results.items():
        if "snowfall" in m.params.index:
            rows.append({
                "Model": f"Count Model {label}",
                "Coef": m.params["snowfall"],
                "p-value": m.pvalues["snowfall"],
            })
    if logit_result is not None and "snowfall" in logit_result.params.index:
        rows.append({
            "Model": "Logistic Model G",
            "Coef": logit_result.params["snowfall"],
            "p-value": logit_result.pvalues["snowfall"],
        })

    print(f"\n  {'Model':<25} {'Coef':>10} {'p-value':>10} {'Significant?':>14}")
    print(f"  {'-'*62}")
    for row in rows:
        sig = ("YES ***" if row["p-value"] < 0.001 else
               ("YES **"  if row["p-value"] < 0.01  else
                ("YES *"  if row["p-value"] < 0.05  else "no")))
        print(f"  {row['Model']:<25} {row['Coef']:>10.4f} "
              f"{row['p-value']:>10.4f} {sig:>14}")


# =============================================================================
# 8. VISUALISATIONS
# =============================================================================

def plot_ols_coefficients(ols_results: dict) -> None:
    """Coefficient comparison plot for main OLS models A–C."""
    vars_to_plot = ["snowfall", "log_gdp", "log_population"]
    var_labels   = ["Snowfall (mm)", "log₁₀(GDP, USD)", "log₁₀(Population)"]
    model_colors = [BLUE, CORAL, TEAL, PURPLE]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "OLS coefficient estimates across model specifications\n"
        "Bars = point estimate  |  Error bars = 95% CI  |  Dashed line = zero",
        fontsize=11, fontweight="bold",
    )

    for ax, var, var_label in zip(axes, vars_to_plot, var_labels):
        coefs, cis, labels = [], [], []
        for label, m in ols_results.items():
            if var in m.params:
                coefs.append(m.params[var])
                ci = m.conf_int().loc[var]
                cis.append((ci[0], ci[1]))
                labels.append(f"Model {label}")

        y_pos = np.arange(len(labels))
        for i, (coef, (lo, hi), color) in enumerate(zip(coefs, cis, model_colors)):
            ax.barh(
                y_pos[i], coef,
                xerr=[[coef - lo], [hi - coef]],
                color=color, height=0.55, capsize=4, alpha=0.85,
                error_kw={"linewidth": 1.2, "ecolor": "#333"},
            )

        ax.axvline(0, color="#999", linewidth=1, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(var_label, fontsize=11)
        ax.set_xlabel("Coefficient (OLS)", fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "10_model_coefficients.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [saved] {path}")


def plot_count_model_fit(count_results: dict, df: pd.DataFrame) -> None:
    """Predicted vs actual plot for the Negative Binomial model."""
    if "F" not in count_results:
        print("  [skipping count model plot — Model F not fitted]")
        return

    m      = count_results["F"]
    pred   = m.predict()
    actual = df["total_medals"].values[:len(pred)]

    mask   = np.isfinite(pred) & np.isfinite(actual)
    pred   = pred[mask]
    actual = actual[mask]

    if len(pred) == 0:
        print("  [skipping count model plot — no valid predictions]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Model F (Negative Binomial) — Predicted vs Actual Medal Counts",
        fontsize=12, fontweight="bold",
    )

    ax1.scatter(pred, actual, alpha=0.4, s=30, color=BLUE, linewidths=0)
    lim = max(actual.max(), pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=CORAL, linewidth=1.5, label="Perfect fit")
    ax1.set_xlabel("Predicted medals")
    ax1.set_ylabel("Actual medals")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(fontsize=9)

    resid = actual - pred
    ax2.hist(resid, bins=40, color=TEAL, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=CORAL, linewidth=1.5, linestyle="--", label="Zero residual")
    ax2.set_xlabel("Residual (actual − predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "11_count_model_fit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_roc_curve(logit_result, y_test_prob_tuple: tuple) -> None:
    """ROC curve for the secondary logistic classification model."""
    if logit_result is None:
        print("  [skipping ROC plot — logistic model not fitted]")
        return

    y_true, y_prob = y_test_prob_tuple
    if y_true is None:
        print("  [skipping ROC plot — no test predictions available]")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    j_scores      = tpr - fpr
    optimal_idx   = np.argmax(j_scores)
    optimal_thr   = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1, label="Random classifier")
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color=CORAL, s=100, zorder=5,
               label=f"Optimal threshold = {optimal_thr:.2f}")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)")
    ax.set_title("Model G — ROC Curve\n(won_any_medal binary classification)")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    path = FIG_DIR / "12_logistic_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data
    df = load()

    # 2. Main OLS models (A, B, C)
    ols_results = run_ols(df)

    # 3. Robustness check — country fixed effects (Model D, optional)
    if RUN_COUNTRY_FE_ROBUSTNESS:
        run_ols_robustness(df)

    # 4. Negative Binomial count model (Model F)
    count_results = run_count_model(df)

    # 5. Secondary classification model (Model G)
    logit_result, y_test_prob = run_logistic(df)

    # 6. Cross-model interpretation summary (A, B, C, F, G)
    print_interpretation(ols_results, count_results, logit_result)

    # 7. Figures
    print(f"\n\n{'='*60}")
    print("  SAVING FIGURES")
    print(f"{'='*60}")
    plot_ols_coefficients(ols_results)   # A, B, C only
    plot_count_model_fit(count_results, df)
    plot_roc_curve(logit_result, y_test_prob)

    print(f"\n✓ Analysis complete → {FIG_DIR}/")
    print("  10_model_coefficients.png  — OLS snowfall coefficient across Models A–C")
    print("  11_count_model_fit.png     — Negative Binomial predicted vs actual")
    print("  12_logistic_roc.png        — ROC curve, AUC, optimal threshold")


if __name__ == "__main__":
    main()