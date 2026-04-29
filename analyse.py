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
H3: Using medals per million reduces the raw size effect and isolates a
    climate/infrastructure signal more cleanly than total medals.

Data Requirements:

Input: data/clean/master.csv 

Outputs:
  Console: model summaries, coefficient tables, fit statistics
  data/figures/10_model_coefficients.png   — OLS coefficient comparison
  data/figures/11_count_model_fit.png      — predicted vs actual (NB model)
  data/figures/12_logistic_roc.png         — ROC curve for Model G

"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

# statsmodel and sklearn are the main modelling libraries

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf   
import statsmodels.api as sm            
from sklearn.metrics import (roc_curve, auc, classification_report, roc_auc_score, )
from sklearn.model_selection import train_test_split  
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# define file paths + plotting style for consistency

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"

# Colour palette for figures
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#BA7517"
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
# load dataset and create modeling variables, consistent, clean input for all models
def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER)

    # rename for simplicity
    df = df.rename(columns={"snowfall_mean_gridcell": "snowfall",
    }, errors="ignore")

    # log transform to reduce skew
    df["log_total_medals"] = np.log1p(df["total_medals"])

    # binary target for classification (model G)
    df["won_any_medal"] = (df["total_medals"] > 0).astype(int)

    # convert year to categorical for fixed effects
    df["year_fe"] = df["year"].astype(str)

    # drop incomplete rows
    model_cols = [ "total_medals", "log_total_medals", "snowfall", "log_gdp", "log_population", "won_any_medal", "year_fe", "country",]
    n_before = len(df)
    df_model = df.dropna(subset=model_cols).copy()
    n_after  = len(df_model)
    n_lost   = n_before - n_after

    print(f"\n{'='*60}")
    print("  DATA LOADING")
    print(f"{'='*60}")
    print(f"  Raw rows: {n_before:,}")
    print(f"  Rows with complete model data: {n_after:,}")
    print(f"  Rows dropped (missing values): {n_lost:,} " f"({n_lost/n_before*100:.1f}%)")
    print(f"  Unique countries: {df_model['country'].nunique()}")
    print(f"  Olympic editions: {sorted(int(y) for y in df_model['year'].unique())}")

    # check for overdispersion in medal counts (choice between model E or F, F if var >> mean)
    mean_m = df_model["total_medals"].mean()
    var_m  = df_model["total_medals"].var()
    ratio  = var_m / mean_m
    print(f"\n  OVERDISPERSION CHECK (for Model E vs F decision)")
    print(f"  mean(medals) = {mean_m:.2f}  |  var(medals) = {var_m:.2f}")
    print(f"  var/mean     = {ratio:.2f}  "
          f"{'→ overdispersed: prefer Negative Binomial (Model F)' if ratio > 2 else '→ mild: Poisson (Model E) may be adequate'}")

    return df_model


# =============================================================================
# 3. HELPER: COMPACT MODEL SUMMARY PRINTER
# =============================================================================

# print compact regression results hide, fixxed effects hidden for readability

def print_model_summary(label: str, result, extra_note: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # number of observations used
    try:
        print(f"  Observations : {int(result.nobs):,}")
    except Exception:
        pass

    # Goodness of fit
    try:
        if hasattr(result, "rsquared"):
            print(f"  R²           : {result.rsquared:.4f}  "
                  f"(adj. R²: {result.rsquared_adj:.4f})")
        elif hasattr(result, "prsquared"):
            print(f"  Pseudo-R²    : {result.prsquared:.4f}")
    except Exception:
        pass

    # lower AIC is better
    try:
        print(f"  AIC          : {result.aic:.1f}")
    except Exception:
        pass

    # filter out FE dummies and only show substantive predictors for readability
    try:
        coef_df = pd.DataFrame({
            "coef":   result.params,
            "se":     result.bse,
            "pvalue": result.pvalues,
        })
        keep = ~(
            coef_df.index.str.startswith("year_fe[") |
            coef_df.index.str.startswith("country[") |
            coef_df.index.str.startswith("C(")
        )
        coef_df = coef_df[keep]

        coef_df["sig"] = coef_df["pvalue"].apply(
            lambda p: "***" if p < 0.001
                      else ("**" if p < 0.01
                            else ("*" if p < 0.05 else ""))
        )

        print(f"\n  {'Variable':<30} {'Coef':>10} {'SE':>10} "
              f"{'p-value':>10} {'':>5}")
        print(f"  {'-'*65}")
        for name, row in coef_df.iterrows():
            print(f"  {name:<30} {row['coef']:>10.4f} {row['se']:>10.4f} "
                  f"{row['pvalue']:>10.4f} {row['sig']:>5}")
    except Exception as e:
        print(f"  [could not extract coefficients: {e}]")

    if extra_note:
        print(f"\n  NOTE: {extra_note}")

# =============================================================================
# 4. OLS MODELS (Models A–D)
# =============================================================================
def run_ols(df: pd.DataFrame) -> dict:
    # linear regression on Log medals. Used to isolate snowfall effect step by step
    print(f"\n\n{'█'*60}")
    print("  OLS MODELS")
    print("  Dependent variable: log(1 + total_medals)")
    print("  Interpretation: coefficients ≈ % effect on expected medals")
    print(f"{'█'*60}")

    results = {}

    # Model A: raw correlation
    mA = smf.ols(
        "log_total_medals ~ snowfall",
        data=df,
    ).fit()
    print_model_summary(
        "Model A - Bivariate: log_medals ~ snowfall",
        mA,
        extra_note="No controls. Coefficient is confounded by GDP and geography."
    )
    results["A"] = mA

    # model B: add economic controls (log_gdp, log_population)
    mB = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population",
        data=df,
    ).fit()
    print_model_summary(
        "Model B - Economic controls: log_medals ~ snowfall + log_gdp + log_pop",
        mB,
        extra_note="Does snowfall matter AFTER accounting for wealth and size?"
    )
    results["B"] = mB

    # model C: add year fixed effects (time trends)
    mC = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population + C(year_fe)",
        data=df,
    ).fit()
    print_model_summary(
        "Model C - Year FE: adds Olympic-year fixed effects",
        mC,
        extra_note="Year FE dummies suppressed. Controls for era-level trends."
    )
    results["C"] = mC

    # model D: add country fixed effects (within estimator)
    mD = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population "
        "+ C(year_fe) + C(country)",
        data=df,
    ).fit()
    print_model_summary(
        "Model D - Country + Year FE: within-country estimation",
        mD,
        extra_note=(
            "If snowfall coef ≈ 0 here, the effect is BETWEEN countries, "
            "not within. This is expected for a slow-moving climate variable."
        )
    )
    results["D"] = mD

    # F-test: do year fixed effects jointly improve fit over Model B?
    try:
        ftest = mC.compare_f_test(mB)
        print(f"\n  F-test: year FEs jointly = 0?")
        print(f"  F = {ftest[0]:.2f}  |  p = {ftest[1]:.4f}  "
              f"({'year FE matter' if ftest[1] < 0.05 else 'year FE not significant'})")
    except Exception:
        pass

    # regression diagnostics for Model B 
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(mB.resid, mB.model.exog)
        print(f"\n  Breusch-Pagan heteroskedasticity test (Model B):")
        print(f"  LM stat = {bp_test[0]:.3f}  |  p = {bp_test[1]:.4f}  "
              f"({'heteroskedastic - consider robust SEs' if bp_test[1] < 0.05 else 'homoskedastic'})")
    except Exception:
        pass

    # VIF check for multicollinearity among predictors in Model B
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_vif = df[["snowfall", "log_gdp", "log_population"]].dropna()
        X_vif = sm.add_constant(X_vif)
        print(f"\n  Variance Inflation Factors (Model B predictors):")
        for i, col in enumerate(X_vif.columns[1:], start=1):
            vif = variance_inflation_factor(X_vif.values, i)
            print(f"  {col:<25} VIF = {vif:.2f} "
                  f"{'⚠ collinearity' if vif > 10 else 'OK'}")
    except Exception:
        pass

    return results

# =============================================================================
# 5. COUNT MODELS (Models E–F)
# =============================================================================
# models for integer medal counts, more appropriate than OLS for count data, check for overdispersion to choose between Poisson and Negative Binomial
def run_count_models(df: pd.DataFrame) -> dict:
    print(f"\n\n{'█'*60}")
    print("  COUNT MODELS")
    print("  Dependent variable: total_medals (raw integer count)")
    print("  Interpretation: exp(coef) = multiplicative effect on medal count")
    print(f"{'█'*60}")

    results = {}

    # create year dummy variables
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X_base = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_base = sm.add_constant(X_base)         
    y = df["total_medals"].reset_index(drop=True)

    # model E: Poisson regression
    print("\n  Fitting Poisson regression...")
    try:
        mE = sm.GLM(y, X_base, family=sm.families.Poisson()).fit()
        print_model_summary(
            "Model E - Poisson: total_medals ~ snowfall + log_gdp + log_pop + year FE",
            mE,
        )
        # Goodness-of-fit: Pearson χ²/df
        # If this >> 1, residual variance exceeds what Poisson predicts.
        pearson_ratio = mE.pearson_chi2 / mE.df_resid
        print(f"\n  Pearson χ²/df = {pearson_ratio:.2f}  "
              f"({'overdispersed → NegBin preferred' if pearson_ratio > 2 else 'acceptable'})")

        # Exponentiated coefficients for interpretation
        print(f"\n  Key exp(coef) - multiplicative effects on medal count:")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mE.params.index:
                print(f"  exp({var:<20}) = {np.exp(mE.params[var]):.3f}  "
                      f"(p={mE.pvalues[var]:.4f})")
        results["E"] = mE
    except Exception as e:
        print(f"  [Poisson failed: {e}]")

    # negative binomial allows overdispersion
    print("\n  Fitting Negative Binomial regression...")
    try:
        mF = sm.NegativeBinomial(y, X_base).fit(disp=False)
        print_model_summary(
            "Model F - Negative Binomial: total_medals ~ snowfall + log_gdp + log_pop + year FE",
            mF,
        )
        # Dispersion parameter α
        alpha = mF.params.get("alpha", np.nan)
        print(f"\n  Dispersion parameter α = {alpha:.4f}  "
              f"({'significant overdispersion → NegBin is correct choice' if alpha > 0.1 else 'mild overdispersion'})")

        print(f"\n  Key exp(coef) - multiplicative effects on medal count:")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mF.params.index:
                print(f"  exp({var:<20}) = {np.exp(mF.params[var]):.3f}  "
                      f"(p={mF.pvalues[var]:.4f})")
        results["F"] = mF
    except Exception as e:
        print(f"  [Negative Binomial failed: {e}]")

    return results

# =============================================================================
# 6. LOGISTIC REGRESSION MODEL (Model G)
# =============================================================================
# binary classification: can we predict which countries win any medals at all?
def run_logistic(df: pd.DataFrame):
    print(f"\n\n{'█'*60}")
    print("  LOGISTIC REGRESSION MODEL")
    print("  Dependent variable: won_any_medal (0 = no medals, 1 = any medal)")
    print("  Interpretation: exp(coef) = odds ratio")
    print(f"{'█'*60}")

    # only main predictors used to avoid multicollinearity
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_with_const = sm.add_constant(X)
    y = df["won_any_medal"].reset_index(drop=True)

    try:
        # fit on full sample for interpretability, then evaluate on train/test split
        X_simple = sm.add_constant(
            df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True)
        )
        mG = sm.Logit(y, X_simple).fit(disp=False)
        print_model_summary(
            "Model G - Logistic: won_any_medal ~ snowfall + log_gdp + log_pop",
            mG,
        )

        # Odds ratios for the main substantive predictors
        print(f"\n  Odds ratios exp(coef) for key predictors:")
        print(f"  {'Variable':<25} {'Odds Ratio':>12} {'p-value':>10} {'':>5}")
        print(f"  {'-'*55}")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mG.params.index:
                coef = mG.params[var]
                pval = mG.pvalues[var]
                sig  = ("***" if pval < 0.001 else
                        ("**"  if pval < 0.01  else
                         ("*"  if pval < 0.05  else "")))
                print(f"  {var:<25} {np.exp(coef):>12.3f} {pval:>10.4f} {sig:>5}")

        # train/test split for predictive evaluation 
        X_np = X_simple.values
        y_np = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=0.20,
            random_state=42,
            stratify=y_np,   # preserve class balance in both splits
        )
        # Fit on training subset
        mG_train = sm.Logit(y_train, X_train).fit(disp=False)
        y_prob   = mG_train.predict(X_test)   # predicted probabilities
        y_pred   = (y_prob >= 0.5).astype(int) # classify at 0.5 threshold

        # Classification report (precision, recall, F1, accuracy)
        print(f"\n  Classification report (threshold = 0.5, test set):")
        print(classification_report(
            y_test, y_pred,
            target_names=["no medal", "any medal"],
        ))

        # auc 
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
# compare snowfall effect across all models
# if snowfall coefficient is positive and significant in Model A but disappears in Model: the bivariate effect was entirely driven by GDP
# if snowfall coefficient is positive and significant through Models B and C: the effect is robust to economic controls and era trends
# if snowfall coefficient is near zero in Model D: the effect is a BETWEEN-country story (stable climate advantage), not a within-country time-series effect (expected and interpretable)
def print_interpretation(ols_results: dict, count_results: dict, logit_result,) -> None:
    print(f"\n\n{'█'*60}")
    print("  INTERPRETATION SUMMARY")
    print("  Snowfall coefficient across all 7 models")
    print(f"{'█'*60}")
    rows = []

    # extract snowfall coefficient and p-value from each model for comparison
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

    # print table for comparison
    print(f"\n  {'Model':<25} {'Coef':>10} {'p-value':>10} {'Significant?':>14}")
    print(f"  {'-'*62}")

    # significance stars for quick visual comparison across models
    for row in rows:
        sig = ("YES ***" if row["p-value"] < 0.001 else
               ("YES **"  if row["p-value"] < 0.01  else
                ("YES *"  if row["p-value"] < 0.05  else "no")))
        print(f"  {row['Model']:<25} {row['Coef']:>10.4f} "
              f"{row['p-value']:>10.4f} {sig:>14}")

# =============================================================================
# 8. VISUALISATIONS
# =============================================================================
# plots to compare models and evaluate fit visually, OLS coefficients with confidence intervals across models A–D, predicted vs actual for best count model, ROC curve for logistic model
def plot_ols_coefficients(ols_results: dict) -> None:
    vars_to_plot  = ["snowfall", "log_gdp", "log_population"]
    var_labels    = ["Snowfall (mm)", "log₁₀(GDP, USD)", "log₁₀(Population)"]
    model_labels  = list(ols_results.keys())
    model_colors  = [BLUE, CORAL, TEAL, PURPLE]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "OLS coefficient estimates across model specifications\n"
        "Bars = point estimate  |  Error bars = 95% CI  |  "
        "Dashed line = zero",
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
        for i, (coef, (lo, hi), color) in enumerate(
                zip(coefs, cis, model_colors)):
            ax.barh(
                y_pos[i], coef,
                xerr=[[coef - lo], [hi - coef]],
                color=color, height=0.55, capsize=4, alpha=0.85,
                error_kw={"linewidth": 1.2, "ecolor": "#333"},
            )

        # reference line
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
    # diagnostic plot for best count model (negative binomial if overdispersion, otherwise Poisson)
    # checks how well the model's predicted medal counts match the actual counts, and examines the distribution of residuals to assess fit quality
    best_key = "F" if "F" in count_results else ("E" if "E" in count_results else None)
    if best_key is None:
        print("  [skipping count model plot - no models fitted]")
        return

    m    = count_results[best_key]
    pred = m.predict()
    actual = df["total_medals"].values[:len(pred)]

    # remove nans invalid predictions for plotting
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred   = pred[mask]
    actual = actual[mask]

    if len(pred) == 0:
        print("  [skipping count model plot - no valid predictions]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Count Model {best_key} - Predicted vs Actual Medal Counts",
        fontsize=12, fontweight="bold",
    )

    # Scatter: predicted vs actual
    ax1.scatter(pred, actual, alpha=0.4, s=30, color=BLUE, linewidths=0)
    lim = max(actual.max(), pred.max()) * 1.05

    # 45-degree reference line for perfect predictions
    ax1.plot([0, lim], [0, lim], "--", color=CORAL, linewidth=1.5,
             label="Perfect fit")
    ax1.set_xlabel("Predicted medals")
    ax1.set_ylabel("Actual medals")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(fontsize=9)

    # residual distribution
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
    # roc curve evaluates classification performance auc summarises model quality (0.5 = random, 1.0 = perfect)
    if logit_result is None:
        print("  [skipping ROC plot - logistic model not fitted]")
        return

    y_true, y_prob = y_test_prob_tuple
    if y_true is None:
        print("  [skipping ROC plot - no test predictions available]")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    
    # reference line for random classifier
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1,
            label="Random classifier")
    
    # Mark the optimal threshold point
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color=CORAL, s=100, zorder=5,
        label=f"Optimal threshold = {optimal_threshold:.2f}",
    )
    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)")
    ax.set_title("Model G - ROC Curve\n(won_any_medal binary classification)")
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
# run full pipeline: load data, fit all models, print interpretation summary, save all figures
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    df = load()

    # Step 2–4: Fit all models
    ols_results   = run_ols(df)
    count_results = run_count_models(df)
    logit_result, y_test_prob = run_logistic(df)

    # Step 5: Interpret
    print_interpretation(ols_results, count_results, logit_result)

    # Step 6: Visualise
    print(f"\n\n{'='*60}")
    print("  SAVING FIGURES")
    print(f"{'='*60}")
    plot_ols_coefficients(ols_results)
    plot_count_model_fit(count_results, df)
    plot_roc_curve(logit_result, y_test_prob)

    print(f"\n✓ Analysis complete → {FIG_DIR}/")
    print("  10_model_coefficients.png  - OLS snow coefficient across models")
    print("  11_count_model_fit.png     - NegBin predicted vs actual")
    print("  12_logistic_roc.png        - ROC curve, AUC, optimal threshold")


if __name__ == "__main__":
    main()