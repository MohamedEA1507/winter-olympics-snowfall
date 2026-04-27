"""
=============================================================================
analyse.py  —  Predictive Modelling Pipeline
Winter Olympics & Snowfall Project
Data Engineering 2025–2026  |  University of Antwerp
=============================================================================

RESEARCH QUESTION
-----------------
To what extent does a country's snowy climate relate to Winter Olympic medal
performance, and does that relationship remain after controlling for GDP
and population?

HYPOTHESES (from project blueprint)
-------------------------------------
H1: Snowier countries win more Winter Olympic medals (positive bivariate
    correlation with snowfall; negative with temperature).
H2: The snow effect attenuates but does not disappear after controlling for
    GDP and population — wealth alone cannot create ski culture or
    infrastructure.
H3: Using medals per million reduces the raw size effect and isolates a
    climate/infrastructure signal more cleanly than total medals.

WHAT THIS FILE DOES
--------------------
This script implements a full, staged predictive modelling pipeline as
described in Section 8 of the project blueprint ("Analysis Plan"). It
follows the ML workflow taught in Lecture 8 (DE_Lecture8_Analytics_NLP_Other):
    1. Problem definition
    2. Data splitting (train/test, stratified)
    3. Data preprocessing (log transforms, standardisation)
    4. Model training — staged progression from simple to complex
    5. Model evaluation — appropriate metrics per model type
    6. Visualisation of results

MODEL OVERVIEW
--------------
We fit 7 models in a deliberate progression (simple → complex):

OLS models  (dependent variable: log(1 + total_medals))
  Model A — Bivariate baseline    : log_medals ~ snowfall
  Model B — Economic controls     : log_medals ~ snowfall + log_gdp + log_pop
  Model C — Year fixed effects    : Model B + year dummies
  Model D — Country fixed effects : Model C + country dummies (within estimator)

Count models  (dependent variable: raw integer total_medals)
  Model E — Poisson regression    : total_medals ~ snowfall + log_gdp + log_pop + year FE
  Model F — Negative Binomial     : same formula, but relaxes Poisson's
                                    variance=mean assumption (better for
                                    overdispersed medal counts)

Binary classification  (dependent variable: won_any_medal = 0 or 1)
  Model G — Logistic regression   : won_any ~ snowfall + log_gdp + log_pop + year FE

WHY THESE MODELS?
-----------------
- OLS with log-transform: standard starting point for skewed count outcomes.
  Taught in Lecture 8 as the linear regression family.
- Fixed effects (C, D): control for Olympic-era trends and stable
  country-level characteristics not in our data (culture, terrain).
- Poisson / NegBin: medal counts are non-negative integers — count models
  are more appropriate than OLS. We check for overdispersion first.
- Logistic: reframes the question as a binary classification task
  (won at least one medal vs did not), which links directly to the
  sklearn workflow taught in Lab Session 5.

DATA REQUIREMENTS
-----------------
Input : data/clean/master.csv  (built by fetch.py and clean.py)
        Expected columns (exact names from master.csv):
          country, year, total_medals, won_any_medal,
          snowfall_mean_gridcell, gdp_per_capita, gdp,
          population, log_gdp, log_population

Outputs:
  Console: model summaries, coefficient tables, fit statistics
  data/figures/10_model_coefficients.png   — OLS coefficient comparison
  data/figures/11_count_model_fit.png      — predicted vs actual (NB model)
  data/figures/12_logistic_roc.png         — ROC curve for Model G

USAGE
-----
  python analyse.py

DEPENDENCIES
------------
  pip install pandas numpy matplotlib statsmodels scikit-learn scipy
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf   # OLS and GLM via R-style formulas
import statsmodels.api as sm            # lower-level GLM / NegBin / Logit
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # Lab 5 pattern
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"

# Colour palette — consistent across all figures
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#BA7517"
GRAY   = "#888780"

# Matplotlib global style — clean, professional appearance
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
    """
    Load and prepare the master analytical table.

    The master.csv was built by the ingest/clean pipeline and already
    contains log-transformed columns (log_gdp, log_population). We add
    the remaining engineered features needed for modelling:
      - log_total_medals  : log(1 + total_medals)  — avoids log(0)
      - won_any_medal     : binary 0/1 flag for the logistic model
      - snowfall          : alias for snowfall_mean_gridcell (mean mm
                            of snow per ERA5 grid cell within the country)
      - year_fe           : year cast to string for use as categorical FE

    Why log-transform medals?
        Medal counts are right-skewed — Norway has 300+, most countries
        have 0. A log transform makes the distribution more symmetric and
        means OLS coefficients represent percentage effects rather than
        absolute ones. log1p(x) = log(1+x) handles zeros gracefully.

    Why log GDP and population?
        They span many orders of magnitude (1e9 to 1e13 for GDP).
        In log-space, an extra year of compound growth is a constant
        additive increment, which matches economic intuition.
    """
    df = pd.read_csv(MASTER)

    # --- Rename for convenience (master.csv already uses correct column names)
    df = df.rename(columns={
        "snowfall_mean_gridcell": "snowfall",
        "gdp_per_capita":         "gdp_per_capita",
    }, errors="ignore")

    # --- Derived modelling columns
    # log1p transform: log(1 + x). Equivalent to log(x) for large values,
    # but defined at x=0. Essential since many countries win zero medals.
    df["log_total_medals"] = np.log1p(df["total_medals"])

    # Binary outcome for logistic regression (Model G).
    # 1 if the country won at least one medal in that edition.
    df["won_any_medal"] = (df["total_medals"] > 0).astype(int)

    # Year as a string so statsmodels treats it as a categorical variable
    # (i.e. creates a dummy for each year) rather than a continuous trend.
    df["year_fe"] = df["year"].astype(str)

    # --- Drop rows missing any column used in modelling
    # Missing snowfall: ERA5 didn't cover some tiny island states.
    # Missing GDP: World Bank data absent for some historical periods.
    # We document what fraction is lost for transparency.
    model_cols = [
        "total_medals", "log_total_medals", "snowfall",
        "log_gdp", "log_population", "won_any_medal", "year_fe", "country",
    ]
    n_before = len(df)
    df_model = df.dropna(subset=model_cols).copy()
    n_after  = len(df_model)
    n_lost   = n_before - n_after

    print(f"\n{'='*60}")
    print("  DATA LOADING")
    print(f"{'='*60}")
    print(f"  Raw rows        : {n_before:,}")
    print(f"  Rows with complete model data: {n_after:,}")
    print(f"  Rows dropped (missing values): {n_lost:,} "
          f"({n_lost/n_before*100:.1f}%)")
    print(f"  Unique countries: {df_model['country'].nunique()}")
    print(f"  Olympic editions: {sorted(df_model['year'].unique())}")

    # --- Overdispersion check
    # For count data, we compare variance to mean. If var >> mean, a simple
    # Poisson model will underestimate standard errors (overdispersion).
    # Rule of thumb: var/mean > 2 suggests we should prefer Negative Binomial.
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

def print_model_summary(label: str, result, extra_note: str = "") -> None:
    """
    Print a compact, readable model summary showing only the substantive
    coefficients (fixed-effect dummies are suppressed for readability).

    Significance stars follow the standard convention:
      *** p < 0.001  (very strong evidence against H0)
      **  p < 0.01   (strong evidence)
      *   p < 0.05   (conventional threshold)
      (blank) not significant at 5% level
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Observation count
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

    # Information criterion — lower AIC = better model
    try:
        print(f"  AIC          : {result.aic:.1f}")
    except Exception:
        pass

    # --- Coefficient table (hiding year/country FE dummies)
    try:
        coef_df = pd.DataFrame({
            "coef":   result.params,
            "se":     result.bse,
            "pvalue": result.pvalues,
        })
        # Keep only substantive predictors — skip fixed-effect dummies
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
    """
    Fit four Ordinary Least Squares models with increasing complexity.

    Dependent variable: log(1 + total_medals)
    Why OLS? It is the simplest and most interpretable starting point.
    The log transform of the DV means coefficients represent approximate
    percentage effects: a coefficient of 0.30 on snowfall means one extra
    unit of snowfall is associated with ~30% more medals.

    The four-model sequence follows the blueprint's "staged" approach:
    each model adds one layer of controls so we can see exactly how the
    snowfall coefficient changes as we account for more confounders.

    Model A — Bivariate (no controls)
        The raw correlation. Expected to be large because wealthy, snowy
        countries (Norway, Finland) dominate, confounding everything.

    Model B — Economic controls (GDP + population)
        Adds log_gdp and log_population. If the snowfall coefficient drops
        dramatically here, much of the bivariate effect was really wealth.
        This tests H2 directly.

    Model C — Year fixed effects (on top of B)
        Adds a dummy variable for each Olympic year. This absorbs:
          - The trend of more events and more nations over time
          - Global shocks (COVID affecting 2022, e.g.)
        Compares countries at the same point in Olympic history.

    Model D — Country + year fixed effects (within estimator)
        Adds a dummy for each country. This is the strictest test:
        only within-country variation over time is used. Since snowfall
        is a slow-moving climate variable (it barely changes year to year
        for a given country), we expect the snowfall coefficient to shrink
        close to zero here. This is not a failure — it confirms that the
        effect is a cross-country story, not a within-country story.

    References: lecture 8 modelling workflow; blueprint Section 8.
    """
    print(f"\n\n{'█'*60}")
    print("  OLS MODELS")
    print("  Dependent variable: log(1 + total_medals)")
    print("  Interpretation: coefficients ≈ % effect on expected medals")
    print(f"{'█'*60}")

    results = {}

    # ── Model A: Bivariate baseline ────────────────────────────────────────
    # The simplest possible model. No controls whatsoever.
    # This tells us: ignoring everything else, does snowfall predict medals?
    mA = smf.ols(
        "log_total_medals ~ snowfall",
        data=df,
    ).fit()
    print_model_summary(
        "Model A — Bivariate: log_medals ~ snowfall",
        mA,
        extra_note="No controls. Coefficient is confounded by GDP and geography."
    )
    results["A"] = mA

    # ── Model B: Add GDP and population ────────────────────────────────────
    # GDP and population are the two most important confounders:
    #   - Rich countries fund more athletes, coaches, and facilities
    #   - Larger countries have a bigger talent pool
    # If the snowfall coefficient drops a lot from A → B, that tells us
    # the bivariate relationship was mainly picking up wealth.
    mB = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population",
        data=df,
    ).fit()
    print_model_summary(
        "Model B — Economic controls: log_medals ~ snowfall + log_gdp + log_pop",
        mB,
        extra_note="Does snowfall matter AFTER accounting for wealth and size?"
    )
    results["B"] = mB

    # ── Model C: Add year fixed effects ────────────────────────────────────
    # C(year_fe) creates one dummy per Olympic year. The reference year
    # (absorbed into the intercept) is the earliest year.
    # This controls for things that changed over all editions simultaneously:
    # more events, more nations participating, geopolitical shifts.
    mC = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population + C(year_fe)",
        data=df,
    ).fit()
    print_model_summary(
        "Model C — Year FE: adds Olympic-year fixed effects",
        mC,
        extra_note="Year FE dummies suppressed. Controls for era-level trends."
    )
    results["C"] = mC

    # ── Model D: Add country fixed effects ─────────────────────────────────
    # Country fixed effects absorb ALL time-invariant country characteristics:
    # geography, culture, winter sport history, terrain, etc.
    # This is the most rigorous specification. It only uses variation within
    # a country across time — e.g. did Finland win more medals in years when
    # it happened to have more snowfall? Since snowfall barely changes year
    # to year, we EXPECT this coefficient to be small. If it is:
    #   → The cross-sectional effect (snowy countries win more) is real,
    #     but it's driven by stable between-country differences, not
    #     within-country year-to-year variation.
    mD = smf.ols(
        "log_total_medals ~ snowfall + log_gdp + log_population "
        "+ C(year_fe) + C(country)",
        data=df,
    ).fit()
    print_model_summary(
        "Model D — Country + Year FE: within-country estimation",
        mD,
        extra_note=(
            "If snowfall coef ≈ 0 here, the effect is BETWEEN countries, "
            "not within. This is expected for a slow-moving climate variable."
        )
    )
    results["D"] = mD

    # ── F-test: do year fixed effects jointly matter? ─────────────────────
    # Compares Model B (no year FE) vs Model C (with year FE).
    # A significant F-test means year dummies jointly explain a meaningful
    # share of variance — i.e. era-level trends are real and worth controlling.
    try:
        ftest = mC.compare_f_test(mB)
        print(f"\n  F-test: year FEs jointly = 0?")
        print(f"  F = {ftest[0]:.2f}  |  p = {ftest[1]:.4f}  "
              f"({'year FE matter' if ftest[1] < 0.05 else 'year FE not significant'})")
    except Exception:
        pass

    # ── Regression diagnostics ────────────────────────────────────────────
    # Heteroskedasticity: OLS assumes constant variance of residuals.
    # If variance grows with fitted values (common with skewed count data),
    # standard errors are biased. We check with the Breusch-Pagan test.
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(mB.resid, mB.model.exog)
        print(f"\n  Breusch-Pagan heteroskedasticity test (Model B):")
        print(f"  LM stat = {bp_test[0]:.3f}  |  p = {bp_test[1]:.4f}  "
              f"({'heteroskedastic — consider robust SEs' if bp_test[1] < 0.05 else 'homoskedastic'})")
    except Exception:
        pass

    # VIF: Variance Inflation Factor measures multicollinearity.
    # VIF > 10 suggests a predictor is nearly a linear combination of others,
    # which inflates standard errors and makes coefficients unreliable.
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

def run_count_models(df: pd.DataFrame) -> dict:
    """
    Fit count regression models using raw integer medal counts as the DV.

    Why count models instead of OLS?
        Medal counts are non-negative integers that arise from a discrete
        counting process. OLS treats the DV as continuous and can predict
        negative values. Count models respect the data-generating process.

    Model E — Poisson GLM
        The Poisson model assumes Var(Y) = E(Y) (variance equals mean).
        This is often violated in practice. We check with the Pearson χ²/df
        statistic: if it exceeds 2, the data are overdispersed and Poisson
        underestimates standard errors → use Negative Binomial instead.

    Model F — Negative Binomial
        Adds a dispersion parameter α that allows Var(Y) = μ + α·μ².
        When α → 0, NegBin reduces to Poisson. A large α confirms
        overdispersion. This is almost always the better fit for medal data
        because a few dominant countries create a very heavy right tail.

    Coefficients in both models are log rate ratios:
        exp(coef) = multiplicative effect on expected medal count.
        Example: exp(0.30) ≈ 1.35 means 35% more expected medals
        per unit increase in the predictor.

    Year fixed effects are included to control for era-level trends.
    We use pd.get_dummies() to create them explicitly (required by
    statsmodels' lower-level GLM interface).
    """
    print(f"\n\n{'█'*60}")
    print("  COUNT MODELS")
    print("  Dependent variable: total_medals (raw integer count)")
    print("  Interpretation: exp(coef) = multiplicative effect on medal count")
    print(f"{'█'*60}")

    results = {}

    # Build design matrix with year dummies
    # drop_first=True avoids perfect multicollinearity (dummy variable trap)
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X_base = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_base = sm.add_constant(X_base)          # add intercept column
    y = df["total_medals"].reset_index(drop=True)

    # ── Model E: Poisson ──────────────────────────────────────────────────
    print("\n  Fitting Poisson regression...")
    try:
        mE = sm.GLM(y, X_base, family=sm.families.Poisson()).fit()
        print_model_summary(
            "Model E — Poisson: total_medals ~ snowfall + log_gdp + log_pop + year FE",
            mE,
        )
        # Goodness-of-fit: Pearson χ²/df
        # If this >> 1, residual variance exceeds what Poisson predicts.
        pearson_ratio = mE.pearson_chi2 / mE.df_resid
        print(f"\n  Pearson χ²/df = {pearson_ratio:.2f}  "
              f"({'overdispersed → NegBin preferred' if pearson_ratio > 2 else 'acceptable'})")

        # Exponentiated coefficients for interpretation
        print(f"\n  Key exp(coef) — multiplicative effects on medal count:")
        for var in ["snowfall", "log_gdp", "log_population"]:
            if var in mE.params.index:
                print(f"  exp({var:<20}) = {np.exp(mE.params[var]):.3f}  "
                      f"(p={mE.pvalues[var]:.4f})")
        results["E"] = mE
    except Exception as e:
        print(f"  [Poisson failed: {e}]")

    # ── Model F: Negative Binomial ─────────────────────────────────────────
    print("\n  Fitting Negative Binomial regression...")
    try:
        mF = sm.NegativeBinomial(y, X_base).fit(disp=False)
        print_model_summary(
            "Model F — Negative Binomial: total_medals ~ snowfall + log_gdp + log_pop + year FE",
            mF,
        )
        # Dispersion parameter α
        alpha = mF.params.get("alpha", np.nan)
        print(f"\n  Dispersion parameter α = {alpha:.4f}  "
              f"({'significant overdispersion → NegBin is correct choice' if alpha > 0.1 else 'mild overdispersion'})")

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
# 6. LOGISTIC REGRESSION MODEL (Model G)
# =============================================================================

def run_logistic(df: pd.DataFrame):
    """
    Fit a logistic regression predicting whether a country won any medal.

    Why logistic regression?
        Medal counts are severely zero-inflated (most country-edition pairs
        have zero medals). Logistic regression reframes the question as a
        binary classification: "does this country win at least one medal?"
        This is a more tractable problem and links directly to the
        sklearn classification workflow taught in Lab Session 5.

    The logistic model works by modelling the log-odds:
        log(p / (1-p)) = β₀ + β₁·snowfall + β₂·log_gdp + ...
    which maps to probabilities via the sigmoid function:
        p = 1 / (1 + exp(−(β₀ + β₁·snowfall + ...)))
    Predictions are always in [0,1], unlike OLS.

    Interpretation:
        exp(β) = odds ratio. exp(β) = 1.4 means the odds of winning any
        medal are 40% higher per unit increase in the predictor.

    Evaluation:
        We use the ROC curve and AUC (Area Under Curve) as the primary
        metric. AUC = 0.5 is random; AUC = 1.0 is perfect. This is
        explicitly covered in Lecture 8 (classification metrics) and
        Lab Session 5 (roc_auc_score from sklearn).

    Train/test split:
        Following the workflow in Lab Session 5, we split into 80% train
        and 20% test. Stratified sampling ensures both splits have the
        same proportion of medal-winners (important for imbalanced data).
    """
    print(f"\n\n{'█'*60}")
    print("  LOGISTIC REGRESSION MODEL")
    print("  Dependent variable: won_any_medal (0 = no medals, 1 = any medal)")
    print("  Interpretation: exp(coef) = odds ratio")
    print(f"{'█'*60}")

    # Build design matrix (same structure as count models)
    year_dummies = pd.get_dummies(df["year_fe"], prefix="yr", drop_first=True)
    X = pd.concat([
        df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True),
        year_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X_with_const = sm.add_constant(X)
    y = df["won_any_medal"].reset_index(drop=True)

    try:
        # --- Full-sample statsmodels fit for coefficients and odds ratios
        # Use only the three substantive predictors (no year dummies)
        # to avoid the near-singular matrix caused by sparse year cells
        X_simple = sm.add_constant(
            df[["snowfall", "log_gdp", "log_population"]].reset_index(drop=True)
        )
        mG = sm.Logit(y, X_simple).fit(disp=False)
        print_model_summary(
            "Model G — Logistic: won_any_medal ~ snowfall + log_gdp + log_pop",
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

        # ── Train/test evaluation (sklearn pattern from Lab 5) ──────────
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
        # Matches the evaluation pattern in Lab Session 5
        print(f"\n  Classification report (threshold = 0.5, test set):")
        print(classification_report(
            y_test, y_pred,
            target_names=["no medal", "any medal"],
        ))

        # AUC — the primary evaluation metric
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

def print_interpretation(
    ols_results: dict,
    count_results: dict,
    logit_result,
) -> None:
    """
    Print a cross-model comparison of the key snowfall coefficient.

    This answers the core research question directly:
    "Does snowfall predict medals, and does the effect survive controls?"

    If the snowfall coefficient is:
    - Positive and significant in Model A but disappears in Model B:
      → The bivariate effect was entirely driven by GDP (H2 rejected)
    - Positive and significant through Models B and C:
      → The effect is robust to economic controls and era trends (H2 supported)
    - Near zero in Model D:
      → The effect is a BETWEEN-country story (stable climate advantage),
        not a within-country time-series effect (expected and interpretable)
    """
    print(f"\n\n{'█'*60}")
    print("  INTERPRETATION SUMMARY")
    print("  Snowfall coefficient across all 7 models")
    print(f"{'█'*60}")

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

    print("""
  HOW TO READ THESE RESULTS
  ──────────────────────────
  OLS Model A  (bivariate, no controls)
    → Raw association. Expected to be inflated because wealthy snowy
      countries dominate both the climate and medal tables.

  OLS Model B  (+ GDP + population)
    → Tests H2: does snowfall matter AFTER accounting for economic scale?
    → If coef drops dramatically from A to B: GDP was doing most of the
      work. If coef survives: snow has an independent signal.

  OLS Model C  (+ year fixed effects)
    → Controls for global trends in Olympic history.
    → Compares countries at the same point in Olympic time.

  OLS Model D  (+ country fixed effects = within estimator)
    → Most rigorous OLS specification.
    → Uses only within-country year-to-year variation.
    → Snowfall barely changes within a country over time, so a small
      coefficient HERE IS EXPECTED and confirms the effect is a
      stable, cross-sectional (between-country) phenomenon.

  Count Models E & F  (Poisson / Negative Binomial)
    → More statistically appropriate for count data.
    → NegBin preferred if overdispersion (var/mean) > 2.
    → Coefficients are log rate ratios: exp(coef) = multiplicative effect.

  Logistic Model G  (binary: won any medal?)
    → Classification framing: can we predict which countries medal?
    → Evaluated by AUC (Lab 5 pattern). AUC > 0.8 = good discrimination.
    """)


# =============================================================================
# 8. VISUALISATIONS
# =============================================================================

def plot_ols_coefficients(ols_results: dict) -> None:
    """
    Coefficient plot across OLS Models A–D for the three key predictors.

    Each bar shows the coefficient estimate. Error bars show the 95%
    confidence interval. If the CI crosses zero, the effect is not
    statistically significant at the 5% level.

    This directly answers: "How does the snowfall coefficient change as
    we add more controls?" — the key question of the project.

    Plot style follows DE_Lecture9_Visualisation best practices:
    - Meaningful title stating the conclusion
    - Axis labels with units
    - Colour-blind-friendly palette
    - Horizontal reference line at zero
    """
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
    """
    Two-panel diagnostic plot for the best-fitting count model (NegBin > Poisson).

    Left panel: Predicted vs actual medal counts.
        Points on the 45° dashed line = perfect prediction.
        Points above the line = model underpredicts.
        Systematic patterns indicate model misspecification.

    Right panel: Residual distribution.
        Well-specified model → residuals centred at zero, roughly symmetric.
        Skewed residuals → the model consistently over/underpredicts
        in a systematic direction.
    """
    # Use Negative Binomial if available, else Poisson
    best_key = "F" if "F" in count_results else ("E" if "E" in count_results else None)
    if best_key is None:
        print("  [skipping count model plot — no models fitted]")
        return

    m    = count_results[best_key]
    pred = m.predict()
    actual = df["total_medals"].values[:len(pred)]

    # Guard against NaN predictions (can occur if NegBin had convergence issues)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred   = pred[mask]
    actual = actual[mask]

    if len(pred) == 0:
        print("  [skipping count model plot — no valid predictions]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Count Model {best_key} — Predicted vs Actual Medal Counts",
        fontsize=12, fontweight="bold",
    )

    # Scatter: predicted vs actual
    ax1.scatter(pred, actual, alpha=0.4, s=30, color=BLUE, linewidths=0)
    lim = max(actual.max(), pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=CORAL, linewidth=1.5,
             label="Perfect fit")
    ax1.set_xlabel("Predicted medals")
    ax1.set_ylabel("Actual medals")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(fontsize=9)

    # Residuals: actual − predicted
    resid = actual - pred
    ax2.hist(resid, bins=40, color=TEAL, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=CORAL, linewidth=1.5, linestyle="--",
                label="Zero residual")
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
    """
    ROC (Receiver Operating Characteristic) curve for Model G.

    The ROC curve plots:
        x-axis: False Positive Rate (1 - Specificity)
                = fraction of non-medalling countries wrongly classified as medalling
        y-axis: True Positive Rate (Sensitivity / Recall)
                = fraction of medalling countries correctly identified

    At each probability threshold, we get a (FPR, TPR) point. The curve
    traces all possible thresholds. A perfect model goes straight to the
    top-left corner. A random classifier follows the diagonal.

    AUC (Area Under Curve):
        Single-number summary of the ROC curve.
        AUC = 0.5: random (no information)
        AUC = 0.7–0.8: acceptable
        AUC > 0.8: good
        AUC > 0.9: excellent

    This evaluation approach is directly taught in Lecture 8 (classification
    metrics section) and implemented in Lab Session 5 using sklearn.
    """
    if logit_result is None:
        print("  [skipping ROC plot — logistic model not fitted]")
        return

    y_true, y_prob = y_test_prob_tuple
    if y_true is None:
        print("  [skipping ROC plot — no test predictions available]")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold: maximises TPR - FPR (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
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
    """
    Runs the full modelling pipeline in sequence:
      1. Load and prepare data
      2. Fit OLS models A–D
      3. Fit count models E–F
      4. Fit logistic model G
      5. Print cross-model interpretation
      6. Save all figures

    The pipeline follows the ML workflow from Lecture 8:
      Problem definition → Data split → Preprocessing →
      Model training → Evaluation → Visualisation
    """
    # Ensure output directory exists
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
    print("  10_model_coefficients.png  — OLS snow coefficient across models")
    print("  11_count_model_fit.png     — NegBin predicted vs actual")
    print("  12_logistic_roc.png        — ROC curve, AUC, optimal threshold")


if __name__ == "__main__":
    main()