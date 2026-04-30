# Pipeline: How to Run the Project

This project follows a reproducible data-engineering pipeline:

```text
Fetch raw data → Clean and integrate data → Descriptive analysis → Statistical modelling → Graph analytics → Interactive visualisation
```

The pipeline is split into separate scripts so that each stage has a clear responsibility. This follows the course idea that data engineering involves collecting, storing, transforming, and analysing data in a structured pipeline. The cleaning step also keeps raw data separate from derived clean data, which supports data lineage and reproducibility.

---

## 1. Fetch raw data

```bash
python fetch.py --skip-snow
```

This script collects the raw input data from the World Bank API, the local Olympics dataset, and Wikipedia scraping. The `--skip-snow` option skips the slow ERA5 snowfall-processing step, which is useful when `snowfall_raw.csv` has already been created before.

Use this instead when you want to run the full collection pipeline, including snowfall:

```bash
python fetch.py
```

---

## 2. Clean and integrate the data

```bash
python clean.py
```

This script reads from `data/raw/`, cleans each source, standardises country identifiers, removes non-country aggregate rows, and creates the final analytical dataset:

```text
data/clean/master.csv
```

This is the main dataset used by the analysis, modelling, graph analytics, and visualisation scripts.

---

## 3. Run descriptive analysis

```bash
python descriptive_analysis.py
```

This script creates exploratory plots and summary tables, such as medal distributions, snowfall distributions, scatterplots, correlations, and medal comparisons by snowfall quartile. The figures are saved in:

```text
data/figures/
```

---

## 4. Run statistical models

```bash
python analyse.py
```

This script answers the main research question:

> To what extent does a country's snowy climate relate to Winter Olympic medal performance, and does that relationship remain after controlling for GDP and population?

It runs the main regression and classification models, including OLS (Models A–C with HC3 robust standard errors), Negative Binomial, and logistic regression. It also saves model-related figures such as coefficient comparisons, count-model fit, and the ROC curve.

---

## 5. Run graph analytics

```bash
python Graph_Analytics.py
```

This script performs supplementary graph analysis. Countries are represented as nodes, and edges connect countries with similar snowfall, GDP, and population profiles. The script calculates centrality metrics, detects communities, and performs shortest-path analysis. This links the project to the graph analytics part of the course.

---

## 6. Create the interactive visualisation

```bash
python visualise.py
```

This script creates an interactive world map showing snowfall and Winter Olympic medal performance by country. The output is saved as:

```text
data/figures/snowfall_medals_map.html
```

Open this file in a browser to view the map.

---

## Full Run Order

Run the scripts in this order:

```bash
python fetch.py --skip-snow
python clean.py
python descriptive_analysis.py
python analyse.py
python Graph_Analytics.py
python visualise.py
```

If the snowfall data has not been processed yet, run this instead at the start:

```bash
python fetch.py
```

---

## Output Structure

After running the full pipeline, the most important outputs are:

```text
data/raw/                              Raw collected data
data/clean/                            Cleaned and integrated data
data/clean/master.csv                  Final analytical dataset
data/figures/                          Static analysis figures
data/figures/graph_analytics/          Graph analytics figures
data/figures/snowfall_medals_map.html  Interactive world map
```