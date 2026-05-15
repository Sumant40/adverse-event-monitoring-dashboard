# Pharmacovigilance Signal Detection — v2

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Domain](https://img.shields.io/badge/Domain-Pharmacovigilance-teal)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

This project analyses FDA Adverse Event Reporting System (FAERS) data to detect potential drug safety signals — directly mirroring the kind of work done in Amgen's Global Safety and pharmacovigilance functions.

The pipeline cleans adverse event records, applies **Proportional Reporting Ratio (PRR)** and **Reporting Odds Ratio (ROR)** disproportionality analysis, tracks how signals evolve over a drug's post-market lifecycle, and models serious-outcome predictors with logistic regression. Results surface through an interactive four-tab Streamlit dashboard.

## Business Problem

Pharmacovigilance teams receive large volumes of spontaneous adverse event reports. Manual triage is slow. This project answers:

> **Which drug-reaction combinations show disproportionate reporting, how has that reporting changed over time, and which signals should be prioritised for safety review?**

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Disproportionality metric | PRR only | **PRR + ROR**, both with 95 % CI |
| Time-series analysis | ✗ | **Quarterly case trends** for top-50 signals |
| Outcome prediction | ✗ | **Logistic regression** — odds ratios by drug/reaction/demographics |
| Biologic tagging | ✗ | **Biologic flag** on every drug, filterable in dashboard |
| Dashboard tabs | 1 (signal table + bar chart) | **4 tabs** — Explorer, Time-Series, Outcome Model, Summary |
| PRR vs ROR scatter | ✗ | **Log-scale scatter** for visual cross-metric validation |
| Annual reporting trend | ✗ | **Year-over-year outcome breakdown** |
| Code structure | Notebooks only | **`src/` modules** with clean separation of concerns |

## Dataset

- **Source:** Kaggle FAERS adverse event dataset
- **Link:** https://www.kaggle.com/datasets/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers
- **Cleaned records:** 528,000
- **Serious-event subset:** ~219,000
- **Years covered:** 2015–2025

## Project Structure

```
pharma_signal_v2/
├── app/
│   └── streamlit_app.py          # 4-tab Streamlit dashboard
├── data/
│   ├── faers_raw.csv
│   ├── faers_clean.csv
│   └── faers_serious.csv
├── notebooks/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   └── 03_signal_detection.py
├── outputs/
│   ├── figures/
│   │   ├── top_10_drugs_by_adverse_events.html
│   │   ├── outcome_distribution.html
│   │   ├── outcome_by_age_group.html
│   │   ├── annual_trend_by_outcome.html       # NEW
│   │   └── biologic_vs_small_molecule_outcomes.html  # NEW
│   └── signals/
│       ├── flagged_signals.csv                # PRR + ROR + CI + priority
│       ├── timeseries_signals.csv             # NEW
│       └── outcome_model_coefs.csv            # NEW
├── src/
│   ├── cleaning.py               # All data cleaning logic
│   └── signal_detection.py      # PRR, ROR, time-series, logistic model
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Cleaning (`src/cleaning.py`)

- Column renaming and text standardisation
- Age group mapping to readable bands (`<2`, `2-11`, `12-17`, `18-64`, `65-85`, `>85`)
- Outcome code derivation with priority ordering (`DE > LT > HO > DS > OT`)
- Biologic drug tagging against a curated keyword list (TNF inhibitors, checkpoint inhibitors, mAbs)
- Deduplication on report-drug-reaction triples
- Serious-event subset extraction

### 2. Signal Detection (`src/signal_detection.py`)

#### PRR

```
PRR = [a / (a + b)] / [c / (c + d)]
```

#### ROR

```
ROR = (a × d) / (b × c)
```

Where `a`, `b`, `c`, `d` are cells of the drug × reaction contingency table.

Both metrics use the **delta method** for log-scale 95 % confidence intervals.

**Flagging criteria** (unchanged from v1, consistent with Evans et al. 2001):

| Criterion | Threshold |
|---|---|
| Minimum cases | ≥ 3 |
| PRR | > 2 |
| Lower 95 % CI (PRR) | > 1 |

**Priority rules:**

| Priority | Rule |
|---|---|
| High | PRR > 3 and cases ≥ 10 |
| Medium | PRR > 2 |

#### Time-Series

For the top 50 high-priority signals, quarterly case counts are extracted to show post-market reporting evolution. A 4-quarter rolling average is overlaid in the dashboard.

#### Logistic Regression (Outcome Model)

A logistic regression model is trained on the full cleaned dataset to predict **serious** (hospitalisation, death, life-threatening, disability) vs **non-serious** outcomes.

Features: biologic flag, sex, age group, top-20 drug dummies, top-20 reaction dummies.

Output: feature coefficients and odds ratios, surfaced in the dashboard.

## Results

| Metric | Result |
|---|---:|
| Cleaned reports | 528,000 |
| Serious reports | ~219,000 |
| Years covered | 2015–2025 |
| Flagged signals | 8,920 |
| High-priority | 1,403 |
| Medium-priority | 7,517 |

### Highest PRR Signals (sample)

| Drug | Reaction | Cases | PRR | ROR | Priority |
|---|---|---:|---:|---:|---|
| IDELVION | FACTOR IX INHIBITION | 3 | 65,690 | — | Medium |
| MONTELUKAST SODIUM | NEUROPSYCHOLOGICAL SYMPTOMS | 31 | 54,713 | — | High |
| BUPIVACAINE HCL | FOETAL EXPOSURE DURING DELIVERY | 23 | 47,502 | — | High |
| DINUTUXIMAB | NEUROBLASTOMA RECURRENT | 21 | 41,421 | — | High |
| LEVONORGESTREL | UTERINE PERFORATION | 133 | 31,183 | — | High |

> High PRR/ROR values indicate disproportionate reporting, **not causality**. All signals are candidates for clinical and regulatory review.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

| Tab | Contents |
|---|---|
| Signal Explorer | Drug/priority/biologic filters; signal table; PRR bar chart; PRR vs ROR scatter |
| Time-Series | Quarterly case trend + 4-quarter rolling average for any flagged signal |
| Outcome Model | Odds ratio bar chart from logistic regression |
| Summary Stats | Headline metrics, top-drug/reaction charts, biologic vs small-molecule breakdown |

## How to Run

```bash
pip install -r requirements.txt

# Run notebooks in order (as .py files with a Jupyter-compatible runner,
# or convert to .ipynb with jupytext)
python notebooks/01_data_cleaning.py
python notebooks/02_eda.py
python notebooks/03_signal_detection.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
scikit-learn>=1.3.0      # new in v2 — outcome model
streamlit>=1.28.0
```

## Limitations

- FAERS is a spontaneous reporting system: under-reporting, stimulated reporting, and reporting bias affect all analyses.
- PRR and ROR detect disproportionality, not causality.
- Drug names are standardised with text normalisation only — not mapped to RxNorm or WHODrug.
- Reaction terms are not mapped to MedDRA hierarchy.
- The logistic regression is a descriptive model, not a causal one.

## Future Work

- RxNorm / WHODrug drug normalisation
- MedDRA PT → HLT → SOC hierarchy mapping
- Bayesian signal detection (BCPNN, GPS/MGPS)
- Country-level and demographic subgroup stratification
- Sensitivity analysis: ROR stability vs PRR at low case counts
- MLflow experiment tracking for the outcome model

## Author

**Sumant Jadiyappagoudar**  
Bioengineering graduate | Data Science & Computational Biology  
[Email](mailto:sumantjadiyappagoudar@gmail.com)