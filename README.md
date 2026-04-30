# Pharmacovigilance Signal Detection - FDA Adverse Event Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Domain](https://img.shields.io/badge/Domain-Pharmacovigilance-teal)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Project Overview

This project analyzes FDA Adverse Event Reporting System (FAERS) data to identify potential drug safety signals. The workflow cleans adverse event records, explores reporting patterns, and applies proportional reporting ratio (PRR) analysis to flag drug-reaction pairs that are reported more often than expected.

The final output is a ranked signal table and a lightweight Streamlit dashboard that helps review flagged drug-reaction combinations by drug and PRR threshold.

## Business Problem

Pharmacovigilance teams receive large volumes of adverse event reports. Reviewing every record manually is slow and difficult to prioritize. This project answers:

**Which drug-reaction combinations show disproportionate reporting patterns, and which of those signals should be prioritized for safety review?**

## Dataset

- **Source:** Kaggle FAERS adverse event dataset
- **Dataset link:** https://www.kaggle.com/datasets/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers
- **Local cleaned dataset:** 528,000 reports
- **Serious-event subset:** 218,977 reports
- **Years covered in local sample:** 2015-2025
- **Key fields:** report ID, report date, drug name, reaction, outcome flags, patient age, sex, country, manufacturer, route, and indication

## Project Structure

```text
pharma ds project/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- faers_raw.csv
|   |-- faers_clean.csv
|   `-- faers_serious.csv
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_eda.ipynb
|   `-- 03_signal_detection.ipynb
|-- outputs/
|   |-- figures/
|   |   |-- outcome_by_age_group.html
|   |   |-- outcome_distribution.html
|   |   |-- streamlit output.pdf
|   |   `-- top_10_drugs_by_adverse_events.html
|   `-- signals/
|       `-- flagged_signals.csv
|-- src/
|-- requirements.txt
`-- README.md
```

## Methodology

### 1. Data Cleaning

The raw FAERS data was standardized and converted into an analysis-ready format.

Key steps:

- Renamed source fields to consistent project names
- Standardized drug names and reactions to uppercase text
- Created outcome codes from serious-event flags
- Converted patient age and report dates into usable formats
- Removed duplicate report-drug-reaction combinations
- Created a serious-event subset for signal detection

Outcome codes used in the project:

| Code | Meaning |
|---|---|
| DE | Death |
| HO | Hospitalised |
| LT | Life-threatening |
| DS | Disabled |
| OT | Other |

### 2. Exploratory Data Analysis

The EDA notebook generates interactive Plotly charts for:

- Top 10 drugs by adverse event reports
- Overall outcome distribution
- Outcome distribution by age group

These charts are saved as HTML files under `outputs/figures/`.

### 3. Signal Detection

The signal detection notebook applies proportional reporting ratio analysis on the serious-event subset.

```text
PRR = (a / (a + b)) / (c / (c + d))
```

Where:

- `a` = reports with drug X and reaction Y
- `b` = reports with drug X but not reaction Y
- `c` = reports without drug X but with reaction Y
- `d` = reports without drug X and without reaction Y

A drug-reaction pair is flagged when:

- cases >= 3
- PRR > 2
- lower 95% confidence interval > 1

Priority rules:

| Priority | Rule |
|---|---|
| High | PRR > 3 and cases >= 10 |
| Medium | PRR > 2 |

## Results

### Data Summary

| Metric | Result |
|---|---:|
| Cleaned reports analyzed | 528,000 |
| Serious reports used for PRR analysis | 218,977 |
| Years represented | 2015-2025 |
| Flagged safety signals | 8,920 |
| High-priority signals | 1,403 |
| Medium-priority signals | 7,517 |

### Outcome Distribution

| Outcome | Reports |
|---|---:|
| Other | 303,157 |
| Hospitalised | 160,047 |
| Death | 54,301 |
| Disabled | 5,866 |
| Life-threatening | 4,629 |

Hospitalisation was the most common serious outcome in the local dataset, followed by death and life-threatening events.

### Top Drugs by Report Count

| Rank | Drug | Reports |
|---:|---|---:|
| 1 | TOFACITINIB | 13,807 |
| 2 | RISPERIDONE | 13,487 |
| 3 | RIVAROXABAN | 12,968 |
| 4 | AVANDIA | 10,091 |
| 5 | ETANERCEPT | 8,751 |
| 6 | DUPILUMAB | 8,246 |
| 7 | ADALIMUMAB | 8,131 |
| 8 | SODIUM OXYBATE | 8,008 |
| 9 | VEDOLIZUMAB | 7,875 |
| 10 | PREGABALIN | 7,845 |

### Most Common Reactions

| Rank | Reaction | Reports |
|---:|---|---:|
| 1 | DRUG INEFFECTIVE | 10,437 |
| 2 | DEATH | 9,371 |
| 3 | PNEUMONIA | 5,510 |
| 4 | DRUG HYPERSENSITIVITY | 5,492 |
| 5 | MYOCARDIAL INFARCTION | 5,317 |
| 6 | GYNAECOMASTIA | 5,153 |
| 7 | CHRONIC KIDNEY DISEASE | 5,083 |
| 8 | OFF LABEL USE | 4,934 |
| 9 | GASTROINTESTINAL HAEMORRHAGE | 4,802 |
| 10 | CEREBROVASCULAR ACCIDENT | 4,096 |

### Highest PRR Signals

The generated `outputs/signals/flagged_signals.csv` file contains 8,920 flagged drug-reaction pairs. Examples from the top ranked results:

| Drug | Reaction | Cases | PRR | Lower CI | Priority |
|---|---|---:|---:|---:|---|
| IDELVION | FACTOR IX INHIBITION | 3 | 65,690.10 | 7,450.37 | Medium |
| MONTELUKAST SODIUM | NEUROPSYCHOLOGICAL SYMPTOMS | 31 | 54,713.25 | 7,527.34 | High |
| BUPIVACAINE HYDROCHLORIDE | FOETAL EXPOSURE DURING DELIVERY | 23 | 47,502.38 | 11,486.34 | High |
| DINUTUXIMAB | NEUROBLASTOMA RECURRENT | 21 | 41,421.08 | 12,906.89 | High |
| LEVONORGESTREL | UTERINE PERFORATION | 133 | 31,183.07 | 4,364.81 | High |
| MAKENA | UTERINE CONTRACTIONS DURING PREGNANCY | 13 | 9,676.16 | 2,202.84 | High |

High PRR values indicate disproportionate reporting, not proof of causality. These signals should be treated as candidates for clinical and regulatory review.

### Drugs With the Most Flagged Signals

| Drug | Flagged Signals |
|---|---:|
| ADALIMUMAB | 178 |
| RIVAROXABAN | 155 |
| TOFACITINIB | 120 |
| NIVOLUMAB | 113 |
| PREGABALIN | 105 |
| ETANERCEPT | 94 |
| HUMAN IMMUNOGLOBULIN G | 91 |
| VEDOLIZUMAB | 87 |
| RITUXIMAB | 83 |
| LENALIDOMIDE | 82 |

## Dashboard

The Streamlit app loads `outputs/signals/flagged_signals.csv` and provides:

- Drug selector
- Minimum PRR filter
- Filtered signal table
- PRR bar chart by reaction

Run it with:

```bash
streamlit run app/streamlit_app.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

```text
1. notebooks/01_data_cleaning.ipynb
2. notebooks/02_eda.ipynb
3. notebooks/03_signal_detection.ipynb
```

Launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Requirements

```text
pandas==2.3.3
numpy==2.3.4
plotly==6.3.1
matplotlib==3.10.7
seaborn==0.13.2
scipy==1.16.2
streamlit==1.51.0
```

## Key Takeaways

- The project successfully converts raw FAERS records into a cleaned analysis dataset and a serious-event subset.
- PRR analysis identified 8,920 disproportionate drug-reaction reporting patterns.
- 1,403 signals met the high-priority rule of PRR > 3 with at least 10 cases.
- Interactive Plotly outputs and a Streamlit dashboard make the results easier to inspect.
- The workflow reflects a practical pharmacovigilance approach for prioritizing potential safety signals.

## Limitations

- FAERS is a spontaneous reporting system and is affected by under-reporting, duplicate reports, reporting bias, and stimulated reporting.
- PRR detects disproportionality, not causality.
- Drug names are standardized with text cleaning, but not fully normalized against RxNorm or another controlled drug dictionary.
- Reaction terms are not mapped to a full MedDRA hierarchy in this version.
- Clinical interpretation requires medical, regulatory, and epidemiological review.

## Future Work

- Add RxNorm or WHODrug-based drug normalization
- Add MedDRA hierarchy mapping for reaction grouping
- Add more robust duplicate detection
- Expand the Streamlit dashboard with trend, demographic, and country-level filters
- Add sensitivity analysis using other signal detection metrics such as ROR or Bayesian methods

## Author

**Sumant Jadiyappagoudar**  
Bioengineering graduate | Data Science & Computational Biology  
[Email](mailto:sumantjadiyappagoudar@gmail.com)
