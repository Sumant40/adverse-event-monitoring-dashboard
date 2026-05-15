"""
src/eda.py
----------
EDA functions for FAERS cleaned data.
Called by notebooks/02_eda.ipynb.

All functions return a pandas DataFrame or Series — no plotting here.
Plotting lives in the notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append("..")
CLEAN_PATH   = Path("..\\data\\faers_clean.csv")
SERIOUS_PATH = Path("..\\data\\faers_serious.csv")

OUTCOME_MAP = {
    "DE": "Death",
    "HO": "Hospitalised",
    "LT": "Life-threatening",
    "DS": "Disabled",
    "OT": "Other",
}

BIOLOGIC_KEYWORDS = [
    "ADALIMUMAB", "ETANERCEPT", "INFLIXIMAB", "CERTOLIZUMAB",
    "GOLIMUMAB", "TOCILIZUMAB", "ABATACEPT", "RITUXIMAB",
    "VEDOLIZUMAB", "USTEKINUMAB", "SECUKINUMAB", "IXEKIZUMAB",
    "RISANKIZUMAB", "GUSELKUMAB", "NIVOLUMAB", "PEMBROLIZUMAB",
    "ATEZOLIZUMAB", "IPILIMUMAB", "TRASTUZUMAB", "BEVACIZUMAB",
    "DENOSUMAB", "DUPILUMAB", "OMALIZUMAB", "MEPOLIZUMAB",
    "DINUTUXIMAB", "BLINATUMOMAB",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    df["report_date"] = pd.to_datetime(df.get("receive_date", df.get("report_date")), errors="coerce")
    df["year"] = df["report_date"].dt.year
    return df


def load_serious() -> pd.DataFrame:
    return pd.read_csv(SERIOUS_PATH, low_memory=False)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> pd.Series:
    """High-level counts printed at the top of the EDA notebook."""
    drug_col     = "drugname" if "drugname" in df.columns else "drug"
    reaction_col = "reaction"
    outcome_col  = "outc_cod" if "outc_cod" in df.columns else "outcome"

    return pd.Series({
        "total_reports":      len(df),
        "unique_drugs":       df[drug_col].nunique(),
        "unique_reactions":   df[reaction_col].nunique() if reaction_col in df.columns else None,
        "date_min":           df["report_date"].min() if "report_date" in df.columns else None,
        "date_max":           df["report_date"].max() if "report_date" in df.columns else None,
        "pct_missing_age":    round(df["patient_age_years"].isna().mean() * 100, 1)
                              if "patient_age_years" in df.columns else None,
        "pct_missing_weight": round(df["patient_weight_kg"].isna().mean() * 100, 1)
                              if "patient_weight_kg" in df.columns else None,
    })


# ---------------------------------------------------------------------------
# Top-N tables (used for bar charts)
# ---------------------------------------------------------------------------

def top_drugs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N drugs by adverse event report count."""
    drug_col = "drugname" if "drugname" in df.columns else "drug"
    return (
        df[drug_col]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={drug_col: "drug", "count": "reports"})
    )


def top_reactions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N reactions by report count."""
    return (
        df["reaction"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"reaction": "reaction", "count": "reports"})
    )


def top_manufacturers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N manufacturers by report count."""
    if "manufacturer" not in df.columns:
        return pd.DataFrame()
    return (
        df["manufacturer"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"manufacturer": "manufacturer", "count": "reports"})
    )


# ---------------------------------------------------------------------------
# Outcome distribution
# ---------------------------------------------------------------------------

def outcome_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of reports per outcome code with readable label.
    Works whether the column is 'outc_cod' or 'outcome'.
    """
    col = "outc_cod" if "outc_cod" in df.columns else "outcome"
    counts = df[col].value_counts().reset_index()
    counts.columns = ["outcome_code", "reports"]
    counts["outcome_label"] = counts["outcome_code"].map(OUTCOME_MAP).fillna(counts["outcome_code"])
    return counts.sort_values("reports", ascending=False)


def outcome_by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins patient_age_years into age groups, then cross-tabs with outcome.
    """
    outcome_col = "outc_cod" if "outc_cod" in df.columns else "outcome"
    if "patient_age_years" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["age_group"] = pd.cut(
        pd.to_numeric(df["patient_age_years"], errors="coerce"),
        bins=[-1, 2, 11, 17, 64, 85, float("inf")],
        labels=["<2", "2-11", "12-17", "18-64", "65-85", ">85"],
    )
    df["age_group"] = df["age_group"].cat.add_categories("Unknown").fillna("Unknown")

    result = (
        df.groupby(["age_group", outcome_col], observed=False)
        .size()
        .reset_index(name="reports")
        .rename(columns={outcome_col: "outcome_code"})
    )
    result["outcome_label"] = result["outcome_code"].map(OUTCOME_MAP).fillna(result["outcome_code"])

    age_order = ["<2", "2-11", "12-17", "18-64", "65-85", ">85", "Unknown"]
    result["age_group"] = pd.Categorical(result["age_group"], categories=age_order, ordered=True)
    return result.sort_values("age_group")


# ---------------------------------------------------------------------------
# Temporal trend
# ---------------------------------------------------------------------------

def annual_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual report counts broken down by outcome.
    Filters to 2015–2025.
    """
    outcome_col = "outc_cod" if "outc_cod" in df.columns else "outcome"
    if "year" not in df.columns:
        df = df.copy()
        df["year"] = pd.to_datetime(df.get("receive_date", df.get("report_date")), errors="coerce").dt.year

    result = (
        df[df["year"].between(2015, 2025)]
        .groupby(["year", outcome_col])
        .size()
        .reset_index(name="reports")
        .rename(columns={outcome_col: "outcome_code"})
    )
    result["outcome_label"] = result["outcome_code"].map(OUTCOME_MAP).fillna(result["outcome_code"])
    return result


def quarterly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly total report counts — used for the time-series overview.
    Uses pre-existing 'quarter' column with values like '2015Q1', '2015Q2', etc.
    """
    if "quarter" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby("quarter")
        .size()
        .reset_index(name="reports")
        .sort_values("quarter")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Biologic vs small-molecule
# ---------------------------------------------------------------------------

def tag_biologics(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_biologic column if not already present."""
    df = df.copy()
    if "is_biologic" not in df.columns:
        drug_col = "drugname" if "drugname" in df.columns else "drug"
        pattern = "|".join(BIOLOGIC_KEYWORDS)
        df["is_biologic"] = df[drug_col].str.contains(pattern, na=False, regex=True)
    return df


def biologic_vs_small_molecule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report counts split by drug class (biologic / small molecule) and outcome.
    """
    df = tag_biologics(df)
    outcome_col = "outc_cod" if "outc_cod" in df.columns else "outcome"

    result = (
        df.groupby(["is_biologic", outcome_col])
        .size()
        .reset_index(name="reports")
        .rename(columns={outcome_col: "outcome_code"})
    )
    result["drug_class"] = result["is_biologic"].map({True: "Biologic", False: "Small Molecule"})
    result["outcome_label"] = result["outcome_code"].map(OUTCOME_MAP).fillna(result["outcome_code"])
    return result


# ---------------------------------------------------------------------------
# Sex & demographic breakdown
# ---------------------------------------------------------------------------

def sex_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Report counts by patient sex."""
    if "patient_sex" not in df.columns:
        return pd.DataFrame()
    sex_map = {"1": "Male", "2": "Female", "0": "Unknown",
               "Male": "Male", "Female": "Female"}
    df = df.copy()
    df["sex_label"] = df["patient_sex"].astype(str).map(sex_map).fillna("Unknown")
    return (
        df["sex_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"sex_label": "sex", "count": "reports"})
    )


def outcome_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of sex × outcome."""
    if "patient_sex" not in df.columns:
        return pd.DataFrame()
    outcome_col = "outc_cod" if "outc_cod" in df.columns else "outcome"
    sex_map = {"1": "Male", "2": "Female", "0": "Unknown",
               "Male": "Male", "Female": "Female"}
    df = df.copy()
    df["sex_label"] = df["patient_sex"].astype(str).map(sex_map).fillna("Unknown")
    result = (
        df.groupby(["sex_label", outcome_col])
        .size()
        .reset_index(name="reports")
        .rename(columns={outcome_col: "outcome_code"})
    )
    result["outcome_label"] = result["outcome_code"].map(OUTCOME_MAP).fillna(result["outcome_code"])
    return result


# ---------------------------------------------------------------------------
# Country breakdown
# ---------------------------------------------------------------------------

def top_countries(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Top N reporting countries by report count."""
    if "country" not in df.columns:
        return pd.DataFrame()
    return (
        df["country"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"country": "country", "count": "reports"})
    )


# ---------------------------------------------------------------------------
# Route of administration
# ---------------------------------------------------------------------------

def top_routes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N drug administration routes."""
    if "drug_route" not in df.columns:
        return pd.DataFrame()
    return (
        df["drug_route"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"drug_route": "route", "count": "reports"})
    )