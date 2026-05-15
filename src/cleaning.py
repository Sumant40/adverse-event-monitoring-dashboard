"""
src/cleaning.py
---------------
FAERS data cleaning pipeline.

Reads faers_raw.csv and produces:
  - data/faers_clean.csv   : full cleaned dataset
  - data/faers_serious.csv : serious-event subset for signal detection
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_PATH   = Path("data/faers_raw.csv")
CLEAN_PATH = Path("data/faers_clean.csv")
SERIOUS_PATH = Path("data/faers_serious.csv")

RENAME_MAP = {
    "safetyreportid": "report_id",
    "receivedate":    "report_date",
    "drugname":       "drug",
    "reactionmeddrapt": "reaction",
    "patientagegroup": "age_group_raw",
    "patientsex":     "sex",
    "primarysourcecountry": "country",
    "manufacturername": "manufacturer",
    "drugadministrationroute": "route",
    "drugindication": "indication",
    "seriousnesscongenitalanomali": "flag_congenital",
    "seriousnessdeath":            "flag_death",
    "seriousnessdisabling":        "flag_disabled",
    "seriousnesshospitalization":  "flag_hosp",
    "seriousnesslifethreatening":  "flag_lt",
    "seriousnessother":            "flag_other",
}

AGE_BAND_MAP = {
    1: "<2",
    2: "2-11",
    3: "12-17",
    4: "18-64",
    5: "65-85",
    6: ">85",
}

OUTCOME_PRIORITY = ["DE", "LT", "HO", "DS", "OT"]

BIOLOGIC_KEYWORDS = [
    "ADALIMUMAB", "ETANERCEPT", "INFLIXIMAB", "CERTOLIZUMAB",
    "GOLIMUMAB", "TOCILIZUMAB", "ABATACEPT", "RITUXIMAB",
    "VEDOLIZUMAB", "USTEKINUMAB", "SECUKINUMAB", "IXEKIZUMAB",
    "RISANKIZUMAB", "GUSELKUMAB", "TILDRAKIZUMAB", "BIMEKIZUMAB",
    "NIVOLUMAB", "PEMBROLIZUMAB", "ATEZOLIZUMAB", "DURVALUMAB",
    "IPILIMUMAB", "TRASTUZUMAB", "PERTUZUMAB", "BEVACIZUMAB",
    "CETUXIMAB", "PANITUMUMAB", "RAMUCIRUMAB", "NECITUMUMAB",
    "DENOSUMAB", "DUPILUMAB", "OMALIZUMAB", "MEPOLIZUMAB",
    "RESLIZUMAB", "BENRALIZUMAB", "TEZEPELUMAB", "CANAKINUMAB",
    "DINUTUXIMAB", "BLINATUMOMAB", "GEMTUZUMAB",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardise_text(series: pd.Series) -> pd.Series:
    """Upper-case, strip whitespace, collapse internal spaces."""
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("NAN", np.nan)
        .replace("", np.nan)
    )


def _parse_date(series: pd.Series) -> pd.Series:
    """Coerce to datetime; accept YYYYMMDD or standard formats."""
    out = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = out.isna()
    out[mask] = pd.to_datetime(series[mask], errors="coerce")
    return out


def _build_outcome_code(row: pd.Series) -> str:
    """Return highest-priority outcome code for a row."""
    mapping = {
        "flag_death":    "DE",
        "flag_lt":       "LT",
        "flag_hosp":     "HO",
        "flag_disabled": "DS",
        "flag_other":    "OT",
    }
    for col, code in mapping.items():
        if str(row.get(col, "")).strip() in {"1", "1.0", "True", "Y"}:
            return code
    return "OT"


def _is_serious(row: pd.Series) -> bool:
    flags = ["flag_death", "flag_lt", "flag_hosp", "flag_disabled"]
    return any(
        str(row.get(f, "")).strip() in {"1", "1.0", "True", "Y"}
        for f in flags
    )


def _tag_biologic(drug_series: pd.Series) -> pd.Series:
    pattern = "|".join(BIOLOGIC_KEYWORDS)
    return drug_series.str.contains(pattern, na=False, regex=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    log.info("Loading raw data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Raw shape: %s", df.shape)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    return df


def standardise_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["drug", "reaction", "country", "manufacturer", "route", "indication"]:
        if col in df.columns:
            df[col] = _standardise_text(df[col])

    df["report_date"] = _parse_date(df.get("report_date", pd.Series(dtype=str)))
    df["year"] = df["report_date"].dt.year

    if "age_group_raw" in df.columns:
        df["age_group"] = (
            pd.to_numeric(df["age_group_raw"], errors="coerce")
            .map(AGE_BAND_MAP)
            .fillna("Unknown")
        )
    return df


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outcome"] = df.apply(_build_outcome_code, axis=1)
    df["is_serious"] = df.apply(_is_serious, axis=1)
    df["is_biologic"] = _tag_biologic(df["drug"])
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate report-drug-reaction triples."""
    before = len(df)
    key_cols = [c for c in ["report_id", "drug", "reaction"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols)
    log.info("  Deduplicated: %d → %d rows", before, len(df))
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("  Saved %d rows → %s", len(df), path)


def run(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw(raw_path)
    df = rename_columns(df)
    df = standardise_fields(df)
    df = build_outcomes(df)
    df = deduplicate(df)

    serious = df[df["is_serious"]].copy()

    save(df, CLEAN_PATH)
    save(serious, SERIOUS_PATH)

    log.info("Clean: %d rows | Serious: %d rows", len(df), len(serious))
    return df, serious


if __name__ == "__main__":
    run()