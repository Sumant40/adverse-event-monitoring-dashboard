"""
src/signal_detection.py
-----------------------
Disproportionality analysis for FAERS serious-event data.

Computes:
  - Proportional Reporting Ratio (PRR) with 95 % CI
  - Reporting Odds Ratio  (ROR) with 95 % CI
  - Time-series of quarterly case counts per flagged drug-reaction pair
  - Logistic regression model predicting serious vs non-serious outcome

Outputs:
  - outputs/signals/flagged_signals.csv
  - outputs/signals/timeseries_signals.csv
  - outputs/signals/outcome_model_coefs.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SERIOUS_PATH   = Path("data/faers_serious.csv")
CLEAN_PATH     = Path("data/faers_clean.csv")
SIGNALS_PATH   = Path("outputs/signals/flagged_signals.csv")
TIMESERIES_PATH = Path("outputs/signals/timeseries_signals.csv")
MODEL_PATH     = Path("outputs/signals/outcome_model_coefs.csv")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_CASES = 3
PRR_THRESHOLD = 2.0
CI_LOWER_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Disproportionality core
# ---------------------------------------------------------------------------

def _contingency(drug: str, reaction: str, df: pd.DataFrame):
    """
    Return the 2×2 contingency table counts (a, b, c, d).

      drug \\ reaction  | reaction Y | not reaction Y
      ----------------  | ---------- | --------------
      drug X            |     a      |       b
      not drug X        |     c      |       d
    """
    mask_d = df["drug"] == drug
    mask_r = df["reaction"] == reaction
    a = (mask_d & mask_r).sum()
    b = (mask_d & ~mask_r).sum()
    c = (~mask_d & mask_r).sum()
    d = (~mask_d & ~mask_r).sum()
    return a, b, c, d


def _prr(a, b, c, d):
    """PRR = (a/(a+b)) / (c/(c+d)), log-scale 95 % CI."""
    if a == 0 or c == 0 or (a + b) == 0 or (c + d) == 0:
        return np.nan, np.nan, np.nan
    prr = (a / (a + b)) / (c / (c + d))
    # Variance of log(PRR) via delta method
    se_log = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    z = 1.96
    log_prr = np.log(prr)
    lower = np.exp(log_prr - z * se_log)
    upper = np.exp(log_prr + z * se_log)
    return round(prr, 4), round(lower, 4), round(upper, 4)


def _ror(a, b, c, d):
    """ROR = (a/b) / (c/d), log-scale 95 % CI."""
    if b == 0 or c == 0 or d == 0 or a == 0:
        return np.nan, np.nan, np.nan
    ror = (a * d) / (b * c)
    se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.96
    log_ror = np.log(ror)
    lower = np.exp(log_ror - z * se_log)
    upper = np.exp(log_ror + z * se_log)
    return round(ror, 4), round(lower, 4), round(upper, 4)


def _priority(prr, cases):
    if prr > 3 and cases >= 10:
        return "High"
    return "Medium"


# ---------------------------------------------------------------------------
# Main signal scan
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all drug-reaction pairs with >= MIN_CASES reports
    and return a DataFrame of flagged signals.
    """
    log.info("Counting drug-reaction pairs …")
    pair_counts = (
        df.groupby(["drug", "reaction"])
        .size()
        .reset_index(name="cases")
    )
    candidates = pair_counts[pair_counts["cases"] >= MIN_CASES].copy()
    log.info("  Candidate pairs (cases >= %d): %d", MIN_CASES, len(candidates))

    records = []
    total = len(candidates)
    for i, row in candidates.iterrows():
        if i % 5000 == 0:
            log.info("  Processing pair %d / %d …", i, total)

        drug, reaction, cases = row["drug"], row["reaction"], row["cases"]
        a, b, c, d = _contingency(drug, reaction, df)

        prr_val, prr_lo, prr_hi = _prr(a, b, c, d)
        ror_val, ror_lo, ror_hi = _ror(a, b, c, d)

        if (
            pd.notna(prr_val)
            and prr_val > PRR_THRESHOLD
            and pd.notna(prr_lo)
            and prr_lo > CI_LOWER_THRESHOLD
        ):
            records.append({
                "drug":     drug,
                "reaction": reaction,
                "cases":    int(cases),
                "a": a, "b": b, "c": c, "d": d,
                "prr":      prr_val,
                "prr_lower_95ci": prr_lo,
                "prr_upper_95ci": prr_hi,
                "ror":      ror_val,
                "ror_lower_95ci": ror_lo,
                "ror_upper_95ci": ror_hi,
                "priority": _priority(prr_val, cases),
            })

    signals = pd.DataFrame(records).sort_values("prr", ascending=False)
    log.info("Flagged signals: %d (High: %d | Medium: %d)",
             len(signals),
             (signals["priority"] == "High").sum(),
             (signals["priority"] == "Medium").sum())
    return signals


# ---------------------------------------------------------------------------
# Time-series component
# ---------------------------------------------------------------------------

def compute_timeseries(df: pd.DataFrame, signals: pd.DataFrame,
                       top_n: int = 50) -> pd.DataFrame:
    """
    For the top_n high-priority signals, compute quarterly case counts
    to show how reporting evolved over the drug's post-market lifecycle.
    """
    if "report_date" not in df.columns or df["report_date"].isna().all():
        log.warning("report_date missing — skipping time-series.")
        return pd.DataFrame()

    df = df.copy()
    df["quarter"] = df["report_date"].dt.to_period("Q").astype(str)

    top_signals = (
        signals[signals["priority"] == "High"]
        .head(top_n)[["drug", "reaction"]]
    )

    records = []
    for _, sig in top_signals.iterrows():
        subset = df[
            (df["drug"] == sig["drug"]) &
            (df["reaction"] == sig["reaction"])
        ]
        ts = (
            subset.groupby("quarter")
            .size()
            .reset_index(name="cases")
        )
        ts["drug"]     = sig["drug"]
        ts["reaction"] = sig["reaction"]
        records.append(ts)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["drug", "reaction", "quarter"])
    return out


# ---------------------------------------------------------------------------
# Outcome prediction model (logistic regression)
# ---------------------------------------------------------------------------

def build_outcome_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression: predict serious (1) vs non-serious (0) outcome.

    Features:
      - is_biologic (bool)
      - sex (encoded)
      - age_group (encoded)
      - top drug dummies (top 20)
      - top reaction dummies (top 20)

    Returns a DataFrame of coefficients and odds ratios.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
    except ImportError:
        log.warning("scikit-learn not installed — skipping outcome model.")
        return pd.DataFrame()

    log.info("Building outcome prediction model …")
    data = df.copy()

    data["target"] = data["is_serious"].astype(int)

    # Encode sex
    data["sex_enc"] = LabelEncoder().fit_transform(
        data["sex"].fillna("UNKNOWN").astype(str)
    )

    # Encode age group
    data["age_enc"] = LabelEncoder().fit_transform(
        data.get("age_group", pd.Series("Unknown", index=data.index))
        .fillna("Unknown").astype(str)
    )

    # is_biologic
    data["bio_enc"] = data["is_biologic"].astype(int)

    # Top 20 drug dummies
    top_drugs = data["drug"].value_counts().head(20).index
    for d in top_drugs:
        safe = re.sub(r"\W", "_", d)
        data[f"drug_{safe}"] = (data["drug"] == d).astype(int)

    # Top 20 reaction dummies
    top_rxns = data["reaction"].value_counts().head(20).index
    for r in top_rxns:
        safe = re.sub(r"\W", "_", r)
        data[f"rxn_{safe}"] = (data["reaction"] == r).astype(int)

    feat_cols = (
        ["sex_enc", "age_enc", "bio_enc"] +
        [c for c in data.columns if c.startswith("drug_") or c.startswith("rxn_")]
    )

    X = data[feat_cols].fillna(0)
    y = data["target"]

    model = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    model.fit(X, y)

    coef_df = pd.DataFrame({
        "feature":      feat_cols,
        "coefficient":  model.coef_[0],
        "odds_ratio":   np.exp(model.coef_[0]),
    }).sort_values("odds_ratio", ascending=False)

    log.info("  Model trained. Top predictor: %s", coef_df.iloc[0]["feature"])
    return coef_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    import re  # needed inside build_outcome_model

    serious = pd.read_csv(SERIOUS_PATH, low_memory=False)
    serious["report_date"] = pd.to_datetime(serious.get("report_date"), errors="coerce")

    signals = compute_signals(serious)
    ts      = compute_timeseries(serious, signals)

    for p in [SIGNALS_PATH, TIMESERIES_PATH, MODEL_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)

    signals.to_csv(SIGNALS_PATH, index=False)
    log.info("Signals saved → %s", SIGNALS_PATH)

    if not ts.empty:
        ts.to_csv(TIMESERIES_PATH, index=False)
        log.info("Time-series saved → %s", TIMESERIES_PATH)

    # Full dataset for outcome model
    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    coefs = build_outcome_model(clean)
    if not coefs.empty:
        coefs.to_csv(MODEL_PATH, index=False)
        log.info("Outcome model coefficients saved → %s", MODEL_PATH)

    return signals, ts, coefs


if __name__ == "__main__":
    import re
    run()