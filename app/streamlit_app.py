"""
app/streamlit_app.py
--------------------
Pharmacovigilance Signal Detection Dashboard — v2

Tabs:
  1. Signal Explorer  — filter by drug, biologic flag, PRR/ROR threshold
  2. Time-Series      — quarterly case trend for a selected signal
  3. Outcome Model    — odds ratios from logistic regression
  4. Summary Stats    — headline metrics and top-10 tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PV Signal Detection | FAERS",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIGNALS_PATH    = Path("outputs/signals/flagged_signals.csv")
TIMESERIES_PATH = Path("outputs/signals/timeseries_signals.csv")
MODEL_PATH      = Path("outputs/signals/outcome_model_coefs.csv")
CLEAN_PATH      = Path("data/faers_clean.csv")
SERIOUS_PATH    = Path("data/faers_serious.csv")

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
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_signals():
    if not SIGNALS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SIGNALS_PATH)
    df["is_biologic"] = df["drug"].str.contains(
        "|".join(BIOLOGIC_KEYWORDS), na=False
    )
    return df


@st.cache_data
def load_timeseries():
    if not TIMESERIES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TIMESERIES_PATH)


@st.cache_data
def load_model():
    if not MODEL_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_PATH)


@st.cache_data
def load_clean():
    if not CLEAN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEAN_PATH, low_memory=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

signals = load_signals()
ts_df   = load_timeseries()
model   = load_model()
clean   = load_clean()

st.title("🔬 Pharmacovigilance Signal Detection Dashboard")
st.caption(
    "FDA FAERS adverse event data · PRR & ROR disproportionality analysis · 2015–2025"
)

if signals.empty:
    st.error(
        "⚠️  No signal data found. Run `src/signal_detection.py` first "
        "to generate `outputs/signals/flagged_signals.csv`."
    )
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Signal Explorer", "📈 Time-Series", "🧮 Outcome Model", "📊 Summary Stats"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Explorer
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Signal Explorer")
    st.markdown(
        "Filter flagged drug-reaction pairs by drug, disproportionality metric, "
        "and priority. Each row meets the minimum signal criteria: "
        "**cases ≥ 3, PRR > 2, lower 95 % CI > 1**."
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])

    with col_f1:
        all_drugs = sorted(signals["drug"].dropna().unique())
        drug_choice = st.multiselect(
            "Drug(s)", all_drugs,
            placeholder="Select one or more drugs …"
        )

    with col_f2:
        min_prr = st.slider("Min PRR", 1.0, 50.0, 2.0, step=0.5)

    with col_f3:
        min_ror = st.slider("Min ROR", 1.0, 50.0, 1.0, step=0.5)

    with col_f4:
        priority_filter = st.multiselect(
            "Priority", ["High", "Medium"], default=["High", "Medium"]
        )

    bio_only = st.checkbox("Biologics only", value=False)

    # Apply filters
    view = signals.copy()
    if drug_choice:
        view = view[view["drug"].isin(drug_choice)]
    view = view[view["prr"] >= min_prr]
    if "ror" in view.columns:
        view = view[view["ror"].fillna(0) >= min_ror]
    view = view[view["priority"].isin(priority_filter)]
    if bio_only:
        view = view[view["is_biologic"]]

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signals shown", f"{len(view):,}")
    m2.metric("High priority", f"{(view['priority']=='High').sum():,}")
    m3.metric("Unique drugs", f"{view['drug'].nunique():,}")
    m4.metric("Unique reactions", f"{view['reaction'].nunique():,}")

    # Display table
    display_cols = [
        "drug", "reaction", "cases", "prr", "prr_lower_95ci", "prr_upper_95ci",
        "ror", "ror_lower_95ci", "ror_upper_95ci", "priority", "is_biologic",
    ]
    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols]
        .sort_values("prr", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    # PRR bar chart for selected drug
    if drug_choice and len(drug_choice) == 1:
        top_rxns = view.nlargest(15, "prr")
        if not top_rxns.empty:
            fig = px.bar(
                top_rxns,
                x="prr", y="reaction",
                orientation="h",
                color="priority",
                color_discrete_map={"High": "#e63946", "Medium": "#f4a261"},
                error_x=top_rxns["prr_upper_95ci"] - top_rxns["prr"],
                title=f"Top 15 Reactions by PRR — {drug_choice[0]}",
                labels={"prr": "PRR", "reaction": "Reaction"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # ROR vs PRR scatter (all filtered signals, up to 2000 for perf)
    if len(view) > 1 and "ror" in view.columns:
        scatter_data = view.dropna(subset=["prr", "ror"]).head(2000)
        fig2 = px.scatter(
            scatter_data,
            x="prr", y="ror",
            color="priority",
            hover_data=["drug", "reaction", "cases"],
            color_discrete_map={"High": "#e63946", "Medium": "#f4a261"},
            title="PRR vs ROR (filtered signals)",
            labels={"prr": "PRR", "ror": "ROR"},
            opacity=0.6,
            log_x=True, log_y=True,
        )
        fig2.add_vline(x=2, line_dash="dash", line_color="gray")
        fig2.add_hline(y=2, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Time-Series
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Quarterly Reporting Trend")
    st.markdown(
        "Select a high-priority signal to see how adverse event reporting "
        "evolved over the drug's post-market lifecycle."
    )

    if ts_df.empty:
        st.info(
            "No time-series data found. Re-run `signal_detection.py` with a "
            "`report_date` column present in the serious dataset."
        )
    else:
        ts_drugs = sorted(ts_df["drug"].unique())
        ts_drug = st.selectbox("Drug", ts_drugs)
        ts_rxns = sorted(
            ts_df[ts_df["drug"] == ts_drug]["reaction"].unique()
        )
        ts_rxn = st.selectbox("Reaction", ts_rxns)

        subset = ts_df[
            (ts_df["drug"] == ts_drug) &
            (ts_df["reaction"] == ts_rxn)
        ].sort_values("quarter")

        if subset.empty:
            st.warning("No quarterly data for this selection.")
        else:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=subset["quarter"],
                y=subset["cases"],
                mode="lines+markers",
                name="Cases",
                line=dict(color="#2a9d8f", width=2),
                marker=dict(size=6),
            ))
            # Rolling 4-quarter average
            subset = subset.copy()
            subset["rolling_avg"] = subset["cases"].rolling(4, min_periods=1).mean()
            fig_ts.add_trace(go.Scatter(
                x=subset["quarter"],
                y=subset["rolling_avg"],
                mode="lines",
                name="4-Quarter Rolling Avg",
                line=dict(color="#e9c46a", width=2, dash="dot"),
            ))
            fig_ts.update_layout(
                title=f"Quarterly Reports: {ts_drug} — {ts_rxn}",
                xaxis_title="Quarter",
                yaxis_title="Reports",
                hovermode="x unified",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total cases", int(subset["cases"].sum()))
            col_b.metric("Peak quarter", subset.loc[subset["cases"].idxmax(), "quarter"])
            col_c.metric("Peak cases", int(subset["cases"].max()))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Outcome Model
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Logistic Regression: Predictors of Serious Outcomes")
    st.markdown(
        "A logistic regression model trained to predict **serious vs non-serious** "
        "adverse event outcomes. Coefficients are expressed as odds ratios (OR > 1 "
        "increases the odds of a serious outcome)."
    )

    if model.empty:
        st.info(
            "No model output found. Run `signal_detection.py` with scikit-learn "
            "installed to generate `outputs/signals/outcome_model_coefs.csv`."
        )
    else:
        top_n_model = st.slider("Show top N features by odds ratio", 10, 50, 20)

        top_features = model.nlargest(top_n_model, "odds_ratio")

        fig_coef = px.bar(
            top_features,
            x="odds_ratio",
            y="feature",
            orientation="h",
            color="odds_ratio",
            color_continuous_scale="RdYlGn_r",
            title=f"Top {top_n_model} predictors of serious adverse event outcome",
            labels={"odds_ratio": "Odds Ratio", "feature": "Feature"},
        )
        fig_coef.add_vline(x=1, line_dash="dash", line_color="gray")
        fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

        with st.expander("Full coefficient table"):
            st.dataframe(
                model.sort_values("odds_ratio", ascending=False),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Summary Stats
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Dataset & Signal Summary")

    # Headline metrics
    if not clean.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cleaned reports", f"{len(clean):,}")
        if "serious" in clean.columns:
          serious_total = pd.to_numeric(clean["serious"], errors="coerce").fillna(0).sum()
          c2.metric("Serious reports", f"{int(serious_total):,}")
        else:
         c2.metric("Serious reports", "—")
        c3.metric("Unique drugs", f"{clean['drugname'].nunique():,}" if "drugname" in clean.columns else "—")
        c4.metric("Unique reactions", f"{clean['reaction'].nunique():,}" if "reaction" in clean.columns else "—")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total flagged signals", f"{len(signals):,}")
    s2.metric("High-priority signals", f"{(signals['priority']=='High').sum():,}")
    s3.metric("Biologic signals", f"{signals['is_biologic'].sum():,}" if "is_biologic" in signals.columns else "—")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Drugs with Most Flagged Signals")
        drug_counts = (
            signals.groupby("drug")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="signals")
        )
        fig_drugs = px.bar(
            drug_counts, x="signals", y="drug", orientation="h",
            color="signals", color_continuous_scale="Tealgrn",
            labels={"signals": "Flagged Signals", "drug": "Drug"},
        )
        fig_drugs.update_layout(yaxis={"categoryorder": "total ascending"},
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_drugs, use_container_width=True)

    with col_right:
        st.markdown("#### Most Reported Reactions (Flagged Signals)")
        rxn_counts = (
            signals.groupby("reaction")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="signals")
        )
        fig_rxns = px.bar(
            rxn_counts, x="signals", y="reaction", orientation="h",
            color="signals", color_continuous_scale="Oranges",
            labels={"signals": "Flagged Signals", "reaction": "Reaction"},
        )
        fig_rxns.update_layout(yaxis={"categoryorder": "total ascending"},
                                coloraxis_showscale=False)
        st.plotly_chart(fig_rxns, use_container_width=True)

    # Biologic vs small-molecule comparison
    if "is_biologic" in signals.columns:
        st.markdown("#### Biologic vs Small-Molecule Signal Distribution")
        bio_summary = (
            signals.groupby(["is_biologic", "priority"])
            .size()
            .reset_index(name="count")
        )
        bio_summary["type"] = bio_summary["is_biologic"].map(
            {True: "Biologic", False: "Small Molecule"}
        )
        fig_bio = px.bar(
            bio_summary,
            x="type", y="count", color="priority",
            color_discrete_map={"High": "#e63946", "Medium": "#f4a261"},
            title="Signal Count by Drug Class and Priority",
            labels={"count": "Signals", "type": "Drug Class"},
            barmode="group",
        )
        st.plotly_chart(fig_bio, use_container_width=True)

    # Outcome distribution (from clean data)
    if not clean.empty and "outc_cod" in clean.columns:
        st.markdown("#### Outcome Distribution (Full Dataset)")
        outcome_counts = clean["outc_cod"].value_counts().reset_index()
        outcome_counts.columns = ["outc_cod", "count"]
        outcome_map = {
            "DE": "Death", "HO": "Hospitalised",
            "LT": "Life-threatening", "DS": "Disabled", "OT": "Other"
        }
        outcome_counts["label"] = outcome_counts["outc_cod"].map(outcome_map).fillna(outcome_counts["outc_cod"])
        fig_out = px.pie(
            outcome_counts, values="count", names="label",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Adverse Event Outcome Distribution",
        )
        st.plotly_chart(fig_out, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚠️  FAERS is a spontaneous reporting system. PRR and ROR detect "
        "disproportionate reporting — not causality. All signals require "
        "clinical, regulatory, and epidemiological review before action."
    )