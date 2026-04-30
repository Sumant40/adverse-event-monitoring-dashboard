import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "outputs" / "signals" / "flagged_signals.csv"

if not DATA_PATH.exists():
    st.error(f"Missing file: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.title("Pharmacovigilance Signal Dashboard")

# Filters
drug = st.selectbox("Select Drug", df['drug'].unique())
min_prr = st.slider("Minimum PRR", 1.0, 5.0, 2.0)

filtered = df[(df['drug']==drug) & (df['prr'] >= min_prr)]

st.dataframe(filtered)

st.bar_chart(filtered[['reaction','prr']].set_index('reaction'))