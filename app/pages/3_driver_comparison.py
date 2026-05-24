"""Page 3 — driver head-to-head comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pitwall.utils.io import read_parquet_glob

st.title("Driver Comparison")
st.caption("Head-to-head pace, sectors, and tyre management vs. teammate.")


@st.cache_data(show_spinner=False)
def load(year: int) -> pd.DataFrame:
    return read_parquet_glob("laps", year=year)


year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = load(year)
if laps.empty:
    st.warning("No data yet — run `make ingest` to populate `data/raw/`.")
    st.stop()

race = st.sidebar.selectbox("Race", sorted(laps["CircuitName"].dropna().unique()))
df = laps[laps["CircuitName"] == race]
drivers = sorted(df["Driver"].dropna().unique())
d1 = st.sidebar.selectbox("Driver A", drivers, index=0)
d2 = st.sidebar.selectbox("Driver B", drivers, index=min(1, len(drivers) - 1))

pair = df[df["Driver"].isin([d1, d2])]
fig = px.box(
    pair, x="Driver", y="LapTimeSeconds", color="Compound",
    title=f"{race} {year} · pace distribution by compound", template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Summary")
summ = (
    pair.groupby("Driver")
        .agg(median_pace=("LapTimeSeconds", "median"),
             best_lap=("LapTimeSeconds", "min"),
             clean_laps=("LapTimeSeconds", "count"))
        .round(3)
)
st.dataframe(summ, use_container_width=True)
