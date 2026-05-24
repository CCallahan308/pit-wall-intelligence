"""Page 5 — championship-wide degradation and pit-cost trends."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pitwall.utils.io import read_parquet_glob

st.title("Season Overview")
st.caption("Championship-wide degradation slopes and pit-cost trends.")

year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = read_parquet_glob("laps", year=year)
if laps.empty:
    st.warning("No data yet — run `make ingest` to populate `data/raw/`.")
    st.stop()

st.subheader("Pace distribution across circuits")
fig = px.box(
    laps.dropna(subset=["LapTimeSeconds"]),
    x="CircuitName", y="LapTimeSeconds", template="plotly_dark",
    title=f"{year} · lap time distribution by circuit",
)
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, use_container_width=True)
