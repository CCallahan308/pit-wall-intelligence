"""Page 1 — stint-level pace and degradation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall.utils.io import read_parquet_glob
from pitwall.viz.stint_plot import stint_degradation_plot

st.title("Stint Analysis")
st.caption("Fuel-corrected lap pace and degradation slopes, per stint.")


@st.cache_data(show_spinner=False)
def load_laps(year: int) -> pd.DataFrame:
    return read_parquet_glob("laps", year=year)


year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = load_laps(year)
if laps.empty:
    st.warning("No data yet — run `make ingest` to populate `data/raw/`.")
    st.stop()

races = sorted(laps["CircuitName"].dropna().unique())
race = st.sidebar.selectbox("Race", races)
df = laps[laps["CircuitName"] == race]

drivers = sorted(df["Driver"].dropna().unique())
chosen = st.sidebar.multiselect("Drivers", drivers, default=drivers[:4])
df = df[df["Driver"].isin(chosen)]

st.plotly_chart(stint_degradation_plot(df, title=f"{race} {year} · Stint Degradation"), use_container_width=True)

st.subheader("Stint summary")
st.dataframe(
    df.groupby(["Driver", "Stint", "Compound"])
      .agg(laps=("LapNumber", "count"), median_pace=("LapTimeSeconds", "median"))
      .round(3)
      .reset_index(),
    use_container_width=True,
)
