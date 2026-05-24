"""Page 3 - driver head-to-head comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand
from pitwall.utils.io import load_fact_lap
from pitwall.viz.driver_names import driver_label, driver_name

apply_theme()
sidebar_brand()
page_header(
    "Driver Comparison",
    "Head-to-head pace and tyre management vs. another driver in the same race.",
)


@st.cache_data(show_spinner=False)
def load(year: int) -> pd.DataFrame:
    return load_fact_lap(year=year)


year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = load(year)
if laps.empty:
    st.warning("No data in the warehouse yet. Run `make ingest` then `make build`.")
    st.stop()

race = st.sidebar.selectbox("Race", sorted(laps["CircuitName"].dropna().unique()))
df = laps[laps["CircuitName"] == race].copy()

driver_codes = sorted(df["Driver"].dropna().unique())
d1 = st.sidebar.selectbox("Driver A", driver_codes, index=0, format_func=driver_label)
default_b_idx = 1 if len(driver_codes) > 1 else 0
d2 = st.sidebar.selectbox("Driver B", driver_codes, index=default_b_idx, format_func=driver_label)

pair = df[df["Driver"].isin([d1, d2])].copy()
pair["DriverName"] = pair["Driver"].map(driver_name)
pair_clean = pair[pair["IsCleanLap"] & (pair["StintPosition"] > 1)].dropna(
    subset=["LapTimeSeconds"]
)

if pair_clean.empty:
    st.info("No clean racing laps for the selected pair.")
    st.stop()


# ============== KPI strip ==============

summ = (
    pair_clean.groupby("DriverName")
    .agg(
        median_pace=("LapTimeSeconds", "median"),
        best_lap=("LapTimeSeconds", "min"),
        median_fuel_corrected=("LapTimeFuelCorrected", "median"),
        clean_laps=("LapTimeSeconds", "count"),
    )
    .round(3)
)

if len(summ) == 2:
    paces = summ["median_pace"]
    delta = float(paces.iloc[0] - paces.iloc[1])
    faster_name = paces.idxmin()
    slower_name = paces.idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{summ.index[0]} - median pace", f"{summ['median_pace'].iloc[0]:.3f}s")
    c2.metric(f"{summ.index[1]} - median pace", f"{summ['median_pace'].iloc[1]:.3f}s")
    c3.metric("Pace delta", f"{abs(delta):.3f} s/lap", f"{faster_name} faster", delta_color="off")
    st.caption(
        f"{faster_name} was on average {abs(delta):.3f} seconds per lap quicker than {slower_name} across clean racing laps."
    )
    st.markdown("&nbsp;")


# ============== Lap-time distribution ==============

fig = px.box(
    pair_clean,
    x="DriverName",
    y="LapTimeSeconds",
    color="Compound",
    title=f"{race} {year} - clean-lap distribution",
    template="plotly_dark",
    points="all",
    labels={"DriverName": "", "LapTimeSeconds": "Lap time (s)"},
    color_discrete_map={
        "SOFT": "#DA291C",
        "MEDIUM": "#FFD700",
        "HARD": "#F0F0F0",
        "INTERMEDIATE": "#43B02A",
        "WET": "#0067AD",
    },
)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=460,
    margin={"l": 60, "r": 30, "t": 60, "b": 40},
    font={"family": "Inter, system-ui, sans-serif", "color": "#E8E8E8"},
    title={"x": 0.02, "xanchor": "left", "font": {"size": 16}},
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "xanchor": "center", "x": 0.5},
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)


# ============== Detail table ==============

st.subheader("Summary table")
display = summ.rename(
    columns={
        "median_pace": "Median pace (s)",
        "best_lap": "Best lap (s)",
        "median_fuel_corrected": "Median fuel-corrected (s)",
        "clean_laps": "Clean laps",
    }
).rename_axis("Driver")
st.dataframe(display, use_container_width=True)
