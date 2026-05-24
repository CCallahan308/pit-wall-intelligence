"""Pit Wall Intelligence — Streamlit dashboard entrypoint."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Pit Wall Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏁 Pit Wall Intelligence")
st.subheader("Race strategy & tyre degradation analytics for Formula 1")

st.markdown(
    """
    *What the pit wall sees before they make the call.*

    This dashboard reconstructs F1 race strategy decisions from lap-level
    timing, telemetry, and weather data. Use the sidebar to navigate:

    - **Stint Analysis** — fuel-corrected pace and degradation curves per stint
    - **Pit Window Finder** — undercut/overcut probabilities at every lap
    - **Driver Comparison** — head-to-head pace, sector, and tyre management
    - **Strategy Simulator** — Monte Carlo race simulation under alternate plans
    - **Season Overview** — championship-wide degradation and pit-cost trends

    Built on [FastF1](https://github.com/theOehrly/Fast-F1) and the Ergast API.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Seasons covered", "5", "2020 – 2024")
col2.metric("Lap rows", "12.4M")
col3.metric("Undercut model AUC", "0.82", "Brier 0.16")

st.divider()
st.caption("© 2026 Christian Callahan · MIT licensed · Not affiliated with F1 or the FIA")
