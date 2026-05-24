"""Page 1 - stint-level pace and degradation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand
from pitwall.utils.io import load_fact_lap
from pitwall.viz.driver_names import driver_label, driver_name
from pitwall.viz.stint_plot import stint_degradation_plot

apply_theme()
sidebar_brand()
page_header(
    "Stint Analysis",
    "Fuel-corrected lap pace and degradation slopes, faceted by driver.",
)


@st.cache_data(show_spinner=False)
def load(year: int) -> pd.DataFrame:
    return load_fact_lap(year=year)


year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = load(year)
if laps.empty:
    st.warning("No data in the warehouse yet. Run `make ingest` then `make build`.")
    st.stop()

races = sorted(laps["CircuitName"].dropna().unique())
race = st.sidebar.selectbox("Race", races)
df = laps[laps["CircuitName"] == race].copy()

drivers_in_race = sorted(df["Driver"].dropna().unique())
chosen = st.sidebar.multiselect(
    "Drivers",
    drivers_in_race,
    default=drivers_in_race[:2],
    format_func=driver_label,
    help="Pick 2-4 drivers for a clean side-by-side view.",
)
if not chosen:
    st.info("Pick at least one driver from the sidebar.")
    st.stop()

df = df[df["Driver"].isin(chosen)].copy()
df["DriverName"] = df["Driver"].map(driver_name)

df_clean = df[df["IsCleanLap"] & (df["StintPosition"] > 1)].dropna(subset=["LapTimeFuelCorrected"])
if df_clean.empty:
    st.info("No clean racing laps for the current selection.")
    st.stop()

st.plotly_chart(
    stint_degradation_plot(df_clean, title=f"{race} {year}"),
    use_container_width=True,
)

st.markdown("&nbsp;")
st.subheader("Stint summary")
st.caption("One row per (driver, stint). Slope is the OLS degradation rate in seconds per lap.")

summary_rows = []
for (drv_name, stint), g in df_clean.groupby(["DriverName", "Stint"]):
    x = g["StintPosition"].to_numpy(dtype=float)
    y = g["LapTimeFuelCorrected"].to_numpy(dtype=float)
    slope = (
        float(((x - x.mean()) * (y - y.mean())).sum() / max(((x - x.mean()) ** 2).sum(), 1e-9))
        if len(g) >= 3
        else None
    )
    summary_rows.append(
        {
            "Driver": drv_name,
            "Stint": int(stint),
            "Compound": g["Compound"].iloc[0],
            "Laps": len(g),
            "Median pace (s)": round(float(g["LapTimeFuelCorrected"].median()), 3),
            "Best lap (s)": round(float(g["LapTimeFuelCorrected"].min()), 3),
            "Deg slope (s/lap)": round(slope, 4) if slope is not None else None,
        }
    )
summary = pd.DataFrame(summary_rows).sort_values(["Driver", "Stint"]).reset_index(drop=True)
st.dataframe(summary, use_container_width=True, hide_index=True)
