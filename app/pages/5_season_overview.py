"""Page 5 - championship-wide degradation and pit-cost trends."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand
from pitwall.utils.io import load_fact_lap, query

apply_theme()
sidebar_brand()
page_header(
    "Season Overview",
    "Championship-wide degradation slopes and pit-cost trends across the ingested races.",
)


@st.cache_data(show_spinner=False)
def load(year: int) -> pd.DataFrame:
    return load_fact_lap(year=year)


@st.cache_data(show_spinner=False)
def load_pit_costs(year: int) -> pd.DataFrame:
    return query(
        """
        select circuit_name,
               count(*)                          as n_stops,
               round(median(pit_loss_s), 2)      as median_loss_s,
               round(min(pit_loss_s), 2)         as min_loss_s,
               round(max(pit_loss_s), 2)         as max_loss_s
        from fact_pit_stop
        where year = ?
        group by 1
        order by median_loss_s desc
        """,
        [year],
    )


@st.cache_data(show_spinner=False)
def load_stint_deg(year: int) -> pd.DataFrame:
    return query(
        """
        select circuit_name, compound,
               round(median(deg_slope_s_per_lap), 4) as median_deg_slope,
               count(*)                              as n_stints
        from fact_stint
        where year = ? and deg_slope_s_per_lap is not null
        group by 1, 2
        order by 1, 2
        """,
        [year],
    )


year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021, 2020], index=0)
laps = load(year)
if laps.empty:
    st.warning("No data in the warehouse yet. Run `make ingest` then `make build`.")
    st.stop()

clean = laps[laps["IsCleanLap"] & (laps["StintPosition"] > 1)].dropna(subset=["LapTimeSeconds"])


# ============== Pace distribution ==============

st.subheader("Pace distribution across circuits")
fig = px.box(
    clean,
    x="CircuitName",
    y="LapTimeSeconds",
    template="plotly_dark",
    labels={"CircuitName": "", "LapTimeSeconds": "Lap time (s)"},
    points=False,
    color_discrete_sequence=["#E10600"],
)
fig.update_xaxes(tickangle=0, showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=420,
    margin={"l": 60, "r": 30, "t": 30, "b": 40},
    font={"family": "Inter, sans-serif", "color": "#E8E8E8"},
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)


# ============== Pit cost + degradation ==============

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pit cost per circuit")
    cost = load_pit_costs(year)
    if cost.empty:
        st.info("No pit-stop records yet. Re-run `make build`.")
    else:
        cost_fig = px.bar(
            cost.iloc[::-1],
            x="median_loss_s",
            y="circuit_name",
            orientation="h",
            text="median_loss_s",
            template="plotly_dark",
            labels={"median_loss_s": "Median pit loss (s)", "circuit_name": ""},
            color="median_loss_s",
            color_continuous_scale=[[0, "#27F4D2"], [0.5, "#FFB800"], [1, "#E10600"]],
        )
        cost_fig.update_traces(texttemplate="%{text:.2f}s", textposition="outside")
        cost_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin={"l": 10, "r": 40, "t": 10, "b": 40},
            coloraxis_showscale=False,
            font={"family": "Inter, sans-serif", "color": "#E8E8E8"},
        )
        cost_fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        cost_fig.update_yaxes(showgrid=False)
        st.plotly_chart(cost_fig, use_container_width=True)

with col2:
    st.subheader("Median degradation by compound")
    deg = load_stint_deg(year)
    if deg.empty:
        st.info("No stint records yet.")
    else:
        deg_fig = px.bar(
            deg,
            x="circuit_name",
            y="median_deg_slope",
            color="compound",
            barmode="group",
            template="plotly_dark",
            labels={
                "median_deg_slope": "Median slope (s/lap)",
                "circuit_name": "",
                "compound": "Compound",
            },
            color_discrete_map={
                "SOFT": "#DA291C",
                "MEDIUM": "#FFD700",
                "HARD": "#F0F0F0",
                "INTERMEDIATE": "#43B02A",
                "WET": "#0067AD",
            },
        )
        deg_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin={"l": 50, "r": 20, "t": 10, "b": 60},
            font={"family": "Inter, sans-serif", "color": "#E8E8E8"},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.32,
                "xanchor": "center",
                "x": 0.5,
            },
        )
        deg_fig.update_xaxes(tickangle=30, showgrid=False)
        deg_fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
        st.plotly_chart(deg_fig, use_container_width=True)
