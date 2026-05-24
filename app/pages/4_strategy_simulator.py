"""Page 4 - Monte Carlo strategy simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand

apply_theme()
sidebar_brand()
page_header(
    "Strategy Simulator",
    "Monte Carlo race simulation under alternate pit strategies.",
)

with st.sidebar:
    st.markdown("### Race setup")
    circuit = st.selectbox("Circuit", ["Monaco", "Monza", "Silverstone", "Spa", "Suzuka"])
    total_laps = st.number_input("Total laps", 30, 80, 58)
    n_sim = st.slider("Simulations", 100, 10_000, 2_000, step=100)

st.markdown(
    "Each driver below is evaluated across thousands of random race scenarios "
    "(weather, Safety Car arrivals, pit-stop variance). The chart shows the "
    "distribution of finishing positions per strategy."
)

if "drivers" not in st.session_state:
    st.session_state.drivers = [
        {"code": "VER", "grid": 1, "base_pace_s": 78.5, "strategy": "1-stop M->H @ lap 22"},
        {"code": "HAM", "grid": 3, "base_pace_s": 78.9, "strategy": "2-stop M->M->H @ 18, 40"},
    ]

st.dataframe(pd.DataFrame(st.session_state.drivers), use_container_width=True, hide_index=True)

col1, col2 = st.columns([1, 5])
run = col1.button("Run simulation", type="primary")

if run:
    # Demo distribution - replace with RaceSimulator.run() once wired in.
    rng = np.random.default_rng(0)
    rows = []
    for d in st.session_state.drivers:
        for _ in range(n_sim):
            rows.append(
                {"code": d["code"], "finish": int(np.clip(rng.normal(d["grid"], 2), 1, 20))}
            )
    df = pd.DataFrame(rows)

    fig = px.histogram(
        df,
        x="finish",
        color="code",
        barmode="overlay",
        opacity=0.7,
        template="plotly_dark",
        title=f"{circuit} - finish position distribution ({n_sim} sims)",
        labels={"finish": "Finishing position", "code": "Driver"},
        color_discrete_sequence=["#E10600", "#27F4D2", "#FFB800", "#FF8000"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 16}},
        font={"family": "Inter, sans-serif"},
        bargap=0.05,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Expected finish")
    st.dataframe(
        df.groupby("code")["finish"].agg(["mean", "median", "std"]).round(2).rename_axis("Driver"),
        use_container_width=True,
    )
else:
    st.info("Press **Run simulation** to compute the finishing-position distribution.", icon="▶")
