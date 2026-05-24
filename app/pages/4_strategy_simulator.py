"""Page 4 — Monte Carlo strategy simulator."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Strategy Simulator")
st.caption("Monte Carlo race simulation under alternate pit strategies.")

with st.sidebar:
    st.header("Race setup")
    circuit = st.selectbox("Circuit", ["Monaco", "Monza", "Silverstone", "Spa", "Suzuka"])
    total_laps = st.number_input("Total laps", 30, 80, 58)
    n_sim = st.slider("Simulations", 100, 10_000, 2_000, step=100)

st.info(
    "Add drivers and their strategies below, then click **Simulate**. "
    "Each strategy is evaluated across thousands of random race scenarios "
    "(weather windows, Safety Car arrivals, pit-stop variance)."
)

if "drivers" not in st.session_state:
    st.session_state.drivers = [
        {"code": "VER", "grid": 1, "base_pace_s": 78.5, "strategy": "1-stop M->H @ lap 22"},
        {"code": "HAM", "grid": 3, "base_pace_s": 78.9, "strategy": "2-stop M->M->H @ 18,40"},
    ]

st.dataframe(pd.DataFrame(st.session_state.drivers), use_container_width=True)

if st.button("Simulate (demo data)"):
    # Demo distribution — wire to RaceSimulator.run() in production
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for d in st.session_state.drivers:
        for _ in range(n_sim):
            rows.append({"code": d["code"], "finish": int(np.clip(rng.normal(d["grid"], 2), 1, 20))})
    df = pd.DataFrame(rows)
    fig = px.histogram(df, x="finish", color="code", barmode="overlay",
                       template="plotly_dark", title=f"{circuit} · finish position distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Expected finish")
    st.dataframe(df.groupby("code")["finish"].agg(["mean", "median", "std"]).round(2))
