"""Page 2 — pit window finder.

Probability-of-gain curve across pit laps, given current race state.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.title("Pit Window Finder")
st.caption("Undercut success probability across pit laps, given current race state.")

st.sidebar.header("Race state")
gap_ahead = st.sidebar.slider("Gap to car ahead (s)", 0.0, 6.0, 1.8, 0.1)
gap_behind = st.sidebar.slider("Gap to car behind (s)", 0.0, 6.0, 2.5, 0.1)
tyre_age = st.sidebar.slider("Current tyre age (laps)", 0, 40, 18)
tyre_age_ahead = st.sidebar.slider("Car-ahead tyre age (laps)", 0, 40, 22)
laps_remaining = st.sidebar.slider("Laps remaining", 5, 60, 32)

# Toy illustrative curve — replace with model.predict_proba in production
laps_axis = np.arange(0, 15)
score = (
    0.55
    + 0.04 * (tyre_age_ahead - tyre_age)
    - 0.03 * gap_ahead
    + 0.02 * gap_behind
    + 0.005 * laps_remaining
    - 0.015 * laps_axis
)
score = np.clip(score, 0.05, 0.95)

fig = go.Figure(
    data=go.Scatter(
        x=laps_axis, y=score, mode="lines+markers",
        line={"color": "#FF8000", "width": 3}, name="P(gain position)"
    )
)
fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="50/50")
fig.update_layout(
    template="plotly_dark",
    title="Undercut probability by pit lap (laps from now)",
    xaxis_title="Laps from now to pit",
    yaxis_title="P(net position gained within 5 laps)",
    yaxis_range=[0, 1],
)
st.plotly_chart(fig, use_container_width=True)

best_lap = int(laps_axis[np.argmax(score)])
st.metric("Recommended pit window", f"+{best_lap} laps", f"P = {score.max():.2f}")
