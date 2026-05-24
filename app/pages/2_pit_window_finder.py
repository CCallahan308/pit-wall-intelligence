"""Page 2 - pit window finder (illustrative heuristic until the real model is wired in)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand

apply_theme()
sidebar_brand()
page_header(
    "Pit Window Finder",
    "Undercut success probability across pit laps, given current race state.",
)

st.info(
    "Heads-up: the curve below is an **illustrative heuristic** so you can play with the inputs. "
    "The calibrated LightGBM classifier from `train_and_validate.py` "
    "will replace this surface in the next pass."
)

with st.sidebar:
    st.markdown("### Race state")
    gap_ahead = st.slider("Gap to car ahead (s)", 0.0, 6.0, 1.8, 0.1)
    gap_behind = st.slider("Gap to car behind (s)", 0.0, 6.0, 2.5, 0.1)
    tyre_age = st.slider("Current tyre age (laps)", 0, 40, 18)
    tyre_age_ahead = st.slider("Car-ahead tyre age (laps)", 0, 40, 22)
    laps_remaining = st.slider("Laps remaining", 5, 60, 32)

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
        x=laps_axis,
        y=score,
        mode="lines+markers",
        line={"color": "#E10600", "width": 3},
        marker={"size": 8, "color": "#E10600"},
        name="P(gain position)",
        hovertemplate="Pit in +%{x} laps<br>P(gain) = %{y:.2f}<extra></extra>",
    )
)
fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="rgba(255,255,255,0.45)",
    annotation_text="50/50",
    annotation_position="top right",
    annotation_font_color="rgba(255,255,255,0.6)",
)
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    title={
        "text": "Undercut probability by pit lap",
        "x": 0.02,
        "xanchor": "left",
        "font": {"size": 18, "family": "Inter, sans-serif"},
    },
    xaxis_title="Laps from now to pit",
    yaxis_title="P(net position gained within 5 laps)",
    yaxis_range=[0, 1],
    height=420,
    margin={"l": 60, "r": 30, "t": 60, "b": 50},
    showlegend=False,
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)

best_lap = int(laps_axis[np.argmax(score)])
col1, col2, col3 = st.columns(3)
col1.metric("Recommended pit window", f"+{best_lap} laps")
col2.metric("Max probability", f"{score.max():.2f}")
col3.metric(
    "Tyre-age advantage if stopped now",
    f"{tyre_age_ahead - tyre_age:+d} laps",
    "vs. car ahead",
    delta_color="off",
)
