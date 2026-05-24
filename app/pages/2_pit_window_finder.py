"""Page 2 - pit window finder backed by the calibrated LightGBM classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pitwall.models.loaders import load_undercut_classifier
from pitwall.models.undercut_classifier import FEATURE_COLS
from pitwall.ui import apply_theme, page_header, sidebar_brand

apply_theme()
sidebar_brand()
page_header(
    "Pit Window Finder",
    "Calibrated undercut-success probability across pit laps, given current race state.",
)

COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_undercut_classifier()


clf = _load_model()
if clf is None or clf.model is None:
    st.warning(
        "Undercut model not found in `data/processed/`. Run `python scripts/train_and_validate.py` "
        "first - that script fits the calibrated LightGBM classifier and saves it as a joblib."
    )
    st.stop()


# ============== Sidebar inputs ==============

with st.sidebar:
    st.markdown("### Current race state")
    current_tyre_age = st.slider("Your tyre age (laps)", 1, 40, 18)
    current_compound = st.selectbox("Your current compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    ahead_tyre_age = st.slider("Car-ahead tyre age (laps)", 1, 40, 22)
    behind_tyre_age = st.slider("Car-behind tyre age (laps)", 1, 40, 14)
    gap_ahead = st.slider("Gap to car ahead (s)", 0.0, 6.0, 1.8, 0.1)
    gap_behind = st.slider("Gap to car behind (s)", 0.0, 6.0, 2.5, 0.1)
    laps_remaining = st.slider("Laps remaining in the race", 5, 60, 32)
    current_deg_slope = st.slider(
        "Current degradation slope (s/lap)",
        0.00,
        0.30,
        0.08,
        0.01,
        help="How much pace you're losing per lap on the current tyre. Fitted on the last 5 stint laps in production.",
    )
    pit_loss = st.slider("Pit-lane loss this circuit (s)", 16.0, 30.0, 22.5, 0.1)
    sc_prob = st.slider("Safety Car probability next 5 laps", 0.0, 0.5, 0.05, 0.01)


# ============== Feature sweep across pit-in-N-laps ==============

n_laps_axis = np.arange(0, 16)
total_laps_for_race = current_tyre_age + laps_remaining  # rough scale

rows = []
for n in n_laps_axis:
    rows.append(
        {
            "gap_ahead_s": gap_ahead,
            "gap_behind_s": gap_behind,
            "tyre_age": current_tyre_age + n,
            "tyre_age_delta_vs_ahead": (current_tyre_age + n) - ahead_tyre_age,
            "tyre_age_delta_vs_behind": (current_tyre_age + n) - behind_tyre_age,
            "compound_idx": COMPOUND_IDX[current_compound],
            "current_deg_slope": current_deg_slope,
            # Pace advantage from going to fresh tyres - rises with tyre age
            "expected_fresh_pace_delta": current_deg_slope * (current_tyre_age + n),
            "laps_remaining": max(laps_remaining - n, 1),
            "race_progress_pct": (current_tyre_age + n) / max(total_laps_for_race, 1),
            "sc_prob_next_5": sc_prob,
            "pit_loss_circuit_s": pit_loss,
            "track_evolution_s_per_lap": -0.02,
        }
    )

X = pd.DataFrame(rows)[FEATURE_COLS]
probs = clf.predict_proba(X)


# ============== Chart ==============

best_idx = int(np.argmax(probs))
best_n = int(n_laps_axis[best_idx])
best_p = float(probs[best_idx])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=n_laps_axis,
        y=probs,
        mode="lines+markers",
        line={"color": "#E10600", "width": 3},
        marker={"size": 8, "color": "#E10600"},
        name="P(gain position)",
        hovertemplate="Pit in +%{x} laps<br>P(gain) = %{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[best_n],
        y=[best_p],
        mode="markers",
        marker={"size": 14, "color": "#FFB800", "line": {"color": "#FFFFFF", "width": 2}},
        name=f"Best window: +{best_n} laps",
        hovertemplate=f"<b>Best window</b><br>+{best_n} laps - P={best_p:.2f}<extra></extra>",
        showlegend=False,
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
        "text": "Undercut probability by pit-in lap (calibrated LightGBM)",
        "x": 0.02,
        "xanchor": "left",
        "font": {"size": 18, "family": "Inter, sans-serif"},
    },
    xaxis_title="Laps from now to pit",
    yaxis_title="P(net position gained within 5 laps)",
    yaxis_range=[0, 1],
    height=440,
    margin={"l": 60, "r": 30, "t": 60, "b": 50},
    showlegend=False,
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, dtick=1)
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)


# ============== Recommendation + breakdown ==============

c1, c2, c3 = st.columns(3)
c1.metric("Recommended pit window", f"+{best_n} laps")
c2.metric("Predicted probability", f"{best_p:.2f}")
c3.metric(
    "Tyre-age delta vs. car ahead at pit",
    f"{(current_tyre_age + best_n) - ahead_tyre_age:+d} laps",
    delta_color="off",
)

st.markdown("&nbsp;")
with st.expander("See the feature vector at the recommended pit window"):
    st.dataframe(
        X.iloc[[best_idx]].T.rename(columns={best_idx: "Value"}),
        use_container_width=True,
    )

st.caption(
    "Calibrated LightGBM trained on historical pit stops. Probabilities are "
    "isotonic-calibrated, so 0.7 should mean ~70% historical hit rate. "
    "Current sample is small (~125 stops); accuracy will sharpen as more "
    "seasons land in the warehouse."
)
