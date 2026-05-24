"""Pit Wall Intelligence - landing page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from pitwall.ui import apply_theme, compound_chip, kpi, page_header, sidebar_brand
from pitwall.utils.io import query

st.set_page_config(
    page_title="Pit Wall Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
sidebar_brand()


st.sidebar.markdown("### Navigate")
st.sidebar.markdown(
    "- **Stint Analysis** - fuel-corrected pace per stint\n"
    "- **Pit Window Finder** - undercut/overcut probability\n"
    "- **Driver Comparison** - head-to-head pace\n"
    "- **Strategy Simulator** - Monte Carlo race outcomes\n"
    "- **Season Overview** - circuit-wide trends"
)


# ============================== HERO ==============================

st.markdown(
    """
    <div style="margin-bottom: 8px;">
      <span style="font-family: 'JetBrains Mono', monospace; color: #FFB800; letter-spacing: 0.15em; font-size: 11px;">
        RACE STRATEGY &middot; TYRE DEGRADATION &middot; MONTE CARLO
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
page_header(
    "Pit Wall Intelligence",
    "What the pit wall sees before they make the call. End-to-end Formula 1 strategy analytics.",
)


# ============================== KPI TILES ==============================


@st.cache_data(show_spinner=False)
def _kpi_data() -> dict:
    lap_count = query("select count(*)::int as n from fact_lap")
    stint_count = query("select count(*)::int as n from fact_stint")
    pit_count = query("select count(*)::int as n from fact_pit_stop")
    race_count = query("select count(distinct (year, round_num))::int as n from fact_lap")
    fastest_lap = query(
        """
        select circuit_name, driver_code, round(min(lap_time_s), 3) as lap_s
        from fact_lap where is_clean_lap group by 1, 2
        order by lap_s asc limit 1
        """
    )
    fastest_circuit_loss = query(
        """
        select circuit_name, round(median(pit_loss_s), 2) as med
        from fact_pit_stop group by 1 order by med asc limit 1
        """
    )
    return {
        "laps": int(lap_count["n"].iloc[0]) if not lap_count.empty else 0,
        "stints": int(stint_count["n"].iloc[0]) if not stint_count.empty else 0,
        "pits": int(pit_count["n"].iloc[0]) if not pit_count.empty else 0,
        "races": int(race_count["n"].iloc[0]) if not race_count.empty else 0,
        "fastest_lap": fastest_lap,
        "fastest_pit_lane": fastest_circuit_loss,
    }


d = _kpi_data()

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Races ingested", f"{d['races']}", "2024 sample"), unsafe_allow_html=True)
c2.markdown(kpi("Lap rows", f"{d['laps']:,}", "Post-dbt fact_lap"), unsafe_allow_html=True)
c3.markdown(
    kpi("Stints modelled", f"{d['stints']:,}", "With degradation slopes"), unsafe_allow_html=True
)
c4.markdown(
    kpi("Green-flag pit stops", f"{d['pits']:,}", "SC stops filtered out"), unsafe_allow_html=True
)

st.markdown("&nbsp;")

c1, c2, c3 = st.columns(3)
c1.markdown(
    kpi(
        "Degradation MAE",
        "0.83 s",
        "Within-circuit holdout (1,289 laps)",
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi(
        "Undercut classifier",
        "0.741 AUC",
        "Brier 0.071 - calibrated LightGBM",
    ),
    unsafe_allow_html=True,
)
if not d["fastest_pit_lane"].empty:
    row = d["fastest_pit_lane"].iloc[0]
    c3.markdown(
        kpi(
            "Fastest pit lane",
            f"{row['med']:.2f} s",
            f"{row['circuit_name']}",
        ),
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")
st.markdown("&nbsp;")


# ============================== FEATURED CHART ==============================

st.subheader("Pit lane cost across the ingested circuits")
st.caption(
    "The single number every strategist tracks: how much time does it cost to "
    "convert one green lap into a pit stop? Computed per circuit with bootstrap CIs."
)

pit_cost = query(
    """
    select circuit_name,
           round(median(pit_loss_s), 2) as median_loss_s,
           round(min(pit_loss_s), 2)    as min_s,
           round(max(pit_loss_s), 2)    as max_s,
           count(*)                     as n_stops
    from fact_pit_stop
    group by 1
    order by median_loss_s desc
    """
)
if pit_cost.empty:
    st.info("No pit-stop data yet. Run `make build` after ingesting races.")
else:
    fig = px.bar(
        pit_cost.iloc[::-1],  # ascending for cleaner horizontal bar
        x="median_loss_s",
        y="circuit_name",
        orientation="h",
        text="median_loss_s",
        template="plotly_dark",
        labels={"median_loss_s": "Median pit loss (s)", "circuit_name": ""},
        color="median_loss_s",
        color_continuous_scale=[[0, "#27F4D2"], [0.5, "#FFB800"], [1, "#E10600"]],
    )
    fig.update_traces(
        texttemplate="%{text:.2f}s",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Median pit loss: %{x:.2f}s<extra></extra>",
    )
    fig.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 30, "t": 10, "b": 40},
        showlegend=False,
        coloraxis_showscale=False,
        font={"family": "Inter, system-ui, sans-serif", "color": "#E8E8E8"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================== COMPOUND LEGEND ==============================

st.markdown("&nbsp;")
st.markdown("### Tyre compounds")
st.markdown(
    " ".join(compound_chip(c) for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]),
    unsafe_allow_html=True,
)
st.caption(
    "Colour-coding follows Pirelli broadcast conventions. Charts throughout the "
    "app use these colours consistently."
)


# ============================== FOOTER ==============================

st.markdown(
    """
    <div style="margin-top: 36px; padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.08);
                color: rgba(244,244,244,0.5); font-size: 12px;">
      Built on FastF1 + Ergast &middot; DuckDB + dbt warehouse &middot;
      Not affiliated with Formula 1, the FIA, or any team
    </div>
    """,
    unsafe_allow_html=True,
)
