"""Page 4 - Monte Carlo strategy simulator backed by the trained DegradationModel."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitwall.models.loaders import load_degradation_model
from pitwall.simulation.race_simulator import DriverPlan, RaceConfig, RaceSimulator
from pitwall.ui import apply_theme, page_header, sidebar_brand
from pitwall.utils.io import query
from pitwall.viz.driver_names import driver_name

apply_theme()
sidebar_brand()
page_header(
    "Strategy Simulator",
    "Monte Carlo race outcomes for alternate pit strategies, using the trained tyre-degradation model.",
)


# ============== Load the trained degradation model ==============


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_degradation_model()


deg = _load_model()
if deg is None:
    st.warning(
        "Degradation model not found. Run `python scripts/train_and_validate.py` first to fit it."
    )
    st.stop()


# ============== Sidebar - race configuration ==============


@st.cache_data(show_spinner=False)
def _circuit_list() -> list[str]:
    df = query("select distinct circuit_name from fact_lap order by 1")
    return df["circuit_name"].tolist() if not df.empty else []


@st.cache_data(show_spinner=False)
def _circuit_pit_loss(circuit: str) -> float:
    df = query(
        "select median(pit_loss_s) as m from fact_pit_stop where circuit_name = ?", [circuit]
    )
    if df.empty or pd.isna(df["m"].iloc[0]):
        return 22.0
    return float(df["m"].iloc[0])


@st.cache_data(show_spinner=False)
def _circuit_typical_laps(circuit: str) -> int:
    df = query("select max(lap_number)::int as n from fact_lap where circuit_name = ?", [circuit])
    if df.empty:
        return 55
    return int(df["n"].iloc[0])


circuits = _circuit_list()
if not circuits:
    st.warning("No data ingested yet. Run `make ingest` then `make build`.")
    st.stop()

with st.sidebar:
    st.markdown("### Race configuration")
    circuit = st.selectbox("Circuit", circuits, index=0)
    total_laps = st.number_input(
        "Total laps",
        min_value=30,
        max_value=80,
        value=_circuit_typical_laps(circuit),
    )
    pit_loss = st.number_input(
        "Pit-lane loss (s)",
        min_value=15.0,
        max_value=35.0,
        value=_circuit_pit_loss(circuit),
        step=0.1,
    )
    sc_rate = st.slider(
        "Mean Safety Cars per race",
        min_value=0.0,
        max_value=2.5,
        value=0.6,
        step=0.1,
    )
    n_sim = st.slider("Number of simulations", 200, 5_000, 1_500, step=100)


# ============== Driver plan editor ==============

st.markdown(
    "Edit each driver's strategy below. **Strategy** is a comma-separated list of pit laps; "
    "**Compounds** is the tyre sequence (one more than the number of pit laps)."
)

if "sim_drivers" not in st.session_state:
    st.session_state.sim_drivers = pd.DataFrame(
        [
            {
                "Driver": "VER",
                "Grid": 1,
                "Base pace (s)": 82.5,
                "Strategy (pit laps)": "22",
                "Compounds": "MEDIUM,HARD",
            },
            {
                "Driver": "HAM",
                "Grid": 3,
                "Base pace (s)": 82.7,
                "Strategy (pit laps)": "18,40",
                "Compounds": "SOFT,MEDIUM,HARD",
            },
            {
                "Driver": "NOR",
                "Grid": 2,
                "Base pace (s)": 82.6,
                "Strategy (pit laps)": "26",
                "Compounds": "MEDIUM,HARD",
            },
        ]
    )

edited = st.data_editor(
    st.session_state.sim_drivers,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Driver": st.column_config.TextColumn(help="3-letter driver code (e.g. VER, HAM)"),
        "Grid": st.column_config.NumberColumn(min_value=1, max_value=20, step=1),
        "Base pace (s)": st.column_config.NumberColumn(min_value=60.0, max_value=130.0, step=0.1),
        "Strategy (pit laps)": st.column_config.TextColumn(help="Comma-separated lap numbers"),
        "Compounds": st.column_config.TextColumn(
            help="Comma-separated: SOFT, MEDIUM, HARD, INTERMEDIATE, WET"
        ),
    },
    hide_index=True,
)
st.session_state.sim_drivers = edited

col1, _ = st.columns([1, 5])
run = col1.button("Run simulation", type="primary")


# ============== Run ==============


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_compounds(s: str) -> list[str]:
    return [x.strip().upper() for x in str(s).split(",") if x.strip()]


if run:
    plans = []
    errors = []
    for _, row in edited.iterrows():
        try:
            pit_laps = _parse_int_list(row["Strategy (pit laps)"])
            compounds = _parse_compounds(row["Compounds"])
            if len(compounds) != len(pit_laps) + 1:
                errors.append(
                    f"{row['Driver']}: compounds ({len(compounds)}) must be exactly one more "
                    f"than pit laps ({len(pit_laps)})."
                )
                continue
            plans.append(
                DriverPlan(
                    code=str(row["Driver"]).strip().upper(),
                    grid=int(row["Grid"]),
                    base_pace_s=float(row["Base pace (s)"]),
                    pit_laps=pit_laps,
                    compounds=compounds,
                    pit_loss_s=pit_loss,
                )
            )
        except Exception as exc:
            errors.append(f"{row['Driver']}: {exc}")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    if len(plans) < 2:
        st.error("Need at least 2 drivers to simulate a race.")
        st.stop()

    config = RaceConfig(
        circuit=circuit,
        total_laps=int(total_laps),
        sc_rate_per_race=float(sc_rate),
        overtake_difficulty_s=0.3,
    )
    sim = RaceSimulator(deg, config, seed=42)
    with st.spinner(f"Running {n_sim:,} simulations..."):
        result = sim.run(plans, n_sim=int(n_sim))

    # ============== Results: position distribution ==============

    long = result.positions.melt(var_name="Driver", value_name="Finish")
    long["DriverName"] = long["Driver"].map(driver_name)
    n_drivers = len(plans)

    fig = px.histogram(
        long,
        x="Finish",
        color="DriverName",
        barmode="group",
        nbins=n_drivers,
        template="plotly_dark",
        title=f"{circuit} - finishing-position distribution across {n_sim:,} simulations",
        labels={"Finish": "Finishing position", "DriverName": "Driver"},
        color_discrete_sequence=["#E10600", "#27F4D2", "#FFB800", "#FF8000", "#64C4FF", "#229971"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 16, "family": "Inter, sans-serif"}},
        font={"family": "Inter, sans-serif"},
        bargap=0.05,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "xanchor": "center", "x": 0.5},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, use_container_width=True)

    # ============== Expected finishing position table ==============

    summary = (
        long.groupby("DriverName")["Finish"]
        .agg(["mean", "median", "std"])
        .round(2)
        .rename_axis("Driver")
        .rename(columns={"mean": "Expected finish", "median": "Median finish", "std": "Std dev"})
    )
    # Add P(win) and P(podium)
    win_prob = long.groupby("DriverName")["Finish"].apply(lambda s: (s == 1).mean()).round(3)
    podium_prob = long.groupby("DriverName")["Finish"].apply(lambda s: (s <= 3).mean()).round(3)
    summary["P(win)"] = win_prob
    summary["P(podium)"] = podium_prob

    st.subheader("Outcome summary")
    st.dataframe(summary, use_container_width=True)

    # ============== Race-time delta plot ==============

    st.subheader("Race-time distribution (vs. fastest sim)")
    times = result.race_times.copy()
    times_long = times.melt(var_name="Driver", value_name="RaceTime")
    times_long["DriverName"] = times_long["Driver"].map(driver_name)
    times_long["DeltaToFastest"] = times_long.groupby(times_long.index // n_drivers)[
        "RaceTime"
    ].transform(lambda s: s - s.min())

    fig2 = go.Figure()
    for drv in plans:
        sub = times_long[times_long["Driver"] == drv.code]
        fig2.add_trace(
            go.Violin(
                y=sub["DeltaToFastest"],
                name=driver_name(drv.code),
                box_visible=True,
                meanline_visible=True,
                line_color="rgba(255,255,255,0.3)",
            )
        )
    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title={
            "text": "Race-time gap to the fastest sim",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 16},
        },
        yaxis_title="Seconds behind fastest sim",
        height=380,
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
        showlegend=False,
        font={"family": "Inter, sans-serif"},
    )
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Press **Run simulation** to compute the finishing-position distribution.")
