"""Page 6 - Race-by-race strategy reconstruction.

For a specific race, show each driver's actual stint plan as a horizontal
Gantt-style timeline coloured by tyre compound. The view a strategist would
sketch in the post-race debrief: who took which compounds and for how long.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitwall.ui import apply_theme, page_header, sidebar_brand
from pitwall.utils.io import query
from pitwall.viz.driver_names import driver_name
from pitwall.viz.team_colors import compound_color

apply_theme()
sidebar_brand()
page_header(
    "Race-by-race",
    "What actually happened: stint plans, pit losses, and finishing-position deltas.",
)


# ============== Race picker ==============


@st.cache_data(show_spinner=False)
def _race_options() -> pd.DataFrame:
    return query(
        """
        select year, round_num, circuit_name,
               count(distinct driver_code) as n_drivers,
               max(lap_number)::int        as total_laps
        from fact_lap
        group by 1, 2, 3
        order by year desc, round_num asc
        """
    )


races = _race_options()
if races.empty:
    st.warning("No data ingested yet. Run `make ingest` then `make build`.")
    st.stop()

races["label"] = races["year"].astype(str) + " - " + races["circuit_name"]
race_label = st.sidebar.selectbox("Race", races["label"].tolist())
row = races[races["label"] == race_label].iloc[0]
year = int(row["year"])
rnd = int(row["round_num"])
circuit = row["circuit_name"]
total_laps = int(row["total_laps"])


# ============== Pull race data ==============


@st.cache_data(show_spinner=False)
def _race_laps(year: int, rnd: int) -> pd.DataFrame:
    df = query(
        """
        select driver_code, team_name, lap_number, stint, stint_position,
               compound, lap_time_s, lap_time_fuel_corrected_s, position,
               is_clean_lap
        from fact_lap
        where year = ? and round_num = ?
        order by driver_code, lap_number
        """,
        [year, rnd],
    )
    if df.empty:
        return df
    df["DriverName"] = df["driver_code"].map(driver_name)
    return df


@st.cache_data(show_spinner=False)
def _race_pits(year: int, rnd: int) -> pd.DataFrame:
    df = query(
        """
        select driver_code, pit_lap, round(pit_loss_s, 2) as pit_loss_s
        from fact_pit_stop
        where year = ? and round_num = ?
        order by driver_code, pit_lap
        """,
        [year, rnd],
    )
    if df.empty:
        return df
    df["DriverName"] = df["driver_code"].map(driver_name)
    return df


laps = _race_laps(year, rnd)
pits = _race_pits(year, rnd)
if laps.empty:
    st.warning("No lap data for this race.")
    st.stop()


# ============== Header strip ==============

n_drivers = laps["driver_code"].nunique()
n_pits = len(pits)
median_pit_loss = float(pits["pit_loss_s"].median()) if not pits.empty else float("nan")
fastest_lap_s = (
    float(laps.loc[laps["is_clean_lap"], "lap_time_s"].min())
    if laps["is_clean_lap"].any()
    else float("nan")
)
fastest_lap_drv = (
    laps.loc[laps["is_clean_lap"]].sort_values("lap_time_s").iloc[0]["DriverName"]
    if laps["is_clean_lap"].any()
    else "n/a"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Race", circuit)
c2.metric("Total laps", f"{total_laps}")
c3.metric(
    "Median pit loss",
    f"{median_pit_loss:.2f} s" if not pd.isna(median_pit_loss) else "n/a",
    f"{n_pits} stops",
    delta_color="off",
)
c4.metric(
    "Fastest lap",
    f"{fastest_lap_s:.3f} s" if not pd.isna(fastest_lap_s) else "n/a",
    fastest_lap_drv,
    delta_color="off",
)

st.markdown("&nbsp;")


# ============== Strategy timeline ==============

st.subheader("Strategy timeline")
st.caption(
    "Each bar shows a continuous stint on a tyre compound. Bar length = stint length. "
    "Drivers are ordered by finishing position."
)

# Build one row per (driver, stint) covering [first_lap, last_lap]
stint_rows = []
for (drv, stint), g in laps.groupby(["driver_code", "stint"]):
    compound = g["compound"].iloc[0]
    if pd.isna(compound):
        continue
    first_lap = int(g["lap_number"].min())
    last_lap = int(g["lap_number"].max())
    stint_rows.append(
        {
            "Driver": driver_name(drv),
            "Team": g["team_name"].iloc[0],
            "Stint": int(stint),
            "Compound": compound,
            "Start": first_lap,
            "End": last_lap + 1,
            "Laps": last_lap - first_lap + 1,
        }
    )
stints_df = pd.DataFrame(stint_rows)

# Determine finishing position per driver (last available position value)
finish_order = (
    laps.dropna(subset=["position"])
    .sort_values(["driver_code", "lap_number"])
    .groupby("driver_code")
    .last()
    .reset_index()[["driver_code", "position"]]
    .sort_values("position")
)
finish_order["DriverName"] = finish_order["driver_code"].map(driver_name)
driver_order = finish_order["DriverName"].tolist()

# Plotly Gantt-style with one bar per stint
fig = go.Figure()
seen_compounds: set[str] = set()
for _, s in stints_df.iterrows():
    show_legend = s["Compound"] not in seen_compounds
    seen_compounds.add(s["Compound"])
    fig.add_trace(
        go.Bar(
            x=[s["Laps"]],
            y=[s["Driver"]],
            base=[s["Start"] - 1],
            orientation="h",
            marker={
                "color": compound_color(s["Compound"]),
                "line": {"color": "rgba(0,0,0,0.6)", "width": 1},
            },
            name=s["Compound"].capitalize(),
            legendgroup=s["Compound"],
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{s['Driver']}</b><br>Stint {s['Stint']} - {s['Compound']}<br>"
                f"Laps {s['Start']}-{s['End'] - 1} ({s['Laps']} laps)<extra></extra>"
            ),
        )
    )

# Pit-loss annotations
for _, p in pits.iterrows():
    fig.add_annotation(
        x=p["pit_lap"],
        y=driver_name(p["driver_code"]),
        text=f"{p['pit_loss_s']:.1f}s",
        showarrow=False,
        font={"size": 9, "color": "rgba(255,255,255,0.85)"},
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="rgba(255,255,255,0.2)",
        borderwidth=1,
        borderpad=2,
        xanchor="center",
        yanchor="middle",
    )

fig.update_layout(
    barmode="stack",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=max(420, 26 * n_drivers),
    margin={"l": 20, "r": 20, "t": 30, "b": 40},
    title={"text": f"{circuit} {year}", "x": 0.02, "xanchor": "left", "font": {"size": 16}},
    font={"family": "Inter, sans-serif"},
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.16, "xanchor": "center", "x": 0.5},
    xaxis={
        "title": "Lap",
        "range": [0, total_laps + 0.5],
        "showgrid": True,
        "gridcolor": "rgba(255,255,255,0.08)",
        "zeroline": False,
    },
    yaxis={
        "categoryorder": "array",
        "categoryarray": driver_order[::-1],  # P1 at the top
        "showgrid": False,
    },
)
st.plotly_chart(fig, use_container_width=True)


# ============== Pit-loss detail + finishing order ==============

c1, c2 = st.columns(2)

with c1:
    st.subheader("Pit-loss detail")
    if pits.empty:
        st.info("No green-flag pit stops recorded for this race.")
    else:
        pit_table = (
            pits.rename(
                columns={
                    "DriverName": "Driver",
                    "pit_lap": "Pit lap",
                    "pit_loss_s": "Pit loss (s)",
                }
            )[["Driver", "Pit lap", "Pit loss (s)"]]
            .sort_values(["Driver", "Pit lap"])
            .reset_index(drop=True)
        )
        st.dataframe(pit_table, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Finishing order")
    if finish_order.empty:
        st.info("Position data not available for this race.")
    else:
        st.dataframe(
            finish_order[["position", "DriverName"]]
            .rename(columns={"position": "Pos", "DriverName": "Driver"})
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ============== Position-vs-lap chart ==============

st.subheader("Track position over the race")
pos = laps.dropna(subset=["position"]).copy()
pos["position"] = pos["position"].astype(int)

fig_pos = px.line(
    pos,
    x="lap_number",
    y="position",
    color="DriverName",
    template="plotly_dark",
    labels={"lap_number": "Lap", "position": "Position", "DriverName": "Driver"},
    color_discrete_sequence=px.colors.qualitative.Set3,
)
fig_pos.update_yaxes(autorange="reversed", dtick=1)  # P1 at top
fig_pos.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=460,
    margin={"l": 60, "r": 30, "t": 30, "b": 50},
    font={"family": "Inter, sans-serif"},
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
)
fig_pos.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
fig_pos.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
st.plotly_chart(fig_pos, use_container_width=True)
