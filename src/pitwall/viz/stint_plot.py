"""Broadcast-style stint and degradation plots using Plotly."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from pitwall.viz.team_colors import compound_color


def stint_degradation_plot(laps: pd.DataFrame, title: str = "Stint Degradation") -> go.Figure:
    """One trace per stint, fuel-corrected lap time vs. stint position."""
    fig = go.Figure()
    for (driver, stint), g in laps.groupby(["Driver", "Stint"]):
        compound = g["Compound"].iloc[0] if "Compound" in g else "SOFT"
        fig.add_trace(
            go.Scatter(
                x=g["StintPosition"],
                y=g["LapTimeFuelCorrected"],
                mode="lines+markers",
                name=f"{driver} S{stint} ({compound})",
                line={"color": compound_color(compound), "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{driver}</b><br>Stint {stint} · {compound}<br>"
                    "Lap %{customdata[0]} · Tyre age %{x}<br>%{y:.3f}s<extra></extra>"
                ),
                customdata=g[["LapNumber"]].values,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Tyre age (laps)",
        yaxis_title="Fuel-corrected lap time (s)",
        template="plotly_dark",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def pit_window_heatmap(pit_window_df: pd.DataFrame) -> go.Figure:
    """Heatmap of strategy outcomes by pit lap × strategy choice."""
    pivot = pit_window_df.pivot(index="strategy", columns="pit_lap", values="net_positions")
    return go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "Net positions"},
        )
    ).update_layout(
        title="Pit window outcomes",
        xaxis_title="Lap pitted",
        yaxis_title="Strategy",
        template="plotly_dark",
    )
