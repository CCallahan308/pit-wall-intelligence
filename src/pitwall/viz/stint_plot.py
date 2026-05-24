"""Broadcast-style stint and degradation plots using Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pitwall.viz.team_colors import compound_color

PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.08)"
AXIS_COLOR = "rgba(255,255,255,0.45)"


def _layout_defaults(title: str | None = None, height: int = 500) -> dict:
    return {
        "title": {
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 18, "color": "#FFFFFF", "family": "Inter, system-ui, sans-serif"},
        },
        "template": "plotly_dark",
        "plot_bgcolor": PLOT_BG,
        "paper_bgcolor": PLOT_BG,
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 70, "b": 60},
        "font": {"family": "Inter, system-ui, sans-serif", "color": "#E8E8E8"},
        "hoverlabel": {"bgcolor": "#111", "font": {"family": "JetBrains Mono, monospace"}},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
    }


def _ols_trendline(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return (predicted y, slope-in-seconds-per-lap) or None if too few points."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return slope * x + intercept, float(slope)


def stint_degradation_plot(
    laps: pd.DataFrame,
    title: str = "Stint Degradation",
    facet_by_driver: bool = True,
) -> go.Figure:
    """Faceted view: one subplot per driver, traces colored by compound.

    Each stint gets a scatter of clean laps + an OLS trendline so the
    degradation slope is visible. The legend collapses duplicates so each
    compound appears only once.
    """
    label_col = "DriverName" if "DriverName" in laps.columns else "Driver"
    drivers = list(laps[label_col].dropna().unique())
    n = max(len(drivers), 1)

    if not facet_by_driver or n == 1:
        fig = go.Figure()
        _add_stint_traces(fig, laps, label_col, show_legend_seen=set(), row=None, col=None)
        fig.update_layout(**_layout_defaults(title))
        _style_axes(fig)
        return fig

    # Decide subplot grid: 1 col if <=2 drivers, else 2 cols
    cols = 1 if n <= 2 else 2
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=drivers,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )
    seen_legend: set[str] = set()
    for i, driver in enumerate(drivers):
        r = i // cols + 1
        c = i % cols + 1
        sub = laps[laps[label_col] == driver]
        _add_stint_traces(fig, sub, label_col, show_legend_seen=seen_legend, row=r, col=c)

    fig.update_layout(**_layout_defaults(title, height=max(360, 280 * rows)))
    for ann in fig.layout.annotations:
        ann.font = {"size": 13, "color": "#E8E8E8", "family": "Inter, system-ui, sans-serif"}
    _style_axes(fig)
    fig.update_xaxes(title_text="Tyre age (laps)", row=rows, col=1)
    if cols == 2:
        fig.update_xaxes(title_text="Tyre age (laps)", row=rows, col=2)
    fig.update_yaxes(title_text="Fuel-corrected lap time (s)", row=1, col=1)
    return fig


def _add_stint_traces(
    fig: go.Figure,
    laps: pd.DataFrame,
    label_col: str,
    show_legend_seen: set[str],
    row: int | None,
    col: int | None,
) -> None:
    for (driver, stint), g in laps.groupby([label_col, "Stint"]):
        if len(g) < 1:
            continue
        compound = (
            g["Compound"].iloc[0]
            if "Compound" in g and pd.notna(g["Compound"].iloc[0])
            else "MEDIUM"
        )
        color = compound_color(compound)
        show_in_legend = compound not in show_legend_seen
        show_legend_seen.add(compound)

        # Raw scatter of clean laps in this stint
        kwargs = {
            "x": g["StintPosition"],
            "y": g["LapTimeFuelCorrected"],
            "mode": "markers",
            "marker": {"size": 6, "color": color, "line": {"width": 0}},
            "name": compound.capitalize(),
            "legendgroup": compound,
            "showlegend": show_in_legend,
            "hovertemplate": (
                f"<b>{driver}</b><br>Stint {stint} - {compound}<br>"
                "Tyre age %{x} - %{y:.3f}s<extra></extra>"
            ),
        }
        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**kwargs))

        # OLS trendline
        trend = _ols_trendline(
            g["StintPosition"].to_numpy(dtype=float),
            g["LapTimeFuelCorrected"].to_numpy(dtype=float),
        )
        if trend is None:
            continue
        yhat, slope = trend
        trend_kwargs = {
            "x": g["StintPosition"],
            "y": yhat,
            "mode": "lines",
            "line": {"color": color, "width": 2, "dash": "solid"},
            "name": f"{compound} trend",
            "legendgroup": compound,
            "showlegend": False,
            "hovertemplate": f"<b>{driver}</b><br>Stint {stint} - deg slope: {slope:+.3f}s/lap<extra></extra>",
        }
        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**trend_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trend_kwargs))


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={"color": AXIS_COLOR, "size": 11},
        title_font={"color": AXIS_COLOR, "size": 12},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont={"color": AXIS_COLOR, "size": 11},
        title_font={"color": AXIS_COLOR, "size": 12},
    )


def pit_window_heatmap(pit_window_df: pd.DataFrame) -> go.Figure:
    """Heatmap of strategy outcomes by pit lap and strategy choice."""
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
