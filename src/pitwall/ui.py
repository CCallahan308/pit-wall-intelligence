"""Shared UI helpers: theming, page chrome, sidebar branding."""

from __future__ import annotations

import streamlit as st

# F1 broadcast-feel palette
ACCENT_RED = "#E10600"  # official F1 red
ACCENT_AMBER = "#FFB800"
BG_DEEP = "#0B0C10"
PANEL = "#15171C"
INK = "#F4F4F4"
INK_DIM = "rgba(244,244,244,0.62)"
RULE = "rgba(255,255,255,0.08)"

CUSTOM_CSS = f"""
<style>
  :root {{
    --f1-red: {ACCENT_RED};
    --f1-amber: {ACCENT_AMBER};
  }}

  /* Page background */
  .stApp {{
    background: radial-gradient(120% 80% at 0% 0%, #131319 0%, {BG_DEEP} 70%);
  }}

  /* Headings */
  h1, h2, h3 {{
    font-family: "Inter", system-ui, sans-serif;
    letter-spacing: -0.01em;
  }}
  h1 {{
    font-weight: 800 !important;
    color: {INK} !important;
  }}
  h2, h3 {{
    color: {INK} !important;
  }}

  /* The accent bar that sits under the page title */
  .pw-title-rule {{
    height: 3px;
    width: 56px;
    background: var(--f1-red);
    border-radius: 2px;
    margin: -8px 0 18px 0;
  }}

  /* KPI tile */
  .pw-kpi {{
    background: {PANEL};
    border: 1px solid {RULE};
    border-radius: 10px;
    padding: 14px 18px;
    min-height: 92px;
  }}
  .pw-kpi-label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {INK_DIM};
    margin-bottom: 6px;
  }}
  .pw-kpi-value {{
    font-family: "JetBrains Mono", "Fira Mono", monospace;
    font-size: 28px;
    font-weight: 700;
    color: {INK};
    line-height: 1.1;
  }}
  .pw-kpi-sub {{
    font-size: 12px;
    color: {INK_DIM};
    margin-top: 4px;
  }}

  /* Compound chip */
  .pw-chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
  }}
  .pw-chip-soft   {{ background: rgba(218, 41, 28, 0.18); color: #FF7A6E; border: 1px solid rgba(218,41,28,0.5); }}
  .pw-chip-medium {{ background: rgba(255, 215, 0, 0.14); color: #FFE16B; border: 1px solid rgba(255,215,0,0.45); }}
  .pw-chip-hard   {{ background: rgba(240, 240, 240, 0.10); color: #E8E8E8; border: 1px solid rgba(255,255,255,0.30); }}
  .pw-chip-inter  {{ background: rgba(67, 176, 42, 0.14); color: #7BD862; border: 1px solid rgba(67,176,42,0.45); }}
  .pw-chip-wet    {{ background: rgba(0, 103, 173, 0.18); color: #6CC0FF; border: 1px solid rgba(0,103,173,0.5); }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: #0E0F13;
    border-right: 1px solid {RULE};
  }}
  section[data-testid="stSidebar"] .pw-sidebar-brand {{
    padding: 6px 0 14px 0;
    border-bottom: 1px solid {RULE};
    margin-bottom: 14px;
  }}
  .pw-sidebar-title {{
    font-family: "Inter", sans-serif;
    font-weight: 800;
    font-size: 18px;
    color: {INK};
    letter-spacing: -0.01em;
  }}
  .pw-sidebar-tag {{
    font-size: 11px;
    color: {INK_DIM};
    margin-top: 2px;
  }}
  .pw-sidebar-accent {{
    display: inline-block;
    height: 8px;
    width: 8px;
    background: var(--f1-red);
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
  }}

  /* Metric component override */
  [data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {RULE};
    border-radius: 10px;
    padding: 12px 16px;
  }}
  [data-testid="stMetricLabel"] {{
    color: {INK_DIM};
  }}

  /* Dataframe */
  div[data-testid="stDataFrame"] {{
    border: 1px solid {RULE};
    border-radius: 8px;
  }}
</style>
"""


def apply_theme() -> None:
    """Inject the global CSS. Call once at the top of every page/script."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Render the branded sidebar header."""
    st.sidebar.markdown(
        '<div class="pw-sidebar-brand">'
        '<div class="pw-sidebar-title"><span class="pw-sidebar-accent"></span>Pit Wall Intelligence</div>'
        '<div class="pw-sidebar-tag">Race strategy analytics</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str | None = None) -> None:
    """Standard page title block with the red accent rule."""
    st.markdown(f"# {title}")
    st.markdown('<div class="pw-title-rule"></div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def kpi(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="pw-kpi-sub">{sub}</div>' if sub else ""
    return (
        '<div class="pw-kpi">'
        f'<div class="pw-kpi-label">{label}</div>'
        f'<div class="pw-kpi-value">{value}</div>'
        f"{sub_html}"
        "</div>"
    )


def compound_chip(compound: str) -> str:
    key = {
        "SOFT": "soft",
        "MEDIUM": "medium",
        "HARD": "hard",
        "INTERMEDIATE": "inter",
        "WET": "wet",
    }.get(compound.upper(), "hard")
    return f'<span class="pw-chip pw-chip-{key}">{compound}</span>'
