"""Official-ish 2024 team colors. Using these is the cheapest credibility win.

Values sourced from public team brand kits, lightly adjusted for screen
contrast.
"""

from __future__ import annotations

TEAM_COLORS: dict[str, str] = {
    "Red Bull Racing":      "#3671C6",
    "Ferrari":              "#E80020",
    "Mercedes":             "#27F4D2",
    "McLaren":              "#FF8000",
    "Aston Martin":         "#229971",
    "Alpine":               "#FF87BC",
    "Williams":             "#64C4FF",
    "RB":                   "#6692FF",
    "Kick Sauber":          "#52E252",
    "Haas":                 "#B6BABD",
    # Pre-2024 fallbacks
    "AlphaTauri":           "#5E8FAA",
    "Alfa Romeo":           "#C92D4B",
}

COMPOUND_COLORS: dict[str, str] = {
    "SOFT":          "#DA291C",
    "MEDIUM":        "#FFD700",
    "HARD":          "#F0F0F0",
    "INTERMEDIATE":  "#43B02A",
    "WET":           "#0067AD",
}


def team_color(team: str, default: str = "#888888") -> str:
    return TEAM_COLORS.get(team, default)


def compound_color(compound: str, default: str = "#888888") -> str:
    return COMPOUND_COLORS.get(compound, default)
