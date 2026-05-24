"""Driver code -> full name mapping.

FastF1 uses 3-letter codes (VER, HAM, ...). The dashboard shows full names so
fans newer to F1 can read the data without memorising the abbreviations.
Covers the 2024 grid plus all 2020-2023 drivers we may ingest later.
"""

from __future__ import annotations

DRIVER_NAMES: dict[str, str] = {
    # 2024 grid
    "VER": "Max Verstappen",
    "PER": "Sergio Perez",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",
    "ALB": "Alexander Albon",
    "SAR": "Logan Sargeant",
    "COL": "Franco Colapinto",
    "TSU": "Yuki Tsunoda",
    "RIC": "Daniel Ricciardo",
    "LAW": "Liam Lawson",
    "BOT": "Valtteri Bottas",
    "ZHO": "Zhou Guanyu",
    "HUL": "Nico Hulkenberg",
    "MAG": "Kevin Magnussen",
    "BEA": "Oliver Bearman",
    "DOO": "Jack Doohan",
    # 2020-2023 drivers who may show up in earlier seasons
    "VET": "Sebastian Vettel",
    "RAI": "Kimi Raikkonen",
    "GIO": "Antonio Giovinazzi",
    "KVY": "Daniil Kvyat",
    "GRO": "Romain Grosjean",
    "FIT": "Pietro Fittipaldi",
    "AIT": "Jack Aitken",
    "MAZ": "Nikita Mazepin",
    "MSC": "Mick Schumacher",
    "KUB": "Robert Kubica",
    "DEV": "Nyck de Vries",
}


def driver_name(code: str) -> str:
    """Full name, with the 3-letter code as a fallback for unknown drivers."""
    return DRIVER_NAMES.get(code, code)


def driver_label(code: str) -> str:
    """`Max Verstappen (VER)` - full name + code in parentheses."""
    name = DRIVER_NAMES.get(code)
    return f"{name} ({code})" if name else code
