"""Central configuration for paths, caches, and constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
DUCKDB_PATH = PROCESSED_DIR / "pitwall.duckdb"

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Physical constants used in fuel correction
# F1 cars start with ~110kg of fuel; burn rate is ~1.6kg/lap on average.
# Each kg of fuel costs roughly 0.03s/lap of pace.
FUEL_BURN_KG_PER_LAP = 1.6
FUEL_TIME_PENALTY_S_PER_KG = 0.03
STARTING_FUEL_KG = 110.0

# Compounds: Pirelli uses C0-C5 (development) mapped to the SOFT/MEDIUM/HARD
# triplet selected per race. INTER and WET are crossovers.
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# Clean-air threshold — laps with > 2.0s gap ahead are considered "free"
CLEAN_AIR_GAP_S = 2.0

# Stint trimming — first/last lap of a stint are in/out laps and excluded
# from degradation slope estimation.
IN_OUT_LAP_TRIM = 1
