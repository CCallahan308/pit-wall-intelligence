"""Sensitivity analysis for the two magic numbers downstream metrics depend on.

We sweep:
  1. Fuel penalty (s/kg)   -- currently 0.03 in src/pitwall/config.py
  2. SC filter threshold   -- currently 1.6x in dbt/models/marts/fact_pit_stop.sql

For each value we recompute the affected metrics from raw fact_lap (we don't
re-run dbt because the SC filter sits at the boundary, and we can replicate
its behaviour in pandas faster than a full dbt rebuild).

Outputs:
  - data/processed/sensitivity_fuel.csv
  - data/processed/sensitivity_sc_filter.csv
  - prints a summary so the result is visible in CI logs and MLflow
"""

from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pitwall.config import PROCESSED_DIR
from pitwall.models.degradation_curve import DegradationModel

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROCESSED_DIR / "pitwall.duckdb"


# ============================================================
# Helper: fuel correction at an arbitrary penalty
# ============================================================


def fuel_correct(
    lap_time_s: np.ndarray, lap_number: np.ndarray, total_laps: int, penalty_s_per_kg: float
) -> np.ndarray:
    """Recompute fuel-corrected lap time at a different `penalty_s_per_kg`."""
    starting_fuel = 110.0
    burn = (starting_fuel - 1.0) / max(total_laps - 1, 1)
    mass = np.maximum(starting_fuel - burn * (lap_number - 1), 0.0)
    return lap_time_s - mass * penalty_s_per_kg


# ============================================================
# Fuel penalty sweep
# ============================================================


def fuel_sweep(laps: pd.DataFrame, penalties: list[float]) -> pd.DataFrame:
    """For each penalty, refit the degradation model and report within-circuit MAE."""
    rows = []
    rng = np.random.default_rng(42)
    stint_keys = laps[["Round", "Driver", "Stint"]].drop_duplicates()
    holdout_mask = rng.random(len(stint_keys)) < 0.2
    holdout = stint_keys[holdout_mask]
    is_holdout = (
        laps.merge(holdout, on=["Round", "Driver", "Stint"], how="left", indicator=True)
        ._merge.eq("both")
        .to_numpy()
    )

    for pen in penalties:
        df = laps.copy()
        # Recompute fuel correction at this penalty per (year, round, driver)
        out = []
        for (_, _, _), g in df.groupby(["Year", "Round", "Driver"]):
            total = int(g["LapNumber"].max())
            corrected = fuel_correct(
                g["LapTimeSeconds"].to_numpy(dtype=float),
                g["LapNumber"].to_numpy(dtype=int),
                total,
                pen,
            )
            sub = g.copy()
            sub["LapTimeFuelCorrected"] = corrected
            out.append(sub)
        df = pd.concat(out, ignore_index=True)

        train = df[~is_holdout]
        test = df[is_holdout]
        model = DegradationModel(min_samples_per_curve=20).fit(stints=pd.DataFrame(), laps=train)
        mae = model.mae(stints=pd.DataFrame(), laps=test)
        rows.append(
            {
                "fuel_penalty_s_per_kg": pen,
                "within_circuit_mae_s": round(mae, 3),
                "n_test_laps": len(test),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SC filter sweep -- simulate fact_pit_stop at different thresholds
# ============================================================


def sc_filter_sweep(laps: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Replicate fact_pit_stop's SC filter at different multipliers.

    Detect pit laps from stint transitions: lap N is a pit lap if the driver's
    stint changes between lap N and lap N+1.
    """
    rows = []
    for thresh in thresholds:
        kept_total = 0
        dropped_total = 0
        loss_values = []
        for (_, _, _), g in laps.sort_values(["Year", "Round", "Driver", "LapNumber"]).groupby(
            ["Year", "Round", "Driver"], sort=False
        ):
            g = g.reset_index(drop=True)
            # Pit laps: lap where stint(lap+1) > stint(lap)
            stint_next = g["Stint"].shift(-1)
            pit_mask = (stint_next.notna()) & (stint_next > g["Stint"])
            in_idx = g.index[pit_mask].tolist()
            for i in in_idx:
                if i + 1 >= len(g):
                    continue
                in_lap = g.loc[i, "LapTimeFuelCorrected"]
                out_lap = g.loc[i + 1, "LapTimeFuelCorrected"]
                # baseline = 10th percentile within +/- 5 laps, excluding in/out
                window = g.loc[max(0, i - 5) : min(len(g) - 1, i + 6)]
                clean = window.drop(index=[i, i + 1], errors="ignore")[
                    "LapTimeFuelCorrected"
                ].dropna()
                if len(clean) < 2 or pd.isna(in_lap) or pd.isna(out_lap):
                    continue
                baseline = float(clean.quantile(0.10))
                # Apply the SC filter at the current threshold
                if in_lap >= thresh * baseline or out_lap >= thresh * baseline:
                    dropped_total += 1
                    continue
                loss = (in_lap + out_lap) - 2 * baseline
                # Also enforce the [5, 45] sanity range we keep at every threshold
                if 5 <= loss <= 45:
                    kept_total += 1
                    loss_values.append(loss)
                else:
                    dropped_total += 1

        loss_arr = np.array(loss_values)
        rows.append(
            {
                "sc_filter_threshold": thresh,
                "n_kept_stops": int(kept_total),
                "n_dropped": int(dropped_total),
                "median_pit_loss_s": round(float(np.median(loss_arr)), 2)
                if len(loss_arr)
                else None,
                "mean_pit_loss_s": round(float(np.mean(loss_arr)), 2) if len(loss_arr) else None,
                "p95_pit_loss_s": round(float(np.percentile(loss_arr, 95)), 2)
                if len(loss_arr)
                else None,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("Loading fact_lap...")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    raw = con.execute("""
        select year, round_num, driver_code, lap_number, stint, stint_position,
               compound, lap_time_s, lap_time_fuel_corrected_s,
               is_clean_lap, position, circuit_name
        from fact_lap
        where lap_time_s is not null
    """).df()
    con.close()
    laps = raw.rename(
        columns={
            "year": "Year",
            "round_num": "Round",
            "driver_code": "Driver",
            "lap_number": "LapNumber",
            "stint": "Stint",
            "stint_position": "StintPosition",
            "compound": "Compound",
            "lap_time_s": "LapTimeSeconds",
            "lap_time_fuel_corrected_s": "LapTimeFuelCorrected",
            "is_clean_lap": "IsCleanLap",
            "position": "Position",
            "circuit_name": "CircuitName",
        }
    )
    print(f"  loaded {len(laps):,} rows")

    # Restrict to clean laps + StintPosition > 1 for the degradation refit
    clean = laps[laps["IsCleanLap"] & (laps["StintPosition"] > 1)].dropna(
        subset=["LapTimeFuelCorrected"]
    )
    print(f"  {len(clean):,} clean laps for degradation sweep")

    print("\n[1/2] Fuel penalty sweep...")
    fuel_table = fuel_sweep(clean, penalties=[0.025, 0.030, 0.035])
    print(fuel_table.to_string(index=False))
    fuel_table.to_csv(PROCESSED_DIR / "sensitivity_fuel.csv", index=False)

    print("\n[2/2] SC filter threshold sweep...")
    sc_table = sc_filter_sweep(laps, thresholds=[1.4, 1.6, 1.8])
    print(sc_table.to_string(index=False))
    sc_table.to_csv(PROCESSED_DIR / "sensitivity_sc_filter.csv", index=False)

    print("\nInterpretation:")
    fuel_range = fuel_table["within_circuit_mae_s"]
    print(
        f"  Fuel penalty: MAE varies by "
        f"{(fuel_range.max() - fuel_range.min()):.3f}s across the 0.025-0.035 sweep. "
        f"Model is {'robust' if (fuel_range.max() - fuel_range.min()) < 0.2 else 'sensitive'} to this constant."
    )
    sc_kept = sc_table["n_kept_stops"]
    print(
        f"  SC filter: keeping {sc_kept.iloc[0]:,} stops at 1.4x vs {sc_kept.iloc[-1]:,} at 1.8x "
        f"(a {abs(sc_kept.iloc[-1] - sc_kept.iloc[0]):,}-stop swing). "
        f"Median pit loss changes by "
        f"{abs(sc_table['median_pit_loss_s'].iloc[-1] - sc_table['median_pit_loss_s'].iloc[0]):.2f}s "
        "across the sweep."
    )


if __name__ == "__main__":
    main()
