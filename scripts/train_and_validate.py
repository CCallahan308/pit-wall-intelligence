"""End-to-end ML pipeline validation on real ingested data.

Runs:
  1. DegradationModel — both cross-circuit and within-circuit holdouts
  2. Pit-cost statistics per circuit
  3. UndercutClassifier — feature build, calibrated LightGBM, AUC/Brier
  4. RaceSimulator — sample scenario, sanity check
"""

from __future__ import annotations

import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd

from pitwall.config import PROCESSED_DIR
from pitwall.models.degradation_curve import DegradationModel
from pitwall.models.pit_cost import circuit_pit_cost
from pitwall.models.undercut_classifier import FEATURE_COLS, UndercutClassifier
from pitwall.simulation.race_simulator import DriverPlan, RaceConfig, RaceSimulator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DUCKDB_PATH = PROCESSED_DIR / "pitwall.duckdb"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    laps = con.execute("select * from fact_lap").df()
    stints = con.execute("select * from fact_stint").df()
    pits = con.execute("select * from fact_pit_stop").df()
    con.close()
    return laps, stints, pits


_RENAME = {
    "year": "Year",
    "round_num": "Round",
    "driver_code": "Driver",
    "circuit_name": "CircuitName",
    "stint": "Stint",
    "lap_number": "LapNumber",
    "stint_position": "StintPosition",
    "compound": "Compound",
    "lap_time_s": "LapTimeSeconds",
    "lap_time_fuel_corrected_s": "LapTimeFuelCorrected",
    "is_clean_lap": "IsCleanLap",
}


def harmonise_all_laps(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.rename(columns=_RENAME)


def harmonise_clean_laps(all_laps_renamed: pd.DataFrame) -> pd.DataFrame:
    return all_laps_renamed[(all_laps_renamed["IsCleanLap"]) & (all_laps_renamed["StintPosition"] > 1)]


def harmonise_stints(stints: pd.DataFrame) -> pd.DataFrame:
    return stints.rename(columns={
        "year": "Year",
        "round_num": "Round",
        "driver_code": "Driver",
        "stint": "Stint",
        "deg_fit_r2": "DegFitR2",
    })


def degradation_two_way_cv(laps: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Cross-circuit holdout AND within-circuit random holdout."""
    rng = np.random.default_rng(42)

    # 1) Within-circuit: hold out 20% of stints randomly
    stint_keys = laps[["Round", "Driver", "Stint"]].drop_duplicates().reset_index(drop=True)
    holdout_mask = rng.random(len(stint_keys)) < 0.2
    holdout_keys = stint_keys[holdout_mask]
    is_holdout = (
        laps.merge(holdout_keys, on=["Round", "Driver", "Stint"], how="left", indicator=True)
        ._merge.eq("both")
        .to_numpy()
    )
    train = laps[~is_holdout]
    test = laps[is_holdout]
    model = DegradationModel(min_samples_per_curve=20).fit(stints, train)
    within_mae = model.mae(stints, test)
    n_pred = sum(
        (row.Compound, row.CircuitName) in model.curves
        or row.Compound in model.global_curves
        for row in test.itertuples()
    )

    # 2) Cross-circuit: hold out one entire circuit
    holdout_circuit = "Italian Grand Prix"
    train_c = laps[laps["CircuitName"] != holdout_circuit]
    test_c = laps[laps["CircuitName"] == holdout_circuit]
    model_c = DegradationModel(min_samples_per_curve=20).fit(stints, train_c)
    cross_mae = model_c.mae(stints, test_c)

    # Train production model on all data
    prod = DegradationModel(min_samples_per_curve=20).fit(stints, laps)

    return {
        "within_circuit_mae_s": within_mae,
        "cross_circuit_mae_s": cross_mae,
        "n_test_within": len(test),
        "n_predictable_within": n_pred,
        "n_test_cross": len(test_c),
        "prod_model": prod,
        "prod_n_curves": len(prod.curves),
        "prod_n_global": len(prod.global_curves),
    }


def build_undercut_features(
    all_laps: pd.DataFrame, pits: pd.DataFrame, deg_model: DegradationModel
) -> pd.DataFrame:
    """Build feature matrix + labels from real pit stops.

    For each historical pit stop, compute the race state at the pit lap and
    label the example +1 if the driver had a higher track position 5 laps
    later (i.e. moved up the field), 0 if they lost positions or stayed put.

    `all_laps` should be the UNFILTERED fact_lap (renamed cols) so that we
    can look up the in-lap position — clean-lap filtering excludes it.
    """
    rows = []
    for _, p in pits.iterrows():
        year = p["year"]
        rnd = p["round_num"]
        drv = p["driver_code"]
        pit_lap = int(p["pit_lap"])
        race_laps = all_laps[(all_laps["Year"] == year) & (all_laps["Round"] == rnd) & (all_laps["Driver"] == drv)]
        if race_laps.empty:
            continue
        all_laps_race = all_laps[(all_laps["Year"] == year) & (all_laps["Round"] == rnd)]
        total_laps = int(all_laps_race["LapNumber"].max())
        pit_row = race_laps[race_laps["LapNumber"] == pit_lap]
        if pit_row.empty:
            continue
        pos_before = pit_row["position"].iloc[0]
        post = race_laps[race_laps["LapNumber"] == pit_lap + 5]
        pos_after = post["position"].iloc[0] if not post.empty else None
        if pos_before is None or pos_after is None or pd.isna(pos_before) or pd.isna(pos_after):
            continue

        circuit = pit_row["CircuitName"].iloc[0]
        compound = pit_row["Compound"].iloc[0] if pd.notna(pit_row["Compound"].iloc[0]) else "MEDIUM"
        tyre_age = int(pit_row["StintPosition"].iloc[0]) if pd.notna(pit_row["StintPosition"].iloc[0]) else 1

        rows.append({
            "gap_ahead_s": 1.5,  # placeholder — Ergast lap-by-lap gaps would fill this
            "gap_behind_s": 2.0,
            "tyre_age": tyre_age,
            "tyre_age_delta_vs_ahead": 0.0,
            "tyre_age_delta_vs_behind": 0.0,
            "compound_idx": {"SOFT": 0, "MEDIUM": 1, "HARD": 2}.get(compound, 1),
            "current_deg_slope": 0.08,  # could be derived per-stint; placeholder
            "expected_fresh_pace_delta": -float(deg_model.predict_delta(compound, circuit, tyre_age)[0])
                                          if not np.isnan(deg_model.predict_delta(compound, circuit, tyre_age)[0])
                                          else 0.0,
            "laps_remaining": total_laps - int(pit_lap),
            "race_progress_pct": float(pit_lap) / max(total_laps, 1),
            "sc_prob_next_5": 0.05,
            "pit_loss_circuit_s": 23.0,
            "track_evolution_s_per_lap": -0.02,
            "label": int(pos_after < pos_before),  # lower pos number = better
        })
    return pd.DataFrame(rows)


def train_undercut(all_laps: pd.DataFrame, pits: pd.DataFrame, deg_model: DegradationModel) -> dict:
    feats = build_undercut_features(all_laps, pits, deg_model)
    if len(feats) < 30:
        return {"error": f"only {len(feats)} examples - need more races for a real model", "n": len(feats)}

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(feats))
    cut = int(len(idx) * 0.75)
    train_idx, test_idx = idx[:cut], idx[cut:]
    X = feats[FEATURE_COLS]
    y = feats["label"]

    clf = UndercutClassifier()
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    metrics = clf.evaluate(X.iloc[test_idx], y.iloc[test_idx])
    metrics["base_rate"] = float(y.mean())
    metrics["n_total"] = int(len(feats))
    metrics["clf"] = clf
    return metrics


def run_simulator(deg_model: DegradationModel) -> dict:
    config = RaceConfig(circuit="Italian Grand Prix", total_laps=53, sc_rate_per_race=1.0,
                        overtake_difficulty_s=0.3)
    sim = RaceSimulator(deg_model, config, seed=7)
    drivers = [
        DriverPlan(code="A_1STOP", grid=1, base_pace_s=82.5, pit_laps=[22], compounds=["MEDIUM", "HARD"], pit_loss_s=23.0),
        DriverPlan(code="B_2STOP", grid=3, base_pace_s=82.4, pit_laps=[16, 36], compounds=["SOFT", "MEDIUM", "HARD"], pit_loss_s=23.0),
        DriverPlan(code="C_1STOP", grid=2, base_pace_s=82.7, pit_laps=[26], compounds=["MEDIUM", "HARD"], pit_loss_s=23.0),
    ]
    result = sim.run(drivers, n_sim=2000)
    return {
        "A_expected_finish": result.expected_finish("A_1STOP"),
        "B_expected_finish": result.expected_finish("B_2STOP"),
        "C_expected_finish": result.expected_finish("C_1STOP"),
        "A_dist": dict(result.position_distribution("A_1STOP").astype(int)),
        "B_dist": dict(result.position_distribution("B_2STOP").astype(int)),
        "C_dist": dict(result.position_distribution("C_1STOP").astype(int)),
    }


def main() -> None:
    print("=" * 64)
    print("Pit Wall Intelligence - End-to-End ML Pipeline Validation")
    print("=" * 64)

    print("\n[1/4] Loading data from DuckDB...")
    laps_raw, stints_raw, pits_raw = load_data()
    all_laps = harmonise_all_laps(laps_raw)
    laps = harmonise_clean_laps(all_laps)
    stints = harmonise_stints(stints_raw)
    print(f"  fact_lap (all):    {len(all_laps):,} rows")
    print(f"  fact_lap (clean):  {len(laps):,} rows across {laps['CircuitName'].nunique()} circuits")
    print(f"  fact_stint:        {len(stints):,} rows")
    print(f"  fact_pit_stop:     {len(pits_raw):,} rows")

    print("\n[2/4] Fitting tyre degradation model with two-way CV...")
    deg = degradation_two_way_cv(laps, stints)
    print(f"  Within-circuit holdout (random 20% of stints):")
    print(f"    test rows:       {deg['n_test_within']:,}")
    print(f"    MAE:             {deg['within_circuit_mae_s']:.3f} s")
    print(f"  Cross-circuit holdout (Italian GP held out):")
    print(f"    test rows:       {deg['n_test_cross']:,}")
    print(f"    MAE:             {deg['cross_circuit_mae_s']:.3f} s")
    print(f"  Production model: {deg['prod_n_curves']} per-circuit curves, {deg['prod_n_global']} fallback")
    joblib.dump(deg["prod_model"], PROCESSED_DIR / "degradation_model.joblib")
    print(f"  -> saved to {PROCESSED_DIR / 'degradation_model.joblib'}")

    print("\n[3a/4] Pit-cost summary by circuit (with bootstrap CI)...")
    cost_df = pits_raw.rename(columns={"circuit_name": "CircuitName", "pit_loss_s": "PitLossS"})
    cost = circuit_pit_cost(cost_df, n_bootstrap=1000)
    for _, r in cost.iterrows():
        print(f"  {r['CircuitName']:30s}  median={r['MedianPitLossS']:5.2f}s  "
              f"CI95=[{r['CI95LowS']:5.2f}, {r['CI95HighS']:5.2f}]  n={r['NStops']}")

    print("\n[3b/4] Training undercut classifier (calibrated LightGBM)...")
    uc = train_undercut(all_laps, pits_raw, deg["prod_model"])
    if "error" in uc:
        print(f"  SKIP: {uc['error']}")
    else:
        print(f"  examples:        {uc['n_total']} ({int(uc['base_rate'] * uc['n_total'])} positive)")
        print(f"  base rate:       {uc['base_rate']:.3f}")
        print(f"  test set:        n={uc['n']}")
        print(f"  AUC:             {uc['auc']:.3f}")
        print(f"  Brier:           {uc['brier']:.3f}")
        joblib.dump(uc["clf"], PROCESSED_DIR / "undercut_classifier.joblib")
        print(f"  -> saved to {PROCESSED_DIR / 'undercut_classifier.joblib'}")

    print("\n[4/4] Running race simulator (2000 sims, 3-driver Monza scenario)...")
    s = run_simulator(deg["prod_model"])
    print(f"  A (1-stop @22, grid P1):  E[finish]={s['A_expected_finish']:.2f}  dist={s['A_dist']}")
    print(f"  C (1-stop @26, grid P2):  E[finish]={s['C_expected_finish']:.2f}  dist={s['C_dist']}")
    print(f"  B (2-stop, grid P3):       E[finish]={s['B_expected_finish']:.2f}  dist={s['B_dist']}")

    print("\n" + "=" * 64)
    print("[OK] All pipeline stages executed successfully.")
    print("=" * 64)


if __name__ == "__main__":
    main()
