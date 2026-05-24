"""Three-race retrospective: run the trained models on famous strategy calls.

For each race we:
  1. Load actual driver strategies (grid, pit laps, compounds) from fact_lap
  2. Score every historical pit stop with the calibrated undercut classifier
  3. Run the Monte Carlo simulator with the actual strategies
  4. Compare predicted vs actual finishing positions
  5. Write a JSON dump + a markdown writeup

Output: docs/writeups/retrospective_<race>.md and data/processed/retrospective.json

This is the file that turns "model exists" into "model is useful."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pitwall.config import PROCESSED_DIR
from pitwall.features.undercut import build_features
from pitwall.models.undercut_classifier import FEATURE_COLS
from pitwall.simulation.race_simulator import DriverPlan, RaceConfig, RaceSimulator
from pitwall.utils.io import query
from pitwall.viz.driver_names import driver_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRITEUPS_DIR = PROJECT_ROOT / "docs" / "writeups"
WRITEUPS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RaceCase:
    year: int
    round_num: int
    name: str
    story: str  # one-line narrative


RACES: list[RaceCase] = [
    RaceCase(
        year=2024,
        round_num=8,
        name="Monaco Grand Prix",
        story="Leclerc broke his Monaco curse from pole. After a lap-1 red flag everyone "
        "switched to hards and ran 77 laps stationary -- a one-stop endurance test.",
    ),
    RaceCase(
        year=2024,
        round_num=13,
        name="Hungarian Grand Prix",
        story="McLaren had a 1-2 in their hands. Norris led after a fast Piastri stop, "
        "team orders eventually flipped them so Piastri took the win.",
    ),
    RaceCase(
        year=2024,
        round_num=16,
        name="Italian Grand Prix",
        story="Leclerc's 1-stop Monza win. McLaren and Mercedes ran 2-stops expecting "
        "tyre degradation that Ferrari engineered around with a soft-pace stint.",
    ),
]


def get_race_state(year: int, rnd: int) -> dict:
    """Extract everything we need: total laps, pit info, finishing order."""
    total_laps = int(
        query(
            "select max(lap_number)::int as n from fact_lap where year = ? and round_num = ?",
            [year, rnd],
        )["n"].iloc[0]
    )

    # Finishing position = last lap with a position value
    finish_df = query(
        """
        with last_lap as (
            select driver_code, max(lap_number) as last_lap
            from fact_lap
            where year = ? and round_num = ? and position is not null
            group by 1
        )
        select fl.driver_code, fl.position, fl.team_name
        from fact_lap fl
        join last_lap ll
          on fl.driver_code = ll.driver_code and fl.lap_number = ll.last_lap
        where fl.year = ? and fl.round_num = ?
        order by fl.position
        """,
        [year, rnd, year, rnd],
    )

    # Grid position = position on earliest available lap per driver
    # (Monaco 2024 had a lap-1 red flag, so the data starts at lap 2-3 for many drivers.)
    grid_df = query(
        """
        with first_lap as (
            select driver_code, min(lap_number) as first_lap
            from fact_lap
            where year = ? and round_num = ? and position is not null
            group by 1
        )
        select fl.driver_code, fl.position as grid
        from fact_lap fl
        join first_lap fl1
          on fl.driver_code = fl1.driver_code and fl.lap_number = fl1.first_lap
        where fl.year = ? and fl.round_num = ?
        """,
        [year, rnd, year, rnd],
    )

    # Base pace: median fuel-corrected clean-lap pace per driver (best estimate)
    pace_df = query(
        """
        select driver_code, median(lap_time_fuel_corrected_s) as base_pace_s
        from fact_lap
        where year = ? and round_num = ? and is_clean_lap and stint_position > 1
        group by 1
        """,
        [year, rnd],
    )

    # Pit laps & compounds per driver (from fact_lap stint info)
    stints_df = query(
        """
        select driver_code, stint,
               min(lap_number) as first_lap,
               max(lap_number) as last_lap,
               max(compound)   as compound
        from fact_lap
        where year = ? and round_num = ?
        group by 1, 2
        order by 1, 2
        """,
        [year, rnd],
    )

    # Median pit loss for this circuit (for the simulator)
    pit_loss = query(
        """
        select circuit_name, median(pit_loss_s) as median_pit_loss_s
        from fact_pit_stop
        where year = ? and round_num = ?
        group by 1
        """,
        [year, rnd],
    )
    median_pit_loss = float(pit_loss["median_pit_loss_s"].iloc[0]) if not pit_loss.empty else 22.0

    return {
        "total_laps": total_laps,
        "finish": finish_df,
        "grid": grid_df,
        "pace": pace_df,
        "stints": stints_df,
        "pit_loss": median_pit_loss,
    }


def build_driver_plans(state: dict) -> list[DriverPlan]:
    plans = []
    skipped = []
    for drv, drv_stints in state["stints"].groupby("driver_code"):
        drv_stints = drv_stints.sort_values("stint").reset_index(drop=True)
        n_stints = len(drv_stints)
        if n_stints == 0:
            skipped.append((drv, "no stints"))
            continue

        # Pit laps = last lap of each stint except the final one
        pit_laps = drv_stints["last_lap"].iloc[:-1].astype(int).tolist()
        compounds = drv_stints["compound"].fillna("MEDIUM").astype(str).tolist()
        # Hard sanity: compounds == pit_laps + 1
        if len(compounds) != len(pit_laps) + 1:
            skipped.append((drv, f"compounds {len(compounds)} != pits+1 {len(pit_laps) + 1}"))
            continue

        grid_match = state["grid"][state["grid"]["driver_code"] == drv]
        if grid_match.empty or pd.isna(grid_match["grid"].iloc[0]):
            skipped.append((drv, "no grid info"))
            continue
        grid = int(grid_match["grid"].iloc[0])

        pace_match = state["pace"][state["pace"]["driver_code"] == drv]
        if pace_match.empty or pd.isna(pace_match["base_pace_s"].iloc[0]):
            skipped.append((drv, "no clean pace"))
            continue
        base_pace_s = float(pace_match["base_pace_s"].iloc[0])

        plans.append(
            DriverPlan(
                code=drv,
                grid=grid,
                base_pace_s=base_pace_s,
                pit_laps=pit_laps,
                compounds=[c.upper() for c in compounds],
                pit_loss_s=state["pit_loss"],
            )
        )

    if skipped:
        print(f"  skipped {len(skipped)} drivers: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return plans


def analyse_race(race: RaceCase, deg_model, clf, all_laps, pits) -> dict:
    state = get_race_state(race.year, race.round_num)
    plans = build_driver_plans(state)
    if len(plans) < 5:
        return {"race": race.name, "error": "insufficient driver data"}

    # Run simulator with actual strategies
    config = RaceConfig(
        circuit=race.name.replace(" Grand Prix", " Grand Prix"),
        total_laps=state["total_laps"],
        sc_rate_per_race=0.6,
        overtake_difficulty_s=0.3,
    )
    sim = RaceSimulator(deg_model, config, seed=42)
    result = sim.run(plans, n_sim=2000)

    # Build predicted vs actual table
    rows = []
    finish = state["finish"]
    for plan in plans:
        actual_row = finish[finish["driver_code"] == plan.code]
        if actual_row.empty:
            continue
        actual = int(actual_row["position"].iloc[0])
        predicted = float(result.expected_finish(plan.code))
        p_win = (
            float((result.positions[plan.code] == 1).mean())
            if plan.code in result.positions.columns
            else 0.0
        )
        p_podium = (
            float((result.positions[plan.code] <= 3).mean())
            if plan.code in result.positions.columns
            else 0.0
        )
        rows.append(
            {
                "driver_code": plan.code,
                "driver_name": driver_name(plan.code),
                "grid": plan.grid,
                "predicted_finish": predicted,
                "actual_finish": actual,
                "abs_error": abs(predicted - actual),
                "p_win": p_win,
                "p_podium": p_podium,
                "n_pits": len(plan.pit_laps),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("actual_finish")

    # Score every historical pit stop with the classifier
    race_pits = pits[(pits["year"] == race.year) & (pits["round_num"] == race.round_num)]
    feats = build_features(all_laps, race_pits, deg_model)
    if not feats.empty:
        probs = clf.predict_proba(feats[FEATURE_COLS])
        pit_score = feats[["driver_code", "pit_lap", "label"]].copy()
        pit_score["prob_gain"] = probs
        pit_score["driver_name"] = pit_score["driver_code"].map(driver_name)
    else:
        pit_score = pd.DataFrame()

    mae_positions = float(comparison["abs_error"].mean())

    return {
        "race": race.name,
        "year": race.year,
        "story": race.story,
        "total_laps": state["total_laps"],
        "median_pit_loss_s": state["pit_loss"],
        "n_drivers_modelled": len(plans),
        "mae_positions": mae_positions,
        "comparison_top10": comparison.head(10).to_dict("records"),
        "pit_classifier_examples": pit_score.head(8).to_dict("records")
        if not pit_score.empty
        else [],
    }


def write_markdown(analysis: dict, path: Path) -> None:
    if "error" in analysis:
        path.write_text(f"# {analysis['race']}\n\n_Error: {analysis['error']}_\n")
        return

    lines = [
        f"# {analysis['race']} {analysis['year']} - retrospective",
        "",
        f"> {analysis['story']}",
        "",
        f"- Total laps: **{analysis['total_laps']}**",
        f"- Median pit loss this circuit: **{analysis['median_pit_loss_s']:.2f}s**",
        f"- Drivers modelled: **{analysis['n_drivers_modelled']}**",
        f"- **Simulator MAE vs actual finishing position: {analysis['mae_positions']:.2f} positions**",
        "",
        "## Simulator vs actual (top 10 finishers)",
        "",
        "| Pos | Driver | Grid | Predicted | Actual | Abs. error | P(win) | P(podium) | Pits |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(analysis["comparison_top10"], 1):
        lines.append(
            f"| {i} | {r['driver_name']} | {r['grid']} | "
            f"{r['predicted_finish']:.2f} | {r['actual_finish']} | "
            f"{r['abs_error']:.2f} | {r['p_win']:.2f} | {r['p_podium']:.2f} | {r['n_pits']} |"
        )

    if analysis["pit_classifier_examples"]:
        lines.extend(
            [
                "",
                "## Undercut classifier on the actual stops",
                "",
                "For each historical pit stop, the calibrated LightGBM predicts the probability of gaining net positions within 5 laps. `Label=1` means it actually did.",
                "",
                "| Driver | Pit lap | Predicted P(gain) | Actual label |",
                "|---|---|---|---|",
            ]
        )
        for r in analysis["pit_classifier_examples"]:
            label_str = "yes" if r["label"] == 1 else "no"
            lines.append(
                f"| {r['driver_name']} | {int(r['pit_lap'])} | {r['prob_gain']:.3f} | {label_str} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Read the table with two questions in mind:",
            "1. *Did the simulator's expected finish track reality?* The MAE is the headline number.",
            "2. *Did the classifier rank actual undercuts correctly?* Compare predicted probabilities for stops that gained vs. stops that lost positions.",
            "",
            "**Honest framing:** the simulator is fed the *actual* pit-lap schedule, so it's not predicting strategy choice -- it's predicting outcome under the chosen strategy. A useful number to a strategist is the MAE between predicted and actual finishing position. The classifier's calibrated probabilities should rank gain-events higher than loss-events on average.",
            "",
            "_Generated by `scripts/race_retrospective.py`. Reproducible against the current MLflow run._",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Loading models...")
    deg = joblib.load(PROCESSED_DIR / "degradation_model.joblib")
    clf = joblib.load(PROCESSED_DIR / "undercut_classifier.joblib")

    # Load all laps once for the classifier feature builder
    import duckdb

    con = duckdb.connect(str(PROCESSED_DIR / "pitwall.duckdb"), read_only=True)
    laps_raw = con.execute("select * from fact_lap").df()
    pits = con.execute("select * from fact_pit_stop").df()
    con.close()
    all_laps = laps_raw.rename(
        columns={
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
            "position": "Position",
        }
    )

    all_results = []
    for race in RACES:
        print(f"\nAnalysing {race.year} {race.name}...")
        analysis = analyse_race(race, deg, clf, all_laps, pits)
        md_path = (
            WRITEUPS_DIR
            / f"retrospective_{race.year}_{race.round_num:02d}_{race.name.lower().replace(' ', '_').replace('grand_prix', 'gp')}.md"
        )
        write_markdown(analysis, md_path)
        print(f"  wrote {md_path.relative_to(PROJECT_ROOT)}")
        if "mae_positions" in analysis:
            print(f"  simulator MAE vs actual finish: {analysis['mae_positions']:.2f} positions")
        all_results.append(analysis)

    summary_path = PROCESSED_DIR / "retrospective.json"
    # Strip DataFrames from records before serialising
    serialisable = []
    for r in all_results:
        if "error" in r:
            serialisable.append(r)
            continue
        # Already dict records; just guard floats
        serialisable.append({k: v for k, v in r.items()})
    summary_path.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"\nSummary written to {summary_path.relative_to(PROJECT_ROOT)}")

    # Overall summary line
    maes = [r["mae_positions"] for r in all_results if "mae_positions" in r]
    if maes:
        print(
            f"\nAcross {len(maes)} retrospective races, simulator MAE = {np.mean(maes):.2f} positions on average."
        )


if __name__ == "__main__":
    main()
