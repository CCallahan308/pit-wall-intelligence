"""Quick smoke test — ingest one session, run fuel correction, fit a curve.

Use this to verify the install works on a fresh machine.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

from pitwall.ingest.fastf1_client import fetch_session, persist, setup_cache
from pitwall.transform.fuel_correction import apply as fuel_apply
from pitwall.transform.stint_features import stint_summary


def main() -> None:
    setup_cache()
    print("Pulling 2024 Bahrain (round 1) race session...")
    frames = fetch_session(2024, 1, "R")
    if frames is None:
        raise SystemExit("No data returned from FastF1")
    persist(frames, 2024, 1, "R")

    laps = frames["laps"]
    print(f"  - {len(laps)} lap rows")

    corrected = fuel_apply(laps)
    print(
        f"  - fuel-corrected lap time range: {corrected['LapTimeFuelCorrected'].min():.2f} - {corrected['LapTimeFuelCorrected'].max():.2f} s"
    )

    summary = stint_summary(corrected.assign(CircuitName="Bahrain"))
    print(f"  - {len(summary)} stints summarised")
    print(summary.head(10).to_string())

    print("\n[OK] Smoke test passed.")


if __name__ == "__main__":
    main()
