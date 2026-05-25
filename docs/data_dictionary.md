# Data Dictionary

## `fact_lap` — one row per driver-lap

| Column | Type | Description |
|---|---|---|
| `year` | int | Season year |
| `round_num` | int | Round number in the season (1 = season opener) |
| `circuit_name` | str | Official event name (e.g. "Monaco Grand Prix") |
| `driver_code` | str | 3-letter driver code (e.g. "VER", "HAM") |
| `team_name` | str | Constructor name (e.g. "Red Bull Racing") |
| `lap_number` | int | 1-indexed lap of the race |
| `stint` | int | 1-indexed stint number for this driver-race |
| `stint_position` | int | 1-indexed position within the stint |
| `compound` | str | SOFT / MEDIUM / HARD / INTERMEDIATE / WET |
| `tyre_life` | float | Laps on this set of tyres (incl. prior sessions) |
| `lap_time_s` | float | Raw lap time in seconds |
| `lap_time_fuel_corrected_s` | float | Lap time corrected for fuel mass |
| `sector1_s`, `sector2_s`, `sector3_s` | float | Sector splits in seconds |
| `position` | int | Track position at end of lap |
| `is_clean_lap` | bool | True if suitable for pace/deg modeling |
| `track_status` | str | FastF1 single-character status code; '1' = green |

## `fact_stint` — one row per (race, driver, stint)

| Column | Type | Description |
|---|---|---|
| `year`, `round_num`, `circuit_name`, `driver_code`, `team_name`, `stint` | — | Composite key + descriptors |
| `compound` | str | Tyre compound for this stint |
| `first_lap`, `last_lap` | int | First and last green-lap number |
| `clean_laps` | int | Count of clean laps in this stint |
| `median_pace_s`, `mean_pace_s` | float | Fuel-corrected pace summaries |
| `deg_slope_s_per_lap` | float | OLS slope of fuel-corrected time vs. stint position |
| `deg_intercept_s` | float | OLS intercept (theoretical lap-0 pace) |
| `deg_fit_r2` | float | R² of the linear degradation fit |

## `fact_pit_stop` — one row per pit stop

| Column | Type | Description |
|---|---|---|
| `year`, `round_num`, `circuit_name`, `driver_code` | — | Composite key |
| `pit_lap` | int | Lap on which the driver entered the pits |
| `in_lap_s`, `out_lap_s` | float | Fuel-corrected in-lap and out-lap times |
| `baseline_s` | float | 10th-percentile clean-air pace within ±5 laps |
| `pit_loss_s` | float | `(in_lap + out_lap) - 2 * baseline` |
| `pit_type` | str | GREEN / SC / VSC / YELLOW / RED — see methodology §8 |
| `in_lap_status`, `out_lap_status` | str | Raw FastF1 track-status string for the in-lap and out-lap |
