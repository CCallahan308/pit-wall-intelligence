{{ config(materialized='table') }}

-- One row per green-flag pit stop with the time lost vs. the driver's
-- clean-air baseline pace.
--
--   pit_loss = (in_lap + out_lap) - 2 * baseline
--
-- where baseline = 10th-percentile fuel-corrected pace within +/- 5 laps,
-- excluding the in-lap and out-lap themselves.
--
-- We aggressively filter out Safety Car / VSC / red-flag stops because their
-- economics aren't comparable to green-flag stops:
--   1) track_status on the in-lap must be '1' (green)
--   2) in-lap and out-lap must be < 1.6x the local baseline
--   3) local baseline must be < 1.4x the driver's race-wide clean median
--      (catches windows where the *entire* window was under SC)
--   4) computed pit_loss must fall in a plausible green-flag range [5, 45]

with driver_baseline as (
    -- One row per (race, driver) with their fastest clean-pace decile
    -- across the whole race. Used to detect SC-contaminated windows.
    select
        year, round_num, driver_code,
        quantile_cont(lap_time_fuel_corrected_s, 0.10) as race_clean_median_s
    from {{ ref('int_clean_laps') }}
    where is_clean_lap
    group by 1, 2, 3
),
all_laps as (
    select
        l.year, l.round_num, l.circuit_name, l.driver_code, l.lap_number,
        l.lap_time_fuel_corrected_s,
        l.pit_in_time_s,
        l.is_clean_lap,
        l.track_status,
        d.race_clean_median_s,
        lead(l.lap_time_fuel_corrected_s) over (
            partition by l.year, l.round_num, l.driver_code
            order by l.lap_number
        )                                              as next_lap_time_s,
        quantile_cont(
            case when l.is_clean_lap then l.lap_time_fuel_corrected_s end,
            0.10
        ) over (
            partition by l.year, l.round_num, l.driver_code
            order by l.lap_number
            rows between 5 preceding and 5 following
        )                                              as baseline_s
    from {{ ref('int_clean_laps') }} l
    left join driver_baseline d
      on l.year = d.year and l.round_num = d.round_num and l.driver_code = d.driver_code
)
select
    year, round_num, circuit_name, driver_code,
    lap_number                                          as pit_lap,
    lap_time_fuel_corrected_s                           as in_lap_s,
    next_lap_time_s                                     as out_lap_s,
    baseline_s,
    (lap_time_fuel_corrected_s + next_lap_time_s) - 2 * baseline_s
                                                        as pit_loss_s
from all_laps
where pit_in_time_s is not null
  and next_lap_time_s is not null
  and baseline_s is not null
  and track_status = '1'
  and lap_time_fuel_corrected_s < 1.6 * baseline_s
  and next_lap_time_s < 1.6 * baseline_s
  and baseline_s < 1.4 * race_clean_median_s
  and (lap_time_fuel_corrected_s + next_lap_time_s) - 2 * baseline_s between 5 and 45
