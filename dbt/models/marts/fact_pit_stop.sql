{{ config(materialized='table') }}

-- One row per pit stop with the time lost vs. the driver's clean-air pace.
--
-- pit_loss = (in_lap + out_lap) - 2 * baseline
-- where baseline = 10th-percentile fuel-corrected pace within +/- 5 laps,
-- excluding the in-lap and out-lap themselves.

with all_laps as (
    select
        year, round_num, circuit_name, driver_code, lap_number,
        lap_time_fuel_corrected_s,
        pit_in_time_s,
        is_clean_lap,
        track_status,
        -- LEAD must run over ALL laps, then we filter to pit laps after.
        lead(lap_time_fuel_corrected_s) over (
            partition by year, round_num, driver_code
            order by lap_number
        )                                              as next_lap_time_s,
        quantile_cont(
            case when is_clean_lap then lap_time_fuel_corrected_s end,
            0.10
        ) over (
            partition by year, round_num, driver_code
            order by lap_number
            rows between 5 preceding and 5 following
        )                                              as baseline_s
    from {{ ref('int_clean_laps') }}
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
  -- Exclude SC/VSC/red-flag pit stops: when the in-lap is more than 1.8x the
  -- clean-air baseline, the pit happened under neutralisation. The economics
  -- of those stops are different and they're modelled separately.
  and lap_time_fuel_corrected_s < 1.8 * baseline_s
  and next_lap_time_s < 1.8 * baseline_s
  and track_status = '1'
