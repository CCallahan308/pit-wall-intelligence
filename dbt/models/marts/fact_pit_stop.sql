{{ config(materialized='table') }}

with pits as (
    select
        year,
        round_num,
        circuit_name,
        driver_code,
        lap_number                                     as pit_lap,
        lap_time_fuel_corrected_s                      as in_lap_s,
        lead(lap_time_fuel_corrected_s) over (
            partition by year, round_num, driver_code
            order by lap_number
        )                                              as out_lap_s
    from {{ ref('int_clean_laps') }}
    where pit_in_time is not null
),
baseline as (
    select
        year, round_num, driver_code, lap_number,
        quantile_cont(lap_time_fuel_corrected_s, 0.10) over (
            partition by year, round_num, driver_code
            order by lap_number
            rows between 5 preceding and 5 following
        ) as baseline_s
    from {{ ref('int_clean_laps') }}
    where is_clean_lap
)
select
    p.year, p.round_num, p.circuit_name, p.driver_code,
    p.pit_lap, p.in_lap_s, p.out_lap_s, b.baseline_s,
    (p.in_lap_s + p.out_lap_s) - 2 * b.baseline_s     as pit_loss_s
from pits p
left join baseline b
  on p.year = b.year and p.round_num = b.round_num
 and p.driver_code = b.driver_code and p.pit_lap = b.lap_number
where p.out_lap_s is not null and b.baseline_s is not null
