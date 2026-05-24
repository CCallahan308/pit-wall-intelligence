{{ config(materialized='view') }}

-- A "clean" lap = no pit, clear track status, not deleted, valid lap time.
-- Adds fuel correction: subtract ~0.03s/kg of fuel mass remaining.
with base as (
    select * from {{ ref('stg_laps') }}
    where session_type = 'R'
),
fuel as (
    select
        *,
        max(lap_number) over (partition by year, round_num, driver_code) as total_laps,
        110.0 - ((110.0 - 1.0) / nullif(max(lap_number) over (
            partition by year, round_num, driver_code) - 1, 0)) * (lap_number - 1)
            as fuel_kg_at_lap_start
    from base
)
select
    *,
    lap_time_s - 0.03 * fuel_kg_at_lap_start                  as lap_time_fuel_corrected_s,
    row_number() over (
        partition by year, round_num, driver_code, stint
        order by lap_number
    )                                                          as stint_position,
    case
        when pit_in_time_s is not null then false
        when pit_out_time_s is not null then false
        when track_status <> '1' then false
        when deleted then false
        else true
    end                                                        as is_clean_lap
from fuel
