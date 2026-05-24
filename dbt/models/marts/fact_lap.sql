{{ config(materialized='table', indexes=[{'columns': ['year', 'round_num', 'driver_code']}]) }}

select
    year,
    round_num,
    circuit_name,
    driver_code,
    team_name,
    lap_number,
    stint,
    stint_position,
    compound,
    tyre_life,
    lap_time_s,
    lap_time_fuel_corrected_s,
    sector1_s,
    sector2_s,
    sector3_s,
    position,
    is_clean_lap,
    track_status
from {{ ref('int_clean_laps') }}
