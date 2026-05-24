{{ config(materialized='table') }}

with clean as (
    select * from {{ ref('int_clean_laps') }}
    where is_clean_lap and stint_position > 1
),
stints as (
    select
        year,
        round_num,
        circuit_name,
        driver_code,
        team_name,
        stint,
        max(compound)                              as compound,
        min(lap_number)                            as first_lap,
        max(lap_number)                            as last_lap,
        count(*)                                   as clean_laps,
        median(lap_time_fuel_corrected_s)          as median_pace_s,
        avg(lap_time_fuel_corrected_s)             as mean_pace_s,
        -- OLS slope of fuel-corrected lap time vs. stint position
        regr_slope(lap_time_fuel_corrected_s, stint_position)      as deg_slope_s_per_lap,
        regr_intercept(lap_time_fuel_corrected_s, stint_position)  as deg_intercept_s,
        regr_r2(lap_time_fuel_corrected_s, stint_position)         as deg_fit_r2
    from clean
    group by 1, 2, 3, 4, 5, 6
)
select * from stints
where clean_laps >= 5    -- short stints produce noisy OLS slopes
