{{ config(materialized='table') }}

-- One row per pit stop with the time lost vs. the driver's clean-air pace,
-- AND a pit_type label that distinguishes green-flag, SC, VSC, yellow, and
-- red-flag stops. This is the distinction every real strategy engineer makes:
--
--   GREEN  : normal racing, full pit-loss penalty (~22s field-wide).
--   VSC    : Virtual Safety Car -- field holds a delta time but doesn't
--            bunch up; pit costs ~12-15s less than green.
--   SC     : Full Safety Car -- field bunches up under SC pace; pit can
--            cost as little as 5-8s of net track position.
--   YELLOW : Local yellow flag at the pit window; minor pace impact.
--   RED    : Red-flag stop; tyre change rules differ, comparable to a free
--            stop in many cases.
--
-- FastF1 packs consecutive status codes for a lap into a single string
-- (e.g. '12' means the lap saw both green (1) and yellow (2) at some point).
-- We detect SC/VSC/etc. by substring presence in the status of any lap in
-- the in-lap + out-lap + 2-lap surrounding window.
--
-- Reference: FastF1 TrackStatus codes
--   1 = Track Clear
--   2 = Yellow Flag
--   4 = Safety Car (full)
--   5 = Red Flag
--   6 = Virtual Safety Car Deployed
--   7 = Virtual Safety Car Ending

with driver_baseline as (
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
        lead(l.lap_time_fuel_corrected_s) over w  as next_lap_time_s,
        lead(l.track_status) over w               as next_lap_status,
        -- Track status of nearby laps so we can classify the pit window
        lag(l.track_status, 1) over w             as prev1_status,
        lag(l.track_status, 2) over w             as prev2_status,
        lead(l.track_status, 2) over w            as fwd2_status,
        quantile_cont(
            case when l.is_clean_lap then l.lap_time_fuel_corrected_s end,
            0.10
        ) over (
            partition by l.year, l.round_num, l.driver_code
            order by l.lap_number
            rows between 5 preceding and 5 following
        )                                          as baseline_s
    from {{ ref('int_clean_laps') }} l
    left join driver_baseline d
      on l.year = d.year and l.round_num = d.round_num and l.driver_code = d.driver_code
    window w as (
        partition by l.year, l.round_num, l.driver_code order by l.lap_number
    )
),
classified as (
    select
        year, round_num, circuit_name, driver_code,
        lap_number                                                       as pit_lap,
        lap_time_fuel_corrected_s                                        as in_lap_s,
        next_lap_time_s                                                  as out_lap_s,
        baseline_s,
        (lap_time_fuel_corrected_s + next_lap_time_s) - 2 * baseline_s   as pit_loss_s,
        track_status                                                     as in_lap_status,
        next_lap_status                                                  as out_lap_status,
        -- Concatenate the window of statuses for classification
        coalesce(prev2_status, '') || '|' || coalesce(prev1_status, '') || '|' ||
            coalesce(track_status, '') || '|' || coalesce(next_lap_status, '') || '|' ||
            coalesce(fwd2_status, '')                                    as window_status,
        race_clean_median_s
    from all_laps
    where pit_in_time_s is not null
      and next_lap_time_s is not null
      and baseline_s is not null
),
labelled as (
    select
        *,
        -- Classification order matters: SC trumps VSC trumps yellow trumps green
        case
            when position('5' in window_status) > 0 then 'RED'
            when position('4' in window_status) > 0 then 'SC'
            when position('6' in window_status) > 0
                 or position('7' in window_status) > 0 then 'VSC'
            when position('2' in window_status) > 0 then 'YELLOW'
            else 'GREEN'
        end                                                              as pit_type
    from classified
)
select
    year, round_num, circuit_name, driver_code, pit_lap,
    in_lap_s, out_lap_s, baseline_s, pit_loss_s,
    pit_type,
    in_lap_status, out_lap_status
from labelled
where
    -- Keep all classified types but bound to physically plausible values.
    -- A GREEN stop should be in [5, 45]; SC/VSC/RED can be much smaller or
    -- even negative (lapped car catches up under neutralisation).
    pit_loss_s between -10 and 50
    -- For GREEN stops only, tighten the upper bound; SC stops have wider
    -- legitimate variance.
    and (pit_type != 'GREEN' or pit_loss_s between 5 and 45)
    -- Drop stops where the baseline window was itself SC-contaminated
    and (pit_type != 'GREEN' or baseline_s < 1.4 * race_clean_median_s)
