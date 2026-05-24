{{ config(materialized='view') }}

-- Read every partitioned parquet under data/raw/laps/ into a unified view.
-- DuckDB's hive_partitioning auto-derives year/round/session from the path.
-- LapTime / Sector*Time / PitIn|OutTime arrive as nanosecond BIGINTs (pandas
-- timedelta64[ns] in parquet) — divide by 1e9 to get seconds.
with raw as (
    select *
    from read_parquet(
        '../data/raw/laps/year=*/round=*/session=*.parquet',
        hive_partitioning = true
    )
)
select
    cast(Year as integer)                              as year,
    cast(Round as integer)                             as round_num,
    SessionType                                        as session_type,
    CircuitName                                        as circuit_name,
    Driver                                             as driver_code,
    Team                                               as team_name,
    cast(LapNumber as integer)                         as lap_number,
    cast(Stint as integer)                             as stint,
    Compound                                           as compound,
    cast(TyreLife as double)                           as tyre_life,
    cast(LapTime as double) / 1e9                      as lap_time_s,
    cast(Sector1Time as double) / 1e9                  as sector1_s,
    cast(Sector2Time as double) / 1e9                  as sector2_s,
    cast(Sector3Time as double) / 1e9                  as sector3_s,
    cast(PitInTime as double) / 1e9                    as pit_in_time_s,
    cast(PitOutTime as double) / 1e9                   as pit_out_time_s,
    cast(TrackStatus as varchar)                       as track_status,
    coalesce(Deleted, false)                           as deleted,
    cast(Position as integer)                          as position
from raw
where LapTime is not null
