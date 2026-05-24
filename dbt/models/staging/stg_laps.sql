{{ config(materialized='view') }}

-- Read every partitioned parquet under data/raw/laps/ into a unified view.
-- DuckDB's hive_partitioning auto-derives year/round/session from the path.
with raw as (
    select *
    from read_parquet(
        '../data/raw/laps/year=*/round=*/session=*.parquet',
        hive_partitioning = true
    )
)
select
    cast(Year as integer)                    as year,
    cast(Round as integer)                   as round_num,
    SessionType                              as session_type,
    CircuitName                              as circuit_name,
    Driver                                   as driver_code,
    Team                                     as team_name,
    cast(LapNumber as integer)               as lap_number,
    cast(Stint as integer)                   as stint,
    Compound                                 as compound,
    cast(TyreLife as double)                 as tyre_life,
    cast(epoch(LapTime) as double)           as lap_time_s,
    cast(epoch(Sector1Time) as double)       as sector1_s,
    cast(epoch(Sector2Time) as double)       as sector2_s,
    cast(epoch(Sector3Time) as double)       as sector3_s,
    PitInTime                                as pit_in_time,
    PitOutTime                               as pit_out_time,
    cast(TrackStatus as varchar)             as track_status,
    coalesce(Deleted, false)                 as deleted,
    cast(Position as integer)                as position
from raw
where LapTime is not null
