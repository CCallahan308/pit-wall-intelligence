"""Pandera schemas — these run as a data-quality gate in CI."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series

from pitwall.config import COMPOUND_ORDER


class LapSchema(pa.DataFrameModel):
    Year: Series[int] = pa.Field(ge=1950, le=2050)
    Round: Series[int] = pa.Field(ge=1, le=30)
    Driver: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 4})
    LapNumber: Series[int] = pa.Field(ge=1, le=100)
    Stint: Series[int] = pa.Field(ge=1, le=10)
    Compound: Series[str] = pa.Field(isin=COMPOUND_ORDER, nullable=True)
    LapTimeSeconds: Series[float] = pa.Field(ge=50.0, le=300.0, nullable=True)

    class Config:
        strict = False
        coerce = True


class StintSchema(pa.DataFrameModel):
    Year: Series[int]
    Round: Series[int]
    Driver: Series[str]
    Stint: Series[int]
    Compound: Series[str] = pa.Field(isin=COMPOUND_ORDER, nullable=True)
    StintLength: Series[int] = pa.Field(ge=1, le=80)
    DegSlopeSPerLap: Series[float] = pa.Field(ge=-1.0, le=2.0, nullable=True)
    MeanPaceCleanS: Series[float] = pa.Field(ge=50.0, le=300.0, nullable=True)

    class Config:
        strict = False
        coerce = True
