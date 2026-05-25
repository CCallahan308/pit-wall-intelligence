"""Tests for the SC/VSC/GREEN pit-type classification in fact_pit_stop.

These run against the actual DuckDB warehouse and skip if it isn't built.
That's the right scope -- the classification logic is pure SQL and the
warehouse is the canonical place to validate it.
"""

from __future__ import annotations

import pytest

from pitwall.config import DUCKDB_PATH


@pytest.fixture
def warehouse():
    if not DUCKDB_PATH.exists():
        pytest.skip("DuckDB warehouse not built; skipping integration tests")
    import duckdb

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    yield con
    con.close()


def test_pit_type_values_are_in_known_set(warehouse):
    types = set(warehouse.execute("select distinct pit_type from fact_pit_stop").df()["pit_type"])
    assert types.issubset({"GREEN", "SC", "VSC", "YELLOW", "RED"})
    # There must be at least some GREEN stops or the warehouse is broken
    assert "GREEN" in types


def test_green_dominates(warehouse):
    """In modern F1, green-flag stops outnumber SC/VSC by ~10:1."""
    counts = warehouse.execute("select pit_type, count(*) as n from fact_pit_stop group by 1").df()
    counts_dict = dict(zip(counts["pit_type"], counts["n"], strict=False))
    green = counts_dict.get("GREEN", 0)
    other = sum(v for k, v in counts_dict.items() if k != "GREEN")
    assert green > other, f"GREEN ({green}) should outnumber all others ({other})"


def test_sc_stops_can_be_cheap(warehouse):
    """The whole point of the SC/VSC distinction: SC stops are often much
    cheaper than green-flag stops. The minimum SC stop should be < 10s
    (the textbook 'SC saves your pit stop' finding)."""
    sc_min = warehouse.execute(
        "select min(pit_loss_s) as m from fact_pit_stop where pit_type = 'SC'"
    ).df()
    if sc_min.empty or sc_min["m"].iloc[0] is None:
        pytest.skip("No SC stops in current warehouse")
    assert sc_min["m"].iloc[0] < 10.0, (
        f"Expected at least one SC stop under 10s, got min={sc_min['m'].iloc[0]:.2f}s"
    )


def test_green_stops_in_realistic_range(warehouse):
    """GREEN stops must fall in [5, 45] seconds. Anything else means the
    SC/VSC filter let through a non-green stop."""
    bad = (
        warehouse.execute(
            "select count(*) as n from fact_pit_stop "
            "where pit_type = 'GREEN' and (pit_loss_s < 5 or pit_loss_s > 45)"
        )
        .df()["n"]
        .iloc[0]
    )
    assert bad == 0, f"{bad} GREEN stops fall outside [5, 45] s"
