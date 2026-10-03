"""The §4KZ cleanup the operator confirmed ("delete 1 2 4 8, rebuild copies,
rest later"): only the listed rows; every kept fact must be live first."""
import runpy
from pathlib import Path

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4kz.py"), run_name="not_main")


def _live():
    return {tuple(c) for c in M["CANONICAL"]} | {tuple(r) for rows in M["DELETE"].values() for r in rows}


def test_every_listed_row_is_removed_and_no_kept_fact_is():
    todo = {tuple(r) for r in M["check"](_live())}
    assert todo.isdisjoint({tuple(c) for c in M["CANONICAL"]})
    assert ("user", "MARRIED_TO", "fotini") not in todo and ("thodoris", "BORN", "~2017-02") in todo


def test_a_missing_kept_fact_applies_nothing():
    live = _live() - {("user", "MARRIED_TO", "fotini")}
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](live)


def test_the_later_items_are_not_touched():
    rows = {tuple(r) for rows in M["DELETE"].values() for r in rows}
    for kept in [("user", "RUNS", "evolmonkey"), ("user", "HAS_ACCOUNT_WITH", "interactive brokers"),
                 ("user", "OWNS", "pista gp rr"), ("fotini", "HAS_BIRTHDATE", "january 10, 1982"),
                 ("user", "HAS_COMPANION", "fotini"), ("user", "HAS_CONDITION", "heart failure")]:
        assert kept not in rows
