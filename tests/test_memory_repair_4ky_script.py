"""The §4KY cleanup the operator confirmed ("delete a b c broken wrist is
wrong"): only listed owner edges, never a life fact the operator kept."""
import runpy
from pathlib import Path

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4ky.py"), run_name="not_main")


def test_the_broken_wrist_is_listed_and_no_kept_life_fact_is():
    edges = {tuple(e) for e in M["EDGES"]}
    assert ("user", "EXPERIENCED", "broken wrist") in edges
    for kept in [("user", "HAS_CONDITION", "asthma"), ("user", "MARRIED_TO", "elena"),
                 ("user", "WORKS_AT", "orbitdata"), ("user", "HAS_ACCOUNT_WITH", "interactive brokers"),
                 ("user", "MANAGES", "mysql cluster"), ("user", "HAS_SON", "kosmas")]:
        assert kept not in edges


def test_only_live_rows_are_touched():
    live = {tuple(M["EDGES"][0])}
    assert M["check"](live) == [M["EDGES"][0]]


def test_an_edge_without_an_owner_end_aborts():
    check = M["check"]
    orig = check.__globals__["EDGES"]
    check.__globals__["EDGES"] = [["elena", "LIKES", "sea"]]
    try:
        with pytest.raises(SystemExit, match="nothing applied"):
            check(set())
    finally:
        check.__globals__["EDGES"] = orig
