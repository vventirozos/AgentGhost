"""The §4LC cleanup the operator confirmed ("1. none 3 yes 4 a i run evolmonkey,
4b yes 4 c no"): only the listed rows; every kept fact must be live first."""
import runpy
from pathlib import Path

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4lc.py"), run_name="not_main")


def _live(n_agent=102):
    agent = {(("ai", "assistant", "system")[i % 3], "DID", f"thing {i}") for i in range(n_agent)}
    return ({tuple(c) for c in M["CANONICAL"]} | {tuple(r) for rows in M["DELETE"].values() for r in rows}
            | agent | {("user", "HAS_SON", "thodoris"), ("user", "GREETED", "ai"), ("ghost", "IS_A", "framework")})


def test_the_agent_edges_and_listed_rows_go_and_nothing_else():
    todo = {tuple(r) for r in M["check"](_live())}
    assert len(todo) == 102 + 5
    assert ("user", "WORKS_AT", "evolmonkey") in todo and ("user", "OWNS", "pista gp rr") in todo
    # kept: the owner's facts, an edge merely POINTING at the agent, an unconfirmed agent-ish subject
    for kept in [("user", "RUNS", "evolmonkey"), ("user", "HAS_SON", "thodoris"), ("user", "GREETED", "ai"),
                 ("ghost", "IS_A", "framework"), ("fotini", "HAS_BIRTHDATE", "january 10, 1982")]:
        assert kept not in todo


def test_a_different_agent_edge_count_applies_nothing():
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_live(n_agent=103))


def test_a_missing_kept_fact_applies_nothing():
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_live() - {("user", "RUNS", "evolmonkey")})


def test_the_profile_gets_the_role_and_both_birth_dates():
    assert {(c, k): v for c, k, v in M["PROFILE_SET"]} == {
        ("root", "role"): "Runs EvolMonkey", ("root", "birthdate"): "1980-01-29",
        ("relationships", "wife_birthdate"): "1982-01-10"}
