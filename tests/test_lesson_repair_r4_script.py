"""The one-off §4KW r4 playbook repair: every named row must resolve to
exactly one row, its new rules pass the write gates, and no rewrite keeps the
request specifics it exists to remove."""
import runpy
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "lesson_repair_4kw_r4.py")
M = runpy.run_path(SCRIPT, run_name="not_main")


def _rows():
    names = M["RETRACT"] + list(M["REKEY"]) + list(M["REWRITE"])
    return [{"trigger": t, "task": t, "mistake": "m", "solution": "s"} for t in names]


def test_the_spec_resolves_and_the_new_rules_pass_every_gate():
    M["check"](_rows())


def test_a_missing_row_applies_nothing():
    rows = _rows()[1:]
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](rows)


def test_a_duplicated_row_applies_nothing():
    rows = _rows()
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](rows + rows[:1])


@pytest.mark.parametrize("word", ["Hetzner", "Leonidas", "Fousekis", "Revolut", "KYC", "war scenarios",
                                  "frag_copy", "IFS", "kc_probe", "farewell", "logslow", "emp1", "dig it",
                                  "delete it"])
def test_no_rewrite_keeps_the_request_specifics(word):
    for new in M["REWRITE"].values():
        assert all(word.lower() not in str(v).lower() for v in new.values())


def test_a_re_keyed_lesson_admits_its_real_request():
    from ghost_agent.memory.lesson_scope import admits
    for _trig, req in M["REKEY"].items():
        assert admits({"scope": "request", "source_request": req}, req)
