"""The one-off r6 playbook repair (lesson producers): exact rows,
all-or-nothing, re-keyed triggers never collide, rewrites pass the gate."""
import runpy
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "lesson_repair_4kw_r6.py")
M = runpy.run_path(SCRIPT, run_name="not_main")


def _rows():
    return [{"trigger": t, "task": t, "mistake": "m", "solution": "s"}
            for t in M["RETRACT"] + list(M["REWRITE"]) + list(M["REKEY"])]


def test_the_spec_resolves():
    M["check"](_rows())


def test_a_missing_row_applies_nothing():
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_rows()[1:])


def test_a_rekey_onto_an_existing_trigger_applies_nothing():
    rows = _rows()
    taken = next(iter(M["REKEY"].values()))
    rows.append({"trigger": taken, "task": taken, "mistake": "m", "solution": "s"})
    with pytest.raises(SystemExit, match="collides"):
        M["check"](rows)


def test_every_rewritten_rule_passes_the_write_gate():
    from ghost_agent.memory.lesson_quality import is_actionable_lesson
    assert all(is_actionable_lesson("none", new["solution"], old) for old, new in M["REWRITE"].items())
