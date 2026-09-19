"""§4IC — an abort from outside is not a verdict on my work.

Request 3cb143fc was cancelled by a process restart 22 s in; §4IB recorded
it (`[ATTEMPT_ABORTED_TURN]`, FAILED for the corpus) and the derived mood
immediately read it as a failure: "mood idle → stuck: 3 of my last 5
verdict-bearing turns failed, including the latest". The diary's verdict
back-scan now skips rows an abort produced.

World where each pin fails: the back-scan counts aborted rows again, or
the marker check narrows to one field.
"""
import datetime as dt
import json

import pytest

from ghost_agent.selfhood.autobiographical import (ABORTED_ROW_MARKS, AUTOBIO_FILENAME,
                                                   AutobiographicalMemory, is_aborted_record)


@pytest.mark.parametrize("row,aborted", [
    ({"answer_gist": "partial\n\n[ATTEMPT_ABORTED_TURN] Turn aborted: cancelled: client disconnected."}, True),
    ({"summary": "I worked on x and it didn't land: Turn aborted: process shutdown."}, True),
    ({"failure_reason": "runtime abort marker [ATTEMPT_ABORTED_TURN]"}, True),
    ({"summary": "I worked on x and it failed: verifier refuted", "outcome": "failed"}, False),
    ({"answer_gist": "[ATTEMPT_ABORTED_NO_PROGRESS] I repeated the same action"}, False),  # the agent's own abort IS a verdict
    ("not a dict", False),
    ({}, False),
])
def test_is_aborted_record(row, aborted):
    assert is_aborted_record(row) is aborted


def test_marks_are_the_recorders_own_strings():
    """Both marks must match what `_record_aborted_turn` writes — read off
    the string constants in its tree (an f-string's literal parts), not
    the text."""
    import ast
    import inspect
    import textwrap
    from ghost_agent.core import agent as ag
    tree = ast.parse(textwrap.dedent(inspect.getsource(ag.GhostAgent._record_aborted_turn)))
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for m in ABORTED_ROW_MARKS:
        assert any(m in c for c in consts), m


def _write_diary(path, rows):
    now = dt.datetime.utcnow()
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            base = {"id": f"e{i}", "trajectory_id": f"t{i}",
                    "timestamp": (now - dt.timedelta(minutes=len(rows) - i)).isoformat() + "Z",
                    "summary": row.get("summary", "I worked on something."),
                    "outcome": row.get("outcome", "unknown")}
            base.update(row)
            f.write(json.dumps(base) + "\n")


def test_recent_verdicts_skips_aborted_rows_but_keeps_real_failures(tmp_path):
    p = tmp_path / AUTOBIO_FILENAME
    _write_diary(p, [
        {"outcome": "passed"},
        {"outcome": "failed", "summary": "verifier refuted it"},
        {"outcome": "failed", "answer_gist": "[ATTEMPT_ABORTED_TURN] Turn aborted: process shutdown."},
        {"outcome": "passed"},
        {"outcome": "failed", "summary": "and it didn't land: Turn aborted: cancelled: client disconnected."},
    ])
    log = AutobiographicalMemory(tmp_path)
    assert log.recent_verdicts(limit=5, max_age_days=7) == ["passed", "failed", "passed"]


def test_recent_verdicts_unchanged_without_aborts(tmp_path):
    p = tmp_path / AUTOBIO_FILENAME
    _write_diary(p, [{"outcome": "passed"}, {"outcome": "failed"}, {"outcome": "unknown"}, {"outcome": "passed"}])
    log = AutobiographicalMemory(tmp_path)
    assert log.recent_verdicts(limit=5, max_age_days=7) == ["passed", "failed", "passed"]
