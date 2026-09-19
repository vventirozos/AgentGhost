"""§4IC — the last budget turn is a report turn.

Request 3a2afac2 spent turn 40 on one more `find`, and the user received
the interstitial narration stitched together ("Found `eckit.geo.Grid`. Let
me explore it thoroughly. Now I understand the spec format…") under a
banner promising "recorded findings (project work log / ledger / notes)"
that did not exist (no project was open). The late verifier refuted it;
the deliverable was still nothing.

World where each pin fails: the last turn runs tools again, the banner
labels a forced report as a working-state dump, or it promises a ledger
with no project open.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (REPORT_TURN_MIN_TURN, blocker_report_alert,
                                    budget_exhausted_note, last_turn_needs_report)


@pytest.mark.parametrize("turn,max_turns,ffr,stop,expected", [
    (38, 40, False, False, True),        # §4IG: turn 39 of 40 — report turn, turn 40 = its retry
    (39, 40, False, False, False),       # the retry turn is already forced-final by then
    (37, 40, False, False, False),
    (38, 40, True, False, False),        # already a final turn
    (38, 40, False, True, False),        # already stopping
    (2, 3, False, False, False),         # short budgets keep their last turns
    (1, 3, False, False, False),         # …even at max−2: the budget floor, not the position, blocks it
    (0, 2, False, False, False),
    (REPORT_TURN_MIN_TURN - 1, REPORT_TURN_MIN_TURN + 1, False, False, True),
    (REPORT_TURN_MIN_TURN, REPORT_TURN_MIN_TURN + 1, False, False, False),
    ("x", 40, False, False, False),
])
def test_last_turn_table(turn, max_turns, ffr, stop, expected):
    assert last_turn_needs_report(turn, max_turns, ffr, stop) is expected


def test_min_turn_is_small():
    assert 1 <= REPORT_TURN_MIN_TURN <= 5


def test_banner_names_a_report_when_the_turn_was_reserved():
    note = budget_exhausted_note(40, report_forced=True, project_active=False)
    assert note.startswith("[TURN BUDGET REACHED]")
    assert "my report of where it stands" in note
    assert "working state" not in note
    assert "nothing is recorded outside this conversation" in note
    assert "project work log" not in note


def test_banner_names_a_dump_when_it_was_not():
    note = budget_exhausted_note(40, report_forced=False, project_active=True)
    assert note.startswith("[TURN BUDGET EXHAUSTED]")
    assert "NOT a finished result" in note
    assert "project work log / ledger / notes" in note


def test_alert_asks_for_the_three_parts_and_forbids_narration():
    a = blocker_report_alert("turn budget", "last turn")
    assert "Tools are OFF" in a
    for part in ("(1)", "(2)", "(3)", "what would unblock it"):
        assert part in a
    assert "'about to'" in a


# --- the sites ---------------------------------------------------------------

def _handle_chat():
    for n in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat":
            return n
    raise AssertionError("handle_chat not found")


def test_loop_reserves_the_last_turn_and_flags_it():
    fn = _handle_chat()
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Call)
              and getattr(n.test.func, "id", "") == "last_turn_needs_report"]
    assert len(guards) == 1
    body = ast.Module(body=guards[0].body, type_ignores=[])
    assigned = {t.id: ast.unparse(s.value) for s in ast.walk(body) if isinstance(s, ast.Assign)
                for t in s.targets if isinstance(t, ast.Name)}
    assert assigned.get("force_final_response") == "True"
    assert assigned.get("_report_turn_forced") == "True"
    alerts = [c for c in ast.walk(body) if isinstance(c, ast.Call)
              and getattr(c.func, "id", "") == "blocker_report_alert"]
    assert len(alerts) == 1 and ast.literal_eval(alerts[0].args[0]) == "turn budget"


def test_exhaustion_note_reads_the_flag_and_the_project():
    fn = _handle_chat()
    notes = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "budget_exhausted_note"]
    assert len(notes) == 1
    kw = {k.arg: ast.unparse(k.value) for k in notes[0].keywords}
    assert "_report_turn_forced" in kw["report_forced"]
    assert "_captured_project_id" in kw["project_active"]
