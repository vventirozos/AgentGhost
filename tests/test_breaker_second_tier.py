"""§4IC — a breaker's second trip forces a report.

Request 3a2afac2: `edit churn` steered twice and the futility breaker once
("probe.py rewritten 6× and rerun 2× — steering strategy shift"); the
model then rewrote probe.py eight more times. Every breaker was a one-shot
advisory. Now the edit-churn check reports once both steers are spent and
the blind edits continue, and the futility breaker reports once the
rewrite/rerun cycle reaches FUTILITY_REPORT_WRITES/RUNS after its steer.

World where each pin fails: the second tier never fires, fires before the
first, fires twice, or the sites stop routing through the two decisions.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (EDIT_CHURN_REPORT_AFTER_STEERS, FUTILITY_REPORT_RUNS,
                                    FUTILITY_REPORT_WRITES, _EDIT_CHURN_MAX_STEERS,
                                    _EDIT_CHURN_STEER_AFTER, edit_churn_decision, futility_tier)


# --- edit churn ---------------------------------------------------------------

def test_live_sequence_steers_twice_then_reports_once():
    churn = {"counts": {}, "steers": 0}
    seq = [edit_churn_decision(churn, "probe.py") for _ in range(4 * _EDIT_CHURN_STEER_AFTER)]
    assert seq.count("steer") == _EDIT_CHURN_MAX_STEERS
    assert seq.count("report") == 1
    assert seq.index("report") > max(i for i, v in enumerate(seq) if v == "steer")
    assert churn["reported"] is True


def test_steer_resets_the_count_and_report_does_not_fire_early():
    churn = {"counts": {}, "steers": 0}
    for _ in range(_EDIT_CHURN_STEER_AFTER - 1):
        assert edit_churn_decision(churn, "a.py") == ""
    assert edit_churn_decision(churn, "a.py") == "steer"
    assert churn["counts"]["a.py"] == 0 and churn["steers"] == 1


def test_a_verified_run_between_edits_keeps_it_quiet():
    """The loop clears counts on a verify tool (unchanged); the decision
    must not carry a hidden counter of its own."""
    churn = {"counts": {}, "steers": 0}
    for _ in range(_EDIT_CHURN_STEER_AFTER - 1):
        edit_churn_decision(churn, "a.py")
    churn["counts"].clear()                    # what the loop does after `execute`
    assert edit_churn_decision(churn, "a.py") == ""


def test_report_tier_sits_after_the_steers():
    assert EDIT_CHURN_REPORT_AFTER_STEERS == _EDIT_CHURN_MAX_STEERS >= 1


# --- futility -----------------------------------------------------------------

def test_futility_steer_then_report_then_silence():
    si = {"probe.py": {"writes": 3, "runs": 2}}
    assert futility_tier(si, False, False)[0] == "steer"
    assert futility_tier(si, True, False)[0] == ""            # not yet the report tier
    si["probe.py"] = {"writes": FUTILITY_REPORT_WRITES, "runs": FUTILITY_REPORT_RUNS}
    kind, bn, rec = futility_tier(si, True, False)
    assert (kind, bn) == ("report", "probe.py") and rec["writes"] == FUTILITY_REPORT_WRITES
    assert futility_tier(si, True, True)[0] == ""              # once


def test_report_tier_never_precedes_the_steer():
    si = {"probe.py": {"writes": 12, "runs": 9}}
    assert futility_tier(si, False, False)[0] == "steer"


def test_futility_thresholds_are_above_the_steer():
    assert FUTILITY_REPORT_WRITES > 3 and FUTILITY_REPORT_RUNS > 2


def test_futility_tier_is_pure_on_garbage():
    assert futility_tier(None, False, False) == ("", None, None)
    assert futility_tier({"x.py": {}}, True, False) == ("", None, None)


# --- the sites ---------------------------------------------------------------

def _dispatch():
    for n in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_dispatch_and_process_tool_batch":
            return n
    raise AssertionError("_dispatch_and_process_tool_batch not found")


def _alert_kinds(node):
    return [ast.literal_eval(c.args[0]) for c in ast.walk(node)
            if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "blocker_report_alert"
            and c.args and isinstance(c.args[0], ast.Constant)]


def test_sites_route_through_the_decisions_and_force_a_final():
    fn = _dispatch()
    calls = {getattr(c.func, "id", ""): c for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "edit_churn_decision" in calls and "futility_tier" in calls
    kinds = _alert_kinds(fn)
    assert kinds.count("edit churn") == 1 and kinds.count("futility") == 1
    # each report branch sets force_final_response = True
    for kind in ("edit churn", "futility"):
        owners = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                  and kind in _alert_kinds(ast.Module(body=n.body, type_ignores=[]))]
        assert owners, kind
        inner = min(owners, key=lambda n: len(ast.unparse(n)))
        assigned = [t.id for s in ast.walk(ast.Module(body=inner.body, type_ignores=[]))
                    if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name)]
        assert "force_final_response" in assigned, kind
