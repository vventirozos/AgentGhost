"""§4IC — a breaker's second trip forced a report. §4UI (2026-11-12, operator:
"don't restrict the model too much while it's coding") REMOVED the report
tiers of the edit-churn and futility breakers: the steers are advice. These
pins now say the second tier stays gone.

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

def test_live_sequence_steers_twice_and_never_reports():
    churn = {"counts": {}, "steers": 0}
    seq = [edit_churn_decision(churn, "probe.py") for _ in range(6 * _EDIT_CHURN_STEER_AFTER)]
    assert seq.count("steer") == _EDIT_CHURN_MAX_STEERS
    assert "report" not in seq


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

def test_futility_steers_once_and_never_reports():
    si = {"probe.py": {"writes": 3, "runs": 2}}
    assert futility_tier(si, False, False)[0] == "steer"
    si["probe.py"] = {"writes": FUTILITY_REPORT_WRITES * 3, "runs": FUTILITY_REPORT_RUNS * 3}
    assert futility_tier(si, True, False)[0] == ""
    assert futility_tier(si, True, True)[0] == ""


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


def test_sites_route_through_the_decisions_and_force_no_report():
    fn = _dispatch()
    calls = {getattr(c.func, "id", ""): c for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "edit_churn_decision" in calls and "futility_tier" in calls
    kinds = _alert_kinds(fn)
    assert "edit churn" not in kinds and "futility" not in kinds


@pytest.mark.parametrize("target", ["results/a.md", "notes.txt", " report.MD ", "dir/x.Txt"])
def test_prose_targets_are_never_counted(target):
    """§4QB-c (operator 2026-10-20): .md/.txt have nothing to run."""
    churn = {"counts": {}, "steers": 0}
    assert {edit_churn_decision(churn, target) for _ in range(4 * _EDIT_CHURN_STEER_AFTER)} == {""}
    assert churn == {"counts": {}, "steers": 0}


@pytest.mark.parametrize("target", ["page.mdx", "a.md.py", "notes.txt.sh", "md", "readme"])
def test_other_targets_still_count(target):
    churn = {"counts": {}, "steers": 0}
    seq = [edit_churn_decision(churn, target) for _ in range(_EDIT_CHURN_STEER_AFTER)]
    assert seq[-1] == "steer"


def test_prose_edits_do_not_reset_a_code_files_count():
    churn = {"counts": {}, "steers": 0}
    for _ in range(_EDIT_CHURN_STEER_AFTER - 1):
        edit_churn_decision(churn, "app.py")
        edit_churn_decision(churn, "notes.md")
    assert "notes.md" not in churn["counts"]
    assert edit_churn_decision(churn, "app.py") == "steer"
