"""The §4LC lesson cleanup the operator confirmed ("go with your
recommendation"): only the listed lessons; the three kept ones must survive."""
import runpy
from pathlib import Path

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4lc_lessons.py"),
                   run_name="not_main")


def _playbook(extra=()):
    rows = [{"trigger": t} for g in M["DELETE"].values() for t in g] + [{"trigger": k} for k in M["KEEP"]]
    return rows + list(extra)


def test_every_listed_lesson_goes_and_the_kept_ones_stay():
    todo = M["check"](_playbook([{"trigger": "When restarting the nginx service"}]))
    assert len(todo) == 45
    assert not set(t.lower() for t in todo) & set(k.lower() for k in M["KEEP"])
    assert "When renaming a parameter or variable in a code file." in M["KEEP"]


def test_a_missing_lesson_applies_nothing():
    pb = [r for r in _playbook() if r["trigger"] != M["DELETE"]["B-private"][0]]
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](pb)


def test_a_missing_kept_lesson_applies_nothing():
    pb = [r for r in _playbook() if r["trigger"] != M["KEEP"][0]]
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](pb)


def test_only_request_lessons_have_their_counters_reset():
    pb = [{"trigger": "a", "scope": "request", "retrievals": 225, "helpful_retrievals": 179},
          {"trigger": "b", "retrievals": 40, "helpful_retrievals": 30}]
    assert M["reset_counters"](pb) == 1
    assert pb[0]["retrievals"] == 0 and pb[0]["helpful_retrievals"] == 0
    assert pb[1]["retrievals"] == 40
