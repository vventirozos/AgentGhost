"""§4HZ — the DONE guard protects a task, not an id.

THE LIVE FAILURE (req d594668e, 2026-09-17). System 3's crisis pivot
replaced the plan with a fresh tree that reused `task_1..task_5` for
different work. `load_from_json` merges by id: it overwrote each
description and — guard keyed on the id — kept the old DONE. The next
planner monologue read *"task_1 (install ecmwf-api) is DONE, task_2
(upgrade pip + retry) is DONE"* for steps that never ran. Both crisis
interventions were neutralised this way.

World where each pin fails: the guard fires on the id alone again, or
`same_task` drifts to a threshold where a rewording counts as a new task
(dropping the protection the guard exists for).
"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.planning import SAME_TASK_JACCARD, TaskStatus, TaskTree, same_task


def _tree(*nodes):
    t = TaskTree()
    t.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                      "children": [dict(n) for n in nodes]})
    return t


def _task(i, desc, status):
    return {"id": f"task_{i}", "description": desc, "status": status, "children": []}


# --- same_task -----------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("check GRIB tooling", "check GRIB tooling", True),
    ("check GRIB tooling", "Check GRIB tooling is installed", True),   # rewording
    ("check GRIB tooling", "install ecmwf-api", False),                 # the live replacement
    ("upgrade pip + retry", "verify the install", False),
    ("", "", True),
    ("", "install ecmwf-api", False),                                   # unset ≠ anything
    ("a b c d e", "a b c", True),                                       # 3/5 = 0.6 → same
    ("a b c d e f", "a b c", False),                                    # 3/6 = 0.5 → new
])
def test_same_task_table(a, b, expected):
    assert same_task(a, b) is expected


def test_threshold_is_the_pinned_constant():
    """The two boundary rows above are exact for 0.6; a moved constant must
    move them deliberately."""
    assert SAME_TASK_JACCARD == 0.6


# --- the guard -----------------------------------------------------------

def test_same_task_keeps_the_done_guard(caplog):
    tree = _tree(_task(1, "check GRIB tooling", "DONE"))
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        tree.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                             "children": [_task(1, "check GRIB tooling is installed", "PENDING")]})
    assert tree.nodes["task_1"].status == TaskStatus.DONE
    assert tree.nodes["task_1"].description == "check GRIB tooling is installed"
    assert any("rejected status regression" in r.message for r in caplog.records)


def test_redescribed_id_adopts_the_incoming_status(caplog):
    """The live shape: System 3 reuses task_1 for 'install ecmwf-api'."""
    tree = _tree(_task(1, "check GRIB tooling", "DONE"),
                 _task(2, "find the MARS request format", "DONE"))
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        tree.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                             "children": [_task(1, "install ecmwf-api", "READY"),
                                          _task(2, "upgrade pip + retry", "PENDING")]})
    assert tree.nodes["task_1"].status == TaskStatus.READY
    assert tree.nodes["task_2"].status == TaskStatus.PENDING
    assert not any("rejected status regression" in r.message for r in caplog.records)
    assert sum("re-described" in r.message for r in caplog.records) == 2


def test_redescribed_id_that_arrives_done_stays_done():
    """Adopting the incoming status cuts both ways: a new task that the
    planner already marks DONE is DONE — no special-casing of the value."""
    tree = _tree(_task(1, "check GRIB tooling", "DONE"))
    tree.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                         "children": [_task(1, "install ecmwf-api", "DONE")]})
    assert tree.nodes["task_1"].status == TaskStatus.DONE


def test_non_done_nodes_are_unaffected_by_the_guard():
    tree = _tree(_task(1, "check GRIB tooling", "READY"))
    tree.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                         "children": [_task(1, "check GRIB tooling", "IN_PROGRESS")]})
    assert tree.nodes["task_1"].status == TaskStatus.IN_PROGRESS


def test_render_after_replan_shows_the_new_tasks_as_not_done():
    """What the planner READS next turn — the render — must not say DONE
    for work that never ran."""
    tree = _tree(_task(1, "check GRIB tooling", "DONE"))
    tree.load_from_json({"id": "root", "description": "goal", "status": "IN_PROGRESS",
                         "children": [_task(1, "install ecmwf-api", "READY")]})
    line = [ln for ln in tree.render().splitlines() if "task_1" in ln][0]
    assert "install ecmwf-api" in line and "(DONE)" not in line
