"""One-task-per-turn enforcement for the interactive agent loop.

A single user go-ahead ("start task 1", "proceed", "next") must advance EXACTLY
one project task and then stop — not grind the whole tree in one request
(observed live: "start task 1" built 8 tasks because the re-injected NEXT TASK
pointer kept pulling the model forward and nothing forced the turn to end).

Two mechanisms guard this:
  * core.agent._manage_projects_closed_a_task — detects, from the tool's store
    read-back, that a task actually closed to DONE (so the loop can force the
    turn to wrap up). Gated/held updates do NOT trip it.
  * core.project_advancer.classify_advance_intent — distinguishes a single
    go-ahead from an explicit batch ("do the next 3", "finish the project"),
    which is allowed to run multiple tasks (via the autoadvance tool).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import (
    _manage_projects_closed_a_task, _scrub_fallback_message,
)
from ghost_agent.core.project_advancer import classify_advance_intent


# ------------------------------------------------ task-close detection (helper)

def test_detects_done_close_from_readback():
    out = json.dumps({"updated": [{"id": "abc", "status": "DONE",
                                   "result_summary": "wrote parser.py"}],
                      "count": 1})
    assert _manage_projects_closed_a_task("manage_projects", out) is True


def test_multi_id_update_with_one_done_trips():
    out = json.dumps({"updated": [
        {"id": "a", "status": "IN_PROGRESS"},
        {"id": "b", "status": "DONE"}], "count": 2})
    assert _manage_projects_closed_a_task("manage_projects", out) is True


def test_held_gated_update_does_not_trip():
    # The DONE-gate held the task at PENDING pending verification evidence —
    # nothing actually closed, so the turn must NOT be forced to stop.
    out = json.dumps({"updated": [{"id": "a", "status": "PENDING"}],
                      "count": 1, "gated_unverified": ["a"]})
    assert _manage_projects_closed_a_task("manage_projects", out) is False


def test_non_done_status_does_not_trip():
    out = json.dumps({"updated": [{"id": "a", "status": "IN_PROGRESS"}]})
    assert _manage_projects_closed_a_task("manage_projects", out) is False


def test_other_tool_never_trips():
    # Even a file_system result that happens to contain the substring.
    assert _manage_projects_closed_a_task(
        "file_system", '{"status": "DONE"}') is False


def test_task_list_readonly_does_not_trip():
    # A task_list showing an already-DONE task is not a close event: its shape
    # is a `tasks` list, not an `updated` list.
    out = json.dumps({"tasks": [{"id": "a", "status": "DONE"}]})
    assert _manage_projects_closed_a_task("manage_projects", out) is False


def test_garbage_result_is_safe():
    assert _manage_projects_closed_a_task("manage_projects", "not json") is False
    assert _manage_projects_closed_a_task("manage_projects", "") is False


# --------------------------------------------------- single vs batch go-ahead

@pytest.mark.parametrize("msg", [
    "start task 1", "proceed", "next task", "continue", "go", "do task 2",
])
def test_single_go_ahead_is_not_batch(msg):
    # These must NOT be treated as a batch — so the one-task gate applies.
    assert classify_advance_intent(msg)["mode"] == "one"


@pytest.mark.parametrize("msg,mode", [
    ("do the next 3 tasks", "n"),
    ("proceed with all remaining tasks", "all"),
    ("finish the project", "all"),
    ("complete the project", "all"),
    ("two more tasks", "n"),
])
def test_batch_go_ahead_detected(msg, mode):
    # These bypass the one-task gate (they route to the autoadvance tool).
    assert classify_advance_intent(msg)["mode"] == mode


@pytest.mark.parametrize("msg,count", [
    ("proceed with task 3 and 4", 2),
    ("yes, proceed with task 3 and 4.", 2),
    ("do tasks 3 and 4", 2),
    ("task 3 and task 4", 2),
    ("tasks 3, 4 and 5", 3),
])
def test_enumerated_tasks_are_batch(msg, count):
    # "task 3 and 4" is TWO tasks — must be a batch, else the one-task gate
    # stops after task 3 and leaves task 4 half-done (observed live).
    out = classify_advance_intent(msg)
    assert out["mode"] == "n"
    assert out["count"] == count


@pytest.mark.parametrize("msg", [
    "start task 1",          # single index, not an enumeration
    "do task 2",
    "proceed with task 3",
])
def test_single_indexed_task_is_not_batch(msg):
    assert classify_advance_intent(msg)["mode"] == "one"


# --------------------------------------- scrub-fallback message (gate vs mismatch)

def test_fallback_message_when_task_closed_is_a_completion_not_a_rephrase():
    # The gate finalized the turn after a task closed — the dropped tool_call
    # was the model trying to start the next task. Don't tell the user to
    # rephrase a request that already succeeded.
    msg = _scrub_fallback_message("file_system", task_closed=True)
    assert "Task complete" in msg
    assert "proceed" in msg.lower() and "next" in msg.lower()
    assert "rephrase" not in msg.lower()
    assert "wasn't executed" not in msg.lower()


def test_fallback_message_for_genuine_routing_mismatch():
    # No task closed — a real planner/model mismatch. Keep the generic guidance.
    msg = _scrub_fallback_message("file_system", task_closed=False)
    assert "wasn't executed" in msg
    assert "file_system" in msg
    assert "rephrase" in msg.lower()


def test_fallback_message_handles_unknown_tool():
    msg = _scrub_fallback_message("", task_closed=False)
    assert "the command" in msg
    assert "Intended tool" not in msg          # no dangling backticks/empty name
