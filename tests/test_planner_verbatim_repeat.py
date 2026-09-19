"""§4IA — a plan repeated byte-for-byte after a new tool result.

MEASURED (durable log, per day): byte-identical consecutive planner
monologues ran 0–2.6% of pairs through 2026-09-14, then 7.6% / 2.9% / 9.7%
on 09-15/16/17 — the step is §4HC (the prefix-aligned planner). All 15
repeats since then followed a tool that actually RAN (file write, execution
ok, browser, one exit-137), never a deferral with an empty delta; the
planner runs at temperature 0.0 and the delta DID carry the result
(`0e6cf008` turn 3 repeated turn 2 verbatim right after the exit-137).

Two changes, one instrument:
  * `build_aligned_planner_tail` puts the delta and the newest tool result
    LAST (after the plan JSON) — recency, where the model reads hardest;
  * `planner_repeat_needs_reask` re-asks ONCE with the newest result named
    when the thought is a verbatim repeat after a new tool result; the
    "Planner Repeat" log line is how the rate is read from now on.

World where each pin fails: the tail order regresses (delta buried again),
the guard fires with nothing new (a deferral loop), the budget is
unbounded, or the site stops routing through the helper.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (_PLANNER_REPEAT_HEAD_CHARS, _PLANNER_REPEAT_MAX_ASKS,
                                    build_aligned_planner_tail, planner_repeat_needs_reask,
                                    planner_repeat_steer)

TOOL = {"role": "tool", "name": "execute",
        "content": "--- EXECUTION RESULT ---\nEXIT CODE: 137 (KILLED after 4s …)\nKilled"}


# --- the tail -------------------------------------------------------------

def test_delta_and_newest_result_pointer_come_after_the_plan():
    tail = build_aligned_planner_tail("PLANNER-SYS", "TOOL (execute): EXIT CODE: 137",
                                      "", "### CURRENT PLAN (JSON)\n{\"id\":\"root\"}", [TOOL])
    i_sys = tail.index("PLANNER-SYS")
    i_plan = tail.index("CURRENT PLAN")
    i_delta = tail.index("NEW SINCE YOUR LAST PLAN")
    i_ptr = tail.index("NEWEST TOOL RESULT")
    assert i_sys < i_plan < i_delta < i_ptr
    assert "(TOOL (execute))" in tail
    # §4HN economy holds: the pointer NAMES the result, it does not repeat it
    assert tail.count("EXIT CODE: 137") == 1
    assert "Killed" not in tail


def test_cap_note_sits_between_plan_and_delta():
    tail = build_aligned_planner_tail("S", "D", "\n### YOUR LAST PLAN WAS CUT\n", "PLAN", [TOOL])
    assert tail.index("PLAN") < tail.index("YOUR LAST PLAN WAS CUT") < tail.index("NEW SINCE")


def test_no_new_tool_means_no_pointer():
    tail = build_aligned_planner_tail("S", "(nothing new)", "", "PLAN", [])
    assert "NEWEST TOOL RESULT" not in tail
    assert tail.rstrip().endswith("(nothing new)")
    tail2 = build_aligned_planner_tail("S", "D", "", "PLAN",
                                       [{"role": "assistant", "content": "x"}])
    assert "NEWEST TOOL RESULT" not in tail2


def test_pointer_names_the_last_tool_and_carries_no_content():
    big = {"role": "tool", "name": "browser", "content": "Z" * 5000}
    tail = build_aligned_planner_tail("S", "D", "", "PLAN", [TOOL, big])
    sec = tail[tail.index("NEWEST TOOL RESULT"):]
    assert "(TOOL (browser))" in sec and "execute" not in sec
    assert "Z" not in sec
    assert len(sec) < 200


def test_tail_survives_garbage_inputs():
    assert "NEW SINCE" in build_aligned_planner_tail("S", "D", None, None, None)
    assert "NEW SINCE" in build_aligned_planner_tail("S", "D", "", "", [None, 3, {"role": "tool"}])


# --- the guard ------------------------------------------------------------

@pytest.mark.parametrize("prev,thought,tools,asks,expected", [
    ("same", "same", [TOOL], 0, True),                    # the live shape
    ("same", "same", [], 0, False),                       # nothing new → correctly the same
    ("same", "same", [{"role": "assistant", "content": "x"}], 0, False),
    ("same", "different", [TOOL], 0, False),
    (None, "same", [TOOL], 0, False),                     # first plan has no previous
    ("same", "same", [TOOL], _PLANNER_REPEAT_MAX_ASKS, False),   # budget spent
    ("same", "same", [TOOL], _PLANNER_REPEAT_MAX_ASKS - 1, True),
    ("", "", [TOOL], 0, False),                           # empty thought is not a repeat
    ("same", "same", [TOOL], "many", False),              # garbage budget → no re-ask
])
def test_repeat_guard_table(prev, thought, tools, asks, expected):
    assert planner_repeat_needs_reask(prev, thought, tools, asks) is expected


def test_budget_is_small_and_positive():
    assert 1 <= _PLANNER_REPEAT_MAX_ASKS <= 5


def test_steer_names_the_result_and_forbids_the_repeat():
    s = planner_repeat_steer("execute", "EXIT CODE: 137")
    assert "REPEATED VERBATIM" in s
    assert "TOOL (execute): EXIT CODE: 137" in s
    assert "Do not repeat your previous thought" in s


# --- the site -------------------------------------------------------------

def _handle_chat_tree():
    for node in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "handle_chat":
            return node
    raise AssertionError("handle_chat not found")


def _calls(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name]


def test_site_routes_through_the_helpers():
    """The aligned planner builds its tail with the helper (one call), asks
    the guard with the previous thought and the new tool rows, and appends
    the steer on re-ask — none of the three is re-implemented inline."""
    tree = _handle_chat_tree()
    tails = _calls(tree, "build_aligned_planner_tail")
    guards = _calls(tree, "planner_repeat_needs_reask")
    steers = _calls(tree, "planner_repeat_steer")
    assert len(tails) == 1 and len(guards) == 1 and len(steers) == 1
    g = guards[0]
    assert [getattr(a, "id", None) for a in g.args[:2]] == ["_prev_planner_thought", "thought_content"]
    # the previous thought is remembered AFTER the guard decided — the
    # assignment `_prev_planner_thought = thought_content` sits below the
    # guard call in the same function
    remembers = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_prev_planner_thought" for t in n.targets)
                 and getattr(n.value, "id", "") == "thought_content"]
    assert len(remembers) == 1
    assert g.lineno < remembers[0].lineno
