"""§4IB — a plan repeated in substance, not only byte-for-byte.

Request 21b295ef: 40 planner monologues, 0 byte-identical (the §4IA guard
never fired), three consecutive pairs at token-Jaccard 0.96 / 0.97 / 0.97 —
the same "let me test this exact format" reworded after a tool result had
just shown the format failing. Measured over 548 consecutive pairs since
09-10: median 0.29, p90 0.60, ≥0.9 in 4.6%, almost all inside the known
loops (d594668e ×6, 552a1ffd ×3, 21b295ef ×3).

World where each pin fails: the threshold drifts down into ordinary
consecutive plans (p90 = 0.60), the minimum-token floor is dropped (two
short thoughts match by accident), or the guard stops consulting the
near-duplicate test.
"""
import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (PLANNER_REPEAT_JACCARD, planner_repeat_needs_reask,
                                    planner_thoughts_repeat)

A = ("The last execute call showed the same 'reduced gaussian grid, npoints=127' SpecError. "
     "I've been hammering the same spec format repeatedly. The root cause: Grid(spec) uses a "
     "regex-based Generator that matches 'reduced gaussian grid' as a type name but then needs "
     "a separate parameter spec. Let me try the exact format from eckit docs: 'reduced gaussian "
     "grid, npoints=127' OR try 'reduced gaussian grid' with npoints as a separate field. "
     "Actually, the most reliable approach: the factory shows 'reduced_gg' as the type. The spec "
     "format for eckit is 'type, param=value'. Let me try 'reduced_gg, npoints=127'. If that "
     "fails, I'll just use the reduced_gg type directly. I've spent 25+ turns on this.")
B = A.replace("The last execute call showed the same", "The Grid.h header shows the same") \
     .replace("Let me try the exact format from eckit docs", "Let me try the format from the docs")
TOOL = [{"role": "tool", "name": "execute", "content": "ERR: ... cannot build grid without 'type'"}]


def test_the_live_pair_is_a_repeat():
    assert planner_thoughts_repeat(A, B)
    assert planner_repeat_needs_reask(A, B, TOOL, 0)


def test_byte_identical_still_counts():
    assert planner_thoughts_repeat(A, A)


def test_an_ordinary_next_plan_is_not_a_repeat():
    nxt = ("The Grid.h header shows Grid::Spec = spec::Spec, a key-value spec, so the string "
           "parser is the wrong entry point. Next: build a dict spec {'type': 'reduced_gg', 'N': 128} "
           "and call Grid with it; if the binding rejects a dict, read _eckit_geo's docstrings "
           "for the constructor signature before trying anything else.")
    assert not planner_thoughts_repeat(A, nxt)
    assert not planner_repeat_needs_reask(A, nxt, TOOL, 0)


def test_short_thoughts_never_match_by_accident():
    assert not planner_thoughts_repeat("run the tests again now", "run the tests again")
    assert not planner_thoughts_repeat("a b c d", "a b c d e")


def test_the_token_floor_is_what_blocks_a_short_near_match():
    """Two 19-token thoughts sharing 18 tokens sit at Jaccard 0.90 — over
    the threshold — and are still NOT a repeat, because both sides are
    under the 20-token floor. Drop the floor and this pair fires."""
    base = [f"w{i}" for i in range(18)]
    a = " ".join(base + ["alpha"])
    b = " ".join(base + ["beta"])
    assert len(set(a.split())) == 19 and len(set(b.split())) == 19
    assert not planner_thoughts_repeat(a, b)
    # the same shape above the floor IS a repeat (so the threshold, not
    # the floor, is doing the work there)
    base20 = [f"w{i}" for i in range(20)]
    assert planner_thoughts_repeat(" ".join(base20 + ["alpha"]), " ".join(base20 + ["beta"]))


def test_threshold_sits_above_the_measured_p90():
    """p90 of consecutive pairs is 0.60 and p95 0.86; anything at or under
    0.86 would fire on one plan in twenty."""
    assert PLANNER_REPEAT_JACCARD >= 0.88
    assert ag._PLANNER_REPEAT_MIN_TOKENS >= 10


def test_garbage_inputs():
    assert not planner_thoughts_repeat(None, A)
    assert not planner_thoughts_repeat(A, "")
    assert not planner_thoughts_repeat(3, A)


def test_guard_still_needs_a_new_tool_result_and_budget():
    assert not planner_repeat_needs_reask(A, B, [], 0)
    assert not planner_repeat_needs_reask(A, B, TOOL, ag._PLANNER_REPEAT_MAX_ASKS)
