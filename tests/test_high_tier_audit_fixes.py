"""Regression tests for the High-tier audit fixes:

* mcts._simulate_parallel: route() returns a content STRING on success;
  without an isinstance guard it hit str.get() → AttributeError →
  swallowed → every candidate collapsed to the 0.3 neutral default.
* dream._VALID_LESSON_DOMAINS: was missing "web_automation", so every
  web-automation self-play lesson was discarded by the generalization guard.
* graph.add_triplets: re-asserting a previously-expired functional fact
  did not reset valid_until → the fact stayed invisible after restart.
* prm.labels._build_state_for_step: pending_count/plan_depth were derived
  from the FUTURE (later steps / total length), leaking the label.
"""

import json
import sqlite3

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.mcts import MCTSReasoner, ActionCandidate


# -----------------------------------------------------------------
# mcts: route() returning a string must be handled (not 0.3 default)
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_simulate_handles_route_returning_string():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    # Production route() returns the content STRING, not a dict.
    client.route = AsyncMock(return_value=json.dumps(
        {"progress": 0.9, "cost": 0.1, "risk": 0.1}
    ))
    mcts = MCTSReasoner(llm_client=client, max_candidates=3)
    candidates = [ActionCandidate(description="run it", tool_name="execute")]

    result = await mcts._simulate_parallel(candidates, "context")

    # Score reflects the parsed JSON, NOT the 0.3 except-branch default.
    assert result[0].score == pytest.approx(0.9, abs=1e-6)
    assert result[0].score != 0.3
    # route returned non-empty → the chat_completion fallback is NOT used.
    client.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_simulate_falls_back_to_chat_completion_when_route_empty():
    client = MagicMock()
    client.route = AsyncMock(return_value=None)  # no worker pool
    client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps(
            {"progress": 0.5, "cost": 0.5, "risk": 0.5})}}]
    })
    mcts = MCTSReasoner(llm_client=client, max_candidates=3)
    candidates = [ActionCandidate(description="x", tool_name="execute")]
    result = await mcts._simulate_parallel(candidates, "context")
    client.chat_completion.assert_called()  # fell back
    assert 0.0 <= result[0].score <= 1.0


# -----------------------------------------------------------------
# dream: web_automation lessons are no longer discarded
# -----------------------------------------------------------------

def test_web_automation_in_valid_lesson_domains():
    from ghost_agent.core.dream import Dreamer
    assert "web_automation" in Dreamer._VALID_LESSON_DOMAINS


def test_generalization_guard_accepts_web_automation_lesson():
    from ghost_agent.core.dream import Dreamer
    lesson = {
        "trigger": "when automating a login flow inside a headless browser",
        "correct_pattern": "always await visibility of the selector before issuing a click",
        "domains": ["web_automation"],
    }
    ok, reason = Dreamer._generalization_guard(
        lesson,
        challenge="Scrape product prices from a paginated catalog page",
        setup_script="# fixture only",
        validation_script="assert result == expected",
    )
    assert ok is True, f"web_automation lesson was rejected: {reason!r}"


# -----------------------------------------------------------------
# graph: re-asserting an expired functional fact revives it
# -----------------------------------------------------------------

def test_reasserted_functional_fact_is_revived(tmp_path):
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)

    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "acme"}])
    # A different object for the functional predicate expires acme.
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "google"}])
    # Re-assert acme — must REVIVE it (valid_until back to NULL).
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "acme"}])

    with sqlite3.connect(gm.db_path) as conn:
        acme_vu = conn.execute(
            "SELECT valid_until FROM triplets WHERE subject=? AND predicate=? AND object=?",
            ("bob", "WORKS_AT", "acme"),
        ).fetchone()[0]
        google_vu = conn.execute(
            "SELECT valid_until FROM triplets WHERE subject=? AND predicate=? AND object=?",
            ("bob", "WORKS_AT", "google"),
        ).fetchone()[0]

    assert acme_vu is None, "re-asserted fact stayed expired (valid_until not reset)"
    assert google_vu is not None, "the superseded object should be expired"

    # And it must be visible to a fresh load (initialize_graph filters valid_until IS NULL).
    gm2 = GraphMemory(tmp_path)
    assert gm2.nx_graph.has_edge("bob", "acme")


# -----------------------------------------------------------------
# prm.labels: prefix-state must not leak the future
# -----------------------------------------------------------------

def test_build_state_for_step_does_not_leak_future():
    from ghost_agent.prm.labels import _build_state_for_step
    from ghost_agent.distill.schema import Trajectory, ToolCall

    short = Trajectory(
        user_request="do x",
        tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
        n_steps=2,
    )
    long = Trajectory(
        user_request="do x",
        tool_calls=[ToolCall(name="a"), ToolCall(name="b"),
                    ToolCall(name="c"), ToolCall(name="d")],
        n_steps=4,
    )

    s_short = _build_state_for_step(short, 0)
    s_long = _build_state_for_step(long, 0)

    # Same prefix (step 0) → identical state regardless of how the turn
    # continued. pending_count/plan_depth must NOT vary with the future.
    assert s_short.pending_count == s_long.pending_count
    assert s_short.plan_depth == s_long.plan_depth
    # Pinned to the neutral inference constants (no leakage, no skew).
    assert s_short.pending_count == 1
    assert s_short.plan_depth == 1


def test_build_state_for_step_prefix_fields_still_reflect_prefix():
    from ghost_agent.prm.labels import _build_state_for_step
    from ghost_agent.distill.schema import Trajectory, ToolCall

    traj = Trajectory(
        user_request="do x",
        tool_calls=[
            ToolCall(name="a", error="boom"),
            ToolCall(name="b"),
            ToolCall(name="c"),
        ],
        n_steps=3,
    )
    s = _build_state_for_step(traj, 2)
    assert s.steps_so_far == 2
    assert s.failures_so_far == 1          # the prior failed 'a'
    assert "a" in s.tools_used_this_turn
    # Future-independent constants regardless of step index.
    assert s.pending_count == 1 and s.plan_depth == 1
