"""Per-request turn facts (core/turn_facts.py) + the §4I Phase 1 router stamp.

The module exists because a plain context attribute belongs to whichever
request is running at trajectory-write time, not to the one being written —
found live, in both directions, on the experiment-arm stamp.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import turn_facts as tf
from tests.helpers import make_agent, make_context


def _ctx():
    return SimpleNamespace()


def test_record_and_read_back():
    ctx = _ctx()
    tf.record(ctx, "req-1", router_label="hard", router_confidence=0.1)
    assert tf.facts_for(ctx, "req-1") == {"router_label": "hard",
                                          "router_confidence": 0.1}


def test_facts_are_request_scoped():
    """The whole point: a later request must not overwrite an earlier one's
    facts while that turn's streamed drain is still pending."""
    ctx = _ctx()
    tf.record(ctx, "req-A", router_label="hard")
    tf.record(ctx, "req-B", router_label="easy")
    assert tf.facts_for(ctx, "req-A") == {"router_label": "hard"}
    assert tf.facts_for(ctx, "req-B") == {"router_label": "easy"}


def test_later_calls_merge_rather_than_replace():
    ctx = _ctx()
    tf.record(ctx, "r", router_label="hard")
    tf.record(ctx, "r", router_confidence=0.2)
    assert tf.facts_for(ctx, "r") == {"router_label": "hard",
                                      "router_confidence": 0.2}


def test_none_values_and_empty_calls_are_ignored():
    ctx = _ctx()
    tf.record(ctx, "r", router_label=None)
    tf.record(ctx, "r")
    tf.record(ctx, "", router_label="x")
    assert tf.facts_for(ctx, "r") == {}
    assert tf.facts_for(ctx, "") == {}


def test_ring_is_bounded_and_lru():
    ctx = _ctx()
    for i in range(tf._MAX_REQUESTS + 5):
        tf.record(ctx, f"r{i}", n=i)
    assert len(getattr(ctx, tf._ATTR)) <= tf._MAX_REQUESTS
    assert tf.facts_for(ctx, "r0") == {}          # evicted, not leaked
    assert tf.facts_for(ctx, f"r{tf._MAX_REQUESTS + 4}") == {"n": tf._MAX_REQUESTS + 4}


def test_recording_refreshes_lru_position():
    ctx = _ctx()
    tf.record(ctx, "keep", n=1)
    for i in range(tf._MAX_REQUESTS - 1):
        tf.record(ctx, f"f{i}", n=i)
    tf.record(ctx, "keep", n=2)                   # touched → newest
    for i in range(5):
        tf.record(ctx, f"g{i}", n=i)
    assert tf.facts_for(ctx, "keep") == {"n": 2}


def test_shallow_copied_contexts_share_the_ring_without_corrupting_it():
    """`subagent.py` and `dream.py` do `copy.copy(context)`. Sharing is
    acceptable; silently losing a live turn's facts is not — so only requests
    that actually record take a slot."""
    ctx = _ctx()
    tf.record(ctx, "user-turn", router_label="hard")
    child = copy.copy(ctx)
    assert tf.facts_for(child, "user-turn") == {"router_label": "hard"}
    child_only = copy.copy(ctx)
    tf.record(child_only, "sub-1", router_label="easy")
    assert tf.facts_for(ctx, "user-turn") == {"router_label": "hard"}


def test_never_raises_on_a_hostile_context():
    class Hostile:
        def __setattr__(self, k, v):
            raise RuntimeError("boom")

        def __getattr__(self, k):
            raise RuntimeError("boom")

    tf.record(Hostile(), "r", router_label="x")
    assert tf.facts_for(Hostile(), "r") == {}


# ── §4I Phase 1 wiring ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_verdict_reaches_the_trajectory(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector

    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="t")
    ctx = make_context(memory_dir=tmp_path / "memory",
                       trajectory_collector=collector)
    ctx.skill_memory.last_playbook_triggers = []
    dispatcher = MagicMock()
    dispatcher.route = MagicMock(return_value=SimpleNamespace(
        label="hard", confidence=0.1, allowed_pools=["full"],
        escalated=True, reason="confidence 0.10 below threshold 0.30"))
    ctx.complexity_dispatcher = dispatcher
    agent = make_agent(ctx)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "done", "tool_calls": []}}]}])

    body = {"messages": [{"role": "user",
                          "content": "please investigate the failing build"}],
            "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    extra = list(collector.iter_trajectories())[0].extra
    assert extra.get("router_label") == "hard"
    assert extra.get("router_confidence") == 0.1
    assert extra.get("router_escalated") is True


@pytest.mark.asyncio
async def test_no_dispatcher_leaves_no_router_keys(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector

    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="t")
    ctx = make_context(memory_dir=tmp_path / "memory",
                       trajectory_collector=collector,
                       complexity_dispatcher=None)
    ctx.skill_memory.last_playbook_triggers = []
    agent = make_agent(ctx)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "done", "tool_calls": []}}]}])
    body = {"messages": [{"role": "user",
                          "content": "please investigate the failing build"}],
            "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    extra = list(collector.iter_trajectories())[0].extra
    assert "router_label" not in extra
