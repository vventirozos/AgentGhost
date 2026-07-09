"""Regression test for agent.py: metacog must record per-TOOL outcomes.

Previously `record_outcome` ran once AFTER the enumerate(results) loop,
keyed on the last tool's name and the turn-wide failure flag — so a turn
that ran [ok_tool, fail_tool] recorded only one (mis-attributed) outcome.
Now each result is recorded with its own tool name and own success.

Also a lightweight guard that the Context-Shield summarizer uses its own
`shield_payload` variable rather than clobbering the loop-level `payload`
(which the Perfect-It path reuses).
"""

import inspect
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext


class _StubMetacog:
    """Minimal metacog bundle: enabled, captures record_outcome, and
    no-ops every other path the agent probes (arbiter must return None so
    it doesn't pause/derail the turn)."""

    def __init__(self):
        self.enabled = True
        self.calls = []

    def record_outcome(self, name, success=None, **kw):
        self.calls.append((name, success))

    async def arbitrate_tool_calls(self, **kw):
        return None

    def __getattr__(self, _name):
        # Any other probed method → benign no-op returning None.
        def _noop(*a, **k):
            return None
        return _noop


def _ctx():
    args = MagicMock()
    args.model = "test-model"
    args.use_planning = False
    args.temperature = 0.0
    args.max_context = 8000
    args.smart_memory = 0.0
    args.enable_metacog = True
    args.metacog_disable_arbiter = True
    args.metacog_disable_logprobs = True
    ctx = GhostContext(args=args, sandbox_dir=None, memory_dir=None, tor_proxy=None)
    ctx.llm_client = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = "Empty"
    return ctx


@pytest.mark.asyncio
async def test_metacog_records_each_tool_with_its_own_outcome():
    ctx = _ctx()
    stub = _StubMetacog()
    ctx.metacog = stub
    agent = GhostAgent(ctx)

    agent.available_tools = {
        "ok_tool": AsyncMock(return_value="SUCCESS: did the thing"),
        "fail_tool": AsyncMock(return_value="Error: boom — it failed"),
    }

    turn1 = {"choices": [{"message": {
        "role": "assistant", "content": "calling both",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "ok_tool", "arguments": "{}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "fail_tool", "arguments": "{}"}},
        ],
    }}]}
    finish = {"choices": [{"message": {
        "role": "assistant", "content": "All done.", "tool_calls": []}}]}

    state = {"n": 0}

    def _cc(*a, **k):
        state["n"] += 1
        return turn1 if state["n"] == 1 else finish

    ctx.llm_client.chat_completion = AsyncMock(side_effect=_cc)

    await agent.handle_chat({"messages": [{"role": "user", "content": "go"}]}, MagicMock())

    recorded = dict(stub.calls)
    assert recorded.get("ok_tool") is True, f"calls={stub.calls}"
    assert recorded.get("fail_tool") is False, f"calls={stub.calls}"


@pytest.mark.asyncio
async def test_prune_context_tracks_episode_archive_task():
    """The episodic-archive write scheduled during compaction must be
    tracked in ctx._pending_background_tasks (so it isn't GC'd mid-flight
    and the lifespan can drain it), not a bare fire-and-forget create_task."""
    import asyncio
    ctx = _ctx()
    ctx.memory_system = MagicMock()
    ctx.memory_system.add = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Condensed summary of prior turns."}}]
    })
    agent = GhostAgent(ctx)

    big = "x " * 2000  # ~1k tokens each
    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"user {i} {big}"})
        messages.append({"role": "assistant", "content": f"assistant {i} {big}"})

    await agent._prune_context(messages, max_tokens=500, model="test-model")

    bg = getattr(ctx, "_pending_background_tasks", None)
    assert bg is not None, "episode-archive task was not tracked (bare create_task)"
    # Drain whatever is still pending, then confirm the write landed exactly once.
    if bg:
        await asyncio.gather(*list(bg), return_exceptions=True)
    ctx.memory_system.add.assert_called_once()


def test_main_drains_background_tasks_on_shutdown():
    """The lifespan shutdown must drain _pending_background_tasks."""
    import inspect
    from ghost_agent import main
    src = inspect.getsource(main)
    assert "_pending_background_tasks" in src
    assert "Background-task drain" in src


def test_context_shield_does_not_clobber_payload_variable():
    """The shield summarizer must bind `shield_payload`, never reassign
    the loop-level `payload` (reused by the Perfect-It generation).

    The shield block moved into `_dispatch_and_process_tool_batch` with
    the #5 step-2 extraction (2026-07-09) — inspect both."""
    src = (inspect.getsource(GhostAgent.handle_chat)
           + inspect.getsource(GhostAgent._dispatch_and_process_tool_batch))
    assert "shield_payload = {" in src
    # The old clobbering assignment must be gone from the shield block.
    assert "Offloading" in src  # shield block still present
    # No bare `payload = {` inside the shield (chat_completion uses shield_payload).
    assert "chat_completion(shield_payload" in src
