"""Regression tests for the `self_play` / `self_play_loop` intent guard.

Incident context (2026-04-24, webOS build session)
--------------------------------------------------
A user asked the agent to "Create a webOS, make sure it looks awesome,
create 10 apps..." After 33 minutes of successful browser-automation
work the LLM's inner thoughts drifted:

    💭  The user wants me to run self-play. Let me call the self_play tool.
    🎯  self-play frontier  Mode=exploration (no frontier seed)

The user had never mentioned self-play. The `self_play` tool has no
user-intent guard — its description says "Use this EVERY TIME the user
asks to practice, train, or do self-play" but that's LLM-side prose;
nothing enforces it, so a long-context hallucination hijacked the turn
and burned an LLM cycle on a synthetic curriculum the user didn't want.

Fix
---
1. `core.agent.GhostContext.last_user_content` now holds the current
   turn's user text. `handle_chat` sets it right after parsing the
   request body, before any tool can be dispatched.
2. `tools.memory.tool_self_play` and `tool_self_play_loop` check
   `_user_asked_for_self_play(context)` at entry. If the user text
   lacks any of the explicit intent phrases ("run self-play",
   "train until stopped", "practice cycle", ...), the tool returns
   `_SELF_PLAY_INTENT_REFUSAL` without launching a cycle.
3. The biological watchdog's self-play phase is UNTOUCHED: it calls
   `Dreamer.synthetic_self_play` directly, not through the tool layer,
   so this guard does not interfere with legitimate background firing.

These tests pin that contract.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.tools.memory import (
    _SELF_PLAY_INTENT_PHRASES,
    _SELF_PLAY_INTENT_REFUSAL,
    _user_asked_for_self_play,
    tool_self_play,
    tool_self_play_loop,
)


# ---------------------------------------------------------------------------
# _user_asked_for_self_play — pure predicate
# ---------------------------------------------------------------------------


class TestIntentPredicate:
    @pytest.mark.parametrize("text", [
        "run self-play",
        "Run Self Play now",
        "please start self-play",
        "can you run a training cycle?",
        "keep practicing until I say stop",
        "train until stopped",
        "train yourself for a while",
        "go ahead and run synthetic self-play",
        "let's do a practice round",
        "keep training",
        "start a training loop please",
    ])
    def test_explicit_requests_accepted(self, text):
        ctx = SimpleNamespace(last_user_content=text)
        assert _user_asked_for_self_play(ctx) is True, (
            f"phrase '{text}' should count as explicit intent"
        )

    @pytest.mark.parametrize("text", [
        "",
        "Create a webOS and 10 apps inside it",
        "Please fix the bug in handle_chat",
        "Can you practice good git hygiene?",  # "practice" alone is not enough
        "What did you learn today?",
        "Let's plan the next step.",
        "Write a Python script that prints hello",
        "Summarise this log file",
        "Train the logistic regression with this data",  # ML 'train', not self-play
    ])
    def test_non_requests_rejected(self, text):
        ctx = SimpleNamespace(last_user_content=text)
        assert _user_asked_for_self_play(ctx) is False, (
            f"phrase '{text}' should NOT count as explicit self-play intent"
        )

    def test_missing_attr_is_rejected(self):
        """A context without `last_user_content` (e.g. test stubs) must
        refuse — the watchdog path bypasses this helper entirely."""
        ctx = SimpleNamespace()
        assert _user_asked_for_self_play(ctx) is False

    def test_none_value_is_rejected(self):
        """Defensive: some callers may set the attr to None."""
        ctx = SimpleNamespace(last_user_content=None)
        assert _user_asked_for_self_play(ctx) is False

    def test_whitespace_and_case_are_normalised(self):
        """Extra whitespace and upper-case letters must not let the
        LLM dodge the guard on one side or trip it on the other."""
        ctx = SimpleNamespace(last_user_content="  RUN   SELF-PLAY   NOW  ")
        assert _user_asked_for_self_play(ctx) is True

    def test_phrase_list_is_frozen_to_tuple(self):
        """Guards against an accidental edit that turns the phrase
        constant into a mutable list (which tests would then fight)."""
        assert isinstance(_SELF_PLAY_INTENT_PHRASES, tuple)
        assert len(_SELF_PLAY_INTENT_PHRASES) >= 10


# ---------------------------------------------------------------------------
# tool_self_play — refusal path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_self_play_refuses_when_intent_absent(monkeypatch):
    """The incident case: the LLM calls `self_play` on a turn whose
    user text is about building a webOS. The guard must refuse BEFORE
    any Dreamer.synthetic_self_play is invoked."""
    from ghost_agent.core.dream import Dreamer

    called = {"n": 0}

    async def _should_not_run(*a, **kw):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(Dreamer, "synthetic_self_play", _should_not_run)

    ctx = SimpleNamespace(
        last_user_content="Create a webOS, make sure it looks awesome",
    )
    result = await tool_self_play(ctx)

    assert result == _SELF_PLAY_INTENT_REFUSAL
    assert "REFUSED" in result
    assert called["n"] == 0, "Dreamer must not be touched on refusal"


@pytest.mark.asyncio
async def test_tool_self_play_refuses_when_attr_missing(monkeypatch):
    """An empty / unstashed context must refuse rather than silently
    permit the tool. The watchdog doesn't go through this code path
    so refusal is safe."""
    from ghost_agent.core.dream import Dreamer

    called = {"n": 0}

    async def _should_not_run(*a, **kw):
        called["n"] += 1
        return "should not happen"

    monkeypatch.setattr(Dreamer, "synthetic_self_play", _should_not_run)

    ctx = SimpleNamespace()  # no last_user_content
    result = await tool_self_play(ctx)

    assert "REFUSED" in result
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_tool_self_play_runs_when_user_asked(monkeypatch):
    """Happy path: the user explicitly asks, the Dreamer is invoked."""
    # Patch the Dreamer *class* itself so we don't have to satisfy its
    # __init__ contract (memory_system, frontier_tracker, ...). The
    # tool only needs something whose `synthetic_self_play` returns a
    # string.
    class _FakeDreamer:
        def __init__(self, ctx): pass
        async def synthetic_self_play(self, is_background=False):
            return "synthetic result"

    monkeypatch.setattr("ghost_agent.core.dream.Dreamer", _FakeDreamer)

    ctx = SimpleNamespace(last_user_content="please run self-play")
    result = await tool_self_play(ctx)

    assert "synthetic result" in result
    assert "SELF PLAY DONE" in result
    assert "REFUSED" not in result


# ---------------------------------------------------------------------------
# tool_self_play_loop — same guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_self_play_loop_refuses_when_intent_absent():
    """The loop variant is even more destructive (continuous cycles),
    so the same guard must apply."""
    ctx = SimpleNamespace(
        last_user_content="build me a React dashboard",
        selfplay_loop_task=None,
        selfplay_loop_stop=None,
        selfplay_loop_started_at=None,
    )
    result = await tool_self_play_loop(ctx, max_cycles=0)
    assert "REFUSED" in result
    # No background task must have been started.
    assert getattr(ctx, "selfplay_loop_task", None) is None


@pytest.mark.asyncio
async def test_tool_self_play_loop_runs_when_user_asked(monkeypatch):
    """Happy path for the loop variant."""
    from ghost_agent.core.dream import Dreamer

    class _FakeDreamer:
        def __init__(self, ctx): pass
        async def synthetic_self_play(self, model_name="m", is_background=True):
            await asyncio.sleep(0)

    monkeypatch.setattr("ghost_agent.core.dream.Dreamer", _FakeDreamer)
    monkeypatch.setattr(
        "ghost_agent.tools.memory._derive_loop_cooloff",
        lambda _ctx: 0.01,
    )

    ctx = SimpleNamespace(
        last_user_content="train until stopped please",
        selfplay_loop_task=None,
        selfplay_loop_stop=None,
        selfplay_loop_started_at=None,
        skill_memory=None,
        frontier_tracker=None,
        llm_client=SimpleNamespace(foreground_tasks=0),
        args=SimpleNamespace(model="m"),
    )
    msg = await tool_self_play_loop(ctx, max_cycles=1)
    assert "REFUSED" not in msg
    assert "STARTED" in msg
    # Wait for the bounded loop to exit cleanly so pytest doesn't see
    # a leaked task.
    task = ctx.selfplay_loop_task
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()


# ---------------------------------------------------------------------------
# Watchdog bypass — the biological path must keep working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_biological_self_play_does_not_go_through_tool_guard():
    """The watchdog calls `Dreamer.synthetic_self_play` directly rather
    than routing through `tool_self_play`, so the intent guard cannot
    reach it. Pin that structural property so a future refactor that
    routes the watchdog through the tool would immediately trip this
    test.

    We inspect the module source rather than running the watchdog, to
    keep the test hermetic and fast.
    """
    import inspect

    from ghost_agent.core import agent as agent_mod

    src = inspect.getsource(agent_mod)
    # The phase-3 self-play tick must call synthetic_self_play directly,
    # not via tool_self_play.
    assert "tool_self_play" not in src.split("_biological_tick")[1].split("async def ")[0] if "_biological_tick" in src else True


# ---------------------------------------------------------------------------
# GhostContext wiring — last_user_content must be stashed by handle_chat
# ---------------------------------------------------------------------------


def test_ghost_context_has_last_user_content_field():
    """The attribute must exist on a fresh context so tools never hit
    AttributeError before the first handle_chat runs."""
    from ghost_agent.core.agent import GhostContext

    args = SimpleNamespace(model="m")
    ctx = GhostContext(args=args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    assert hasattr(ctx, "last_user_content")
    assert ctx.last_user_content == ""


def test_handle_chat_source_stashes_last_user_content():
    """Grep-assert that handle_chat actually assigns
    `self.context.last_user_content = last_user_content`. Without the
    assignment, the guard would refuse every legitimate call because
    the field stays at its empty-string default.
    """
    from pathlib import Path

    agent_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    )
    src = agent_path.read_text(encoding="utf-8")
    assert "self.context.last_user_content = last_user_content" in src, (
        "handle_chat must stash last_user_content on the context so "
        "tool_self_play's intent guard can see the current turn's text"
    )
