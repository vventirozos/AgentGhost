"""Canonical test builders (IMPROVEMENTS.md #26).

Pins the shared `make_context` / `make_agent` helpers so the ~90 files that
hand-roll a GhostContext can migrate to one seam — the change that de-risks the
agent.py extraction (#5) by decoupling the suite from the monolith's internals.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import make_context, make_agent, FakeBgTasks


def test_make_context_has_common_fields():
    ctx = make_context()
    assert ctx.args.model == "Qwen-Test"
    assert ctx.args.max_context == 8000
    assert ctx.llm_client.foreground_requests == 0
    assert ctx.memory_system.search_items(("q",)) == []
    assert ctx.profile_memory.get_context_string() == ""
    # New-flag defaults present so agent __init__ reads them cleanly.
    assert ctx.args.bio_time_scale == 1.0
    assert ctx.args.bio_deterministic is False


def test_overrides_replace_fields():
    ctx = make_context(memory_system=None, sandbox_dir="/custom")
    assert ctx.memory_system is None
    assert ctx.sandbox_dir == "/custom"


def test_make_agent_builds_real_agent():
    ag = make_agent()
    from ghost_agent.core.agent import GhostAgent
    assert isinstance(ag, GhostAgent)
    # Init-time wiring is exercised: turns serialized (#22), flags read (#4).
    assert ag.agent_semaphore._value == 1
    assert ag._bio_time_scale == 1.0
    assert ag._bio_deterministic is False


def test_make_agent_accepts_prebuilt_context():
    ctx = make_context()
    ctx.args.max_context = 12345
    ag = make_agent(ctx)
    assert ag.context.args.max_context == 12345


@pytest.mark.asyncio
async def test_built_agent_can_handle_a_trivial_turn():
    """Smoke: an agent from the builder runs a minimal chat without wiring
    errors — proving the canonical context is complete enough for handle_chat."""
    ag = make_agent()
    body = {"messages": [{"role": "user", "content": "hello there"}], "model": "Qwen-Test"}
    result, _, _ = await ag.handle_chat(body, background_tasks=FakeBgTasks())
    assert isinstance(result, str)


def test_fake_bg_tasks_records_without_running():
    bg = FakeBgTasks()
    marker = []
    bg.add_task(lambda: marker.append(1), )
    assert len(bg.tasks) == 1
    assert marker == []  # not executed
