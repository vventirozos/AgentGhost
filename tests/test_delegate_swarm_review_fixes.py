"""Regression tests for the 2026-07-22 delegation/swarm review fixes.

Three findings from the LLM-stack review, all corroborated and live-log
confirmed (delegate_to_swarm was burning a strike on every call):

1. CRIT — ``delegate(wait=True)`` deadlocks against the parent request.
   A sub-agent's LLM calls are forced ``is_background=True`` and park on
   ``_wait_for_foreground_clear`` while a user request is active; ``wait=True``
   blocks the parent inside ``reg.wait`` while it still holds
   ``foreground_requests`` up — circular. Fix: downgrade to fire-and-forget
   when ``foreground_requests > 0`` (an interactive turn is in flight).

2. HIGH — ``delegate_to_swarm`` advertised in the schema while unconfigured.
   Fix: drop it from ``get_active_tool_definitions`` when no swarm clients
   exist (mirrors the ``image_generation`` gating).

3. HIGH — the ``"delegate" -> "delegate_to_swarm"`` alias hijacked name
   variants of the real ``delegate`` tool. Fix: removed the alias.
"""
import asyncio

import pytest
from unittest.mock import patch

from tests.helpers import make_context

from ghost_agent.tools import delegate as delegate_mod
from ghost_agent.tools.delegate import tool_delegate
from ghost_agent.tools.registry import get_active_tool_definitions
from ghost_agent.core.jobs import JobRegistry
from ghost_agent.core.agent import GhostAgent


# ── Finding 1: delegate(wait=True) deadlock guard ──────────────────────────

def _ctx_with_registry(foreground_requests: int):
    ctx = make_context()
    ctx.llm_client.foreground_requests = foreground_requests
    ctx.job_registry = JobRegistry()
    return ctx


@pytest.mark.asyncio
async def test_delegate_wait_downgraded_when_foreground_active():
    """wait=True during an interactive turn (foreground_requests>0) must NOT
    block on reg.wait — it downgrades to fire-and-forget and returns fast."""
    ctx = _ctx_with_registry(foreground_requests=1)

    # No real sub-agent runs — the coro is never awaited by the test.
    async def _noop(*a, **k):
        await asyncio.sleep(3600)  # would hang if awaited synchronously

    with patch.object(delegate_mod, "run_subagent", _noop), \
            patch.object(delegate_mod, "spawn_bg",
                         lambda coro, **k: asyncio.ensure_future(coro)):
        # If the downgrade fails, this awaits reg.wait(timeout=630) → the
        # wait_for ceiling below trips and the test fails loudly.
        result = await asyncio.wait_for(
            tool_delegate(task="analyze this", wait=True, context=ctx),
            timeout=2.0,
        )

    assert "background" in result.lower()
    assert "wait=true is unavailable" in result.lower()
    # A job WAS spawned (fire-and-forget), just not awaited.
    assert len(ctx.job_registry.list()) == 1


@pytest.mark.asyncio
async def test_delegate_wait_honored_when_no_foreground():
    """wait=True in an idle/background context (foreground_requests==0) is
    safe — the tool blocks on the job and returns its result."""
    ctx = _ctx_with_registry(foreground_requests=0)

    async def _quick(context, *, job_id, task, **k):
        return f"DONE: {task}"

    def _spawn(coro, **k):
        return asyncio.ensure_future(coro)

    with patch.object(delegate_mod, "run_subagent", _quick), \
            patch.object(delegate_mod, "spawn_bg", _spawn):
        result = await asyncio.wait_for(
            tool_delegate(task="quick job", wait=True, context=ctx),
            timeout=5.0,
        )

    assert "DONE: quick job" in result
    assert "wait=true is unavailable" not in result.lower()


# ── Finding 2: delegate_to_swarm schema gating ─────────────────────────────

def _tool_names(defs):
    return {d.get("function", {}).get("name") for d in defs}


def test_delegate_to_swarm_hidden_when_unconfigured():
    ctx = make_context()
    ctx.llm_client.swarm_clients = []
    names = _tool_names(get_active_tool_definitions(ctx))
    assert "delegate_to_swarm" not in names


def test_delegate_to_swarm_advertised_when_configured():
    ctx = make_context()
    ctx.llm_client.swarm_clients = [{"client": object(), "model": "m"}]
    names = _tool_names(get_active_tool_definitions(ctx))
    assert "delegate_to_swarm" in names


# ── Finding 3: no bare "delegate" alias hijack ─────────────────────────────

def test_delegate_variant_not_hijacked_to_swarm():
    """A case/paren variant of the real delegate tool must canonicalise to
    'delegate', never to 'delegate_to_swarm'."""
    available = ["delegate", "delegate_to_swarm", "file_system", "execute"]
    for variant in ("Delegate", "DELEGATE", "delegate()"):
        assert GhostAgent._canonicalise_tool_name(variant, available) == "delegate"


def test_swarm_name_variants_still_reach_swarm():
    """The swarm-specific aliases still resolve — only the bare 'delegate'
    hijack was removed."""
    available = ["delegate", "delegate_to_swarm", "file_system"]
    for variant in ("delegate_to_swarm", "delegate-to-swarm", "DelegateToSwarm"):
        assert (GhostAgent._canonicalise_tool_name(variant, available)
                == "delegate_to_swarm")
