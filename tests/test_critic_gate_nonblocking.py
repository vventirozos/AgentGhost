"""Non-blocking post-response verifier gate (the `_gated` wrapper).

With a dedicated `--critic-nodes` pool the (slower) verdict must not
serialise behind the user's response: the gate spawns it as a background
task and ships the answer, while a high-confidence REFUTED that lands
late still scrubs the turn's lessons. Without a critic pool the gate is
byte-for-byte the legacy inline `await`. `GHOST_CRITIC_GATE_TIMEOUT`
tunes the inline wait either way.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.smart_memory = 0.0
    ctx.llm_client = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.verifier = None
    return GhostAgent(ctx)


def _verdict(v, conf=0.95, issues=None, reasoning="reason"):
    return VerifyResult(verdict=v, confidence=conf, reasoning=reasoning, issues=issues or [])


def _critic_llm():
    llm = MagicMock()
    llm.critic_clients = [{"url": "http://mini:8001", "model": "qwen3:9b"}]
    return llm


# --------------------------------------------------------------------------
# _critic_gate_timeout
# --------------------------------------------------------------------------

def test_gate_timeout_infinite_without_critic_pool(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = None
    assert agent._critic_gate_timeout() == float("inf")


def test_gate_timeout_zero_with_critic_pool(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = MagicMock()
    agent.context.verifier.llm_client = _critic_llm()
    assert agent._critic_gate_timeout() == 0.0


def test_gate_timeout_env_override(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "3.5")
    # Even with no critic pool, an explicit budget wins.
    agent.context.verifier = None
    assert agent._critic_gate_timeout() == 3.5


def test_gate_timeout_bad_env_ignored(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "not-a-number")
    agent.context.verifier = None
    assert agent._critic_gate_timeout() == float("inf")


# --------------------------------------------------------------------------
# _compute_verifier_verdict_gated
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inline_await_without_critic_pool(agent, monkeypatch):
    """No critic pool → block inline and return the real verdict."""
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = None

    last_tool = {"name": "execute", "content": "out"}
    inner = AsyncMock(return_value=(_verdict(VerifyVerdict.CONFIRMED), last_tool))
    agent._compute_verifier_verdict = inner

    v_result, lt = await agent._compute_verifier_verdict_gated(
        tools_run_this_turn=[last_tool],
        messages=[],
        final_ai_content="answer",
        last_user_content="q",
        lc="q",
        trajectory_id="traj-1",
    )

    assert v_result is not None
    assert v_result.verdict == VerifyVerdict.CONFIRMED
    inner.assert_awaited_once()


@pytest.mark.asyncio
async def test_pure_async_releases_without_waiting(agent, monkeypatch):
    """Critic pool + default (gate 0) → return (None, last_tool) at once,
    never blocking on the verdict."""
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = MagicMock()
    agent.context.verifier.llm_client = _critic_llm()

    last_tool = {"name": "execute", "content": "out"}
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_verdict(**kwargs):
        started.set()
        await release.wait()  # would hang forever if awaited inline
        return _verdict(VerifyVerdict.CONFIRMED), last_tool

    agent._compute_verifier_verdict = _slow_verdict

    with patch(
        "ghost_agent.core.agent._find_substantive_tool_for_verifier",
        return_value=last_tool,
    ):
        v_result, lt = await asyncio.wait_for(
            agent._compute_verifier_verdict_gated(
                tools_run_this_turn=[last_tool],
                messages=[],
                final_ai_content="answer",
                last_user_content="q",
                lc="q",
                trajectory_id="traj-1",
            ),
            timeout=1.0,
        )

    # Released immediately, unverified, even though the verdict is still
    # blocked on `release`.
    assert v_result is None
    assert lt is last_tool
    release.set()
    await asyncio.sleep(0)  # let the background task drain


