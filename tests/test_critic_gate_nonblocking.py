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
    ctx.skill_memory = MagicMock()
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


@pytest.mark.asyncio
async def test_late_refuted_scrubs_lessons(agent, monkeypatch):
    """A REFUTED verdict that lands after the response retracts the turn's
    lessons via the done-callback."""
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = MagicMock()
    agent.context.verifier.llm_client = _critic_llm()
    agent.context.skill_memory = MagicMock()

    last_tool = {"name": "execute", "content": "out"}
    agent._compute_verifier_verdict = AsyncMock(
        return_value=(_verdict(VerifyVerdict.REFUTED, conf=0.9, issues=["wrong"]), last_tool)
    )

    with patch(
        "ghost_agent.core.agent._find_substantive_tool_for_verifier",
        return_value=last_tool,
    ):
        v_result, lt = await agent._compute_verifier_verdict_gated(
            tools_run_this_turn=[last_tool],
            messages=[],
            final_ai_content="answer",
            last_user_content="q",
            lc="q",
            trajectory_id="traj-1",
        )

    assert v_result is None  # released unverified
    # Give the background task + done-callback + to_thread time to run.
    await asyncio.sleep(0.2)
    agent.context.skill_memory.retract_lessons_from_trajectory.assert_called_once()
    args, kwargs = agent.context.skill_memory.retract_lessons_from_trajectory.call_args
    assert args[0] == "traj-1"


@pytest.mark.asyncio
async def test_late_confirmed_does_not_scrub(agent, monkeypatch):
    """A late CONFIRMED verdict must NOT retract anything."""
    monkeypatch.delenv("GHOST_CRITIC_GATE_TIMEOUT", raising=False)
    agent.context.verifier = MagicMock()
    agent.context.verifier.llm_client = _critic_llm()
    agent.context.skill_memory = MagicMock()

    last_tool = {"name": "execute", "content": "out"}
    agent._compute_verifier_verdict = AsyncMock(
        return_value=(_verdict(VerifyVerdict.CONFIRMED), last_tool)
    )

    with patch(
        "ghost_agent.core.agent._find_substantive_tool_for_verifier",
        return_value=last_tool,
    ):
        await agent._compute_verifier_verdict_gated(
            tools_run_this_turn=[last_tool],
            messages=[],
            final_ai_content="answer",
            last_user_content="q",
            lc="q",
            trajectory_id="traj-1",
        )

    await asyncio.sleep(0.2)
    agent.context.skill_memory.retract_lessons_from_trajectory.assert_not_called()
