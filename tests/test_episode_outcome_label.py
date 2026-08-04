"""Episode outcome labels must use the real signals, not a substring guess.

Measured on the live store before the fix: 250/259 episodes (96.5%) labelled
success, all NINE "failures" were false negatives, and the agent's own
total-failure sentinel was stored as SUCCESS three times.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import make_agent, make_context


def _agent_with_episodes():
    em = MagicMock()
    em.record_episode = MagicMock(return_value=None)
    ctx = make_context(episodic_memory=em)
    return make_agent(ctx), em


def _recorded(em):
    assert em.record_episode.called, "no episode recorded"
    kwargs = em.record_episode.call_args.kwargs
    if "outcome_success" in kwargs:
        return kwargs["outcome_success"]
    return kwargs.get("success")


@pytest.mark.asyncio
async def test_a_refuted_verdict_is_not_a_success():
    agent, em = _agent_with_episodes()
    await agent._record_episode_safe(
        "do the thing", [{"name": "execute", "content": "ok"}],
        "Here you go, all done!", verifier_verdict="failed")
    assert _recorded(em) is False


@pytest.mark.asyncio
async def test_the_hard_limit_sentinel_is_not_a_success():
    """The agent's own total-failure reply. Stored as SUCCESS three times
    live, because the word 'failures' sits past the 80-char cut."""
    agent, em = _agent_with_episodes()
    await agent._record_episode_safe(
        "do the thing", [{"name": "execute", "content": "boom"}],
        "I hit a hard limit after repeated failures and could not complete "
        "this task. Please rephrase or break it into smaller steps.",
        execution_failure_count=6)
    assert _recorded(em) is False


@pytest.mark.asyncio
async def test_explaining_an_error_is_not_a_failure():
    """All nine live 'failures' were this shape — a correct answer that
    happens to mention an error."""
    agent, em = _agent_with_episodes()
    await agent._record_episode_safe(
        "what does 0/0 do?", [{"name": "execute", "content": "ok"}],
        "In Python, 0/0 raises a ZeroDivisionError — here is why that failed.",
        verifier_verdict="passed")
    assert _recorded(em) is True


@pytest.mark.asyncio
async def test_a_clean_turn_with_no_verdict_is_a_success():
    agent, em = _agent_with_episodes()
    await agent._record_episode_safe(
        "read the file", [{"name": "file_system", "content": "contents"}],
        "The file contains three lines.")
    assert _recorded(em) is True
