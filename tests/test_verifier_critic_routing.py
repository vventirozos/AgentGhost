"""Verifier routes its LLM call to the critic pool when one is configured.

`Verifier._call_llm` prefers a dedicated critic pool (`use_critic=True`)
over the worker route, so the verdict runs on the spare-box judge model
and off both the worker pool and the foreground inference slot. When no
critic pool exists it must behave exactly as before (worker route →
direct call), and a broken/empty critic pool must fall through, not crash.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.core.verifier import Verifier, VerifyVerdict


_VERDICT_JSON = (
    '{"verdict": "CONFIRMED", "confidence": 0.9, '
    '"reasoning": "ok", "issues": []}'
)


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_verifier_uses_critic_pool_when_present():
    llm = MagicMock()
    llm.critic_clients = [{"url": "http://mini:8001", "model": "qwen3:9b"}]
    llm.chat_completion = AsyncMock(return_value=_completion(_VERDICT_JSON))
    # Worker route is present but must NOT be consulted once the critic
    # pool answered.
    llm.route = AsyncMock(return_value=None)

    verifier = Verifier(llm_client=llm)
    result = await verifier.verify_claim("claim", "evidence")

    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED
    # critic pool consulted with use_critic=True...
    assert llm.chat_completion.await_count == 1
    args, kwargs = llm.chat_completion.await_args
    assert kwargs.get("use_critic") is True
    # ...as a FOREGROUND call, never is_background. The verifier is invoked
    # from inside a live user request; an is_background critic call would
    # park on _wait_for_foreground_clear waiting for THAT request to end —
    # a self-deadlock that hangs the turn. A bounded timeout must be set so
    # an unreachable node falls through instead of blocking.
    assert kwargs.get("is_background") is not True
    assert kwargs.get("timeout") is not None
    # ...with thinking DISABLED and a small token cap — the verdict is a
    # tiny JSON object; a <think> prelude is pure latency on an off-host
    # judge model. Both switches (portable + chat-template) must be set.
    critic_payload = args[0] if args else kwargs.get("payload")
    assert critic_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert critic_payload["messages"][-1]["content"].rstrip().endswith("/no_think")
    assert critic_payload["max_tokens"] <= 512
    # ...and the worker route was never reached.
    llm.route.assert_not_awaited()


@pytest.mark.asyncio
async def test_verifier_falls_back_to_worker_when_no_critic_pool():
    llm = MagicMock()
    llm.critic_clients = []  # no critic pool
    llm.route = AsyncMock(return_value=_VERDICT_JSON)
    llm.chat_completion = AsyncMock(return_value=_completion(_VERDICT_JSON))

    verifier = Verifier(llm_client=llm)
    result = await verifier.verify_claim("claim", "evidence")

    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED
    # Legacy path: worker route consulted, critic pool branch skipped.
    llm.route.assert_awaited()


@pytest.mark.asyncio
async def test_verifier_critic_pool_failure_falls_through():
    """A throwing critic pool must not crash the verdict — it falls
    through to the worker route / direct call."""
    llm = MagicMock()
    llm.critic_clients = [{"url": "http://mini:8001", "model": "qwen3:9b"}]
    llm.chat_completion = AsyncMock(side_effect=Exception("critic offline"))
    llm.route = AsyncMock(return_value=_VERDICT_JSON)

    verifier = Verifier(llm_client=llm)
    result = await verifier.verify_claim("claim", "evidence")

    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED
    llm.route.assert_awaited()
