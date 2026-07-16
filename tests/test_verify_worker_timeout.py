"""VERIFY-specific worker timeout (2026-07-16).

Regression target: the verifier's worker-route call rode `route()`'s
default `_ROUTE_TIMEOUT_S` (12s — sized for sub-second routing chores
like query expansion). A verdict takes 7–11s UNCONTENDED on the live
worker, so any node contention (the finalize burst fires verify +
hydration-judge together) pushed it past the ceiling:
`Nova: ReadTimeout` → no gate verdict → a hallucinated answer shipped
unchecked (req 738c/35, the "Everest pizza" turn), caught only by the
late async verdict a full turn later.

The fix, in two halves:
- `LLMClient.route()` accepts a per-call `timeout` override;
- `Verifier._call_llm` passes `_VERIFY_WORKER_TIMEOUT_S` (default 45s,
  `GHOST_VERIFY_WORKER_TIMEOUT` env override) instead of inheriting the
  routing ceiling.
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.llm import LLMClient, _ROUTE_TIMEOUT_S
from ghost_agent.core.verifier import Verifier, _VERIFY_WORKER_TIMEOUT_S


def _stub_client(captured):
    """An LLMClient whose chat_completion just records its kwargs."""
    client = LLMClient.__new__(LLMClient)
    client.worker_clients = [{"model": "stub-worker"}]

    async def fake_chat_completion(payload, **kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    client.chat_completion = fake_chat_completion
    return client


class TestRouteTimeoutOverride:
    async def test_default_is_route_ceiling(self):
        captured = {}
        client = _stub_client(captured)
        await client.route("EXPAND_QUERY", {"messages": []})
        assert captured["timeout"] == _ROUTE_TIMEOUT_S

    async def test_explicit_timeout_wins(self):
        captured = {}
        client = _stub_client(captured)
        await client.route("VERIFY", {"messages": []}, timeout=45.0)
        assert captured["timeout"] == 45.0

    async def test_zero_timeout_is_respected_not_replaced(self):
        """0 is a valid explicit value — the `is not None` guard must not
        silently swap it for the default."""
        captured = {}
        client = _stub_client(captured)
        await client.route("X", {"messages": []}, timeout=0.0)
        assert captured["timeout"] == 0.0


class TestVerifierUsesVerifyBudget:
    async def test_default_budget_generous_vs_route_ceiling(self):
        """The whole point: the verify budget must comfortably exceed the
        routing ceiling that was killing contended verdicts."""
        assert _VERIFY_WORKER_TIMEOUT_S >= 30.0
        assert _VERIFY_WORKER_TIMEOUT_S > _ROUTE_TIMEOUT_S

    async def test_call_llm_passes_verify_timeout_to_route(self):
        captured = {}

        class StubLLM:
            async def route(self, task, payload, **kwargs):
                captured["task"] = task
                captured.update(kwargs)
                return json.dumps({
                    "verdict": "CONFIRMED", "confidence": 0.9,
                    "reasoning": "fine", "issues": [],
                })

        verifier = Verifier(llm_client=StubLLM())
        result = await verifier.verify_claim(
            claim="the sky is blue", evidence="sky: blue", context="")
        assert captured["task"] == "VERIFY"
        assert captured["timeout"] == _VERIFY_WORKER_TIMEOUT_S
        # The verdict must still parse through the widened path.
        assert result is not None and result.confidence == 0.9

    async def test_critic_pool_budget_untouched(self):
        """With a critic pool configured, the (already-generous) critic
        budget path is used — route() isn't even consulted."""
        captured = {}

        class StubLLM:
            critic_clients = [{"model": "stub-critic"}]

            async def chat_completion(self, payload, **kwargs):
                captured.update(kwargs)
                return {"choices": [{"message": {"content": json.dumps({
                    "verdict": "CONFIRMED", "confidence": 0.8,
                    "reasoning": "", "issues": [],
                })}}]}

            async def route(self, *a, **k):  # pragma: no cover
                raise AssertionError("critic path must not fall to route()")

        verifier = Verifier(llm_client=StubLLM())
        result = await verifier.verify_claim("c", "e")
        assert result is not None
        assert captured.get("use_critic") is True
        # Critic budget is its own constant (120s default), not the
        # worker verify budget.
        assert captured["timeout"] != _ROUTE_TIMEOUT_S
