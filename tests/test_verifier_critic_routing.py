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
async def test_verifier_uses_critic_pool_when_present(monkeypatch):
    # Pin the classic single-stage path: this test asserts ROUTING
    # invariants per _call_llm call, and the two-stage default would add
    # a second (fallback) call because the stub returns a verdict JSON —
    # no "suspects" key — to the stage-1 enumeration prompt too. The
    # two-stage pipeline itself is covered by test_verifier_two_stage.py.
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
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
    #
    # ⚠ THIS IS THE NON-PRODUCTION BRANCH. `bin/start-ghost-agent.sh:293`
    # exports GHOST_CRITIC_NO_THINK=0, so the live agent runs the `else`
    # arm and passes the payload through untouched. This test only reached
    # the no-think arm because pytest leaves the variable unset and the
    # default is "1". See the sibling test below, which covers what
    # production actually does — it had NO coverage until R5 (replacing the
    # live branch with `raise AssertionError` passed 188 tests).
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


# ---------------------------------------------------------------------------
# LLM review 2026-08-18: the client's SILENT pool→main fallback
# ---------------------------------------------------------------------------

def test_a_critic_call_served_by_MAIN_is_reported_as_main(monkeypatch):
    """⚠ The whole suite above stubs `chat_completion`, so the client's
    internal fallback is invisible to it — and that fallback is the live
    case: `--worker-nodes` and `--critic-nodes` are the SAME box, so one
    outage (or our own NodeSaturated) sends the verdict to the 35B.

    The client returned a dict identical in shape to a critic-served one, so
    this stamped `route="critic"` on a MAIN-model verdict and §4BR's
    degradation guard — abort the self-consistency vote when the route is
    main/failed — could never fire. All n samples then serialised on the
    single foreground slot, the exact condition the guard exists to stop.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.llm import _stamp_leg
    from ghost_agent.core.verifier import Verifier

    for served, fell_back, want in (("critic", "", "critic"),
                                    ("main", "critic", "main")):
        v = Verifier.__new__(Verifier)
        v.llm_client = MagicMock()
        v.llm_client.critic_clients = [{"url": "http://node", "model": "m"}]
        v.llm_client.chat_completion = AsyncMock(return_value=_stamp_leg(
            {"choices": [{"message": {"content": '{"verdict": "PASS"}'}}]},
            served, fell_back))
        v._parse_json = lambda t: {"verdict": "PASS"}
        # The critic leg lives inside `_call_llm`; drive it there.
        v.context = MagicMock()
        route_out = {}
        got = asyncio.run(v._call_llm("prompt", route_out=route_out))
        assert isinstance(got, dict)
        assert route_out.get("route") == want, (
            f"a verdict served by {served!r} was reported as "
            f"{route_out.get('route')!r} — the §4BR guard reads this field")
        if fell_back:
            assert route_out.get("fell_back_from") == fell_back


@pytest.mark.asyncio
async def test_the_LIVE_critic_config_passes_the_payload_through_untouched(
        monkeypatch):
    """The branch the shipping agent runs: GHOST_CRITIC_NO_THINK=0.

    Suppressing thinking on this judge was benched and REJECTED — it
    produced false REFUTEs — so the live config deliberately leaves the
    payload alone. That decision was carried entirely by an env var read at
    IMPORT time, so no test could set it: the branch production actually
    executes had zero coverage, and replacing it with `raise AssertionError`
    passed 188 tests across ten verifier files (R5 lens B). Meanwhile the
    test above pinned the arm that is switched OFF live.
    """
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.setenv("GHOST_CRITIC_NO_THINK", "0")
    llm = MagicMock()
    llm.critic_clients = [{"url": "http://mini:8001", "model": "qwen3:9b"}]
    llm.chat_completion = AsyncMock(return_value=_completion(_VERDICT_JSON))
    llm.route = AsyncMock(return_value=None)

    verifier = Verifier(llm_client=llm)
    result = await verifier.verify_claim("claim", "evidence")
    assert result is not None

    args, kwargs = llm.chat_completion.await_args
    payload = args[0] if args else kwargs.get("payload")
    assert "chat_template_kwargs" not in payload, (
        "thinking was suppressed on the critic despite "
        "GHOST_CRITIC_NO_THINK=0 — this judge produced false REFUTEs "
        "without a reasoning prelude, which is why the switch exists")
    assert not payload["messages"][-1]["content"].rstrip().endswith(
        "/no_think"), "the soft switch leaked into the live payload"
    assert kwargs.get("use_critic") is True
    assert kwargs.get("timeout") is not None
    assert kwargs.get("slot_wait") is not None, (
        "the critic leg inherits the 90s operator ceiling; worker and critic "
        "are the same physical box, so the pair queues on it twice per "
        "verdict")
