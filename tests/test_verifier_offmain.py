"""Verifier last-resort fallback: bounded + background-aware (2026-07-22).

Regression target: `Verifier._call_llm`'s final fallback — the direct
call on the MAIN model, reached when the critic pool and the worker
route are both absent or unusable — was a FOREGROUND-marked
`chat_completion(payload)` with NO timeout. It therefore:

- rode the shared httpx client's 1200s default, so an exhausted worker
  path landed an unbounded thinking-enabled generation on the single
  main inference slot, in direct contention with a live user stream;
- always inflated `foreground_tasks`, even when the verify was invoked
  from a BACKGROUND flow (dream/self-play verify shares
  `context.verifier`), making other background work misread a live
  user.

The fix (`_bounded_fallback_kwargs`):

- always pass `timeout=_VERIFY_FALLBACK_TIMEOUT_S` (default 90s,
  `GHOST_VERIFY_FALLBACK_TIMEOUT` override) when the client accepts it;
- pass `is_background=True` ONLY when `llm_client.foreground_requests`
  says no user request is live — the verifier runs from INSIDE a user
  turn (in-loop auto-repair verdict), so marking a user-path verify
  background would park it on `_wait_for_foreground_clear` waiting for
  its own request: the self-stall already documented on the critic
  path. A foreground-ambiguous case stays foreground and relies on the
  bounded timeout;
- kwargs are passed only when `chat_completion`'s signature accepts
  them, so duck-typed clients (older stubs, wrappers) keep working and
  a TypeError can never eat the verdict.

The verdict payload itself is untouched: parsing and semantics are the
same, and `tests/test_verifier_two_stage.py::
test_classic_path_keeps_original_payload` continues to pin that.
"""

import json

from ghost_agent.core.verifier import (
    Verifier,
    VerifyVerdict,
    _VERIFY_FALLBACK_TIMEOUT_S,
    _VERIFY_WORKER_TIMEOUT_S,
    _bounded_fallback_kwargs,
)

CONFIRM_JSON = json.dumps({
    "verdict": "CONFIRMED", "confidence": 0.9,
    "reasoning": "supported", "issues": [],
})


class _FallbackStub:
    """Duck-typed llm_client with no critic pool and no router — every
    `_call_llm` lands on the direct main-model fallback. Records the
    kwargs of every chat_completion call."""

    critic_clients = None

    def __init__(self, foreground_requests=None, response=CONFIRM_JSON):
        if foreground_requests is not None:
            self.foreground_requests = foreground_requests
        self._response = response
        self.calls = []  # list of (payload, kwargs)

    async def chat_completion(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return {"choices": [{"message": {"content": self._response}}]}


class _LegacyPositionalStub:
    """A client whose chat_completion accepts NOTHING but the payload —
    the shape several older test stubs use. The fallback must not
    TypeError it (which the broad except would silently turn into a
    skipped verdict)."""

    critic_clients = None

    async def chat_completion(self, payload):
        return {"choices": [{"message": {"content": CONFIRM_JSON}}]}


# ---------- the timeout bound itself ----------


def test_fallback_timeout_is_bounded_and_roomier_than_worker():
    """The whole point: no more 1200s httpx default. Bounded, but with
    more headroom than the worker's budget because the main model may
    think before the JSON."""
    assert 0 < _VERIFY_FALLBACK_TIMEOUT_S <= 300.0
    assert _VERIFY_FALLBACK_TIMEOUT_S >= _VERIFY_WORKER_TIMEOUT_S


def test_fallback_timeout_env_override(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_FALLBACK_TIMEOUT", "60")
    import importlib
    import ghost_agent.core.verifier as vmod
    importlib.reload(vmod)
    try:
        assert vmod._VERIFY_FALLBACK_TIMEOUT_S == 60.0
    finally:
        monkeypatch.delenv("GHOST_VERIFY_FALLBACK_TIMEOUT", raising=False)
        importlib.reload(vmod)


def test_fallback_timeout_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_FALLBACK_TIMEOUT", "not-a-number")
    import importlib
    import ghost_agent.core.verifier as vmod
    importlib.reload(vmod)
    try:
        assert vmod._VERIFY_FALLBACK_TIMEOUT_S == 90.0
    finally:
        monkeypatch.delenv("GHOST_VERIFY_FALLBACK_TIMEOUT", raising=False)
        importlib.reload(vmod)


# ---------- fallback call carries the bound ----------


async def test_fallback_call_passes_bounded_timeout():
    stub = _FallbackStub(foreground_requests=1)
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("the sky is blue", "sky: blue", "")
    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED
    # Every call this stub saw was the direct main-model fallback
    # (no critic pool, no route attr) — each must be bounded.
    assert stub.calls, "fallback was never reached"
    for _payload, kwargs in stub.calls:
        assert kwargs.get("timeout") == _VERIFY_FALLBACK_TIMEOUT_S


async def test_fallback_call_never_unbounded_via_code_output_path():
    stub = _FallbackStub(foreground_requests=1)
    v = Verifier(llm_client=stub)
    result = await v.verify_code_output(
        code="print(6)", output="6", intent="print six", response="6")
    assert result is not None
    for _payload, kwargs in stub.calls:
        assert kwargs.get("timeout") == _VERIFY_FALLBACK_TIMEOUT_S


# ---------- is_background awareness ----------


async def test_background_context_marks_call_background():
    """No live user request (foreground_requests == 0) → the verify came
    from a background flow (dream/self-play/idle) or a late async
    verdict; the fallback must ride the background lane so it stops
    inflating foreground_tasks."""
    stub = _FallbackStub(foreground_requests=0)
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert result is not None
    assert stub.calls
    for _payload, kwargs in stub.calls:
        assert kwargs.get("is_background") is True
        assert kwargs.get("timeout") == _VERIFY_FALLBACK_TIMEOUT_S


async def test_live_user_request_stays_foreground():
    """foreground_requests > 0: the verify may BE part of the live user
    turn (in-loop auto-repair verdict). Marking it background would park
    it on _wait_for_foreground_clear behind its own request — it must
    stay foreground and rely on the bounded timeout."""
    stub = _FallbackStub(foreground_requests=1)
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert result is not None
    for _payload, kwargs in stub.calls:
        assert "is_background" not in kwargs


async def test_unknown_foreground_state_stays_foreground():
    """No foreground_requests attribute at all (duck-typed client) —
    assume a user request may be live; never self-park."""
    stub = _FallbackStub()  # attribute absent
    v = Verifier(llm_client=stub)
    result = await v.verify_claim("c", "e", "x")
    assert result is not None
    for _payload, kwargs in stub.calls:
        assert "is_background" not in kwargs


def test_non_numeric_foreground_counter_stays_foreground():
    """A wrapper whose counter isn't int-coercible must read as 'user
    may be live': keep the timeout bound but never mark background."""
    stub = _FallbackStub()
    stub.foreground_requests = "n/a"  # int() raises ValueError
    kwargs = _bounded_fallback_kwargs(stub)
    assert "is_background" not in kwargs
    assert kwargs.get("timeout") == _VERIFY_FALLBACK_TIMEOUT_S


def test_bare_magicmock_client_gets_no_kwargs_at_all():
    """A bare MagicMock/AsyncMock client (the tests/test_verifier.py
    fixture shape) has a non-introspectable signature — Mock auto-creates
    a bogus __signature__ — so the helper must degrade to NO kwargs:
    byte-for-byte the pre-fix call, never a TypeError'd verdict."""
    from unittest.mock import AsyncMock, MagicMock
    client = MagicMock()
    client.chat_completion = AsyncMock()
    assert _bounded_fallback_kwargs(client) == {}


# ---------- duck-typing back-compat ----------


async def test_positional_only_stub_still_verifies():
    """Clients whose chat_completion takes only the payload must keep
    working: the guards are skipped rather than TypeError-ing the call
    into a silently skipped verdict."""
    v = Verifier(llm_client=_LegacyPositionalStub())
    result = await v.verify_claim("2+2=4", "math")
    assert result is not None
    assert result.verdict == VerifyVerdict.CONFIRMED


def test_bounded_kwargs_positional_only_signature_yields_nothing():
    assert _bounded_fallback_kwargs(_LegacyPositionalStub()) == {}


def test_bounded_kwargs_no_chat_completion_yields_nothing():
    class _NotAClient:
        pass
    assert _bounded_fallback_kwargs(_NotAClient()) == {}


def test_bounded_kwargs_named_params_without_var_kw():
    """A real-LLMClient-shaped signature (named timeout/is_background,
    no **kwargs) gets both guards."""
    class _RealShaped:
        foreground_requests = 0

        async def chat_completion(self, payload, use_worker=False,
                                  use_critic=False, is_background=False,
                                  timeout=None):
            return {}

    kwargs = _bounded_fallback_kwargs(_RealShaped())
    assert kwargs["timeout"] == _VERIFY_FALLBACK_TIMEOUT_S
    assert kwargs["is_background"] is True


def test_real_llmclient_signature_accepts_both_guards():
    """Pin against the coordinator-owned client drifting away from the
    kwargs the fallback relies on."""
    import inspect
    from ghost_agent.core.llm import LLMClient
    params = inspect.signature(LLMClient.chat_completion).parameters
    assert "timeout" in params
    assert "is_background" in params


# ---------- payload untouched (parsing/semantics unchanged) ----------


async def test_fallback_payload_content_not_mutated(monkeypatch):
    """The bound lives in the call kwargs, not the payload: the classic
    single-prompt payload must keep its thinking-sized budget and carry
    no /no_think suffix (also pinned by test_verifier_two_stage.py)."""
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _FallbackStub(foreground_requests=1)
    v = Verifier(llm_client=stub)
    await v.verify_claim("c", "e", "x")
    payload, kwargs = stub.calls[0]
    assert payload["max_tokens"] == 2048
    assert "stop" not in payload
    assert "chat_template_kwargs" not in payload
    assert not payload["messages"][0]["content"].rstrip().endswith(
        "/no_think")
    assert kwargs.get("timeout") == _VERIFY_FALLBACK_TIMEOUT_S
