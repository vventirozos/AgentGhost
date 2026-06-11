"""Tests for background-LLM yielding to active foreground requests.

Regression target (post-req-70 starvation): `foreground_tasks` only
counts LLM calls in flight, but a user turn spends much of its wall-clock
BETWEEN LLM calls (tools, file I/O, browser). Background cycles grabbed
the single llama slot in those gaps and the user's next prompt queued
behind full background generations for ~12 minutes. The fix adds
`foreground_requests` — active user requests at the API layer — and
parks background callers while either signal is hot.
"""
import asyncio
import inspect

from ghost_agent.core.llm import LLMClient


def _bare_client():
    c = LLMClient.__new__(LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c._foreground_lock = asyncio.Lock()
    return c


def _instant_sleep(monkeypatch, counter):
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        counter.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


async def test_returns_immediately_when_idle(monkeypatch):
    c = _bare_client()
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert sleeps == []


async def test_waits_for_active_request_then_releases(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)

    async def finish_request():
        # let the waiter spin a few ticks, then clear the request
        for _ in range(3):
            await asyncio.sleep(0)
        c.foreground_requests = 0

    await asyncio.gather(c._wait_for_foreground_clear(), finish_request())
    assert len(sleeps) >= 1  # it actually parked


async def test_active_request_outlasts_the_old_30s_cap(monkeypatch):
    """With a user request active, background must keep waiting well past
    the 30-tick budget that used to let it barge in."""
    c = _bare_client()
    c.foreground_requests = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)

    async def clear_after_120_ticks():
        while len(sleeps) < 120:
            await asyncio.sleep(0)
        c.foreground_requests = 0

    await asyncio.gather(
        c._wait_for_foreground_clear(), clear_after_120_ticks())
    assert len(sleeps) >= 120


async def test_task_only_blocking_keeps_30s_cap(monkeypatch):
    """No active request, just an in-flight LLM call: the old 30s
    proceed-anyway behavior is preserved (a stale background result
    beats none)."""
    c = _bare_client()
    c.foreground_tasks = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert 28 <= len(sleeps) <= 35


async def test_leaked_counter_hard_ceiling(monkeypatch):
    """A leaked foreground_requests must not deadlock background work
    forever — the 600-tick hard ceiling releases it."""
    c = _bare_client()
    c.foreground_requests = 1  # never cleared
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert 595 <= len(sleeps) <= 605


def test_api_layer_marks_requests_on_both_paths():
    """Both chat handlers (streaming + non-streaming) must bracket
    handle_chat with the foreground_requests counter — increment before,
    guaranteed decrement after."""
    import ghost_agent.api.routes as routes
    src = inspect.getsource(routes)
    assert src.count("_mark_foreground(agent, +1)") >= 2
    assert src.count("_mark_foreground(agent, -1)") >= 2


def test_mark_foreground_counts_and_floors_at_zero():
    from types import SimpleNamespace
    from ghost_agent.api.routes import _mark_foreground

    llm = SimpleNamespace(foreground_requests=0)
    agent = SimpleNamespace(context=SimpleNamespace(llm_client=llm))
    _mark_foreground(agent, +1)
    _mark_foreground(agent, +1)
    assert llm.foreground_requests == 2
    _mark_foreground(agent, -1)
    _mark_foreground(agent, -1)
    _mark_foreground(agent, -1)  # extra decrement must floor, not go negative
    assert llm.foreground_requests == 0


def test_mark_foreground_tolerates_mocked_clients():
    """Test apps wire MagicMock contexts — instrumentation must no-op,
    never raise (the original inline version broke 11 API tests with
    `max(0, MagicMock - 1)`)."""
    from unittest.mock import MagicMock
    from ghost_agent.api.routes import _mark_foreground

    _mark_foreground(MagicMock(), +1)   # mock counter → skipped
    _mark_foreground(object(), +1)      # no context at all → skipped
    _mark_foreground(None, -1)          # no agent → skipped


def test_background_gates_use_the_shared_wait():
    src = inspect.getsource(LLMClient.chat_completion)
    assert "_wait_for_foreground_clear" in src
    src_stream = inspect.getsource(LLMClient.stream_chat_completion)
    assert "_wait_for_foreground_clear" in src_stream
