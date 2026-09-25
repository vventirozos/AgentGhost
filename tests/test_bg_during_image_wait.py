"""Background LLM calls may run while the foreground is waiting on the
image node — 2026-09-24.

`foreground_requests` parks background callers for the whole life of a user
request (the post-req-70 starvation fix). But an image turn spends 200-500 s
blocked on a node that is NOT the main LLM slot, and every background call
sat parked for all of it. `generate_image` now marks that window; the park
admits a caller while the window is open and no LLM call is in flight.
"""
import asyncio

import pytest

from ghost_agent.core.llm import LLMClient, _bg_during_image_wait_enabled


def _bare_client():
    c = LLMClient.__new__(LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c.main_slot_idle_windows = 0
    c._foreground_lock = asyncio.Lock()
    return c


def _instant_sleep(monkeypatch, counter):
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        counter.append(seconds)
        if len(counter) > 5:            # a parked caller: release it so the test ends
            counter[:] = counter[:6]
            raise RuntimeError("still parked")
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


async def test_admitted_during_an_image_wait(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    sleeps = []
    _instant_sleep(monkeypatch, sleeps)
    with c.main_slot_idle():               # the window grants its one admission
        await c._wait_for_foreground_clear()
    assert sleeps == []


async def test_still_parked_when_an_llm_call_is_in_flight(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    c.main_slot_idle_windows = 1
    c._idle_window_admits = 1           # the window HAS an admission: only the condition under test may park
    c.foreground_tasks = 1
    sleeps = []
    _instant_sleep(monkeypatch, sleeps)
    with pytest.raises(RuntimeError, match="still parked"):
        await c._wait_for_foreground_clear()


async def test_still_parked_with_no_window_open(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    sleeps = []
    _instant_sleep(monkeypatch, sleeps)
    with pytest.raises(RuntimeError, match="still parked"):
        await c._wait_for_foreground_clear()


async def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GHOST_BG_DURING_IMAGE_WAIT", "0")
    assert _bg_during_image_wait_enabled() is False
    c = _bare_client()
    c.foreground_requests = 1
    c.main_slot_idle_windows = 1
    c._idle_window_admits = 1           # the window HAS an admission: only the condition under test may park
    sleeps = []
    _instant_sleep(monkeypatch, sleeps)
    with pytest.raises(RuntimeError, match="still parked"):
        await c._wait_for_foreground_clear()


def test_the_window_is_a_counter_that_survives_nesting_and_errors():
    c = _bare_client()
    with c.main_slot_idle():
        assert c.main_slot_idle_windows == 1
        with c.main_slot_idle():
            assert c.main_slot_idle_windows == 2
        assert c.main_slot_idle_windows == 1
    assert c.main_slot_idle_windows == 0
    with pytest.raises(ValueError):
        with c.main_slot_idle():
            raise ValueError("render failed")
    assert c.main_slot_idle_windows == 0


async def test_generate_image_opens_the_window_around_the_node_post(monkeypatch):
    """The window is open exactly while the node renders: observed from
    inside the mocked POST, closed again after."""
    from contextlib import asynccontextmanager
    c = _bare_client()
    seen = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"b64_json": "AA=="}]}

    class _Client:
        async def post(self, path, json=None):
            seen["windows_during_post"] = c.main_slot_idle_windows
            return _Resp()

    node = {"model": "Ghost", "client": _Client(), "url": "http://img"}
    c.image_gen_clients = [node]
    c.get_image_gen_node = lambda *_a, **_k: node

    @asynccontextmanager
    async def _slot(*_a, **_k):
        seen["windows_during_slot_wait"] = c.main_slot_idle_windows   # R2 review: the wait for a busy node is idle time too
        yield
    c._node_slot = _slot
    from unittest.mock import MagicMock
    c.circuit_breaker = MagicMock()
    out = await c.generate_image({"prompt": "a cat"})
    assert out == {"data": [{"b64_json": "AA=="}]}
    assert seen["windows_during_post"] == 1 and seen["windows_during_slot_wait"] == 1
    assert c.main_slot_idle_windows == 0



async def test_one_admission_per_render_window(monkeypatch):
    """R4 review: every parked caller polled the same flag and all passed in
    the same tick. Exactly one caller is admitted per window."""
    c = _bare_client()
    c.foreground_requests = 1
    admitted = []

    async def caller(i):
        try:
            await asyncio.wait_for(c._wait_for_foreground_clear(), timeout=0.5)
            admitted.append(i)
        except asyncio.TimeoutError:
            pass
    with c.main_slot_idle():
        await asyncio.gather(*(caller(i) for i in range(6)))
    assert len(admitted) == 1, admitted



async def test_a_second_active_request_keeps_the_window_shut(monkeypatch):
    """R4 pins review: the window is process-global; with two requests active
    and only one rendering, request B's gap between tool calls is NOT idle."""
    c = _bare_client()
    c.foreground_requests = 2
    sleeps = []
    _instant_sleep(monkeypatch, sleeps)
    with c.main_slot_idle():
        with pytest.raises(RuntimeError, match="still parked"):
            await c._wait_for_foreground_clear()
