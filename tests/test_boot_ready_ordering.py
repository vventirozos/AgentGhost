"""The READY banner must be the LAST line of boot.

The main-prefix warmup is spawned in the background (it prefills ~24k tokens
and must not hold up the socket), so it was the one boot step that logged
AFTER "system ready" — which made the banner read as a lie in the operator's
stream, since the stream is what the operator watches boot in.

Only the LOG LINE waits. The socket is opened by the `yield` in `lifespan`
regardless, so nothing here delays actual serving.
"""

import asyncio
import time

import pytest

import ghost_agent.main as main_mod


@pytest.mark.asyncio
async def test_ready_is_announced_after_the_warmup(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _warm():
        await asyncio.sleep(0.05)
        seen.append("Prefix Warmup Done")

    await main_mod._announce_ready_when_warm(asyncio.ensure_future(_warm()))
    assert seen == ["Prefix Warmup Done", "System Ready"], (
        f"ready must come last, got {seen}")


@pytest.mark.asyncio
async def test_ready_is_immediate_when_there_is_no_warmup(monkeypatch):
    """A mocked client or GHOST_MAIN_PREFIX_WARMUP=0 spawns no task."""
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))
    t0 = time.monotonic()
    await main_mod._announce_ready_when_warm(None)
    assert seen == ["System Ready"]
    assert time.monotonic() - t0 < 0.5, "must not wait on nothing"


@pytest.mark.asyncio
async def test_a_wedged_warmup_cannot_suppress_the_ready_line(monkeypatch):
    """A log line must never be able to hide. A hung upstream would otherwise
    mean NO ready banner for a server that is already serving."""
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))
    hung = asyncio.ensure_future(asyncio.sleep(30))
    try:
        t0 = time.monotonic()
        await main_mod._announce_ready_when_warm(hung, timeout=0.2)
        assert "System Ready" in seen
        assert "Prefix Warmup Slow" in seen, "the operator is told why"
        assert time.monotonic() - t0 < 2.0
        # …and the wait is SHIELDED: timing out must not cancel the warmup.
        assert not hung.done(), "the warmup must survive the announce timeout"
    finally:
        hung.cancel()


@pytest.mark.asyncio
async def test_a_failed_warmup_still_announces_ready(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _boom():
        raise RuntimeError("upstream down")

    task = asyncio.ensure_future(_boom())
    await main_mod._announce_ready_when_warm(task)
    assert seen == ["System Ready"]


@pytest.mark.asyncio
async def test_an_already_finished_warmup_does_not_stall(monkeypatch):
    seen = []
    monkeypatch.setattr(main_mod, "pretty_log",
                        lambda title, content="", **kw: seen.append(title))

    async def _done():
        return None

    task = asyncio.ensure_future(_done())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    t0 = time.monotonic()
    await main_mod._announce_ready_when_warm(task)
    assert seen == ["System Ready"]
    assert time.monotonic() - t0 < 0.5


def test_the_lifespan_announces_through_the_helper():
    """The banner must not be re-inlined by a later edit — that is exactly
    how it ended up printing before the warmup in the first place."""
    import inspect
    src = inspect.getsource(main_mod)
    assert "_announce_ready_when_warm(_warmup_task)" in src
    assert src.count('pretty_log("System Ready"') == 1, (
        "one announce site only — the helper's")
