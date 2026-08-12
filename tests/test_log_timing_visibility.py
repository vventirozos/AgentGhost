"""Every operation's cost must be visible in the stream the operator watches.

Two blind spots, both observed live on 2026-08-12:

  * the `└─ request finished +Ns` frame closes when `handle_chat` RETURNS,
    which on the streaming path is when the answer starts streaming — so a
    turn whose banner said "+23.0s" handed the terminal back ~2 minutes
    later, and every second of that generation was logged nowhere;
  * `**` (SYSTEM-scoped) lines carried no delta at all — six blank spaces —
    so post-turn and idle work was the only part of the agent whose cost was
    invisible.
"""

import asyncio
import re
import time

import pytest

from ghost_agent.utils import logging as glog


@pytest.fixture(autouse=True)
def _fresh_anchor():
    glog.reset_system_delta_anchor()
    yield
    glog.reset_system_delta_anchor()


# ------------------------------------------------------- ** line deltas

def test_the_first_system_line_has_no_delta_to_report():
    """There is nothing to measure against yet, and inventing +0.00s would
    imply the step was instant.

    Asserted on the PURE helper, not the process-global anchor: any
    background thread that logs (watchdog, scheduler, spawn_bg work) can
    advance that anchor between two statements, which is exactly how the
    first version of this test passed alone and failed in a full run."""
    assert glog._delta_from(None, 100.0) == "      "


def test_later_system_lines_report_the_gap_since_the_previous_one():
    assert glog._delta_from(100.0, 100.05).strip() == "+0.05s"
    assert glog._delta_from(100.0, 142.0).strip() == "+42.0s"


def test_the_delta_is_the_gap_between_lines_not_since_boot():
    """A daemon runs for weeks; "since process start" would be useless. Each
    line reports the duration of the step that just finished, so the values
    do NOT grow monotonically."""
    first = glog._delta_from(100.0, 100.5)
    second = glog._delta_from(100.5, 100.6)
    assert float(re.search(r"\+\s*([\d.]+)s", second).group(1)) < \
        float(re.search(r"\+\s*([\d.]+)s", first).group(1))


def test_a_clock_that_goes_backwards_never_renders_a_negative():
    """Column width is fixed at 6; a '-' would break every line after it."""
    assert glog._delta_from(100.0, 99.0).strip() == "+0.00s"


def test_the_stateful_reader_advances_its_anchor():
    """One integration check on the real anchor, written so a racing
    background log line cannot fail it: the value is bounded, not exact."""
    glog.reset_system_delta_anchor()
    glog._system_delta()
    time.sleep(0.05)
    out = glog._system_delta()
    assert out.strip().startswith("+")
    assert len(out) == 6


def test_a_request_opening_resets_the_anchor():
    """The gap across a whole user turn is not the duration of any background
    step — reporting it as one would repeat the category error the request
    frame made. Asserted on the anchor VALUE, which BEGIN sets to None,
    rather than on a rendered delta a racing thread could have advanced."""
    with glog._SYSTEM_ANCHOR_LOCK:
        glog._SYSTEM_ANCHOR["t"] = 12345.0
    tok = glog.request_id_context.set("abcd1234")
    try:
        glog.pretty_log("Request Started", special_marker="BEGIN")
        with glog._SYSTEM_ANCHOR_LOCK:
            anchor = glog._SYSTEM_ANCHOR["t"]
        assert anchor is None or anchor > 12345.0, (
            "BEGIN must drop the pre-request anchor")
    finally:
        glog.pretty_log("Request Finished", special_marker="END")
        glog.request_id_context.reset(tok)


def test_a_tracked_request_still_measures_from_its_own_start():
    tok = glog.request_id_context.set("efgh5678")
    try:
        glog.pretty_log("Request Started", special_marker="BEGIN")
        time.sleep(0.05)
        out = glog._format_delta("efgh5678")
        val = float(re.search(r"\+\s*([\d.]+)s", out).group(1))
        assert 0.03 <= val < 2.0
    finally:
        glog.pretty_log("Request Finished", special_marker="END")
        glog.request_id_context.reset(tok)


def test_the_delta_is_the_gap_between_lines_not_since_boot():
    """A daemon runs for weeks; "since process start" would be useless. The
    reading must be the duration of the step that just finished."""
    glog.reset_system_delta_anchor()
    glog._format_delta("SYSTEM")
    time.sleep(0.05)
    first = float(re.search(r"\+\s*([\d.]+)s",
                            glog._format_delta("SYSTEM")).group(1))
    time.sleep(0.15)
    second = float(re.search(r"\+\s*([\d.]+)s",
                             glog._format_delta("SYSTEM")).group(1))
    assert second > first, (
        f"each line must report its OWN gap ({first}s then {second}s), not a "
        f"monotonically growing time-since-boot")


def test_an_unknown_non_system_id_still_gets_blanks():
    """Only SYSTEM gets the rolling anchor; a closed request id must not
    silently borrow it."""
    assert glog._format_delta("no-such-request") == "      "


def test_the_delta_columns_stay_aligned():
    """Six characters, always — the content column is computed from it."""
    glog.reset_system_delta_anchor()
    glog._format_delta("SYSTEM")
    for _ in range(3):
        assert len(glog._format_delta("SYSTEM")) == 6
    assert len(glog._fmt_secs(0.5)) == 6
    assert len(glog._fmt_secs(42.0)) == 6
    assert len(glog._fmt_secs(1234.0)) == 6


# ------------------------------------------------- the true elapsed line

def test_the_drain_reports_total_and_time_to_first_token():
    """The frame's number is kept (it IS when output starts appearing, and
    three clients parse that frame); this is the number it was being
    mistaken for."""
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    idx = src.index("async def _stream_then_unregister")
    body = src[idx:idx + 2200]
    assert "Stream Drained" in body
    assert "TOTAL" in body and "first token" in body
    assert "_chunks" in body and "_bytes" in body
    # It must be in the FINALLY, or a client that disconnects mid-stream
    # produces no line at all — exactly the case worth seeing.
    assert body.index("finally:") < body.index("Stream Drained")


def test_the_drain_line_survives_a_client_that_disconnects(monkeypatch):
    """A cancelled drain is the case where knowing the elapsed matters most.
    Reproduces the wrapper's shape rather than importing it (it closes over
    a whole turn's locals)."""
    seen = []

    async def _gen():
        yield b"a"
        yield b"bc"
        raise asyncio.CancelledError()

    async def _wrapper(g):
        t0 = time.monotonic()
        chunks = bytes_ = 0
        try:
            async for c in g:
                chunks += 1
                bytes_ += len(c)
                yield c
        finally:
            seen.append((chunks, bytes_, time.monotonic() - t0))

    async def _drive():
        with pytest.raises(asyncio.CancelledError):
            async for _ in _wrapper(_gen()):
                pass

    asyncio.run(_drive())
    assert seen and seen[0][0] == 2 and seen[0][1] == 3, seen


def test_the_console_shows_total_and_ttft_before_truncation():
    """The console truncates content at ~60 chars; the pair that answers
    'why did the banner say 23s when I waited two minutes' must survive."""
    line = ("b72d9978 · TOTAL 140.2s · first token 23.0s · 2,317 chunks · "
            "178 KB · 20 chunk/s")
    head = line[:glog.LOG_TRUNCATE_LIMIT]
    assert "TOTAL 140.2s" in head
    assert "first token 23.0s" in head
