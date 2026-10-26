"""§4RR (2026-10-26) — two YouTube transcription faults found by an end-to-end run.

1. yt-dlp 2026.10 prints an "n challenge solving failed … Ensure you have a
   supported JavaScript runtime …" warning on EVERY fetch. The stack-fault
   regex matched "javascript runtime", so the first bot-walled circuit ended
   the whole fetch as "repair the stack" instead of rotating to the next one.
2. Audio windows were 720 s (~18k tokens), sized for a 32k slot; nova's slots
   are 16,384 since 2026-10-25, so every 12-minute window failed with HTTP 400
   and was stored as a gap. Windows are now fitted to the node's slot.

The stderr fixtures are VERBATIM from yt-dlp 2026.10.21 over Tor (2026-10-26).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.memory import audio_ingest as ai
from ghost_agent.memory import youtube_ingest as yi

VID = "UF8uR6Z6KLc"

# Printed on every fetch by yt-dlp 2026.10.21 with deno installed but no EJS solver script.
EJS_WARNINGS = (
    "WARNING: [youtube] [jsc] Remote components challenge solver script (deno) and NPM package (deno) "
    "were skipped. These may be required to solve JS challenges. You can enable these downloads with  "
    "--remote-components ejs:github  (recommended) or  --remote-components ejs:npm , respectively. For "
    "more information and alternatives, refer to  https://github.com/yt-dlp/yt-dlp/wiki/EJS\n"
    f"WARNING: [youtube] {VID}: n challenge solving failed: Some formats may be missing. Ensure you have "
    "a supported JavaScript runtime and challenge solver script distribution installed. Review any "
    "warnings presented before this message. For more details, refer to  "
    "https://github.com/yt-dlp/yt-dlp/wiki/EJS\n"
)
BOT_WALL = (f"ERROR: [youtube] {VID}: Sign in to confirm you’re not a bot. Use --cookies-from-browser or "
            "--cookies for the authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-t")
# yt-dlp's own missing-runtime line (extractor/youtube/_video.py).
NO_RUNTIME = ("WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by "
              "default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config.\n")


# ── 1. the challenge warning is not a stack fault ────────────────────────────

def test_a_bot_wall_under_the_challenge_warning_is_retried_not_terminal():
    kind, reason = yi.classify_failure(EJS_WARNINGS + BOT_WALL)
    assert kind == "retry", reason
    assert "deno" not in reason and "Sign in to confirm" in reason


def test_a_missing_runtime_is_still_a_terminal_stack_fault():
    kind, reason = yi.classify_failure(NO_RUNTIME + "ERROR: [youtube] x: Requested format is not available")
    assert kind == "terminal" and "deno" in reason


def test_the_rotation_moves_past_a_walled_first_circuit():
    """Live 2026-10-26: circuit 1 walled → the fetch stopped after ONE circuit."""
    calls = []

    def do(proxy):
        calls.append(proxy)
        return (True, "ok") if len(calls) == 3 else (False, EJS_WARNINGS + BOT_WALL)

    st = yi.FetchStats()
    ok, payload = yi._rotate(VID, "resolve", "socks5://127.0.0.1:9050", 8, do, st)
    assert ok and payload == "ok" and len(calls) == 3
    assert st.bot_walls == 2 and len(set(calls)) == 3


# ── 2. the audio window fits the node's slot ─────────────────────────────────

def _tokens(seconds):
    """What one window costs a slot, by the same rates the fit uses."""
    return seconds * (ai.AUDIO_TOKENS_PER_S + ai.TRANSCRIPT_TOKENS_PER_S) + ai.PROMPT_RESERVE_TOKENS


@pytest.mark.parametrize("ctx", [8192, 16384, 32768])
def test_a_fitted_window_fits_the_slot(ctx):
    w = ai.fit_window_seconds(720.0, ctx)
    assert w <= 720.0 and _tokens(w) <= ctx


def test_the_measured_failure_no_longer_fits():
    """720 s measured at 18,068 prompt tokens against a 16,384 slot (HTTP 400)."""
    assert 720.0 * ai.AUDIO_TOKENS_PER_S > 16384
    assert ai.fit_window_seconds(720.0, 16384) < 600.0, "600 s was cut off mid-transcript live"
    assert ai.fit_window_seconds(720.0, 16384) >= 480.0, "480 s finished live — do not shrink below it"


def test_the_fit_never_grows_the_window_and_keeps_it_when_unknown():
    assert ai.fit_window_seconds(300.0, 65536) == 300.0
    assert ai.fit_window_seconds(480.0, None) == 480.0
    assert ai.fit_window_seconds(480.0, 0) == 480.0
    assert ai.fit_window_seconds(480.0, 600) == ai.MIN_WINDOW_SECONDS


def test_the_default_window_fits_a_16k_slot_without_props():
    assert _tokens(ai.WINDOW_SECONDS) <= 16384


def _drive(monkeypatch, *, total, window, slot_ctx_fn=None, post_fn="fake", node_ctx=None):
    cuts = []
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: total)
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: cuts.append(d) or b"WAV")
    reply = {"choices": [{"message": {"content": "words spoken here"}, "finish_reason": "stop"}]}
    if post_fn == "fake":
        post_fn = lambda url, payload, timeout: reply  # noqa: E731
    else:  # the default node path: the real transcribe call, network stubbed
        monkeypatch.setattr(ai, "_post_json", lambda url, payload, timeout: reply)
        monkeypatch.setattr(ai, "node_slot_ctx", lambda: node_ctx)
    st = ai.AudioIngestStats()
    list(ai.iter_audio_chunks(Path("t.m4a"), "t.m4a", stats=st, post_fn=post_fn, window_seconds=window,
                              window_overlap=15, sleep=lambda s: None, slot_ctx_fn=slot_ctx_fn))
    return st, cuts


def test_windows_are_cut_to_the_slot_and_the_whole_recording_is_covered(monkeypatch):
    st, cuts = _drive(monkeypatch, total=904.0, window=720.0, slot_ctx_fn=lambda: 16384)
    assert cuts and max(cuts) <= ai.fit_window_seconds(720.0, 16384)
    assert all(_tokens(d) <= 16384 for d in cuts)
    assert st.seconds == 904.0 and not st.gaps


def test_the_real_node_path_reads_the_slot_by_default(monkeypatch):
    """No injected post_fn = the windows go to AUDIO_NODE_URL, so its slot sizes them."""
    _, cuts = _drive(monkeypatch, total=904.0, window=720.0, post_fn=None, node_ctx=8192)
    assert max(cuts) <= ai.fit_window_seconds(720.0, 8192) < 720.0


def test_an_unreadable_slot_keeps_the_configured_window(monkeypatch):
    _, cuts = _drive(monkeypatch, total=904.0, window=480.0, post_fn=None, node_ctx=None)
    assert max(cuts) == 480.0


def test_an_overlap_wider_than_the_fitted_window_is_shrunk(monkeypatch):
    monkeypatch.setattr(ai, "probe_duration_seconds", lambda p: 400.0)
    cuts = []
    monkeypatch.setattr(ai, "extract_window_wav", lambda p, s, d: cuts.append((s, d)) or b"WAV")
    reply = {"choices": [{"message": {"content": "w"}, "finish_reason": "stop"}]}
    st = ai.AudioIngestStats()
    list(ai.iter_audio_chunks(Path("t.m4a"), "t.m4a", stats=st, post_fn=lambda u, p, t: reply,
                              window_seconds=480.0, window_overlap=200.0, sleep=lambda s: None,
                              slot_ctx_fn=lambda: 2048))
    starts = [s for s, _ in cuts]
    assert len(cuts) < 20 and all(b > a for a, b in zip(starts, starts[1:]))


def test_node_slot_ctx_reads_props_and_degrades_to_none(monkeypatch):
    import httpx

    class _Resp:
        def __init__(self, body): self._b = body
        def json(self): return self._b

    class _Client:
        body = {"default_generation_settings": {"n_ctx": 16384}, "total_slots": 4}
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            assert url == f"{ai.AUDIO_NODE_URL}/props"
            if isinstance(self.body, Exception):
                raise self.body
            return _Resp(self.body)

    monkeypatch.setattr(httpx, "Client", _Client)
    assert ai.node_slot_ctx() == 16384
    _Client.body = {"total_slots": 4}
    assert ai.node_slot_ctx() is None
    _Client.body = httpx.ConnectError("down")
    assert ai.node_slot_ctx() is None
