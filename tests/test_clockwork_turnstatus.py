"""uConsole client — live turn-status caption (turnstatus.py).

Two kinds of guard here:

  * **Behaviour** — the corridor parser is exercised against REAL log lines
    lifted from ``Logs/ghost-agent.log`` (ANSI codes and all), because the
    parser's whole job is surviving that exact format.
  * **Drift** — ``turnstatus.py`` is a hand port of tables that live in
    ``interface/static/app.js``, and the handheld never sees app.js, so nothing
    at runtime can notice when the two diverge. These tests parse both files and
    fail when they do.

Unlike ``test_clockwork_client_ui.py`` (source-level AST guards, because
``client.py`` imports PyQt6 which this venv does not have), this module imports
the real thing: ``turnstatus.py`` is deliberately Qt-free.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CLIENT_DIR = _ROOT / "interface" / "externals" / "clockwork_ghost"
_APP_JS = _ROOT / "interface" / "static" / "app.js"

sys.path.insert(0, str(_CLIENT_DIR))
import turnstatus as ts  # noqa: E402


# ── real log lines, copied verbatim (ANSI intact) ───────────────────────────
HEADER = ("\x1b[38;5;213m┌─ \x1b[1m99\x1b[0m\x1b[38;5;213m 9961f364\x1b[0m  "
          "\x1b[2mrequest started  15:16:26\x1b[0m \x1b[38;5;213m─────────────\x1b[0m")
FILE_READ = ("\x1b[38;5;213m│\x1b[0m  \x1b[38;5;213m99\x1b[0m  📖  \x1b[2m+8.06s\x1b[0m  "
             "\x1b[36m\x1b[1mfile read         \x1b[0m  notes.md")
THINKING = ("\x1b[38;5;213m│\x1b[0m  \x1b[38;5;213m99\x1b[0m  💭  \x1b[2m+8.04s\x1b[0m  "
            "\x1b[2m\x1b[1mthinking          \x1b[0m  The user has attached an image")
WORKER = ("\x1b[38;5;213m│\x1b[0m  \x1b[38;5;213m99\x1b[0m  🔧  \x1b[2m+38.1s\x1b[0m  "
          "\x1b[36m\x1b[1mworker compute    \x1b[0m  verify → Worker Node (Nova)")
PREFILL = ("\x1b[38;5;213m│\x1b[0m  \x1b[38;5;213m99\x1b[0m  🧩  \x1b[2m+29.9s\x1b[0m  "
           "\x1b[36m\x1b[1mprefill cache     \x1b[0m  sys h=9ab1b534 chars=16757")
CONTINUATION = "                                       vision_analysis tool to describe"
OTHER_CORRIDOR = ("\x1b[38;5;141m│\x1b[0m  \x1b[38;5;141m6A\x1b[0m  🌐  \x1b[2m+0.01s\x1b[0m  "
                  "\x1b[36m\x1b[1mweb search        \x1b[0m  onion engines")


def _started(connected=True):
    t = ts.TurnTicker(clock=lambda: 0.0)
    t.set_connected(connected)
    t.start()
    return t


# ── caption behaviour ───────────────────────────────────────────────────────
def test_idle_ticker_ignores_everything():
    """No turn in flight → the socket's constant background traffic is inert."""
    t = ts.TurnTicker()
    assert t.note_line(HEADER) is False
    assert t.note_line(FILE_READ) is False
    assert t.req_id is None


def test_placeholder_says_starting_when_the_log_stream_is_up():
    assert _started(connected=True).desc == ts.DESC_STARTING


def test_placeholder_falls_back_to_cogitating_with_no_log_stream():
    """A dead interface must degrade to the OLD behaviour, not promise a
    description that will never arrive."""
    assert _started(connected=False).desc == ts.DESC_OFFLINE


def test_a_tool_line_becomes_a_description_and_its_icon():
    t = _started()
    t.note_line(HEADER)
    assert t.note_line(FILE_READ) is True
    assert t.desc == "reading a file · notes.md"
    assert t.icon == "📖"


def test_unmapped_titles_fall_back_to_the_raw_title():
    t = _started()
    t.note_line(HEADER)
    line = FILE_READ.replace("file read         ", "sandbox mount     ")
    t.note_line(line)
    assert t.desc.startswith("sandbox mount")


def test_thinking_lines_say_thinking_rather_than_going_stale():
    """The gap between two tools is minutes of reasoning; leaving the previous
    tool's caption up would read as a hang."""
    t = _started()
    t.note_line(HEADER)
    t.note_line(FILE_READ)
    assert t.note_line(THINKING) is True
    assert (t.icon, t.desc) == (ts.ICON_THINKING, "thinking…")


def test_plumbing_titles_are_never_shown():
    t = _started()
    t.note_line(HEADER)
    t.note_line(FILE_READ)
    assert t.note_line(PREFILL) is False
    assert t.desc == "reading a file · notes.md"


def test_continuation_lines_are_ignored():
    """Wrapped thinking text has no id column — it must not be parsed as a step."""
    t = _started()
    t.note_line(HEADER)
    t.note_line(FILE_READ)
    assert t.note_line(CONTINUATION) is False
    assert t.desc == "reading a file · notes.md"


def test_other_corridors_are_ignored():
    """Dreams and self-play stream over the same socket. Narrating them while
    the operator waits on their own question is the bug this prevents."""
    t = _started()
    t.note_line(HEADER)
    assert t.note_line(OTHER_CORRIDOR) is False
    assert t.desc == ts.DESC_STARTING


def test_a_corridor_already_open_is_never_adopted():
    """Adoption keys on `request started` — a corridor whose header predates the
    send has none to offer, so its body lines cannot capture the ticker."""
    t = _started()
    assert t.note_line(OTHER_CORRIDOR) is False
    assert t.req_id is None


def test_restart_re_arms_adoption():
    t = _started()
    t.note_line(HEADER)
    t.note_line(FILE_READ)
    t.start()
    assert t.req_id is None
    assert t.desc == ts.DESC_STARTING
    # The NEXT turn's corridor is a different id; the old one is now noise.
    assert t.note_line(FILE_READ) is False


def test_repeated_identical_lines_report_no_change():
    """The client repaints on True, and a repaint re-fits a QLabel — so an
    unchanged caption must report False."""
    t = _started()
    t.note_line(HEADER)
    assert t.note_line(FILE_READ) is True
    assert t.note_line(FILE_READ) is False


def test_long_detail_is_truncated_with_an_ellipsis():
    t = _started()
    t.note_line(HEADER)
    t.note_line(FILE_READ.replace("notes.md", "x" * 80))
    assert t.desc.endswith("…")
    assert len(t.desc) < 60


def test_clock_counts_from_start():
    now = {"t": 100.0}
    t = ts.TurnTicker(clock=lambda: now["t"])
    t.start()
    assert t.clock_text() == "0:00"
    now["t"] = 100.0 + 74
    assert t.clock_text() == "1:14"


def test_losing_the_socket_mid_wait_reverts_only_the_placeholder():
    t = _started()
    assert t.set_connected(False) is True
    assert t.desc == ts.DESC_OFFLINE
    # …but once a real step is on screen, keep it: a blinking socket must not
    # erase what the agent is actually doing.
    t.set_connected(True)
    t.note_line(HEADER)
    t.note_line(FILE_READ)
    assert t.set_connected(False) is False
    assert t.desc == "reading a file · notes.md"


# ── icon extraction ─────────────────────────────────────────────────────────
def test_corridor_rails_and_arrows_are_not_mistaken_for_icons():
    """`│ ┌ ─` and `→` are the format's own furniture and appear in content.
    Matching them would give every line a bogus icon (and the `think` class)."""
    assert ts.extract_icon("┌─ 99 9961f364  request started") is None
    assert ts.extract_icon("│  99  worker → Nova") is None


@pytest.mark.parametrize("icon", ["📖", "⏳", "✅", "❌", "⚡", "🔶", "🫀", "⏩"])
def test_known_icons_extract_from_a_real_line(icon):
    line = ts.clean_log_line(FILE_READ).replace("📖", icon)
    assert ts.extract_icon(line) == icon


def test_worker_line_keeps_its_arrow_detail():
    """`→` inside the content must survive — it is only barred from being read
    as the ICON."""
    t = _started()
    t.note_line(HEADER)
    t.note_line(WORKER)
    assert t.desc == "delegating to the worker node · verify → Worker Node (Nova)"[:len(t.desc)]
    assert t.icon == "🔧"


# ── the rendered caption ────────────────────────────────────────────────────
def _caption(desc, icon="📖", secs=0):
    t = ts.TurnTicker(clock=lambda: 0.0)
    t.start()
    t._t0 = -secs
    t.icon, t.desc = icon, desc
    return ts.caption_html(t, dim="#aaa", mono="mono")


def test_caption_reads_clock_then_description_then_icon():
    """Same field order as the web UI's turn-status line."""
    html = _caption("reading a file · notes.md", "📖", 34)
    assert html.index("0:34") < html.index("reading a file") < html.index("📖")


def test_caption_escapes_log_content():
    """The description is raw log text. An unescaped `<` is not a typo in the
    caption — the QLabel is in RichText mode, so it swallows the rest."""
    html = _caption("executing code · <script>x</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_caption_ampersands_survive_escaping():
    assert "&amp;" in _caption("running a command · make a & b")


def test_long_captions_are_elided_to_one_line():
    """A wrapped caption changes the bubble's HEIGHT every few seconds, which
    reads as the transcript twitching. Measured on-device: ~46 chars."""
    html = _caption("delegating to the worker node · verify → Worker Node (Nova)")
    body = re.search(r"<i>(.*?)</i>", html).group(1)
    assert len(body) == ts.CAPTION_DESC_CHARS
    assert body.endswith("…")


def test_short_captions_are_left_alone():
    body = re.search(r"<i>(.*?)</i>", _caption("thinking…")).group(1)
    assert body == "thinking…"


# ── drift guards against app.js ─────────────────────────────────────────────
def _js_object(name: str) -> dict:
    """Pull `const <name> = { 'k': 'v', … }` out of app.js."""
    src = _APP_JS.read_text()
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    assert m, f"{name} not found in app.js — the tables moved, re-point this test"
    return dict(re.findall(r"'([^']+)':\s*'([^']+)'", m.group(1)))


def _js_set(name: str) -> set:
    src = _APP_JS.read_text()
    m = re.search(r"const %s = new Set\(\[(.*?)\]\);" % name, src, re.S)
    assert m, f"{name} not found in app.js — the tables moved, re-point this test"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_icon_class_matches_the_web_client():
    assert ts.ICON_CLASS == _js_object("ICON_CLASS")


def test_ticker_verbs_match_the_web_client():
    assert ts.TICKER_VERBS == _js_object("TICKER_VERBS")


def test_ticker_noise_matches_the_web_client():
    assert ts.TICKER_NOISE == _js_set("TICKER_NOISE")


# ── the log socket ──────────────────────────────────────────────────────────
def test_ws_url_carries_the_key_in_the_query_string():
    """The endpoint is key-gated and WebSockets take no custom headers, so
    `?key=` is the only dialect the interface accepts."""
    url = ts.log_ws_url("eva", "s3cr/et")
    assert url.startswith("wss://eva:8080/ws?key=")
    assert "s3cr%2Fet" in url


def test_stream_survives_a_dead_interface(monkeypatch):
    """Connect failures must be swallowed and retried — a down interface costs
    a caption, never the client."""
    states, sleeps = [], []

    class _FakeWS:
        def connect(self, url, **kw):
            raise OSError("connection refused")

    monkeypatch.setitem(sys.modules, "websockets", _FakeWS())

    async def _fake_sleep(d):
        sleeps.append(d)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError
    monkeypatch.setattr(ts.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ts.stream_log_lines(
            "wss://eva:8080/ws?key=x", on_line=lambda _l: None,
            on_state=states.append))
    assert states == [False, False, False]      # never claimed to be connected
    assert sleeps == [1.0, 2.0, 4.0]            # exponential backoff


def test_stream_feeds_only_log_frames():
    """The socket may carry other frame types; only `type: log` is a log line."""
    seen = []

    class _FakeConn:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def __aiter__(self):  # pragma: no cover - replaced below
            pass

    class _Conn(_FakeConn):
        def __init__(self, frames): self._frames = frames

        async def __aiter__(self):
            for f in self._frames:
                yield f

    frames = [
        json.dumps({"type": "log", "content": "line one"}),
        json.dumps({"type": "status", "content": "not a log"}),
        "{ not json",
        json.dumps({"type": "log", "content": "line two"}),
    ]

    class _FakeWS:
        def connect(self, url, **kw): return _Conn(frames)

    sys.modules["websockets"] = _FakeWS()
    try:
        async def _run():
            task = asyncio.ensure_future(ts.stream_log_lines(
                "ws://eva:8080/ws?key=x", on_line=seen.append))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        asyncio.run(_run())
    finally:
        del sys.modules["websockets"]
    assert seen == ["line one", "line two"]


def test_missing_websockets_package_is_survivable(monkeypatch):
    """The handheld's venv is not this repo's — the client must still run."""
    states = []
    monkeypatch.setitem(sys.modules, "websockets", None)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_websockets(name, *a, **kw):
        if name == "websockets":
            raise ImportError("no module named websockets")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _no_websockets)
    asyncio.run(ts.stream_log_lines("wss://x/ws", on_line=lambda _l: None,
                                    on_state=states.append))
    assert states == [False]
