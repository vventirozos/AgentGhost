"""Live turn status for the uConsole client — what the agent is doing *now*.

The waiting bubble used to read ``cogitating ···`` for the whole 30-90 s a turn
takes, which says nothing. This module gives it the same caption the web UI
shows under its waiting bubble: an elapsed clock, a plain-English description of
the step the agent is on, and that step's icon::

    0:34   reading a file · notes.md   📖

**Where the data comes from.** The agent's ``/api/chat`` stream carries content
tokens only — it says nothing about tools, memory or thinking, so the client
cannot learn any of this from the reply it is already reading. The web UI gets
it from a SECOND channel: the interface server (port 8080) tails the agent's
pretty log and rebroadcasts every line over a websocket, and ``app.js`` turns
those lines into captions. This module is the Python half of exactly that —
:func:`stream_log_lines` is the socket, :class:`TurnTicker` is the caption logic.

**Corridor filtering.** Background work (self-play, dreams, other clients)
streams over the same socket with its own request ids. The ticker adopts the id
of the FIRST ``request started`` corridor that opens after the operator sends,
and ignores every line that does not carry it — otherwise the handheld would
narrate the agent's dreams while the operator waits on their own question.

**Kept in sync by hand.** ``ICON_CLASS`` / ``TICKER_VERBS`` / ``TICKER_NOISE``
are ports of the tables in ``interface/static/app.js`` (which themselves mirror
``utils/logging.py``'s ``Icons``). Nothing at runtime can catch drift — the
device never sees app.js — so ``tests/test_clockwork_turnstatus.py`` parses BOTH
files and fails when they diverge. Update the JS and the Python together.
"""

from __future__ import annotations

import asyncio
import json
import re
import ssl
import time
from urllib.parse import quote

# CSI colour/cursor sequences from the agent's pretty stream.
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Body lines are column-formatted with 2+ spaces between fields:
#   │  99  📖  +8.04s  file read           notes.md
_GAP_RE = re.compile(r"\s{2,}")
# Header:  ┌─ 99 9961f364  request started  15:16:26 ────────────
# The SHORT id (`99`) is what body lines carry, so that is what is captured.
_HEADER_ID_RE = re.compile(r"^\S+\s+(\S+)\s")
_DELTA_RE = re.compile(r"^\+[\d.\s]*m?s$")

# ── icon → priority class (port of app.js ICON_CLASS) ───────────────────────
ICON_CLASS = {
    # --- lifecycle / status (accent) ---
    "⚡": "accent",
    "🚀": "accent",
    "🌅": "accent",
    "💤": "idle",
    "😴": "idle",
    "🎬": "accent",
    "🏁": "accent",
    "⏳": "accent",
    "✅": "accent",
    "❌": "accent",
    "🔶": "accent",
    "🛑": "accent",
    "🔄": "accent",
    "💡": "accent",
    "🐛": "accent",
    "🔒": "accent",
    "📣": "accent",
    "🪞": "accent",
    "🎓": "accent",
    "👍": "accent",   # FEEDBACK_POS — human outcome label (/api/feedback)
    "👎": "accent",   # FEEDBACK_NEG

    # --- tools (external action) ---
    "🌐": "tool",
    "🌎": "tool",
    "🧅": "tool",
    "🔬": "tool",
    "🐍": "tool",
    "🐚": "tool",
    "🧰": "tool",
    "📥": "tool",
    "🎨": "tool",
    "📄": "tool",
    "🎮": "tool",
    "🎲": "tool",
    "🧪": "tool",
    "📦": "tool",
    "🏃": "tool",   # JOB_PROMOTE — long sandbox command detached as a job
    "🔧": "tool",
    "🛸": "tool",

    # --- memory / filesystem ---
    "💾": "memory",
    "📖": "memory",
    "📙": "memory",   # TOOL_FILE_M — multi-path batch read
    "🔍": "memory",
    "👀": "memory",
    "📝": "memory",
    "🔎": "memory",
    "📍": "memory",
    "📚": "memory",
    "📑": "memory",
    "🔻": "memory",
    "🧬": "memory",
    "🧮": "memory",
    "🧹": "memory",
    "📔": "memory",
    "💪": "memory",
    "📇": "memory",
    "📓": "memory",
    "🪢": "memory",
    "🧾": "memory",
    "📶": "memory",
    "🎥": "memory",
    "📡": "memory",
    "🔀": "memory",
    "👤": "memory",
    "🐘": "memory",

    # --- planning / routing ---
    "📋": "plan",
    "🧩": "plan",
    "🧭": "plan",
    "🎯": "plan",
    "🌳": "plan",
    "🔮": "plan",
    "🪄": "plan",
    "🌀": "plan",
    "🦋": "plan",
    "🚜": "plan",   # GEPA_AUTONOMY — §4DC supply watch / live judge
    "🔗": "plan",

    # --- raw thinking (the floor) ---
    "🧠": "think",
    "💭": "think",
    "💬": "think",
    "🤖": "think",

    # --- idle / skipped ---
    "⏩": "idle",
    "🌙": "plan",
    "🫀": "idle",

    # --- metacognition (§LOG-7 — mirrors app.js) ---
    "🫧": "plan",     # METACOG (generic uplift)
    "🌱": "plan",     # METACOG_BOOT
    "📊": "plan",     # METACOG_SUMMARY
    "📈": "plan",     # METACOG_CONF
    "📐": "plan",     # METACOG_CALIB
    "🥇": "plan",     # METACOG_ARBITER
    "🚧": "plan",     # METACOG_VALID
    "💻": "plan",     # METACOG_HOST
    "🚦": "plan",     # METACOG_REPLAN
    "🚪": "plan",     # METACOG_GATE

    # --- misc ---
    "🫥": "accent",
    "🔥": "accent",
}

# Terminal-bookkeeping glyphs (human feedback receipts) that must never
# read as turn activity. Mirrors app.js _NON_WORKING_ICONS — drift-pinned
# by tests/test_clockwork_turnstatus.py.
NON_WORKING_ICONS = {"👍", "👎"}

# Titles that narrate plumbing, not progress — never shown.
TICKER_NOISE = {
    "prefill cache", "memory bus", "llm request",
    "constraint check", "metacog conf", "turn outcome", "agent parser",
}

# Friendly phrasings for the most common step titles; anything not listed
# falls back to the raw title (already short and readable).
TICKER_VERBS = {
    "worker compute": "delegating to the worker node",
    "sandbox tree": "scanning the workspace",
    "sandbox exec": "running a command",
    "execution task": "executing code",
    "file read": "reading a file",
    "file write": "writing a file",
    "web search": "searching the web",
    "web read": "reading a page",
    "browser": "driving the browser",
    "verifier": "verifying the answer",
    "memory search": "recalling memories",
    "memory save": "saving a memory",
    "graph updated": "updating the knowledge graph",
    "belief revision": "revisiting a belief",
    "hydrated context": "gathering context",
    "system weather": "checking the weather",
    "vision": "looking at an image",
    "delegation": "delegating a subtask",
}

# Shown before the first corridor line lands. "starting…" mirrors the web UI;
# the old "cogitating" is kept for the case where the log stream is DOWN, so an
# unreachable interface degrades to the previous behaviour instead of lying
# about a step that will never be described.
DESC_STARTING = "starting…"
DESC_OFFLINE = "cogitating"
ICON_STARTING = "⏳"
ICON_THINKING = "💭"

# Extended_Pictographic, hand-rolled: Python's `re` has no \p{...}, and the
# `regex` module is not installed on the handheld. Ranges are deliberately
# TIGHT around the pictographic blocks — box-drawing (U+2500–U+257F, the
# corridor rails `│ ┌ ─`) and arrows (U+2190–U+21FF, the `→` in log content)
# must NOT match, or every line would "contain an icon".
_PICTOGRAPHIC_RANGES = (
    (0x00A9, 0x00A9), (0x00AE, 0x00AE),
    (0x203C, 0x203C), (0x2049, 0x2049),
    (0x2122, 0x2122), (0x2139, 0x2139),
    (0x231A, 0x231B), (0x2328, 0x2328),
    (0x23CF, 0x23CF), (0x23E9, 0x23FA),
    (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B00, 0x2BFF),
    (0x1F000, 0x1FAFF),
)


def _is_pictographic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _PICTOGRAPHIC_RANGES)


def clean_log_line(raw) -> str:
    """Strip ANSI colour codes and trailing whitespace (app.js cleanLogLine)."""
    return ANSI_ESCAPE_RE.sub("", str(raw or "")).rstrip()


def extract_icon(line: str):
    """First pictographic character in the line, or None."""
    for ch in line:
        if _is_pictographic(ch):
            return ch
    return None


def icon_class(icon: str) -> str:
    """Priority class for an icon. Unknown icons are raw thought (the floor)."""
    return ICON_CLASS.get(icon, "think")


class TurnTicker:
    """Caption state for one in-flight turn.

    Fed :meth:`note_line` for every log line the socket delivers; the client
    reads :attr:`icon`, :attr:`desc` and :meth:`clock_text` to draw. Every
    mutator returns True when the visible caption actually changed, so the
    client can skip a repaint (each one re-measures and re-fits a QLabel, which
    is not free on the CM4) for the many lines that change nothing.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._t0 = 0.0
        self.active = False
        self.connected = False
        self.req_id = None
        self.icon = ICON_STARTING
        self.desc = DESC_OFFLINE

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        """A turn just went out; reset and begin counting."""
        self.active = True
        self.req_id = None
        self._t0 = self._clock()
        self.icon = ICON_STARTING
        self.desc = DESC_STARTING if self.connected else DESC_OFFLINE

    def stop(self) -> None:
        self.active = False
        self.req_id = None

    def set_connected(self, ok: bool) -> bool:
        """Note whether the log socket is up. Returns True if the caption moved."""
        ok = bool(ok)
        changed = ok != self.connected
        self.connected = ok
        # Only the pre-adoption placeholder depends on this: once a corridor is
        # adopted the last real step stays on screen, which is more useful than
        # reverting to a placeholder because the socket blinked.
        if changed and self.active and self.req_id is None:
            self.desc = DESC_STARTING if ok else DESC_OFFLINE
            return True
        return False

    def clock_text(self) -> str:
        secs = max(0, int(self._clock() - self._t0))
        return f"{secs // 60}:{secs % 60:02d}"

    # ── the corridor parser (port of app.js noteTickerLine) ──────────────
    def note_line(self, raw) -> bool:
        if not self.active:
            return False
        clean = clean_log_line(raw)

        # Corridor adoption: the first corridor that OPENS after our send. A
        # corridor already open when we sent can never be adopted.
        if self.req_id is None:
            if "request started" in clean:
                m = _HEADER_ID_RE.match(clean)
                if m:
                    self.req_id = m.group(1)
            return False

        # Body lines: │  <id>  <icon>  <+delta>  <title>  <content…>
        parts = [p.strip() for p in _GAP_RE.split(clean) if p.strip()]
        if len(parts) < 3 or parts[1] != self.req_id:
            return False
        icon = extract_icon(clean)
        if not icon:
            return False
        cls = icon_class(icon)
        if cls == "think":
            # Long reasoning stretches between tools: say so instead of leaving
            # the last tool's caption to go stale.
            return self._set(ICON_THINKING, "thinking…")
        if cls not in ("tool", "memory", "plan", "accent"):
            return False
        # Feedback receipts (👍/👎) are terminal bookkeeping, not turn
        # activity — mirrors the web client's exclusion so the uConsole
        # ticker doesn't read out a label left on an old reply.
        if icon in NON_WORKING_ICONS:
            return False

        icon_idx = next((i for i, p in enumerate(parts) if icon in p), -1)
        if icon_idx < 0:
            return False
        ti = icon_idx + 1
        if ti < len(parts) and _DELTA_RE.match(parts[ti]):
            ti += 1
        title = (parts[ti] if ti < len(parts) else "").lower()[:22].strip()
        if not title or title in TICKER_NOISE:
            return False
        desc = TICKER_VERBS.get(title, title)
        # Specificity when the line carries it ("reading a file · notes.md").
        detail = " ".join(parts[ti + 1:]).strip()
        if detail:
            desc += f" · {detail[:30]}" + ("…" if len(detail) > 30 else "")
        return self._set(icon, desc)

    def _set(self, icon: str, desc: str) -> bool:
        if (icon, desc) == (self.icon, self.desc):
            return False
        self.icon, self.desc = icon, desc
        return True


# ── rendering ───────────────────────────────────────────────────────────────

# Measured on the device: past ~46 characters a caption wraps in a 0.56-width
# bubble at 20px. The web UI gets one-lining from `white-space: nowrap` +
# `text-overflow: ellipsis`; a QLabel has no equivalent, and a caption that
# wraps also changes the bubble's HEIGHT every few seconds, which reads as the
# transcript twitching.
CAPTION_DESC_CHARS = 46


def caption_html(ticker, dim: str, mono: str, max_chars: int = CAPTION_DESC_CHARS) -> str:
    """The waiting bubble's caption: ``0:34   reading a file · notes.md   📖``.

    Field order mirrors the web UI's turn-status line (clock, description,
    icon) so the two clients read the same way; spacing does the separating —
    the literal ``:``/``-`` punctuation the web tried first read as soup.

    Lives here, away from Qt, so the escaping and the elide are unit-testable:
    the description is LOG CONTENT, and an unescaped ``<`` from a tool's output
    would be swallowed as rich-text markup by the QLabel that draws this.
    """
    desc = ticker.desc if len(ticker.desc) <= max_chars \
        else ticker.desc[:max_chars - 1] + "…"
    desc = (desc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (f"<span style='color:{dim};'>"
            f"<span style='font-family:{mono}; font-size:16px;'>"
            f"{ticker.clock_text()}</span>"
            f"&nbsp;&nbsp;<i>{desc}</i>"
            f"&nbsp;&nbsp;{ticker.icon}</span>")


# ── the log socket ──────────────────────────────────────────────────────────

def log_ws_url(host: str, key: str, port: int = 8080, scheme: str = "wss") -> str:
    """URL of the interface server's log broadcast.

    The key rides in the QUERY STRING, not a header: the endpoint is gated the
    same way the web UI is, and the browser client cannot set custom headers on
    a WebSocket — so `?key=` is the only dialect the server accepts.
    """
    # safe="" — quote() leaves `/` alone by default, which would break a key
    # containing one against the server's exact-match compare.
    return f"{scheme}://{host}:{port}/ws?key={quote(key, safe='')}"


async def stream_log_lines(url, on_line, on_state=None, verify_tls=False,
                           backoff_max=30.0, connect_kwargs=None):
    """Forever-loop: read the interface's log broadcast, call ``on_line(text)``.

    Reconnects with exponential backoff and NEVER raises — a dead interface
    must cost the operator a status caption, not the client. ``on_state(bool)``
    (optional) is called on every connect/disconnect so the caption can fall
    back to the offline placeholder.

    The import is lazy so a device without the `websockets` package still runs
    the client: the status line simply stays on its offline placeholder.
    """
    try:
        import websockets
    except ImportError:
        if on_state:
            on_state(False)
        return

    ssl_ctx = None
    if url.startswith("wss://") and not verify_tls:
        # The interface serves a self-signed cert and the hop is inside the
        # tailnet (WireGuard-authenticated), exactly like the voice endpoints
        # this client already calls with verification off.
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    kwargs = {"ping_interval": 20, "ping_timeout": 20, "open_timeout": 10}
    kwargs.update(connect_kwargs or {})
    if ssl_ctx is not None:
        kwargs["ssl"] = ssl_ctx

    delay = 1.0
    while True:
        try:
            async with websockets.connect(url, **kwargs) as ws:
                delay = 1.0
                if on_state:
                    on_state(True)
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(data, dict) and data.get("type") == "log":
                        on_line(data.get("content", ""))
        except asyncio.CancelledError:
            if on_state:
                on_state(False)
            raise
        except Exception:  # noqa: BLE001 — connect refused, TLS, protocol, …
            pass
        if on_state:
            on_state(False)
        await asyncio.sleep(delay)
        delay = min(delay * 2, backoff_max)
