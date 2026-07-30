#!/usr/bin/env python3
"""TUI over the agent's in-flight turns — inspect and cancel/kill them.

    python3 scripts/ghost_kill_sessions.py [--url URL] [--key-file PATH]
                                           [--interval SECS] [--list]

Talks to the live agent's turn registry:
  GET  /api/turns        — every in-flight turn: full request_id, age,
                           RUNNING/QUEUED/cancelled, session, preview
  GET  /api/sessions     — summaries used to ENRICH each turn's session id
                           with its title, message count and last activity
                           (enrichment only — a sessions-disabled agent or
                           a failed fetch just leaves the raw ids)
  POST /api/turn/cancel  — {"request_id", "hard"}: cooperative by default
                           (stops at the next boundary, returns partial
                           work); hard cancels the asyncio task outright —
                           the guaranteed release for a turn wedged inside
                           a long upstream call. Queued turns are always
                           hard-cancelled server-side.

The 8-char tags in ghost-agent.log ("┌─ E7 e7e55cb8") are PREFIXES of the
full request ids shown here — this TUI is how you go from a log tag to a
kill without hunting for the full id.

Display: RUNNING green, QUEUED yellow, CANCELLING red. The AGE column
escalates — yellow past 2m, red past 10m — because turns are globally
serialized: one old RUNNING turn is blocking every other client. The pane
under the table shows the SELECTED turn in full: untruncated request id,
start time, the session join (title · message count · last activity), and
the whole preview. Colors degrade to bold/dim on mono terminals and are
disabled entirely by $NO_COLOR; --list prints ANSI color only to a TTY.

Keys:  ↑/↓ or j/k  select      c  cancel (cooperative)
       K           hard-kill (asks y/n)
       r           refresh now  q  quit

Auth: $GHOST_API_KEY, else --key-file (default ~/Data/AI/.ghost_api_key).
Read-only except the cancel actions you explicitly invoke. Stdlib only.
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_KEY_FILE = Path.home() / "Data" / "AI" / ".ghost_api_key"
HTTP_TIMEOUT_S = 5.0

# One wedged turn holds the global semaphore for everyone — age turns
# yellow, then red, on that clock.
AGE_WARN_S = 120
AGE_BAD_S = 600

# Sessions change far slower than turns; re-join every Nth auto-tick.
SESSIONS_EVERY_TICKS = 5


# ──────────────────────────────────────────────────────────────────────
# API client (pure helpers — unit-tested without a terminal)
# ──────────────────────────────────────────────────────────────────────

def load_api_key(key_file=None) -> str:
    """A NON-EMPTY $GHOST_API_KEY wins; else the key file; '' when neither
    yields a key (matching an agent started with auth disabled). Set-but-
    empty env must fall through — interactive shells commonly carry an
    empty export (observed live: it made this script send no key and 403)."""
    env = (os.environ.get("GHOST_API_KEY") or "").strip()
    if env:
        return env
    path = Path(key_file) if key_file else DEFAULT_KEY_FILE
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _request(url: str, key: str, body=None):
    """One JSON round-trip. Returns (payload, error_string) — exactly one
    of the two is None. Never raises: the TUI must survive a dead agent."""
    req = urllib.request.Request(url)
    if key:
        req.add_header("X-Ghost-Key", key)
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data=data,
                                    timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return None, "auth rejected (403) — check the key/--key-file"
        # /api/turn/cancel answers 404 with a JSON body ("no active turn")
        # — surface that message rather than a bare status code.
        try:
            detail = json.loads(e.read().decode("utf-8", "replace"))
            msg = detail.get("error") or detail.get("detail") or str(detail)
        except Exception:  # noqa: BLE001
            msg = ""
        return None, f"HTTP {e.code}" + (f": {msg}" if msg else "")
    except urllib.error.URLError as e:
        return None, f"agent unreachable ({getattr(e, 'reason', e)})"
    except Exception as e:  # noqa: BLE001 — malformed JSON, timeout, …
        return None, f"{type(e).__name__}: {e}"


def fetch_turns(base_url: str, key: str):
    """(turns list, error) from GET /api/turns. The endpoint returns
    {"turns": [...]} or a bare list depending on version — accept both."""
    payload, err = _request(base_url.rstrip("/") + "/api/turns", key)
    if err:
        return [], err
    if isinstance(payload, dict):
        payload = payload.get("turns") or []
    if not isinstance(payload, list):
        return [], None
    # A lying agent must not crash the TUI: keep only dict-shaped turns.
    return [t for t in payload if isinstance(t, dict)], None


def fetch_sessions(base_url: str, key: str) -> dict:
    """session_id → summary dict from GET /api/sessions. Enrichment only:
    any failure (dead agent, sessions disabled, odd shape) collapses to {}
    so the turn table still renders with raw session ids."""
    payload, err = _request(
        base_url.rstrip("/") + "/api/sessions?limit=200", key)
    if err or not isinstance(payload, dict):
        return {}
    out = {}
    for s in payload.get("sessions") or []:
        if isinstance(s, dict) and s.get("id"):
            out[str(s["id"])] = s
    return out


def cancel_turn(base_url: str, key: str, request_id: str, hard: bool):
    """(result dict, error) from POST /api/turn/cancel."""
    return _request(base_url.rstrip("/") + "/api/turn/cancel", key,
                    body={"request_id": request_id, "hard": bool(hard)})


# ──────────────────────────────────────────────────────────────────────
# Row/detail building (pure — every string the UI shows is made here)
# ──────────────────────────────────────────────────────────────────────

def fmt_age(seconds) -> str:
    # OverflowError: json.loads accepts bare Infinity, and int(inf) throws.
    try:
        s = int(float(seconds))
    except (TypeError, ValueError, OverflowError):
        return "?"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def turn_state(t: dict) -> str:
    if t.get("cancelled"):
        return "CANCELLING"
    return "RUNNING" if t.get("running") else "QUEUED"


_STATE_ROLE = {"RUNNING": "running", "QUEUED": "queued",
               "CANCELLING": "cancelling"}


def age_severity(seconds) -> str:
    """'' | 'age_warn' | 'age_bad' — the color escalation for a turn that
    has been holding (or waiting on) the global semaphore too long."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return ""
    if s >= AGE_BAD_S:
        return "age_bad"
    if s >= AGE_WARN_S:
        return "age_warn"
    return ""


def session_label(session_id, sessions: dict):
    """(display label, message-count string) for a turn's session — the
    title from the /api/sessions join when there is one, else the raw id
    prefix, else '-' for a turn that carries no session at all."""
    sid = str(session_id or "")
    if not sid:
        return "-", ""
    s = (sessions or {}).get(sid)
    if not isinstance(s, dict):
        return sid[:8], ""
    title = " ".join(str(s.get("title") or "").split())
    n = s.get("message_count")
    return (title or sid[:8]), (str(n) if isinstance(n, int) else "")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


# Fixed column budget: rid 13 + state 11 + age 7 + gap 2 + session 19
# + msgs 6 = 58; preview gets the rest.
_FIXED_W = 58


def column_header() -> str:
    return (f"{'REQUEST-ID':<13}{'STATE':<11}{'AGE':>7}  "
            f"{'SESSION':<19}{'MSGS':>4}  PREVIEW")


def row_cells(t: dict, width: int, sessions: dict = None):
    """One table row as (text, role) segments. Roles are resolved to
    curses attrs (TUI) or ANSI codes (--list) by the caller."""
    rid = str(t.get("request_id") or "?")
    state = turn_state(t)
    label, msgs = session_label(t.get("session_id"), sessions or {})
    preview = " ".join(str(t.get("preview") or "").split())
    preview_w = max(10, width - _FIXED_W)
    return [
        (f"{rid[:12]:<13}", "plain"),
        (f"{state:<11}", _STATE_ROLE[state]),
        (f"{fmt_age(t.get('age_s')):>7}", age_severity(t.get("age_s"))
         or "plain"),
        ("  ", "plain"),
        (f"{_trunc(label, 18):<19}", "session"),
        (f"{msgs:>4}  ", "dim"),
        (preview[:preview_w], "plain"),
    ]


def render_rows(turns, width: int, sessions: dict = None):
    """Plain-text table rows (row_cells joined) — the --list body, and the
    shape the tests pin down."""
    return ["".join(text for text, _ in row_cells(t, width, sessions))
            for t in turns]


def find_turn_index(turns, request_id, fallback: int = 0) -> int:
    """Index of request_id in turns, else fallback. The selection must
    follow the TURN across refreshes, not the row number — the running
    turn finishing shifts every queued turn up one, and a cancel keyed
    off a stale index would land on the wrong request."""
    rid = str(request_id)
    for i, t in enumerate(turns):
        if str(t.get("request_id")) == rid:
            return i
    return fallback


def detail_lines(t: dict, sessions: dict, width: int, now: float = None):
    """The selected-turn pane, as 3 lines of (text, role) segments: full
    request id + timing, the session join, the untruncated preview."""
    now = time.time() if now is None else now
    rid = str(t.get("request_id") or "?")
    age = t.get("age_s")
    # localtime() raises OverflowError/OSError on out-of-range timestamps
    # (a corrupt age_s must not kill the draw loop).
    try:
        started = time.strftime("%H:%M:%S", time.localtime(now - float(age)))
    except (TypeError, ValueError, OverflowError, OSError):
        started = "?"
    line1 = [("id ", "dim"), (rid, "accent"),
             ("   started ", "dim"), (started, "plain"),
             ("   age ", "dim"), (fmt_age(age), age_severity(age) or "plain")]
    if t.get("cancelled"):
        line1.append(("   cancel requested — waiting on a boundary",
                      "cancelling"))

    sid = str(t.get("session_id") or "")
    if not sid:
        line2 = [("session ", "dim"), ("- (turn carries no session id)",
                                       "dim")]
    else:
        line2 = [("session ", "dim"), (sid, "session")]
        s = (sessions or {}).get(sid)
        if isinstance(s, dict):
            title = " ".join(str(s.get("title") or "").split())
            if title:
                line2.append((f'  “{title}”', "plain"))
            n = s.get("message_count")
            if isinstance(n, int):
                line2.append((f"  · {n} msg{'s' if n != 1 else ''}", "dim"))
            upd = s.get("updated_at")
            if isinstance(upd, (int, float)) and upd > 0:
                line2.append((f"  · active {fmt_age(now - upd)} ago", "dim"))
        else:
            line2.append(("  (not in /api/sessions — brand new, or sessions"
                          " disabled)", "dim"))

    preview = " ".join(str(t.get("preview") or "").split())
    line3 = [("» ", "dim"),
             (_trunc(preview, max(10, width - 4)) or "(no preview)", "plain")]
    return [line1, line2, line3]


# ──────────────────────────────────────────────────────────────────────
# ANSI (--list) coloring
# ──────────────────────────────────────────────────────────────────────

_ANSI = {
    "running": "\x1b[1;32m", "queued": "\x1b[33m", "cancelling": "\x1b[1;31m",
    "age_warn": "\x1b[33m", "age_bad": "\x1b[1;31m",
    "session": "\x1b[36m", "accent": "\x1b[1;36m",
    "err": "\x1b[1;31m", "ok": "\x1b[32m", "dim": "\x1b[2m",
    "colhead": "\x1b[1m",
}
_ANSI_RESET = "\x1b[0m"


def ansi_enabled(stream=None) -> bool:
    """Color only on a real terminal, never under $NO_COLOR or TERM=dumb."""
    stream = sys.stdout if stream is None else stream
    return (bool(getattr(stream, "isatty", lambda: False)())
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM", "") != "dumb")


def paint(segments, color: bool) -> str:
    """(text, role) segments → one printable string; ANSI-wrapped per
    segment when color is on, byte-identical to the plain join when off."""
    if not color:
        return "".join(text for text, _ in segments)
    out = []
    for text, role in segments:
        code = _ANSI.get(role)
        out.append(f"{code}{text}{_ANSI_RESET}" if code else text)
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────
# TUI
# ──────────────────────────────────────────────────────────────────────

def _init_palette() -> dict:
    """role → curses attr. Mono terminals (and $NO_COLOR) get a bold/dim
    rendering of the same roles; init_pair failures fall back silently."""
    p = {
        "plain": curses.A_NORMAL,
        "header": curses.A_REVERSE | curses.A_BOLD,
        "colhead": curses.A_BOLD,
        "running": curses.A_BOLD, "queued": curses.A_NORMAL,
        "cancelling": curses.A_BOLD,
        "age_warn": curses.A_BOLD, "age_bad": curses.A_BOLD,
        "session": curses.A_NORMAL, "accent": curses.A_BOLD,
        "err": curses.A_BOLD, "ok": curses.A_NORMAL, "dim": curses.A_DIM,
    }
    if os.environ.get("NO_COLOR"):
        return p
    try:
        if not curses.has_colors():
            return p
        curses.start_color()
        curses.use_default_colors()

        def pair(n, fg, bg=-1):
            curses.init_pair(n, fg, bg)
            return curses.color_pair(n)

        p["running"] = pair(1, curses.COLOR_GREEN) | curses.A_BOLD
        p["ok"] = pair(1, curses.COLOR_GREEN)
        p["queued"] = pair(2, curses.COLOR_YELLOW)
        p["age_warn"] = pair(2, curses.COLOR_YELLOW)
        p["cancelling"] = pair(3, curses.COLOR_RED) | curses.A_BOLD
        p["age_bad"] = pair(3, curses.COLOR_RED) | curses.A_BOLD
        p["err"] = pair(3, curses.COLOR_RED) | curses.A_BOLD
        p["session"] = pair(4, curses.COLOR_CYAN)
        p["accent"] = pair(4, curses.COLOR_CYAN) | curses.A_BOLD
        p["header"] = pair(5, curses.COLOR_BLACK,
                           curses.COLOR_CYAN) | curses.A_BOLD
    except curses.error:
        pass
    return p


def _draw_line(stdscr, y, segments, palette, width, extra=0, fill_role=None):
    """Paint one line of (text, role) segments, clipped at width-1. Never
    raises — a resize mid-draw must not kill the TUI. fill_role pads the
    remainder (selection bar / header band)."""
    x = 0
    for text, role in segments:
        if x >= width - 1:
            break
        try:
            stdscr.addnstr(y, x, text, width - 1 - x,
                           palette.get(role, curses.A_NORMAL) | extra)
        except curses.error:
            break
        x += len(text)
    if fill_role is not None and x < width - 1:
        try:
            stdscr.addnstr(y, x, " " * (width - 1 - x), width - 1 - x,
                           palette.get(fill_role, curses.A_NORMAL) | extra)
        except curses.error:
            pass


_KEYBAR = [(" ↑/↓ ", "accent"), ("select", "dim"),
           ("   c ", "accent"), ("cancel", "dim"),
           ("   K ", "accent"), ("hard-kill", "dim"),
           ("   r ", "accent"), ("refresh", "dim"),
           ("   q ", "accent"), ("quit", "dim")]


def _tui(stdscr, base_url: str, key: str, interval: float):
    palette = _init_palette()
    curses.curs_set(0)
    stdscr.timeout(int(interval * 1000))   # getch doubles as refresh tick
    selected, top = 0, 0
    status = ("loading…", "dim")
    turns, fetch_err = [], None
    sessions: dict = {}
    ticks = 0

    def refresh(with_sessions=True):
        nonlocal turns, fetch_err, sessions, selected
        prev_rid = (turns[selected].get("request_id")
                    if 0 <= selected < len(turns) else None)
        turns, fetch_err = fetch_turns(base_url, key)
        if prev_rid is not None:
            selected = find_turn_index(
                turns, prev_rid,
                fallback=min(selected, max(0, len(turns) - 1)))
        if with_sessions:
            sessions = fetch_sessions(base_url, key)

    refresh()
    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        # Header band + column header.
        running_n = sum(1 for t in turns if t.get("running"))
        hdr = (f" ghost turns · {base_url} · {len(turns)} in flight "
               f"({running_n} running) · {time.strftime('%H:%M:%S')}")
        _draw_line(stdscr, 0, [(hdr, "header")], palette, w,
                   fill_role="header")
        _draw_line(stdscr, 1, [(column_header(), "colhead")], palette, w)

        # Table (scrolls) + selected-turn detail pane.
        show_detail = bool(turns) and not fetch_err and h >= 13
        detail_h = 4 if show_detail else 0
        list_h = max(1, h - 2 - detail_h - 2)

        if fetch_err:
            _draw_line(stdscr, 3, [("  ⚠ " + fetch_err, "err")], palette, w)
        elif not turns:
            _draw_line(stdscr, 3,
                       [("  ✓ no turns in flight — the agent is idle", "ok")],
                       palette, w)
        else:
            selected = max(0, min(selected, len(turns) - 1))
            if selected < top:
                top = selected
            elif selected >= top + list_h:
                top = selected - list_h + 1
            top = max(0, min(top, max(0, len(turns) - list_h)))
            for i, t in enumerate(turns[top:top + list_h]):
                is_sel = (top + i == selected)
                _draw_line(stdscr, 2 + i, row_cells(t, w, sessions), palette,
                           w, extra=curses.A_REVERSE if is_sel else 0,
                           fill_role="plain" if is_sel else None)
            if show_detail:
                t = turns[selected]
                y0 = 2 + list_h
                rid12 = str(t.get("request_id") or "?")[:12]
                _draw_line(stdscr, y0,
                           [(f"─ {rid12} " + "─" * max(0, w - len(rid12) - 4),
                             "dim")], palette, w)
                for j, segs in enumerate(detail_lines(t, sessions, w)):
                    _draw_line(stdscr, y0 + 1 + j, segs, palette, w)

        _draw_line(stdscr, h - 2, _KEYBAR, palette, w, fill_role="dim")
        _draw_line(stdscr, h - 1, [(" " + status[0], status[1])], palette, w)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == -1:                                   # tick → auto-refresh
            ticks += 1
            refresh(with_sessions=(ticks % SESSIONS_EVERY_TICKS == 0))
            continue
        if ch in (ord("q"), 27):
            return
        if ch == ord("r"):
            refresh()
            status = ("refreshed", "ok")
            continue
        if ch in (curses.KEY_UP, ord("k")):
            selected = max(0, selected - 1)
            continue
        if ch in (curses.KEY_DOWN, ord("j")):
            selected = min(max(0, len(turns) - 1), selected + 1)
            continue
        if ch in (ord("c"), ord("K")) and turns:
            t = turns[selected]
            rid = str(t.get("request_id") or "")
            hard = (ch == ord("K"))
            if hard:
                _draw_line(stdscr, h - 1,
                           [(f" hard-kill {rid[:12]}? (y/n) ", "cancelling")],
                           palette, w, extra=curses.A_REVERSE,
                           fill_role="cancelling")
                stdscr.refresh()
                stdscr.timeout(-1)                     # block for the answer
                confirm = stdscr.getch()
                stdscr.timeout(int(interval * 1000))
                if confirm not in (ord("y"), ord("Y")):
                    status = ("hard-kill aborted", "dim")
                    continue
            result, err = cancel_turn(base_url, key, rid, hard)
            if err:
                status = (f"{rid[:12]}: {err}", "err")
            else:
                status = (f"{rid[:12]}: cancelled "
                          f"({result.get('mode', '?')})", "ok")
            refresh()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TUI over the agent's in-flight turns (list + kill).")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"agent base URL (default {DEFAULT_URL})")
    ap.add_argument("--key-file", default=None,
                    help=f"API key file (default {DEFAULT_KEY_FILE}; "
                         f"$GHOST_API_KEY wins)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="auto-refresh seconds (default 2)")
    ap.add_argument("--list", action="store_true",
                    help="print the turns once (no TUI) and exit")
    args = ap.parse_args()

    key = load_api_key(args.key_file)
    if args.list:
        turns, err = fetch_turns(args.url, key)
        if err:
            print(paint([(err, "err")], ansi_enabled(sys.stderr)),
                  file=sys.stderr)
            return 1
        color = ansi_enabled()
        if not turns:
            print(paint([("no turns in flight", "ok")], color))
            return 0
        sessions = fetch_sessions(args.url, key)
        print(paint([(column_header(), "colhead")], color))
        for t in turns:
            print(paint(row_cells(t, 120, sessions), color))
        return 0

    try:
        curses.wrapper(_tui, args.url, key, max(0.5, args.interval))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
