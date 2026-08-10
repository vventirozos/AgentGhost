#!/usr/bin/env python3
"""Read the agent's log the way a human wants to read it.

WHY THIS EXISTS (2026-08-09 log audit). `$GHOST_HOME/system/ghost-agent.log`
is deliberately a COMPLETE record — every mirror line, including DEBUG, so a
turn can be reconstructed after a restart and instruments can count events.
That is the right design for an archive and the wrong one for monitoring:
measured on a 4000-line window, **1652 lines (41%) were a single repeated
string** — `critic compute — Routing verification to Critic Node (Nova)` —
while only **6 lines carried a verdict**. The log announced intent 275 times
for every outcome it reported.

Two fixes were needed and this is the second. The first was at the source:
background plumbing (self-play, REM, failure-dimension tagging) now logs at
DEBUG, so it stays in the archive and leaves the operator's view. This tool
is the lens that view needs.

    ghostlog.py                 # INFO+, repeats collapsed — ambient health
    ghostlog.py -f              # follow
    ghostlog.py --req 178defd6  # ONE request, in full, DEBUG included
    ghostlog.py --since 30      # last 30 minutes
    ghostlog.py --all           # nothing hidden (the raw archive)

`--req` is the "how was this request actually processed" view: every line
that carries the request id, in order, at every level — sub-steps, tool
calls, verifier activity, timings — without the surrounding background noise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 2026-08-09 20:04:12 - GhostStream - INFO - [178defd6 +45.7s] request finished — …
_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - (?P<logger>\S+) - "
    r"(?P<level>[A-Z]+) - (?:\[(?P<tag>[^\]]*)\] )?(?P<msg>.*)$")

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_C = {"DEBUG": "\033[2m", "INFO": "", "WARNING": "\033[33m",
      "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
_RESET = "\033[0m"
_DIM = "\033[2m"


def _color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _default_log() -> Path:
    home = os.getenv("GHOST_HOME", "")
    return Path(home) / "system" / "ghost-agent.log"


def _parse(line: str):
    m = _LINE.match(line.rstrip("\n"))
    if not m:
        return None
    d = m.groupdict()
    tag = (d.get("tag") or "").strip()
    # "[178defd6 +45.7s]" → req 178defd6; "[SYSTEM]" → background
    d["req"] = "" if tag in ("", "SYSTEM") else tag.split()[0]
    d["tag"] = tag
    return d


def _render(rec, show_req: bool, colour: bool, width: int = 0) -> str:
    lvl = rec["level"]
    c = _C.get(lvl, "") if colour else ""
    r = _RESET if colour and c else ""
    who = rec["req"][:8] if rec["req"] else ("·" if show_req else "")
    head = f"{rec['ts'][11:]} "
    if show_req:
        head += f"{who:<8} "
    if lvl not in ("INFO",):
        head += f"{lvl[:4]:<5}"
    msg = rec["msg"]
    if width and len(msg) > width:
        # Cut in the MIDDLE: the head says what the event is, the tail
        # usually carries the outcome. Chopping the tail loses the answer.
        keep = width // 2 - 4
        msg = f"{msg[:keep]} …{len(msg) - 2 * keep} more… {msg[-keep:]}"
    return f"{c}{head}{msg}{r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default="", help="log path (default $GHOST_HOME/system/ghost-agent.log)")
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("--req", default="", help="show ONE request in full (implies --all)")
    ap.add_argument("--since", type=float, default=0.0, help="only the last N minutes")
    ap.add_argument("--level", choices=_LEVELS, default="INFO")
    ap.add_argument("--all", action="store_true", help="include DEBUG — the raw archive")
    ap.add_argument("--no-collapse", action="store_true")
    ap.add_argument("--width", type=int, default=190,
                    help="truncate long messages (0 = never). A hydration or "
                         "monologue line can run 1500+ chars and destroy the "
                         "view it is supposed to explain.")
    ap.add_argument("-n", "--lines", type=int, default=200,
                    help="how many lines back to start from (0 = whole file)")
    args = ap.parse_args()

    path = Path(args.file) if args.file else _default_log()
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 2

    min_lvl = 0 if (args.all or args.req) else _LEVELS.index(args.level)
    cutoff = (datetime.now() - timedelta(minutes=args.since)) if args.since else None
    colour = _color()

    # Collapse state: consecutive identical (req, msg) print once with ×N.
    # This is the SAME defect the source-level fix addresses, met from the
    # reader's side — a burst that is genuinely one event should read as one.
    last = {"key": None, "n": 0, "rec": None}

    def flush():
        if last["key"] is not None and last["rec"] is not None:
            line = _render(last["rec"], not args.req, colour, args.width)
            if last["n"] > 1:
                sfx = f"  ×{last['n']}"
                line += f"{_DIM}{sfx}{_RESET}" if colour else sfx
            print(line, flush=True)
        last["key"], last["n"], last["rec"] = None, 0, None

    def handle(raw: str):
        rec = _parse(raw)
        if rec is None:
            return
        # An UNKNOWN level is never filtered out — a line we cannot classify
        # is more likely to matter than less. (The nested conditional this
        # replaces read correctly but was a trap for the next edit.)
        if rec["level"] in _LEVELS and _LEVELS.index(rec["level"]) < min_lvl:
            return
        if args.req and not rec["req"].startswith(args.req):
            return
        if cutoff:
            try:
                if datetime.strptime(rec["ts"], "%Y-%m-%d %H:%M:%S") < cutoff:
                    return
            except ValueError:
                pass
        key = (rec["req"], rec["msg"], rec["level"])
        if not args.no_collapse and key == last["key"]:
            last["n"] += 1
            return
        flush()
        last["key"], last["n"], last["rec"] = key, 1, rec

    with path.open("r", errors="ignore") as fh:
        body = fh.readlines()
        start = 0 if (args.lines == 0 or args.req or args.since) else max(0, len(body) - args.lines)
        for raw in body[start:]:
            handle(raw)
        flush()
        if args.follow:
            fh.seek(0, os.SEEK_END)
            try:
                while True:
                    raw = fh.readline()
                    if not raw:
                        flush()
                        time.sleep(0.4)
                        continue
                    handle(raw)
            except KeyboardInterrupt:
                flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
