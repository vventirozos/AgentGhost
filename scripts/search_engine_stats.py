#!/usr/bin/env python3
"""§4UN-onion / §4UN-read: per-engine search outcomes and web_search page reads, from the dated agent log.

Reads the durable log (``$GHOST_HOME/system/ghost-agent.log``, default ``~/Data/AI/Data/system/ghost-agent.log``,
plus its ``.1`` rotation when present). Its lines are day-stamped::

    2026-11-13 10:00:00 - GhostStream - INFO - [abc +1.0s] web search — yandex won wave 0 … | engines: duckduckgo=blocked yandex=win …
    2026-11-13 10:00:06 - GhostStream - INFO - [abc +7.0s] search page read — read python.org in 900 ms — …

Every search wave's line ends with `` | engines: <engine>=<outcome> …`` (win, empty, blocked, timeout,
conn-error, off-topic, low-relevance, error, abandoned — "late" in older logs — unfinished); every page web_search opened logs one
``search page read — <outcome> …`` line (read, failed, timeout, thin, off-topic).

    python scripts/search_engine_stats.py --days 14
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) \d{2}:\d{2}:\d{2} ")
_ENGINES_RE = re.compile(r" \| engines: ((?:[\w.-]+=[\w-]+ ?)+)\s*$")
_PAGE_READ_RE = re.compile(r"\bsearch page read — ([\w-]+) ")


def default_logs() -> List[str]:
    home = os.environ.get("GHOST_HOME") or os.path.expanduser("~/Data/AI/Data")
    base = os.path.join(home, "system", "ghost-agent.log")
    return [p for p in (base + ".1", base) if os.path.exists(p)]


def tally(paths: Iterable[str], since: Optional[dt.date] = None) -> Dict:
    """{"engines": {engine: {outcome: n}}, "page_reads": {outcome: n}, "waves": n} for lines on or after
    ``since``."""
    engines: Dict[str, Dict[str, int]] = {}
    reads: Dict[str, int] = {}
    waves = 0
    for path in paths:
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                m = _DATE_RE.match(line)
                if not m:
                    continue
                if since is not None and dt.date(int(m[1]), int(m[2]), int(m[3])) < since:
                    continue
                e = _ENGINES_RE.search(line)
                if e:
                    waves += 1
                    for tok in e.group(1).split():
                        name, _, outcome = tok.partition("=")
                        d = engines.setdefault(name, {})
                        d[outcome] = d.get(outcome, 0) + 1
                    continue
                r = _PAGE_READ_RE.search(line)
                if r:
                    reads[r.group(1)] = reads.get(r.group(1), 0) + 1
    return {"engines": engines, "page_reads": reads, "waves": waves}


def render(stats: Dict) -> str:
    kinds = ["win", "empty", "blocked", "timeout", "conn-error", "abandoned", "late", "unfinished", "off-topic",
             "low-relevance", "error"]
    out = [f"waves: {stats['waves']}", "",
           f"{'engine':<20}" + "".join(f"{k:>14}" for k in kinds) + f"{'win %':>8}"]
    for name in sorted(stats["engines"], key=lambda n: -stats["engines"][n].get("win", 0)):
        d = stats["engines"][name]
        total = sum(d.values())
        out.append(f"{name:<20}" + "".join(f"{d.get(k, 0):>14}" for k in kinds)
                   + f"{(100.0 * d.get('win', 0) / total if total else 0):>7.1f}%")
    out += ["", "page reads: " + (", ".join(f"{k}={v}" for k, v in sorted(stats["page_reads"].items()))
                                  or "none")]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="lines from the last N days (default 14)")
    ap.add_argument("--log", action="append", help="log file(s); default: the dated agent log and its .1")
    a = ap.parse_args(argv)
    since = dt.date.today() - dt.timedelta(days=max(0, a.days - 1))
    print(render(tally(a.log or default_logs(), since=since)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
