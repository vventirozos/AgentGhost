#!/usr/bin/env python3
"""How far along is a long run? — READ the answer, never derive it.

    python scripts/runstatus.py <progress.json> [--watch]

The whole point is that this tool does exactly one thing: read the file the
run itself writes. It does not count cache entries, parse logs, or infer
position from call-shape ratios. On 2026-08-09 every one of those inference
methods was tried on a 90-minute bench and every one gave a wrong number,
because each was reconstructing a fact the run already knew.

It reports four states honestly, and three of them are not "a percentage":

    running   — position and a MEASURED rate
    finished  — done
    STALLED   — the file stopped moving; do NOT read `done` as current
    missing   — position UNKNOWN. Not zero, not "probably fine" — unknown.

`missing` and `STALLED` are first-class answers. A status tool whose only
vocabulary is percentages will always find a percentage to report, which is
how a wedged run reads as a healthy one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.eval.runprogress import read_progress, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--watch", type=float, nargs="?", const=30.0, default=0.0,
                    help="re-read every N seconds (default 30)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    while True:
        if args.json:
            import json
            print(json.dumps(read_progress(args.path), indent=1))
        else:
            print(render(args.path))
        if not args.watch:
            break
        b = read_progress(args.path)
        if b.get("status") in ("finished", "missing", "unreadable"):
            break
        time.sleep(args.watch)

    st = read_progress(args.path).get("status")
    # Exit codes so a script can branch without parsing prose.
    return {"running": 0, "finished": 0, "STALLED": 3,
            "missing": 2, "unreadable": 2}.get(st, 2)


if __name__ == "__main__":
    sys.exit(main())
