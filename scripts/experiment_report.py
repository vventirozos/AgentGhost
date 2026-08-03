#!/usr/bin/env python3
"""Read the live randomized-arm results off the trajectory corpus.

The agent surfaces the same report through `introspect action='experiments'`;
this is the operator/headless entry point (and the one that can point at an
arbitrary corpus, e.g. an archived day).

Usage:
    PYTHONPATH=src python scripts/experiment_report.py
    PYTHONPATH=src python scripts/experiment_report.py --day 2026-08-05
    PYTHONPATH=src python scripts/experiment_report.py \
        --trajectories /path/to/trajectories --alpha 0.01 --json

Exit codes: 0 always (a report is not a gate). Nothing here writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core.experiments import (  # noqa: E402
    compare_arms, render_report, summarize_streaming,
)
from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402


def _default_root() -> Path:
    base = os.getenv("GHOST_HOME")
    if base:
        return Path(base) / "system" / "trajectories"
    return Path.home() / ".ghost" / "trajectories"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectories", default="",
                    help="corpus root (default $GHOST_HOME/system/trajectories)")
    ap.add_argument("--day", default="", help="restrict to one YYYY-MM-DD partition")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="confidence-sequence level (default 0.05)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable summary instead of the report")
    args = ap.parse_args()

    root = Path(args.trajectories) if args.trajectories else _default_root()
    if not root.exists():
        print(f"no trajectory corpus at {root}", file=sys.stderr)
        return 0

    collector = TrajectoryCollector(root=root, session_id="reader")
    summary, triggered, coverage = summarize_streaming(
        collector.iter_trajectories(day=args.day or None))

    if args.json:
        out = {"coverage": coverage}
        for name, arms in summary.items():
            out[name] = {
                "arms": {
                    arm: {"n": s.n, "unknown": s.unknown,
                          "means": {m: s.mean(m) for m in (s.values or {})}}
                    for arm, s in arms.items()
                },
                "comparisons": [asdict(c) | {"verdict": c.verdict}
                                for c in compare_arms(arms, alpha=args.alpha)],
                "triggered_comparisons": [
                    asdict(c) | {"verdict": c.verdict}
                    for c in compare_arms(triggered.get(name, {}),
                                          alpha=args.alpha)],
            }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(render_report(summary, alpha=args.alpha, triggered=triggered,
                        coverage=coverage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
