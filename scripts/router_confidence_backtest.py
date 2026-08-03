#!/usr/bin/env python3
"""§4I Phase 2 — does router confidence actually discriminate?

The gate the depth curve had to pass, applied to the one PROSPECTIVE signal
this agent computes and never used. Buckets recorded turns by the complexity
router's turn-0 confidence and measures the ACTUAL failure rate in each.

Two outcomes, and they lead different places:

  * it discriminates  → a real prior, worth using for variance reduction
    (Phase 3) and possibly for an early gate (Phase 4, and ONLY then);
  * it is flat        → an escalation heuristic that happens to be logged.
    **The plan stops here.** That is a result, not a failure: this project's
    history is full of signals that looked usable and measured dead
    (`uncertainty_pressure`: 2 distinct values across a whole epoch;
    the competence component: separation 0.0023).

Requires Phase 1 to have been deployed long enough to stamp turns — turns
recorded before it carry no `router_confidence` and are reported separately
rather than silently dropped.

Usage:
    PYTHONPATH=src python scripts/router_confidence_backtest.py
    PYTHONPATH=src python scripts/router_confidence_backtest.py --json
    PYTHONPATH=src python scripts/router_confidence_backtest.py --min-per-bucket 40

Exit codes: 0 = discriminates; 1 = flat or insufficient data; 2 = no corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core.experiments import asymp_cs_radius  # noqa: E402
from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402

# Confidence buckets. The router's own escalation threshold is 0.30, so that
# boundary is kept as a bucket edge — if the signal is real, the escalate/
# don't-escalate split should be visible right there.
BUCKETS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.75), (0.75, 1.01))

# A bucket below this many RESOLVED turns is reported but never used for the
# verdict — the same "no verdict on thin data" discipline the experiment
# analysis uses.
DEFAULT_MIN_PER_BUCKET = 30

# The spread (worst bucket minus best bucket failure rate) below which the
# signal is called flat. 0.10 is deliberately modest: the depth curve spans
# 0.178 → 0.606, and anything under 10 points is not worth a behaviour change.
DISCRIMINATION_THRESHOLD = 0.10


def _default_root() -> Path:
    base = os.getenv("GHOST_HOME")
    if base:
        return Path(base) / "system" / "trajectories"
    return Path.home() / "ghost_llamacpp" / "system" / "trajectories"


def _bucket_of(conf: float):
    for lo, hi in BUCKETS:
        if lo <= conf < hi:
            return (lo, hi)
    return None


def collect(trajectories, *, min_per_bucket: int):
    """→ (per_bucket stats, coverage counters). One streaming pass."""
    stats = defaultdict(lambda: {"n": 0, "failed": 0, "unknown": 0,
                                 "outcomes": []})
    cov = {"user_turns": 0, "stamped": 0, "resolved": 0}
    for traj in trajectories:
        try:
            if str(getattr(traj, "task_kind", "") or "") != "user_request":
                continue
            cov["user_turns"] += 1
            extra = getattr(traj, "extra", None) or {}
            if not isinstance(extra, dict) or "router_confidence" not in extra:
                continue
            conf = float(extra["router_confidence"])
            key = _bucket_of(conf)
            if key is None:
                continue
            cov["stamped"] += 1
            s = stats[key]
            s["n"] += 1
            outcome = str(getattr(traj, "outcome", "") or "").lower()
            if outcome == "failed":
                s["failed"] += 1
                s["outcomes"].append(1.0)
                cov["resolved"] += 1
            elif outcome == "passed":
                s["outcomes"].append(0.0)
                cov["resolved"] += 1
            else:
                s["unknown"] += 1
        except Exception:
            continue
    return stats, cov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectories", default="")
    ap.add_argument("--min-per-bucket", type=int, default=DEFAULT_MIN_PER_BUCKET)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.trajectories) if args.trajectories else _default_root()
    if not root.exists():
        print(f"no trajectory corpus at {root}", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": "corpus_missing", "root": str(root)}))
        return 2

    collector = TrajectoryCollector(root=root, session_id="reader")
    stats, cov = collect(collector.iter_trajectories(),
                         min_per_bucket=args.min_per_bucket)

    rows = []
    for key in BUCKETS:
        s = stats.get(key)
        if not s or not s["outcomes"]:
            rows.append({"bucket": f"{key[0]:.2f}-{key[1]:.2f}", "n": 0,
                         "resolved": 0, "failure_rate": None, "ci": None,
                         "usable": False})
            continue
        resolved = len(s["outcomes"])
        rate = sum(s["outcomes"]) / resolved
        rad = asymp_cs_radius(s["outcomes"], alpha=0.05)
        rows.append({
            "bucket": f"{key[0]:.2f}-{key[1]:.2f}",
            "n": s["n"], "resolved": resolved, "unknown": s["unknown"],
            "failure_rate": round(rate, 4),
            "ci": None if rad is None else round(rad, 4),
            "usable": resolved >= args.min_per_bucket,
        })

    usable = [r for r in rows if r["usable"]]
    spread = None
    verdict = "insufficient data"
    if len(usable) >= 2:
        lo = min(r["failure_rate"] for r in usable)
        hi = max(r["failure_rate"] for r in usable)
        spread = round(hi - lo, 4)
        verdict = ("DISCRIMINATES" if spread >= DISCRIMINATION_THRESHOLD
                   else "FLAT — router confidence is an escalation heuristic, "
                        "not a difficulty prior; §4I stops at Phase 2")

    if args.json:
        print(json.dumps({"coverage": cov, "buckets": rows,
                          "spread": spread, "verdict": verdict}, indent=2))
    else:
        print("§4I Phase 2 — router confidence vs actual failure rate")
        print(f"  corpus: {cov['user_turns']} user turns, "
              f"{cov['stamped']} carry a router stamp, "
              f"{cov['resolved']} of those resolved pass/fail")
        if cov["stamped"] == 0:
            print("  ⚠ NO stamped turns — Phase 1 either has not deployed or "
                  "has regressed. Nothing to measure yet.")
        print()
        print(f"  {'confidence':<14}{'n':>6}{'resolved':>10}"
              f"{'failure rate':>14}{'±CS':>9}")
        for r in rows:
            fr = "—" if r["failure_rate"] is None else f"{r['failure_rate']:.3f}"
            ci = "—" if r["ci"] is None else f"{r['ci']:.3f}"
            flag = "" if r["usable"] else "   (thin)"
            print(f"  {r['bucket']:<14}{r['n']:>6}{r['resolved']:>10}"
                  f"{fr:>14}{ci:>9}{flag}")
        print()
        if spread is not None:
            print(f"  spread across usable buckets: {spread:.3f} "
                  f"(threshold {DISCRIMINATION_THRESHOLD})")
        print(f"  VERDICT: {verdict}")
        print()
        print("  Reminder: a FLAT result is a real answer. Do not proceed to "
              "Phase 4 on a signal that does not separate the classes — the "
              "depth curve earned its place on 342 labelled trajectories.")

    return 0 if verdict == "DISCRIMINATES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
