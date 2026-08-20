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
    ap.add_argument("--bench", action="store_true",
                    help="report the BENCH population instead of live turns "
                         "(§4BF 1c: reads $GHOST_HOME/system/bench/"
                         "trajectories, admits task_kind=bench only)")
    args = ap.parse_args()

    if args.bench and not args.trajectories:
        # The WRITER's root resolution, not a re-derivation (R5 review:
        # this script's own fallback disagreed with banks.py's — the
        # silent-inert split R4 fixed one file over, re-created here).
        from ghost_agent.eval.banks import trajectories_root
        root = trajectories_root()
    else:
        root = Path(args.trajectories) if args.trajectories else _default_root()
    if not root.exists():
        print(f"no trajectory corpus at {root}", file=sys.stderr)
        if args.json:
            # R3 finding 5: a bare early return starved --json of the very
            # keys added so that "absent ≠ healthy" — a pipe consumer got
            # empty stdout and a JSON parse error instead of a document
            # saying what is unknown.
            print(json.dumps({"error": f"no trajectory corpus at {root}",
                              "coverage": {},
                              "population": ("bench" if args.bench
                                             else "live"),
                              "registry_degraded": None,
                              "enabled_unstamped": None},
                             indent=2, sort_keys=True))
        return 0

    # Scope filter, DENY form (R6/R7 reviews): exclude every name
    # DECLARED in the other scope, enabled or not. The R5 cut
    # ALLOW-listed this scope's enabled names (erased a disabled spec's
    # history behind a false "stamp regressed" alarm); the R6 cut denied
    # only ENABLED other-scope names (re-scope-then-disable resurrected
    # the stale stamps). `enabled: false` must stay cost-free to read.
    _deny = None
    _expected = None
    _degraded = False
    try:
        from ghost_agent.core.experiments import (
            SCOPE_BENCH, SCOPE_LIVE, load_registry, registry_path)
        _reg = load_registry(registry_path())
        _deny = set(_reg.names_in_scope(
            SCOPE_LIVE if args.bench else SCOPE_BENCH))
        _degraded = bool(getattr(_reg, "degraded", False))
        # Introspect review R2 MAJOR-1 — this file's OWN R5 comment above
        # records the "fixed one file over, re-created here" pattern, and
        # it happened again: the enabled-but-unstamped zero-row fix (C1,
        # the shape that hid verify_depth's three inert days) landed in
        # the introspect surface while this "same report" CLI kept
        # enumerating only stamped arms. Skipped when the registry is
        # degraded: the substituted defaults would raise false
        # "enrollment broken" alarms for specs the operator never listed.
        if not _degraded:
            _expected = set(_reg.names_for_scope(
                SCOPE_BENCH if args.bench else SCOPE_LIVE))
    except Exception:
        _deny = None
        _expected = None
    if _degraded:
        print("WARNING: system/experiments.json exists but is unreadable — "
              "scope filter reflects the CODE DEFAULTS, and "
              "enabled-but-unstamped arms cannot be detected",
              file=sys.stderr)

    collector = TrajectoryCollector(root=root, session_id="reader")
    summary, triggered, coverage = summarize_streaming(
        collector.iter_trajectories(day=args.day or None),
        admit_task_kinds=(("bench",) if args.bench else ("user_request",)),
        deny_names=_deny)

    if args.json:
        out = {"coverage": coverage,
               "population": ("bench" if args.bench else "live"),
               "registry_degraded": _degraded,
               # Enabled specs with no stamped turn — the machine-readable
               # form of the n=0 rows (absent key ≠ healthy).
               "enabled_unstamped": sorted((_expected or set())
                                           - set(summary))}
        for name, arms in summary.items():
            out[name] = {
                "arms": {
                    arm: {"n": s.n, "unknown": s.unknown,
                          "means": {m: s.mean(m) for m in (s.values or {})}}
                    for arm, s in arms.items()
                },
                # experiment=name on BOTH (review R4 C1). The human path
                # below threads it and this machine path did not, so
                # `--json` dropped every per-experiment caveat — including
                # the CIRCULAR annotation on `failure_rate`, which for
                # tts_bon is the ONLY metric with any n on the bench
                # population and is already past the verdict floor. A bench
                # consumer would have read TREATMENT WORSE where the human
                # report reads NO VERDICT. This file's own comments record
                # the same "fixed one file over, re-created here" pattern
                # twice already.
                "comparisons": [asdict(c) | {"verdict": c.verdict}
                                for c in compare_arms(arms, alpha=args.alpha,
                                                      experiment=name)],
                "triggered_comparisons": [
                    asdict(c) | {"verdict": c.verdict}
                    for c in compare_arms(triggered.get(name, {}),
                                          alpha=args.alpha,
                                          experiment=name)],
            }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(render_report(summary, alpha=args.alpha, triggered=triggered,
                        coverage=coverage,
                        population=("bench" if args.bench else "live"),
                        expected_names=_expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
