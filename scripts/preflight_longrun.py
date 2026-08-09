#!/usr/bin/env python3
"""Pre-flight gate for any run long enough to waste real time or compute.

WHY THIS EXISTS. Three separate long runs on this project were wasted or
unreadable, and every one failed a precondition that was checkable BEFORE
launch rather than discoverable after:

  2026-08-09 (~8h, wasted)  the B3 ablation's two arms resolved to BYTE-
      IDENTICAL flags, so the comparison compared a configuration with
      itself and could only ever report a tie. A single smoke run diffing
      the two arms' resolved flags would have caught it in seconds.
  2026-08-09 (~8h, invalid) the warm-seeded metric counted ABSOLUTE store
      contents, so all three arms reported the seed's own 50/5/19. The
      metric measured the setup, not the run.
  2026-08-09 (90min, blind) the verifier bench was launched with a block-
      buffered stdout, so its own progress counter was unreadable. Four
      derived progress figures were reported to the operator and every one
      was wrong.

None of those were subtle in hindsight. They were unasked questions.

THE GATE. Five preconditions; any failure blocks the launch:

  1. OBSERVABLE   progress is readable WHILE running — an unbuffered stream
                  or a `progress.json` written under the RunProgress contract.
  2. BOUNDED      the denominator comes from the TOOL, not from an estimate.
                  "How many items?" must have an authoritative answer.
  3. RESUMABLE    if killed at any moment, the work already paid for is not
                  lost (a response cache, a checkpoint, an idempotent skip).
  4. DISCRIMINATING  a smoke run proved the configuration produces
                  DISTINGUISHABLE output. For an A/B this means the arms
                  actually differ; for a measurement, that the metric moves
                  when the thing it measures moves.
  5. MEASURED     the cost estimate comes from a real timed smoke, expressed
                  as a RANGE. Extrapolating from an unrepresentative pilot is
                  how "90 minutes" became two hours.

Usage — declare what you checked, and it holds you to it:

    python scripts/preflight_longrun.py --name "verifier re-bench" \\
        --observable progress-file:$GHOST_HOME/system/eval/rebench.progress.json \\
        --total-from-tool 464 \\
        --resumable "response cache --cache-mode read" \\
        --smoke "16 trials, arms differ on --frontier-selfplay" \\
        --measured-rate 3.5 --rate-source "90s window over live trials"

Exit 0 = cleared to launch. Exit 1 = blocked, with the failing precondition
named. `--force` records an override WITH a reason rather than silently
skipping the gate, because an unexplained override is the same as no gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fail(checks, name, why, fix):
    checks.append({"check": name, "ok": False, "why": why, "fix": fix})


def _ok(checks, name, evidence):
    checks.append({"check": name, "ok": True, "evidence": evidence})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="what you are launching")
    ap.add_argument("--observable", default="",
                    help="'progress-file:PATH' or 'unbuffered' (python -u)")
    ap.add_argument("--total-from-tool", type=int, default=0,
                    help="item count as PRINTED BY THE TOOL, not estimated")
    ap.add_argument("--resumable", default="",
                    help="what makes a kill cheap (cache/checkpoint/skip)")
    ap.add_argument("--smoke", default="",
                    help="what a smoke run PROVED is distinguishable")
    ap.add_argument("--measured-rate", type=float, default=0.0,
                    help="items/min measured in a real timed run")
    ap.add_argument("--rate-source", default="",
                    help="where that rate was measured")
    ap.add_argument("--force", default="",
                    help="override the gate WITH a stated reason")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    checks: list = []

    # 1. OBSERVABLE
    if not args.observable:
        _fail(checks, "OBSERVABLE",
              "no way to read progress while the run is in flight",
              "pass --observable unbuffered (launch with `python -u`) or "
              "progress-file:PATH using eval.runprogress.RunProgress")
    elif args.observable.startswith("progress-file:"):
        p = Path(args.observable.split(":", 1)[1])
        # Not required to exist yet (the run creates it), but its directory
        # must — a progress file that cannot be written is no observability.
        if not p.parent.exists():
            _fail(checks, "OBSERVABLE",
                  f"progress file directory does not exist: {p.parent}",
                  "create it, or point at a writable path")
        else:
            _ok(checks, "OBSERVABLE", f"progress file at {p}")
    elif args.observable == "unbuffered":
        _ok(checks, "OBSERVABLE", "stdout unbuffered (python -u)")
    else:
        _fail(checks, "OBSERVABLE",
              f"unrecognised observability claim: {args.observable!r}",
              "use 'unbuffered' or 'progress-file:PATH'")

    # 2. BOUNDED
    if args.total_from_tool <= 0:
        _fail(checks, "BOUNDED",
              "no authoritative item count",
              "run the tool's own dry-run/banner and pass the number it "
              "PRINTS. Do not compute it yourself — a derived denominator "
              "(35 instead of 58) is what made every progress report wrong "
              "on 2026-08-09")
    else:
        _ok(checks, "BOUNDED", f"{args.total_from_tool} items (from the tool)")

    # 3. RESUMABLE
    if not args.resumable:
        _fail(checks, "RESUMABLE",
              "a kill at any moment would discard everything done so far",
              "add a response cache / checkpoint / idempotent skip, then "
              "pass --resumable describing it")
    else:
        _ok(checks, "RESUMABLE", args.resumable)

    # 4. DISCRIMINATING
    if not args.smoke:
        _fail(checks, "DISCRIMINATING",
              "nothing proved this configuration can produce a "
              "distinguishable result",
              "run a small smoke first and state what it proved. The 8h "
              "ablation whose two arms were byte-identical passed every "
              "other check and still measured nothing")
    else:
        _ok(checks, "DISCRIMINATING", args.smoke)

    # 5. MEASURED
    if args.measured_rate <= 0 or not args.rate_source:
        _fail(checks, "MEASURED",
              "no measured throughput, so any ETA would be invented",
              "time a real slice and pass --measured-rate with "
              "--rate-source. Extrapolating from an unrepresentative pilot "
              "is how '90 minutes' became two hours")
    else:
        eta = (args.total_from_tool / args.measured_rate
               if args.measured_rate > 0 and args.total_from_tool else 0)
        _ok(checks, "MEASURED",
            f"{args.measured_rate}/min via {args.rate_source}"
            + (f" -> ETA ~{eta*0.75:.0f}-{eta*1.5:.0f} min (RANGE)"
               if eta else ""))

    failed = [c for c in checks if not c["ok"]]
    cleared = not failed or bool(args.force)

    if args.json:
        print(json.dumps({"name": args.name, "cleared": cleared,
                          "forced": bool(args.force), "checks": checks},
                         indent=1))
        return 0 if cleared else 1

    print("=" * 78)
    print(f"PRE-FLIGHT — {args.name}")
    print("=" * 78)
    for c in checks:
        mark = "PASS" if c["ok"] else "BLOCK"
        print(f"  [{mark:5}] {c['check']}")
        if c["ok"]:
            print(f"           {c['evidence']}")
        else:
            print(f"           {c['why']}")
            print(f"           fix: {c['fix']}")
    print()
    if not failed:
        print("  CLEARED FOR LAUNCH — all five preconditions hold.")
        print("=" * 78)
        return 0
    if args.force:
        print(f"  ⚠ OVERRIDDEN ({len(failed)} failing) — reason: {args.force}")
        print("  Recorded, not hidden. If this run is wasted, the reason is "
              "above.")
        print("=" * 78)
        return 0
    print(f"  BLOCKED — {len(failed)} precondition(s) failed. Each cost a "
          f"real run\n  on this project. Fix them, or --force with a reason.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
