#!/usr/bin/env python3
"""Record a verifier bench run as the incumbent baseline — the §4T way.

The baseline file is what `verify_bench_status.py` (the staleness oracle)
compares the tree against and what `verify_bench_compare.py` pairs a new run
with, so it must carry the run's FULL provenance block, the class mix and
the honest interval. This script writes it from a `results.json`, using the
oracle's own interval function and the compare tool's own scoring helpers —
one formula each, never a replica (§4T: a pool replica said 35 where the
tool said 58).

Refuses a run whose route health is not clean (a cheap-leg fall-through or
an unanswered escalation measured a different pipeline) unless --force
names why. Preserves the previous baseline beside the new one.

    PYTHONPATH=src python scripts/record_verifier_baseline.py RESULTS.json \\
        --why "§4JE: artifact-scan clause restored + conceded-corroboration uphold"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_bench_status import _wilson_half_width  # noqa: E402
import verify_bench_compare as vbc  # noqa: E402

DEFAULT_BASELINE = "system/eval/verifier_incumbent_baseline.json"


def build_baseline(results: dict, results_path: str, why: str, previous: dict | None,
                   arm: str = "two_stage_on") -> dict:
    trials = vbc._trials(results, arm)
    keys = list(trials)
    p_nr, n_nr = vbc._rate(trials, keys, want_non_refute=True)
    p_rf, n_rf = vbc._rate(trials, keys, want_non_refute=False)
    if p_nr is None or p_rf is None:
        raise SystemExit("one class has no trials — a balanced score would be a raw score "
                         "wearing a balanced label; refusing to record")
    bal = 0.5 * p_nr + 0.5 * p_rf
    half = 0.5 * ((_wilson_half_width(p_nr, n_nr) ** 2
                   + _wilson_half_width(p_rf, n_rf) ** 2) ** 0.5)
    ev = ((results.get("arms") or {}).get(arm) or {}).get("metrics", {}).get("escalation_events") or {}
    prov = results.get("provenance") or {}
    return {
        "kind": "verifier-incumbent-baseline",
        "source": "scripts/record_verifier_baseline.py (from scripts/verify_bench.py results)",
        "supersedes": (f"{previous.get('recorded_utc')} balanced {previous.get('private_incumbent_balanced')} "
                       f"({previous.get('n_private_trials')} trials)" if previous else ""),
        "why": why,
        "recorded_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "results_path": str(results_path),
        "metric": "balanced = 0.5*mean(non-refute correct) + 0.5*mean(refute correct)",
        "private_incumbent_balanced": round(bal, 4),
        "ci95_half_width": round(half, 4),
        "ci95": [round(bal - half, 4), round(bal + half, 4)],
        "ci_method": "0.5*sqrt(wilson_hw(p_nr,n_nr)^2 + wilson_hw(p_rf,n_rf)^2), z=1.96 — "
                     "verify_bench_status._wilson_half_width",
        "nonrefute_mean": round(p_nr, 4),
        "refute_mean": round(p_rf, 4),
        "n_private_cases": results.get("n_cases"),
        "n_private_trials": results.get("n_trials"),
        "class_mix": {"refute_expecting": n_rf, "non_refute": n_nr},
        "resolution_note": "use ci95_half_width; do NOT reintroduce smallest_resolvable_delta",
        "escalation_ledger": {
            "overturns": ev.get("refute_overturned"),
            "rescues": ev.get("overturn_rescues"),
            "damage": ev.get("overturn_damage"),
            "replaced_uncertain": ev.get("replaced_uncertain"),
            "unavailable": ev.get("escalation_unavailable"),
        },
        "provenance": prov,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--why", required=True, help="what changed that makes this the number")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--arm", default="two_stage_on")
    ap.add_argument("--force", default="", help="record despite unclean route health, WITH a reason")
    args = ap.parse_args()

    home = Path(os.environ.get("GHOST_HOME") or "/Users/vasilis/Data/AI/Data")
    out = Path(args.baseline) if args.baseline else home / DEFAULT_BASELINE
    results = json.loads(Path(args.results).read_text())
    rh = ((results.get("provenance") or {}).get("escalation") or {}).get("route_health") or {}
    if not rh.get("clean") and not args.force:
        print(f"REFUSING: route health is not clean ({rh}) — this run did not measure the "
              f"pipeline; re-run, or --force '<reason>'", file=sys.stderr)
        return 1
    previous = None
    if out.exists():
        previous = json.loads(out.read_text())
        stamp = str(previous.get("recorded_utc", "prev"))[:10]
        keep = out.with_name(f"{out.stem}.{stamp}.json")
        if not keep.exists():
            shutil.copy2(out, keep)
        print(f"previous baseline preserved → {keep}")
    base = build_baseline(results, args.results, args.why, previous, arm=args.arm)
    if args.force:
        base["forced"] = args.force
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(base, indent=2))
    tmp.replace(out)
    print(f"recorded → {out}\n  balanced {base['private_incumbent_balanced']} "
          f"95% CI {base['ci95']} (±{base['ci95_half_width']}) "
          f"n={base['class_mix']['non_refute']} non-refute / {base['class_mix']['refute_expecting']} refute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
