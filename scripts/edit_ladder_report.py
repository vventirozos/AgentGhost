#!/usr/bin/env python3
"""§4KB step 1 — what the tolerance ladder actually costs.

Reads `<GHOST_HOME>/system/edits/ladder.jsonl` (written by
`ghost_agent.utils.edit_ledger`) and answers the three questions that decided
this work item:

  Q1  What fraction of APPLIED edits rest on a rung below `exact`?
  Q2  How often is a non-exact apply followed by a corrective edit to the same
      file soon after — the "landed wrong" proxy?
  Q3  Does that corrective rate differ by rung?

⚠ **Q2 is a PROXY and the report says so in its own output.** A second applied
edit to the same file inside the window can be a correction of a bad match OR
a second planned edit. It cannot tell them apart. What makes it useful is the
COMPARISON across rungs: if `exact` and `fuzzy` are both followed by a second
edit at the same rate, tolerance is not buying trouble; if `fuzzy` is followed
far more often, that gap is the ladder's cost. Read the delta, never the level.

usage: edit_ladder_report.py [--home DIR] [--window N] [--since-hours H] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ghost_agent.utils.edit_ledger import (                    # noqa: E402
    RUNG_EXACT,
    read_ledger,
    rung_family,
)


def _families(row):
    """Families that APPLIED in this row, deduped, order-preserving."""
    out = []
    for f in row.get("families") or []:
        if f and f not in out:
            out.append(f)
    return out


def corrective_index(rows, window: int):
    """For each row index, whether a LATER applied edit hits the same path
    within `window` subsequent applied-edit rows of the same request.

    Same-request scoping matters: two edits to one file in two different turns
    hours apart are not a correction, they are ordinary work.
    """
    flagged = {}
    applied = [(i, r) for i, r in enumerate(rows) if r.get("applied")]
    for pos, (i, r) in enumerate(applied):
        path, req = r.get("path", ""), r.get("req_id", "")
        hit = False
        for j, later in applied[pos + 1: pos + 1 + window]:
            if later.get("path") == path and later.get("req_id", "") == req:
                hit = True
                break
        flagged[i] = hit
    return flagged


def build(rows, window: int):
    applied = [r for r in rows if r.get("applied")]
    rejected = [r for r in rows if not r.get("applied")]
    corr = corrective_index(rows, window)

    per_family = defaultdict(lambda: {"applied": 0, "corrected": 0})
    for i, r in enumerate(rows):
        if not r.get("applied"):
            continue
        for fam in _families(r) or [""]:
            per_family[fam]["applied"] += 1
            if corr.get(i):
                per_family[fam]["corrected"] += 1

    n_applied = len(applied)
    n_tolerant = sum(
        1 for r in applied if any(f != RUNG_EXACT for f in _families(r)))

    reasons = defaultdict(int)
    for r in rejected:
        reasons[r.get("reason") or "(none)"] += 1

    # §4KC: `edit` and `replace` on ONE yardstick. Same `applied` rule (see
    # `file_system._edit_applied`), same ledger, so the comparison the
    # one-cycle overlap exists to make is a row per op here.
    per_op = defaultdict(lambda: {"rows": 0, "applied": 0, "rejected": 0,
                                  "tolerant": 0, "reasons": defaultdict(int)})
    for r in rows:
        op = r.get("op") or "replace"
        e = per_op[op]
        e["rows"] += 1
        if r.get("applied"):
            e["applied"] += 1
            if any(f != RUNG_EXACT for f in _families(r)):
                e["tolerant"] += 1
        else:
            e["rejected"] += 1
            e["reasons"][r.get("reason") or "(none)"] += 1

    return {
        "per_op": {
            k: {
                "rows": v["rows"], "applied": v["applied"],
                "rejected": v["rejected"], "tolerant": v["tolerant"],
                "applied_rate": (v["applied"] / v["rows"]) if v["rows"] else 0.0,
                "reasons": dict(sorted(v["reasons"].items(), key=lambda kv: -kv[1])),
            }
            for k, v in sorted(per_op.items())
        },
        "rows": len(rows),
        "applied": n_applied,
        "rejected": len(rejected),
        "tolerant_applied": n_tolerant,
        "tolerant_share": (n_tolerant / n_applied) if n_applied else 0.0,
        "per_family": {
            k: {
                "applied": v["applied"],
                "corrected": v["corrected"],
                "corrective_rate": (v["corrected"] / v["applied"]) if v["applied"] else 0.0,
            }
            for k, v in sorted(per_family.items())
        },
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "window": window,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME"))
    ap.add_argument("--window", type=int, default=3,
                    help="how many later applied edits count as 'soon after'")
    ap.add_argument("--since-hours", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    home = Path(a.home) if a.home else None
    rows = read_ledger(home=home)
    if a.since_hours > 0:
        cut = time.time() - a.since_hours * 3600
        rows = [r for r in rows if float(r.get("ts") or 0) >= cut]

    rep = build(rows, a.window)
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0

    if not rep["rows"]:
        print("No edit-ladder rows yet. The ledger fills from live replace "
              "calls; give it traffic before reading anything off it.")
        return 0

    print(f"edit-ladder report — {rep['rows']} rows "
          f"({rep['applied']} applied, {rep['rejected']} rejected)")
    print()
    print(f"Q1  applied edits resting below `exact`: "
          f"{rep['tolerant_applied']}/{rep['applied']} "
          f"= {rep['tolerant_share']:.1%}")
    print()
    print(f"Q2/Q3  corrective-edit rate by rung "
          f"(same path, same request, within {rep['window']} later applies)")
    print("       ⚠ PROXY: a second edit may be a correction OR planned work.")
    print("       Read the DELTA between rungs, never the level.")
    print()
    print(f"       {'rung':<12} {'applied':>8} {'corrected':>10} {'rate':>8}")
    for fam, v in rep["per_family"].items():
        print(f"       {(fam or '(none)'):<12} {v['applied']:>8} "
              f"{v['corrected']:>10} {v['corrective_rate']:>7.1%}")
    if rep["rejection_reasons"]:
        print()
        print("       rejections by reason:")
        for k, n in rep["rejection_reasons"].items():
            print(f"       {k:<40} {n:>6}")
    if len(rep["per_op"]) > 1:
        print()
        print("Q4  edit vs replace — same ledger, same `applied` rule")
        print(f"       {'op':<28} {'rows':>6} {'applied':>8} {'rate':>7} {'tolerant':>9}")
        for op, v in rep["per_op"].items():
            print(f"       {op:<28} {v['rows']:>6} {v['applied']:>8} "
                  f"{v['applied_rate']:>6.1%} {v['tolerant']:>9}")
        for op, v in rep["per_op"].items():
            if v["reasons"]:
                print(f"       {op} rejections: " + ", ".join(
                    f"{k}={n}" for k, n in list(v["reasons"].items())[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
