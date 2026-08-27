#!/usr/bin/env python3
"""Override-precision report for the verifier's ground-truth checks.

WHY (§4DI, 2026-08-27). The FILE-ARTIFACT / WEB-EXEC overrides ran from
2026-07-16 to 2026-08-27 with NO durable trace of when they fired — the one
question that matters ("how often is the override RIGHT?") could not be asked
of the record, and the three refutes recoverable by hand were all false. The
sidecar's `override` field closes that; this script is the consumer, so the
measurement is one command instead of a promise.

Reads Data/system/verdicts/*.jsonl (override provenance, seq-ordered) and
cross-references corrections.jsonl + human labels where present. Run it after
a few weeks of traffic; before that it will honestly report thin data.

Usage: python scripts/verdict_override_report.py [--days N] [--home DIR]
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--home", default=os.getenv(
        "GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
    args = ap.parse_args()
    sysdir = Path(args.home) / "system"
    vdir = sysdir / "verdicts"
    if not vdir.is_dir():
        print(f"no verdicts dir at {vdir}")
        return 1

    rows = []
    files = sorted(vdir.glob("*.jsonl"))[-args.days:]
    for f in files:
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        print("no verdict rows yet")
        return 0

    # LAST row per trajectory is the shipped answer's verdict (the sidecar's
    # own join contract — an auto-repaired turn verifies twice).
    # ⚠ (at, seq), not seq alone: seq is a per-process counter and resets on
    # restart, so a repair spanning a restart would join to the PRE-restart
    # row on max(seq). The timestamp breaks the tie in wall order.
    last = {}
    for r in sorted(rows, key=lambda r: (str(r.get("at", "")),
                                         r.get("seq", 0))):
        if r.get("trajectory_id"):
            last[r["trajectory_id"]] = r

    # human labels + late corrections, where recorded
    # ⚠ label PROVENANCE matters: 45 of the first 65 joined labels were
    # `source: "verifier_late"` — the verifier's own late verdict backfilled
    # into corrections. Counting those as agreement is the instrument
    # grading itself; they are reported in their own column.
    human = {}
    corr = sysdir / "trajectories" / "corrections.jsonl"
    if corr.is_file():
        for line in corr.open(encoding="utf-8", errors="replace"):
            try:
                c = json.loads(line)
            except Exception:
                continue
            tid = c.get("trajectory_id") or c.get("id")
            if tid:
                human[tid] = (c.get("outcome"), str(c.get("source", "")))

    by = collections.defaultdict(collections.Counter)
    agree = collections.defaultdict(collections.Counter)
    for tid, r in last.items():
        tag = r.get("override") or "(text judge)"
        by[tag][r.get("verdict")] += 1
        if tid in human:
            v, (h, src) = r.get("verdict"), human[tid]
            if h not in ("passed", "failed"):
                continue                     # "unknown" is not agreement
            bucket = ("self" if "verifier" in src else "human")
            key = ("agrees" if (v == "REFUTED") == (h == "failed")
                   else "DISAGREES")
            agree[tag][f"{bucket}:{key}"] += 1

    print(f"{len(last)} shipped verdicts across {len(files)} day-file(s); "
          f"{sum(1 for t in last if t in human)} with a correction/label\n")
    print(f"{'override':22} {'total':>5}  verdicts / label-agreement")
    for tag in sorted(by, key=lambda t: -sum(by[t].values())):
        verdicts = ", ".join(f"{k}={v}" for k, v in by[tag].most_common())
        lab = ("  |  " + ", ".join(f"{k}={v}" for k, v in agree[tag].items())
               if agree[tag] else "")
        print(f"{tag:22} {sum(by[tag].values()):>5}  {verdicts}{lab}")
    if not any(r.get("override") for r in last.values()):
        print("\nNOTE: no override-tagged rows yet — provenance recording "
              "began 2026-08-27; re-run after real traffic accrues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
