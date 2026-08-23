#!/usr/bin/env python
"""One-shot reconciliation: give the diary the verdicts the corpus already has.

WHY THIS EXISTS (queue #7, 2026-08-21). The autobiographical log's verdict
backfill was wired to exactly one live feed — finalize's inline
``verifier_backfill`` leg, which only fires when the bounded in-loop critic
await wins its 25s race. The LATE async verdict (~85% of production
verdicts) and every human 👍/👎 wrote the trajectory corpus and never the
diary. That is fixed forward in ``core/agent.py`` and ``core/feedback.py``;
this script repairs the BACKLOG those two legs never wrote — on the live
store when it was found, 331 diary rows sat at ``unknown`` while the
corrections sidecar already carried a passed/failed label for the very same
``trajectory_id``.

THE RULE IS THE SAME AS THE FIX: the diary FOLLOWS the corpus. The resolved
outcome is read through ``iter_trajectories``, i.e. through the corrections
OVERLAY, so every authority guard the collector applied at write time
(human label > machine verdict, bench oracle > machine verdict, the
passed-carries-no-reason coherence rule) is already baked into what this
reads. It invents nothing and it arbitrates nothing.

Bounded by construction: it only ever rewrites diary rows that ALREADY
EXIST and are ALREADY ``unknown``. A row the diary never captured is not
created (capture is real_only-gated and this must not launder a sim/bench
turn into the diary), and a row that already carries a verdict is left
alone — this is a repair, not a re-label.

Usage:
    PYTHONPATH=src python scripts/heal_diary_outcomes.py            # dry run
    PYTHONPATH=src python scripts/heal_diary_outcomes.py --apply    # write
    ... --home /path/to/GHOST_HOME     (default: $GHOST_HOME)
    ... --limit N                      (cap the rows changed)

``--apply`` takes a timestamped copy of the diary next to it first.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402
from ghost_agent.selfhood.autobiographical import (  # noqa: E402
    AutobiographicalMemory,
)

RESOLVED = ("passed", "failed")


def resolved_outcomes(traj_root: Path) -> dict:
    """{trajectory_id: (outcome, failure_reason)} for every RESOLVED turn,
    read through the corrections overlay — the corpus's own final word."""
    col = TrajectoryCollector(root=traj_root, session_id="heal", enabled=True)
    out = {}
    for t in col.iter_trajectories():
        outcome = (getattr(t, "outcome", "") or "").strip().lower()
        if outcome in RESOLVED:
            out[t.id] = (outcome, getattr(t, "failure_reason", "") or "")
    return out


def diary_gaps(diary_path: Path, resolved: dict) -> list:
    """Diary rows that exist, are `unknown`, and that the corpus resolved."""
    gaps = []
    if not diary_path.exists():
        return gaps
    for line in diary_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (d.get("outcome") or "").strip().lower() != "unknown":
            continue
        hit = resolved.get(d.get("trajectory_id") or "")
        if hit:
            gaps.append((d.get("trajectory_id"), hit[0], hit[1],
                         (d.get("timestamp") or "")[:10]))
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", default=os.getenv("GHOST_HOME", ""),
                    help="GHOST_HOME (default: $GHOST_HOME)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the diary (default: dry run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N rows (0 = no cap)")
    args = ap.parse_args()

    if not args.home:
        print("ERROR: pass --home or set GHOST_HOME", file=sys.stderr)
        return 2
    system = Path(args.home) / "system"
    traj_root = system / "trajectories"
    diary_path = system / "selfhood" / "autobiographical.jsonl"
    for p in (traj_root, diary_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            return 2

    resolved = resolved_outcomes(traj_root)
    gaps = diary_gaps(diary_path, resolved)
    if args.limit > 0:
        gaps = gaps[:args.limit]

    by_day = collections.Counter(g[3] for g in gaps)
    by_outcome = collections.Counter(g[1] for g in gaps)
    print(f"corpus resolved outcomes : {len(resolved)}")
    print(f"diary rows to heal       : {len(gaps)}  {dict(by_outcome)}")
    if by_day:
        span = f"{min(by_day)} … {max(by_day)}"
        print(f"spanning                 : {span}")
    if not gaps:
        print("nothing to do.")
        return 0
    if not args.apply:
        for tid, outcome, _reason, day in gaps[:10]:
            print(f"  would set {tid[:8]} ({day}) → {outcome}")
        if len(gaps) > 10:
            print(f"  … and {len(gaps) - 10} more")
        print("\nDRY RUN — re-run with --apply to write.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = diary_path.with_suffix(f".jsonl.pre-heal-{stamp}")
    shutil.copy2(diary_path, backup)
    print(f"backup written           : {backup}")

    diary = AutobiographicalMemory(diary_path.parent, enabled=True)
    changed = failed = 0
    for tid, outcome, reason, _day in gaps:
        if diary.update_outcome(tid, outcome, failure_reason=reason):
            changed += 1
        else:
            failed += 1
    print(f"healed                   : {changed}")
    if failed:
        print(f"no-op / not found        : {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
