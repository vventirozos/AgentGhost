#!/usr/bin/env python
"""Retire constraints left ACTIVE on projects that are already terminal.

WHY (queue #10, 2026-08-21). Since 2026-08-01 every DONE transition retires
the project's constraints — the fix for an incident where a 07-28 deliverable
constraint ("Start with: What it means to BE ghost") replayed into every
request for four days after the work closed, polluting artifacts and driving
verifier refutes whose follow-up tasks reopened the project: a self-feeding
loop.

Retirement fires ON THE TRANSITION, so a project whose last DONE predates the
fix still carries its constraints armed. Measured on the live store: **WebOS
is DONE with 7 active constraints and none retired**; its last
`project_auto_rollup {"new_status": "DONE"}` was 2026-07-31 22:00, one day
before the fix. The code is correct; the data is stale.

That matters because the constraints re-arm the moment the project becomes
active again — which is exactly the situation the incident describes, and the
next DONE transition (the thing that would retire them) only happens AFTER
they have had their run.

This calls the store's own `retire_constraints`, so the audit trail
(`metadata.constraints_retired` + a `constraints_retired` event) is written by
the same code the live path uses. Nothing is invented here.

Usage:
    PYTHONPATH=src python scripts/retire_stale_constraints.py            # dry run
    PYTHONPATH=src python scripts/retire_stale_constraints.py --apply
    ... --home /path/to/GHOST_HOME     (default: $GHOST_HOME)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.memory.projects import ProjectStore  # noqa: E402

#: Statuses at which constraints should no longer be armed. RELEASED and
#: ARCHIVED are terminal too, but only DONE has the documented automatic
#: retirement, so those are reported and left alone unless --all-terminal.
TERMINAL = ("DONE",)
ALL_TERMINAL = ("DONE", "RELEASED", "ARCHIVED", "FAILED")


def _constraints(store, project) -> list:
    """Active constraints on a project row.

    ⚠ `list_projects` returns `metadata` ALREADY PARSED (a dict), not the raw
    `metadata_json` column — the first version of this script read the column
    name straight off the SQL schema, found nothing anywhere, and reported
    "nothing to retire" on a store with seven live examples. A reconciler that
    silently finds nothing is indistinguishable from a clean store, which is
    the failure mode this whole queue is about.
    """
    meta = project.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(meta, dict):
        return []
    raw = meta.get("constraints") or []
    return [str(c) for c in raw if str(c).strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", default=os.getenv("GHOST_HOME", ""))
    ap.add_argument("--apply", action="store_true",
                    help="actually retire (default: dry run)")
    ap.add_argument("--all-terminal", action="store_true",
                    help="include RELEASED/ARCHIVED/FAILED, not just DONE")
    args = ap.parse_args()

    if not args.home:
        print("ERROR: pass --home or set GHOST_HOME", file=sys.stderr)
        return 2
    home = Path(args.home)
    store = ProjectStore(home / "system" / "memory",
                         sandbox_root=home / "sandbox")

    wanted = ALL_TERMINAL if args.all_terminal else TERMINAL
    targets = []
    for p in store.list_projects():
        if str(p.get("status") or "").upper() not in wanted:
            continue
        act = _constraints(store, p)
        if act:
            targets.append((p, act))

    if not targets:
        print("nothing to retire — no terminal project carries an active "
              "constraint.")
        return 0

    for p, act in targets:
        print(f"{p['id']}  {str(p.get('title'))[:36]:36s} "
              f"[{p.get('status')}]  {len(act)} active")
        for c in act:
            print(f"     - {str(c)[:88]}")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to retire these. They stay in "
              "metadata.constraints_retired, and the user restating one "
              "re-arms it.")
        return 0

    total = 0
    for p, _act in targets:
        moved = store.retire_constraints(
            p["id"], reason="terminal project predating the 2026-08-01 "
                            "constraint lifecycle (queue #10 reconciliation)")
        total += len(moved or [])
        print(f"retired {len(moved or [])} on {p['id']}")
    print(f"\ntotal retired: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
