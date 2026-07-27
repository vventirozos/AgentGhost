#!/usr/bin/env python3
"""Backfill `context` / `cluster_id` on pre-2026-07-26 episodes.

Both fields are populated at write time from the turn's tool actions
(`agent.py::_record_episode_safe`), but that wiring only landed 2026-07-26 —
episodes recorded before it have empty values. Live coverage was 9/174
(5.2%), so cluster-keyed retrieval (`get_episodes_by_cluster`) and the
recovery scan (`search_recoveries`, which greps `context` for FAILED
markers) were blind to almost the entire corpus.

The actions themselves were always persisted to `episode_actions`, so both
fields can be reconstructed EXACTLY as the live path would have written
them — this is a replay of the same derivation, not a guess:

    cluster_id = _domain_for_tool(<first substantive tool>)
    context    = "tools: a → b(FAILED) → c ; recovered after failures in: b"

Idempotent: only touches rows whose `context` AND `cluster_id` are empty,
so re-running is a no-op. Writes nothing without --apply.

Usage:
    python scripts/backfill_episode_context.py            # dry run
    python scripts/backfill_episode_context.py --apply
    python scripts/backfill_episode_context.py --db /path/to/episodic_memory.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.core.metacog import _domain_for_tool  # noqa: E402

# Mirrors the live path: these never count as the episode's "first
# substantive tool" for clustering purposes.
_NON_SUBSTANTIVE = {"", "none", "notify_operator"}


def _default_db() -> Path:
    home = os.environ.get("GHOST_HOME") or str(Path.home() / "Data/AI/Data")
    return Path(home) / "system" / "memory" / "episodic_memory.db"


def _derive(actions):
    """Rebuild (cluster_id, context) from an episode's ordered actions.

    Mirrors `_record_episode_safe` so backfilled rows are indistinguishable
    from natively-written ones.
    """
    first_real = ""
    fail_tools = []
    for tool_name, success in actions:
        name = (tool_name or "").strip()
        if not name or name.lower() in _NON_SUBSTANTIVE:
            continue
        if not first_real:
            first_real = name
        if not success:
            fail_tools.append(name)

    cluster_id = _domain_for_tool(first_real) if first_real else ""

    chain = " → ".join(
        f"{(t or '?')}{'' if ok else '(FAILED)'}" for t, ok in actions[:8]
    ) or "no tools"
    parts = [f"tools: {chain}"]
    # "recovered after failures in: …" is only meaningful on a successful
    # episode; the caller passes episode success in.
    return cluster_id, parts, fail_tools


def backfill(db_path: Path, *, apply: bool) -> int:
    if not db_path.exists():
        print(f"no episodic DB at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, outcome_success FROM episodes "
            "WHERE (context IS NULL OR context = '') "
            "  AND (cluster_id IS NULL OR cluster_id = '') "
            "ORDER BY id"
        ).fetchall()
        if not rows:
            print("nothing to backfill — every episode already has "
                  "context/cluster_id")
            return 0

        updates, skipped = [], 0
        for row in rows:
            actions = [
                (a["tool_name"], bool(a["success"]))
                for a in conn.execute(
                    "SELECT tool_name, success FROM episode_actions "
                    "WHERE episode_id = ? ORDER BY action_order",
                    (row["id"],),
                ).fetchall()
            ]
            if not actions:
                # No stored actions → nothing faithful to reconstruct.
                # Leave it alone rather than inventing a placeholder.
                skipped += 1
                continue
            cluster_id, parts, fail_tools = _derive(actions)
            if fail_tools and row["outcome_success"]:
                parts.append("recovered after failures in: "
                             + ", ".join(dict.fromkeys(fail_tools)))
            context = " ; ".join(parts)[:2000]
            updates.append((context, cluster_id, row["id"]))

        print(f"episodes missing both fields : {len(rows)}")
        print(f"  reconstructable from actions: {len(updates)}")
        print(f"  skipped (no stored actions) : {skipped}")
        if updates:
            print("\nsample:")
            for context, cluster_id, ep_id in updates[:3]:
                print(f"  #{ep_id} cluster={cluster_id!r}")
                print(f"      {context[:110]}")

        if not apply:
            print("\nDRY RUN — re-run with --apply to write.")
            return 0

        conn.executemany(
            "UPDATE episodes SET context = ?, cluster_id = ? WHERE id = ?",
            updates,
        )
        conn.commit()
        print(f"\napplied: {len(updates)} episodes updated")

        total, with_ctx = conn.execute(
            "SELECT COUNT(*), SUM(context IS NOT NULL AND context != '') "
            "FROM episodes"
        ).fetchone()
        print(f"coverage now: {with_ctx}/{total} "
              f"({100.0 * (with_ctx or 0) / max(1, total):.1f}%)")
        return 0
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    args = ap.parse_args()
    return backfill(args.db or _default_db(), apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
