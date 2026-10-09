#!/usr/bin/env python3
"""§4MM one-off (2026-10-08, the real-turn audit): stale moving-target facts.

"What is the latest version of postgresql?" was answered "18.4, released May
14, 2026" twice (10-04, 10-05) although the agent had read 18.6 on
postgresql.org: the graph held `postgresql HAS_VERSION 18.4` (July) beside
18.6 and ranked it higher, and five stored episodes asserting 18.4 were
hydrated as precedent. The code fix stops both at the source
(`TRANSIENT_WORLD_PREDICATE` at every graph writer; moving-target episodes
never hydrated). This removes what is already stored:

1. graph: every LIVE non-owner triplet whose predicate is a moving target
   (HAS_VERSION, HAS_PRICE, …) — archived to the graph archive, then deleted;
2. episodes: the five that assert the stale 18.4 answer — each re-checked to
   still say "18.4" — forgotten (archived, 30-day retention; their vector
   twins are reaped by the dream's reconcile).

Run ONLY with the agent stopped. Dry run by default; `--apply` writes.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4mm.py [--apply]
"""
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
STALE_EPISODES = (20, 92, 266, 270, 548)
STAMP = time.strftime("%Y%m%dT%H%M%S")


def transient_rows(db_path) -> list:
    """Read-only (r2: the dry run constructed GraphMemory, which opens the
    live db for writing)."""
    from ghost_agent.memory.graph import TRANSIENT_WORLD_PREDICATE, GraphMemory
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        rows = c.execute("SELECT subject, predicate, object, COALESCE(weight, 1), timestamp "
                         "FROM triplets WHERE valid_until IS NULL").fetchall()
    return [r for r in rows if TRANSIENT_WORLD_PREDICATE.search(r[1] or "")
            and "user" not in (r[0], r[2])
            and not GraphMemory._is_owner_fact(r[0], r[1], r[2])]


def stale_episode_ids(db: Path) -> list:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        found = []
        for eid in STALE_EPISODES:
            row = c.execute("SELECT outcome, lesson, context FROM episodes WHERE id=?", (eid,)).fetchone()
            if row and "18.4" in " ".join(str(x or "") for x in row):
                found.append(eid)
    return found


def main() -> int:
    if APPLY:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "memory_repair_4mm.py")
    rows = transient_rows(MEM / "knowledge_graph.db")
    eids = stale_episode_ids(MEM / "episodic_memory.db")
    report = {"graph_rows": [f"{s} {p} {o}" for s, p, o, _w, _t in rows], "episodes": eids}
    if APPLY:
        from ghost_agent.memory.graph import GraphMemory
        g = GraphMemory(MEM)
        shutil.copy2(g.db_path, g.db_path.with_name(g.db_path.name + f".pre-4mm-{STAMP}.bak"))
        shutil.copy2(MEM / "episodic_memory.db", MEM / f"episodic_memory.db.pre-4mm-{STAMP}.bak")
        if rows:
            if not g._archive_rows("§4MM moving-target world fact", rows):
                print("graph archive failed — nothing deleted", file=sys.stderr)
                return 1
            with sqlite3.connect(g.db_path) as c:
                for s, p, o, _w, _t in rows:
                    c.execute("DELETE FROM triplets WHERE subject=? AND predicate=? AND object=?", (s, p, o))
        if eids:
            from ghost_agent.memory.episodes import EpisodicMemory
            report["episodes_forgotten"] = EpisodicMemory(MEM).delete_episodes(
                eids, None, reason="§4MM: stale moving-target answer (PostgreSQL 18.4)")
    report["applied"] = APPLY
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
