#!/usr/bin/env python3
"""§4SN (retrieval review 2026-10-31): re-date the profile's graph mirror
edges (``user HAS_<CATEGORY>_<KEY> <value>``) to when the PROFILE learned each
value. The 10-30 mirror re-sync stated every owner edge "as of 2026-10-30";
the owner-fact lookup printed that date beside the profile's "as of
2026-07-07" and the model picked "the most recent data".

Only ``valid_from`` of LIVE mirror edges whose value is a CURRENT profile
value changes, and only to an EARLIER date; nothing is added or removed.
Dry run by default.

    PYTHONPATH=src .agent.venv/bin/python scripts/restamp_owner_mirrors_4sn.py [--memory-dir DIR] [--apply]

--apply copies knowledge_graph.db to knowledge_graph.db.pre-4sn-<ts> first.
--apply refuses while another writer (the agent) holds the memory dir: stop it
first (gate idle → launchctl kill SIGTERM → bootout the replacement → run →
bootstrap).
"""
import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghost_agent.memory.profile import ProfileMemory, mirror_name  # noqa: E402
from ghost_agent.memory.store_lock import assert_no_other_writer  # noqa: E402
from ghost_agent.tools.memory import _profile_learned_epochs  # noqa: E402


def plan(memory_dir: Path) -> list:
    pm = ProfileMemory(memory_dir)
    raw = pm.load_raw() or {}
    db = memory_dir / "knowledge_graph.db"
    out = []
    with sqlite3.connect(db) as c:
        for cat, sub in raw.items():
            if not isinstance(sub, dict):
                continue
            for key in sub:
                pred = "HAS_" + mirror_name(cat, key).upper().replace(" ", "_")
                for value, epoch in _profile_learned_epochs(pm, cat, key).items():
                    for rowid, vf in c.execute(
                            "SELECT rowid, valid_from FROM triplets WHERE valid_until IS NULL AND lower(subject)='user' "
                            "AND predicate=? AND lower(trim(object))=?", (pred, value)):
                        # only EARLIER: an edge the graph knew before the
                        # profile did keeps its own (older) date
                        if vf is None or epoch < float(vf) - 86400:
                            out.append((rowid, pred, value, vf, epoch))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-dir", default=str(Path.home() / "Data/AI/Data/system/memory"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    md = Path(a.memory_dir)
    rows = plan(md)
    fmt = lambda e: time.strftime("%Y-%m-%d", time.gmtime(float(e))) if e else "none"  # noqa: E731
    for _rid, pred, value, vf, epoch in rows:
        print(f"  {pred} {value!r}: stated {fmt(vf)} → {fmt(epoch)}")
    print(f"{len(rows)} mirror edge(s) to re-date" + ("" if a.apply else " (dry run; --apply to write)"))
    if a.apply and rows:
        assert_no_other_writer(md, "restamp_owner_mirrors_4sn.py")   # the agent must be stopped
        db = md / "knowledge_graph.db"
        bak = db.with_name(f"knowledge_graph.db.pre-4sn-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(db, bak)
        with sqlite3.connect(db) as c:
            c.executemany("UPDATE triplets SET valid_from=? WHERE rowid=?", [(e, r) for r, _p, _v, _vf, e in rows])
        print(f"applied; backup at {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
