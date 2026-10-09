#!/usr/bin/env python3
"""§4MJ (2026-10-08, operator: "clean up episodes that name deleted projects").

Every project in `deleted_projects` gets its episodes forgotten the way a hard
delete now does it (`EpisodicMemory.forget_project`): by project id always,
by title only when the title is distinctive and no LIVE project's title
contains it or is contained in it. Archived to `episodes_forgotten.jsonl`
(30-day retention) and the episodic db is backed up first. Dry run by
default; `--apply` deletes (run it with the agent STOPPED). The vector twins
are not touched here: the dream's `_reconcile_memory_stores` →
`VectorMemory.reconcile_indexes` deletes vector rows whose episode id is no
longer live.

    python scripts/memory_repair_4mj_episodes.py --home <GHOST_HOME> [--apply]
"""
import argparse, json, shutil, sqlite3, sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    mem = Path(a.home) / "system" / "memory"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if a.apply:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, "memory_repair_4mj_episodes.py")
    from ghost_agent.memory.episodes import EpisodicMemory
    con = sqlite3.connect(f"file:{mem / 'projects.db'}?mode=ro", uri=True)
    live = [r[0] for r in con.execute("SELECT title FROM projects")]
    deleted = con.execute("SELECT id, title FROM deleted_projects ORDER BY deleted_at").fetchall()
    con.close()
    em = EpisodicMemory(mem)
    plan, seen = [], set()
    for pid, title in deleted:
        ids = [i for i in em.project_mention_ids(pid, title, live) if i not in seen]
        seen.update(ids)
        if ids:
            plan.append({"project": pid, "title": title,
                         "by_title": em.project_title_is_distinctive(title, live), "episodes": ids})
    with sqlite3.connect(mem / "episodic_memory.db") as c:
        total = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        prev = {i: t for i, t in c.execute("SELECT id, trigger FROM episodes")}
    report = {"episodes": total, "to_forget": len(seen), "projects": len(plan),
              "plan": [{**p, "triggers": [prev.get(i, "")[:70] for i in p["episodes"][:4]]} for p in plan],
              "applied": bool(a.apply)}
    if a.apply and seen:
        shutil.copy2(mem / "episodic_memory.db", mem / "episodic_memory.db.pre-4mj.bak")
        n = 0
        for p in plan:
            n += em.delete_episodes(p["episodes"], None, reason=f"project deleted: {p['project']}")
        report["deleted"] = n
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
