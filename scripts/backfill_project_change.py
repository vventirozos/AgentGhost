#!/usr/bin/env python3
"""Add a past change to a project's change log by hand (§4UM).

The change log (``project_changes`` in projects.db, mirrored to the project's
PROJECT_CHANGES.md) records every change the owner asks for once planned work is
done. Work done BEFORE the log existed is not in it; this script adds one
such change, with the details read from the request log by the operator.

Every detail is an argument — nothing about any project is built in:

  python scripts/backfill_project_change.py \\
      --memory-dir "$GHOST_HOME/system/memory" --sandbox-root "$GHOST_HOME/sandbox" \\
      --project <id> --ask "add a game to it" \\
      --added js/apps/game.js --changed index.html \\
      --summary "The game is in and live." \\
      --ts "2026-11-12 19:57" --req <request id> \\
      --deployed --url https://host.example:20000/ --how put --lab-id lab_xxxxxxxx

Dry run by default (prints the entry it would add); ``--apply`` writes it,
and refuses while the agent runs (stop it, apply, start it again).
With ``--deployed --url`` it also records the push (a deploy entry at the
same time, and where the project is live — never over a newer deploy);
``--deployed`` alone marks only this entry live. ``--reopen`` opens a DONE
project the owner closed (one the old roll-up closed by itself already reads
"ready"). Idempotent: the same --req, or the same time and ask, is not added
again unless ``--force``. The dry run opens the database read-only.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ghost_agent.memory.projects import ProjectStore, backfill_change, change_line  # noqa: E402
from ghost_agent.memory.store_lock import assert_no_other_writer  # noqa: E402


def _ghost_home() -> Path:
    return Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))


def _parse_ts(s: str) -> float:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.mktime(time.strptime(s.strip(), fmt))
        except ValueError:
            continue
    raise SystemExit(f"--ts {s!r}: use local time like '2026-11-12 19:57'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-dir", default=None, help="$GHOST_HOME/system/memory (holds projects.db)")
    ap.add_argument("--sandbox-root", default=None, help="$GHOST_HOME/sandbox (holds projects/<id>/)")
    ap.add_argument("--project", required=True, help="the project id")
    ap.add_argument("--ask", required=True, help="what the owner asked for")
    ap.add_argument("--added", action="append", default=[], help="a file the change added (project-relative)")
    ap.add_argument("--changed", action="append", default=[], help="a file the change changed")
    ap.add_argument("--removed", action="append", default=[], help="a file the change removed")
    ap.add_argument("--summary", default="", help="one line: what the change did")
    ap.add_argument("--ts", required=True, help="when it was made, local time 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--req", default="", help="the request id that made it")
    ap.add_argument("--deployed", action="store_true", help="it went live")
    ap.add_argument("--url", default="", help="where it went live")
    ap.add_argument("--how", default="put", help="how it went live (put, deploy, switch)")
    ap.add_argument("--lab-id", default="", help="the kept lab it was pushed to")
    ap.add_argument("--reopen", action="store_true",
                    help="a project the old roll-up closed (DONE) is opened again (ACTIVE: ready for changes)")
    ap.add_argument("--apply", action="store_true", help="write it (default: dry run)")
    ap.add_argument("--force", action="store_true", help="add it even if --req is already logged")
    a = ap.parse_args(argv)

    mem = Path(a.memory_dir) if a.memory_dir else _ghost_home() / "system" / "memory"
    sbx = Path(a.sandbox_root) if a.sandbox_root else _ghost_home() / "sandbox"
    if not (mem / "projects.db").exists():
        print(f"no projects.db under {mem}", file=sys.stderr)
        return 2
    if a.url and not a.deployed:
        print("--url is where it went live: pass --deployed with it", file=sys.stderr)
        return 2
    norm = [str(p).strip().lstrip("/") for p in a.added + a.changed + a.removed]
    if any(not p or ".." in p.split("/") for p in norm):
        print("file paths are project-relative (no leading /, no ..)", file=sys.stderr)
        return 2
    ts = _parse_ts(a.ts)
    if not a.apply:
        return _dry_run(mem, a, ts)
    assert_no_other_writer(mem, Path(__file__).name)          # the agent is stopped while it writes
    store = ProjectStore(mem, sandbox_root=sbx)
    proj = store.get_project(a.project)
    if not proj:
        print(f"no project {a.project!r}", file=sys.stderr)
        return 2
    if a.reopen and str(proj.get("status") or "").upper() == "DONE" and not store.closed_by_rollup(a.project):
        store.update_project(a.project, status="ACTIVE")
        store.log_event(a.project, None, "project_reopened",
                        {"reason": "the change log backfill (only the owner closes a project)",
                         "from_status": "DONE"})
        print("reopened: DONE → ACTIVE (ready)")
    e = backfill_change(store, a.project, a.ask, added=a.added, changed=a.changed, removed=a.removed,
                        summary=a.summary, ts=ts, req=a.req, deployed=a.deployed, url=a.url,
                        lab_id=a.lab_id, how=a.how, allow_duplicate=a.force)
    if not e:
        print("not written", file=sys.stderr)
        return 1
    if (e.get("meta") or {}).get("existing"):
        print(f"already in the change log as #{e['num']} — nothing written (--force adds it again)")
        return 0
    print(f"written: change #{e['num']}")
    for e2 in store.list_changes(a.project, limit=3):
        print("  " + change_line(e2))
    return 0


def _dry_run(mem: Path, a, ts: float) -> int:
    """What --apply would do, read from the database opened READ-ONLY (no schema change, no lock)."""
    import sqlite3
    conn = sqlite3.connect(f"file:{mem / 'projects.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT id, title, status FROM projects WHERE id = ?", (a.project,)).fetchone()
        if not row:
            print(f"no project {a.project!r}", file=sys.stderr)
            return 2
        try:
            dup = conn.execute("SELECT num FROM project_changes WHERE project_id = ? AND kind = 'change' AND "
                               "(req_id = ? AND ? != '' OR (ABS(ts - ?) <= 60 AND ask = ?))",
                               (a.project, a.req, a.req, ts, " ".join(a.ask.split())[:300])).fetchone()
        except sqlite3.OperationalError:
            dup = None                                       # no change log table yet
    finally:
        conn.close()
    preview = {"kind": "change", "num": "N", "ts": ts, "ask": a.ask, "added": a.added, "changed": a.changed,
               "removed": a.removed, "summary": a.summary, "deployed": a.deployed}
    print(f"project {a.project} ({row['title']}, {row['status']})")
    print("entry:  " + change_line(preview))
    if a.deployed and a.url:
        print(f"deploy: {a.how} → {a.url}" + (f" ({a.lab_id})" if a.lab_id else ""))
    if dup is not None and not a.force:
        print(f"already in the change log as #{dup['num']} — --apply would write nothing")
    if a.reopen and str(row["status"] or "").upper() == "DONE":
        print("reopen: DONE → ACTIVE (unless the old roll-up closed it: that reads ready already)")
    print("dry run — nothing written (--apply writes it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
