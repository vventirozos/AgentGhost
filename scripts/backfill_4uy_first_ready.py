#!/usr/bin/env python3
"""§4UY P2: repair the change log of projects whose first plan finished BEFORE §4UM.

Such a project got ``metadata.first_ready_at`` / ``first_ready_req`` at its NEXT plan finish, naming that later
request — so the later request's work read as "the first plan" and no change entry was written for it. The
code now reads the older roll-up first (`memory.projects._was_ready_before`) and stamps a legacy finish with no
request; this script repairs the records already written:

  1. the stamp: ``first_ready_at`` = the first finished roll-up, ``first_ready_req`` = "" (no request);
  2. one change entry for the swallowed request's work, rebuilt from the records: the tasks that plan closed
     (created after the last change entry, closed by the stamp), the files they registered, the files the work
     log and the deploy entries of that span name, and the files in the folder whose content moved in that span.
     A file born in the span is "added", else "changed"; the entry is live when a deploy entry of the span
     carried every file. The ask is the work log's request text of the span (read from the database at run time
     — nothing about any project is built in).

Dry run by default (works on a COPY of projects.db: nothing is written); ``--apply`` writes, and refuses while
the agent runs (stop it, apply, start it again). Idempotent: a project already repaired (stamp with no request)
or an entry already logged for the request is left as it is.

  python scripts/backfill_4uy_first_ready.py --memory-dir "$GHOST_HOME/system/memory" \\
      --sandbox-root "$GHOST_HOME/sandbox" [--project <id>] [--apply]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ghost_agent.memory.projects import (ProjectStore, _skip_change_path, backfill_change,  # noqa: E402
                                         change_line, legacy_ready_at, push_covers)

#: how far before the swallowed plan's first task, and after its finish, the request's own writes are looked for
SPAN_SLACK_S = 600.0


def _ghost_home() -> Path:
    return Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))


def _rel(project_id: str, p: Any) -> str:
    s = str(p or "").strip().replace("\\", "/")
    for pre in (f"/workspace/projects/{project_id}/", f"projects/{project_id}/", "./"):
        if s.startswith(pre):
            s = s[len(pre):]
    return "" if (not s or s.startswith("/") or ".." in s.split("/")) else s


def plan_repair(store: ProjectStore, project_id: str) -> Optional[Dict[str, Any]]:
    """What the repair writes for ``project_id``, or None when it needs none."""
    proj = store.get_project(project_id)
    if not proj:
        return None
    meta = proj.get("metadata") or {}
    try:
        first_at = float(meta.get("first_ready_at") or 0.0)
    except (TypeError, ValueError):
        return None
    req = str(meta.get("first_ready_req") or "")
    if not first_at or not req:
        return None                                     # never stamped, or already a legacy stamp
    legacy = legacy_ready_at(store, project_id, before=first_at - 1.0)
    if legacy is None:
        return None                                     # the stamp IS the first plan's: nothing swallowed
    out: Dict[str, Any] = {"project": project_id, "title": proj.get("title"), "req": req,
                           "stamp": {"first_ready_at": legacy, "first_ready_req": ""}, "entry": None}
    changes = store.list_changes(project_id)
    if any(c.get("kind") == "change" and c.get("req_id") == req for c in changes):
        return out                                      # its entry exists: only the stamp
    prev_cut = max([legacy] + [float(c["ts"]) for c in changes
                               if c.get("kind") == "change" and float(c["ts"]) < first_at])
    next_cut = min([float(c["ts"]) for c in changes if c.get("kind") == "change" and float(c["ts"]) > first_at]
                   or [float("inf")])
    tasks = [t for t in store.list_tasks(project_id) or []
             if str(t.get("status") or "").upper() == "DONE"
             and prev_cut < float(t.get("created_at") or 0) and float(t.get("updated_at") or 0) <= first_at + 1.0]
    if not tasks:
        return out
    lo = max(prev_cut, min(float(t["created_at"]) for t in tasks) - SPAN_SLACK_S)
    hi = min(next_cut, first_at + SPAN_SLACK_S)
    files: Dict[str, str] = {}                          # rel -> evidence
    tids = {t["id"] for t in tasks}
    for a in store.list_artifacts(project_id=project_id):
        if a.get("task_id") in tids and str(a.get("kind")) == "file":
            r = _rel(project_id, a.get("payload"))
            if r:
                files.setdefault(r, "artifact")
    asks: List[str] = []
    for ev in store.list_events(project_id, limit=500, event_type="work_log") or []:
        try:
            ts = float(ev.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if not (lo <= ts <= hi):
            continue
        pl = ev.get("payload") or {}
        for f in pl.get("files") or []:
            r = _rel(project_id, f)
            if r:
                files.setdefault(r, "work log")
        q = str(pl.get("request") or "").strip()
        if q and not q.startswith("[autoadvance]"):
            asks.append((ts, q))
    deploys = [c for c in changes if c.get("kind") == "deploy" and lo <= float(c["ts"]) <= hi]
    for d in deploys:
        for f in d.get("changed") or []:
            r = _rel(project_id, f)
            if r and not r.endswith("/"):
                files.setdefault(r, "deploy")
    root = Path(store._default_workspace(project_id) or "")
    if root.is_dir():
        for p in root.rglob("*"):
            try:
                if p.is_file() and lo <= p.stat().st_mtime <= hi:
                    files.setdefault(p.relative_to(root).as_posix(), "folder")
            except OSError:
                continue
    added, changed, last_t = [], [], 0.0
    for r in sorted(files):
        if _skip_change_path(r):
            continue
        full = root / r
        if not full.is_file():
            continue                                    # gone since: not this entry's to name
        st = full.stat()
        born = getattr(st, "st_birthtime", None)
        (added if born is not None and float(born) >= lo else changed).append(r)
        if lo <= st.st_mtime <= hi:
            last_t = max(last_t, st.st_mtime)
    if not (added or changed):
        return out
    live = added + changed
    deployed = any(all(push_covers(d.get("changed") or [], r) for r in live) for d in deploys) or any(
        (d.get("meta") or {}).get("whole") for d in deploys)
    ask = (sorted(asks)[-1][1] if asks else f"(backfilled §4UY: the work of request {req})")
    out["entry"] = {"ask": ask, "added": added, "changed": changed, "ts": last_t or first_at,
                    "deployed": bool(deployed), "summary": f"backfilled: the work that finished the plan "
                                                           f"({len(tasks)} task(s)) in request {req}"}
    return out


def apply_repair(store: ProjectStore, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pid = plan["project"]
    st = plan["stamp"]

    def _mut(meta):
        meta["first_ready_at"], meta["first_ready_req"] = st["first_ready_at"], st["first_ready_req"]
        return meta
    store._atomic_metadata_update(pid, _mut)
    e = plan.get("entry")
    if not e:
        return None
    return backfill_change(store, pid, e["ask"], added=e["added"], changed=e["changed"], summary=e["summary"],
                           ts=e["ts"], req=plan["req"], deployed=e["deployed"])


def _print(plan: Dict[str, Any]) -> None:
    print(f"project {plan['project']} ({plan['title']}): stamp first_ready_req {plan['req']} → none "
          f"(first plan finished {plan['stamp']['first_ready_at']:.0f})")
    e = plan.get("entry")
    if e:
        print("  entry: " + change_line({"kind": "change", "num": "N", **e}))
    else:
        print("  entry: none needed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-dir", default=None, help="$GHOST_HOME/system/memory (holds projects.db)")
    ap.add_argument("--sandbox-root", default=None, help="$GHOST_HOME/sandbox (holds projects/<id>/)")
    ap.add_argument("--project", default="", help="one project id (default: every project)")
    ap.add_argument("--apply", action="store_true", help="write it (default: dry run on a copy)")
    a = ap.parse_args(argv)
    mem = Path(a.memory_dir) if a.memory_dir else _ghost_home() / "system" / "memory"
    sbx = Path(a.sandbox_root) if a.sandbox_root else _ghost_home() / "sandbox"
    if not (mem / "projects.db").exists():
        print(f"no projects.db under {mem}", file=sys.stderr)
        return 2
    tmp = None
    if a.apply:
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, Path(__file__).name)
        db_dir = mem
    else:
        tmp = tempfile.mkdtemp(prefix="backfill-4uy-")
        for f in mem.glob("projects.db*"):
            shutil.copy2(f, Path(tmp) / f.name)
        db_dir = Path(tmp)
    try:
        store = ProjectStore(db_dir, sandbox_root=sbx)
        ids = [a.project] if a.project else [p["id"] for p in store.list_projects() or []]
        n = 0
        for pid in ids:
            plan = plan_repair(store, pid)
            if plan is None:
                continue
            n += 1
            _print(plan)
            if a.apply:
                e = apply_repair(store, plan)
                if e is not None:
                    print(f"  written: change #{e['num']}"
                          + (" (already logged)" if (e.get("meta") or {}).get("existing") else ""))
                print("  stamp repaired")
        if not n:
            print("nothing to repair")
        if not a.apply:
            print("dry run — nothing written (--apply writes it)")
        return 0
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
