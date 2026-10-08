#!/usr/bin/env python3
"""§4MI (2026-10-08) — remove non-owner residue from the owner's memory stores.

Three reviewers measured, on a snapshot of the live `system/` dir:

* the autobiography (`selfhood/autobiographical.jsonl`) held 21 rows captured
  from channel MEMBERS' turns (2026-09-23/24, before the §4KJ member gate)
  and 17 rows from PROBE turns whose ids carried no `probe-` prefix — and
  `recall_relevant` served them to the owner ("what is my name" → five
  member rows);
* the knowledge graph held triplets written during those member threads
  (`generated images DEPICTS classical roman masturbatorium`, `user
  REQUESTED dvda schematic`, …) that the owner's prompts hydrated;
* the graph mirrored an ON-DEMAND profile field (`user HAS_FOTINI_DESCRIPTION
  …`) that any query word inside it seeded into a prompt;
* the workspace activity log held probe-driven events (§4LN/§4LO, 10-04/05)
  with no project id, which every project's wake-up prefix kept forever, and
  the workspace narrative was built from them.

The repair is a JOIN against the trajectory corpus — rows are classified by
their trajectory's `requester_role` / `task_kind`, or by their timestamp
falling inside a member or probe turn's window — never by content. Every
removed row is ARCHIVED beside its store (`*.removed-4mi.jsonl`, or the
graph's own archive with reason `member-thread-4mi` / `on-demand-mirror-4mi`),
and the stores are backed up first. Dry run by default; `--apply` writes.

    python scripts/memory_repair_4mi.py --home /path/to/GHOST_HOME [--apply]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path

WINDOW_PAD_S = 120.0          # a member/probe turn's writes land within this of its stamp


def _parse_ts(s: str) -> float | None:
    s = str(s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        try:
            return dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        except Exception:  # noqa: BLE001
            return None


def _iter_trajectory_rows(root: Path):
    for day in sorted(root.iterdir()) if root.is_dir() else []:
        if day.is_dir() and len(day.name) == 10:
            for f in sorted(day.glob("*.jsonl")):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if ln:
                            try:
                                yield json.loads(ln)
                            except Exception:  # noqa: BLE001
                                continue
    arch = root / "archive"
    if arch.is_dir():
        for tgz in sorted(arch.glob("*.tar.gz")):
            try:
                with tarfile.open(tgz) as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.endswith(".jsonl"):
                            fh = tf.extractfile(m)
                            for ln in (fh.read().decode("utf-8", "replace").splitlines() if fh else []):
                                if ln.strip():
                                    try:
                                        yield json.loads(ln)
                                    except Exception:  # noqa: BLE001
                                        continue
            except Exception:  # noqa: BLE001
                continue


def classify_trajectories(home: Path):
    """id → class ('member' | 'probe' | 'owner'), and the member/probe time windows."""
    cls, windows = {}, []
    for row in _iter_trajectory_rows(home / "system" / "trajectories"):
        tid = str(row.get("id") or "")
        extra = row.get("extra") or {}
        role = str(extra.get("requester_role") or extra.get("source_requester_role") or "").lower()
        kind = str(row.get("task_kind") or "")
        rid = str(extra.get("req_id") or "")
        if role == "member":
            c = "member"
        elif kind == "probe" or rid.startswith("probe-"):
            c = "probe"
        else:
            c = "owner"
        if tid:
            cls[tid] = c
        ts = _parse_ts(row.get("timestamp"))
        if ts is not None:
            dur = float(row.get("duration_s") or 0.0)
            # r2 review: a trajectory's timestamp is stamped when the turn
            # ENDS (built after the reply) — the turn's body is BEFORE it
            windows.append((c, ts - dur - 5.0, ts + WINDOW_PAD_S))
    return cls, windows


def _in_window(ts: float | None, windows, kinds=("member", "probe")) -> str:
    if ts is None:
        return ""
    for c, a, b in windows:
        if c in kinds and a <= ts <= b:
            return c
    return ""


def repair_autobiography(home: Path, cls, apply: bool) -> dict:
    path = home / "system" / "selfhood" / "autobiographical.jsonl"
    if not path.is_file():
        return {"present": False}
    keep, drop = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except Exception:  # noqa: BLE001
                keep.append(ln.rstrip("\n")); continue
            c = cls.get(str(row.get("trajectory_id") or ""), "owner")
            (drop if c in ("member", "probe") else keep).append((c, ln.rstrip("\n")) if c != "owner" else ln.rstrip("\n"))
    out = {"present": True, "rows": len(keep) + len(drop), "removed": len(drop),
           "by_class": {c: sum(1 for x in drop if x[0] == c) for c in ("member", "probe")}}
    if apply and drop:
        shutil.copy2(path, path.with_suffix(".jsonl.pre-4mi.bak"))
        with open(path.with_suffix(".removed-4mi.jsonl"), "a", encoding="utf-8") as fh:
            for c, ln in drop:
                fh.write(json.dumps({"reason": f"{c}-turn-4mi", "row": json.loads(ln)}, ensure_ascii=False) + "\n")
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        os.replace(tmp, path)
    return out


def repair_graph(home: Path, windows, apply: bool) -> dict:
    db = home / "system" / "memory" / "knowledge_graph.db"
    if not db.is_file():
        return {"present": False}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT rowid, subject, predicate, object, weight, timestamp FROM triplets").fetchall()
    doomed = []
    for rid, s, p, o, w, ts in rows:
        _t = _parse_ts(ts)
        c = _in_window(_t, windows)
        # a row written during a member/probe turn — unless an OWNER turn
        # overlapped that window too (ambiguous → kept)
        if c and not _in_window(_t, windows, kinds=("owner",)):
            doomed.append((c + "-thread-4mi", rid, s, p, o, w, ts))
        elif str(s).lower() == "user" and str(p).upper().endswith("_DESCRIPTION"):
            doomed.append(("on-demand-mirror-4mi", rid, s, p, o, w, ts))
    out = {"present": True, "rows": len(rows), "removed": len(doomed),
           "samples": [(r[0], r[2], r[3], str(r[4])[:40]) for r in doomed[:12]]}
    if apply and doomed:
        shutil.copy2(db, db.with_suffix(".db.pre-4mi.bak"))
        arch = db.parent / "graph_pruned_archive.jsonl"
        now = dt.datetime.now().timestamp()
        with open(arch, "a", encoding="utf-8") as fh:
            for reason, rid, s, p, o, w, ts in doomed:
                fh.write(json.dumps({"archived_at": now, "reason": reason, "subject": s, "predicate": p,
                                     "object": o, "weight": w, "timestamp": ts}, ensure_ascii=False) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        con.executemany("DELETE FROM triplets WHERE rowid = ?", [(r[1],) for r in doomed])
        con.commit()
    con.close()
    return out


def repair_workspace(home: Path, windows, apply: bool) -> dict:
    ws = home / "system" / "workspace"
    path = ws / "activity.jsonl"
    if not path.is_file():
        return {"present": False}
    keep, drop = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except Exception:  # noqa: BLE001
                keep.append(ln.rstrip("\n")); continue
            _t = _parse_ts(row.get("timestamp"))
            c = _in_window(_t, windows)
            if c and not _in_window(_t, windows, kinds=("owner",)) and not str(row.get("project_id") or ""):
                drop.append((c, ln.rstrip("\n")))
            else:
                keep.append(ln.rstrip("\n"))
    narrative = ws / "narrative.txt"
    out = {"present": True, "rows": len(keep) + len(drop), "removed": len(drop),
           "narrative_reset": narrative.is_file()}
    if apply:
        if drop:
            shutil.copy2(path, path.with_suffix(".jsonl.pre-4mi.bak"))
            with open(path.with_suffix(".removed-4mi.jsonl"), "a", encoding="utf-8") as fh:
                for c, ln in drop:
                    fh.write(json.dumps({"reason": f"{c}-turn-4mi", "row": json.loads(ln)}, ensure_ascii=False) + "\n")
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            os.replace(tmp, path)
        # the narrative was built from the removed rows: move it aside with
        # its input key so the next idle phase regenerates it from owner rows
        for f in (narrative, ws / "narrative.txt.inputkey", ws / "narrative.inputkey"):
            if f.is_file():
                shutil.move(str(f), str(f) + ".pre-4mi.bak")
    return out


def correct_r2(home: Path, windows, apply: bool) -> dict:
    """r2: the first live run used windows anchored at the wrong end of each
    turn. Re-judge every row it removed with the corrected windows and put
    back the ones they no longer condemn (workspace: merged back in
    timestamp order; graph: re-inserted). The residue the wrong windows
    missed is then removed by the ordinary repair."""
    out = {"workspace_restored": 0}
    ws = home / "system" / "workspace"
    removed = ws / "activity.removed-4mi.jsonl"
    act = ws / "activity.jsonl"
    if removed.is_file() and act.is_file():
        keep_removed, back = [], []
        for ln in removed.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            row = rec.get("row") or {}
            t = _parse_ts(row.get("timestamp"))
            c = _in_window(t, windows)
            if c and not _in_window(t, windows, kinds=("owner",)):
                keep_removed.append(ln)
            else:
                back.append(row)
        out["workspace_restored"] = len(back)
        out["workspace_restored_samples"] = [str(r.get("summary") or "")[:60] for r in back[:5]]
        if apply and back:
            rows = [json.loads(l) for l in act.read_text(encoding="utf-8").splitlines() if l.strip()]
            have = {r.get("id") for r in rows}
            rows += [r for r in back if r.get("id") not in have]
            rows.sort(key=lambda r: _parse_ts(r.get("timestamp")) or 0.0)
            shutil.copy2(act, act.with_suffix(".jsonl.pre-4mi-r2.bak"))
            tmp = act.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            os.replace(tmp, act)
            removed.write_text("\n".join(keep_removed) + ("\n" if keep_removed else ""), encoding="utf-8")
    # The GRAPH is not re-judged: its rows are written by the consolidation
    # drain AFTER the turn ends (and an upsert re-stamps them), so the
    # body window cannot place them — the first run's post-turn window is
    # the one that matches how they were written, and none of its 47 rows
    # fell inside an owner turn (r2 review, verified).
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--correct-r2", action="store_true",
                    help="restore rows the first (wrongly windowed) run removed, then repair the residue")
    a = ap.parse_args(argv)
    home = Path(a.home)
    cls, windows = classify_trajectories(home)
    if a.correct_r2:
        print(json.dumps({"correction": correct_r2(home, windows, a.apply)}, indent=2, ensure_ascii=False))
    report = {
        "trajectories": {"classified": len(cls),
                         "member": sum(1 for v in cls.values() if v == "member"),
                         "probe": sum(1 for v in cls.values() if v == "probe"),
                         "windows": len(windows)},
        "autobiography": repair_autobiography(home, cls, a.apply),
        # r2: the graph pass is NOT re-run — graph rows are written after
        # the turn (consolidation drain, upserts re-stamp), so a turn's body
        # window cannot place them; the first run's 48 removals stand
        "graph": (repair_graph(home, windows, a.apply) if not a.correct_r2 else {"skipped": "r2"}),
        "workspace": repair_workspace(home, windows, a.apply),
        "applied": bool(a.apply),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
