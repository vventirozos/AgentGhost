"""§4KJ one-off (2026-09-24): member data that reached the owner's stores.

Run ONLY with the agent stopped (the vector store has one writer):
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/member_data_cleanup_4kj.py [--apply]

1. Trajectories recorded before `extra["requester_role"]` existed whose
   request id is a `slack-` id are stamped `member` (conservative: an owner
   Slack turn that no longer teaches is harmless; a member turn that teaches
   is the defect). The idle phases read the role, never the id.
2. Playbook lessons derived from member turns are retracted from the JSON
   playbook AND the vector store (`agent_memory`, by source_trajectory_id).
3. (R10, 2026-09-25) Foresight ledger rows written by member turns are
   removed: a row's `req_id` is a member trajectory's session id (after
   step 1). The gate (`gate.json`) is re-aggregated from the ledger by the
   idle phase. The competence profile keeps only Beta counts per domain, so
   its member share cannot be separated and is left as is.
Backups are written next to every file touched. Dry run by default.
"""
import glob, json, os, shutil, sys, time
from pathlib import Path

APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
SYS = HOME / "system"
MEM = SYS / "memory"
STAMP = time.strftime("%Y%m%dT%H%M%S")
# the three lessons whose task is a member's request (mapped by task text to
# slack- trajectories on 2026-09-24; see the journal §4KJ round 5)
MEMBER_LESSON_SOURCES = [s for s in sys.argv[1:] if not s.startswith("--")]


def backup(p: Path):
    b = p.with_name(p.name + f".pre-4kj-{STAMP}.bak")
    shutil.copy2(p, b)
    return b


def stamp_trajectories():
    n_rows = 0
    for f in sorted(glob.glob(str(SYS / "trajectories" / "*" / "*.jsonl"))):
        lines = open(f, encoding="utf-8").read().splitlines()
        out, changed = [], 0
        for line in lines:
            try:
                t = json.loads(line)
            except Exception:
                out.append(line); continue
            ex = t.get("extra") or {}
            if str(t.get("session_id", "")).startswith("slack-") and not ex.get("requester_role"):
                ex["requester_role"] = "member"
                t["extra"] = ex
                changed += 1
                out.append(json.dumps(t, ensure_ascii=False))
            else:
                out.append(line)
        if changed:
            n_rows += changed
            print(f"  {Path(f).name}: {changed} row(s)")
            if APPLY:
                backup(Path(f))
                tmp = f + ".tmp"
                open(tmp, "w", encoding="utf-8").write("\n".join(out) + "\n")
                os.replace(tmp, f)
    print(f"trajectories stamped member: {n_rows}{'' if APPLY else ' (dry run)'}")


def retract_lessons():
    if not MEMBER_LESSON_SOURCES:
        print("no lesson sources given; skipping retraction")
        return
    pb = MEM / "skills_playbook.json"
    lessons = json.load(open(pb, encoding="utf-8"))
    hits = [x for x in lessons if x.get("source_trajectory_id") in MEMBER_LESSON_SOURCES]
    for x in hits:
        print("  lesson:", (x.get("task") or "")[:70].replace("\n", " "))
    if not APPLY:
        print(f"lessons to retract: {len(hits)} (dry run)")
        return
    backup(pb)
    shutil.copy2(MEM / "chroma.sqlite3", MEM / f"chroma.sqlite3.pre-4kj-{STAMP}.bak")
    import chromadb
    from ghost_agent.memory.skills import SkillMemory

    class _Shim:  # the retraction reads `.collection`
        collection = chromadb.PersistentClient(path=str(MEM)).get_collection("agent_memory")
    sm = SkillMemory(MEM)
    total = 0
    for sid in MEMBER_LESSON_SOURCES:
        total += sm.retract_lessons_from_trajectory(sid, memory_system=_Shim())
    print(f"lessons retracted: {total}")


def _member_request_ids() -> set:
    ids = set()
    for f in glob.glob(str(SYS / "trajectories" / "*" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                t = json.loads(line)
            except Exception:
                continue
            ex = t.get("extra") or {}
            sid = str(t.get("session_id", ""))
            # legacy slack- ids are covered by the ledger rule in purge_foresight
            if sid and ex.get("requester_role") == "member":
                ids.add(sid)
    return ids


def _owner_slack_ids() -> set:
    """Slack ids whose trajectory says OWNER explicitly — kept."""
    out = set()
    for f in glob.glob(str(SYS / "trajectories" / "*" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                t = json.loads(line)
            except Exception:
                continue
            if (t.get("extra") or {}).get("requester_role") == "owner":
                out.add(str(t.get("session_id", "")))
    return out


def purge_foresight():
    ids = _member_request_ids()
    owners = _owner_slack_ids()
    n = 0
    for f in sorted(glob.glob(str(SYS / "foresight" / "predictions.jsonl*"))):
        if f.endswith((".bak", ".tmp")):
            continue
        keep, drop = [], 0
        for line in open(f, encoding="utf-8").read().splitlines():
            try:
                rid = json.loads(line).get("req_id")
            except Exception:
                keep.append(line); continue
            rid = rid if isinstance(rid, str) else ""
            # step 1's rule applies to ledger ids too: a legacy slack- id with
            # no trajectory is a member's unless its trajectory says owner (R11)
            if rid and (rid in ids or (rid.startswith("slack-") and rid not in owners)):
                drop += 1
            else:
                keep.append(line)
        if drop:
            n += drop
            print(f"  {Path(f).name}: {drop} row(s)")
            if APPLY:
                backup(Path(f))
                tmp = f + ".tmp"
                open(tmp, "w", encoding="utf-8").write("\n".join(keep) + ("\n" if keep else ""))
                os.replace(tmp, f)
    print(f"foresight rows removed: {n}{'' if APPLY else ' (dry run)'}")


if __name__ == "__main__":
    stamp_trajectories()
    retract_lessons()
    purge_foresight()
