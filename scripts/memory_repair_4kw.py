"""§4KW one-off (2026-10-02) — memory repair the operator chose after the
fresh reviews of the memory and lesson systems.

1. PROFILE: restore `root.name` and the `relationships` category. A channel
   member's empty `update_profile` removed root.name (slack-3851e437,
   2026-09-24); the relationships category vanished by an unattributed path.
   Values from the 2026-09-04 backups (the consistent "sons" line from
   `pre-temporal-…083301`, the birthdates from `pre-consolidate-…`) and the
   owner's 2026-09-06 `fotini_description` write. Every other key is kept.
2. EPISODES: remove the episodes written by probe turns and by channel
   members before the member wall (ids from `--episode-ids`, a JSON file
   {"probe": [...], "member": [...]} produced by matching each episode's
   trigger + time to a probe / member trajectory), with their action rows
   and their vector copies (`episode_id` metadata).
3. LESSONS: retract the dream restatements of test prompts and the one
   lesson minted from a sub-agent leaf, by exact trigger (`remove_by_trigger`
   archives each first and deletes its vector copy).

Run ONLY with the agent stopped (the vector store and the episodic DB have
one writer):
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4kw.py \\
    --episode-ids <file.json> [--apply]
Backups are written next to every file touched. Dry run by default.
"""
import json, os, shutil, sqlite3, sys, time
from pathlib import Path

APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = HOME / "system" / "memory"
STAMP = time.strftime("%Y%m%dT%H%M%S")

PROFILE_RESTORE = {
    ("root", "name"): ("Vasilis", "2026-09-04T07:52:51"),
    ("relationships", "wife_name"): ("Fotini", "2026-09-04T07:52:51"),
    ("relationships", "sons"): ("Thodoris (born 2016-11-25) and Leonidas (born 2026-03-12)", "2026-09-04T08:33:01"),
    ("relationships", "son_thodoris_birthdate"): (
        "Thodoris born November 25, 2016 (Vasilis and Fotini's older son)", "2026-09-04T07:52:51"),
    ("relationships", "son_lleonidas_birthdate"): (
        "Leonidas born March 12, 2026 (Vasilis and Fotini's younger son)", "2026-09-04T07:52:51"),
    ("relationships", "fotini_description"): (
        "Fotini is Vasilis's wife. Recognizable appearance: tan/sunny complexion, long straight hair with "
        "brown/blonde highlights and some gray at the temples, light skin detail. In one photo "
        "(photo-20260906-102919.jpg) she wears a pink/lavender tank top + lavender shorts on an outdoor tiled "
        "terrace.", "2026-09-06T07:35:23"),
}

RETRACT_TRIGGERS = (
    "When executing commands in the sandbox, use the `echo` command followed by a con",
    "When responding to greetings, use system_utility to check the time or weather if",
    "When performing web searches requiring specific constraints (e.g., 'in at most t",
    "When performing system checks, adhere strictly to the requested output format (e",
    "When managing services, use action='restart' instead of relying on implicit rest",
    "When performing complex tasks, ensure the output is strictly limited to the requ",
)
#: the leaf lesson's trigger is long; matched by this exact prefix
RETRACT_TRIGGER_PREFIXES = (
    "BUILD TASK (one leaf of a project): Extend app.py: add create_app(storage_path=None)",
)


_CLIENT = None


def _collection():
    """ONE Chroma client per process: a second PersistentClient on the same
    path raises "an instance of Chroma already exists … different settings"."""
    global _CLIENT
    import chromadb
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(MEM))
    return _CLIENT.get_collection("agent_memory")


def backup(p: Path) -> Path:
    b = p.with_name(p.name + f".pre-4kw-repair-{STAMP}.bak")
    shutil.copy2(p, b)
    return b


def restore_profile():
    p = MEM / "user_profile.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    changes = []
    for (cat, key), (val, as_of) in PROFILE_RESTORE.items():
        cur = (data.get(cat) or {}).get(key)
        cur_v = cur.get("v") if isinstance(cur, dict) else cur
        if cur_v == val:
            continue
        changes.append((cat, key, cur_v, val))
        data.setdefault(cat, {})
        if not isinstance(data[cat], dict):
            data[cat] = {}
        data[cat][key] = {"v": val, "as_of": as_of}
    for cat, key, old, new in changes:
        print(f"  profile {cat}.{key}: {old!r} -> {new[:60]!r}")
    print(f"profile keys restored: {len(changes)}{'' if APPLY else ' (dry run)'}")
    if changes and APPLY:
        backup(p)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)


def remove_episodes(ids_file: str):
    ids = json.loads(Path(ids_file).read_text())
    eids = sorted({int(i) for k in ("probe", "member") for i in ids.get(k, [])})
    db = MEM / "episodic_memory.db"
    con = sqlite3.connect(str(db))
    present = [r[0] for r in con.execute(
        f"select id from episodes where id in ({','.join('?' * len(eids))})", eids)] if eids else []
    print(f"episodes to remove: {len(present)} of {len(eids)} listed"
          f" (probe {len(ids.get('probe', []))}, member {len(ids.get('member', []))}){'' if APPLY else ' (dry run)'}")
    if not (APPLY and present):
        con.close()
        return
    con.close()
    backup(db)
    con = sqlite3.connect(str(db))
    with con:
        q = ",".join("?" * len(present))
        con.execute(f"delete from episode_actions where episode_id in ({q})", present)
        con.execute(f"delete from episodes where id in ({q})", present)
    con.close()
    col = _collection()
    n = 0
    for e in present:
        got = col.get(where={"episode_id": int(e)})
        if got.get("ids"):
            col.delete(ids=got["ids"])
            n += len(got["ids"])
    print(f"episodes removed: {len(present)}; vector copies removed: {n}")


def retract_lessons():
    pb = MEM / "skills_playbook.json"
    lessons = json.loads(pb.read_text(encoding="utf-8"))
    want = {t.strip().lower() for t in RETRACT_TRIGGERS}
    hits = []
    for x in lessons:
        t = (x.get("trigger") or x.get("task") or "").strip()
        if t.lower() in want or any(t.startswith(pfx) for pfx in RETRACT_TRIGGER_PREFIXES):
            hits.append(t)
    for t in hits:
        print("  lesson:", t[:90].replace("\n", " "))
    print(f"lessons to retract: {len(hits)}{'' if APPLY else ' (dry run)'}")
    if not (APPLY and hits):
        return
    backup(pb)
    shutil.copy2(MEM / "chroma.sqlite3", MEM / f"chroma.sqlite3.pre-4kw-repair-{STAMP}.bak")
    from ghost_agent.memory.skills import SkillMemory

    class _Shim:
        collection = _collection()
    sm = SkillMemory(MEM)
    done = sum(bool(sm.remove_by_trigger(t, memory_system=_Shim())) for t in hits)
    print(f"lessons retracted: {done}")


# §4MR: a store writer must be the ONLY writer — refuse beside the running
# agent (two chroma writers left the store segfaulting on open, §4MN)
if __name__ == "__main__" and "--apply" in __import__("sys").argv:
    import os as _os4mr, sys as _sys4mr
    from pathlib import Path as _P4mr
    _sys4mr.path.insert(0, str(_P4mr(__file__).resolve().parents[1] / "src"))
    from ghost_agent.memory.store_lock import assert_no_other_writer as _no_other_writer
    _no_other_writer(_P4mr(_os4mr.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory",
                     _P4mr(__file__).name)


if __name__ == "__main__":
    ids_file = sys.argv[sys.argv.index("--episode-ids") + 1] if "--episode-ids" in sys.argv else None
    if (MEM / "chroma.sqlite3").exists() and APPLY:
        # a DISTINCT name: retract_lessons() takes its own copy later in the
        # run, and one shared name let it overwrite this one (data audit)
        shutil.copy2(MEM / "chroma.sqlite3", MEM / f"chroma.sqlite3.pre-4kw-repair-{STAMP}-start.bak")
    restore_profile()
    if ids_file:
        remove_episodes(ids_file)
    retract_lessons()
