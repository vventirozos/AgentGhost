"""§4LL follow-up (2026-10-04, operator: "also fix the 3 selfhood entries about
the old git test turns and the separate lesson from 09-23"):
  1 DROP the diary rows (`selfhood/autobiographical.jsonl`) of the §4KB test
    turns that used the `git` tool or wrote `.git/config` — six rows, not the
    three the review counted (by id; each must still be a "tool call(s) only,
    then stop" test turn, or nothing is applied). The two non-git test rows of
    that series (dup_probe, r5) are not touched.
  2 RETIRE the request-scoped image lesson of 09-23 that names the same person
    as lesson #16 (`SkillMemory.remove_by_trigger`: archived, twin removed).
Every check runs before anything is written. Agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4ll_b.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
SYSTEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system"
MEM = SYSTEM / "memory"
DIARY = SYSTEM / "selfhood" / "autobiographical.jsonl"

DIARY_IDS = {  # id prefix -> what the turn was
    "941fcc3c": "kb_probe symbols + git status",
    "977d51d5": "git status",
    "e20c93e9": "rv_probe outline + git status",
    "b4808853": "r3probe outline/write + git",
    "5d6136ed": "r4 outline + .git/config write",
    "410a93d4": "r6 outline/replace + .git/config write",
}
DIARY_MARK = "only, then stop"
LESSON = ("Create a photorealistic image that looks like a real photograph rather than digital art or CGI. "
          "Use physically accurate lighting")
LESSON_NAME = "ζωή Κωνσταντοπούλου"


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def diary_split(lines: list) -> tuple:
    """(kept lines, dropped ids) — or SystemExit when an id matches no row,
    more than one, or a row that is not a test turn."""
    keep, hits = [], {p: [] for p in DIARY_IDS}
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            keep.append(ln)
            continue
        p = next((p for p in DIARY_IDS if str(r.get("id", "")).startswith(p)), None)
        if p is None:
            keep.append(ln)
            continue
        if DIARY_MARK not in (r.get("summary") or "")[:120]:
            raise SystemExit(f"diary row {p} is not a test turn — nothing applied")
        hits[p].append(r["id"])
    bad = {p: len(v) for p, v in hits.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"diary ids not matching exactly one row: {bad} — nothing applied")
    return keep, [v[0] for v in hits.values()]


def lesson_trigger(playbook: list) -> str:
    def t(r):
        return (r.get("trigger") or r.get("task") or "") if isinstance(r, dict) else ""
    hits = [t(r) for r in playbook if t(r).startswith(LESSON) and LESSON_NAME in t(r)]
    if len(hits) != 1:
        raise SystemExit(f"the image lesson matches {len(hits)} lessons — nothing applied")
    return hits[0]


def main():
    lines = DIARY.read_text(encoding="utf-8").splitlines()
    keep, dropped = diary_split(lines)
    raw = json.loads((MEM / "skills_playbook.json").read_text())
    trig = lesson_trigger(raw if isinstance(raw, list) else (raw.get("playbook") or raw.get("lessons") or []))
    print(f"drop {len(dropped)} diary rows ({len(lines)} -> {len(keep)}), retire 1 lesson"
          f"{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copy2(DIARY, DIARY.with_name(f"autobiographical.jsonl.pre-4ll-{stamp}.bak"))
    shutil.copytree(MEM, SYSTEM / f"memory.pre-4ll-b-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    tmp = DIARY.with_suffix(".jsonl.tmp-4ll")
    tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
    tmp.replace(DIARY)
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    gone = SkillMemory(MEM).remove_by_trigger(trig, memory_system=vm)
    print(f"diary rows dropped {len(dropped)}; lesson retired {bool(gone)}")


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
    main()
