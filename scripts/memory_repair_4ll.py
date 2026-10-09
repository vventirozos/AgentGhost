"""§4LL one-off (2026-10-04, operator: "proceed"): the three file-handling
data actions the review listed —
  1 RETIRE 9 lessons that teach wrong or one-off file advice (or were taught
    by a request that is not a lesson): matched by trigger PREFIX, each must
    match exactly one live lesson; removed via `SkillMemory.remove_by_trigger`
    (archived; vector twin removed)
  2 DELETE the §4KB `git`-tool test episodes 502/508/510/511 (+ their vector
    twins 25977/25989/25996/26000) via `EpisodicMemory.delete_episodes`
    (archived to episodes_forgotten.jsonl). Each must still be a "N tool calls
    only, then stop" test turn, or nothing is applied.
  3 RELABEL the 24 test turns I sent without `X-Ghost-Origin: probe` to
    task_kind="probe" (the label real probes carry, which the collector
    filters); the old kind is kept in extra.relabelled_from.
Every check runs before anything is written. Agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4ll.py [--apply]
"""
import json, os, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
SYSTEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system"
MEM = SYSTEM / "memory"
TRAJ = SYSTEM / "trajectories"

RETIRE = [  # trigger prefixes (review §4LL, lesson numbers from its listing)
    "When using the `file_system` tool to modify a file, always ensure the operation ",          # 127
    "When attempting to modify or search for content in a file, verify that the remem",        # 113
    "Create an image of ζωή Κωνσταντοπούλου",                                                   # 16
    "When querying knowledge bases, include necessary parameters like 'pattern' for f",        # 48
    "When using the file_system tool to read or check for files, verify the exact fil",        # 110
    "When performing file system operations, verify the success of the operation, esp",        # 109
    "When using the 'file_system' tool for downloads, ensure the operation is not blo",        # 138
    "When performing multi-step tool calls, ensure the required number of calls is me",        # 120
    "create a functional web OS , make it look great with glassmorphism",                      # 82
]
EPISODES = [502, 508, 510, 511]
EPISODE_MARK = " tool calls only, then stop."
RELABEL = ("9dcb182b 9ed2d376 865079f9 268550fc 4b0afbc2 85cc2da8 6a8a8d3a 0bf059b0 76b3602e 5e9051bd "
           "75488203 40249c21 07c0c588 3b827526 d5162795 961d939d adc79ddb 0c510d86 626b6232 389be12b "
           "81dbd7cd 5582d869 d940f2b6 9f910ca0").split()


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _trig(row) -> str:
    return (row.get("trigger") or row.get("task") or "") if isinstance(row, dict) else ""


def lessons_to_retire(playbook: list) -> list:
    """The full trigger of each prefix — or SystemExit when a prefix matches
    no lesson or more than one."""
    out = []
    for p in RETIRE:
        hits = [_trig(r) for r in playbook if _trig(r).strip().lower().startswith(p.strip().lower())]
        if len(hits) != 1:
            raise SystemExit(f"lesson {p[:50]!r} matches {len(hits)} lessons — nothing applied")
        out.append(hits[0])
    return out


def check_episodes(db: Path) -> None:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        rows = dict(c.execute(f"SELECT id, trigger FROM episodes WHERE id IN ({','.join('?' * len(EPISODES))})",
                              EPISODES).fetchall())
    for i in EPISODES:
        if EPISODE_MARK not in (rows.get(i) or ""):
            raise SystemExit(f"episode {i} is missing or not a test turn — nothing applied")


def rows_to_relabel(traj: Path) -> dict:
    """{file: [line index, ...]} — every id must match exactly one row."""
    found, where = {i: 0 for i in RELABEL}, {}
    for f in sorted(traj.glob("*/*.jsonl")):
        for n, ln in enumerate(f.read_text(encoding="utf-8").splitlines()):
            try:
                rid = str(((json.loads(ln) or {}).get("extra") or {}).get("req_id") or "")
            except (ValueError, AttributeError):
                continue
            for i in RELABEL:
                if rid.startswith(i):
                    found[i] += 1
                    where.setdefault(f, []).append(n)
    bad = {i: k for i, k in found.items() if k != 1}
    if bad:
        raise SystemExit(f"trajectory ids not matching exactly one row: {bad} — nothing applied")
    return where


def relabel(where: dict) -> int:
    n = 0
    for f, idx in where.items():
        lines = f.read_text(encoding="utf-8").splitlines()
        for i in idx:
            r = json.loads(lines[i])
            if r.get("task_kind") != "probe":
                r.setdefault("extra", {})["relabelled_from"] = r.get("task_kind")
                r["task_kind"] = "probe"
                lines[i] = json.dumps(r, ensure_ascii=False)
                n += 1
        tmp = f.with_suffix(".jsonl.tmp-4ll")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(f)
    return n


def main():
    raw = json.loads((MEM / "skills_playbook.json").read_text())
    playbook = raw if isinstance(raw, list) else (raw.get("playbook") or raw.get("lessons") or [])
    trigs = lessons_to_retire(playbook)
    check_episodes(MEM / "episodic_memory.db")
    where = rows_to_relabel(TRAJ)
    print(f"retire {len(trigs)} lessons, delete {len(EPISODES)} episodes, relabel "
          f"{sum(map(len, where.values()))} rows in {len(where)} files{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, SYSTEM / f"memory.pre-4ll-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    tb = SYSTEM / f"trajectories.pre-4ll-{stamp}"
    tb.mkdir()
    for f in where:
        (tb / f.parent.name).mkdir(exist_ok=True)
        shutil.copy2(f, tb / f.parent.name / f.name)
    from ghost_agent.memory.episodes import EpisodicMemory
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in trigs)
    em = EpisodicMemory(MEM)
    eg = em.delete_episodes(EPISODES, vm, reason="§4LL: §4KB git-tool test turns (operator confirmed)")
    nr = relabel(where)
    print(f"lessons retired {gone}/{len(trigs)}; episodes deleted {eg}/{len(EPISODES)} "
          f"(twin failures: {em.last_twin_failures}); rows relabelled {nr}")


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
