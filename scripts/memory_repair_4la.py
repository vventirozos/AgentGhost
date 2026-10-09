"""§4LA one-off (2026-10-03): the episode cleanup the operator confirmed — "yes
to all, proceed":
  1 RELABEL the episodes that said SUCCESS for a turn the agent's own records
    call failed (trajectory outcome failed / a late refute in corrections.jsonl,
    or the agent's failure sentinels in the reply);
  2 DELETE legacy probe/test episodes and chess-engine prompts (written before
    the probe gate);
  3 DELETE the episodes on the sensitive search topics (the graph copies were
    removed in §4KY/§4KZ);
  4 PRUNE backups: keep the newest two `memory.pre-*` folders, delete the older
    ones and the single-file `*.bak` / `*.pre-*` copies inside memory/.
Deletions go through `EpisodicMemory.delete_episodes` (archived 30 days, twins
removed). Every id must exist, or nothing is applied. Agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4la.py [--apply]
"""
import os, re, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
SYSTEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system"
MEM = SYSTEM / "memory"
RELABEL = [1, 3, 6, 12, 14, 15, 18, 27, 28, 33, 42, 45, 48, 52, 54, 58, 59, 62, 63, 64, 65, 69, 72, 73, 74, 75, 76, 95, 101, 103, 104, 106, 110, 116, 117, 123, 125, 127, 129, 130, 133, 136, 137, 139, 140, 141, 144, 145, 147, 148, 149, 150, 154, 155, 157, 158, 160, 161, 166, 170, 172, 174, 175, 185, 187, 188, 191, 192, 194, 198, 199, 202, 206, 207, 211, 212, 214, 218, 219, 222, 228, 232, 233, 234, 237, 241, 243, 244, 251, 253, 254, 257, 278, 281, 284, 288, 289, 291, 296, 297, 302, 306, 309, 311, 315, 321, 331, 333, 334, 342, 347, 349, 360, 367, 369, 370, 374, 397, 402, 403, 408, 413, 414, 418, 436, 441, 483, 494, 502, 505, 507, 508, 509, 510, 511, 538]
DELETE = {
    "probe": [176, 177, 180, 237, 238, 241, 250, 251, 321, 375, 503, 504, 505, 507, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523],
    "chess": [12, 14, 15, 45, 47, 48, 49, 52],
    "sensitive": [60, 61, 64, 65, 305, 312, 313, 531, 543],
}
KEEP_BACKUP_DIRS = 2
_SINGLE_BACKUP = re.compile(r"\.bak($|-)|\.pre[-_]|\.bak-")


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def check(ids: set) -> tuple:
    """(relabel, delete) restricted to existing ids — or SystemExit when a
    listed id is missing (the store moved under the list)."""
    dele = [i for g in DELETE.values() for i in g]
    missing = [i for i in RELABEL + dele if i not in ids]
    if missing:
        raise SystemExit(f"ids no longer present: {missing[:10]} — nothing applied")
    if set(RELABEL) & set(dele):
        relabel = [i for i in RELABEL if i not in set(dele)]
    else:
        relabel = list(RELABEL)
    return relabel, dele


def _stamp(p: Path) -> str:
    """The backup's own time, from its NAME (copytree copies the source
    folder's mtime, so a fresh backup can look older than an earlier one)."""
    m = re.search(r"(\d{8}T\d{6})", p.name)
    return m.group(1) if m else time.strftime("%Y%m%dT%H%M%S", time.localtime(p.stat().st_mtime))


def backups_to_delete(system: Path, memdir: Path, keep: Path = None) -> list:
    dirs = sorted((p for p in system.glob("memory.pre-*") if p.is_dir() and p != keep), key=_stamp)
    n_keep = KEEP_BACKUP_DIRS - (1 if keep is not None else 0)
    old = dirs[:-n_keep] if n_keep > 0 and len(dirs) > n_keep else (dirs if n_keep <= 0 else [])
    files = [p for p in memdir.iterdir() if p.is_file() and _SINGLE_BACKUP.search(p.name)]
    return old + files


def main():
    with sqlite3.connect(f"file:{MEM / 'episodic_memory.db'}?mode=ro", uri=True) as c:
        ids = {r[0] for r in c.execute("SELECT id FROM episodes")}
    relabel, dele = check(ids)
    print(f"relabel {len(relabel)}, delete {len(dele)} episodes{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        print("backups that would go:", len(backups_to_delete(SYSTEM, MEM)))
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    _mine = SYSTEM / f"memory.pre-4la-{stamp}.bak"
    shutil.copytree(MEM, _mine,
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*", "episodes_forgotten*"))
    from ghost_agent.memory.episodes import EpisodicMemory
    from ghost_agent.memory.vector import VectorMemory
    em = EpisodicMemory(MEM)
    with sqlite3.connect(em.db_path) as c:
        c.executemany("UPDATE episodes SET outcome_success = 0, outcome = substr('[the turn failed] ' || outcome, 1, 1000) "
                      "WHERE id = ? AND outcome NOT LIKE '[the turn failed]%'", [(i,) for i in relabel])
        c.commit()
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    gone = 0
    for group, g in DELETE.items():
        gone += em.delete_episodes(g, vm, reason=f"§4LA cleanup: {group}")
    pruned = []
    for p in backups_to_delete(SYSTEM, MEM, keep=_mine):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        pruned.append(p.name)
    print(f"relabelled {len(relabel)}; deleted {gone} (twin failures: {em.last_twin_failures}); "
          f"backups removed {len(pruned)}")


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
