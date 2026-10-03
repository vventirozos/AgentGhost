"""§4KX one-off (2026-10-03): data repair after the re-review of the §4KX
fixes. Rows named by EXACT trigger; any that does not resolve aborts.
  * RETRACT 3 dream rules minted from the self-play harness (one written
    after the §4KX deploy: "When executing self-play tasks, state the
    findings…"; "avoid embedding an AI opponent…"; "check for role violations
    (coded stand-ins)…" — r6 had re-keyed it).
  * REWRITE 2 rules carrying the episode prompt's leaked "the trigger" word.
  * RESTORE 4 owner facts the graph decay pruned before owner facts were
    protected (home, address, residence, a vehicle), from
    graph_pruned_archive.jsonl — only those, not the project/skill noise.
Run with the agent STOPPED; memory dir backed up first (without *.bak).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4kx_r7.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
RETRACT = [
 "When executing self-play tasks, state the findings and name the sources read if ",
 "When performing self-play, avoid embedding an AI opponent when the user specifie",
 "When executing Python scripts, check for and address role violations (e.g., code"
]
REWRITE = {
 "When the trigger requires a specific output format (e.g., 'exactly one sentence'": {
  "trigger": "When a request requires a specific output format, follow it exactly in the final answer",
  "solution": "When a request requires a specific output format (e.g., 'exactly one sentence', 'strict JSON'), always make the final answer follow that format exactly."
 },
 "When performing file system operations, verify the creation and content of the f": {
  "solution": "When performing file system operations, verify the creation and content of the file before proceeding to subsequent steps, especially when the request asks for verification (e.g., 'read it back')."
 }
}
RESTORE_EDGES = [["user", "RESIDES_IN", "athens"], ["user", "HAS_HOME_IN", "thrakomakedones"], ["user", "HAS_ADDRESS", "thrakomakedones"], ["user", "OWNS", "pista gp rr"]]


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _key(x) -> str:
    return x.get("trigger") or x.get("task") or ""


def check(playbook: list, archive_rows: list) -> list:
    """Every named row resolves; every restored edge was pruned by decay and
    is an owner fact now — or SystemExit. Returns the edges to restore."""
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.lesson_quality import is_actionable_lesson
    for group, names in (("retract", RETRACT), ("rewrite", list(REWRITE))):
        for t in names:
            n = sum(1 for x in playbook if _key(x) == t)
            if n != 1:
                raise SystemExit(f"{group}: {n} rows for {t[:70]!r} — nothing applied")
    for old, new in REWRITE.items():
        if not is_actionable_lesson("none", new["solution"], new.get("trigger", old)):
            raise SystemExit(f"rewrite fails the write gate: {old[:60]!r}")
    pruned = {(a.get("subject"), a.get("predicate"), a.get("object")) for a in archive_rows
              if a.get("reason") == "prune_stale_edges"}
    out = []
    for s, p, o in RESTORE_EDGES:
        if (s, p, o) not in pruned or not GraphMemory._is_owner_fact(s, p, o):
            raise SystemExit(f"restore: {s} {p} {o} is not a pruned owner fact — nothing applied")
        out.append({"subject": s, "predicate": p, "object": o})
    return out


def main():
    playbook = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    arch = [json.loads(l) for l in open(MEM / "graph_pruned_archive.jsonl", encoding="utf-8") if l.strip()]
    edges = check(playbook, arch)
    print(f"retract {len(RETRACT)}, rewrite {len(REWRITE)}, restore {len(edges)} edges{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kx-r7-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.skills import SkillMemory, _delete_lesson_twin
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(MEM)
    gone = sum(bool(sm.remove_by_trigger(t, memory_system=vm)) for t in RETRACT)
    with sm._get_lock():
        pb = sm._load_playbook()
        for x in pb:
            k = _key(x)
            if k in REWRITE:
                _delete_lesson_twin(vm, dict(x))
                if "trigger" in REWRITE[k]:
                    x["trigger"] = x["task"] = REWRITE[k]["trigger"]
                x["solution"] = x["correct_pattern"] = REWRITE[k]["solution"]
        sm._save_playbook_unlocked(pb)
    healed = sm.heal_missing_twins(vm)
    GraphMemory(MEM).add_triplets(edges)
    print(f"retracted {gone}; rewritten {len(REWRITE)}; twins written {healed}; edges restored {len(edges)}")


if __name__ == "__main__":
    main()
