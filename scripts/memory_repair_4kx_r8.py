"""§4KX r8 one-off (2026-10-03): restore the owner's health, profession and
companion facts the graph decay pruned before those predicates were durable
(heart failure, blood clot, Entresto twice, companion; the profession edge was wrong), from
graph_pruned_archive.jsonl. Only these five; each must be in the archive as a
`prune_stale_edges` row, be an owner fact under the CURRENT rule, and not be
live already — else nothing is applied.
Run with the agent STOPPED; memory dir backed up first (without *.bak).
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4kx_r8.py [--apply]
"""
import json, os, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
RESTORE_EDGES = [["user", "HAS_CONDITION", "heart failure"], ["user", "HAS_CONDITION", "blood clot"],
                 ["user", "HAS_MEDICATION", "entresto"], ["user", "TAKES_MEDICATION", "entresto"],
                 ["user", "HAS_COMPANION", "fotini"]]
#: (2026-10-03, operator: "im not a doctor") `user HAS_PROFESSION doctor` was restored by the first run and then
#: removed at the operator's word — never restore it again


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def check(archive_rows: list, live: set) -> list:
    """The edges to restore, or SystemExit (nothing applied)."""
    from ghost_agent.memory.graph import GraphMemory
    pruned = {(a.get("subject"), a.get("predicate"), a.get("object")) for a in archive_rows
              if a.get("reason") == "prune_stale_edges"}
    out = []
    for s, p, o in RESTORE_EDGES:
        if (s, p, o) not in pruned or not GraphMemory._is_owner_life_fact(s, p, o):
            raise SystemExit(f"restore: {s} {p} {o} is not a pruned owner fact — nothing applied")
        if (s, p, o) not in live:
            out.append({"subject": s, "predicate": p, "object": o})
    return out


def main():
    arch = [json.loads(l) for l in open(MEM / "graph_pruned_archive.jsonl", encoding="utf-8") if l.strip()]
    with sqlite3.connect(f"file:{MEM / 'knowledge_graph.db'}?mode=ro", uri=True) as c:
        live = set(c.execute("SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL"))
    edges = check(arch, live)
    print(f"restore {len(edges)} edges{'' if APPLY else ' (dry run)'}: " + "; ".join(" ".join(e.values()) for e in edges))
    if not APPLY or not edges:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kx-r8-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    GraphMemory(MEM).add_triplets(edges)
    print(f"edges restored {len(edges)}")


if __name__ == "__main__":
    main()
