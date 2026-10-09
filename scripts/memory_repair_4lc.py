"""§4LC one-off (2026-10-03): the owner-fact answers to the §4KZ/§4LB questions —
"1. none 3 yes 4 a i run evolmonkey, 4b yes 4 c no":
  3  the 102 graph edges whose subject is the agent itself (ai/assistant/system:
     log noise such as "assistant GENERATED image", never facts about the owner)
  4a the owner RUNS EvolMonkey: `user WORKS_AT evolmonkey` removed, the profile
     gains root.role ("Runs EvolMonkey")
  4b birth dates into the profile: root.birthdate 1980-01-29,
     relationships.wife_birthdate 1982-01-10 (mirrors synced)
  4c NOT owned / not real: the AGV Pista GP RR and Shoei X-Spirit III helmets,
     the Interactive Brokers account with €100 (graph edges + the one synthesis
     row stating it). The episodes recording that he ASKED about helmets stay.
Graph rows via `delete_edge` (archived). Run with the agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4lc.py [--apply]
"""
import os, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
AGENT_SUBJECTS = ("ai", "assistant", "system")
AGENT_EDGES_EXPECTED = 102
DELETE = {
    "4a-works-at": [["user", "WORKS_AT", "evolmonkey"]],
    "4c-not-owned": [["user", "OWNS", "pista gp rr"], ["user", "OWNS", "shoei x-spirit iii"],
                     ["user", "HAS_ACCOUNT_WITH", "interactive brokers"], ["user", "HAS_FUNDS", "100 euros"]],
}
#: the vector rows stating a 4c fact (exact id + the text it must still carry)
VECTOR_DELETE = {"b2c437fcc4eaec0196bb23181f0ce43c": "Interactive Brokers account with 100 euros"}
#: kept — each must be live, or nothing is applied
CANONICAL = [["user", "RUNS", "evolmonkey"], ["evolmonkey", "IS_A", "postgresql services company"],
             ["user", "HAS_BIRTHDATE", "january 29, 1980"], ["fotini", "HAS_BIRTHDATE", "january 10, 1982"],
             ["user", "MARRIED_TO", "fotini"]]
PROFILE_SET = [["root", "role", "Runs EvolMonkey"], ["root", "birthdate", "1980-01-29"],
               ["relationships", "wife_birthdate", "1982-01-10"]]


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def check(live: set) -> list:
    """The rows to delete (agent-subject edges + the listed rows still live),
    or SystemExit (nothing applied) when a kept fact is missing, a delete
    would take one, or the agent-edge count is not the one the operator saw."""
    canon = {tuple(c) for c in CANONICAL}
    for c in canon:
        if c not in live:
            raise SystemExit(f"canonical {c} is not live — nothing applied")
    agent = sorted(r for r in live if str(r[0]).lower() in AGENT_SUBJECTS)
    if len(agent) != AGENT_EDGES_EXPECTED:
        raise SystemExit(f"{len(agent)} agent-subject edges, the operator confirmed {AGENT_EDGES_EXPECTED} — nothing applied")
    todo = [list(r) for r in agent]
    for group, rows in DELETE.items():
        for r in rows:
            if tuple(r) in canon:
                raise SystemExit(f"{group} would delete a kept fact {r} — nothing applied")
            if tuple(r) in live:
                todo.append(r)
    return todo


def main():
    with sqlite3.connect(f"file:{MEM / 'knowledge_graph.db'}?mode=ro", uri=True) as c:
        live = set(c.execute("SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL"))
    todo = check(live)
    print(f"remove {len(todo)} edges, {len(VECTOR_DELETE)} vector row(s); set {len(PROFILE_SET)} profile fields"
          f"{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4lc-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.memory.vector import VectorMemory
    g = GraphMemory(MEM)
    n = sum(g.delete_edge(*r) for r in todo)
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    got = vm.collection.get(ids=list(VECTOR_DELETE), include=["documents"])
    doomed = [i for i, d in zip(got.get("ids") or [], got.get("documents") or []) if VECTOR_DELETE[i] in str(d)]
    if doomed:
        vm.collection.delete(ids=doomed)
    pm = ProfileMemory(MEM)
    synced = 0
    for cat, k, v in PROFILE_SET:
        pm.update(cat, k, v)
        synced += g.sync_owner_field(k, [v])[0] + vm.sync_owner_field(k, [v])[0]
    print(f"edges removed {n}; vector rows removed {len(doomed)}; profile set {len(PROFILE_SET)}; mirrors added {synced}")


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
