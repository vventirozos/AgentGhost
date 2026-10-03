"""§4KZ one-off (2026-10-03): the owner-fact cleanup the operator confirmed —
"delete 1 2 4 8, rebuild copies, rest later":
  1 wrong values (a stale trip location/hotel, Thodoris "born ~2017-02", stored
    ages), 2 duplicates of one fact under other predicates (the canonical edge
    is KEPT and verified live first), 4 one-off search topics stored as the
    owner's wants/interests, 8 noise (car-wash tasks, a deadline, a helmet only
    considered, assistant-event logs, the lesson telling the model to doubt the
    owner's home). Plus: the two per-son birthdate profile fields (duplicates of
    relationships.sons) and cli_tools.info (duplicate of preferences.grep_tool);
    `PREFER download` → `PREFERS download`; and the vector identity copies of
    EVERY profile field rebuilt (`VectorMemory.sync_owner_field`; the store had
    none since ≥09-24). Items 3, 5, 6, 7 are left for later.
Graph rows via `delete_edge` (archived). Run with the agent STOPPED; backup first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4kz.py [--apply]
"""
import os, shutil, sqlite3, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
DELETE = {
    "1-wrong": [["user", "LOCATED_AT", "kyllini"], ["user", "STAYS_AT", "grecotel kyllini"], ["thodoris", "BORN", "~2017-02"], ["leonidas", "AGE", "5"], ["leonidas", "HAS_AGE", "5"], ["thodoris", "AGE", "9"], ["thodoris", "HAS_AGE", "9"]],
    "2-duplicate": [["user", "IS_NAMED", "vasilis"], ["user", "WAS_BORN_ON", "january 29, 1980"], ["vasilis", "HAS_BIRTHDATE", "january 29, 1980"], ["user", "RESIDES_IN", "thrakomakedones"], ["user", "HAS_HOME_IN", "thrakomakedones"], ["user", "RESIDES_IN", "athens"], ["user", "IS_LOCATED_IN", "athens"], ["user", "HAS_FAMILY_MEMBER", "fotini"], ["user", "HAS_FAMILY", "fotini"], ["user", "HAS_RELATIONSHIP", "fotini"], ["user", "HAS_CHILD", "thodoris"], ["user", "HAS_CHILD", "leonidas"], ["user", "HAS_FAMILY_MEMBER", "thodoris"], ["user", "HAS_FAMILY_MEMBER", "leonidas"], ["user", "HAS_RELATIONSHIP", "thodoris"], ["user", "HAS_RELATIONSHIP", "leonidas"], ["user", "HAS_FAMILY", "boys"], ["user", "HAS_RELATIONSHIP", "boys"], ["leonidas", "IS_SON_OF", "vasilis"], ["user", "HAS_SON_LLEONIDAS_BIRTHDATE", "leonidas born march 12, 2026 (vasilis and fotini's younger son)"], ["user", "HAS_SON_THODORIS_BIRTHDATE", "thodoris born november 25, 2016 (vasilis and fotini's older son)"], ["thodoris", "HAS_BIRTH_DATE", "nov 25, 2016"], ["thodoris", "BORN_ON", "november 25, 2016"], ["leonidas", "HAS_BIRTH_DATE", "march 12, 2026"], ["leonidas", "BORN_ON", "march 12, 2026"], ["user", "USES", "bmw"], ["user", "HAS_ASSET", "motorcycle"], ["user", "HAS_ASSET", "car"], ["user", "HAS_PROPERTY", "motorcycle"], ["user", "HAS_ITEM", "car"], ["user", "TAKES_MEDICATION", "entresto"], ["user", "HAS_INTEREST_IN", "bjj"], ["user", "HAS_ACTIVITY", "bjj training"], ["user", "HAS_ACTIVITY", "jiu-jitsu class"], ["user", "REQUIRES", "concise output"], ["user", "REQUIRES", "one-sentence answer"], ["user", "PREFER", "download"]],
    "4-search-topic": [["user", "INTERESTED_IN", "interracial sexuality"], ["user", "INTERESTED_IN", "sexual capital"], ["user", "INTERESTED_IN", "stealthing"], ["user", "INTERESTED_IN", "classical roman masturbatorium"], ["user", "WANTS", "coca plant cultivation plan"], ["user", "WANTS", "cocaine production plan"], ["user", "WANTS", "cocaine"], ["user", "WANTS", "hiring a hitman"], ["user", "SEEKING", "ecstasy pills"], ["user", "SEARCHES", "dark web"], ["user", "REQUESTED", "diagram with four men penetrating a woman"]],
    "8-noise": [["user", "HAS_TASK", "wash car"], ["user", "NEEDS", "car wash"], ["user", "NEEDS_TO_PERFORM", "car wash"], ["user", "WANTS_TO_PERFORM", "wash car"], ["user", "HAS_CLIENT_DEADLINE", "imminent"], ["user", "CONSIDERS", "shoei x-spr pro"], ["ai", "IDENTIFIED", "vasilis"], ["ai", "MENTIONED", "vasilis"], ["assistant", "GREETED", "vasilis"], ["assistant", "MENTION", "vasilis"], ["assistant", "PROVIDED_BRIEFING_TO", "vasilis"], ["assistant", "GAVE_BRIEFING_TO", "vasilis"]],
}
#: kept (one per fact) — each must be live, or nothing is applied
CANONICAL = [["user", "MARRIED_TO", "fotini"], ["user", "HAS_SON", "thodoris"], ["user", "HAS_SON", "leonidas"],
             ["thodoris", "HAS_BIRTHDATE", "november 25, 2016"], ["leonidas", "HAS_BIRTHDATE", "march 12, 2026"],
             ["user", "HAS_BIRTHDATE", "january 29, 1980"], ["user", "HAS_NAME", "vasilis"],
             ["user", "LIVES_IN", "thrakomakedones near athens"], ["user", "HAS_MEDICATION", "entresto"],
             ["user", "PREFERS", "concise answers"], ["user", "HAS_HOBBY", "brazilian jiu jitsu"]]
ADD = [["user", "PREFERS", "download"]]
PROFILE_FIELDS = [["relationships", "son_thodoris_birthdate"], ["relationships", "son_lleonidas_birthdate"],
                  ["cli_tools", "info"]]
LESSON_TRIGGERS = ["how much time is from here to home ?"]


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def check(live: set) -> list:
    """The listed rows still live, or SystemExit (nothing applied) when a
    kept canonical edge is missing or a delete would take one."""
    canon = {tuple(c) for c in CANONICAL}
    for c in canon:
        if c not in live:
            raise SystemExit(f"canonical {c} is not live — nothing applied")
    todo = []
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
    print(f"remove {len(todo)} edges, {len(PROFILE_FIELDS)} profile fields, {len(LESSON_TRIGGERS)} lesson; "
          f"add {len(ADD)}; rebuild identity copies{'' if APPLY else ' (dry run)'}")
    if not APPLY:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4kz-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    g = GraphMemory(MEM)
    n = sum(g.delete_edge(*r) for r in todo)
    g.add_triplets([{"subject": s, "predicate": p, "object": o} for s, p, o in ADD])
    pm = ProfileMemory(MEM)
    f = [pm.delete(cat, k, exact=True) for cat, k in PROFILE_FIELDS]
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    gone = sum(bool(SkillMemory(MEM).remove_by_trigger(t, memory_system=vm)) for t in LESSON_TRIGGERS)
    rebuilt = 0
    for cat, sub in (pm.load() or {}).items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                rebuilt += vm.sync_owner_field(k, v if isinstance(v, list) else [v])[0]
    print(f"edges removed {n}; profile {f}; lessons removed {gone}; identity copies written {rebuilt}")


if __name__ == "__main__":
    main()
