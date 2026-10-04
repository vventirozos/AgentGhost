"""§4LC part 2 (2026-10-03, operator: "do that too, proceed"): three profile
fields stored as one comma/"and" sentence become lists, one item per fact, so a
single item can be updated or forgotten (`remove_item`) without rewriting the
rest. Each item keeps the field's original as_of. Values are listed verbatim
and must equal the live value, or nothing is applied. The vector identity rows
(`User <key> is …`) are synced to the items; the graph already holds the
per-item facts (HAS_SON ×2, OWNS ×3, HAS_HOBBY), so no HAS_SONS/HAS_VEHICLES
edges are added beside them.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4lc_lists.py [--apply]
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
SPLIT = {
    ("relationships", "sons"): ("Thodoris (born 2016-11-25) and Leonidas (born 2026-03-12)",
                                ["Thodoris (born 2016-11-25)", "Leonidas (born 2026-03-12)"]),
    ("interests", "hobbies"): ("Brazilian jiu jitsu, cars, motorcycles, Technology (special interest in AI)",
                               ["Brazilian jiu jitsu", "cars", "motorcycles", "Technology (special interest in AI)"]),
    ("assets", "vehicles"): ("BMW 118i, Ducati Streetfighter V4s, Sym scooter",
                             ["BMW 118i", "Ducati Streetfighter V4s", "Sym scooter"]),
}


def _agent_listening(port: int = 0) -> bool:
    import socket
    port = port or int(os.environ.get("GHOST_AGENT_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def plan(raw: dict) -> dict:
    """{(cat, key): (as_of, items)} to write, or SystemExit (nothing applied)
    when a live value is not the one listed. Already-split fields are skipped."""
    todo = {}
    for (cat, key), (sentence, items) in SPLIT.items():
        cur = (raw.get(cat) or {}).get(key)
        if isinstance(cur, list) and [str(x.get("v") if isinstance(x, dict) else x) for x in cur] == items:
            continue
        v = cur.get("v") if isinstance(cur, dict) else cur
        if v != sentence:
            raise SystemExit(f"{cat}.{key} is {v!r}, not the listed value — nothing applied")
        todo[(cat, key)] = ((cur or {}).get("as_of") if isinstance(cur, dict) else None, items)
    return todo


def main():
    raw = json.loads((MEM / "user_profile.json").read_text())
    todo = plan(raw)
    print(f"split {len(todo)} field(s): {[f'{c}.{k}' for c, k in todo]}{'' if APPLY else ' (dry run)'}")
    if not APPLY or not todo:
        return
    if _agent_listening():
        raise SystemExit("the agent is running — stop it first (single writer); nothing applied")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4lc-lists-{stamp}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.memory.vector import VectorMemory
    pm = ProfileMemory(MEM)
    for (cat, key), (as_of, items) in todo.items():
        pm.delete(cat, key, exact=True)
        for it in items:
            pm.update(cat, key, it, as_of=as_of)
    got = {(c, k): (pm.load().get(c) or {}).get(k) for c, k in todo}
    bad = {f"{c}.{k}": v for (c, k), v in got.items() if v != todo[(c, k)][1]}
    if bad:
        raise SystemExit(f"profile did not take the lists: {bad} — restore memory.pre-4lc-lists-{stamp}.bak")
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    synced = [vm.sync_owner_field(k, items) for (_c, k), (_a, items) in todo.items()]
    print(f"profile split {len(todo)}; identity rows (added, removed) {synced}")


if __name__ == "__main__":
    main()
