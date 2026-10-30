#!/usr/bin/env python3
"""KB / graph / recall review one-off (2026-10-30, operator: "you can change
stored data"). Rule-based, re-derived from the store at run time:

  1. mirrors: `resync_all_owner_mirrors` (the category-qualified names) FIRST,
     so every profile field has its new graph/vector copy before step 2;
  2. graph, the owner's legacy-extractor duplicates of profile fields
     (OWNS, HAS_HOBBY, HAS_SON, HAS_SPOUSE, MARRIED_TO, LIVES_IN, LOCATED_IN,
     HAS_ADDRESS) — only when the profile still holds the value; they never
     retire when the profile changes (review E F3);
  3. graph, the owner's CHATTER logged as facts (ASKS, DEMANDS, REJECTED,
     WANTS…) and every non-owner row except the owner's family (a subject the
     profile names) and the projects (review E F5: ~89 % noise; the graph has
     been out of prompts since §4NC and only mirrors/projects write it, §4OM);
  4. episodes a Slack MEMBER asked that were stored as the owner's (review D:
     same text as a member trajectory within 4 min, no owner trajectory with
     that text) — `EpisodicMemory.delete_episodes` (archived, vector twins too).

Graph rows go through `GraphMemory.delete_edge` (archived first). A full copy
of the memory directory is taken first. Run ONLY with the agent stopped.
Dry run by default; `--apply` writes.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_kb1030.py [--apply]
"""
import asyncio
import datetime as dt
import glob
import json
import os
import shutil
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = Path(os.environ.get("GHOST_MEMORY_DIR") or HOME / "system" / "memory")
TRAJ = Path(os.environ.get("GHOST_TRAJECTORIES_DIR") or HOME / "system" / "trajectories")
STAMP = time.strftime("%Y%m%dT%H%M%S")

LEGACY_DUP = {"OWNS", "HAS_HOBBY", "HAS_SON", "HAS_SPOUSE", "MARRIED_TO", "LIVES_IN", "LOCATED_IN", "HAS_ADDRESS"}
OWNER_DURABLE = {"PREFERS", "SPECIALIZES_IN", "USES", "PLAYS", "VISITED", "INTERESTED_IN", "HAS_INTEREST"}
KEEP_SUBJECTS = {"project", "webos", "jiu jitsu calendar"}


def fold(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").casefold())
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def live_graph_rows() -> list:
    with sqlite3.connect(f"file:{MEM / 'knowledge_graph.db'}?mode=ro", uri=True) as c:
        return [(s, p, o) for s, p, o in c.execute(
            "SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL")]


def project_titles() -> set:
    try:
        with sqlite3.connect(f"file:{MEM / 'projects.db'}?mode=ro", uri=True) as c:
            return {fold(r[0]) for r in c.execute("SELECT title FROM projects") if r[0]}
    except Exception:  # noqa: BLE001
        return set()


_FILLER = {"near", "in", "of", "the", "and", "at"}


def _profile_words(prof) -> tuple:
    """(every folded word of the profile, the first word of each value — a
    person's name: "Fotini", "Leonidas (born …)")."""
    words, firsts = set(), set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, (str, int, float)):
            ws = [w for w in "".join(c if c.isalnum() else " " for c in fold(v)).split()]
            words.update(ws)
            if ws:
                firsts.add(ws[0])
    walk(prof)
    return words, firsts


def classify(rows, profile_text: str, titles: set, pwords=(set(), set())):
    """[(row, why)] to delete. Kept: the owner's HAS_* facts (mirrors, graph-only
    health facts) and durable attributes, family, projects."""
    out = []
    for s, p, o in rows:
        fs, fo = fold(s), fold(o)
        if fs == "user":
            if p in LEGACY_DUP:
                _ow = {w for w in "".join(c if c.isalnum() else " " for c in fo).split()} - _FILLER
                if _ow and _ow <= pwords[0]:
                    out.append(((s, p, o), "legacy duplicate of a profile field"))
                continue
            if p.startswith("HAS_") and p != "HAS_ACTIVITY":
                continue
            if p in OWNER_DURABLE:
                continue
            out.append(((s, p, o), "owner chatter logged as a fact"))
            continue
        if fs in KEEP_SUBJECTS or fs in titles or (" " not in fs and fs in pwords[1]):
            continue
        out.append(((s, p, o), "non-owner extractor noise"))
    return out


def _ts(v):
    try:
        t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return t.timestamp() if t.tzinfo else t.replace(tzinfo=dt.timezone.utc).timestamp()
    except Exception:  # noqa: BLE001
        return None


def member_episodes() -> list:
    """(id, trigger head) of episodes whose text a MEMBER sent within 4 min of
    it and the owner never sent in that window."""
    by_text: dict = {}
    for f in glob.glob(str(TRAJ / "*" / "*.jsonl")):
        for ln in open(f, errors="replace"):
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            u = (d.get("user_request") or "").strip()
            t = _ts(d.get("timestamp"))
            if not u or t is None:
                continue
            role = (d.get("extra") or {}).get("requester_role") or ""
            by_text.setdefault(u, []).append((t, role))
    out = []
    with sqlite3.connect(f"file:{MEM / 'episodic_memory.db'}?mode=ro", uri=True) as c:
        for eid, trig, ts in c.execute("SELECT id, trigger, timestamp FROM episodes WHERE req_id = ''"):
            hits = [r for t, r in by_text.get((trig or "").strip(), []) if abs(t - float(ts)) <= 240]
            if any(r == "member" for r in hits) and not any(r == "" for r in hits):
                out.append((eid, (trig or "")[:60]))
    return out


def main() -> int:
    prof = json.loads((MEM / "user_profile.json").read_text(encoding="utf-8"))
    profile_text = fold(json.dumps(prof, ensure_ascii=False))
    pwords = _profile_words(prof)
    titles = project_titles()
    rows = live_graph_rows()
    eps = member_episodes()
    plan = classify(rows, profile_text, titles, pwords)
    kinds: dict = {}
    for _, why in plan:
        kinds[why] = kinds.get(why, 0) + 1
    print(f"graph live {len(rows)}: delete {len(plan)} {kinds}; keep {len(rows) - len(plan)}")
    print(f"member episodes: {len(eps)} {[e for e, _ in eps]}")
    if not APPLY:
        for (s, p, o), why in plan:
            if why != "non-owner extractor noise":
                print(f"  - {s} {p} {o[:60]}  [{why}]")
        kept = sorted(set(rows) - {r for r, _ in plan})
        print("  kept:", "; ".join(f"{s} {p} {o[:30]}" for s, p, o in kept))
        return 0
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(MEM, Path(__file__).name)          # §4MN: never beside the agent
    shutil.copytree(MEM, MEM.parent / f"memory.pre-kb1030-{STAMP}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.vector import VectorMemory
    from ghost_agent.memory.graph import GraphMemory
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.memory.episodes import EpisodicMemory
    from ghost_agent.tools.memory import resync_all_owner_mirrors
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    g = GraphMemory(MEM)
    pm = ProfileMemory(MEM)
    synced, lagging = asyncio.run(resync_all_owner_mirrors(pm, g, vm))
    if lagging:
        print(f"mirror resync: {lagging} field(s) lagging — stopping before any delete")
        return 1
    # step 1 changed the graph: classify again on what is live now
    plan = classify(live_graph_rows(), profile_text, titles, pwords)
    removed = sum(g.delete_edge(s, p, o, mirror=True) for (s, p, o), _ in plan)
    em = EpisodicMemory(MEM)
    ep_removed = em.delete_episodes([e for e, _ in eps], vector_memory=vm,
                                    reason="KB review 10-30: a channel member's request stored as the owner's")
    after = live_graph_rows()
    still = [r for r, _ in plan if r in set(after)]
    print(json.dumps({"mirrors_synced": synced, "graph_removed": removed, "graph_still_live": len(still),
                      "graph_live_after": len(after), "episodes_removed": ep_removed,
                      "episode_twin_failures": getattr(em, "last_twin_failures", []),
                      "backup": f"memory.pre-kb1030-{STAMP}.bak"}, indent=1))
    return 1 if still or ep_removed != len(eps) else 0


if __name__ == "__main__":
    sys.exit(main())
