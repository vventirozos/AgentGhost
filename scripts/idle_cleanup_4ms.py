#!/usr/bin/env python3
"""§4MS one-off (2026-10-09, operator: "preview, then I confirm"): clean what
the idle cycle wrote wrong.

The §4MS value lens graded idle-written memory: a third of a 42-item sample
was wrong, duplicate, junk or rested on probe traffic. The code now gates
new writes; this cleans what is already stored:

1. playbook — RETRACT (archived to skills_pruned_archive.jsonl, reason
   `retract:4ms-<why>`):
   * "use the available imagination/creative parameters" (dream) — no such
     parameter exists; it reached an owner turn on 10-06;
   * "confirm the item ID … when deletion attempts fail due to token issues"
     (dream) — a workaround for the confirm-token bug fixed in §4LU;
   * the reflection rewrite resting on trajectory de485e9e… — a PROBE turn.
2. playbook — RESET the retrieval/helpful counters of three rows the lens
   found credited with zero owner hydrations (two credits from unlabelled
   probe turns): "verify the file path exists first" (44/42), "web searches
   requiring specific version or release data" (29/24), "`manage_services`
   … confirm service binding" (26/24). Pruning and graduation read them.
3. (dropped, r1 review) the backfill of 80-character dream triggers: each
   row's vector twin and `task` are keyed by the OLD trigger, so a new
   trigger orphaned the row from retrieval and froze its counters — and 17
   of the 36 would have been cut again at 160 characters. Not applied.
4. vector memory — DELETE the auto memory "The user is aware of potential
   pro-China alignment biases…" (a mischaracterised Slack joke, recalled 8×).
5. acquired skill — RETIRE `extract_html_content` (graduated from a
   self-play Playwright lesson; no owner use) the way the store retires one:
   code to `retired/`, registry entry and vector rows removed.

A verified snapshot is taken first (`memory/snapshot.py`). Run ONLY with the
agent stopped. Dry run by default (prints the exact plan); `--apply` writes.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/idle_cleanup_4ms.py [--apply]
"""
import datetime
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
MEM = HOME / "system" / "memory"
PLAYBOOK = MEM / "skills_playbook.json"
ARCHIVE = MEM / "skills_pruned_archive.jsonl"
SKILLS_DIR = MEM / "acquired_skills"
RETIRE_SKILL = "extract_html_content"
VECTOR_TEXT = "The user is aware of potential pro-China alignment biases"

RETRACT = (
    ("imagination-parameters", lambda r: "imagination/creative parameters" in str(r.get("trigger") or "")
     and r.get("source") == "dream"),
    ("stale-confirm-token-workaround", lambda r: str(r.get("trigger") or "").startswith(
        "When deleting a knowledge base document, confirm the specific item ID") and r.get("source") == "dream"),
    ("probe-rooted", lambda r: str(r.get("source_trajectory_id") or "").startswith("de485e9e")
     and r.get("source") == "reflection"),
)
RESET = (
    lambda r: str(r.get("trigger") or "").startswith("When running commands that operate on a file, verify the file path exists"),
    lambda r: str(r.get("trigger") or "").startswith("When performing web searches requiring specific version or release data"),
    lambda r: str(r.get("trigger") or "").startswith("When using the `manage_services` tool, confirm service binding"),
)
COUNTERS = ("retrievals", "helpful_retrievals")


def plan_playbook(rows):
    from ghost_agent.memory.skills import trigger_text
    retract, reset, backfill = [], [], []
    for i, r in enumerate(rows):
        for why, match in RETRACT:
            if match(r):
                retract.append((i, why))
        if any(m(r) for m in RESET):
            reset.append(i)
        if r.get("source") == "dream":
            trig = str(r.get("trigger") or "")
            rule = " ".join(str(r.get("solution") or r.get("correct_pattern") or "").split())
            full = trigger_text(rule)
            # cut MID-sentence (the next character continues the text) — not a
            # trigger that merely lacks the rule's closing period
            rest = rule[len(trig):].strip()
            if trig and rule.startswith(trig) and len(trig) < len(full) and rest not in ("", ".", "!", "?"):
                backfill.append((i, trig, full))
    return retract, reset, backfill


def find_vector_ids():
    with sqlite3.connect(f"file:{MEM / 'chroma.sqlite3'}?mode=ro", uri=True) as c:
        return [r[0] for r in c.execute(
            "SELECT e.embedding_id FROM embedding_metadata m JOIN embeddings e ON e.id = m.id "
            "WHERE m.key = 'chroma:document' AND m.string_value LIKE ?", (VECTOR_TEXT + "%",))]


def main() -> int:
    if APPLY:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "idle_cleanup_4ms.py")
    rows = json.loads(PLAYBOOK.read_text(encoding="utf-8"))
    retract, reset, backfill = plan_playbook(rows)
    vec_ids = find_vector_ids()
    registry = json.loads((SKILLS_DIR / "skills_registry.json").read_text(encoding="utf-8"))
    plan = {
        "playbook_rows": len(rows),
        "retract": [{"index": i, "why": why, "source": rows[i].get("source"),
                     "trigger": str(rows[i].get("trigger"))[:110]} for i, why in retract],
        "reset_counters": [{"index": i, "trigger": str(rows[i].get("trigger"))[:90],
                            **{k: rows[i].get(k) for k in COUNTERS}} for i in reset],
        "backfill_triggers": [{"index": i, "from": a[-40:], "to_len": len(b)} for i, a, b in backfill],
        "delete_vector_memories": vec_ids,
        "retire_acquired_skill": RETIRE_SKILL if RETIRE_SKILL in registry else None,
    }
    plan["backfill_triggers"] = []          # r1: not applied (see the docstring)
    if not APPLY:
        print(json.dumps({"dry_run": True, **plan}, indent=1, ensure_ascii=False))
        return 0
    expected = {"imagination-parameters", "stale-confirm-token-workaround", "probe-rooted"}
    # idempotent (r1: a crash after the playbook write left a re-run refused):
    # whatever is still there and EXPECTED is done; anything unexpected stops it
    if not {w for _, w in retract} <= expected or len(reset) > 3:
        print(json.dumps({"error": "the plan no longer matches what was previewed — nothing changed",
                          **plan}, indent=1, ensure_ascii=False))
        return 2
    from ghost_agent.memory.snapshot import take_snapshot
    snap = take_snapshot(HOME, None, "pre-4ms-cleanup")
    if not snap.get("ok"):
        print(json.dumps({"error": "snapshot failed — nothing changed", "snapshot": snap}))
        return 2
    # 4 + 5 FIRST: the vector store (chroma directly — the agent is stopped
    # and the writer lock is ours), then the skill; the playbook last
    import chromadb
    from chromadb.config import Settings
    col = chromadb.PersistentClient(path=str(MEM), settings=Settings(anonymized_telemetry=False)) \
        .get_collection("agent_memory")
    if vec_ids:
        col.delete(ids=vec_ids)
    retired = None
    if RETIRE_SKILL in registry:
        (SKILLS_DIR / "retired").mkdir(exist_ok=True)
        src = SKILLS_DIR / f"{RETIRE_SKILL}.py"
        if src.exists():
            shutil.move(str(src), str(SKILLS_DIR / "retired" / f"{RETIRE_SKILL}.py"))
        try:
            col.delete(where={"$and": [{"name": RETIRE_SKILL}, {"type": "acquired_skill"}]})
        except Exception as e:  # noqa: BLE001
            print(f"skill vector rows not removed: {e}", file=sys.stderr)
        registry.pop(RETIRE_SKILL, None)
        reg_tmp = SKILLS_DIR / "skills_registry.json.tmp"
        reg_tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        reg_tmp.replace(SKILLS_DIR / "skills_registry.json")
        retired = RETIRE_SKILL
    # 1-2: the playbook, one write. Archived as OPERATOR retractions (the
    # tombstone reason the store refuses to re-learn), each lesson a dict.
    now = datetime.datetime.now().isoformat()
    gone = {i for i, _ in retract}
    with open(ARCHIVE, "a", encoding="utf-8") as fh:
        for i, why in retract:
            fh.write(json.dumps({"pruned_at": now, "reason": "removed_by_trigger", "detail": f"4ms-{why}",
                                 "lesson": rows[i]}, ensure_ascii=False, default=str) + "\n")
    for i in reset:
        for k in ("retrievals", "helpful_retrievals"):     # r1: the outcome counters stay
            if k in rows[i]:
                rows[i][k] = 0
    kept = [r for j, r in enumerate(rows) if j not in gone]
    tmp = PLAYBOOK.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PLAYBOOK)
    print(json.dumps({"applied": True, "snapshot": snap["path"], "retracted": len(gone),
                      "counters_reset": len(reset), "vector_deleted": len(vec_ids),
                      "skill_retired": retired, "playbook_rows_after": len(kept)}, indent=1))
    return 0


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
    sys.exit(main())
