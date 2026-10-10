#!/usr/bin/env python3
"""§4NA one-off (2026-10-10, operator: "All 56 (Recommended)").

The lesson-reach review labelled 86 real owner requests against the
playbook: 56 lessons were judged bad by independent readers (1 harmful —
the greeting lesson that recites internal metrics against the owner's
adopted rule 1; 29 self-play/bench test-harness drills; 14 vague or
confused rules; 12 stale one-request answer scripts), and 1 of the 56
applied to any request in two weeks. Each is archived and removed from the
JSON playbook AND the vector store (`remove_by_trigger`, which tombstones
the trigger so an idle cycle does not re-mint it). The version rule the
operator approved on 2026-10-09 is marked `origin: owner_rule` so it rides
the adopted-rules block.

The list is `scripts/lesson_retire_4na.json` (trigger texts, matched
exactly). Run ONLY with the agent stopped. Backup first. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_retire_4na.py [--apply]
"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
VERSION_RULE = "when asked for the latest version of a piece of software"


def main() -> int:
    apply = "--apply" in sys.argv
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    if apply:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, "lesson_retire_4na.py")
    want = json.loads((Path(__file__).with_suffix(".json")).read_text())
    playbook = mem / "skills_playbook.json"
    rows = json.loads(playbook.read_text())
    norm = lambda t: (t or "").strip().lower()
    have = {norm(r.get("trigger") or r.get("task")) for r in rows}
    missing = [w for w in want if norm(w["trigger"]) not in have]
    print(f"to retire: {len(want)}; present: {len(want) - len(missing)}; rows now {len(rows)}")
    for m in missing:
        print("  missing:", m["trigger"][:80])
    if missing:
        print("the playbook changed since the review — nothing done")
        return 1
    if not apply:
        print("dry run — pass --apply")
        return 0
    shutil.copy2(playbook, mem / "skills_playbook.json.pre-4na.bak")
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(mem, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(mem)
    gone = sum(1 for w in want if sm.remove_by_trigger(w["trigger"], memory_system=vm))
    marked = sm._update_lesson_fields(lambda r: norm(r.get("trigger")) == VERSION_RULE,
                                      lambda r: r.__setitem__("origin", "owner_rule"))
    # §4NB: the operator-approved version rule gets a ledger number, so
    # "show rule N" / "forget rule N" reach it like any adopted rule
    import time
    from ghost_agent.core import failure_replay as FR
    ledger = FR.load(home / "system")
    vrow = next((r for r in json.loads(playbook.read_text())
                 if norm(r.get("trigger")) == VERSION_RULE), None)
    if vrow is not None and not any(e.get("source_id") == "operator-4mw-version" for e in ledger):
        # no `request`: a request on the ledger is never replayed again (r1)
        ledger.append({"source_id": "operator-4mw-version", "stage": "done", "created": time.time(),
                       "request": "", "rule": vrow.get("solution"), "trigger": vrow.get("trigger"),
                       "diagnosis": {"cause": "memory_over_evidence", "when": vrow.get("trigger"),
                                     "explanation": "the §4MW manual round: 18.4 from a stale snippet; 18.6 "
                                                    "on the vendor's table — approved by the operator 2026-10-09"},
                       "base": [], "test": [], "adopted_at": time.time(),
                       "n": 1 + max([int(e.get("n") or 0) for e in ledger] + [len(ledger)])})
        FR.save(home / "system", ledger)
    numbered = next((e.get("n") for e in FR.load(home / "system") if e.get("source_id") == "operator-4mw-version"), None)
    after = json.loads(playbook.read_text())
    ok = gone == len(want) and marked and len(after) == len(rows) - len(want) and numbered is not None
    print(json.dumps({"retired": gone, "version_rule_adopted": marked, "version_rule_number": numbered,
                      "rows": [len(rows), len(after)],
                      "owner_rules": [r.get("trigger", "")[:60] for r in sm.owner_rules()], "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
