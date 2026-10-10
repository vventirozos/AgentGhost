#!/usr/bin/env python3
"""§4MZ one-off (2026-10-10, operator: "do 1" — adopt candidate rule 1).

The owner adopted replay candidate rule 1 ("Do not assert system health
status … unless the tool output explicitly confirms them via a dedicated
health check action"). Its case was diagnosed before the diagnosis gave a
`when`, so the lesson was keyed by the rule's own first 120 characters —
a poor match for the requests it is for ("hello ghost, how's things
today ?"). Re-keyed to the situation, the rule text unchanged; the ledger's
diagnosis gets the same `when` so "show rule 1" shows it.

Run ONLY with the agent stopped (single-writer stores). Backups first. Dry
run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_rekey_4mz.py [--apply]
"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
NEW_TRIGGER = ("when asked how you are, how things are going, or about the system's status, health "
               "or performance")


def main() -> int:
    apply = "--apply" in sys.argv
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    if apply:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, "lesson_rekey_4mz.py")
    from ghost_agent.core import failure_replay as FR
    ledger = FR.load(home / "system")
    case = next((e for e in ledger if e.get("n") == 1 and e.get("adopted_at")), None)
    if case is None:
        print("no adopted rule 1 — nothing done")
        return 1
    rule = case["rule"]
    old_trigger = rule[:120]
    playbook = mem / "skills_playbook.json"
    rows = json.loads(playbook.read_text())
    hits = [x for x in rows if (x.get("trigger") or "").strip().lower() == old_trigger.strip().lower()]
    print(f"rows keyed by the rule head: {len(hits)}")
    for x in hits:
        print("  -", x.get("trigger", "")[:80], "|", str(x.get("solution", ""))[:80])
    if len(hits) != 1 or (hits[0].get("solution") or "").strip() != rule.strip():
        print("unexpected rows — nothing done")
        return 1
    if not apply:
        print("dry run — pass --apply")
        return 0
    shutil.copy2(playbook, mem / "skills_playbook.json.pre-4mz.bak")
    lp = FR._path(home / "system")
    shutil.copy2(lp, lp.with_suffix(".pre-4mz.bak"))
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(mem, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(mem)
    removed = sm.remove_by_trigger(old_trigger, memory_system=vm)
    w = sm.learn_lesson(NEW_TRIGGER, hits[0].get("mistake") or "unsupported_claim", rule, memory_system=vm,
                        trigger=NEW_TRIGGER, verified=True, source="learn_skill", origin="owner_rule",
                        source_trajectory_id=str(case.get("source_id") or ""),
                        generality_context=f"learn this rule: {rule}")
    case.setdefault("diagnosis", {})["when"] = NEW_TRIGGER
    FR.save(home / "system", ledger)
    after = json.loads(playbook.read_text())
    now = [x for x in after if (x.get("trigger") or "").strip().lower() == NEW_TRIGGER.lower()]
    ok = (removed and w is not None and len(now) == 1 and now[0].get("solution") == rule
          and not any((x.get("trigger") or "").strip().lower() == old_trigger.strip().lower() for x in after)
          and len(after) == len(rows))
    print(json.dumps({"removed_old_key": removed, "rekeyed": bool(now), "rows": [len(rows), len(after)], "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
