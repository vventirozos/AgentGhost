#!/usr/bin/env python3
"""§4MX one-off (2026-10-10, operator: "fix the open items").

Two lessons about "the latest version of a piece of software" disagreed:

1. 2026-09-15 (unverified, reflection): "Prefer the most recent, specific
   version number … in the results" — reading the version off search
   SNIPPETS, the exact failure the §4MW replay found (PostgreSQL 18.4 from a
   stale snippet; 18.6 on the vendor's page). Removed (archived first,
   vector twin deleted).
2. 2026-10-09 (verified, learn_skill, the §4MW rule): its example
   "e.g. 18.6, not just 18" is the replay's own answer — a moving-target
   fact the model would read as current. Re-learned under the same trigger
   with a neutral example; the dedup hit rewrites the row and its twin.

Run ONLY with the agent stopped (single-writer stores). Backup of the
playbook first. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_fix_4mx.py [--apply]
"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
RULE_TRIGGER = "when asked for the latest version of a piece of software"
OLD_TRIGGER = "When answering about the latest version or release of software from search results"
NEW_SOLUTION = ("Answer with the latest POINT release (e.g. X.Y.Z, not just the major version X), read "
                "from the vendor's own version, support or download table, and name the page. A major "
                "version's first release date is not the date of the latest point release; if the page "
                "does not give that date, say so.")


def main() -> int:
    apply = "--apply" in sys.argv
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    if apply:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, "lesson_fix_4mx.py")
    playbook = mem / "skills_playbook.json"
    rows = json.loads(playbook.read_text())
    rule = [x for x in rows if (x.get("trigger") or "").strip().lower() == RULE_TRIGGER]
    old = [x for x in rows if (x.get("trigger") or "").strip().lower() == OLD_TRIGGER.lower()]
    print(f"rule rows: {len(rule)}; old rows: {len(old)}")
    for x in rule + old:
        print("  -", x.get("trigger", "")[:70], "|", str(x.get("solution", ""))[:110])
    if len(rule) != 1 or len(old) != 1:
        print("unexpected row count — nothing done")
        return 1
    if not apply:
        print("dry run — pass --apply")
        return 0
    shutil.copy2(playbook, mem / "skills_playbook.json.pre-4mx.bak")
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(mem, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(mem)
    removed = sm.remove_by_trigger(OLD_TRIGGER, memory_system=vm)
    r = rule[0]
    w = sm.learn_lesson(r["task"], r["mistake"], NEW_SOLUTION, memory_system=vm, trigger=RULE_TRIGGER,
                        verified=True, source="learn_skill", replace_text=True)
    after = json.loads(playbook.read_text())
    now = [x for x in after if (x.get("trigger") or "").strip().lower() == RULE_TRIGGER]
    ok = (removed and w is not None and len(now) == 1 and now[0].get("solution") == NEW_SOLUTION
          and not any((x.get("trigger") or "").strip().lower() == OLD_TRIGGER.lower() for x in after)
          and len(after) == len(rows) - 1)
    print(json.dumps({"removed_old": removed, "rule_rewritten": bool(now and now[0].get("solution") == NEW_SOLUTION),
                      "rows_before": len(rows), "rows_after": len(after), "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
