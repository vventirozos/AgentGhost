#!/usr/bin/env python3
"""§4NC one-off (2026-10-10, operator: "Retire them (Recommended)").

The producer review replayed 86 real owner requests: the 44 dream-written
lessons surfaced 3 times with no relevant hit (once self-derived items are
set aside); distilled lessons were re-bumped rules. Producer lessons are now
off (`LESSON_SOURCES_ALLOWED`). Every remaining lesson with source dream /
dream_pattern / distilled is archived and removed from the playbook AND the
vector store (`remove_rows`, tombstoned) — never a verified or owner-rule row.

Run ONLY with the agent stopped. Backup first. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/lesson_retire_4nc.py [--apply]
"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
SOURCES = {"dream", "dream_pattern", "distilled"}


def _target(r) -> bool:
    return (isinstance(r, dict) and str(r.get("source") or "") in SOURCES
            and str(r.get("verified")).lower() != "true" and r.get("origin") != "owner_rule")


def main() -> int:
    apply = "--apply" in sys.argv
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    if apply:   # §4MN: never beside the agent (single-writer stores)
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mem, "lesson_retire_4nc.py")
    playbook = mem / "skills_playbook.json"
    rows = json.loads(playbook.read_text())
    hit = [r for r in rows if _target(r)]
    print(f"rows {len(rows)}; to retire {len(hit)}")
    if not apply:
        for r in hit[:5]:
            print("  -", (r.get("trigger") or "")[:80])
        print("dry run — pass --apply")
        return 0
    shutil.copy2(playbook, mem / "skills_playbook.json.pre-4nc.bak")
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(mem, upstream_url="http://127.0.0.1:8088")
    sm = SkillMemory(mem)
    n = sm.remove_rows(_target, memory_system=vm)
    after = json.loads(playbook.read_text())
    ok = n == len(hit) and len(after) == len(rows) - len(hit) and not any(_target(r) for r in after)
    print(json.dumps({"retired": n, "rows": [len(rows), len(after)],
                      "owner_rules_kept": len(sm.owner_rules()), "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
