#!/usr/bin/env python3
"""§4MK one-off (2026-10-08, operator: "1 yes").

Trajectory 3d27a6b0 ("give me a morning briefinf", 2026-10-06) was REFUTED
twice, both falsely: "30 GB / 37 GB" against "30048MB / 36864MB" (decimal
rounding, the binder rule fixed in §4MK) and "Chess Coach v4 -> FAILED",
which came from an earlier turn's system notice the judges never saw (the
§4MK CRIT). Reflection wrote a "hallucinated" corrective lesson from it
(vector row 26344, trigger "give me a morning briefinf").

1. The trajectory is relabelled PASSED in the corrections overlay
   (source "operator").
2. Every lesson sourced from it, the corrective included, is retracted
   from the JSON playbook AND the vector store (`retract_lessons_from_trajectory`,
   include_correctives=True: the turn was fine).

Run ONLY with the agent stopped (the vector store has one writer). Backup
of the playbook first. Dry run by default.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4mk_lesson.py [--apply]
"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
TRAJ = "3d27a6b083ad463f87462c347b0e4f22"
REASON = ("operator 2026-10-08 (§4MK): both refutes were false — decimal GB rounding, and a fact "
          "from an earlier turn's system notice the judges never saw")


def main() -> int:
    apply = "--apply" in sys.argv
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    playbook = mem / "skills_playbook.json"
    hits = [x for x in json.loads(playbook.read_text()) if x.get("source_trajectory_id") == TRAJ]
    print(f"lessons sourced from {TRAJ[:8]}: {len(hits)}")
    for x in hits:
        print("  -", x.get("task", "")[:80], "|", str(x.get("mistake", ""))[:100])
    if not apply:
        print("dry run — pass --apply")
        return 0
    shutil.copy2(playbook, mem / "skills_playbook.json.pre-4mk.bak")
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory.vector import VectorMemory
    from ghost_agent.distill.collector import TrajectoryCollector
    vm = VectorMemory(mem, upstream_url="http://127.0.0.1:8088")   # delete by metadata: no embedding call
    n = SkillMemory(mem).retract_lessons_from_trajectory(TRAJ, memory_system=vm, include_correctives=True)
    ok = TrajectoryCollector(home / "system" / "trajectories").update_outcome(
        TRAJ, "passed", REASON, source="operator")
    print(json.dumps({"retracted": n, "relabelled": bool(ok)}))
    return 0 if n == len(hits) and ok else 1


if __name__ == "__main__":
    sys.exit(main())
