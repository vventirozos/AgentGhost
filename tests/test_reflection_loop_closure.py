"""Step 3 of the post-hunt strategy: close ONE learning loop end-to-end.

Target = the reflection loop (FAILED trajectory → Reflector → lesson →
SkillMemory → retrievable on the next similar turn). Step 2 just made
verifier-caught wrong answers land in the corpus as FAILED, so they now feed
this loop.

Two things are verified here IN-PROCESS (the capability lift itself needs a
live agent — see docs/self_improvement.md for the A/B protocol):
  1. the loop CLOSES structurally — a FAILED trajectory produces a lesson that
     lands in SkillMemory (mocked critique, so no live LLM needed);
  2. the dedup set now PERSISTS across restarts, so the loop progresses through
     the failure backlog instead of re-reflecting the oldest failures each boot.
"""

import json
import types

import pytest

from ghost_agent.distill.schema import Trajectory, Outcome


# ══════════════════════════════════════════════════════════════════════
# 1. Structural loop closure: FAILED traj → lesson in SkillMemory
# ══════════════════════════════════════════════════════════════════════

class TestReflectionLoopClosure:
    async def test_failed_trajectory_becomes_a_lesson(self, tmp_path):
        from ghost_agent.reflection.loop import Reflector
        from ghost_agent.memory.skills import SkillMemory

        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True, exist_ok=True)
        sm = SkillMemory(mem_dir)
        assert sm.list_lessons() == []  # start empty

        async def critique(_prompt):
            return (
                "DIAGNOSIS: the agent read a hardcoded path without checking it exists\n"
                "REVISED PLAN:\n1. list the directory first\n2. then read the located file"
            )

        reflector = Reflector(critique_fn=critique)

        # The production sink (main.py::_reflection_sink), minimally.
        def sink(refl):
            sm.learn_lesson(
                task=(refl.user_request or "")[:400],
                mistake=str(refl.extra.get("source_failure_reason") or "failure")[:400],
                solution=str(refl.planning_output or refl.final_response)[:1200],
                memory_system=None,
                source_trajectory_id=str(refl.extra.get("reflected_from") or ""),
                source="reflection",
            )

        failed = Trajectory(
            id="src-1",
            user_request="parse the logfile at /data/app.log and count errors",
            outcome=Outcome.FAILED.value,
            failure_reason="read a hardcoded path that did not exist",
        )

        report = await reflector.run(failed_source=[failed], sink=sink)
        assert report.reflected_ok >= 1, "reflection did not succeed"

        lessons = sm.list_lessons()
        assert len(lessons) >= 1, "no lesson landed in SkillMemory"
        blob = json.dumps(lessons)
        # The corrective plan the critique produced is now in the playbook,
        # tagged with the source failure's id for retraction provenance.
        assert "list the directory" in blob
        assert "src-1" in blob


# ══════════════════════════════════════════════════════════════════════
# 2. Dedup set persists across restarts (the "keeps progressing" fix)
# ══════════════════════════════════════════════════════════════════════

def _agent(memory_dir):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = types.SimpleNamespace(memory_dir=memory_dir)
    return a


class TestReflectedIdsPersistence:
    def test_ids_survive_a_restart(self, tmp_path):
        a1 = _agent(tmp_path)
        ids = a1._get_reflected_ids()
        assert ids == set()
        ids.add("traj-1")
        ids.add("traj-2")
        a1._persist_reflected_ids()

        # "Restart" — a fresh agent on the same memory dir loads the set.
        a2 = _agent(tmp_path)
        assert a2._get_reflected_ids() == {"traj-1", "traj-2"}

    def test_no_memory_dir_is_safe(self):
        a = _agent(None)
        ids = a._get_reflected_ids()
        ids.add("x")
        a._persist_reflected_ids()  # no-op, must not raise
        assert "x" in a._get_reflected_ids()

    def test_set_is_bounded(self, tmp_path):
        a = _agent(tmp_path)
        ids = a._get_reflected_ids()
        for i in range(50):
            ids.add(f"t{i}")
        a._persist_reflected_ids(cap=10)
        assert len(a.context._reflected_trajectory_ids) == 10
        # And the bounded set round-trips.
        assert len(_agent(tmp_path)._get_reflected_ids()) == 10

    def test_load_is_once_then_cached(self, tmp_path):
        a = _agent(tmp_path)
        first = a._get_reflected_ids()
        first.add("only-in-memory")
        # Second call returns the SAME cached set (not a reload from disk).
        assert a._get_reflected_ids() is first
        assert "only-in-memory" in a._get_reflected_ids()
