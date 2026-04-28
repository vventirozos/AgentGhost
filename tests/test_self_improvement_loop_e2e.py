"""End-to-end integration test for Ghost's stage-1 self-improvement
loop.

Pins the entire chain that the audit identified as previously
*technically* closed but practically open:

  user-correction → trajectory promoted to FAILED →
  Reflector.reflect_one (post-turn, fire-and-forget) →
  composite sink (JSONL + SkillMemory.learn_lesson) →
  next user query retrieves the lesson via SkillMemory.get_playbook_context

The test wires REAL ``TrajectoryCollector``, ``Reflector``, and
``SkillMemory`` against a temp dir. The LLM client is stubbed (the
critique returns a fixed parseable response), but every other hop is
the production code path. This test is the regression gate for the
closed-loop claim — if any link in the chain is broken by a future
refactor, this test goes red.

Why a single test file: the per-module unit tests exercise each link
in isolation. This test exists *because* every individual link can
be green while their composition is broken. It's the integration
ratchet, not a duplicate of the unit coverage.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory, Outcome
from ghost_agent.memory.skills import SkillMemory
from ghost_agent.reflection.loop import Reflector


# A reflection-prompt response that ``parse_reflection_output`` will
# parse into both a diagnosis and a revised plan. Keep this terse —
# the chain test isn't about prompt quality, it's about wiring.
_FIXED_CRITIQUE = (
    "DIAGNOSIS: agent answered with go files when the user asked for "
    "python files\n"
    "REVISED PLAN:\n"
    "1. parse the user request literally for the language token\n"
    "2. filter the workspace listing on the matching extension\n"
    "3. return only that filtered subset\n"
)


def _bare_agent(ctx) -> GhostAgent:
    a = GhostAgent.__new__(GhostAgent)
    a.context = ctx
    return a


def _build_composite_sink(traj_collector: TrajectoryCollector,
                          skill_memory: SkillMemory):
    """Mirror of the production composite sink wired in main.py:
    JSONL append + SkillMemory.learn_lesson. Kept inline (not
    imported) so a refactor of main.py's wiring doesn't silently
    bypass this regression gate."""
    def _sink(reflected_trajectory: Trajectory):
        traj_collector.append(reflected_trajectory)
        src_reason = (
            reflected_trajectory.extra.get("source_failure_reason", "")
            or "failure"
        )
        plan_text = (
            reflected_trajectory.planning_output
            or reflected_trajectory.final_response
        )
        skill_memory.learn_lesson(
            task=(reflected_trajectory.user_request or "")[:400],
            mistake=str(src_reason)[:400],
            solution=str(plan_text)[:1200],
            memory_system=None,
        )
    return _sink


async def test_user_correction_lesson_lands_and_is_retrievable(tmp_path):
    # ----- 1. Wire the production-shaped components in tmp_path -----
    traj_root = tmp_path / "trajectories"
    traj_collector = TrajectoryCollector(root=traj_root, session_id="e2e")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    skill_memory = SkillMemory(memory_dir)

    async def critique(_prompt):
        return _FIXED_CRITIQUE

    reflector = Reflector(critique_fn=critique, per_call_timeout_s=10.0)
    composite_sink = _build_composite_sink(traj_collector, skill_memory)

    ctx = SimpleNamespace(
        trajectory_collector=traj_collector,
        reflector=reflector,
        reflection_sink=composite_sink,
        last_user_content="",
    )
    agent = _bare_agent(ctx)

    # ----- 2. Simulate the prior turn that the user is about to correct -----
    failed_user_request = "list every python file in the workspace directory"
    bad_assistant_response = (
        "Here are the go files in your workspace: a.go b.go c.go"
    )
    traj = Trajectory(
        id="prior-turn",
        user_request=failed_user_request,
        final_response=bad_assistant_response,
        outcome=Outcome.UNKNOWN.value,
    )
    traj_collector.append(traj)
    agent._stash_trajectory_for_correction_lookup(traj)

    # ----- 3. The user's correction turn arrives -----
    correction_text = (
        "no, list every python file in the workspace - python not go"
    )
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": failed_user_request},
        {"role": "assistant", "content": bad_assistant_response},
        {"role": "user", "content": correction_text},
    ]
    agent._maybe_promote_prior_turn_via_user_correction(
        messages, correction_text
    )

    # The promotion must have happened in-memory + on the sidecar.
    assert traj.outcome == Outcome.FAILED.value
    sidecar = traj_root / "corrections.jsonl"
    assert sidecar.exists()
    rec = json.loads(sidecar.read_text().splitlines()[0])
    assert rec["trajectory_id"] == "prior-turn"
    assert rec["outcome"] == Outcome.FAILED.value
    assert rec["source"] == "user_correction"

    # ----- 4. Drain the scheduled reflection task -----
    pending = ctx._pending_reflection_tasks
    assert pending, "no reflection task was scheduled"
    await asyncio.gather(*list(pending))

    # ----- 5. Reflection landed: JSONL has a `task_kind=reflection` line -----
    reflection_records = [
        t for t in traj_collector.iter_trajectories()
        if t.task_kind == "reflection"
    ]
    assert len(reflection_records) == 1
    assert reflection_records[0].extra["reflected_from"] == "prior-turn"

    # ----- 6. Reflection landed: SkillMemory has a lesson ------------
    playbook = skill_memory._load_playbook()
    assert len(playbook) >= 1
    lesson = playbook[0]
    # Lesson should reference the original request as its trigger /
    # task — that's the field the next-turn retrieval matches against.
    trigger = (lesson.get("trigger") or lesson.get("task") or "").lower()
    assert "python" in trigger
    assert "workspace" in trigger or "file" in trigger

    # ----- 7. NEXT user turn: a similar query retrieves the lesson ---
    # Simulates the planner's playbook-prefetch on the agent's NEXT
    # turn. We pass the BM25 path (no memory_system) — this is the
    # path that fires when the vector store is unavailable, and the
    # one we can exercise without ChromaDB.
    similar_query = "show me all python files in workspace"
    surfaced = skill_memory.get_playbook_context(
        query=similar_query, memory_system=None
    )
    assert surfaced, (
        "next-turn retrieval returned empty — the loop is open"
    )
    # The retrieved context must reference the lesson we just learned.
    surfaced_lower = surfaced.lower()
    assert (
        "python" in surfaced_lower
        and ("file" in surfaced_lower or "workspace" in surfaced_lower)
    ), f"retrieved context didn't include the new lesson; got: {surfaced!r}"

    # ----- 8. Negative control: an unrelated query must NOT retrieve --
    # the lesson. (This pins the bm25_fallback fix — recency-only
    # retrieval would surface an unrelated lesson.) The query is
    # chosen to share NO tokens with the lesson trigger; the
    # production ``_bm25_like_score`` doesn't strip common-stopword
    # overlap (e.g. "the"), so a query containing "the" against a
    # trigger containing "the" surfaces a non-zero score. This is a
    # quirk of the existing BM25 helper, not the loop logic — keep
    # the unrelated query orthogonal to avoid coupling this loop
    # test to that BM25 detail.
    unrelated = skill_memory.get_playbook_context(
        query="tell me about astronomy and black holes",
        memory_system=None,
    )
    if unrelated:
        # If anything came back, it must NOT be our python-file lesson.
        assert "python" not in unrelated.lower(), (
            "unrelated query surfaced the python-file lesson — "
            "retrieval is leaking via recency"
        )


async def test_loop_is_idempotent_under_repeated_correction(tmp_path):
    """Two correction-shaped follow-ups in a row must not result in
    duplicate lessons or duplicate reflection scheduling. Pin the
    'one promotion per stashed trajectory' contract end-to-end."""
    traj_root = tmp_path / "trajectories"
    traj_collector = TrajectoryCollector(root=traj_root, session_id="e2e")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    skill_memory = SkillMemory(memory_dir)

    async def critique(_p):
        return _FIXED_CRITIQUE

    reflector = Reflector(critique_fn=critique, per_call_timeout_s=10.0)
    composite_sink = _build_composite_sink(traj_collector, skill_memory)
    ctx = SimpleNamespace(
        trajectory_collector=traj_collector,
        reflector=reflector,
        reflection_sink=composite_sink,
        last_user_content="",
    )
    agent = _bare_agent(ctx)

    bad_assistant = "Here is incorrect information about postgres"
    traj = Trajectory(
        id="t-only",
        user_request="how do postgres indexes work",
        final_response=bad_assistant,
        outcome=Outcome.UNKNOWN.value,
    )
    traj_collector.append(traj)
    agent._stash_trajectory_for_correction_lookup(traj)

    correction = (
        "no, how do postgres indexes work specifically btree gin gist"
    )
    msgs = [
        {"role": "user", "content": "how do postgres indexes work"},
        {"role": "assistant", "content": bad_assistant},
        {"role": "user", "content": correction},
    ]
    agent._maybe_promote_prior_turn_via_user_correction(msgs, correction)
    agent._maybe_promote_prior_turn_via_user_correction(msgs, correction)
    # Drain whatever scheduled.
    pending = ctx._pending_reflection_tasks or set()
    if pending:
        await asyncio.gather(*list(pending))

    # Exactly ONE reflection record + ONE lesson, despite TWO calls.
    reflection_records = [
        t for t in traj_collector.iter_trajectories()
        if t.task_kind == "reflection"
    ]
    assert len(reflection_records) == 1
    playbook = skill_memory._load_playbook()
    assert len(playbook) == 1
