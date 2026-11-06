"""§4NM batch 5 (2026-10-11): the inline planner, utility pruning, the
selfhood wake-up prefix, the derived-mood idle job, the production-off
switches and the bench seams were deleted; the idle "dream" job became
"maintenance". Each test names the world it fails in."""
import asyncio
import datetime
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── the inline planner is gone ───────────────────────────────────────────────

def test_the_use_planning_flag_is_rejected(monkeypatch):
    """Fails where `--use-planning` still parses: a launcher that passed it
    would boot a planner path that no longer exists."""
    from ghost_agent import main as M
    monkeypatch.setattr(sys, "argv", ["ghost", "--use-planning"])
    with pytest.raises(SystemExit):
        M.parse_args()


def test_no_planner_symbol_survives_in_the_turn_loop():
    import ghost_agent.core.agent as A
    import ghost_agent.core.prompts as P
    for name in ("_prepare_planning_context", "_get_recent_transcript"):
        assert not hasattr(A.GhostAgent, name), name
    for name in ("_NoPlannerSnapshot", "salvage_truncated_plan", "planner_repeat_needs_reask",
                 "_PLANNER_TIMEOUT_S", "_SELFHOOD_PREFIX_ENABLED", "_PLAN_IS_NOT_A_PROJECT_NOTE"):
        assert not hasattr(A, name), name
    assert not hasattr(P, "PLANNING_SYSTEM_PROMPT")
    # (§4NN: the experiments registry that listed `use_planning` is gone)


def test_the_project_task_tree_is_kept():
    """TaskTree is live for projects — only the turn-loop planner went."""
    from ghost_agent.core.planning import TaskTree, TaskStatus
    t = TaskTree()
    assert TaskStatus.DONE is not None and hasattr(t, "render")


# ── utility pruning and the derived-mood idle job ────────────────────────────


def test_the_idle_tick_moves_no_mood():
    """The idle phase that minted "idle" moods was dropped by the operator
    (§4NM); the derived updater itself is gone (§4TC). No idle job writes the
    mood: it decays when it is read, and only an owner's turn, the owner's
    reactions and the owner's word move it."""
    import ast
    import inspect
    import ghost_agent.core.agent as A
    import textwrap
    writers = {"update_derived_mood", "note_owner_turn", "note_owner_reaction", "settle_mood",
               "nudge", "set_owner_mood", "reset_mood", "set_manner"}
    tick = ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent._biological_tick)))
    assert not [n for n in ast.walk(tick) if isinstance(n, ast.Attribute) and n.attr in writers]
    assert not hasattr(A.GhostAgent, "_DERIVED_MOOD_COOLDOWN")
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(A)))
             if isinstance(n, ast.Attribute) and n.attr == "update_derived_mood"]
    assert not calls, "the derived updater has no caller left"


# ── dream → maintenance ──────────────────────────────────────────────────────

def test_a_pre_rename_heartbeat_reads_as_maintenance(tmp_path):
    """Fails where the liveness table showed a DEAD "maintenance" beside a
    healthy "dream" for the first window after the deploy."""
    from ghost_agent.core.autonomous_activity import (
        ATTEMPT_DECLINED, attempts_path, canonical_phase, read_attempts)
    led = tmp_path / "autonomous_activity.jsonl"
    now = time.time()
    attempts_path(led).write_text(json.dumps({
        "dream": {"ts": now - 10, "result": ATTEMPT_DECLINED},
        "maintenance": {"ts": now - 500, "result": ATTEMPT_DECLINED},
    }))
    got = read_attempts(led)
    assert set(got) == {"maintenance"} and got["maintenance"]["ts"] == pytest.approx(now - 10)
    assert canonical_phase("dream") == "maintenance" and canonical_phase("self_play") == "self_play"


@pytest.mark.asyncio
async def test_the_pass_reports_what_it_did(tmp_path):
    from ghost_agent.core.dream import Dreamer
    ctx = MagicMock()
    ctx.memory_system.collection = MagicMock()
    ctx.project_store = None
    ctx.trajectory_collector = None
    ctx.memory_dir = None
    d = Dreamer(ctx)
    with patch.object(Dreamer, "_backfill_file_manifests", AsyncMock(return_value=3)), \
         patch.object(Dreamer, "_maintenance_due", return_value=False):
        await d.dream(model_name="m")
    assert d.last_dream_outcome["summary"] == (   # §4PQ: in words
        "0 project activity digests written; 3 project file descriptions written; "
        "no store backup this pass (one a day); memory graph cleanup not due (every 6 h)")


@pytest.mark.asyncio
async def test_the_idle_tick_logs_and_ledgers_the_summary():
    """Fails where the tick announced "Entering spontaneous REM cycle..." and
    ledgered under the old "dream" phase."""
    from ghost_agent.core.agent import GhostAgent
    ctx = MagicMock()
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.journal = None
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=900)
    ctx.args = MagicMock()
    ctx.args.model = "m"
    ctx.self_model = None
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._record_autonomous_activity = MagicMock()
    fake = MagicMock()
    fake.dream = AsyncMock(return_value="x")
    fake.last_dream_outcome = {"phase": "skipped", "side_output": True,
                               "summary": "maintenance: project digests 2, manifests 0, "
                                          "snapshot no, graph pruned 0, graph merged 0"}
    logged = []
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1), \
         patch("ghost_agent.core.agent.pretty_log",
               side_effect=lambda *a, **k: logged.append(" ".join(str(x) for x in a))):
        await agent._biological_tick()
    rows = [c.args for c in agent._record_autonomous_activity.call_args_list]
    assert ("maintenance", fake.last_dream_outcome["summary"]) in rows
    assert not any(r[0] == "dream" for r in rows)
    assert any("maintenance: project digests 2" in line for line in logged)
    assert not any("REM" in line for line in logged)


# ── the bench seams ──────────────────────────────────────────────────────────
