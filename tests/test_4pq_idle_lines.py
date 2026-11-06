"""§4PQ (2026-10-16) — the idle-cycle lines say what each job is FOR and
what it did. The operator: "make the logs for the idle cycle tasks more
descriptive, right now it's not clear what each does". The lines read
"ran tidy (idle 15m)", "ran store-hygiene (idle 29m)" and "maintenance:
project digests 0, manifests 0, snapshot no, graph pruned 0, graph merged 0".
"""
import ast
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.agent import (IDLE_JOB_PURPOSE, _store_hygiene_note,
                                    idle_cycle_line)
from ghost_agent.core.dream import maintenance_summary

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


# ── the maintenance summary ─────────────────────────────────────────────────

def test_a_quiet_maintenance_pass_says_nothing_was_needed():
    """Fails where an all-zero pass read like a list of counters."""
    s = maintenance_summary(0, 0, "", False)
    assert s.startswith("nothing needed — ")
    assert "no store backup this pass (one a day)" in s
    assert "memory graph cleanup not due (every 6 h)" in s


def test_a_working_pass_names_each_thing_it_changed():
    s = maintenance_summary(2, 1, "snapshot store-20261016.tar.gz", True, 3, 1)
    assert not s.startswith("nothing needed")
    assert "2 project activity digests written" in s
    assert "1 project file description written" in s
    assert "store backup taken (store-20261016.tar.gz)" in s
    assert "3 stale links removed, 1 duplicate entity merged" in s


def test_a_failed_backup_is_not_called_nothing_needed():
    """Fails where a FAILED snapshot hid behind the quiet-pass wording."""
    s = maintenance_summary(0, 0, "store snapshot FAILED", False)
    assert "store backup FAILED" in s
    assert not s.startswith("nothing needed")


def test_a_graph_pass_that_ran_says_ran_not_due():
    assert "memory graph cleaned: 0 stale links removed" in maintenance_summary(0, 0, "", True)


def test_a_graph_pass_that_raised_says_failed():
    """r1 m4: fails where a raising store maintenance read "cleaned"."""
    s = maintenance_summary(0, 0, "", "failed")
    assert "memory graph cleanup FAILED" in s and "cleaned" not in s
    assert not s.startswith("nothing needed")


def test_a_reconcile_only_pass_is_not_nothing_needed():
    """r1 M2: fails where a pass whose only work was the cross-store
    reconcile (ledgered as work) read "nothing needed"."""
    s = maintenance_summary(0, 0, "", True, 0, 0, "memory reconcile: 2 orphan outline(s) dropped")
    assert not s.startswith("nothing needed")
    assert s.endswith("; memory reconcile: 2 orphan outline(s) dropped")


@pytest.mark.asyncio
async def test_the_dream_pass_carries_the_reconcile_report():
    from ghost_agent.core.dream import Dreamer
    from unittest.mock import AsyncMock
    ctx = MagicMock()
    ctx.memory_system.collection = MagicMock()
    ctx.project_store = None
    ctx.trajectory_collector = None
    ctx.memory_dir = None
    ctx.graph_memory = None
    d = Dreamer(ctx)
    with patch.object(Dreamer, "_backfill_file_manifests", AsyncMock(return_value=0)), \
         patch.object(Dreamer, "_maintenance_due", return_value=True), \
         patch.object(Dreamer, "_compress_graph_nodes", AsyncMock(return_value=0)), \
         patch.object(Dreamer, "_reconcile_memory_stores",
                      AsyncMock(return_value="memory reconcile: 1 catalogue entry dropped")):
        await d.dream(model_name="m")
    out = d.last_dream_outcome
    assert out["side_output"] is True
    assert "memory reconcile: 1 catalogue entry dropped" in out["summary"]
    assert not out["summary"].startswith("nothing needed")


@pytest.mark.asyncio
@pytest.mark.parametrize("summary,expect", [
    ("nothing needed — x; no store backup this pass (one a day); y", "nothing needed"),
    ("0 project activity digests written; store backup taken (s.tar); y", "did work (see the Idle Maintenance line)"),
    ("0 project activity digests written; store backup FAILED; y", "something FAILED (see the Idle Maintenance line)"),
])
async def test_the_cycle_note_follows_the_maintenance_summary(summary, expect):
    """r1 M1: fails where the cycle line said "nothing needed" for a pass
    whose only work was the backup (side_output False), or a FAILED one."""
    from tests.test_biological_watchdog import _make_agent
    from unittest.mock import AsyncMock
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    agent.context.project_store = None
    fake = MagicMock()
    fake.dream = AsyncMock(return_value="x")
    fake.last_dream_outcome = {"phase": "skipped", "side_output": False, "summary": summary}
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle" and "maintenance (" in c]
    assert idle, calls
    assert f"maintenance ({IDLE_JOB_PURPOSE['maintenance']}): {expect}" in idle[0]


# ── the Idle Cycle line ─────────────────────────────────────────────────────

def _appended_job_names():
    """Every literal job name the scheduler appends to ``_idle_ran``."""
    tree = ast.parse((SRC / "core/agent.py").read_text(encoding="utf-8"))
    out = set()
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "append" and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "_idle_ran" and c.args
                and isinstance(c.args[0], ast.Constant)):
            out.add(c.args[0].value)
    return out


def test_every_idle_job_has_a_purpose():
    """Enumerated over the source: fails where a new idle job is added
    without saying what it is for."""
    names = _appended_job_names()
    assert {"maintenance", "store-hygiene", "tidy"} <= names and "failure-replay" not in names   # §4SZ
    assert names <= set(IDLE_JOB_PURPOSE), names - set(IDLE_JOB_PURPOSE)


def test_the_line_carries_purpose_and_outcome_per_job():
    line = idle_cycle_line(["tidy", "store-hygiene"],
                           {"tidy": "nothing to remove in 3 project workspace(s)",
                            "store-hygiene": "nothing to fix"}, 15.2)
    assert line.startswith("idle 15m — ran tidy (")
    assert IDLE_JOB_PURPOSE["tidy"] in line and IDLE_JOB_PURPOSE["store-hygiene"] in line
    assert "nothing to remove in 3 project workspace(s)" in line
    assert " | store-hygiene (" in line and line.endswith(": nothing to fix")


def test_a_failed_job_says_failed_with_its_purpose():
    line = idle_cycle_line(["maintenance(failed)"], {"maintenance": "nothing needed"}, 20)
    assert (f"maintenance ({IDLE_JOB_PURPOSE['maintenance']}): had an error (see the log); "
            "then nothing needed") in line
    assert line.count("had an error") == 1


def test_store_hygiene_note_names_counts_and_failures():
    assert _store_hygiene_note({"removed": [], "marked": []}) == "nothing to fix"
    assert _store_hygiene_note({"removed": ["a"], "marked": []}, ["temporary-document sweep"]) == (
        "1 expired video transcript(s) removed; temporary-document sweep FAILED")
    # a failure alone is not "nothing to fix"
    assert _store_hygiene_note({}, ["temporary-document sweep"]) == "temporary-document sweep FAILED"
    assert _store_hygiene_note(None) == "nothing to fix"


# ── through the real tick ───────────────────────────────────────────────────

def _tick_lines(agent):
    calls = []

    def _cap(t, c=None, **k):
        calls.append((t, str(c)))
    return calls, _cap


@pytest.mark.asyncio
async def test_the_tick_prints_tidy_with_its_purpose_and_outcome():
    """Through ``_biological_tick``: fails where the console line went
    back to a bare job name."""
    from tests.test_biological_watchdog import _make_agent
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    _store = MagicMock()
    _store.list_projects.return_value = [{"id": "p1"}, {"id": "p2"}]
    agent.context.project_store = _store
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.workspace_cleanup.tidy_project_workspace",
               return_value={"deleted": [], "freed_bytes": 0}), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle"]
    assert idle, calls
    assert f"tidy ({IDLE_JOB_PURPOSE['tidy']}): nothing to remove in 2 project workspace(s)" in idle[0]


@pytest.mark.asyncio
async def test_the_tick_prints_store_hygiene_counts():
    from tests.test_biological_watchdog import _make_agent
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    agent.context.project_store = None
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.memory.temp_documents.sweep",
               return_value={"removed": ["a", "b"], "marked": [], "dropped": []}), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle"]
    assert idle and "store-hygiene (" in idle[0], calls
    assert "2 expired video transcript(s) removed" in idle[0]


def test_purposes_are_keyed_by_the_scheduler_labels():
    """Fails where a purpose is keyed by a name the scheduler does not use
    (a rename that left the line without its description)."""
    from ghost_agent.core.agent import IDLE_SCHEDULER_JOBS
    labels = {label for label, _t, _f in IDLE_SCHEDULER_JOBS}
    assert set(IDLE_JOB_PURPOSE) <= labels, set(IDLE_JOB_PURPOSE) - labels


@pytest.mark.asyncio
async def test_a_raising_store_maintenance_reads_failed():
    """r1 m4 through the pass: fails where a raise inside the 6-hourly
    store maintenance was printed as "memory graph cleaned"."""
    from ghost_agent.core.dream import Dreamer
    from unittest.mock import AsyncMock
    ctx = MagicMock()
    ctx.memory_system.collection = MagicMock()
    ctx.project_store = None
    ctx.trajectory_collector = None
    ctx.memory_dir = None
    d = Dreamer(ctx)
    with patch.object(Dreamer, "_backfill_file_manifests", AsyncMock(return_value=0)), \
         patch.object(Dreamer, "_maintenance_due", return_value=True), \
         patch.object(Dreamer, "_store_maintenance", AsyncMock(side_effect=RuntimeError("x"))):
        await d.dream(model_name="m")
    s = d.last_dream_outcome["summary"]
    assert "memory graph cleanup FAILED" in s and "cleaned" not in s


@pytest.mark.asyncio
async def test_a_maintenance_error_outcome_reads_failed():
    """r2 m2: fails where phase "error" (memory system gone) read "did work"."""
    from tests.test_biological_watchdog import _make_agent
    from unittest.mock import AsyncMock
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    agent.context.project_store = None
    fake = MagicMock()
    fake.dream = AsyncMock(return_value="x")
    fake.last_dream_outcome = {"phase": "error", "side_output": False,
                               "summary": "maintenance: memory system not available"}
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle" and "maintenance (" in c]
    assert idle and "): FAILED: memory system not available" in idle[0], calls


@pytest.mark.asyncio
async def test_tidy_counts_only_the_workspaces_it_checked():
    """r2 m3: fails where a skipped project (keep-set unreadable) was
    reported as checked and clean; a project with no workspace yet is
    not a fault."""
    from tests.test_biological_watchdog import _make_agent
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    _store = MagicMock()
    _store.list_projects.return_value = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    agent.context.project_store = _store
    res = {"a": {"status": "ok", "deleted": []},
           "b": {"status": "skipped: artifact read failed"},
           "c": {"status": "skipped: no project dir"}}
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.workspace_cleanup.tidy_project_workspace",
               side_effect=lambda st, pid: res[pid]), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle"]
    assert idle and "nothing to remove in 2 project workspace(s) (1 not checked)" in idle[0], calls


@pytest.mark.asyncio
async def test_a_raising_narrative_is_listed_as_failed():
    """r2 m1: fails where the narrative's FAILED note never reached the
    line (the job was listed only on success)."""
    from tests.test_biological_watchdog import _make_agent
    from unittest.mock import AsyncMock
    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    agent.context.project_store = None
    sm = MagicMock(enabled=True)
    sm.consolidate_narrative = AsyncMock(side_effect=RuntimeError("worker down"))
    agent.context.self_model = sm
    calls, cap = _tick_lines(agent)
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.agent.pretty_log", side_effect=cap), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    idle = [c for t, c in calls if t == "Idle Cycle"]
    assert idle and "selfhood-narrative (" in idle[0] and "FAILED" in idle[0], calls
