"""Edit-churn brake + late-refute follow-up tasks (2026-07-25).

Two halves of the same incident pair:

* req 646e9b7e made 16 successive blind edits to index.html before its
  first browser load (turn 23 of 40), burned the whole turn budget, and
  failed. After ``_EDIT_CHURN_STEER_AFTER`` consecutive successful
  mutations of ONE file with no intervening run/render, a steer now
  demands verification; execute/browser/manage_services success resets
  the counters; at most ``_EDIT_CHURN_MAX_STEERS`` steers per request.

* the same request's LATE REFUTED named concrete leftovers ("line 1981
  still passes count") and the knowledge went nowhere but a banner.
  High-confidence late refutes now file the named issues as bounded,
  deduped project tasks (``_file_refute_followup_tasks``) so the next
  turn / autoadvance picks them up.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import (
    GhostAgent,
    GhostContext,
    _EDIT_CHURN_MAX_STEERS,
    _EDIT_CHURN_STEER_AFTER,
)
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


# --------------------------------------------------------------------------
# Harness (mirrors test_verifier_auto_repair.py)
# --------------------------------------------------------------------------

@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


def _write_call(path, content, tid):
    return {"choices": [{"message": {"content": "editing", "tool_calls": [
        {"id": tid, "function": {"name": "file_system", "arguments":
            f'{{"operation": "write", "path": "{path}", '
            f'"content": "{content}"}}'}}]}}]}


def _exec_call(tid):
    return {"choices": [{"message": {"content": "running", "tool_calls": [
        {"id": tid, "function": {"name": "execute",
                                 "arguments": '{"content": "ls"}'}}]}}]}


def _final(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


async def _run_spying(agent, user, llm_side_effects):
    """Drive handle_chat; return (per-call message histories, pretty_log
    spy). Steer counting goes through the pretty_log spy — asserting on
    message-history presence proved fragile because context PRUNING can
    evict an old steer message from the final call's history (observed
    when tokenizer state left by earlier test files inflated token
    estimates)."""
    calls = []
    seq = list(llm_side_effects)

    async def _spy(payload, *a, **k):
        calls.append([str(m.get("content", ""))
                      for m in payload.get("messages", [])])
        return seq[len(calls) - 1]

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_spy)
    body = {"messages": [{"role": "user", "content": user}],
            "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log") as plog:
        await agent.handle_chat(body, background_tasks=MagicMock())
    return calls, plog


def _steer_count(plog):
    """Number of edit-churn steer INJECTIONS, counted at the pretty_log
    call site (one "Edit Churn" log per injected steer)."""
    return sum(1 for c in plog.call_args_list
               if c.args and c.args[0] == "Edit Churn")


# --------------------------------------------------------------------------
# Edit-churn brake
# --------------------------------------------------------------------------

async def test_three_unverified_edits_same_file_get_steered(agent):
    writes = [f"SUCCESS: wrote {n} chars to 'index.html'." for n in (10, 20, 30)]
    agent.available_tools["file_system"] = AsyncMock(side_effect=writes)
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: clean")

    calls, plog = await _run_spying(agent, "fix the calendar", [
        _write_call("index.html", "v1", "t1"),
        _write_call("index.html", "v2", "t2"),
        _write_call("index.html", "v3", "t3"),
        _exec_call("t4"),               # verification after the steer
        _final("Done, verified."),
    ])

    assert _steer_count(plog) == 1
    steer_log = next(c for c in plog.call_args_list
                     if c.args and c.args[0] == "Edit Churn")
    assert "index.html" in steer_log.args[1]
    # The steer text itself reached the conversation.
    assert any("edit-churn check" in m for call in calls for m in call)


async def test_interleaved_verification_resets_counter(agent):
    writes = [f"SUCCESS: wrote {n} chars to 'index.html'." for n in (10, 20, 30)]
    agent.available_tools["file_system"] = AsyncMock(side_effect=writes)
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: clean")

    calls, plog = await _run_spying(agent, "fix the calendar", [
        _write_call("index.html", "v1", "t1"),
        _write_call("index.html", "v2", "t2"),
        _exec_call("t3"),               # verification resets the counter
        _write_call("index.html", "v3", "t4"),
        _exec_call("t5"),
        _final("Done."),
    ])

    assert _steer_count(plog) == 0


async def test_edits_across_different_files_do_not_trip(agent):
    # A scaffold turn writing N DIFFERENT files once each is legitimate.
    writes = [f"SUCCESS: wrote 10 chars to 'f{i}.py'." for i in range(4)]
    agent.available_tools["file_system"] = AsyncMock(side_effect=writes)
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: clean")

    calls, plog = await _run_spying(agent, "scaffold the app", [
        _write_call("f0.py", "a", "t1"),
        _write_call("f1.py", "b", "t2"),
        _write_call("f2.py", "c", "t3"),
        _write_call("f3.py", "d", "t4"),
        _exec_call("t5"),
        _final("Scaffolded."),
    ])

    assert _steer_count(plog) == 0


async def test_steers_are_bounded_per_request(agent):
    n_writes = _EDIT_CHURN_STEER_AFTER * (_EDIT_CHURN_MAX_STEERS + 2)
    writes = [f"SUCCESS: wrote {i} chars to 'index.html'."
              for i in range(n_writes)]
    agent.available_tools["file_system"] = AsyncMock(side_effect=writes)
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: clean")

    seq = [_write_call("index.html", f"v{i}", f"t{i}")
           for i in range(n_writes)]
    seq += [_exec_call("tx"), _final("Done.")]
    calls, plog = await _run_spying(agent, "keep fixing it", seq)

    assert _steer_count(plog) == _EDIT_CHURN_MAX_STEERS


# --------------------------------------------------------------------------
# Late-refute follow-up tasks
# --------------------------------------------------------------------------

def _refute(issues, conf=0.95):
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=conf,
                        reasoning="refuted", issues=issues)


def _store(status="ACTIVE", existing=()):
    store = MagicMock()
    store.get_project.return_value = {"id": "p1", "status": status}
    store.list_tasks.return_value = [
        {"description": d, "status": "PENDING"} for d in existing]
    return store


ISSUE_A = "Line 1981 still passes count to the Moves Drilled modal"
ISSUE_B = "The detail modal never renders the nested weight entry"
ISSUE_C = "Calendar month view still renders only one week of days"


def test_files_bounded_deduped_tasks_on_active_project(agent):
    store = _store()
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(
            _refute([ISSUE_A, ISSUE_B, ISSUE_C]), "p1")
    # Bounded at _REFUTE_TASK_MAX (2) even with 3 issues.
    assert store.add_task.call_count == 2
    filed = [c.args[1] for c in store.add_task.call_args_list]
    assert all(d.startswith("Verifier follow-up: ") for d in filed)
    assert ISSUE_A in filed[0]


def test_existing_task_is_not_duplicated(agent):
    store = _store(existing=[f"Verifier follow-up: {ISSUE_A}"])
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(_refute([ISSUE_A, ISSUE_B]), "p1")
    filed = [c.args[1] for c in store.add_task.call_args_list]
    assert len(filed) == 1 and ISSUE_B in filed[0]


def test_short_issues_and_missing_project_are_skipped(agent):
    store = _store()
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(_refute(["too short"]), "p1")
        agent._file_refute_followup_tasks(_refute([ISSUE_A]), None)
    assert store.add_task.call_count == 0


@pytest.mark.parametrize("status", ["RELEASED", "ARCHIVED"])
def test_terminal_immutable_projects_never_reopened(agent, status):
    store = _store(status=status)
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(_refute([ISSUE_A]), "p1")
    assert store.add_task.call_count == 0


def test_done_project_respects_reopen_cap(agent):
    """Refute-driven reopens share the defect_reopens budget: when the
    atomic gate says the cap is hit, nothing is filed."""
    store = _store(status="DONE")

    def _gate_denies(pid, fn):
        # Simulate a metadata window already at the cap: the callback runs
        # but must NOT record ok (recent list stays full).
        import time as _t
        fn({"defect_reopens": [_t.time(), _t.time()]})
    store._atomic_metadata_update.side_effect = _gate_denies
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(_refute([ISSUE_A]), "p1")
    assert store.add_task.call_count == 0


def test_done_project_under_cap_reopens_and_files(agent):
    store = _store(status="DONE")

    def _gate_allows(pid, fn):
        fn({"defect_reopens": []})
    store._atomic_metadata_update.side_effect = _gate_allows
    agent.context.project_store = store
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._file_refute_followup_tasks(_refute([ISSUE_A]), "p1")
    assert store.add_task.call_count == 1


def test_kill_switch_disables_filing(agent, monkeypatch):
    monkeypatch.setenv("GHOST_REFUTE_FOLLOWUP_TASKS", "0")
    store = _store()
    agent.context.project_store = store
    agent._file_refute_followup_tasks(_refute([ISSUE_A]), "p1")
    assert store.add_task.call_count == 0


async def test_record_late_verdict_wires_filing_with_project_id(agent):
    agent.context.skill_memory = None
    agent.context.project_store = _store()
    agent._backfill_trajectory_outcome = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_late_verdict(
            _refute([ISSUE_A]), trajectory_id="traj1", conv_fp="c",
            last_tool={"name": "execute", "content": "x"},
            project_id="p1")
    assert agent.context.project_store.add_task.call_count == 1


async def test_low_confidence_refute_files_nothing(agent):
    agent.context.skill_memory = None
    agent.context.project_store = _store()
    agent._backfill_trajectory_outcome = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_late_verdict(
            _refute([ISSUE_A], conf=0.5), trajectory_id="traj1",
            conv_fp="c", last_tool={"name": "execute", "content": "x"},
            project_id="p1")
    assert agent.context.project_store.add_task.call_count == 0
