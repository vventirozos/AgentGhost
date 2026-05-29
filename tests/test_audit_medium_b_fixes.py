"""Regression tests for Medium-tier batch B:

* optim.run_gepa._GhostLMAdapter.__call__: from inside a running loop it
  fell through to asyncio.run() and raised. Now runs on a worker thread.
* tools.registry: acquired-skill telemetry logged success=True even when
  tool_execute RETURNED an error string (EXIT CODE 1). Now classified.
* tools.browser extract_text: `length` reported the capped length, not
  the true page length, on truncation.
* tools.projects: zip(ids, subtasks) misaligned graph links when a blank
  subtask was dropped from ids. Now pairs are built correctly.
* tools.swarm: get_swarm_node was called twice per task (dispatch +
  worker), double-advancing the round-robin index. Now resolved once.
"""

import inspect
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.optim.run_gepa import _GhostLMAdapter
from ghost_agent.tools.registry import _acquired_skill_result_ok
from ghost_agent.tools.swarm import tool_delegate_to_swarm
from ghost_agent.tools import projects as projmod
from ghost_agent.memory.projects import ProjectStore


# -----------------------------------------------------------------
# run_gepa adapter — callable from within a running event loop
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_gepa_adapter_callable_from_running_loop():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "hello from worker"}}]
    })
    adapter = _GhostLMAdapter(llm, "test-model")
    # We ARE inside a running loop (pytest-asyncio). The old code raised
    # "asyncio.run() cannot be called from a running event loop".
    out = adapter("some prompt")
    assert out == ["hello from worker"]


# -----------------------------------------------------------------
# registry — acquired-skill result classification
# -----------------------------------------------------------------

def test_acquired_skill_result_ok_classifies_exit_code():
    assert _acquired_skill_result_ok("--- EXECUTION RESULT ---\nEXIT CODE: 0\nhi") is True
    assert _acquired_skill_result_ok("--- EXECUTION RESULT ---\nEXIT CODE: 1\nboom") is False
    assert _acquired_skill_result_ok("[SYSTEM ERROR] sandbox unavailable") is False
    assert _acquired_skill_result_ok("Error: something went wrong") is False
    # No exit-code banner, no error marker → treated as success.
    assert _acquired_skill_result_ok("Computed result: 42") is True


# -----------------------------------------------------------------
# browser — extract_text reports true (pre-truncation) length
# -----------------------------------------------------------------

def test_browser_extract_text_reports_true_length():
    from ghost_agent.tools import browser
    src = inspect.getsource(browser)
    # Both extract_text sites capture full length BEFORE truncating.
    assert src.count("full_len = len(text)") >= 2
    assert '"length": full_len' in src


# -----------------------------------------------------------------
# projects — graph links pair to the correct description
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_decompose_links_correct_pairs_when_blank_survives(monkeypatch, tmp_path):
    # Simulate a blank subtask surviving coercion (non-string item / split
    # edge) by making coercion a pass-through.
    monkeypatch.setattr(projmod, "_coerce_str_list", lambda v: v)
    captured = []
    monkeypatch.setattr(
        projmod, "_link_task_in_graph",
        lambda context, pid, tid, desc: captured.append((tid, desc)),
    )
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    context = SimpleNamespace(
        project_store=store, scratchpad=MagicMock(),
        graph_memory=None, current_project_id=None,
    )
    pid = store.create_project("P")

    await projmod.tool_manage_projects(
        context, action="task_decompose", project_id=pid,
        subtasks=["design the API", "", "implement it"],
    )

    pairs = {desc: tid for tid, desc in captured}
    assert "" not in pairs, "blank subtask must never be linked"
    assert "design the API" in pairs
    assert "implement it" in pairs, "real subtask dropped/shifted by misaligned zip"
    assert pairs["design the API"] != pairs["implement it"]


# -----------------------------------------------------------------
# swarm — node resolved exactly once per task (no double-advance)
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_swarm_resolves_node_once_per_task():
    calls = {"n": 0}
    node = {"client": AsyncMock(), "model": "m", "url": "http://node"}

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    node["client"].post = AsyncMock(return_value=resp)

    llm = MagicMock()
    llm.swarm_clients = [node]
    llm.circuit_breaker = MagicMock()

    def get_node(target=None):
        calls["n"] += 1
        return node

    llm.get_swarm_node = get_node

    result = await tool_delegate_to_swarm(
        llm, "fallback-model", MagicMock(),
        tasks=[{"instruction": "do x", "input_data": "data", "output_key": "k1"}],
        await_results=True,
    )

    assert "SUCCESS" in result or "completed" in result
    # Dispatch pre-validates (1 call); the worker reuses that node on
    # attempt 0. Before the fix it re-resolved → 2 calls per task.
    assert calls["n"] == 1, f"get_swarm_node called {calls['n']}x (double-advance)"
