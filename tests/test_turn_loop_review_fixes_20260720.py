"""Regression tests for the 2026-07-20 three-stack review fixes that live in
core/agent.py (the turn-loop coordinator's slice). Behavioral where feasible,
source-pinned where the logic is inside the monolithic dispatch/stream loop
that can't be exercised in isolation.
"""

import inspect
import types

import pytest

import ghost_agent.core.agent as agent_mod
from tests.helpers import make_agent


# ---------------------------------------------------------------- H4: prune

@pytest.mark.asyncio
async def test_prune_few_messages_huge_content_is_capped():
    # The <=5-message branch of _prune_context used to return the tail
    # VERBATIM (no _cap_oversized_tail), so a handful of giant tool results
    # shipped oversize → HTTP 400 → destructive recovery. It must now be
    # capped like the other two return paths.
    agent = make_agent()
    huge = "x " * 60000  # ~120k chars, well over any small budget
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": huge},
        {"role": "tool", "content": huge},
    ]
    out = await agent._prune_context(messages, max_tokens=2000, model="test-model")
    total = agent_mod._estimate_messages_tokens(out)
    # Capped near the budget, not the ~130k-token verbatim tail.
    assert total < 2000 * 3, f"few-messages branch not capped: {total} tokens"


# ------------------------------------------- H1: work_log helper behavior

@pytest.mark.asyncio
async def test_work_log_helper_writes_and_resets():
    agent = make_agent()
    calls = []

    class _Store:
        def add_work_log(self, pid, **kw):
            calls.append((pid, kw))

    agent.context.current_project_id = "proj1234abcd"
    agent.context.project_store = _Store()
    agent.context._project_work_files = {"README.md"}
    agent.context._project_work_tools = {"file_system": 1}
    agent.context._project_work_cmds = []
    agent.context._turn_failure_texts = []

    await agent._write_project_work_log_safe(
        last_user_content="add a readme",
        final_ai_content="done",
        execution_failure_count=0,
        verifier_backfill=None,
    )
    assert len(calls) == 1
    pid, kw = calls[0]
    assert pid == "proj1234abcd"
    assert "README.md" in kw["files"]           # original case preserved
    assert kw["outcome"] == "completed"
    # accumulators consumed so a follow-up turn can't re-attribute
    assert agent.context._project_work_files == set()
    assert agent.context._project_work_tools == {}


@pytest.mark.asyncio
async def test_work_log_helper_noop_without_project():
    agent = make_agent()
    calls = []

    class _Store:
        def add_work_log(self, pid, **kw):
            calls.append(pid)

    agent.context.current_project_id = None
    agent.context.project_store = _Store()
    agent.context._project_work_files = {"a.py"}
    await agent._write_project_work_log_safe(
        last_user_content="x", final_ai_content="y",
        execution_failure_count=0, verifier_backfill=None,
    )
    assert calls == []


# ---------------------------------------------- source-pinned dispatch fixes

def test_is_mutating_covers_unzip_git_clone_and_image_gen():
    src = inspect.getsource(agent_mod.GhostAgent._dispatch_and_process_tool_batch)
    seg = src[src.index("is_mutating ="):src.index("is_mutating =") + 600]
    for needle in ('"unzip"', '"git_clone"', '"image_generation"'):
        assert needle in seg, f"is_mutating missing {needle}"


def test_batch_collapse_unsafe_excludes_stateful_tools():
    unsafe = agent_mod._BATCH_COLLAPSE_UNSAFE
    for name in ("browser", "manage_projects", "notify_operator", "delegate"):
        assert name in unsafe
    src = inspect.getsource(agent_mod.GhostAgent._dispatch_and_process_tool_batch)
    assert "_collapse_unsafe = is_mutating or fname in _BATCH_COLLAPSE_UNSAFE" in src


def test_failure_attribution_uses_failing_tool_name():
    src = inspect.getsource(agent_mod.GhostAgent._dispatch_and_process_tool_batch)
    # captured at the failure sites, used for note_failure + fallback hint
    assert "failed_fname = fname" in src
    assert "_fail_fname = failed_fname or fname" in src
    assert "strikes.note_failure(\n                            _fail_fname" in src \
        or "note_failure(_fail_fname" in src.replace("\n", "").replace(" ", "")


def test_promotion_nudge_counts_only_successful_writes():
    src = inspect.getsource(agent_mod.GhostAgent._finalize_and_return)
    assert "_is_fs_write" in src
    # reads (raw content) and errors must not count
    assert 'startswith("SUCCESS")' in src


def test_scratchpad_injection_is_capped_at_source():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    assert "_SCRATCH_INJECT_CAP" in src


def test_prune_reserves_injection_headroom():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    assert "_history_budget" in src
    assert "_INJECTION_RESERVE_TOKENS" in src
    # prune is called against the reserved budget, not raw max_context
    assert "_prune_context(messages, max_tokens=_history_budget" in src


def test_streamed_traj_pending_field_removed():
    # The inert backfill machinery was removed; the field must not survive.
    src = inspect.getsource(agent_mod.GhostAgent)
    assert "_streamed_traj_pending" not in src
