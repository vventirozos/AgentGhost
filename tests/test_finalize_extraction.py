"""#5 decomposition step 3 — `_finalize_and_return` + FinalizeState.

The post-turn-loop finalization chain (output scrubbers → deferred
Perfect-It → final verifier gate/calibration → competence & skill credit →
episode recording → correction stash → response return) extracted VERBATIM
from handle_chat. Zero control-flow rewrites: the region's own `return` is
the method's return. FinalizeState is read-only — nothing after the region
reads handle_chat locals, so there is no repack.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import FinalizeState, GhostAgent


def _make_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    ctx.args.perfect_it = False
    agent = GhostAgent(ctx)
    return agent


def _fs(**over):
    fields = dict(
        body={"messages": [], "stream": False},
        created_time=1234567890,
        current_plan_json={},
        current_trajectory_id="traj-1",
        execution_failure_count=0,
        final_ai_content="All done.",
        force_stop=False,
        forget_was_called=False,
        last_user_content="do the thing",
        last_was_failure=False,
        lc="do the thing",
        messages=[{"role": "user", "content": "do the thing"}],
        model="test-model",
        payload=None,
        req_id="ab12cd34",
        thought_content="",
        tools_run_this_turn=[],
        wakeup_prefix="",
        was_complex_task=False,
        _stable_conv_fp="fp",
        _verdict_is_fresh=False,
        _verifier_verdict_cache=None,
    )
    fields.update(over)
    return FinalizeState(**fields)


# ── contract shape ───────────────────────────────────────────────────────────

def test_finalize_state_is_read_only_inputs():
    # no repack contract — FinalizeState must not grow MUTATED_FIELDS
    assert not hasattr(FinalizeState, "MUTATED_FIELDS")
    assert "final_ai_content" in FinalizeState.__dataclass_fields__
    assert "payload" in FinalizeState.__dataclass_fields__


def test_handle_chat_delegates_to_finalize():
    src = inspect.getsource(GhostAgent.handle_chat)
    assert "_finalize_and_return(FinalizeState(" in src
    # the chain's landmarks must be gone from handle_chat
    assert "FINAL OUTPUT SCRUBBER" not in src
    assert "bleed_markers" not in src


def test_method_carries_the_finalization_landmarks():
    src = inspect.getsource(GhostAgent._finalize_and_return)
    for marker in ("FINAL OUTPUT SCRUBBER", "bleed_markers",
                   "_deferred_perfect_it", "verifier_backfill",
                   "return final_ai_content, created_time, req_id"):
        assert marker in src, f"finalization landmark missing: {marker}"


# ── behavior ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_the_response_triple():
    agent = _make_agent()
    out, created, rid = await agent._finalize_and_return(_fs())
    assert "All done." in out
    assert created == 1234567890
    assert rid == "ab12cd34"


@pytest.mark.asyncio
async def test_bleed_marker_scrubber_still_fires():
    agent = _make_agent()
    fs = _fs(final_ai_content="Real answer.\n# Tools\n<tools> LEAKED SCHEMA")
    out, _, _ = await agent._finalize_and_return(fs)
    assert "LEAKED SCHEMA" not in out
    assert "Real answer." in out


@pytest.mark.asyncio
async def test_tool_response_scrubber_still_fires():
    agent = _make_agent()
    fs = _fs(final_ai_content=(
        "Answer.\n<tool_response>raw dump</tool_response>\nMore."))
    out, _, _ = await agent._finalize_and_return(fs)
    assert "raw dump" not in out


# ── revived finalize gates (2026-07-20) ─────────────────────────────────────
# Both gates silently died in the step-3 extraction: they read handle_chat
# locals via `locals().get(...)` from inside the extracted method, which can
# never see them. They now ride FinalizeState — these tests pin that the
# fields actually reach their consumers.

@pytest.mark.asyncio
async def test_plan_postcondition_gate_receives_the_plan():
    agent = _make_agent()
    plan = {
        "id": "root", "description": "answer with a number",
        "postconditions": ["the reply contains a number"],
        "subtasks": [],
    }
    fs = _fs(current_plan_json=plan)
    # The gate must at least ATTEMPT to load the plan — pin via the
    # TaskTree loader getting called (MagicMock context keeps every
    # downstream dependency inert, so a note may or may not append; the
    # dead-variable regression made even the attempt impossible).
    import ghost_agent.core.planning as planning_mod
    called = {}
    orig = planning_mod.TaskTree.load_from_json
    planning_mod.TaskTree.load_from_json = (
        lambda self, pj: called.setdefault("plan", pj) or orig(self, pj))
    try:
        await agent._finalize_and_return(fs)
    finally:
        planning_mod.TaskTree.load_from_json = orig
    assert called.get("plan") == plan


@pytest.mark.asyncio
async def test_wakeup_prefix_reaches_reference_counter():
    agent = _make_agent()
    sm = MagicMock()
    sm.enabled = True
    agent.context.self_model = sm
    fs = _fs(wakeup_prefix="=== WAKING UP ===\npast experience text")
    await agent._finalize_and_return(fs)
    assert sm.note_referenced_experiences.called
    _, kwargs = sm.note_referenced_experiences.call_args
    assert kwargs.get("prefix_text", "").startswith("=== WAKING UP ===")
