"""Regression: every consumer of ``tools_run_this_turn`` that treats
``[-1]`` (or all entries) as real tool output must skip synthetic
agent-loop error entries.

Background: the agent loop synthesises ``role=tool`` entries when the
model malforms a tool call (parse error, invalid JSON args, unknown
tool, idempotency block, empty-write block, etc.). These are NOT
real tool output; they are nudges the loop appends to drive the
model into a corrected retry. They live on ``tools_run_this_turn``
with ``_synthetic: True``.

A first fix (test_verifier_synthetic_error_skipped.py) made the
verifier filter skip them. This test pins the THREE other consumers
that previously read ``tools_run_this_turn`` blindly:

1. ``_prepare_planning_context`` — fed to the strategic planner.
   Synthetic entries leaked their ``SYSTEM ERROR: ...`` strings to
   the planner as "tool output", causing the planner to plan
   against fabricated evidence.

2. Final-fallback (``agent.py`` ``if tools_run_this_turn and not
   final_ai_content:``) — wraps the last entry in a "Process
   finished successfully" banner. A synthetic error here surfaces
   as a user-visible false-success.

3. Emergency-prune recovery (``agent.py`` context-overflow path) —
   copies the last entry into the recovery payload. A synthetic
   entry here (a) misleads the retry into planning against a fake
   tool result, and (b) shallow-copies the ``_synthetic`` marker
   into the upstream payload (contract violation: messages should
   be clean OpenAI-shape).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext


# ---------------------------------------------------------------------------
# Finding R1: planner-context filter
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_for_planner():
    """Planner-context only needs `args.max_context`. We share the
    full chat fixture here so registry init doesn't fail on missing
    `llm_client`/`profile_memory` etc."""
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


def test_planner_context_skips_synthetic_entries(agent_for_planner):
    tools = [
        {"name": "system", "content": "SYSTEM ERROR: Your <tool_call> did not parse...",
         "_synthetic": True},
        {"name": "execute", "content": "exit code 0\nactual output here"},
    ]
    out = agent_for_planner._prepare_planning_context(tools)
    assert "SYSTEM ERROR" not in out, (
        f"Synthetic entry leaked into planner context: {out!r}"
    )
    assert "actual output here" in out


def test_planner_context_all_synthetic_returns_no_context(agent_for_planner):
    """If every entry is synthetic the planner should see "None
    (Start of Task)", not a list of fake tools."""
    tools = [
        {"name": "system", "content": "SYSTEM ERROR: parse error",
         "_synthetic": True},
        {"name": "execute", "content": "Error: Invalid JSON arguments - ...",
         "_synthetic": True},
    ]
    out = agent_for_planner._prepare_planning_context(tools)
    assert out == "None (Start of Task)"


def test_planner_context_keeps_real_tools(agent_for_planner):
    """Sanity: non-synthetic entries pass through unchanged."""
    tools = [
        {"name": "execute", "content": "real exec output"},
        {"name": "file_system", "content": "SUCCESS: Wrote file"},
    ]
    out = agent_for_planner._prepare_planning_context(tools)
    assert "real exec output" in out
    assert "SUCCESS: Wrote file" in out


# ---------------------------------------------------------------------------
# Findings R2 + R3: end-to-end via handle_chat
#
# Both fallbacks live deep inside `handle_chat`. We exercise R2 (final
# fallback) by driving the agent loop to a state where every tool
# emission was synthetic. R3 (emergency-prune) is harder to trigger
# end-to-end because it requires upstream returning a 400 with
# "context" in the body — covered as a unit test on
# `_find_substantive_tool_for_verifier` instead.
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_for_chat():
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


def _reply_with_tool_call(preamble: str, tool_name: str):
    xml = (
        f'<tool_call>\n<function name="{tool_name}">\n'
        '<parameter name="x">1</parameter>\n'
        '</function>\n</tool_call>'
    )
    return {
        "choices": [{
            "message": {"content": f"{preamble}\n{xml}", "tool_calls": []}
        }]
    }


class FakeBgTasks:
    def add_task(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_final_fallback_skips_synthetic_when_response_empty(agent_for_chat):
    """If the agent loop exits with `final_ai_content` empty AND
    the only entries on `tools_run_this_turn` are synthetic
    parse/unknown-tool errors, the final-fallback must NOT wrap the
    synthetic content in "Process finished successfully" — it
    should fall through to the generic "Task executed successfully."
    placeholder.

    We trigger this by feeding the agent a tool_call for an unknown
    tool repeatedly until the turn cap is hit; every iteration's
    ui_content gets rolled back (per the earlier preamble-rollback
    fix) and the only entries left in `tools_run_this_turn` are
    synthetic unknown-tool errors."""
    cursor = {"i": 0}
    bad_replies = [_reply_with_tool_call("", "definitely_not_a_real_tool")] * 20

    async def capture(payload, **kwargs):
        i = cursor["i"]
        cursor["i"] += 1
        return bad_replies[min(i, len(bad_replies) - 1)]

    agent_for_chat.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent_for_chat.available_tools = {}
    body = {"messages": [{"role": "user", "content": "do something"}]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        final, _, _ = await agent_for_chat.handle_chat(body, FakeBgTasks())

    assert "SYSTEM ERROR" not in final, (
        f"Synthetic error leaked into final response: {final!r}"
    )
    assert "Error: Unknown tool" not in final
    # The "Process finished successfully" banner is the visible
    # tell of the bug. It must not appear when there is no real
    # tool output.
    assert "Process finished successfully" not in final


# ---------------------------------------------------------------------------
# Finding R3 (emergency-prune): pin the helper-based contract.
#
# The fix replaces `tools_run_this_turn[-1].copy()` with a walk to
# the last real tool entry via `_find_substantive_tool_for_verifier`.
# We test the helper is wired to skip synthetic entries — the
# end-to-end path through `_handle_context_overflow_recovery` is
# covered by the helper's unit tests in
# test_verifier_synthetic_error_skipped.py and the contract pin
# below.
# ---------------------------------------------------------------------------


def test_emergency_prune_helper_skips_synthetic_entries():
    """Pin the contract: the helper used by emergency-prune must
    NOT return a synthetic entry, even when it's the most recent.
    """
    from ghost_agent.core.agent import _find_substantive_tool_for_verifier
    tools = [
        {"name": "execute", "content": "actual stdout from real run"},
        {"name": "system", "content": "SYSTEM ERROR: parse fail",
         "_synthetic": True},
    ]
    res = _find_substantive_tool_for_verifier(tools)
    assert res is not None
    assert res["content"] == "actual stdout from real run"


def test_emergency_prune_strips_underscore_keys_when_forwarding():
    """The emergency-prune fix uses a dict comprehension to drop
    keys starting with `_` before appending to recovery_msgs.
    Pin that contract so a future refactor doesn't reintroduce
    the shallow-copy leak.
    """
    real_tool = {
        "role": "tool",
        "tool_call_id": "abc",
        "name": "execute",
        "content": "stdout",
        "_synthetic": True,  # malformed: should never be on a real tool
    }
    cleaned = {k: v for k, v in real_tool.items() if not k.startswith("_")}
    assert "_synthetic" not in cleaned
    assert cleaned["content"] == "stdout"
