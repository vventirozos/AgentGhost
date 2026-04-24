"""abort_attempt: solver-facing escape-hatch tool.

When a solver has proven the current task is structurally unwinnable
(e.g. broken validator, contradictory spec, missing dependency), it
needs a way to STOP retrying instead of spiraling. The `abort_attempt`
tool returns a sentinel (`CHALLENGE_ABORTED_BY_SOLVER`) that:
  1. triggers `force_stop=True` in the agent turn loop, and
  2. causes `dream.synthetic_self_play` to skip remaining attempts.

Regression target: 2026-04-17 09:07 log — solver spent 10+ turns
re-deriving the same impossibility proof with no way to exit.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.tools.registry import get_available_tools, TOOL_DEFINITIONS


@pytest.fixture
def agent():
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


class TestToolRegistration:
    """abort_attempt must be registered in both the definitions list
    (LLM sees it) and the handlers dict (agent can execute it)."""

    def test_schema_in_tool_definitions(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert "abort_attempt" in names

    def test_schema_requires_reason(self):
        entry = next(
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] == "abort_attempt"
        )
        required = entry["function"]["parameters"]["required"]
        assert "reason" in required

    def test_schema_description_mentions_escape_hatch_semantics(self):
        entry = next(
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] == "abort_attempt"
        )
        desc = entry["function"]["description"].lower()
        # Must make clear this is NOT for normal retries.
        assert "only" in desc or "escape" in desc
        assert "unwinnable" in desc or "cannot be completed" in desc


class TestHandlerReturnsSentinel:
    @pytest.mark.asyncio
    async def test_handler_emits_canonical_sentinel(self):
        ctx = MagicMock()
        ctx.args = MagicMock()
        ctx.args.default_db = None
        ctx.llm_client = MagicMock()
        ctx.llm_client.image_gen_clients = None
        ctx.sandbox_dir = None
        ctx.memory_system = None
        tools = get_available_tools(ctx)
        assert "abort_attempt" in tools
        result = await tools["abort_attempt"](reason="validator line 54 references undefined `best_group_stats`")
        assert isinstance(result, str)
        # The sentinel is what agent.handle_chat greps for.
        assert "CHALLENGE_ABORTED_BY_SOLVER" in result
        # The reason must be surfaced so post-mortems can see WHY.
        assert "best_group_stats" in result


class TestAgentForceStopsOnAbortSentinel:
    """The agent must force_stop out of the turn loop as soon as
    `abort_attempt` returns its sentinel. Without this, the model
    could emit additional tool calls in the SAME turn and the loop
    would continue spinning."""

    @pytest.mark.asyncio
    async def test_abort_tool_exits_turn_loop_immediately(self, agent):
        # LLM would keep trying on turn 2+ if we didn't stop it; we
        # assert the stop by counting how many LLM calls happen.
        turn_calls = {"n": 0}

        async def capture(payload, **kwargs):
            turn_calls["n"] += 1
            # Turn 1: emit abort_attempt. Turn 2+: never reached.
            if turn_calls["n"] == 1:
                xml = (
                    '<tool_call>\n<function name="abort_attempt">\n'
                    '<parameter name="reason">validator has a structural bug that makes exp==[] unreachable</parameter>\n'
                    '</function>\n</tool_call>'
                )
                return {"choices": [{"message": {"content": xml, "tool_calls": []}}]}
            # This branch MUST NOT execute if the abort worked.
            return {"choices": [{"message": {"content": "should not see this", "tool_calls": []}}]}

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)

        abort_mock = AsyncMock(return_value=(
            "[CHALLENGE_ABORTED_BY_SOLVER] Reason: validator has a structural bug\n"
            "SYSTEM: The agent has declared this task unsolvable as specified."
        ))
        agent.available_tools = {"abort_attempt": abort_mock}

        body = {"messages": [{"role": "user", "content": "solve this impossible task"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "abort_attempt"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # Tool fired exactly once
        assert abort_mock.call_count == 1
        # LLM was called exactly once — no turn 2
        assert turn_calls["n"] == 1, (
            f"Agent issued {turn_calls['n']} LLM calls. The abort "
            "sentinel did NOT force-stop the turn loop."
        )
        # The sentinel propagates to final_ai_content so self-play can
        # detect it.
        assert "CHALLENGE_ABORTED_BY_SOLVER" in final
