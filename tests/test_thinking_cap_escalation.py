"""Thinking-cap escalation: second cap event in one attempt → force stop.

Background (2026-04-17 09:07 log): the solver hit the 12 000-char
thinking cap on Turn 6, got the "stop re-deriving the same paragraph"
SYSTEM ALERT, spiraled back into the same derivation, hit the cap
again on Turn 9, and STILL kept going. Before this fix there was no
escalation ladder — each cap event just incremented
`execution_failure_count` by 1, so a patient paralysis loop could
burn 8+ turns before the strike cap finally fired.

The fix: track `thinking_cap_events` per attempt. First event retains
the existing retry path (SYSTEM ALERT + retry). Second event in the
same attempt force-stops the request with a distinct
`ATTEMPT_ABORTED_THINKING_LOOP` sentinel so the outer caller
(dream.synthetic_self_play) can move on instead of retrying.
"""

import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext


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
    agent_instance = GhostAgent(context)
    # Tighten the cap so the test can trigger paralysis cheaply.
    # 600 chars of deterministic repeating content will trip both
    # the cap guard and the intra-stream n-gram detector — either
    # path increments thinking_cap_events via the shared handler.
    agent_instance.max_thinking_chars_override = 500
    return agent_instance


class TestCapEscalation:
    """Both cap paths (extended-cap overrun AND n-gram loop detection)
    converge on the same `thinking_loop_detected` branch. Once
    `thinking_cap_events >= 2` inside a single request, the agent
    must force-stop with the canonical abort marker."""

    def test_attribute_exists_on_agent(self, agent):
        # The override is what self-play sets on temp_agent; without
        # it the production agent has much larger caps.
        assert agent.max_thinking_chars_override == 500


    @pytest.mark.asyncio
    async def test_second_cap_event_force_stops(self, agent):
        """Two consecutive cap-triggering turns → stop after turn 2.

        Uses the conftest's stream adapter which echoes chat_completion
        results back through `stream_chat_completion`. Each turn
        returns 800 chars of reasoning (well past the 500-char cap)
        with enough repetition for the intra-stream guard to bite.
        """
        runaway = ("The validator expects 0 lines but split returns "
                   "a list with one empty string. ") * 20  # ~1200 chars, repetitive
        turn_calls = {"n": 0}

        async def capture(payload, **kwargs):
            turn_calls["n"] += 1
            xml = (
                '<tool_call>\n<function name="noop">\n'
                '<parameter name="x">1</parameter>\n</function>\n</tool_call>'
            )
            # Runaway thinking in every turn — the cap + loop detector
            # should fire each time.
            return {
                "choices": [{
                    "message": {
                        "content": f"<think>{runaway}</think>\n{xml}",
                        "tool_calls": [],
                    }
                }]
            }

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        noop = AsyncMock(return_value="ok")
        agent.available_tools = {"noop": noop}
        body = {"messages": [{"role": "user", "content": "thinking-cap escalation test"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log") as plog, \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "noop"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # Either path (pure-cap escalation OR cross-turn repetition)
        # is acceptable — both are valid responses to the same
        # paralysis pattern. What must NOT happen: the solver runs
        # for the full 40-turn budget.
        acceptable_markers = (
            "ATTEMPT_ABORTED_THINKING_LOOP",
            "ATTEMPT_ABORTED_CROSS_TURN_LOOP",
        )
        assert any(m in final for m in acceptable_markers), (
            f"Expected abort marker in final output. Got: {final[:300]!r}"
        )
        assert turn_calls["n"] < 10, (
            f"Agent ran {turn_calls['n']} turns — escalation should "
            "have stopped the loop within a few turns."
        )

    @pytest.mark.asyncio
    async def test_first_cap_event_alone_does_not_abort(self, agent):
        """Single cap event should still retry — not force-stop.
        Only the SECOND event in the same attempt escalates. Without
        this, a one-off over-think would kill entire requests."""
        # First turn: runaway. Second turn: short + valid tool call.
        runaway = ("Thinking thinking thinking around and around. ") * 20
        replies = iter([
            {
                "choices": [{
                    "message": {
                        "content": f"<think>{runaway}</think>",
                        "tool_calls": [],
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "content": "Done with the task now.",
                        "tool_calls": [],
                    }
                }]
            },
        ])

        async def capture(payload, **kwargs):
            try:
                return next(replies)
            except StopIteration:
                # Any follow-up turns: short no-op.
                return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        agent.available_tools = {}
        body = {"messages": [{"role": "user", "content": "single cap event test"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # One cap event alone must NOT produce the abort sentinel.
        assert "ATTEMPT_ABORTED_THINKING_LOOP" not in final, (
            "A single cap event should invoke retry, not force-stop."
        )
