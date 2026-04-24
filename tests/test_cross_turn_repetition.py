"""Cross-turn paragraph-repetition detector.

The intra-stream `_detect_thinking_loop` only sees one turn at a time.
When the model's stream is CUT OFF (thinking cap, strike reset) and
the NEXT turn begins with the same derivation, intra-stream probes
miss it — the 2026-04-17 09:07 log shows the solver opening
Turns 7/8/9/10 with near-identical "`''.split('\\n')` returns `['']`"
reasoning, each turn within its own budget.

The cross-turn guard fingerprints the first ~300 chars of each turn's
<think> block into a word set and compares against the prior turn via
Jaccard. Two consecutive hits ≥ 0.7 → abort.
"""

import json
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
    return GhostAgent(context)


# The exact paralysis opening from the 09:07 log, shortened to the
# word-set fingerprint the detector uses.
PARALYSIS_THOUGHT = (
    "The validator expects 0 lines but `out.stdout.strip().split('\\n')` "
    "returns `['']` which has length 1. Looking at the data: none of "
    "the groups have count 3, so expected should be empty. But the "
    "validator's strip().split('\\n') on empty string gives `['']` "
    "with length 1. So it always fails with `len(act) != len(exp)`."
)


def _think_reply(thought: str, tool_name: str = "noop"):
    """Build a chat-completion reply that carries `thought` inside
    <think> tags AND emits a tool call. The tool call is required
    because a pure text response breaks the turn loop at line 3066
    (`break` when `not tool_calls`), which would end the loop after
    a single turn and prevent the cross-turn guard from ever comparing
    turn N+1 against turn N. Cross-turn repetition is only meaningful
    when the loop is actually *looping*."""
    xml = (
        f'<tool_call>\n<function name="{tool_name}">\n'
        '<parameter name="x">1</parameter>\n'
        '</function>\n</tool_call>'
    )
    return {
        "choices": [{
            "message": {
                "content": f"<think>{thought}</think>\n{xml}",
                "tool_calls": [],
            }
        }]
    }


class TestCrossTurnRepetition:

    @pytest.mark.asyncio
    async def test_three_identical_openings_abort_attempt(self, agent):
        """Opening word-set repeats across three consecutive turns →
        the detector must force-stop with the cross-turn-loop marker."""
        turn_calls = {"n": 0}

        async def capture(payload, **kwargs):
            turn_calls["n"] += 1
            # Three identical openings → Jaccard 1.0 vs prior turn on
            # both comparison windows → 2 consecutive hits → abort.
            return _think_reply(PARALYSIS_THOUGHT)

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        noop = AsyncMock(return_value="ok")
        agent.available_tools = {"noop": noop}
        body = {"messages": [{"role": "user", "content": "solve the impossible validator task"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "noop"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # The abort fires on the THIRD call (2 consecutive Jaccard hits
        # against turns 1 and 2). Turn 4+ must never reach the LLM.
        assert turn_calls["n"] == 3, (
            f"Agent made {turn_calls['n']} LLM calls. Expected abort on turn 3."
        )
        assert "ATTEMPT_ABORTED_CROSS_TURN_LOOP" in final


    @pytest.mark.asyncio
    async def test_diverse_turns_do_not_trigger_abort(self, agent):
        """Distinct thoughts across turns must NOT fire the guard.
        This is the false-positive guard — normal task progress has
        LOW cross-turn lexical overlap."""
        diverse_thoughts = [
            "I need to read the orders CSV first and inspect its schema before writing solution.",
            "The inventory.csv contains item ids mapped to warehouses — that's the join key for later.",
            "Now writing a pandas-free aggregation using defaultdict grouped by region and month.",
            "Script produced expected output; wrapping up the response to the user now.",
        ]
        cursor = {"i": 0}

        async def capture(payload, **kwargs):
            t = diverse_thoughts[min(cursor["i"], len(diverse_thoughts) - 1)]
            cursor["i"] += 1
            return _think_reply(t)

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        noop = AsyncMock(return_value="ok")
        agent.available_tools = {"noop": noop}
        body = {"messages": [{"role": "user", "content": "analyze orders"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "noop"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # Must reach the end naturally (no abort marker).
        assert "ATTEMPT_ABORTED_CROSS_TURN_LOOP" not in final

    @pytest.mark.asyncio
    async def test_short_thoughts_do_not_trigger(self, agent):
        """Below the 8-word threshold, Jaccard is noisy and the
        guard skips. Prevents false-positive on short
        'Thinking.', 'Done.' style openings."""
        cursor = {"i": 0}

        async def capture(payload, **kwargs):
            cursor["i"] += 1
            # 4-word thought — below min word threshold.
            return _think_reply("Brief quick short thought.")

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        noop = AsyncMock(return_value="ok")
        agent.available_tools = {"noop": noop}
        body = {"messages": [{"role": "user", "content": "quick task"}]}

        class FakeBgTasks:
            def add_task(self, *a, **k): pass

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "noop"}}]):
            # Short opening stays below the 8-word threshold → guard
            # skips and the loop either exits naturally or hits the
            # turn cap. Either way: no cross-turn abort marker.
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        assert "ATTEMPT_ABORTED_CROSS_TURN_LOOP" not in final
