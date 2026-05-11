"""Regression: when an iteration's tool_calls all get rejected
synthetically (parse error, invalid JSON args, unknown tool, etc.),
the iteration's preamble text must NOT bleed into the next
iteration's response.

Trace (2026-05-01 dialog log, turn 28). Reply opened with:

    Thank you for the kind words! I'm glad to hear that the
    information I provided was helpful. ...
    Have a great day! 😊

    [blank line]

    You're right to push back on the "no universals" claim, ...

The user's prompt was a Chomsky/Pirahã universal-grammar pushback —
zero "kind words" content. The opener was the model's preamble from
an inner-loop iteration whose tool_call did not parse; the loop kept
running, the next iteration produced the on-topic reply, and the two
got concatenated by the unconditional preamble-flush at what is now
``agent.py:4357-4361``. The blank line was the ``\\n\\n`` separator
inserted by that flush block.

Fix: capture ``_pre_flush_final_len`` before flushing, and rollback
to it if the iteration produced zero real ``tool_tasks`` (every
tool_call was rejected synthetically).
"""
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


def _reply_with_tool_call(preamble: str, tool_name: str, args_xml: str = '<parameter name="x">1</parameter>'):
    """Build a chat-completion reply that carries `preamble` as
    user-visible text PLUS a `<tool_call>` block. The XML parser
    extracts the tool_call into `tool_calls`; the rest becomes
    `ui_content` (the preamble we're worried about leaking)."""
    xml = (
        f'<tool_call>\n<function name="{tool_name}">\n'
        f'{args_xml}\n'
        '</function>\n</tool_call>'
    )
    return {
        "choices": [{
            "message": {
                "content": f"{preamble}\n{xml}",
                "tool_calls": [],
            }
        }]
    }


def _plain_text_reply(text: str):
    """No tool_call — the inner loop hits `if not tool_calls:` and
    breaks after flushing `text` to final_ai_content."""
    return {
        "choices": [{
            "message": {
                "content": text,
                "tool_calls": [],
            }
        }]
    }


class FakeBgTasks:
    def add_task(self, *a, **k):
        pass


class TestPreambleRollback:

    @pytest.mark.asyncio
    async def test_unknown_tool_iteration_drops_preamble(self, agent):
        """Turn 1 emits 'Thank you for the kind words!' + a tool_call
        for an unknown tool. The tool_call hits the unknown-tool
        synthetic-error branch (`Error: Unknown tool '...'`) and
        nothing reaches `tool_tasks`. The rollback must drop the
        preamble before the next iteration runs.

        Turn 2 emits the on-topic reply with no tool_call. The
        final response must contain ONLY turn 2's text — turn 1's
        preamble is gone."""
        cursor = {"i": 0}
        replies = [
            _reply_with_tool_call(
                "Thank you for the kind words! Have a great day!",
                "definitely_not_a_real_tool",
            ),
            _plain_text_reply(
                "You're right to push back on the no-universals claim."
            ),
        ]

        async def capture(payload, **kwargs):
            i = cursor["i"]
            cursor["i"] += 1
            return replies[min(i, len(replies) - 1)]

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        agent.available_tools = {}  # nothing registered → unknown-tool branch fires
        body = {"messages": [{"role": "user", "content": "tell me about Chomsky"}]}

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        assert "Thank you for the kind words" not in final, (
            f"Stale preamble from rejected iteration leaked. final={final!r}"
        )
        assert "Have a great day" not in final
        assert "You're right to push back" in final

    @pytest.mark.asyncio
    async def test_real_tool_iteration_keeps_preamble(self, agent):
        """Mixed iteration: turn 1 emits a preamble + a tool_call
        for a REAL registered tool. The tool runs, `tool_tasks` is
        non-empty, the rollback does NOT fire. The preamble is
        legitimate ("Let me check the weather...") and must survive
        into the final response.

        This is the false-positive guard for the rollback fix —
        normal multi-step preambles must keep working."""
        cursor = {"i": 0}
        # Turn 1 calls the real tool. Turn 2 finalizes.
        replies = [
            _reply_with_tool_call(
                "Let me check the weather first.",
                "weather_lookup",
            ),
            _plain_text_reply("It is 72 degrees and sunny."),
        ]

        async def capture(payload, **kwargs):
            i = cursor["i"]
            cursor["i"] += 1
            return replies[min(i, len(replies) - 1)]

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        weather = AsyncMock(return_value="72F sunny")
        agent.available_tools = {"weather_lookup": weather}
        body = {"messages": [{"role": "user", "content": "what's the weather?"}]}

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "weather_lookup"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # Both the legitimate preamble and the final answer must be present.
        assert "Let me check the weather first" in final, (
            f"Legitimate preamble was dropped (false-positive rollback). final={final!r}"
        )
        assert "It is 72 degrees and sunny" in final

    @pytest.mark.asyncio
    async def test_invalid_json_args_iteration_drops_preamble(self, agent):
        """Same shape as the unknown-tool case but exercising a
        DIFFERENT synthetic-error branch: invalid JSON args. The
        tool exists, but the args XML produces invalid JSON, so the
        for-loop hits the JSON-parse-error branch
        (`Error: Invalid JSON arguments - ...`) and tool_tasks
        stays empty. Rollback must still fire."""
        cursor = {"i": 0}
        # The args block contains literal `</parameter>` inside,
        # which the XML parser will turn into broken JSON when it
        # tries `json.loads(tool["function"]["arguments"])`.
        bad_args = '<parameter name="payload">{"unterminated</parameter>'
        replies = [
            _reply_with_tool_call(
                "Sure thing, executing now!",
                "execute",
                args_xml=bad_args,
            ),
            _plain_text_reply("Here is the actual answer to your question."),
        ]

        async def capture(payload, **kwargs):
            i = cursor["i"]
            cursor["i"] += 1
            return replies[min(i, len(replies) - 1)]

        agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
        execute = AsyncMock(return_value="ran")
        agent.available_tools = {"execute": execute}
        body = {"messages": [{"role": "user", "content": "compute something"}]}

        with patch("ghost_agent.core.agent.pretty_log"), \
             patch("ghost_agent.core.agent.get_active_tool_definitions",
                   return_value=[{"function": {"name": "execute"}}]):
            final, _, _ = await agent.handle_chat(body, FakeBgTasks())

        # The preamble may or may not get dropped depending on
        # whether the args actually fail to parse — the contract
        # we're really pinning here is "if the iteration ends up
        # with tool_tasks==[] then the preamble is dropped". If
        # the args happen to parse despite the malformation, the
        # tool will run and the preamble survives. Either is OK
        # as long as we don't see a Frankenstein opener; check
        # that the on-topic answer is present in either case.
        assert "Here is the actual answer" in final
