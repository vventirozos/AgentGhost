"""Reasoning-channel divergence guard — Layer 2 of the dual-channel emit fix.

Some models (Qwen-class reasoning variants in particular) emit BOTH
`reasoning_content` explicitly disclaiming tool use ("I can answer this
directly without using any tools") AND a structured `tool_call` in the
same response. The two channels are generated separately, so the model
can talk itself out of a tool call in prose while still emitting one in
the function-calling slot.

Pre-fix, the agent dispatched the contradicting tool_call and burned a
strike on it (see the trace where a "write me a checklist" prompt fired
a spurious `knowledge_base` call). Post-fix, the divergence guard scans
`reasoning_content` for an explicit no-tool disclaimer and, if found,
drops the tool_calls and re-runs the turn in final-generation mode.

The guard is intentionally narrow: it only fires when the reasoning
channel uses one of a small set of explicit phrases. A reasoning trace
that says "I'll use the recall tool" must NOT trigger a drop — that's
the negative-test case.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext


def _make_agent(*, native_tools=False, use_planning=False):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = use_planning
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.native_tools = native_tools

    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    )
    ctx.llm_client.worker_clients = None

    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="profile-data")
    ctx.profile_memory.load = MagicMock(return_value={})
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.graph_memory = MagicMock()
    ctx.graph_memory.get_neighborhood = MagicMock(return_value=[])
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)


def _all_content(payload):
    return "\n".join(m.get("content", "") for m in payload["messages"])


def _bogus_kb_call():
    return {
        "id": "tc-1",
        "function": {
            "name": "knowledge_base",
            "arguments": json.dumps({
                "action": "ingest_document",
                "filename": "migration_checklist",
            }),
        },
    }


# -----------------------------------------------------------------
# Positive: reasoning disclaims tools → tool_calls dropped, regen
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_disclaim_drops_tool_calls_and_regens():
    """The bug pattern from the production trace: the model thinks 'I
    can answer directly without using any tools' AND emits a
    knowledge_base tool_call. The guard must drop the call, set
    force_final_response, and re-run the turn so the model produces
    prose instead."""
    agent = _make_agent()

    kb_mock = AsyncMock(return_value="<should not be invoked>")
    agent.available_tools["knowledge_base"] = kb_mock

    captured_payloads = []

    async def mock_chat(payload, *a, **kw):
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": (
                            "The user is asking for a checklist. This is a "
                            "general knowledge question and I can answer this "
                            "directly without using any tools."
                        ),
                        "tool_calls": [_bogus_kb_call()],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Here is the checklist.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat
    body = {
        "messages": [{"role": "user", "content": "write me a checklist for a database migration"}],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())

    kb_mock.assert_not_called()
    assert len(captured_payloads) >= 2, (
        f"Expected at least one regen call after the disclaim drop; "
        f"got {len(captured_payloads)} call(s). The guard should `continue` "
        f"the turn loop rather than break."
    )

    second_prompt = _all_content(captured_payloads[1])
    assert "Final-generation turn" in second_prompt, (
        "After dropping disclaim-contradicting tool_calls, the regen "
        "turn must run in final-generation mode (slim 'answer directly' "
        "header)."
    )
    assert "DO NOT emit any <tool_call>" in second_prompt


@pytest.mark.asyncio
async def test_disclaim_phrase_no_tools_needed_also_caught():
    """The pattern set covers several phrasings. 'No tools are needed' is
    one of them — verify it triggers the same drop."""
    agent = _make_agent()
    kb_mock = AsyncMock(return_value="<should not be invoked>")
    agent.available_tools["knowledge_base"] = kb_mock

    captured = []

    async def mock_chat(payload, *a, **kw):
        captured.append(payload)
        if len(captured) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": "No tools are needed for this. I'll explain directly.",
                        "tool_calls": [_bogus_kb_call()],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat
    body = {"messages": [{"role": "user", "content": "explain transactions"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    kb_mock.assert_not_called()
    assert len(captured) >= 2


# -----------------------------------------------------------------
# Negative: reasoning supports tools → tool_calls dispatched
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_supportive_reasoning_does_not_drop():
    """Critical regression guard: if the reasoning trace says the model
    INTENDS to call a tool, the guard must NOT fire. False positives
    here would silently break every legitimate tool-calling turn."""
    agent = _make_agent()

    kb_mock = AsyncMock(return_value="LIBRARY CONTENTS (0 files):")
    agent.available_tools["knowledge_base"] = kb_mock

    captured = []

    async def mock_chat(payload, *a, **kw):
        captured.append(payload)
        if len(captured) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": (
                            "I should use the knowledge_base tool to list the "
                            "ingested documents and report back to the user."
                        ),
                        "tool_calls": [{
                            "id": "tc-1",
                            "function": {
                                "name": "knowledge_base",
                                "arguments": json.dumps({"action": "list_docs"}),
                            },
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Here's your library.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat
    body = {"messages": [{"role": "user", "content": "show me my saved docs"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    kb_mock.assert_called_once()


@pytest.mark.asyncio
async def test_no_reasoning_content_does_not_drop():
    """If reasoning_content is empty or absent, the guard has no signal
    to act on — tool_calls must dispatch normally."""
    agent = _make_agent()
    kb_mock = AsyncMock(return_value="LIBRARY CONTENTS (0 files):")
    agent.available_tools["knowledge_base"] = kb_mock

    captured = []

    async def mock_chat(payload, *a, **kw):
        captured.append(payload)
        if len(captured) == 1:
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "tc-1",
                            "function": {
                                "name": "knowledge_base",
                                "arguments": json.dumps({"action": "list_docs"}),
                            },
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat
    body = {"messages": [{"role": "user", "content": "list docs"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    kb_mock.assert_called_once()
