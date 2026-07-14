"""warm_up_main_prefix — boot-time main-node prompt-cache warmup (2026-07-14).

The first request of a session prefills the ~30k-token rendered head
(system slot + native tool schemas) at ~450 tok/s ≈ 70s on the user's
critical path. The warmup sends the SAME byte-stable head at startup with
max_tokens=1 so the server's prefix cache absorbs it in the background.

Byte-exactness is the contract under test: the warmup must build its system
slot and tools through the same code paths a live request uses, target the
main node with is_background=True (yields to any live foreground request),
and never let a failure escape (best-effort, like warm_up_workers).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.prompts import SYSTEM_PROMPT

pytestmark = pytest.mark.asyncio


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


async def test_system_slot_is_byte_exact(agent):
    """The warmup's system message must equal the exact splice a live
    request performs: SYSTEM_PROMPT with {{PROFILE}} replaced by the
    profile string (same _RequestState.get_profile_str path)."""
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = False

    await agent.warm_up_main_prefix()

    payload = agent.context.llm_client.chat_completion.call_args[0][0]
    expected = SYSTEM_PROMPT.replace("{{PROFILE}}", "User Profile Data")
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == expected
    assert payload["max_tokens"] == 1
    assert payload["stream"] is False


async def test_targets_main_slot_politely(agent):
    """is_background=True with no off-main flags: lands on the main node
    but waits for any live foreground request to clear first."""
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = False

    await agent.warm_up_main_prefix()

    kwargs = agent.context.llm_client.chat_completion.call_args.kwargs
    assert kwargs.get("is_background") is True
    assert kwargs.get("task_label") == "main-prefix-warmup"
    assert kwargs.get("timeout") >= 120  # this IS the ~70s prefill


async def test_native_tools_attached_via_live_builder(agent):
    """With native tools on, the warmup ships the same tools list the live
    builder produces (plus the live payload's companion flags)."""
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = True

    canned = [{"type": "function",
               "function": {"name": "web_search", "parameters": {}}}]
    with patch("ghost_agent.tools.registry.get_active_tool_definitions",
               return_value=list(canned)) as mock_defs:
        await agent.warm_up_main_prefix()

    payload = agent.context.llm_client.chat_completion.call_args[0][0]
    assert payload["tools"] == canned
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    # Routed through the SAME builder a live request uses, neutral query.
    assert mock_defs.call_args.kwargs.get("query") == ""


async def test_disabled_tools_filtered(agent):
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = True
    agent.disabled_tools = {"delegate"}

    canned = [
        {"type": "function", "function": {"name": "web_search", "parameters": {}}},
        {"type": "function", "function": {"name": "delegate", "parameters": {}}},
    ]
    with patch("ghost_agent.tools.registry.get_active_tool_definitions",
               return_value=list(canned)):
        await agent.warm_up_main_prefix()

    payload = agent.context.llm_client.chat_completion.call_args[0][0]
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["web_search"]


async def test_perfect_it_splice_mirrored(agent):
    """The one config-driven system-slot mutation a live request applies
    must be mirrored, or the warmed prefix diverges at that byte."""
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = True
    agent.context.args.native_tools = False

    await agent.warm_up_main_prefix()

    payload = agent.context.llm_client.chat_completion.call_args[0][0]
    assert 'THE "PERFECT IT" PROTOCOL' in payload["messages"][0]["content"]


async def test_no_native_tools_means_no_tools_key(agent):
    agent.context.llm_client.chat_completion = AsyncMock(return_value={})
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = False

    await agent.warm_up_main_prefix()

    payload = agent.context.llm_client.chat_completion.call_args[0][0]
    assert "tools" not in payload


async def test_failure_never_escapes(agent):
    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=RuntimeError("upstream down"))
    agent.context.args.perfect_it = False
    agent.context.args.native_tools = False

    await agent.warm_up_main_prefix()  # must not raise


async def test_no_llm_client_is_clean_noop(agent):
    agent.context.llm_client = None
    await agent.warm_up_main_prefix()  # must not raise
