"""The agent honors `thinking_budget_override` set on the instance.

The override was introduced so synthetic self-play could force a tighter
tier on its temp_agent (§4NM: that tier and caller are gone; the override
now admits only "tight" / "extended" and is used by coding leaves).

These tests lock in two contracts:
1. When `thinking_budget_override` is set on the agent, the classifier
   is bypassed and the override value wins.
2. Unknown/invalid override values do NOT win — the classifier still
   runs, so a typo doesn't silently fall back to TIGHT for a real task.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.prompts import (
    THINK_BUDGET_TIGHT,
    THINK_BUDGET_EXTENDED,
)


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
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


async def _run_single_turn_capture_payload(agent, user_content: str):
    """Drive one LLM call and return the payload that was sent to it.
    We make the LLM return a trivial text response so the turn loop
    terminates after a single request."""
    captured = {}

    async def capture(payload, **kwargs):
        # Record the payload only on the first call (subsequent turns
        # are irrelevant to the budget question).
        captured.setdefault("payload", payload)
        return {"choices": [{"message": {"content": "done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent.available_tools = {}
    body = {"messages": [{"role": "user", "content": user_content}]}

    class FakeBgTasks:
        def add_task(self, *a, **k): pass

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await agent.handle_chat(body, FakeBgTasks())
    return captured.get("payload")


def _combined_user_content(payload) -> str:
    """The think-budget guidance is spliced into the transient user-
    message header (see `tool_header_block` at agent.py ~1815). Join
    all user-message contents to search across the whole payload."""
    parts = []
    for m in payload["messages"]:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                parts.append(c)
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_override_tight_forces_tight(agent):
    """Sanity: a coding-heavy query that would auto-route to EXTENDED
    can be clamped back to TIGHT via the override."""
    agent.thinking_budget_override = "tight"
    payload = await _run_single_turn_capture_payload(
        agent,
        "Debug this traceback and optimize the query.",  # 2 EXTENDED keywords
    )
    body = _combined_user_content(payload)
    assert "Think as much as THIS step needs" in body  # TIGHT wording
    assert "HARD violations" not in body  # EXTENDED wording absent


@pytest.mark.asyncio
async def test_no_override_still_auto_classifies(agent):
    """Without an override, the classifier runs as before — a coding
    query with a strong keyword still lands in EXTENDED. This is the
    regression guard against the override silently short-circuiting
    normal routing."""
    # Explicitly make sure the attribute isn't set.
    if hasattr(agent, "thinking_budget_override"):
        delattr(agent, "thinking_budget_override")
    payload = await _run_single_turn_capture_payload(
        agent,
        "Debug this traceback and optimize the query.",
    )
    body = _combined_user_content(payload)
    assert "HARD violations" in body, (
        "Without an override, a debug+optimize query must reach EXTENDED."
    )


@pytest.mark.asyncio
async def test_bogus_override_is_ignored(agent):
    """A typo like `thinking_budget_override = "tite"` must NOT be
    treated as a valid override. The classifier runs and the real
    tier is used."""
    agent.thinking_budget_override = "tite"  # deliberate typo
    payload = await _run_single_turn_capture_payload(
        agent,
        "Debug this traceback and optimize the query.",
    )
    body = _combined_user_content(payload)
    # Two EXTENDED keywords → classifier would pick EXTENDED.
    assert "HARD violations" in body, (
        "Invalid override value should fall through to the classifier, not "
        "silently clamp to TIGHT."
    )
