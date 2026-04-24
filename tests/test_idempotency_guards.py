"""Regression tests for the production tool-loop bug.

A real production log showed `update_profile(root.location=Athens, Greece)`
fired 9× in a single request before the model produced a response. The fix
is defense-in-depth across three layers — these tests pin each layer."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.bus import MemoryBus
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.tools.memory import (
    tool_remember, tool_update_profile, tool_learn_skill,
)


# ============================================================ bus dedup LRU


def test_bus_dedup_lru_signature_stable():
    bus = MemoryBus()
    sig_a = bus._fact_signature("update_profile", {"key": "loc", "value": "Athens"})
    sig_b = bus._fact_signature("update_profile", {"value": "Athens", "key": "loc"})
    assert sig_a == sig_b  # order-independent
    sig_c = bus._fact_signature("update_profile", {"key": "loc", "value": "Berlin"})
    assert sig_a != sig_c


@pytest.mark.asyncio
async def test_bus_publish_dedup_blocks_repeat():
    """Two identical publish_fact calls → second one short-circuits."""
    vec = MagicMock()
    vec.add = MagicMock()
    bus = MemoryBus(vector_memory=vec)

    fact = {"text": "User loves coffee"}
    rep1 = await bus.publish_fact("insert_fact", fact)
    rep2 = await bus.publish_fact("insert_fact", fact)

    assert rep1["vector"] == "ok"
    assert rep2["vector"] == "dedup"
    vec.add.assert_called_once()  # second call never reached the store


@pytest.mark.asyncio
async def test_bus_publish_dedup_distinct_facts_pass_through():
    vec = MagicMock()
    vec.add = MagicMock()
    bus = MemoryBus(vector_memory=vec)

    await bus.publish_fact("insert_fact", {"text": "A"})
    await bus.publish_fact("insert_fact", {"text": "B"})
    assert vec.add.call_count == 2


@pytest.mark.asyncio
async def test_bus_publish_dedup_lru_caps():
    """The dedup ledger has a hard cap so it can't leak unbounded memory."""
    bus = MemoryBus()
    for i in range(MemoryBus._DEDUP_LRU_MAX + 50):
        await bus.publish_fact("insert_fact", {"text": f"chunk_{i}"})
    assert len(bus._publish_lru) <= MemoryBus._DEDUP_LRU_MAX


# ============================================================== tool dedup


@pytest.mark.asyncio
async def test_tool_update_profile_short_circuits_when_value_unchanged():
    profile = MagicMock()
    profile.load = MagicMock(return_value={"root": {"location": "Athens, Greece"}})
    res = await tool_update_profile(
        category="root", key="location", value="Athens, Greece",
        profile_memory=profile,
    )
    assert "NOOP" in res
    profile.update.assert_not_called()  # never wrote


@pytest.mark.asyncio
async def test_tool_update_profile_writes_when_value_differs():
    profile = MagicMock()
    profile.load = MagicMock(return_value={"root": {"location": "Athens, Greece"}})
    res = await tool_update_profile(
        category="root", key="location", value="Berlin, Germany",
        profile_memory=profile,
    )
    assert "SUCCESS" in res
    profile.update.assert_called_once_with("root", "location", "Berlin, Germany")


@pytest.mark.asyncio
async def test_tool_remember_skips_existing_text():
    """vector.collection.get returns the existing id → tool returns NOOP."""
    vec = MagicMock()
    vec.collection = MagicMock()
    vec.collection.get = MagicMock(return_value={"ids": ["abcd1234"]})
    res = await tool_remember(text="The user owns a husky", memory_system=vec)
    assert "NOOP" in res
    vec.add.assert_not_called()


@pytest.mark.asyncio
async def test_tool_learn_skill_skips_duplicate_lesson(tmp_path):
    """Identical (task, mistake, solution) triplet must not be re-learned."""
    import json
    file_path = tmp_path / "skills_playbook.json"
    file_path.write_text(json.dumps([{
        "task": "parse json", "mistake": "missing key", "solution": "use .get()"
    }]))

    skill = MagicMock()
    skill.file_path = file_path
    skill.learn_lesson = MagicMock()

    res = await tool_learn_skill(
        task="parse json", mistake="missing key", solution="use .get()",
        skill_memory=skill,
    )
    assert "NOOP" in res
    skill.learn_lesson.assert_not_called()


# ===================================================== agent loop guard


def _make_agent_for_loop_test():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False

    ctx.llm_client = AsyncMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx.profile_memory.load = MagicMock(return_value={})  # so dedup falls through
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


@pytest.mark.asyncio
async def test_agent_loop_guard_blocks_duplicate_update_profile():
    """Simulate the production bug: model emits two identical update_profile
    tool calls in one turn. The second must be refused with an idempotency
    banner instead of dispatched to the tool."""
    agent = _make_agent_for_loop_test()

    duplicate_call = {
        "id": "call_a",
        "function": {
            "name": "update_profile",
            "arguments": '{"category":"root","key":"location","value":"Athens, Greece"}',
        },
    }
    second_call = {
        "id": "call_b",
        "function": {
            "name": "update_profile",
            "arguments": '{"category":"root","key":"location","value":"Athens, Greece"}',
        },
    }

    # Stub the LLM: first turn returns both tool calls, second turn returns
    # a final answer to break the loop.
    call_counter = {"n": 0}
    async def mock_chat(*a, **kw):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [duplicate_call, second_call],
            }}]}
        return {"choices": [{"message": {"content": "done", "tool_calls": []}}]}
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=mock_chat)

    update_profile_handler = AsyncMock(return_value="SUCCESS: Profile updated.")
    agent.available_tools = {"update_profile": update_profile_handler}

    body = {"messages": [{"role": "user", "content": "I live in Athens, Greece"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    # The handler must have been called EXACTLY ONCE despite two tool calls.
    assert update_profile_handler.call_count == 1
