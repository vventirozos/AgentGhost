"""Integration tests for MemoryBus wiring through agent.handle_chat
and the bus-aware paths in tools/memory.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.bus import MemoryBus
from ghost_agent.tools.memory import (
    tool_remember, tool_update_profile, tool_learn_skill,
)


# =============================================================== fixtures


def _make_agent_context():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False

    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    )

    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="Mocked Vector Memory")
    ctx.graph_memory = MagicMock()
    ctx.graph_memory.get_neighborhood = MagicMock(
        return_value=["- (Neo) -[HACKS]-> (Matrix)"]
    )
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.sandbox_dir = "/tmp/sandbox"
    return ctx


# ======================================================= agent → bus path


@pytest.mark.asyncio
async def test_agent_handle_chat_uses_memory_bus_for_hydration():
    """handle_chat must call hydrate_context on a MemoryBus and inject the
    fused output into the user message, replacing the old sequential block."""
    ctx = _make_agent_context()
    agent = GhostAgent(ctx)

    body = {
        "messages": [
            {"role": "user", "content": "Tell me about (Neo's) relationship with the Matrix!"}
        ],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())

    # Both subsystems must have been queried via the bus's asyncio.to_thread
    ctx.memory_system.search.assert_called()
    ctx.graph_memory.get_neighborhood.assert_called()
    # Bus extracts query terms before hitting graph
    graph_call = ctx.graph_memory.get_neighborhood.call_args
    words = graph_call.args[0]
    assert "neo's" in words
    assert "matrix" in words
    assert "user" not in words  # "user" is no longer auto-seeded (ego-graph bug)

    # The fused Markdown must land in the LLM payload's user message
    llm_payload = ctx.llm_client.chat_completion.call_args.args[0]
    msgs = llm_payload["messages"]
    user_msg = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in user_msg
    assert "- (Neo) -[HACKS]-> (Matrix)" in user_msg
    assert "MEMORY CONTEXT" in user_msg
    assert "Mocked Vector Memory" in user_msg


@pytest.mark.asyncio
async def test_agent_uses_explicit_bus_when_set_on_context():
    """When `context.memory_bus` is a real MemoryBus, the agent must use
    that one rather than building its own ad-hoc bus."""
    ctx = _make_agent_context()
    bus = MemoryBus(
        vector_memory=ctx.memory_system,
        graph_memory=ctx.graph_memory,
        skill_memory=ctx.skill_memory,
        profile_memory=ctx.profile_memory,
    )
    bus.hydrate_context = AsyncMock(return_value="### MEMORY CONTEXT:\nExplicit bus used\n\n")
    ctx.memory_bus = bus

    agent = GhostAgent(ctx)
    body = {"messages": [{"role": "user", "content": "Please remember this"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    bus.hydrate_context.assert_awaited()
    user_msg = next(
        m["content"] for m in reversed(ctx.llm_client.chat_completion.call_args.args[0]["messages"])
        if m["role"] == "user"
    )
    assert "Explicit bus used" in user_msg


@pytest.mark.asyncio
async def test_agent_skips_bus_on_fact_check_query():
    """should_fetch_memory bypass remains: fact-check queries skip the bus."""
    ctx = _make_agent_context()
    agent = GhostAgent(ctx)

    body = {
        "messages": [{"role": "user", "content": "Please fact-check this claim about X."}],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())

    # No memory subsystem should have been touched.
    ctx.memory_system.search.assert_not_called()
    ctx.graph_memory.get_neighborhood.assert_not_called()


# ========================================================= tools → bus path


@pytest.mark.asyncio
async def test_tool_remember_routes_through_bus_when_supplied():
    bus = MagicMock()
    bus.publish_fact = AsyncMock()

    res = await tool_remember(
        text="The user owns a husky named Max.",
        memory_bus=bus,
    )
    assert "stored" in res.lower()
    bus.publish_fact.assert_awaited_once()
    event_type, fact_data = bus.publish_fact.await_args.args
    assert event_type == "insert_fact"
    assert fact_data["text"] == "The user owns a husky named Max."
    assert fact_data["metadata"]["type"] == "manual"
    # No LLM client → no triplets extracted
    assert fact_data["triplets"] == []


@pytest.mark.asyncio
async def test_tool_remember_publishes_immediately_then_extracts_triplets_in_background():
    # New contract (functional hunt unit 3): the fact is published IMMEDIATELY
    # with empty triplets, and triplet extraction runs OFF the critical path in
    # a background task (the old inline-await hung the turn and lost the write —
    # see tests/test_insert_fact_hang.py). Triplets land in the graph, not in
    # the publish_fact payload.
    from ghost_agent.tools import memory as M
    bus = MagicMock()
    bus.publish_fact = AsyncMock()
    graph = MagicMock()
    graph.add_triplets = MagicMock(return_value=1)
    bus.graph = graph
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"graph_triplets": [{"subject": "user", "predicate": "OWNS", "object": "max"}]}'}}]
    })

    await tool_remember(
        text="user owns max",
        memory_bus=bus,
        llm_client=llm,
        model_name="test-model",
    )

    # Fact published immediately, without waiting on extraction.
    bus.publish_fact.assert_awaited_once()
    _, fact_data = bus.publish_fact.await_args.args
    assert fact_data["text"] == "user owns max"
    assert fact_data["triplets"] == []

    # Triplets are extracted + added to the graph off the critical path.
    tasks = list(M._GRAPH_EXTRACT_TASKS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
    graph.add_triplets.assert_called_once_with(
        [{"subject": "user", "predicate": "OWNS", "object": "max"}])


@pytest.mark.asyncio
async def test_tool_remember_legacy_path_unchanged():
    """Without a bus, the legacy direct path still calls memory_system.add."""
    mem = MagicMock()
    mem.add = MagicMock()
    res = await tool_remember(text="legacy fact", memory_system=mem)
    assert "stored" in res.lower()
    mem.add.assert_called_once()


@pytest.mark.asyncio
async def test_tool_update_profile_routes_through_bus():
    bus = MagicMock()
    bus.publish_fact = AsyncMock()

    res = await tool_update_profile(
        category="preferences",
        key="favorite language",
        value="Python",
        memory_bus=bus,
    )
    assert "SUCCESS" in res
    bus.publish_fact.assert_awaited_once()
    event_type, fact_data = bus.publish_fact.await_args.args
    assert event_type == "update_profile"
    assert fact_data["profile_update"] == {
        "category": "preferences",
        "key": "favorite language",
        "value": "Python",
    }
    assert fact_data["triplets"] == [
        {"subject": "user", "predicate": "HAS_FAVORITE_LANGUAGE", "object": "python"}
    ]
    assert fact_data["text"] == "User favorite language is Python"


@pytest.mark.asyncio
async def test_tool_update_profile_legacy_path_still_works():
    profile = MagicMock()
    res = await tool_update_profile(
        category="root", key="name", value="Vasilis",
        profile_memory=profile,
    )
    assert "SUCCESS" in res
    profile.update.assert_called_once_with("root", "name", "Vasilis")


@pytest.mark.asyncio
async def test_tool_learn_skill_routes_through_bus():
    bus = MagicMock()
    bus.publish_fact = AsyncMock()

    res = await tool_learn_skill(
        task="parse JSON",
        mistake="forgot to handle null",
        solution="check with .get()",
        memory_bus=bus,
    )
    assert "SUCCESS" in res
    bus.publish_fact.assert_awaited_once()
    event_type, fact_data = bus.publish_fact.await_args.args
    assert event_type == "learn_skill"
    assert fact_data["skill"] == {
        "task": "parse JSON",
        "mistake": "forgot to handle null",
        "solution": "check with .get()",
    }


@pytest.mark.asyncio
async def test_tool_learn_skill_legacy_path_still_works():
    skill = MagicMock()
    skill.learn_lesson = MagicMock()
    res = await tool_learn_skill(
        task="t", mistake="m", solution="s",
        skill_memory=skill,
    )
    assert "SUCCESS" in res
    skill.learn_lesson.assert_called_once()


# ============================================================= GhostContext


def test_ghost_context_has_memory_bus_field():
    ctx = GhostContext.__new__(GhostContext)
    GhostContext.__init__(ctx, args=MagicMock(), sandbox_dir=None, memory_dir=None, tor_proxy=None)
    assert hasattr(ctx, "memory_bus")
    assert ctx.memory_bus is None
