"""2026-07-22 — KV-pin stable-block instability fix (core/agent.py).

Under GHOST_PIN_TOOL_SCHEMAS=1 (prod) the "stable" injection pinned to the
first user message must be byte-identical across the turns of one request for
the upstream KV-cache to reuse it. Two per-turn-varying pieces used to sit in
that block and bust it: the skill playbook (its query is rebuilt from the
planner's per-turn tool + thought) and the final-generation header flip. Both
now ride the VOLATILE block under pin; the UNPINNED path is unchanged.

Proof chain: `test_compose_injection.py::test_pinned_first_message_identical_
across_turns` already proves that an identical stable string yields a
byte-identical pinned first message. These tests prove the stable string no
longer contains the per-turn playbook/directive under pin (and that the
unpinned composition is untouched).
"""
import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext


def _make_agent(*, playbook="", native_tools=True):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.native_tools = native_tools
    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hello there", "tool_calls": []}}]})
    ctx.llm_client.worker_clients = None
    ctx.llm_client.vision_clients = None
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
    ctx.skill_memory.get_playbook_context = MagicMock(return_value=playbook)
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)


def _last_nonstream_payload(agent):
    return agent.context.llm_client.chat_completion.await_args.args[0]


def _msgs(payload):
    return payload["messages"]


def _first_user(payload):
    return next(m for m in _msgs(payload) if m["role"] == "user")


@pytest.mark.asyncio
async def test_pinned_playbook_rides_volatile_not_the_pinned_block(monkeypatch):
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    agent = _make_agent(playbook="ALWAYS wrap paths in quotes.")
    body = {"messages": [{"role": "user", "content": "Explain the tradeoffs between two database migration strategies for me?"}], "model": "test", "stream": False}
    await agent.handle_chat(body, MagicMock())

    payload = _last_nonstream_payload(agent)
    pinned = _first_user(payload)["content"]
    whole = "\n".join(m.get("content", "") for m in _msgs(payload))

    # The pinned session_context is present...
    assert "<session_context>" in pinned
    # ...but the per-turn playbook is NOT inside it (it would bust the cache).
    assert "SKILL PLAYBOOK" not in pinned
    # The playbook still reaches the model — in the volatile block.
    assert "SKILL PLAYBOOK" in whole


@pytest.mark.asyncio
async def test_unpinned_composition_unchanged(monkeypatch):
    monkeypatch.delenv("GHOST_PIN_TOOL_SCHEMAS", raising=False)
    agent = _make_agent(playbook="ALWAYS wrap paths in quotes.")
    body = {"messages": [{"role": "user", "content": "Explain the tradeoffs between two database migration strategies for me?"}], "model": "test", "stream": False}
    await agent.handle_chat(body, MagicMock())

    payload = _last_nonstream_payload(agent)
    whole = "\n".join(m.get("content", "") for m in _msgs(payload))
    # Unpinned still surfaces the playbook (single-message injection); no
    # session_context pinning.
    assert "SKILL PLAYBOOK" in whole
    assert "<session_context>" not in whole


class TestSourceInvariants:
    def test_pinned_stable_injection_excludes_playbook(self):
        src = inspect.getsource(GhostAgent.handle_chat)
        # Under pin, _stable_injection is built WITHOUT fetched_playbook.
        assert 'if _pin_stable:' in src
        assert '_stable_injection = f"{tool_header_block}\\n\\n{active_persona}{fetched_context}{_continuity_tail}"' in src
        # The unpinned else-branch keeps playbook inline (unchanged layout).
        assert '{active_persona}{fetched_playbook}{fetched_context}' in src

    def test_pinned_final_gen_directive_routed_to_volatile(self):
        src = inspect.getsource(GhostAgent.handle_chat)
        # Slim header flip only on the UNPINNED path...
        assert "if _is_final_generation_for_schema and not _pin_stable:" in src
        # ...pinned final-gen emits the directive for the volatile block.
        assert "if _is_final_generation_for_schema and _pin_stable:" in src
        assert "_final_gen_directive" in src
