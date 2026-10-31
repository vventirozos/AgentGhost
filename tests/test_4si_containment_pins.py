"""§4SI r2: containment pins that lived only in the removed sub-agent tests
(test_subagent_containment, the 4lb delegate tests, the subagent half of the
replay union). The guards are live — coding-loop leaves and isolated replays
use them — and each pin was shown to fail on its mutant (a broken copy of the
guard passed the whole remaining suite before these existed)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from tests.test_4lb_prompt_context import _profile


def test_leaf_carries_no_owner_memory(tmp_path):
    """Fails where a leaf gets the owner's memory bus or a writable vector store."""
    from ghost_agent.core.coding_loop import build_leaf_context
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.memory.readonly import ReadOnlyVectorMemory
    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.args = SimpleNamespace(sandbox_dir=str(tmp_path))
    ctx.scratchpad = MagicMock(list_all=MagicMock(return_value="MARK-SCRATCH"))
    iso = build_leaf_context(ctx, leaf_id="l1")
    assert iso.owner_memory_isolated is True
    assert "MARK-SCRATCH" not in str(iso.scratchpad.list_all())
    bus = GhostAgent._get_memory_bus(SimpleNamespace(context=iso))
    assert isinstance(bus, MemoryBus) and bus.vector is None and bus.graph is None
    assert isinstance(iso.memory_system, ReadOnlyVectorMemory)
    assert iso.episodic_memory is None and iso.memory_bus is None


async def test_isolated_context_cannot_recall(tmp_path):
    """Fails where a contained context can `recall` the owner's store."""
    from ghost_agent.core.coding_loop import build_leaf_context
    from ghost_agent.tools.registry import get_available_tools
    from tests.helpers import make_context
    ctx = make_context()
    ctx.sandbox_dir = tmp_path
    ctx.profile_memory = _profile(tmp_path)
    ctx.memory_system = MagicMock()
    ctx.tor_proxy = None
    iso = build_leaf_context(ctx, leaf_id="l1")
    out = get_available_tools(iso)["recall"](query="user home")
    out = await out if asyncio.iscoroutine(out) else out
    assert "not available" in out
    ctx.memory_system.search_advanced.assert_not_called()
    # and the leaf's web_search is scrubbed of the owner's street (locality kept)
    import ghost_agent.tools.registry as R
    seen = {}

    async def _search(**kw):
        seen.update(kw)
        return "results"
    real = R.tool_search
    try:
        R.tool_search = _search
        await get_available_tools(iso)["web_search"](query="pizza Makedonias 83")
    finally:
        R.tool_search = real
    assert "Makedonias 83" not in seen["query"] and "Thrakomakedones" in seen["query"]


def _agent(ctx):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = ctx
    return a


def test_rebuild_renarrows_to_allowlist():
    """Fails where the rebuild after a tool-name miss restores the full surface."""
    a = _agent(SimpleNamespace(_subagent_allowed_tools=frozenset({"recall"})))
    a.disabled_tools = set()
    with patch("ghost_agent.core.agent.get_available_tools",
               return_value={"recall": 1, "jobs": 2, "manage_services": 3}):
        assert set(a._rebuild_available_tools()) == {"recall"}


def test_rebuild_redrops_disabled():
    """Fails where the rebuild brings a disabled tool back."""
    a = _agent(SimpleNamespace())
    a.disabled_tools = {"web_search"}
    with patch("ghost_agent.core.agent.get_available_tools",
               return_value={"recall": 1, "web_search": 2}):
        assert set(a._rebuild_available_tools()) == {"recall"}


async def test_leaf_turn_hides_and_blocks_everything_but_the_allowlist(monkeypatch):
    """Fails where a leaf can see or call a tool outside its allowlist."""
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core import agent as agent_mod
    seen = {}

    class FakeAgent:
        def __init__(self, iso):
            self.iso = iso
            self.available_tools = {"file_system": 1, "execute": 1, "browser": 1, "jobs": 1}
            self.disabled_tools = set(); self.max_turns_override = None

        async def handle_chat(self, body, background_tasks=None, request_id=None):
            seen["disabled"] = set(self.disabled_tools); seen["avail"] = set(self.available_tools)
            return "ok", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    base = SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=None, memory_system=object(),
                           skill_memory=object(), graph_memory=None, scratchpad=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object())
    await cl.run_leaf_turn(base, leaf_id="L", prompt="p", is_background=False, project_id="p1")
    assert seen["avail"] == {"file_system", "execute"}
    for t in ("browser", "jobs", "web_search", "manage_projects", "notify_operator"):
        assert t in seen["disabled"], t
    assert not {"file_system", "execute"} & seen["disabled"]


async def test_leaf_turn_fails_closed(monkeypatch):
    """Fails where a leaf runs open when its containment cannot be applied."""
    import sys
    from ghost_agent.core import coding_loop as cl
    from ghost_agent.core import agent as agent_mod
    ran = []

    class FakeAgent:
        def __init__(self, iso):
            self.available_tools = {"file_system": 1}; self.disabled_tools = set()
        async def handle_chat(self, *a, **k):
            ran.append(1); return "ok", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    base = SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=None, memory_system=object(),
                           skill_memory=object(), graph_memory=None, scratchpad=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object())
    with patch("ghost_agent.tools.registry.TOOL_DEFINITIONS", new=[{"function": {}}]):
        with pytest.raises(RuntimeError, match="containment"):
            await cl.run_leaf_turn(base, leaf_id="L", prompt="p", is_background=False, project_id="p1")
    assert not ran


def test_replay_list_keeps_the_old_subagent_denials():
    """Fails where the replay denylist loses a name the removed sub-agent list denied."""
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    assert {"manage_tasks", "manage_services", "manage_projects", "self_state", "update_profile"} <= REPLAY_FORBIDDEN_TOOLS


def test_a_zero_parameter_tool_is_still_detected_as_a_pure_trigger():
    """Fails where `_compute_pure_trigger_tools` stops seeing a `properties:{}`
    tool (none ships since dream_mode left; the gate still reads the set)."""
    from ghost_agent.core import agent as A
    defs = [{"type": "function", "function": {"name": "zz_noargs", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "execute", "parameters": {"type": "object",
                                                                              "properties": {"command": {}}}}}]
    with patch.object(A, "TOOL_DEFINITIONS", new=defs):
        assert A._compute_pure_trigger_tools() == frozenset({"zz_noargs"})
