"""Delegated sub-agent tool containment (bug-hunt 2026-07-14).

The old restriction filtered only `agent.available_tools` (the dispatch dict)
and stored `_subagent_allowed_tools` UNUSED. Two escapes resulted:
  1. the SCHEMA the model sees is filtered by `disabled_tools`, which the
     sub-agent never set → the model was shown delegate/jobs/manage_* and
     invited to call them;
  2. on any dispatch miss, `_rebuild_available_tools` healed the dict back to
     the FULL registry → the filter was undone the moment the model emitted a
     tool not in the narrowed dict.

The fix: run_subagent sets `disabled_tools` = advertised − allowlist (filters
schema AND blocks dispatch by name), narrows the dispatch dict, AND the
rebuild re-narrows to `context._subagent_allowed_tools`.
"""

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.subagent import (
    resolve_allowed_tools, build_subagent_context, run_subagent,
    DEFAULT_ALLOWED_TOOLS, FORBIDDEN_TOOLS,
)


class TestResolveAllowedTools:
    def test_default_is_allowlist_minus_forbidden(self):
        allowed = set(resolve_allowed_tools())
        assert allowed == set(DEFAULT_ALLOWED_TOOLS) - FORBIDDEN_TOOLS
        assert allowed.isdisjoint(FORBIDDEN_TOOLS)

    def test_forbidden_request_is_dropped(self):
        allowed = set(resolve_allowed_tools(["recall", "delegate", "manage_services"]))
        assert allowed == {"recall"}
        assert "delegate" not in allowed
        assert "manage_services" not in allowed

    def test_unknown_request_is_dropped(self):
        assert resolve_allowed_tools(["not_a_real_tool"]) == []


def _fake_context(tmp_path):
    ctx = MagicMock()
    ctx.sandbox_dir = str(tmp_path)
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.graph_memory = MagicMock()
    ctx.args = SimpleNamespace(model="m", perfect_it=True, smart_memory=0.9,
                               native_tools=False)
    ctx.llm_client = MagicMock()
    return ctx


class TestBuildContext:
    def test_sets_allowlist_and_isolates(self, tmp_path):
        ctx = _fake_context(tmp_path)
        iso = build_subagent_context(ctx, job_id="j1", allowed_tools=["recall"])
        assert iso._subagent_allowed_tools == frozenset({"recall"})
        # Isolation invariants.
        assert iso.workspace_model is None
        assert iso.trajectory_collector is None
        assert iso.episodic_memory is None
        assert iso.journal is None
        assert iso.memory_bus is None
        assert iso.args.perfect_it is False
        assert iso.args.smart_memory == 0.0
        # Read-only memory wrap.
        from ghost_agent.memory.readonly import ReadOnlyVectorMemory
        assert isinstance(iso.memory_system, ReadOnlyVectorMemory)
        assert iso.memory_system.is_read_only is True


class TestRebuildGuard:
    def test_rebuild_renarrows_for_subagent(self):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)  # bypass heavy __init__
        agent.context = SimpleNamespace(_subagent_allowed_tools=frozenset({"recall"}))
        with patch("ghost_agent.core.agent.get_available_tools",
                   return_value={"recall": 1, "delegate": 2, "manage_services": 3}):
            rebuilt = agent._rebuild_available_tools()
        assert set(rebuilt) == {"recall"}  # forbidden tools NOT healed back in

    def test_rebuild_unrestricted_for_normal_agent(self):
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace()  # no _subagent_allowed_tools
        full = {"recall": 1, "delegate": 2, "execute": 3}
        with patch("ghost_agent.core.agent.get_available_tools", return_value=full):
            rebuilt = agent._rebuild_available_tools()
        assert set(rebuilt) == set(full)  # a normal agent keeps the full set

    def test_rebuild_drops_disabled_for_dream(self):
        """Dream/self-play contains by SETTING disabled_tools (+ popping
        available_tools), NOT by _subagent_allowed_tools. A dispatch-miss
        rebuild must re-drop the disabled set, or the network-egress tools
        self-play disables (web_search/deep_research) heal back in (2026-07-22)."""
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace()  # no _subagent_allowed_tools (like dream)
        agent.disabled_tools = {"web_search", "deep_research", "delegate_to_swarm"}
        full = {"recall": 1, "web_search": 2, "deep_research": 3,
                "execute": 4, "delegate_to_swarm": 5}
        with patch("ghost_agent.core.agent.get_available_tools", return_value=full):
            rebuilt = agent._rebuild_available_tools()
        assert set(rebuilt) == {"recall", "execute"}  # disabled tools NOT healed back
        for banned in ("web_search", "deep_research", "delegate_to_swarm"):
            assert banned not in rebuilt

    def test_rebuild_drops_disabled_and_honors_allowlist_together(self):
        """When both guards are present the intersection wins."""
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace(_subagent_allowed_tools=frozenset({"recall", "web_search"}))
        agent.disabled_tools = {"web_search"}  # disabled overrides the allowlist
        full = {"recall": 1, "web_search": 2, "execute": 3}
        with patch("ghost_agent.core.agent.get_available_tools", return_value=full):
            rebuilt = agent._rebuild_available_tools()
        assert set(rebuilt) == {"recall"}  # allowlist ∩ (not disabled)


class TestRunSubagentRestriction:
    async def test_disabled_and_available_enforce_allowlist(self, tmp_path):
        """run_subagent must set disabled_tools = advertised − allowlist and
        narrow available_tools, so the schema hides forbidden tools and
        dispatch blocks them."""
        ctx = _fake_context(tmp_path)
        captured = {}

        async def fake_handle_chat(self, body, **kw):
            captured["disabled"] = set(self.disabled_tools)
            captured["available"] = set(self.available_tools)
            return ("done", 0, "sub-j1")

        with patch("ghost_agent.core.agent.GhostAgent.handle_chat",
                   new=fake_handle_chat):
            await run_subagent(ctx, job_id="j1", task="do it",
                               allowed_tools=["recall"], timeout_s=30)

        # available_tools narrowed to the allowlist (∩ registry).
        assert captured["available"] <= {"recall"}
        # Forbidden tools are disabled → filtered from the advertised schema
        # AND blocked at dispatch.
        for forbidden in ("delegate", "jobs", "delegate_to_swarm",
                          "manage_services", "manage_tasks", "update_profile"):
            assert forbidden in captured["disabled"], forbidden
        # The allowed tool is NOT disabled.
        assert "recall" not in captured["disabled"]
