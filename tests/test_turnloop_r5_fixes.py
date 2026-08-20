"""Turn-loop review, Round 5 (§ turn-loop R5, 2026-08-19) — behavioral pins.

R5's single MAJOR: `_STATIC_TOOL_NAMES` derived from TOOL_DEFINITIONS alone,
but two BUILT-INS (vision_analysis, image_generation) have their schemas
appended inside get_active_tool_definitions — so they were misclassified as
runtime tools and a degenerate empty-args native `vision_analysis {}` shadowed
a fully-specified XML call (an R4-introduced regression on the verify_ui
channel). Fixed via CONDITIONALLY_ADVERTISED_BUILTIN_NAMES in the registry,
drift-guarded here by DERIVING the set from the live function output.
"""

import json

from ghost_agent.core.agent import GhostAgent, _STATIC_TOOL_NAMES
from ghost_agent.tools.registry import (
    CONDITIONALLY_ADVERTISED_BUILTIN_NAMES, TOOL_DEFINITIONS,
    get_active_tool_definitions)
from unittest.mock import MagicMock


def _parser_agent(names=("vision_analysis", "recall", "web_search")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in names}
    return agent


def _names(tcs):
    return [(t.get("function") or {}).get("name") for t in tcs]


def _args0(tcs):
    x = tcs[0]["function"]["arguments"]
    return json.loads(x) if isinstance(x, str) else x


class TestConditionalBuiltinsAreStatic:
    def test_membership(self):
        assert "vision_analysis" in _STATIC_TOOL_NAMES
        assert "image_generation" in _STATIC_TOOL_NAMES

    def test_constant_matches_live_derivation(self, monkeypatch):
        # Ground truth: the names get_active_tool_definitions itself appends
        # beyond TOOL_DEFINITIONS, with every conditional gate enabled and the
        # runtime (macro/skill) sources disabled. If a future built-in is
        # advertised conditionally but not added to the constant, this fails.
        import ghost_agent.tools.registry as reg
        monkeypatch.setattr(reg, "register_composed_skills",
                            lambda *a, **k: None)
        ctx = MagicMock()
        ctx.sandbox_dir = None       # disables acquired-skill advertising
        ctx.memory_system = None
        ctx.llm_client.image_gen_clients = [object()]  # enables image_generation
        active = reg.get_active_tool_definitions(ctx)
        active_names = {(d.get("function") or {}).get("name") for d in active}
        static_names = {(d.get("function") or {}).get("name")
                        for d in TOOL_DEFINITIONS}
        derived = {n for n in (active_names - static_names) if n}
        assert derived == set(CONDITIONALLY_ADVERTISED_BUILTIN_NAMES), (
            f"conditionally-advertised built-ins drifted: derived={derived}")


class TestVisionEmptyArgsIsDegenerate:
    """The R5 regression pinned: a built-in with required params, natively
    called with empty args, must yield to a competing fully-specified XML
    call — same-tool and cross-tool."""

    def test_empty_vision_native_yields_to_rich_vision_xml(self):
        agent = _parser_agent()
        xml = ('<tool_call><function name="vision_analysis">'
               '<parameter name="action">verify_ui</parameter>'
               '<parameter name="target">/shot.png</parameter>'
               '<parameter name="prompt">is the ball in play?</parameter>'
               '</function></tool_call>')
        msg = {"content": xml, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "vision_analysis", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(xml, msg)
        assert _names(tcs) == ["vision_analysis"]
        args = _args0(tcs)
        assert args.get("action") == "verify_ui"
        assert args.get("target") == "/shot.png"

    def test_empty_vision_native_yields_to_rich_recall_xml(self):
        agent = _parser_agent()
        xml = ('<tool_call><function name="recall">'
               '<parameter name="query">real question</parameter>'
               '</function></tool_call>')
        msg = {"content": xml, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "vision_analysis", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(xml, msg)
        assert _names(tcs) == ["recall"]
