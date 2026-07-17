"""Grammar-constrained tool-call decoding (core/tool_grammar.py, 2026-07-17).

Pure-text tests — the grammar was validated against the LIVE llama-server
before wiring (per-request GBNF + pattern-type lazy triggers confirmed on
/completion and /v1/chat/completions; full 39-tool grammar compiled and
produced canonical calls; an invalid enum was coerced to a legal value).
These tests pin the generator's structure so refactors can't silently
change what the sampler is allowed to emit.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import _freeze_funcs
from ghost_agent.core.tool_grammar import (
    build_tool_grammar, grammar_payload_fields, grammar_enabled,
)
from ghost_agent.tools.registry import TOOL_DEFINITIONS


def _grammar_for(funcs):
    return build_tool_grammar(_freeze_funcs(funcs))


_DEMO = [
    {"name": "introspect", "description": "d",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string",
                    "enum": ["summary", "stats", "activity"]},
         "limit": {"type": "integer"},
         "query": {"type": "string"},
     }, "required": []}},
    {"name": "no_args_tool", "description": "d",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


class TestGrammarStructure:
    def test_canonical_framing_and_tool_alternation(self):
        g = _grammar_for(_DEMO)
        assert g.startswith("root ::= ws tool-call")
        assert '"<tool_call>"' in g and '"</tool_call>"' in g
        assert '"<function name=\\"introspect\\">"' in g
        assert '"<function name=\\"no_args_tool\\">"' in g

    def test_enum_param_constrained_and_tight(self):
        """The value rule lists exactly the legal literals with NO
        whitespace padding — a leading ws gave the sampler an escape
        hatch (live probe: forbidden enum → unbounded tab stream)."""
        g = _grammar_for(_DEMO)
        line = next(l for l in g.split("\n")
                    if l.startswith("v-introspect-action"))
        assert line == 'v-introspect-action ::= ("summary" | "stats" | "activity")'

    def test_integer_param_uses_int_val_and_string_uses_sval(self):
        g = _grammar_for(_DEMO)
        assert 'v-introspect-limit ::= int-val' in g
        # String params share the free-text rule; no per-param rule.
        assert "v-introspect-query" not in g
        assert '{sval}' not in g and "sval ::= svc*" in g

    def test_sval_exclusion_chain_blocks_close_tag_only(self):
        g = _grammar_for(_DEMO)
        # The 11-step prefix-exclusion chain for '</parameter>'.
        assert 'sv9 ::= [^r] | "r" [^>]' in g

    def test_scalar_rules_are_tight(self):
        g = _grammar_for(_DEMO)
        assert 'int-val ::= "-"? [0-9]+' in g
        assert 'bool-val ::= "true" | "false" | "True" | "False"' in g

    def test_full_registry_builds_and_covers_every_tool(self):
        g = _grammar_for([t["function"] for t in TOOL_DEFINITIONS])
        assert g
        for t in TOOL_DEFINITIONS:
            assert f'"<function name=\\"{t["function"]["name"]}\\">"' in g

    def test_empty_tools_yield_empty_grammar(self):
        assert build_tool_grammar(tuple()) == ""

    def test_memoized_on_frozen_signature(self):
        a = _grammar_for(_DEMO)
        b = _grammar_for(_DEMO)
        assert a is b  # lru_cache hit


class TestPayloadFields:
    def test_fields_shape(self, monkeypatch):
        monkeypatch.setenv("GHOST_TOOL_GRAMMAR", "1")
        fields = grammar_payload_fields(
            [{"function": f} for f in _DEMO])
        assert fields["grammar_lazy"] is True
        assert fields["grammar_triggers"] == [
            {"type": 2, "value": "\n<tool_call>"}]
        assert "tool-call" in fields["grammar"]

    def test_off_by_default_after_think_incident(self, monkeypatch):
        """Opt-in since req 9f1c3173: with thinking on, the model drafts
        literal <tool_call> inside reasoning, the trigger armed mid-think
        and generation died at '<tool' every turn. Default must stay OFF
        until the trigger uses think-aware PATTERN_FULL semantics."""
        monkeypatch.delenv("GHOST_TOOL_GRAMMAR", raising=False)
        assert not grammar_enabled()
        assert grammar_payload_fields(
            [{"function": f} for f in _DEMO]) == {}

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("GHOST_TOOL_GRAMMAR", "0")
        assert not grammar_enabled()
        assert grammar_payload_fields(
            [{"function": f} for f in _DEMO]) == {}

    def test_empty_and_garbage_are_safe(self, monkeypatch):
        monkeypatch.setenv("GHOST_TOOL_GRAMMAR", "1")
        assert grammar_payload_fields([]) == {}
        assert grammar_payload_fields([{"no": "function"}]) == {}


class TestWiring:
    def test_payload_attachment_gated_on_non_final_turns(self):
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "core" / "agent.py").read_text()
        idx = src.find("from .tool_grammar import grammar_payload_fields")
        assert idx != -1
        window = src[idx - 600:idx]
        assert "not is_final_generation and all_tools" in window

    def test_launcher_keeps_native_parsing_with_revert_notes(self):
        """The --no-native-tools experiment was REVERTED same-day: moving
        the schemas into the prompt broke the prefix-cache warmup (every
        conversation re-paid a ~20K-token prefill) and the grammar killed
        thinking turns. The launcher must NOT carry the flag until the
        warmup + PATTERN_FULL prerequisites (documented in its comment
        block) are built."""
        launcher = Path.home() / "Data" / "AI" / "bin" / "start-ghost-agent.sh"
        if not launcher.exists():
            pytest.skip("no launcher on this machine")
        src = launcher.read_text()
        assert "--no-native-tools \\" not in src
        assert "PATTERN_FULL" in src  # the re-attempt checklist stays
