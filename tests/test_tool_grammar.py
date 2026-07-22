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


# ---------------------------------------------------------------------------
# 2026-07-22 fixes: sval trailing-partial-prefix termination, collision-free
# rule names, control-char-safe GBNF literals, OFF-path inertness.
#
# The termination property is checked against the ACTUAL generated grammar
# via a mini interpreter for the GBNF subset the generator emits (literals
# with escapes, char classes, groups, `| * ? +`, rule refs). llama-cpp-python
# is not installed in this venv; when it is, TestRealGBNFParser compiles the
# same grammars with the real parser.
# ---------------------------------------------------------------------------

import re as _re


def _gbnf_esc(body, k):
    """Decode one escape sequence starting after the backslash."""
    c = body[k]
    if c == "n":
        return "\n", k + 1
    if c == "r":
        return "\r", k + 1
    if c == "t":
        return "\t", k + 1
    if c == "x":
        return chr(int(body[k + 1:k + 3], 16)), k + 3
    return c, k + 1  # \" \\ \] \^ \- and friends


def _gbnf_tokenize(body):
    toks, k, n = [], 0, len(body)
    while k < n:
        c = body[k]
        if c in " \t":
            k += 1
        elif c == '"':
            k += 1
            buf = []
            while k < n and body[k] != '"':
                if body[k] == "\\":
                    ch, k = _gbnf_esc(body, k + 1)
                else:
                    ch, k = body[k], k + 1
                buf.append(ch)
            assert k < n, f"unterminated literal in: {body!r}"
            k += 1
            toks.append(("lit", "".join(buf)))
        elif c == "[":
            k += 1
            neg = body[k] == "^"
            if neg:
                k += 1
            chars, ranges = set(), []
            while body[k] != "]":
                if body[k] == "\\":
                    ch, k = _gbnf_esc(body, k + 1)
                else:
                    ch, k = body[k], k + 1
                if body[k] == "-" and body[k + 1] != "]":
                    k += 1
                    if body[k] == "\\":
                        hi, k = _gbnf_esc(body, k + 1)
                    else:
                        hi, k = body[k], k + 1
                    ranges.append((ch, hi))
                else:
                    chars.add(ch)
            k += 1
            toks.append(("class", (neg, frozenset(chars), tuple(ranges))))
        elif c in "()|*?+":
            toks.append((c, c))
            k += 1
        else:
            m = _re.match(r"[A-Za-z0-9_-]+", body[k:])
            assert m, f"bad GBNF token at: {body[k:]!r}"
            toks.append(("ref", m.group(0)))
            k += m.end()
    return toks


class _GbnfParser:
    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def alts(self):
        out = [self.seq()]
        while self.peek()[0] == "|":
            self.i += 1
            out.append(self.seq())
        return out

    def seq(self):
        items = []
        while self.peek()[0] not in (None, "|", ")"):
            items.append(self.item())
        return items

    def item(self):
        t, v = self.toks[self.i]
        self.i += 1
        if t == "(":
            node = ("group", self.alts())
            assert self.toks[self.i][0] == ")", "unbalanced parens"
            self.i += 1
        else:
            node = (t, v)
        if self.peek()[0] in ("*", "?", "+"):
            op = self.toks[self.i][0]
            self.i += 1
            node = ("rep", node, op)
        return node


def _parse_gbnf(text):
    """Parse full grammar text into {rule_name: alternatives-AST}."""
    rules = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "::=" not in line:
            continue
        name, body = line.split("::=", 1)
        name = name.strip()
        assert name not in rules, f"duplicate rule definition: {name}"
        p = _GbnfParser(_gbnf_tokenize(body))
        rules[name] = p.alts()
        assert p.peek()[0] is None, f"trailing tokens in rule {name}"
    return rules


def _match_node(node, s, i, rules):
    kind = node[0]
    if kind == "lit":
        v = node[1]
        return {i + len(v)} if s.startswith(v, i) else set()
    if kind == "class":
        neg, chars, ranges = node[1]
        if i < len(s):
            c = s[i]
            hit = c in chars or any(a <= c <= b for a, b in ranges)
            if hit != neg:
                return {i + 1}
        return set()
    if kind == "ref":
        return _match_alts(rules[node[1]], s, i, rules)
    if kind == "group":
        return _match_alts(node[1], s, i, rules)
    if kind == "rep":
        inner, op = node[1], node[2]
        if op == "?":
            return {i} | _match_node(inner, s, i, rules)
        results = {i} if op == "*" else set()
        frontier, seen = {i}, {i}
        while frontier:
            nxt = set()
            for j in frontier:
                for k in _match_node(inner, s, j, rules):
                    if k not in seen:
                        seen.add(k)
                        nxt.add(k)
            results |= nxt
            frontier = nxt
        return results
    raise AssertionError(f"unknown node kind {kind}")


def _match_seq(items, s, i, rules):
    positions = {i}
    for it in items:
        nxt = set()
        for p in positions:
            nxt |= _match_node(it, s, p, rules)
        positions = nxt
        if not positions:
            break
    return positions


def _match_alts(alts, s, i, rules):
    out = set()
    for seq in alts:
        out |= _match_seq(seq, s, i, rules)
    return out


def _accepts(rules, rule, s):
    """True iff `s` is FULLY derivable from `rule`."""
    return len(s) in _match_alts(rules[rule], s, 0, rules)


_CLOSE = "</parameter>"


def _param_text(value, pname="query"):
    return f'<parameter name="{pname}">{value}{_CLOSE}'


_COLLIDING = [
    {"name": "web_search", "description": "d",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
     }, "required": []}},
    {"name": "web", "description": "d",
     "parameters": {"type": "object", "properties": {
         "search_query": {"type": "string"},
     }, "required": []}},
]

_HOSTILE_ENUMS = [
    {"name": "t", "description": "d",
     "parameters": {"type": "object", "properties": {
         "mode": {"type": "string",
                  "enum": ['a"b', "c\nd", "e\\f", "g\th", "z\x01w"]},
     }, "required": []}},
]

_DUP_TOOLS = [
    {"name": "dup", "description": "d",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "integer"}}, "required": []}},
    {"name": "dup", "description": "d",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "integer"}}, "required": []}},
]


class TestSvalTermination:
    """Bug: the prefix-exclusion chain (`"<" sv0` demands a continuation
    char) could not ACCEPT a value ending in a proper prefix of
    '</parameter>', so the real close tag was absorbed into the value and
    generation ran away until max_tokens. Fixed by `sv-tail`."""

    def test_tail_rule_structure(self):
        g = _grammar_for(_DEMO)
        assert "sval ::= svc* sv-tail?" in g
        line = next(l for l in g.split("\n") if l.startswith("sv-tail"))
        # Every proper prefix of the close tag, lengths 1..11.
        expected = " | ".join(
            f'"{_CLOSE[:k]}"' for k in range(1, len(_CLOSE)))
        assert line == f"sv-tail ::= {expected}"

    def test_value_ending_in_lt_terminates(self):
        rules = _parse_gbnf(_grammar_for(_DEMO))
        assert _accepts(rules, "p-introspect-query", _param_text("foo<"))

    def test_value_ending_in_any_partial_prefix_terminates(self):
        rules = _parse_gbnf(_grammar_for(_DEMO))
        for k in range(1, len(_CLOSE)):
            value = "x" + _CLOSE[:k]
            assert _accepts(rules, "p-introspect-query",
                            _param_text(value)), (
                f"value ending in partial prefix {_CLOSE[:k]!r} "
                "cannot reach the close tag")

    def test_value_containing_close_tag_still_rejected(self):
        rules = _parse_gbnf(_grammar_for(_DEMO))
        assert not _accepts(rules, "p-introspect-query",
                            _param_text(f"a{_CLOSE}b"))

    def test_ordinary_values_still_accepted(self):
        rules = _parse_gbnf(_grammar_for(_DEMO))
        for value in ("", "hello", "a < b <= c", "<html><p>x</p></html>",
                      "if x < y:\n    pass", "</param", "< / p a r"):
            assert _accepts(rules, "p-introspect-query",
                            _param_text(value)), f"rejected {value!r}"

    def test_full_tool_call_with_trailing_lt_value_terminates(self):
        rules = _parse_gbnf(_grammar_for(_DEMO))
        call = ('<tool_call><function name="introspect">'
                '<parameter name="query">foo<</parameter>'
                "</function></tool_call>")
        assert _accepts(rules, "tool-call", call)


class TestHostileSchemas:
    """Bug: `_` -> `-` rule-name mapping collided across distinct
    (tool, param) pairs, and `_gbnf_string_literal` passed newlines and
    control chars through raw — both built 'successfully' and produced
    grammar llama-server would 400 on at request time."""

    def test_collision_pair_distinct_rule_names(self):
        g = _grammar_for(_COLLIDING)
        names = [l.split(" ::=")[0] for l in g.splitlines() if " ::= " in l]
        assert len(names) == len(set(names)), "duplicate rule definitions"
        # Both pairs map to base p-web-search-query; second gets -x2.
        assert "p-web-search-query" in names
        assert "p-web-search-query-x2" in names
        assert '"<parameter name=\\"query\\">"' in g
        assert '"<parameter name=\\"search_query\\">"' in g
        rules = _parse_gbnf(g)  # duplicate defs would assert inside
        assert _accepts(rules, "p-web-search-query", _param_text("a"))
        assert _accepts(rules, "p-web-search-query-x2",
                        _param_text("b", pname="search_query"))

    def test_duplicate_tool_names_get_distinct_rules(self):
        g = _grammar_for(_DUP_TOOLS)
        names = [l.split(" ::=")[0] for l in g.splitlines() if " ::= " in l]
        assert len(names) == len(set(names))
        assert "fn-dup" in names and "fn-dup-x2" in names
        alternation = next(l for l in g.splitlines()
                           if l.startswith("tool-call ::="))
        assert "fn-dup" in alternation and "fn-dup-x2" in alternation

    def test_no_duplicate_rules_across_full_registry(self):
        g = _grammar_for([t["function"] for t in TOOL_DEFINITIONS])
        names = [l.split(" ::=")[0] for l in g.splitlines() if " ::= " in l]
        assert len(names) == len(set(names))

    def test_hostile_enum_values_escaped_and_roundtrip(self):
        g = _grammar_for(_HOSTILE_ENUMS)
        # No raw control characters anywhere in the grammar text: a raw
        # newline splits a rule line, a raw \x01 is a parse error.
        for line in g.splitlines():
            assert not any(ord(c) < 32 for c in line), f"raw ctl in {line!r}"
        v_line = next(l for l in g.splitlines() if l.startswith("v-t-mode"))
        assert '\\n' in v_line and '\\\\' in v_line and '\\"' in v_line
        assert '\\t' in v_line and '\\x01' in v_line
        # Round-trip: escape-then-decode must be identity — the grammar
        # accepts exactly the raw enum values, nothing else.
        rules = _parse_gbnf(g)
        for value in ('a"b', "c\nd", "e\\f", "g\th", "z\x01w"):
            assert _accepts(rules, "v-t-mode", value), f"lost {value!r}"
        assert not _accepts(rules, "v-t-mode", "a")
        assert not _accepts(rules, "v-t-mode", "c d")

    def test_param_name_with_quote_and_newline_builds_valid_literal(self):
        funcs = [{"name": "odd", "description": "d",
                  "parameters": {"type": "object", "properties": {
                      'we"ird\nname': {"type": "string"}},
                   "required": []}}]
        g = _grammar_for(funcs)
        for line in g.splitlines():
            assert not any(ord(c) < 32 for c in line)
        rules = _parse_gbnf(g)  # must be structurally valid
        p_rule = next(n for n in rules if n.startswith("p-odd-"))
        raw = '<parameter name="we"ird\nname">v</parameter>'
        assert _accepts(rules, p_rule, raw)


class TestRealGBNFParser:
    def test_grammars_compile_with_llama_cpp(self):
        pytest.importorskip(
            "llama_cpp",
            reason="llama-cpp-python not installed in this venv; the "
                   "mini-interpreter + structural tests above still pin "
                   "GBNF validity")
        from llama_cpp import LlamaGrammar
        for funcs in (_DEMO, _COLLIDING, _HOSTILE_ENUMS, _DUP_TOOLS,
                      [t["function"] for t in TOOL_DEFINITIONS]):
            g = _grammar_for(funcs)
            LlamaGrammar.from_string(g, verbose=False)


class TestOffPathInert:
    def test_off_path_never_builds_grammar(self, monkeypatch):
        """With GHOST_TOOL_GRAMMAR unset the payload path must return {}
        WITHOUT ever invoking the grammar builder (a bomb would be
        swallowed by the best-effort except, so spy instead)."""
        monkeypatch.delenv("GHOST_TOOL_GRAMMAR", raising=False)
        import ghost_agent.core.tool_grammar as tg
        calls = []
        monkeypatch.setattr(tg, "build_tool_grammar",
                            lambda *a, **k: calls.append(a) or "")
        assert tg.grammar_payload_fields([{"function": _DEMO[0]}]) == {}
        assert calls == [], "grammar built despite OFF switch"
