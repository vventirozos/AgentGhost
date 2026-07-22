# src/ghost_agent/core/tool_grammar.py
"""Grammar-constrained tool-call decoding (2026-07-17).

Generates a llama.cpp GBNF grammar from the registry's tool schemas and
the request fields that attach it LAZILY: generation runs unconstrained
(thinking, prose) until the model emits ``<tool_call>``, then the grammar
snaps on and only syntactically valid calls — known tool names, known
parameter names, enum-valid values, numeric-valid integers — can decode.
The malformed-call class (truncated tags, invalid enum actions, stringly
integers) dies at the sampler; the agent's robust XML parser stays as the
consumer (and as the fallback for the shapes the grammar cannot cover).

Verified against the live llama-server build before wiring (2026-07-17):
``/completion`` and ``/v1/chat/completions`` both honour per-request
``grammar`` + ``grammar_lazy`` + ``grammar_triggers``; word-type triggers
demand pre-registered preserved tokens, but PATTERN-type (``type: 2``)
triggers work as-is, dormant through free text and firing on the literal
tag. With no ``tools`` field in the payload the server leaves the XML in
``content`` — pairing this module with ``--no-native-tools`` retires the
upstream native parser and its two known corruption shapes (merged
multi-tool args; value duplication) in one move.

Design constraints:

* One CANONICAL emission shape — ``<tool_call><function name="x">
  <parameter name="y">value</parameter>…</function></tool_call>`` — the
  exact shape the prompt teaches and the parser prefers. The grammar
  does not admit the ``<function=name>`` equals-variant; variants exist
  downstream for lenient PARSING, not for generation.
* String parameter values admit ANY text that does not contain the
  literal ``</parameter>`` close (prefix-exclusion chain, plus a
  trailing-partial-prefix tail so a value may END in e.g. a bare ``<``
  and the real close tag still parses) — file writes with HTML/code
  inside keep working.
* Required-ness and non-duplication of parameters are NOT enforced —
  GBNF can't express unordered required subsets without a combinatorial
  blow-up, and the tool layer already validates and teaches on missing
  args. The grammar's job is framing, names, enums, and numbers.
* Grammar text is memoized on the same frozen-schema signature the XML
  schema serialiser uses, so a stable tool loadout costs one build.

Kill switch: ``GHOST_TOOL_GRAMMAR=0`` disables attachment entirely.
"""

from __future__ import annotations

import functools
import os
import re
from typing import Optional

# The literal the lazy trigger fires on. Pattern-type (2) — the live
# build rejects word-type triggers for non-preserved tokens. Anchored on
# a preceding newline so a `<tool_call>` merely QUOTED inline in the
# thinking block (the historical think-strip incident shape) does not
# arm the grammar mid-reasoning; real calls start on their own line. A
# reply that OPENS with a call (no preceding newline) simply never
# triggers — the parser handles it exactly as today.
_TRIGGER_PATTERN = "\n<tool_call>"
_TRIGGER_TYPE_PATTERN = 2

_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9-]")


def _rule_name(prefix: str, name: str) -> str:
    """GBNF rule names allow [a-zA-Z0-9-]; tool/arg names carry
    underscores. The ``_`` → ``-`` mapping CAN collide across distinct
    (tool, param) pairs (``web_search``+``query`` and ``web``+
    ``search_query`` both map to ``p-web-search-query``), so this base
    name is only ever emitted through ``build_tool_grammar``'s per-build
    allocator, which suffixes ``-x2``, ``-x3``, … until the name is
    unique within the grammar."""
    return f"{prefix}-{_NAME_SAFE_RE.sub('-', str(name))}"


def _gbnf_string_literal(text: str) -> str:
    """A GBNF double-quoted literal for ``text``. Escapes backslash and
    quote, AND newlines/control characters — a raw newline inside a
    literal splits the rule across lines and a raw control char is
    rejected by llama-server's grammar parser at request time, so a
    hostile enum value or parameter name used to build 'successfully'
    here and 400 later. llama.cpp's parser accepts \\n \\r \\t and
    \\xHH escapes inside literals."""
    out = []
    for ch in str(text):
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


# "Any text not containing '</parameter>'" — the standard prefix-
# exclusion chain. Each step consumes the matched prefix plus one
# diverging character; the real close tag is consumed by the param rule
# instead. Shared by every string-typed parameter.
#
# ``sv-tail`` (2026-07-22 fix): the chain alone cannot ACCEPT a value
# that ENDS in a proper prefix of the close tag — ``"<" sv0`` demands a
# continuation character, so for a value ending in a bare ``<`` the only
# live parse absorbed the real ``</parameter>`` into the value text and
# generation could never legally terminate (runaway until max_tokens).
# ``sv-tail`` lets a value end in any proper prefix of the sentinel; the
# parse where the value stops there and the param rule's literal close
# tag matches next keeps termination reachable. Values still cannot
# CONTAIN the full close tag: mid-value ``<`` continues through the
# chain, and the tail only ever matches at the very end of ``sval``.
_SVAL_RULES = r"""
sval ::= svc* sv-tail?
svc ::= [^<] | "<" sv0
sv0 ::= [^/] | "/" sv1
sv1 ::= [^p] | "p" sv2
sv2 ::= [^a] | "a" sv3
sv3 ::= [^r] | "r" sv4
sv4 ::= [^a] | "a" sv5
sv5 ::= [^m] | "m" sv6
sv6 ::= [^e] | "e" sv7
sv7 ::= [^t] | "t" sv8
sv8 ::= [^e] | "e" sv9
sv9 ::= [^r] | "r" [^>]
sv-tail ::= "<" | "</" | "</p" | "</pa" | "</par" | "</para" | "</param" | "</parame" | "</paramet" | "</paramete" | "</parameter"
""".strip()

_SCALAR_RULES = """
ws ::= [ \\t\\n]*
int-val ::= "-"? [0-9]+
num-val ::= "-"? [0-9]+ ("." [0-9]+)?
bool-val ::= "true" | "false" | "True" | "False"
""".strip()


def _value_rule_for(p_type: str, p_enum: tuple, rule: str) -> Optional[str]:
    """Return the GBNF for one parameter's VALUE rule, or None to use the
    shared free-text ``sval``. Enum wins over declared type.

    Deliberately TIGHT — no whitespace padding around constrained
    values. A leading ``ws`` gave the sampler an escape hatch: when the
    model wanted an enum value the grammar forbids, it stalled emitting
    tabs forever instead of committing to a legal literal (observed on
    the live probe, action='bogus_action' → an unbounded tab stream).
    With a tight rule the only sampleable tokens at the value position
    are the legal literals' first characters, so the model is coerced to
    the nearest valid value and the call proceeds."""
    if p_enum:
        alts = " | ".join(_gbnf_string_literal(v) for v in p_enum)
        return f"{rule} ::= ({alts})"
    if p_type == "integer":
        return f"{rule} ::= int-val"
    if p_type == "number":
        return f"{rule} ::= num-val"
    if p_type == "boolean":
        return f"{rule} ::= bool-val"
    return None  # string / object / array / unknown → free text


@functools.lru_cache(maxsize=64)
def build_tool_grammar(frozen_funcs: tuple) -> str:
    """GBNF grammar text for a frozen tool-schema tuple (the exact shape
    ``agent._freeze_funcs`` produces: ``(name, description, prop_items,
    required)`` with ``prop_items = ((p_name, p_type, p_desc, p_enum),
    …)``). Descriptions and required-sets are ignored by design."""
    fn_rules, param_rules, value_rules = [], [], []
    fn_alts = []

    # Rule names go through this allocator so distinct (tool, param)
    # pairs NEVER share a rule name: the `_` → `-` mapping collides
    # (web_search+query vs web+search_query → p-web-search-query), and
    # a duplicated tool name in the schema would otherwise emit two
    # identical rule definitions — both shapes built "successfully" and
    # then 400'd at llama-server request time. Iteration order over the
    # frozen tuple is deterministic, so suffixes are stable per build.
    used_names: set = set()

    def _alloc(prefix: str, raw: str) -> str:
        base = _rule_name(prefix, raw)
        rule, n = base, 1
        while rule in used_names:
            n += 1
            rule = f"{base}-x{n}"
        used_names.add(rule)
        return rule

    for name, _desc, prop_items, _required in frozen_funcs:
        if not name:
            continue  # schema-less/garbage entry — nothing to constrain to
        fn_rule = _alloc("fn", name)
        fn_alts.append(fn_rule)
        p_alts = []
        for p_name, p_type, _p_desc, p_enum in prop_items:
            p_rule = _alloc("p", f"{name}-{p_name}")
            v_rule = _alloc("v", f"{name}-{p_name}")
            custom = _value_rule_for(p_type, p_enum, v_rule)
            if custom is None:
                used_names.discard(v_rule)  # unused — release the name
                v_rule = "sval"
            else:
                value_rules.append(custom)
            p_open = _gbnf_string_literal(f'<parameter name="{p_name}">')
            param_rules.append(
                f'{p_rule} ::= {p_open} '
                f"{v_rule} \"</parameter>\" ws"
            )
            p_alts.append(p_rule)
        params = f"({' | '.join(p_alts)})*" if p_alts else ""
        fn_open = _gbnf_string_literal(f'<function name="{name}">')
        fn_rules.append(
            f'{fn_rule} ::= {fn_open} ws '
            f"{params + ' ' if params else ''}\"</function>\""
        )
    if not fn_alts:
        return ""
    parts = [
        "root ::= ws tool-call (ws tool-call)* ws",
        f"tool-call ::= \"<tool_call>\" ws ({' | '.join(fn_alts)}) "
        "ws \"</tool_call>\"",
        *fn_rules,
        *param_rules,
        *value_rules,
        _SCALAR_RULES,
        _SVAL_RULES,
    ]
    return "\n".join(parts)


def grammar_enabled() -> bool:
    """OPT-IN (default off) since the 2026-07-17 live incident: with
    thinking enabled the model DRAFTS the literal ``<tool_call>`` inside
    its reasoning block, the pattern trigger armed mid-think, and
    generation died at ``<tool`` on every turn (req 9f1c3173 — hard-fail
    loop; the pre-wiring probes all ran with thinking disabled, which is
    exactly the blind spot). Re-enable with GHOST_TOOL_GRAMMAR=1 only
    after the trigger is upgraded to llama.cpp's think-aware
    PATTERN_FULL semantics and validated on a THINKING turn."""
    return (os.getenv("GHOST_TOOL_GRAMMAR", "0").strip().lower()
            in ("1", "true", "yes"))


def grammar_payload_fields(all_tools: list) -> dict:
    """The request fields that attach the lazy grammar for ``all_tools``
    (a list of ``{"function": {...}}`` dicts). ``{}`` when disabled,
    empty, or the schema shape is unusable — attachment is best-effort
    by contract and must never break payload construction."""
    if not grammar_enabled() or not all_tools:
        return {}
    try:
        # Local import: agent.py imports this module; _freeze_funcs lives
        # there next to the XML-schema serialiser that shares its key.
        from .agent import _freeze_funcs
        frozen = _freeze_funcs([t.get("function") or {} for t in all_tools])
        grammar = build_tool_grammar(frozen)
        if not grammar:
            return {}
        return {
            "grammar": grammar,
            "grammar_lazy": True,
            "grammar_triggers": [
                {"type": _TRIGGER_TYPE_PATTERN, "value": _TRIGGER_PATTERN},
            ],
        }
    except Exception:
        return {}
