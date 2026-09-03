"""§4EC — the usable-native gate pinned over its WHOLE input space (2026-09-02).

§4BY R3/R4 shipped `_native_call_usable` (agent.py, a closure inside
`_parse_assistant_tool_calls`): a native `tool_calls` entry wins over a rich
XML call in `content` only when it is USABLE — a known name AND either real
(non-empty dict) arguments or, for empty arguments, a pure-trigger tool or a
runtime-registered tool.  The §R battery found three of its arms deletable
with every pin green:

  * `if isinstance(raw, dict)` → False   (dict-typed native args never usable)
  * `elif raw is None`          → False   (None args never usable)
  * `if args is None`           → False   (list/scalar/garbage args reach the
                                           pure-trigger check and pass for it)

The existing R3 pins used static tools (`web_search`, `execute`), where the
fixed and broken worlds AGREE (both yield to XML).  This table lists the
expected verdict for every (argument shape × tool kind) cell as DATA — not a
re-derivation of the gate — and drives each cell through the real parser: a
usable native call is dispatched and the XML call is dropped; a degenerate one
yields to the XML call.

World where a row fails: any of the three mutants above (dict_full×static,
none×pure, list_type×pure / str_json_list×pure flip).
"""
import json
import itertools

import pytest

from ghost_agent.core.agent import GhostAgent, _PURE_TRIGGER_TOOLS, _STATIC_TOOL_NAMES

PURE = "dream_mode"          # properties:{} in TOOL_DEFINITIONS
STATIC = "web_search"        # static, takes arguments
RUNTIME = "zz_runtime_macro"  # registered at runtime, not in TOOL_DEFINITIONS
UNKNOWN = "no_such_tool_xyz"  # not exposed at all
XML_TOOL = "file_system"

XML = ('<tool_call>\n<function name="file_system">\n'
       '<parameter name="operation">read</parameter>\n'
       '<parameter name="path">a.py</parameter>\n</function>\n</tool_call>')

RAW = {
    "dict_full": {"query": "x"},
    "dict_empty": {},
    "none": None,
    "str_empty": "",
    "str_braces": "{}",
    "str_braces_padded": " {} ",     # (json.loads tolerates this either way)
    "str_ws_only": "   ",            # L13755 `raw.strip()`: whitespace-only is EMPTY,
    "str_null_padded": " null ",     #   not unparseable / not a JSON null
    "str_null": "null",
    "str_brackets": "[]",
    "str_json_dict": '{"query": "x"}',
    "str_json_list": "[1, 2]",
    "str_json_scalar": "42",
    "str_garbage": "{not json",
    "list_type": [1],
    "int_type": 7,
}
NONEMPTY = {"dict_full", "str_json_dict"}
EMPTY = {"dict_empty", "none", "str_empty", "str_braces", "str_braces_padded",
         "str_ws_only", "str_null_padded", "str_null", "str_brackets"}
DEGENERATE = {"str_json_list", "str_json_scalar", "str_garbage", "list_type", "int_type"}
assert NONEMPTY | EMPTY | DEGENERATE == set(RAW)

# Expected verdict per cell, as data.
EXPECTED = {}
for shape, kind in itertools.product(RAW, ("pure", "static", "runtime", "unknown")):
    if kind == "unknown":
        EXPECTED[(shape, kind)] = False
    elif shape in NONEMPTY:
        EXPECTED[(shape, kind)] = True
    elif shape in EMPTY:
        EXPECTED[(shape, kind)] = kind in ("pure", "runtime")
    else:
        EXPECTED[(shape, kind)] = False

NAME = {"pure": PURE, "static": STATIC, "runtime": RUNTIME, "unknown": UNKNOWN}


def test_fixture_premises():
    # The table only discriminates if the tool kinds are what they claim.
    assert PURE in _PURE_TRIGGER_TOOLS and PURE in _STATIC_TOOL_NAMES
    assert STATIC in _STATIC_TOOL_NAMES and STATIC not in _PURE_TRIGGER_TOOLS
    assert RUNTIME not in _STATIC_TOOL_NAMES and RUNTIME not in _PURE_TRIGGER_TOOLS
    assert UNKNOWN not in _STATIC_TOOL_NAMES


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {n: (lambda **kw: None)
                         for n in (PURE, STATIC, RUNTIME, XML_TOOL, "execute")}
    return a


def _drive(shape, kind):
    native = {"id": "call_native", "type": "function",
              "function": {"name": NAME[kind], "arguments": RAW[shape]}}
    tcs, ui, reason = _agent()._parse_assistant_tool_calls(XML, {"tool_calls": [native]})
    return [t["function"]["name"] for t in tcs]


@pytest.mark.parametrize("shape,kind", sorted(EXPECTED))
def test_gate_cell(shape, kind):
    names = _drive(shape, kind)
    if EXPECTED[(shape, kind)]:
        assert names == [NAME[kind]], (shape, kind, names)   # native wins, XML dropped
    else:
        assert names == [XML_TOOL], (shape, kind, names)      # degenerate → XML wins


def test_one_usable_call_in_a_mixed_native_batch_still_wins():
    """L13782: `_native_tcs_present = ... any(_native_call_usable(tc) ...)`.
    A batch of [degenerate, usable] must take the native path (any), not
    yield to XML (all) — the usable call would otherwise be dropped."""
    degenerate = {"id": "c1", "type": "function",
                  "function": {"name": STATIC, "arguments": "{}"}}
    usable = {"id": "c2", "type": "function",
              "function": {"name": STATIC, "arguments": '{"query": "x"}'}}
    tcs, ui, reason = _agent()._parse_assistant_tool_calls(
        XML, {"tool_calls": [degenerate, usable]})
    names = [t["function"]["name"] for t in tcs]
    assert STATIC in names and XML_TOOL not in names, names
