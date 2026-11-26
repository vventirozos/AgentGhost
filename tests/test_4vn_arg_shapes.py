"""§4VN — a list or an object that arrived as a string.

A tool call written as text carries every parameter as text; for a parameter
the tool declares as an array or an object, the JSON the model wrote there
must reach the tool as the value. Everything the schema types as a string
stays exactly as written (the §4RD rule: no blanket decoding of arguments).
"""
from __future__ import annotations

import ast
import inspect
import json

import pytest

from ghost_agent.core import arg_shapes as S
from ghost_agent.tools.registry import TOOL_DEFINITIONS

ACTIONS = [{"action": "goto", "url": "https://example.com"},
           {"action": "evaluate", "js": "document.querySelector('h1') ? 1 : 'NO H1'"}]


def test_the_browser_call_that_failed_live_now_carries_its_list():
    """The exact arguments of the failing call (request probe-48, 2026-11-26)."""
    args = {"operation": "interact", "actions": json.dumps(ACTIONS, separators=(",", ":"))}
    assert S.decode_structured("browser", args, TOOL_DEFINITIONS) == ["actions"]
    assert args == {"operation": "interact", "actions": ACTIONS}


def test_a_native_call_is_left_as_it_is():
    args = {"operation": "interact", "actions": list(ACTIONS)}
    before = json.dumps(args, sort_keys=True)
    assert S.decode_structured("browser", args, TOOL_DEFINITIONS) == []
    assert json.dumps(args, sort_keys=True) == before


@pytest.mark.parametrize("tool, args", [
    # what the schema types as a STRING is never decoded, whatever it looks like
    ("file_system", {"operation": "write", "path": "a.json", "content": '[1, 2, 3]'}),
    ("file_system", {"operation": "write", "path": "a.json", "content": '{"a": "&quot;x&quot;"}'}),
    ("execute", {"command": '["not", "a", "list"]'}),
    # a string where a list is declared, but not that JSON: the tool gets it and says so
    ("browser", {"operation": "interact", "actions": "click the button"}),
    ("browser", {"operation": "interact", "actions": "[{'action': 'goto'}]"}),          # Python, not JSON
    ("browser", {"operation": "interact", "actions": '[{"action": "goto"'}),             # cut off
    ("browser", {"operation": "interact", "actions": '{"action": "goto"}'}),             # an object where a list is declared
    ("browser", {"operation": "interact", "actions": ""}),
    ("manage_projects", {"action": "create", "metadata": "[1, 2]"}),                     # a list where an object is declared
    ("no_such_tool", {"actions": "[1]"}),
])
def test_everything_else_reaches_the_tool_as_written(tool, args):
    before = json.dumps(args, sort_keys=True)
    assert S.decode_structured(tool, args, TOOL_DEFINITIONS) == []
    assert json.dumps(args, sort_keys=True) == before


def test_several_parameters_and_both_kinds():
    args = {"action": "create", "topics": ' ["a", "b"] ', "metadata": '{"k": 1}', "name": '["kept"]'}
    assert sorted(S.decode_structured("manage_projects", args, TOOL_DEFINITIONS)) == ["metadata", "topics"]
    assert args == {"action": "create", "topics": ["a", "b"], "metadata": {"k": 1}, "name": '["kept"]'}
    cmd = {"command": "ls", "args": '["-l", "-a"]'}
    assert S.decode_structured("execute", cmd, TOOL_DEFINITIONS) == ["args"] and cmd["args"] == ["-l", "-a"]


@pytest.mark.parametrize("args", [None, "x", ["a"], 3])
def test_arguments_that_are_not_an_object(args):
    assert S.decode_structured("browser", args, TOOL_DEFINITIONS) == []


def test_which_parameters_count():
    defs = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {
        "a": {"type": "array"}, "o": {"type": "object"}, "s": {"type": "string"}, "n": {"type": "integer"},
        "either": {"type": ["array", "string"]},                    # text is a legitimate value: left out
        "any": {"anyOf": [{"type": "array"}, {"type": "string"}]},
        "odd": "not a schema"}}}},
        {"function": {"name": "empty"}}, None, {}]
    assert S.structured_params("t", defs) == {"a": "array", "o": "object"}
    assert S.structured_params("empty", defs) == {} and S.structured_params("missing", defs) == {}
    args = {"a": "[1]", "o": "{}", "s": "[1]", "n": "[1]", "either": "[1]", "any": "[1]"}
    assert sorted(S.decode_structured("t", args, defs)) == ["a", "o"]
    assert args == {"a": [1], "o": {}, "s": "[1]", "n": "[1]", "either": "[1]", "any": "[1]"}
    assert S.decode_structured("t", {"a": "[1]"}, None) == []


def test_every_declared_list_or_object_in_the_real_tools_is_covered():
    seen = {(d["function"]["name"], p): k for d in TOOL_DEFINITIONS
            for p, k in S.structured_params(d["function"]["name"], TOOL_DEFINITIONS).items()}
    assert seen[("browser", "actions")] == "array" and seen[("execute", "args")] == "array"
    assert seen[("manage_projects", "metadata")] == "object" and len(seen) >= 15
    # and no tool declares its file content, command or query as a list (they must stay text)
    for (tool, param) in seen:
        assert param not in ("content", "command", "query", "replace_with", "old_string", "new_string"), (tool, param)


def test_the_decoded_arguments_are_stored_back_in_the_call():
    """What the tool runs with, what the history shows and what a later
    duplicate-call check hashes must be the same arguments."""
    from ghost_agent.core.agent import shape_call_arguments
    text = json.dumps({"operation": "interact", "actions": json.dumps(ACTIONS)})
    tool = {"id": "c1", "function": {"name": "browser", "arguments": text}}
    args = json.loads(text)
    assert shape_call_arguments(tool, "browser", args, TOOL_DEFINITIONS) == ["actions"]
    assert args["actions"] == ACTIONS
    assert json.loads(tool["function"]["arguments"]) == {"operation": "interact", "actions": ACTIONS}
    # nothing to decode: the call's own text is not rewritten
    same = {"id": "c2", "function": {"name": "browser", "arguments": '{"operation":"navigate",  "url":"x"}'}}
    assert shape_call_arguments(same, "browser", {"operation": "navigate", "url": "x"}, TOOL_DEFINITIONS) == []
    assert same["function"]["arguments"] == '{"operation":"navigate",  "url":"x"}'


def test_tools_defined_per_context_are_covered_too():
    """image_generation is not in the static list; its `reference_images` and
    `subjects` are lists — the same failure the decode exists for."""
    assert S.structured_params("image_generation", TOOL_DEFINITIONS) == {}      # why the static list is not enough
    active = list(TOOL_DEFINITIONS) + [{"type": "function", "function": {"name": "image_generation", "parameters": {
        "type": "object", "properties": {"prompt": {"type": "string"},
                                         "reference_images": {"type": "array", "items": {"type": "string"}},
                                         "subjects": {"type": "array", "items": {"type": "string"}}}}}}]
    args = {"prompt": '["not this"]', "reference_images": '["gen_1.png"]', "subjects": '["a", "b"]'}
    assert sorted(S.decode_structured("image_generation", args, active)) == ["reference_images", "subjects"]
    assert args == {"prompt": '["not this"]', "reference_images": ["gen_1.png"], "subjects": ["a", "b"]}


def test_the_agent_reads_the_active_definitions_and_falls_back_to_the_static_ones(monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.core import agent as A
    seen = {}

    def active(context, query=None, *, serve_tuned=True, disabled=None):
        seen.update(context=context, serve_tuned=serve_tuned)
        return [{"function": {"name": "only_here"}}]
    monkeypatch.setattr(A, "get_active_tool_definitions", active)
    me = SimpleNamespace(context="CTX")
    assert A.GhostAgent._arg_shape_definitions(me) == [{"function": {"name": "only_here"}}]
    assert seen == {"context": "CTX", "serve_tuned": False}             # not the prompt being built: no tuned-text draw
    monkeypatch.setattr(A, "get_active_tool_definitions", lambda *a, **k: 1 / 0)
    assert A.GhostAgent._arg_shape_definitions(me) is A.TOOL_DEFINITIONS
    monkeypatch.setattr(A, "get_active_tool_definitions", lambda *a, **k: [])
    assert A.GhostAgent._arg_shape_definitions(me) is A.TOOL_DEFINITIONS


def test_a_value_nested_past_any_parser_is_left_as_written():
    args = {"operation": "interact", "actions": "[" * 200000}
    assert S.decode_structured("browser", args, TOOL_DEFINITIONS) == [] and args["actions"] == "[" * 200000


def test_the_agent_applies_it_where_every_calls_arguments_are_prepared():
    from ghost_agent.core import agent as A
    calls = [c for c in ast.walk(ast.parse(inspect.getsource(A))) if isinstance(c, ast.Call)
             and ast.unparse(c.func) == "shape_call_arguments"]
    assert [[ast.unparse(a) for a in c.args] for c in calls] == [
        ["tool", "_cname", "t_args", "self._arg_shape_definitions()"]]
