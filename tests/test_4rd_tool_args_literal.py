"""§4RD — tool arguments reach the tool exactly as the model wrote them.

Req e2c7c7f4 (2026-10-24, a WebOS build at xhigh): a blanket
``html.unescape`` + ``\\"``→``"`` pass over every string argument of every
call turned the model's JS escape table ``'"': "&quot;"`` into
three bare double quotes (a syntax error), and REJECTED two repair edits as "old_string
and new_string are identical" — both sides decoded to the same text. ~1000 s
of a deadline-cut request went into diagnosing it.

World where each test fails: the blanket decode comes back (dispatch tests),
or the text-dialect parser stops decoding a quoted attribute (parser tests),
or a host path reaches the model in a syntax-check note.
"""
import asyncio
import json
from pathlib import Path

import pytest

from ghost_agent.core.agent import GhostAgent
from tests.test_turnloop_r1_fixes import _dispatch_agent, _ts, _call

ENTITY_ARGS = {
    "content": 'const ESC = {"&": "&amp;", "<": "&lt;", \'"\': "&quot;"};',
    "old_string": 'print(\\"hi\\")',
    "new_string": "echo it\\'s",
    "nested": ["&gt;", {"k": "&#39;"}],
    "n": 1,
}


async def _dispatched(args):
    agent = _dispatch_agent(); seen = {}

    async def t(**kw):
        seen.update(kw); return "ok"
    agent.available_tools = {"t": t}
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[_call("t", args)]))
    return seen


@pytest.mark.asyncio
async def test_entities_and_backslash_quotes_reach_the_tool_untouched():
    assert await _dispatched(ENTITY_ARGS) == ENTITY_ARGS


@pytest.mark.asyncio
async def test_the_e2c7c7f4_edit_pair_stays_distinct():
    # the model's repair: the raw `"""` back to the entity it meant
    args = {"operation": "edit", "path": "webos/apps.js",
            "old_string": '\'"\': """', "new_string": '\'"\': "&quot;"'}
    seen = await _dispatched(args)
    assert seen["old_string"] != seen["new_string"]
    assert seen["new_string"] == '\'"\': "&quot;"'


def _parse(content):
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"execute": (lambda **kw: None), "file_system": (lambda **kw: None)}
    tcs, _ui, reason = a._parse_assistant_tool_calls(content, {})
    assert len(tcs) == 1 and reason == "", (tcs, reason)
    args = tcs[0]["function"]["arguments"]
    return json.loads(args) if isinstance(args, str) else args


def test_text_dialect_element_body_is_literal():
    args = _parse('<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n'
                  '<parameter=path>a.html</parameter>\n'
                  '<parameter=content>&lt;p&gt;a &amp;&amp; b&lt;/p&gt;</parameter>\n'
                  '</function>\n</tool_call>')
    assert args["content"] == "&lt;p&gt;a &amp;&amp; b&lt;/p&gt;"


@pytest.mark.parametrize("call", [
    # Format 2: name/value attributes
    '<tool_call>\n<function name="execute">\n'
    '<parameter name="command" value="echo &quot;hi&quot; &amp;&amp; ls" />\n</function>\n</tool_call>',
    # Format 4: attribute tag
    '<tool_call>\n<function name="execute">\n'
    '<parameter command="echo &quot;hi&quot; &amp;&amp; ls" />\n</function>\n</tool_call>',
    # Format 5: direct attribute tag
    '<tool_call>\n<function name="execute">\n'
    '<command="echo &quot;hi&quot; &amp;&amp; ls">\n</function>\n</tool_call>',
    # attributes on the function tag itself
    '<tool_call>\n<function name="execute" command="echo &quot;hi&quot; &amp;&amp; ls">\n'
    '</function>\n</tool_call>',
])
def test_text_dialect_quoted_attribute_is_decoded(call):
    assert _parse(call)["command"] == 'echo "hi" && ls'


@pytest.mark.asyncio
async def test_syntax_note_names_the_file_not_the_host_path(tmp_path):
    from ghost_agent.tools import file_system as fs
    if not fs._find_node():
        pytest.skip("node not on PATH")
    d = tmp_path / "webos"; d.mkdir()
    f = d / "apps.js"
    f.write_text("const a = {'\"': \"\"\"};\n", encoding="utf-8")
    note = await fs._syntax_feedback(f, "webos/apps.js")
    assert "SYNTAX CHECK FAILED" in note
    assert str(tmp_path) not in note
    assert "webos/apps.js:1" in note


@pytest.mark.asyncio
async def test_bash_syntax_note_names_the_file_not_the_host_path(tmp_path):
    import shutil
    from ghost_agent.tools import file_system as fs
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")
    f = tmp_path / "run.sh"
    f.write_text("if true; then\n  echo hi\n", encoding="utf-8")
    note = await fs._syntax_feedback(f, "scripts/run.sh")
    assert "SYNTAX CHECK FAILED" in note
    assert str(tmp_path) not in note
    assert "scripts/run.sh" in note


@pytest.mark.asyncio
async def test_a_resolved_path_that_contains_the_given_one_is_replaced_whole(tmp_path, monkeypatch):
    # r1: /private/tmp/x/a.js contains /tmp/x/a.js — the shorter one first left "/private" behind
    from ghost_agent.tools import file_system as fs
    if not fs._find_node():
        pytest.skip("node not on PATH")
    f = tmp_path / "a.js"
    f.write_text("const = ;\n", encoding="utf-8")
    note = await fs._syntax_feedback(f, "a.js")
    assert "SYNTAX CHECK FAILED" in note
    assert "a.js:1" in note
    assert str(f) not in note and str(f.resolve()) not in note


def test_the_regex_fallback_unescapes_its_json_slice_quotes():
    # r1: the extreme fallback cuts values out of broken JSON source
    args = _parse('<tool_call>\n{"name": "file_system", "operation": "write", "path": "a.py", '
                  '"content": "print(\\"hi\\")"\n</tool_call>')
    assert args.get("content") == 'print("hi")', args
