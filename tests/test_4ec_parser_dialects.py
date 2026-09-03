"""§4EC — the parser's healing / extraction DIALECTS, pinned as a class (2026-09-02).

The §R re-verification of §4BY ran a whole-function mutation battery over
`_parse_assistant_tool_calls` and found that every argument-extraction dialect
the parser heals — CDATA envelopes, `value=` attributes, bare tags, attribute
tags, direct-attribute tags, the bounds-aware repair, JSON-in-wrapper, the
`<tool>` wrapper, `<tool_call name=…>`, the text fallbacks — could be DELETED
with every pin file green.  Eighteen extraction sinks, zero driven tests.

The class-level pin: every dialect row below must dispatch the SAME
(name, arguments) as the canonical form.  The drift guard at the bottom counts
the parser's extraction sinks from its AST, so a new sink without a row here
fails loudly instead of joining the unpinned set.  (The count is a coverage
guard on THIS table, not a proxy for behaviour — behaviour is the rows.)

World where each row fails: the dialect's heal/extraction is deleted (the
exact mutants that survived: agent.py L13627/13628/13850/13888-13894/13909/
13925/13948/13953-13956/13962/13968/13982-14035) — the row then parses no
call, or the wrong arguments.
"""
import ast
import inspect
import json

import pytest

from ghost_agent.core.agent import GhostAgent

TOOLS = ("execute", "file_system", "deep_research", "deep_think",
         "image_generation", "vision_analysis", "web_search")


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {n: (lambda **kw: None) for n in TOOLS}
    return a


def _one(content):
    tcs, ui, reason = _agent()._parse_assistant_tool_calls(content, {})
    assert len(tcs) == 1, (tcs, reason)
    # R2 (fresh-eye): a dialect that dispatches the right call but leaves a
    # parse-failure REASON behind is still broken — the reason feeds the
    # model's recovery hint ("your output was cut off") and the operator's
    # "Upstream Truncation" warning. The `</tool>` heal survived without this.
    assert reason == "", (reason, tcs)
    f = tcs[0]["function"]
    args = f["arguments"]
    return f["name"], (json.loads(args) if isinstance(args, str) else args)


def _names(content):
    tcs, ui, reason = _agent()._parse_assistant_tool_calls(content, {})
    return [t["function"]["name"] for t in tcs], reason


CANON = ('<tool_call>\n<function name="execute">\n'
         '<parameter name="command">echo hi</parameter>\n'
         '</function>\n</tool_call>')
EXP = ("execute", {"command": "echo hi"})

ROWS = {
    "canonical": (CANON, EXP),
    # L13627/13628 — `<tool>` wrapper normalised to `<tool_call>`
    "tool_wrapper_xml": (
        '<tool>\n<function name="execute">\n<parameter name="command">echo hi'
        '</parameter>\n</function>\n</tool>', EXP),
    "tool_wrapper_json": (
        '<tool>{"name": "execute", "arguments": {"command": "echo hi"}}</tool>', EXP),
    # L13850 — `<tool_call name="x">` hallucination healed into a function tag
    "tool_call_name_attr": (
        '<tool_call name="execute">\n<parameter name="command">echo hi</parameter>'
        '\n</tool_call>', EXP),
    # L14073 (`else: t_data = extract_json_from_text(block_content)`) — pure
    # JSON inside the wrapper, no <function>. NOT L13886-13894: that block is
    # dead (it only ever sets func_match=None and L14073 recomputes the same
    # value) — recorded as F5, equivalent mutant, to be deleted.
    "json_in_wrapper": (
        '<tool_call>{"name": "execute", "arguments": {"command": "echo hi"}}'
        '</tool_call>', EXP),
    "json_root_key_is_tool": (
        '<tool_call>{"execute": {"command": "echo hi"}}</tool_call>', EXP),
    "json_openai_nested": (
        '<tool_call>{"function": {"name": "execute", "arguments": '
        '{"command": "echo hi"}}}</tool_call>', EXP),
    "json_stringified_args": (
        '<tool_call>{"name": "execute", "arguments": "{\\"command\\": \\"echo hi\\"}"}'
        '</tool_call>', EXP),
    "json_unparseable_string_args_dispatch_empty": (
        '<tool_call>{"name": "execute", "arguments": "not json at all"}</tool_call>',
        ("execute", {})),
    # L13909 — stray attributes on the <function> tag are arguments
    "function_tag_attrs": (
        '<tool_call>\n<function name="execute" command="echo hi">\n</function>\n'
        '</tool_call>', EXP),
    # L13943 — Format 1 sink. The repair pass (L14034) re-derives most values,
    # so only shapes the repair CANNOT reach discriminate: an EMPTY body (repair
    # skips `body_end <= body_start`) and an UNCLOSED parameter (no
    # `</parameter>` in range → repair leaves it alone).
    "format1_empty_body": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">'
        '</parameter>\n</function>\n</tool_call>', ("execute", {"command": ""})),
    "format1_unclosed_parameter": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">'
        'echo hi\n</function>\n</tool_call>', EXP),
    # L13925 — Format 0a CDATA envelope keeps a literal </parameter> inside
    "cdata_envelope": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">'
        '<![CDATA[echo "</parameter>" & done]]></parameter>\n</function>\n'
        '</tool_call>', ("execute", {"command": 'echo "</parameter>" & done'})),
    # L13948 — Format 2 value= attribute (self-closing)
    "value_attr": (
        '<tool_call>\n<function name="execute">\n<parameter name="command" '
        'value="echo hi" />\n</function>\n</tool_call>', EXP),
    # L13953-13956 — Format 3 bare tags
    "bare_tags": (
        '<tool_call>\n<function name="execute">\n<command>echo hi</command>\n'
        '</function>\n</tool_call>', EXP),
    # L13954 — Format 3 skips STRUCTURAL tags (`<parameters>` must not become
    # an argument), and L13955/13961/13967 — an explicit <parameter> is never
    # overwritten by a colliding bare / attribute / direct-attribute tag.
    "bare_tags_inside_parameters_wrapper": (
        '<tool_call>\n<function name="execute">\n<parameters>\n<command>echo hi'
        '</command>\n</parameters>\n</function>\n</tool_call>', EXP),
    "explicit_parameter_beats_bare_tag": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">echo hi'
        '</parameter>\n<command>other</command>\n</function>\n</tool_call>', EXP),
    "explicit_parameter_beats_attribute_tag": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">echo hi'
        '</parameter>\n<parameter command="other" />\n</function>\n</tool_call>', EXP),
    "explicit_parameter_beats_direct_attribute": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">echo hi'
        '</parameter>\n<command="other">\n</function>\n</tool_call>', EXP),
    # L13962 — Format 4 attribute tag
    "attribute_tag": (
        '<tool_call>\n<function name="execute">\n<parameter command="echo hi" />'
        '\n</function>\n</tool_call>', EXP),
    # L13968 — Format 5 direct attribute tag
    "direct_attribute_tag": (
        '<tool_call>\n<function name="execute">\n<command="echo hi">\n'
        '</function>\n</tool_call>', EXP),
    # L13982-14033 — Format 5b bounds-aware repair: literal </parameter> in a body
    "repair_literal_close_tag": (
        '<tool_call>\n<function name="execute">\n<parameter name="command">'
        'echo "</parameter>" done</parameter>\n</function>\n</tool_call>',
        ("execute", {"command": 'echo "</parameter>" done'})),
    # Format 6 — JSON inside <function>
    "json_in_function": (
        '<tool_call>\n<function name="execute">{"command": "echo hi"}</function>\n'
        '</tool_call>', EXP),
    # Format 7 — single-argument text fallbacks (args_val sinks L14060-14065)
    "text_fallback_deep_research": (
        '<tool_call>\n<function name="deep_research">quantum computing</function>'
        '\n</tool_call>', ("deep_research", {"query": "quantum computing"})),
    "text_fallback_deep_think": (
        '<tool_call>\n<function name="deep_think">why?</function>\n</tool_call>',
        ("deep_think", {"query": "why?"})),
    "text_fallback_image_generation": (
        '<tool_call>\n<function name="image_generation">a red fox</function>\n'
        '</tool_call>', ("image_generation", {"prompt": "a red fox"})),
    "text_fallback_vision_analysis": (
        '<tool_call>\n<function name="vision_analysis">photo.png</function>\n'
        '</tool_call>', ("vision_analysis",
                         {"target": "photo.png", "action": "describe_picture"})),
    # L14057 — Format 7 refuses a body that starts with `<` or `{` (a comment
    # is not a query); the call still dispatches, with no argument.
    "text_fallback_refuses_tag_led_body": (
        '<tool_call>\n<function name="deep_research"><!-- x --></function>\n'
        '</tool_call>', ("deep_research", {})),
    # Un-nesting hallucinated <arguments>/<parameters>/<kwargs>/<args> wrappers
    "unnest_parameters_wrapper_str": (
        '<tool_call>\n<function name="execute">\n<parameter name="parameters">'
        '{"command": "echo hi"}</parameter>\n</function>\n</tool_call>', EXP),
    "unnest_kwargs_wrapper_str": (
        '<tool_call>\n<function name="execute">\n<parameter name="kwargs">'
        '{"command": "echo hi"}</parameter>\n</function>\n</tool_call>', EXP),
    "unnest_args_wrapper_str": (
        '<tool_call>\n<function name="execute">\n<parameter name="args">'
        '{"command": "echo hi"}</parameter>\n</function>\n</tool_call>', EXP),
    "unnest_arguments_wrapper_str": (
        '<tool_call>\n<function name="execute">\n<parameter name="arguments">'
        '{"command": "echo hi"}</parameter>\n</function>\n</tool_call>', EXP),
    "unnest_arguments_wrapper_dict": (
        '<tool_call>{"name": "execute", "arguments": {"arguments": '
        '{"command": "echo hi"}}}</tool_call>', EXP),
    # Fallback 1 — hallucinated <tool_name> tags (args_fallback sinks L14140-14148)
    "tool_name_tag_bare_args": (
        '<tool_call>\n<execute>\n<command>echo hi</command>\n</execute>\n'
        '</tool_call>', EXP),
    "tool_name_tag_text_deep_research": (
        '<tool_call>\n<deep_research>quantum computing</deep_research>\n'
        '</tool_call>', ("deep_research", {"query": "quantum computing"})),
    "tool_name_tag_text_deep_think": (
        '<tool_call>\n<deep_think>why?</deep_think>\n</tool_call>',
        ("deep_think", {"query": "why?"})),
    "tool_name_tag_text_image_generation": (
        '<tool_call>\n<image_generation>a red fox</image_generation>\n'
        '</tool_call>', ("image_generation", {"prompt": "a red fox"})),
    "tool_name_tag_text_vision_analysis": (
        '<tool_call>\n<vision_analysis>photo.png</vision_analysis>\n</tool_call>',
        ("vision_analysis", {"target": "photo.png", "action": "describe_picture"})),
    # Raw-JSON path (no wrapper at all, L14400-14415): stringified arguments
    # pass through as the JSON string they are — never double-encoded.
    "raw_json_stringified_args": (
        '{"name": "execute", "arguments": "{\\"command\\": \\"echo hi\\"}"}', EXP),
    # CDATA token inside a JSON LIST argument: the un-masking recurses into lists
    "cdata_inside_json_list_argument": (
        '<tool_call>{"name": "execute", "arguments": {"argv": ["<![CDATA[a<b>c]]>", "x", 2], "n": 1}}'
        '</tool_call>', ("execute", {"argv": ["a<b>c", "x", 2], "n": 1})),
    # Extreme regex fallback (L14160-14245): a <name>/<tool_name> tag or a
    # `"name":` literal plus loose `"key": "value"` pairs — five `args_dict`
    # sinks the first version of this table did not count.
    "extreme_name_tag_file_system_read": (
        '<tool_call>\n<name>file_system</name> "operation": "read", "path": "a.py"'
        '\n</tool_call>', ("file_system", {"operation": "read", "path": "a.py"})),
    "extreme_file_system_replace_with_content": (
        '<tool_call>\n<name>file_system</name> "operation": "replace", '
        '"path": "a.py", "content": "old", "replace_with": "new"\n</tool_call>',
        ("file_system", {"operation": "replace", "path": "a.py",
                         "content": "old", "replace_with": "new"})),
    "extreme_execute_filename_content": (
        '<tool_call>{"filename": "a.py", "content": "print(1)"}</tool_call>',
        ("execute", {"filename": "a.py", "content": "print(1)"})),
    "extreme_vision_target_tag": (
        '<tool_call>\n<name>vision_analysis</name>\n<target>photo.png</target>\n'
        '</tool_call>', ("vision_analysis",
                         {"target": "photo.png", "action": "describe_picture"})),
    # L14166-14170: the `"name": "…"` literal form, and a path that merely CONTAINS
    # the word "filename" must not flip a named tool to `execute`.
    "extreme_name_literal_file_system_read": (
        '<tool_call>\n"name": "file_system", "operation": "read", "path": "a.py"\n'
        '</tool_call>', ("file_system", {"operation": "read", "path": "a.py"})),
    "extreme_named_tool_wins_over_filename_heuristic": (
        '<tool_call>\n<name>file_system</name> "operation": "read", '
        '"path": "my_filename.txt"\n</tool_call>',
        ("file_system", {"operation": "read", "path": "my_filename.txt"})),
    "extreme_deep_research_query_literal": (
        '<tool_call>\n<name>deep_research</name> "query": "x"\n</tool_call>',
        ("deep_research", {"query": "x"})),
}


@pytest.mark.parametrize("row", sorted(ROWS))
def test_dialect_dispatches_like_canonical(row):
    content, expected = ROWS[row]
    assert _one(content) == expected


def test_tool_wrapper_close_tag_is_healed_for_the_truncation_detector():
    """L13628 (`</tool>` → `</tool_call>`): the call dispatches either way,
    but with the close tag unhealed the truncation counters read 1 open /
    0 closes and the parse reason becomes 'truncated' — the model is told
    its output was cut off (shorten it!) for a syntax error, and the operator
    stream gets an 'Upstream Truncation' warning per reply."""
    names, reason = _names(
        '<tool>\n<function name="execute">\n<parameter name="command">echo hi'
        '</parameter>\n</function>\n</tool>\n<tool>garbage</tool>')
    assert names == ["execute", "system_parse_error"], names
    assert reason == "no_function_tag", reason


@pytest.mark.parametrize("content", [
    "<tool_call>not a call at all</tool_call>",
    '<tool_call>{"foo": 1}</tool_call>',           # JSON without a name, `foo` is no tool
    '<tool_call>{"nottool": {"a": 1}}</tool_call>',  # root key is not a tool
    # L14131 — Fallback 1 accepts a leading tag as the tool name ONLY if it is
    # an exposed tool: `<parameter>` and `<b>` are not tools.
    '<tool_call>\n<parameter name="command">echo hi</parameter>\n</tool_call>',
    '<tool_call>\n<b>bold</b> not a call\n</tool_call>',
    # L14079 — the root-key heal needs exactly ONE root key
    '<tool_call>{"execute": {"command": "x"}, "web_search": {"query": "q"}}</tool_call>',
    # L14084 — the OpenAI-nested heal needs a name inside "function"
    '<tool_call>{"function": {"arguments": {"command": "x"}}}</tool_call>',
])
def test_non_calls_become_one_system_parse_error(content):
    # The parser's contract for an unparseable wrapper is ONE synthetic
    # `system_parse_error` call (a strike + the recovery hint), never a
    # silently-dropped block and never a real tool with guessed args.
    tcs, ui, reason = _agent()._parse_assistant_tool_calls(content, {})
    assert [t["function"]["name"] for t in tcs] == ["system_parse_error"]


def test_cdata_wins_over_format1_and_is_not_double_parsed():
    # Format 1 also matches the CDATA opener; CDATA must have populated the
    # value first so Format 1's `not in args_val` guard keeps the full body.
    content = ('<tool_call>\n<function name="file_system">\n'
               '<parameter name="operation">write</parameter>\n'
               '<parameter name="path">a.py</parameter>\n'
               '<parameter name="content"><![CDATA[x = "</parameter>"\n'
               'y = 1 < 3 > 2]]></parameter>\n</function>\n</tool_call>')
    name, args = _one(content)
    assert name == "file_system"
    assert args == {"operation": "write", "path": "a.py",
                    "content": 'x = "</parameter>"\ny = 1 < 3 > 2'}


_F1F2_HISTORY = (  # kept as the record of why these two pins exist
    "§4EC F1/F2 (2026-09-02): CDATA WAS extracted at Format 0a but the body was "
    "never masked from the working block, so every structural regex still sees "
    "it — F1: the <tool_call> block split (~L13851) and the truncation counters "
    "run BEFORE CDATA extraction, so a body containing a literal <tool_call> is "
    "split in half (truncated write + a system_parse_error strike); F2: Formats "
    "3/5 run AFTER and harvest tags inside the body as arguments (`<b>` becomes "
    "args['b']), which the tool then rejects as an unexpected keyword. CDATA's "
    "own comment promises 'ANYTHING'. Strict: flips to a failure the day the "
    "block is CDATA-masked, at which point these become plain pins.")


def test_cdata_protects_a_literal_tool_call_marker():
    content = ('<tool_call>\n<function name="file_system">\n'
               '<parameter name="operation">write</parameter>\n'
               '<parameter name="path">a.py</parameter>\n'
               '<parameter name="content"><![CDATA[y = "<tool_call>"]]></parameter>'
               '\n</function>\n</tool_call>')
    assert _one(content) == ("file_system", {"operation": "write", "path": "a.py",
                                             "content": 'y = "<tool_call>"'})


def test_cdata_body_tags_are_not_harvested_as_arguments():
    content = ('<tool_call>\n<function name="file_system">\n'
               '<parameter name="operation">write</parameter>\n'
               '<parameter name="path">a.html</parameter>\n'
               '<parameter name="content"><![CDATA[<p>hi</p><b>x</b>]]></parameter>'
               '\n</function>\n</tool_call>')
    assert _one(content) == ("file_system", {"operation": "write", "path": "a.html",
                                             "content": "<p>hi</p><b>x</b>"})


def test_repair_never_shrinks_a_clean_value():
    # Repair only REPLACES when strictly longer; a clean two-param call is untouched.
    content = ('<tool_call>\n<function name="file_system">\n'
               '<parameter name="operation">read</parameter>\n'
               '<parameter name="path">a.py</parameter>\n</function>\n</tool_call>')
    assert _one(content) == ("file_system", {"operation": "read", "path": "a.py"})


# --- parse-failure diagnosis (agent.py L14265-14290) ---------------------------
@pytest.mark.parametrize("content,names,reason", [
    ("<tool_call>garbage</tool_call>", ["system_parse_error"], "no_function_tag"),
    ('<tool_call>\n<function name="execute"', ["system_parse_error"], "truncated"),
    # the per-BLOCK truncation arm (L14273): wrapper closed, function tag never
    # closed — the whole-reply pre-check counts 1/1 wrappers and sees no `<function…>`
    ('<tool_call>\n<function name="execute"\n</tool_call>', ["system_parse_error"], "truncated"),
    # first failure's reason sticks; the batch does NOT stop at a bad block
    ("<tool_call>garbage</tool_call>\n" + CANON, ["system_parse_error", "execute"], "no_function_tag"),
    # a truncation is reported once per reply — later fragments are folded into it
    ('<tool_call>\n<function name="execute"\n<tool_call>garbage</tool_call>', ["system_parse_error"], "truncated"),
    # non-truncation failures are NOT deduped: two garbage blocks, two strikes
    ("<tool_call>garbage</tool_call>\n<tool_call>junk</tool_call>",
     ["system_parse_error", "system_parse_error"], "no_function_tag"),
])
def test_parse_failure_reason_and_strike_count(content, names, reason):
    assert _names(content) == (names, reason)


def test_an_empty_function_name_is_a_malformed_strike_not_silence():
    assert _names('<tool_call>\n<function name=>\n</function>\n</tool_call>') == (
        ["system_parse_error"], "malformed")


def test_a_swallowed_block_error_does_not_overwrite_an_earlier_reason():
    # first failure wins: a truncated reply followed by an empty-name block
    # keeps "truncated" (the recovery hint the model needs) — the handler only
    # fills an EMPTY reason
    names, reason = _names('<tool_call>\n<function name="execute"\n'
                           '<tool_call>\n<function name=>\n</function>\n</tool_call>')
    assert reason == "truncated" and names.count("system_parse_error") >= 1


def test_xml_path_scrubs_tool_xml_from_the_ui_text():
    tcs, ui, reason = _agent()._parse_assistant_tool_calls("Sure, running it.\n" + CANON, {})
    assert [t["function"]["name"] for t in tcs] == ["execute"]
    assert ui == "Sure, running it." and "<" not in ui


# --- drift guard -------------------------------------------------------------
# Every `<name>[...] = ...` assignment in the parser is an extraction sink
# (there is no other dict-key write in the method). Counted by SHAPE, not by
# variable name, so a sink under a new name (the first version of this guard
# missed the five `args_dict` sinks of the extreme fallback) still counts.
EXPECTED_EXTRACTION_SINKS = 19   # 16 args_val/args_fallback + 3 args_dict (subscript writes only); §4EC removed the Format-0a CDATA sink (masking) and the repair-None arm (F4)


def test_every_extraction_sink_has_a_dialect_row():
    """If this fails, a sink was added or removed: add or remove its ROWS
    entry above, THEN update the count. Never update the count alone — a
    decoy assignment satisfies the count; only a row can prove the sink."""
    src = inspect.getsource(GhostAgent._parse_assistant_tool_calls)
    tree = ast.parse("class _X:\n" + "\n".join("    " + l for l in src.splitlines()))
    sinks = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Subscript)
             and isinstance(n.targets[0].value, ast.Name)]
    names = sorted({n.targets[0].value.id for n in sinks})
    assert len(sinks) == EXPECTED_EXTRACTION_SINKS, (len(sinks), names)
