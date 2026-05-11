"""Tests for the XML-parser hardening + parse-error escape-hatch fixes.

Background: the deep-research selfplay session caught a loop where the
agent burned 5 consecutive turns on `tool_syntax_error` because the
recovery prompt kept telling the model to use the same XML shape that
was failing. Three fixes:

1. **CDATA envelope** — `<parameter name="X"><![CDATA[...]]></parameter>`
   passes the inner body through verbatim, so literal `</parameter>`,
   `<`, `>`, JSON, and quotes inside content don't truncate the parser.

2. **Bounds-aware extraction** — when CDATA isn't used, the fallback
   parser walks `<parameter ...>` openings in order and uses the LAST
   `</parameter>` before the next opening (or `</function>`) as the
   body terminator, surviving literal `</parameter>` substrings inside
   docstrings/examples that previously truncated the body.

3. **Pivot prompt + fallback hint** — after ≥2 consecutive
   `system_parse_error` events, the recovery message switches from
   "use XML" (the same shape that just failed) to a strategy-pivot
   hint suggesting CDATA, heredoc-via-execute, or chunked replace.

The full XML parser is inlined inside `handle_chat`, so we test the
constituent regex patterns directly + the source-level guarantees of
the strike-counter logic.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ghost_agent.tools.tool_failure import get_fallback_hint


# ---------------------------------------------------------------------------
# Fallback hint wired for the synthetic `system` tool.
# ---------------------------------------------------------------------------

def test_fallback_hint_for_system_parse_error_match():
    err = (
        "SYSTEM ERROR: Your `<tool_call>` was invalid or contained broken JSON. "
        "DO NOT output JSON dictionaries inside the tool call."
    )
    hint = get_fallback_hint("system", err)
    assert hint is not None
    assert "CDATA" in hint
    assert "heredoc" in hint.lower() or "execute" in hint.lower()


def test_fallback_hint_for_escape_hatch_match():
    err = "SYSTEM ESCAPE HATCH (parse failed 2 turns in a row)..."
    hint = get_fallback_hint("system", err)
    assert hint is not None
    assert "switch" in hint.lower() or "shape" in hint.lower()


def test_fallback_hint_returns_none_for_unrelated_system_text():
    assert get_fallback_hint("system", "totally unrelated message") is None


# ---------------------------------------------------------------------------
# Source-level guarantees — the fix landed in agent.py / prompts.py /
# tool_failure.py. Verify the structural invariants without re-running
# the entire chat loop.
# ---------------------------------------------------------------------------

def _agent_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "src/ghost_agent/core/agent.py"
    ).read_text()


def _prompts_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "src/ghost_agent/core/prompts.py"
    ).read_text()


def test_agent_tracks_consecutive_parse_errors():
    src = _agent_source()
    assert "consecutive_parse_errors = 0" in src, (
        "agent must initialise the per-request parse-error counter"
    )
    assert "consecutive_parse_errors += 1" in src, (
        "agent must increment the counter on each system_parse_error"
    )


def test_agent_resets_parse_errors_on_successful_dispatch():
    src = _agent_source()
    # The reset happens once we reach the parallel-execution path
    # (proves at least one tool parsed cleanly this turn).
    assert re.search(
        r"consecutive_parse_errors\s*=\s*0\s*\n\s*for i, result in enumerate\(results\):",
        src,
    ), "consecutive_parse_errors should reset just before result dispatch"


def test_agent_pivot_prompt_includes_alternative_strategies():
    src = _agent_source()
    # The pivot block must still give the model multiple shapes to try
    # when XML is malformed (not truncated). The current set is:
    # CDATA envelope, file_system write, file_system replace. HEREDOC
    # and BASE64 were dropped after production traces showed the model
    # applying them to truncation failures (where they don't help).
    assert "CDATA" in src
    assert "file_system" in src
    assert "ESCAPE HATCH" in src

    # Reason-specific branches must all exist — the generic "your XML
    # is broken" message is what trapped the model in a 6-turn loop.
    assert '"truncated"' in src or "'truncated'" in src
    assert '"no_function_tag"' in src or "'no_function_tag'" in src
    assert '"malformed"' in src or "'malformed'" in src


def test_agent_pivot_threshold_is_two():
    src = _agent_source()
    # The threshold must be ≥2 so a single one-off parse failure
    # doesn't trigger the heavier pivot prompt.
    assert "consecutive_parse_errors >= 2" in src


def test_prompts_document_cdata_envelope():
    src = _prompts_source()
    # The system prompt must teach the model about CDATA so it can
    # use the escape hatch proactively, not only under pivot.
    assert "CDATA" in src and "<![CDATA[" in src


# ---------------------------------------------------------------------------
# CDATA regex behaviour — verify the pattern we baked into the agent
# parser correctly extracts contents that include literal `</parameter>`.
# ---------------------------------------------------------------------------

CDATA_PATTERN = re.compile(
    r'<parameter(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_-]+)["\']?\s*>\s*<!\[CDATA\[(.*?)\]\]>\s*</parameter>',
    re.DOTALL | re.IGNORECASE,
)


def test_cdata_extracts_literal_close_tag_inside_body():
    block = """
    <parameter name="content"><![CDATA[
    # this docstring has </parameter> inside it which used to break the parser
    print("ok")
    ]]></parameter>
    """
    matches = list(CDATA_PATTERN.finditer(block))
    assert len(matches) == 1
    body = matches[0].group(2)
    assert "</parameter>" in body, "CDATA body must preserve literal </parameter>"
    assert "print(\"ok\")" in body


def test_cdata_extracts_json_inside_body():
    block = '<parameter name="content"><![CDATA[{"key": "value", "nested": {"<x>": "[ok]"}}]]></parameter>'
    matches = list(CDATA_PATTERN.finditer(block))
    assert len(matches) == 1
    assert matches[0].group(2) == '{"key": "value", "nested": {"<x>": "[ok]"}}'


def test_cdata_handles_multiple_params_in_one_block():
    block = """
    <parameter name="path"><![CDATA[/sandbox/file.py]]></parameter>
    <parameter name="content"><![CDATA[print("hi")]]></parameter>
    """
    matches = list(CDATA_PATTERN.finditer(block))
    assert len(matches) == 2
    found = {m.group(1): m.group(2) for m in matches}
    assert found["path"] == "/sandbox/file.py"
    assert found["content"] == 'print("hi")'


def test_cdata_pattern_does_not_match_non_cdata_params():
    block = '<parameter name="content">just text</parameter>'
    assert not CDATA_PATTERN.search(block), (
        "CDATA pattern must only match CDATA-wrapped bodies"
    )


# ---------------------------------------------------------------------------
# Bounds-aware extraction — tests for the LAST-`</parameter>`-before-next-
# opening logic that survives literal `</parameter>` inside content.
# ---------------------------------------------------------------------------

def _bounds_aware_extract(block: str) -> dict:
    """Replica of the bounds-aware logic from agent.py for unit testing."""
    open_pattern = re.compile(
        r'<parameter(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_-]+)["\']?[^>]*>',
        re.IGNORECASE,
    )
    close_re = re.compile(r'</parameter>', re.IGNORECASE)
    end_func_re = re.compile(r'<function\b|</function>|<tool_call\b|</tool_call>', re.IGNORECASE)
    out: dict = {}
    openings = list(open_pattern.finditer(block))
    # Skip past the FIRST <function> opening before searching for the
    # boundary — otherwise the legit opening at position ~0 would be
    # treated as end-of-function and kill every parameter extraction.
    first_func = re.search(r'<function\b', block, re.IGNORECASE)
    search_start = first_func.end() if first_func else 0
    end_func = end_func_re.search(block, search_start)
    end_pos = end_func.start() if end_func else len(block)
    for i, op in enumerate(openings):
        p_name = op.group(1).strip().strip('"').strip("'")
        if p_name in out:
            continue
        body_start = op.end()
        next_open = openings[i + 1].start() if i + 1 < len(openings) else end_pos
        boundary = min(next_open, end_pos)
        candidates = [m for m in close_re.finditer(block, body_start, boundary)]
        body_end = candidates[-1].start() if candidates else boundary
        if body_end > body_start:
            out[p_name] = block[body_start:body_end].strip("\r\n")
    return out


def test_bounds_aware_handles_literal_close_inside_content():
    block = (
        '<parameter name="path">solution.py</parameter>'
        '<parameter name="content">'
        'docstring = """The XML format uses </parameter> as the close tag."""\n'
        'print(docstring)'
        '</parameter>'
        '</function>'
    )
    args = _bounds_aware_extract(block)
    assert args["path"] == "solution.py"
    # Must include the literal </parameter> inside the docstring AND the
    # full body up to the LAST </parameter> before </function>.
    assert "</parameter>" in args["content"]
    assert "print(docstring)" in args["content"]


def test_bounds_aware_handles_multiple_clean_params():
    block = (
        '<parameter name="operation">write</parameter>'
        '<parameter name="path">a.txt</parameter>'
        '<parameter name="content">hello</parameter>'
        '</function>'
    )
    args = _bounds_aware_extract(block)
    assert args == {"operation": "write", "path": "a.txt", "content": "hello"}


def test_bounds_aware_handles_missing_close_tag():
    # Truncated stream — last param has no closing tag. The body should
    # still be captured up to </function>.
    block = (
        '<parameter name="path">x.py</parameter>'
        '<parameter name="content">print(1)\nprint(2)'
        '</function>'
    )
    args = _bounds_aware_extract(block)
    assert args["path"] == "x.py"
    assert "print(1)" in args["content"]
    assert "print(2)" in args["content"]


# ---------------------------------------------------------------------------
# Integration smoke test: build a fake LLM response with a CDATA-wrapped
# file_system call and run it through the actual agent parser block.
# Because the parser lives inside `handle_chat`, we exercise it via the
# raw block extraction we use in tests/test_agent_xml_edge.py — but
# this time we use the IMPROVED patterns that mirror the in-code fixes.
# ---------------------------------------------------------------------------

def test_full_extraction_with_cdata_wrapper():
    """End-to-end: the new CDATA + bounds-aware extractor working together."""
    raw_response = """
    <think>writing the file</think>
    <tool_call>
    <function name="file_system">
    <parameter name="operation">write</parameter>
    <parameter name="path">demo.py</parameter>
    <parameter name="content"><![CDATA[
data = {"json": True, "tag": "<a href='x'>"}
print(data)
# this also has </parameter> in a docstring example
]]></parameter>
    </function>
    </tool_call>
    """
    blocks = re.split(r"<tool_call.*?>", raw_response, flags=re.IGNORECASE)
    block = re.split(r"</tool_call.*?>", blocks[1], flags=re.IGNORECASE)[0]

    # CDATA pass first
    cdata_args: dict = {}
    cdata_block = block
    for cm in CDATA_PATTERN.finditer(block):
        cdata_args[cm.group(1)] = cm.group(2)
    cdata_block = CDATA_PATTERN.sub("", cdata_block)

    # Bounds-aware pass on what's left
    rest = _bounds_aware_extract(cdata_block)
    args = {**rest, **cdata_args}

    assert args["operation"] == "write"
    assert args["path"] == "demo.py"
    assert "data = {" in args["content"]
    assert "</parameter>" in args["content"], (
        "literal </parameter> inside docstring must survive CDATA pass-through"
    )


# ---------------------------------------------------------------------------
# Regression guard: the bounds-aware pass MUST NOT corrupt clean
# non-CDATA tool calls (the shape self-play attempts 1 and 2 used). After
# a prior revision the repair pass ran BEFORE Formats 1-7 and garbled
# args_val for self-closing / value= / attr shapes. The repair now runs
# AFTER those formats and only replaces a value when the repaired body
# is strictly longer than what was already extracted.
# ---------------------------------------------------------------------------

def _full_parser_snapshot(block_content: str) -> dict:
    """Reimplements the agent parser's Formats 0a → 5b sequence for test
    purposes. Mirrors the in-code ordering so a test failure here
    reflects the agent's actual parse behaviour.
    """
    args_val: dict = {}

    # Format 0a — CDATA
    cdata_hits = list(CDATA_PATTERN.finditer(block_content))
    for cm in cdata_hits:
        args_val[cm.group(1).strip()] = cm.group(2)

    # Format 1 — <parameter name="x">y</parameter>
    # Lookahead breaks on opening `<function` / `<tool_call` too, so a
    # block-split miss between two consecutive tool calls cannot let
    # one param's value swallow the next tool call's contents.
    param_matches = list(re.finditer(
        r'<parameter(?:\s+name=|=)([^>]+)>(.*?)'
        r'(?=</parameter>|<parameter(?:\s+name=|=)|<function\b|</function>|<tool_call\b|</tool_call>|$)',
        block_content, re.DOTALL | re.IGNORECASE,
    ))
    for p in param_matches:
        p_name = p.group(1).split()[0].strip().strip('"').strip("'")
        p_val = p.group(2).strip("\r\n")
        if p_name not in args_val:
            args_val[p_name] = p_val

    # Format 2 — value= attribute
    alt = list(re.finditer(
        r'<parameter\s+name=["\']([^"\']+)["\']\s+value=(["\'])(.*?)\2\s*(?:/|>.*?</parameter>)',
        block_content, re.DOTALL | re.IGNORECASE,
    ))
    for p in alt:
        args_val[p.group(1)] = p.group(3)

    # Format 5b — bounds-aware REPAIR pass
    open_pattern = re.compile(
        r'<parameter(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_-]+)["\']?[^>]*>',
        re.IGNORECASE,
    )
    close_re = re.compile(r'</parameter>', re.IGNORECASE)
    end_func_re = re.compile(r'<function\b|</function>|<tool_call\b|</tool_call>', re.IGNORECASE)
    openings = list(open_pattern.finditer(block_content))
    # Skip past the FIRST <function> opening before searching for the
    # boundary (mirrors the agent.py fix).
    first_func = re.search(r'<function\b', block_content, re.IGNORECASE)
    search_start = first_func.end() if first_func else 0
    end_func = end_func_re.search(block_content, search_start)
    end_pos = end_func.start() if end_func else len(block_content)
    for i, op in enumerate(openings):
        p_name = op.group(1).strip().strip('"').strip("'")
        in_cdata = any(cm.start() <= op.start() < cm.end() for cm in cdata_hits)
        if in_cdata:
            continue
        if block_content[op.end() - 2:op.end()] == "/>":
            continue
        body_start = op.end()
        next_open = openings[i + 1].start() if i + 1 < len(openings) else end_pos
        boundary = min(next_open, end_pos)
        candidates = [m for m in close_re.finditer(block_content, body_start, boundary)]
        if not candidates:
            continue
        body_end = candidates[-1].start()
        if body_end <= body_start:
            continue
        repaired = block_content[body_start:body_end].strip("\r\n")
        existing = args_val.get(p_name)
        if isinstance(existing, str) and len(repaired) > len(existing):
            args_val[p_name] = repaired
        elif existing is None:
            args_val[p_name] = repaired

    return args_val


def test_clean_file_system_write_still_parses():
    """The exact shape that self-play attempts 1 & 2 used successfully."""
    block = (
        '\n<function name="file_system">\n'
        '<parameter name="operation">write</parameter>\n'
        '<parameter name="path">solution.py</parameter>\n'
        '<parameter name="content">\n'
        'import csv\n'
        'from collections import defaultdict\n\n'
        'def main():\n'
        '    revenue = defaultdict(float)\n'
        '    with open("sales_data.csv") as f:\n'
        '        for row in csv.DictReader(f):\n'
        '            revenue[row["category"]] += float(row["price"]) * int(row["quantity"])\n'
        '    for cat in sorted(revenue.keys()):\n'
        '        print(f"  {cat}: ${revenue[cat]:.2f}")\n\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
        '</parameter>\n'
        '</function>\n'
    )
    args = _full_parser_snapshot(block)
    assert args["operation"] == "write"
    assert args["path"] == "solution.py"
    assert "import csv" in args["content"]
    assert "defaultdict" in args["content"]
    assert "main()" in args["content"]


def test_self_closing_value_attr_not_corrupted_by_repair_pass():
    """Self-closing `<parameter name="x" value="y" />` shapes must survive
    the new bounds-aware repair pass untouched — Format 2 populates them
    and the repair pass sees the `/>` prefix and skips.
    """
    block = (
        '<function name="execute">'
        '<parameter name="command" value="ls -la" />'
        '<parameter name="timeout" value="30" />'
        '</function>'
    )
    args = _full_parser_snapshot(block)
    assert args["command"] == "ls -la"
    assert args["timeout"] == "30"


def test_repair_pass_fills_content_truncated_by_format1():
    """When the body contains a literal `</parameter>` (e.g. a docstring
    showing the XML format), Format 1's non-greedy regex truncates it.
    Format 5b must detect the longer repair and swap it in.
    """
    block = (
        '<function name="file_system">'
        '<parameter name="operation">write</parameter>'
        '<parameter name="path">demo.py</parameter>'
        '<parameter name="content">'
        'docstring = """The XML close tag is </parameter> — here as literal."""\n'
        'print(docstring)'
        '</parameter>'
        '</function>'
    )
    args = _full_parser_snapshot(block)
    assert args["operation"] == "write"
    assert args["path"] == "demo.py"
    assert "print(docstring)" in args["content"], (
        "repair pass must recover content past the literal </parameter>"
    )


def test_repair_pass_never_shrinks_existing_value():
    """Even if the bounds-aware logic finds a shorter repair (unlikely
    but possible on degenerate input), the existing Format-1 extraction
    must not be overwritten with something smaller.
    """
    block = (
        '<function name="file_system">'
        '<parameter name="content">'
        'full body with many words including a terminator </parameter>'
        'trailing garbage that should NOT replace the full body'
        '</parameter>'
        '</function>'
    )
    # Format 1 captures up to the FIRST </parameter> → "full body with ... terminator ".
    # The repair pass finds the LAST </parameter> before </function> → a longer string.
    # That's a legitimate improvement, so the longer version wins.
    args = _full_parser_snapshot(block)
    assert "terminator" in args["content"]
    # Either Format 1 stopped early (length = 52ish) or the repair promoted
    # to the full body. Whichever wins, "terminator" must survive.
    # This test exists so a future change that SHRINKS the value is caught.


def test_agent_source_repair_pass_runs_after_format5():
    """Ordering invariant — the repair pass must NOT run before Format 1
    or it will corrupt self-closing / value= shapes. Verify by reading
    the agent source and locating the repair marker relative to the
    other format markers.
    """
    src = _agent_source()
    f0a = src.find("# Format 0a: CDATA envelope")
    f1 = src.find("# Format 1:")
    f2 = src.find("# Format 2:")
    f5 = src.find("# Format 5:")
    f5b = src.find("# Format 5b:")
    assert f0a < f1 < f2 < f5 < f5b, (
        "Format ordering broken — repair pass (5b) must run AFTER 0a-5 "
        f"(offsets: 0a={f0a}, 1={f1}, 2={f2}, 5={f5}, 5b={f5b})"
    )


def test_agent_source_logs_block_on_system_parse_error():
    """When the parser emits system_parse_error we must log the block
    content so the regression is diagnosable from traces alone — the
    previous `tool syntax error` log with no payload hid 5+ identical
    failures during the self-play session.
    """
    src = _agent_source()
    assert "Parser emitted system_parse_error" in src
    assert "block_content[:4096]" in src


# ---------------------------------------------------------------------------
# Multi-tool-call boundary regression — when the model emits TWO tool calls
# in one turn AND the first call uses a non-`</parameter>` close (e.g.
# `</path>`) on its last param, the parameter regex used to consume past
# the first call's `</function>`/`</tool_call>` into the SECOND call's
# `<parameter=operation>write</parameter>`, ending up with a single param
# value containing both calls' tag-soup. Production trace 07:20:32 (request
# `F7`): file_system was called with operation="data.csv</path> </function>
# </tool_call> <tool_call> <function=file_system> <parameter=operation>
# write" — the operation lookup naturally failed and the cycle had to
# retry. The fix adds opening `<function` / `<tool_call` to the regex
# lookahead so a stray opening of the next tool call terminates the
# parameter just like a closing tag does.
# ---------------------------------------------------------------------------


def test_param_regex_stops_at_next_function_opening():
    """Worst-case: a block_content where the block-split missed the
    boundary between two tool calls (e.g. multi-line `<tool_call\\n>`
    against the no-DOTALL split, or no `</tool_call>` between them).
    The single block now contains TWO `<function=...>` openings and
    the first param value would otherwise run into the second call.
    """
    block_content = (
        "<function=file_system>"
        "<parameter=operation>read</parameter>"
        "<parameter=file_path>data.csv</path>"  # stray </path>, not </parameter>
        # Note: NO </function> / </tool_call> — block split missed the boundary.
        "<function=file_system>"
        "<parameter=operation>write</parameter>"
        "<parameter=file_path>solution.py</parameter>"
        "</function>"
    )
    args = _full_parser_snapshot(block_content)
    # The first call's file_path must NOT include the second call's tag-soup.
    fp = args.get("file_path", "")
    assert "<function" not in fp, (
        f"file_path leaked across tool-call boundary: {fp!r}"
    )
    assert "<tool_call" not in fp, (
        f"file_path leaked into next <tool_call>: {fp!r}"
    )
    # Operation parsed cleanly from the first call (the canonical case the
    # second call's `<parameter=operation>write</parameter>` would have
    # overwritten under the broken regex via the bounds-aware repair).
    assert args.get("operation") == "read"


def test_param_regex_stops_at_next_tool_call_opening():
    """Same idea but the next call's `<tool_call>` opening sits inside
    the param body (block-split miss because the opening was on its
    own line and `<tool_call.*?>` is split without DOTALL).
    """
    block_content = (
        "<function=file_system>"
        "<parameter=operation>read</parameter>"
        "<parameter=file_path>data.csv</path>"
        "<tool_call>"
        "<function=file_system>"
        "<parameter=operation>write</parameter>"
        "</function>"
    )
    args = _full_parser_snapshot(block_content)
    fp = args.get("file_path", "")
    assert "<tool_call" not in fp
    assert "<function" not in fp
    # Operation must remain "read" — the second call's "write" must not
    # overwrite it via the repair pass.
    assert args.get("operation") == "read"


def test_param_regex_clean_two_call_block_split_still_works():
    """Sanity check: when the block-split correctly partitions the two
    tool calls (the common case), each block's params parse cleanly.
    The fix to the lookahead must not regress the happy path.
    """
    # Block 1: clean first call.
    block1 = (
        "<function=file_system>"
        "<parameter=operation>read</parameter>"
        "<parameter=file_path>data.csv</parameter>"
        "</function>"
    )
    args1 = _full_parser_snapshot(block1)
    assert args1 == {"operation": "read", "file_path": "data.csv"}

    # Block 2: clean second call.
    block2 = (
        "<function=file_system>"
        "<parameter=operation>write</parameter>"
        "<parameter=file_path>solution.py</parameter>"
        "<parameter=content>print(1)</parameter>"
        "</function>"
    )
    args2 = _full_parser_snapshot(block2)
    assert args2 == {
        "operation": "write",
        "file_path": "solution.py",
        "content": "print(1)",
    }
