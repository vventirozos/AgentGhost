"""Regression tests for the req 65d8cf76 parse-failure cluster (2026-08-01).

One lesioned emission — '<tool_call>function=knowledge_base>' (the '\n<'
between the opener and the function tag lost upstream) — struck out five
times and aborted the request, because:

  1. the fallback parser had no heal for a dropped '<' on the function
     opener (every other sloppy shape had one),
  2. the broken bytes were replayed verbatim as the assistant turn, so
     the model copied its own mistake on every retry, and
  3. the no_function_tag recovery hint taught the attribute dialect
     (`<function name="…">`) on the native path whose template speaks
     the equals dialect (`<function=…>`) — the dual-dialect problem
     root-caused on 2026-07-31.

These tests pin the three fixes: the heal, the history scrub, and the
path-aware hint example.
"""
import inspect
import json

import pytest

from ghost_agent.core.agent import (
    GhostAgent,
    _render_assistant_with_tool_calls,
    _scrub_unparsed_tool_call_text,
    _tool_call_format_example,
)


def make_agent(tool_names=("web_search", "execute", "file_system",
                           "knowledge_base", "manage_projects")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in tool_names}
    return agent


def parse(content, msg=None, agent=None):
    agent = agent or make_agent()
    return agent._parse_assistant_tool_calls(content, msg if msg is not None else {})


def call_names(tool_calls):
    return [tc["function"]["name"] for tc in tool_calls]


def args_of(tool_call):
    args = tool_call["function"]["arguments"]
    return json.loads(args) if isinstance(args, str) else args


# The exact turn-1 emission from the live boundary recording
# (llm_recordings/2026-08-01.jsonl, req 65d8cf76eb814134, ordinal 238).
REQ_65D8CF76_TURN1 = (
    "<tool_call>function=knowledge_base>\n"
    "<parameter=action>\nquery\n</parameter>\n"
    "<parameter=filename>\nverifier_report.md\n</parameter>\n"
    "<parameter=question>\nWhat is the pending issue with task 6? "
    "What claim needs evidence?\n</parameter>\n"
    "</function>\n</tool_call>"
)


class TestDroppedAngleBracketHeal:
    def test_req_65d8cf76_turn1_now_parses(self):
        tool_calls, _, reason = parse(REQ_65D8CF76_TURN1)
        assert reason == ""
        assert call_names(tool_calls) == ["knowledge_base"]
        args = args_of(tool_calls[0])
        assert args["action"] == "query"
        assert args["filename"] == "verifier_report.md"
        assert args["question"].startswith("What is the pending issue")

    def test_double_call_lesion_both_recover(self):
        # Turn 3 of the incident emitted TWO lesioned calls in one reply.
        content = (
            "<tool_call>function=file_system>\n"
            "<parameter=operation>\nread\n</parameter>\n"
            "<parameter=path>\nprojects/x/answer.md\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>function=manage_projects>\n"
            "<parameter=action>\nartifact_add\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["file_system", "manage_projects"]

    def test_lesion_with_surviving_newline_heals(self):
        content = (
            "<tool_call>\nfunction=web_search>\n"
            "<parameter=query>\ntor engines\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["web_search"]
        assert args_of(tool_calls[0])["query"] == "tor engines"

    def test_lesioned_attribute_dialect_heals_too(self):
        content = (
            "<tool_call>function name=\"web_search\">\n"
            "<parameter name=\"query\">postgres</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["web_search"]

    def test_function_eq_inside_parameter_body_is_not_rewritten(self):
        # The heal is anchored to the <tool_call> opener; 'function=' in a
        # parameter body must survive byte-for-byte.
        content = (
            "<tool_call>\n<function name=\"execute\">\n"
            "<parameter name=\"content\">x = dict(function=len)</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        name = tool_calls[0]["function"]["name"]
        assert name == "execute"
        assert args_of(tool_calls[0])["content"] == "x = dict(function=len)"

    def test_prose_mentioning_function_eq_is_untouched(self):
        tool_calls, ui, reason = parse("Set function=main in the config.")
        assert tool_calls == []
        assert reason == ""
        assert "function=main" in ui

    def test_well_formed_equals_dialect_still_parses(self):
        # Guard: the heal must not disturb the healthy template dialect.
        content = (
            "<tool_call>\n<function=file_system>\n"
            "<parameter=operation>\nread\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["file_system"]

    def test_lesion_behind_unclosed_think_still_heals(self):
        # Review catch: _THINK_UNCLOSED_RE only anchored on well-formed
        # openers, so the lesioned call after an unclosed <think> was
        # stripped to EOS — swallowed silently (no strike, no hint,
        # empty turn). The lookahead now knows the lesion shape.
        content = "<think>Let me query the KB\n" + REQ_65D8CF76_TURN1
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["knowledge_base"]

    def test_lesion_with_attributed_opener_heals(self):
        # Review catch: sibling heals accept attribute-bearing openers,
        # the dropped-'<' heal required a bare '<tool_call>'.
        content = (
            "<tool_call id=\"1\">function=web_search>\n"
            "<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["web_search"]

    def test_unquoted_function_name_variant_parses(self):
        # Review catch: '<tool_call>function_name=web_search>' healed
        # into '<function_name=web_search>' which nothing downstream
        # could parse (struck out as "malformed"). The tag now
        # normalizes to the equals dialect.
        content = (
            "<tool_call>function_name=web_search>\n"
            "<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"
        )
        tool_calls, _, reason = parse(content)
        assert reason == ""
        assert call_names(tool_calls) == ["web_search"]

    def test_truncated_reason_with_fully_parsed_call_has_no_synthetic_entry(self):
        # Premise of the scrub gate: "truncated" can be stamped by the
        # pre-parse open/close counter while the call itself parses
        # completely (Format-1 '$' fallback). Such a turn must carry NO
        # system_parse_error entry — the gate keys on the entry, not
        # the reason, so this executed call's history is never scrubbed.
        content = (
            "<tool_call>\n<function=manage_projects>\n"
            "<parameter=action>\nartifact_add\n</parameter>\n"
            "<parameter=payload>\nreport.md"  # cut mid-param: no closers
        )
        tool_calls, _, reason = parse(content)
        assert reason == "truncated"
        assert "system_parse_error" not in call_names(tool_calls)
        assert "manage_projects" in call_names(tool_calls)


class TestBrokenReplayScrub:
    def test_broken_block_replaced_with_note(self):
        out = _scrub_unparsed_tool_call_text(REQ_65D8CF76_TURN1, "no_function_tag")
        assert "function=knowledge_base" not in out
        assert "malformed tool_call removed" in out
        assert "no_function_tag" in out

    def test_surrounding_prose_survives(self):
        text = "Let me check the report.\n" + REQ_65D8CF76_TURN1 + "\nDone."
        out = _scrub_unparsed_tool_call_text(text, "no_function_tag")
        assert "Let me check the report." in out
        assert "Done." in out
        assert "<parameter=" not in out

    def test_unclosed_block_scrubbed_to_end(self):
        text = "Prose first.\n<tool_call>function=execute>\n<parameter=command>\nls"
        out = _scrub_unparsed_tool_call_text(text, "truncated")
        assert "Prose first." in out
        assert "<parameter=command>" not in out
        assert "truncated" in out

    def test_bare_function_block_scrubbed(self):
        text = "<function name=\"execute\">\n<parameter name=\"command\">rm -rf /</parameter>\n</function>"
        out = _scrub_unparsed_tool_call_text(text, "malformed")
        assert "rm -rf" not in out
        assert "malformed tool_call removed" in out

    def test_adjacent_blocks_collapse_to_one_note(self):
        text = REQ_65D8CF76_TURN1 + "\n" + REQ_65D8CF76_TURN1
        out = _scrub_unparsed_tool_call_text(text, "no_function_tag")
        assert out.count("malformed tool_call removed") == 1

    def test_output_is_bounded_single_giant_block(self):
        big = "<tool_call>function=execute>\n<parameter=content>\n" + ("x" * 200_000)
        out = _scrub_unparsed_tool_call_text(big, "truncated")
        assert len(out) < 500

    def test_output_is_bounded_degenerate_many_blocks(self):
        # Review catch: 2000 tiny closed blocks each became a ~124-char
        # note (separators defeated the adjacency collapse) — a 5x
        # context AMPLIFIER in the retry path. Now only the first
        # _SCRUB_MAX_NOTES regions become notes; the rest are deleted.
        big = "<tool_call>x</tool_call>." * 2000
        out = _scrub_unparsed_tool_call_text(big, "no_function_tag")
        assert len(out) < len(big)
        assert out.count("malformed tool_call removed") <= 2

    def test_tool_hallucination_shape_is_scrubbed(self):
        # Review catch: '<tool>' (the shape common enough to have its
        # own parser heal) matched neither scrub pattern and was
        # replayed verbatim — the imitation loop, reopened.
        text = "<tool>\n<action>check_health</action>\n</tool>"
        out = _scrub_unparsed_tool_call_text(text, "no_function_tag")
        assert "<tool>" not in out
        assert "check_health" not in out
        assert "malformed tool_call removed" in out

    def test_function_name_shape_is_scrubbed(self):
        # Review catch: '\b' fails before '_', so '<function_name…' was
        # invisible to the '<function\b' pattern.
        text = "<function_name=web_search>\n<parameter=query>\nx\n</parameter>"
        out = _scrub_unparsed_tool_call_text(text, "malformed")
        assert "<function_name" not in out
        assert "malformed tool_call removed" in out

    def test_backtick_quoted_mentions_survive(self):
        # Review catch: retry turns QUOTE `<tool_call>` from the SYSTEM
        # ERROR hint while reasoning; eating from the quoted mention
        # onward destroyed legitimate analysis.
        text = (
            "The system said my `<tool_call>` was malformed. Key finding: "
            "the bug is in x().\n" + REQ_65D8CF76_TURN1
        )
        out = _scrub_unparsed_tool_call_text(text, "no_function_tag")
        assert "Key finding: the bug is in x()." in out
        assert "`<tool_call>`" in out
        assert "function=knowledge_base" not in out

    def test_reason_is_never_a_regex_template(self):
        # Review catch: the note went through re.sub as a replacement
        # TEMPLATE — a reason containing '\1' or '\s' crashed the
        # replay path; '\g<0>' silently re-injected the broken block.
        for evil in ("back\\slash", "group \\1 ref", "\\g<0>"):
            out = _scrub_unparsed_tool_call_text(REQ_65D8CF76_TURN1, evil)
            assert "function=knowledge_base" not in out
            assert evil in out

    def test_non_string_and_empty_passthrough(self):
        assert _scrub_unparsed_tool_call_text("", "x") == ""
        assert _scrub_unparsed_tool_call_text(None, "x") is None

    def test_handle_chat_wires_the_scrub_gated_on_failed_block(self):
        # Source guard: the replay site must scrub on parse failure —
        # replaying the broken bytes is what turned one lesion into five.
        # The gate must key on an ACTUAL system_parse_error entry, not
        # parse_failure_reason alone ("truncated" accompanies fully
        # executed calls; scrubbing those asks for a re-run of a
        # mutation that already happened).
        src = inspect.getsource(GhostAgent.handle_chat)
        assert "_scrub_unparsed_tool_call_text(" in src
        idx_strip = src.index("clean_content_for_history = _strip_think_blocks")
        idx_scrub = src.index("_scrub_unparsed_tool_call_text(")
        idx_assign = src.index('msg["content"] = clean_content_for_history')
        assert idx_strip < idx_scrub < idx_assign
        gate_region = src[idx_strip:idx_scrub]
        assert '"system_parse_error"' in gate_region

    def test_renderer_native_dialect_round_trips_and_matches_path(self):
        # The history renderer must speak the dialect of the active path:
        # on native, the equals dialect the model itself generated —
        # attribute-style renders were a standing dual-dialect nudge in
        # intra-request history (req 93 drifted to text-path emissions
        # late in a request full of them).
        call = {"id": "c1", "type": "function",
                "function": {"name": "execute",
                             "arguments": json.dumps({
                                 "command": "ls -la",
                                 "env": {"FOO": "bar"},
                                 "dry_run": True,
                             })}}
        native_out = _render_assistant_with_tool_calls("done.", [call], native=True)
        assert "<function=execute>" in native_out
        # Pin the full value-on-own-line shape, not just the tag: the
        # model's real recorded emissions put values on their own lines,
        # and an inline-value render would quietly re-create a shape
        # nudge while staying green on a tag-only assertion.
        assert "<parameter=command>\nls -la\n</parameter>" in native_out
        assert 'name="' not in native_out
        legacy_out = _render_assistant_with_tool_calls("done.", [call], native=False)
        assert '<function name="execute">' in legacy_out
        assert "<function=" not in legacy_out
        # Default stays legacy (back-compat with existing 2-arg callers).
        assert _render_assistant_with_tool_calls("done.", [call]) == legacy_out
        # Both dialects round-trip through our own fallback parser with
        # identical names and argument values.
        for rendered in (native_out, legacy_out):
            tool_calls, _, reason = parse(rendered)
            assert reason == ""
            assert call_names(tool_calls) == ["execute"]
            args = args_of(tool_calls[0])
            assert args["command"] == "ls -la"
            assert args["env"] == '{"FOO": "bar"}'
            assert args["dry_run"] == "true"

    def test_renderer_native_parallel_calls_all_render(self):
        # Mirror of the legacy 4-parallel-calls regression, equals dialect.
        calls = [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "update_profile",
                          "arguments": json.dumps({"key": f"k{i}", "value": f"v{i}"})}}
            for i in range(4)
        ]
        out = _render_assistant_with_tool_calls("", calls, native=True)
        assert out.count("<tool_call>") == 4
        assert out.count("<function=update_profile>") == 4
        for i in range(4):
            assert f"k{i}" in out and f"v{i}" in out

    def test_renderer_call_site_passes_path_flag(self):
        # Source guard: the single production call site must key the
        # dialect on args.native_tools.
        src = inspect.getsource(GhostAgent.handle_chat)
        assert "native=_render_native" in src
        assign = src.split("_render_native = ", 1)[1].split("req_messages", 1)[0]
        assert '"native_tools"' in assign

    def test_specialist_native_guidance_teaches_zero_xml(self):
        # The specialist prompt was the one dialect-teaching surface the
        # 07-31 header split missed: attribute-dialect examples + CDATA
        # rule on every native coding turn — the prompt-block class the
        # ablation showed corrupts stacked native calls 8/8. The native
        # variant must carry the same workflow with ZERO tool-call XML
        # and no CDATA (the server template parses values as raw text;
        # a CDATA wrapper would land verbatim inside the argument).
        from ghost_agent.core.prompts import (
            SPECIALIST_SYSTEM_PROMPT,
            SPECIALIST_TOOL_XML_LEGACY,
            SPECIALIST_TOOL_XML_NATIVE,
        )
        assert "<tool_call" not in SPECIALIST_TOOL_XML_NATIVE
        assert "<function" not in SPECIALIST_TOOL_XML_NATIVE
        # No CDATA *example* — the prose "no CDATA wrappers"
        # anti-instruction is allowed (and wanted).
        assert "<![CDATA[" not in SPECIALIST_TOOL_XML_NATIVE
        # Workflow survives the format removal.
        assert "fix_edit.py" in SPECIALIST_TOOL_XML_NATIVE
        assert "RAW" in SPECIALIST_TOOL_XML_NATIVE
        # Legacy keeps its contract verbatim.
        assert '<function name="file_system">' in SPECIALIST_TOOL_XML_LEGACY
        assert "<![CDATA[" in SPECIALIST_TOOL_XML_LEGACY
        # The shared body carries the placeholder and no attribute
        # dialect of its own.
        assert "{{TOOL_XML_GUIDANCE}}" in SPECIALIST_SYSTEM_PROMPT
        assert '<function name=' not in SPECIALIST_SYSTEM_PROMPT

    def test_specialist_splice_and_grammar_gate_are_path_aware(self):
        # Source guards: the specialist splice selects the guidance per
        # args.native_tools, and the GBNF grammar (attribute dialect,
        # hard-coded) is refused with a loud warning on the native path
        # instead of silently forcing a third dialect.
        src = inspect.getsource(GhostAgent.handle_chat)
        assert "SPECIALIST_TOOL_XML_NATIVE if _specialist_native" in src
        assert "SPECIALIST_TOOL_XML_LEGACY" in src
        assert "GHOST_TOOL_GRAMMAR=1 ignored" in src
        gate_region = src.split("grammar_payload_fields", 1)[0]
        assert '"native_tools"' in gate_region.rsplit("is_final_generation and all_tools", 1)[1]

    def test_watchdog_break_text_is_path_aware(self):
        # The think-loop sever injects a synthetic replan call that is
        # persisted and replayed as assistant history — it must speak
        # the active path's dialect.
        src = inspect.getsource(GhostAgent._stream_final_generation)
        assert "<function=replan>" in src
        assert '<function name=\\"replan\\">' in src
        assert '"native_tools"' in src

    def test_renderer_never_renders_synthetic_parse_error_entries(self):
        # Review catch: after the scrub removed '<tool_call' from the
        # replayed content, the history serializer's already-inline
        # suppression vanished and it rendered the synthetic
        # system_parse_error entries as a well-formed attribute-dialect
        # call to a nonexistent tool — imitable, and the exact
        # dual-dialect teaching the hint fix removed.
        synthetic = {"id": "call_x", "type": "function",
                     "function": {"name": "system_parse_error", "arguments": "{}"}}
        real = {"id": "call_y", "type": "function",
                "function": {"name": "web_search",
                             "arguments": json.dumps({"query": "x"})}}
        note = "[malformed tool_call removed]"
        out = _render_assistant_with_tool_calls(note, [synthetic])
        assert "system_parse_error" not in out
        assert out == note
        out_mixed = _render_assistant_with_tool_calls(note, [synthetic, real])
        assert "system_parse_error" not in out_mixed
        assert "web_search" in out_mixed


class TestPathAwareHintExample:
    def test_native_example_is_equals_dialect(self):
        ex = _tool_call_format_example(True)
        assert "<function=the_tool_name>" in ex
        assert 'name="' not in ex

    def test_legacy_example_is_attribute_dialect(self):
        ex = _tool_call_format_example(False)
        assert '<function name="the_tool_name">' in ex
        assert "<function=" not in ex

    @pytest.mark.parametrize("native", [True, False])
    def test_hint_examples_parse_in_our_own_fallback(self, native):
        # Self-consistency: whatever dialect we teach in a recovery hint
        # must round-trip through the fallback parser.
        ex = _tool_call_format_example(native).replace("the_tool_name", "web_search")
        tool_calls, _, reason = parse(ex)
        assert reason == ""
        assert call_names(tool_calls) == ["web_search"]
        assert args_of(tool_calls[0])["arg1"] == "value1"

    def test_dispatch_hints_use_the_helper_and_flag(self):
        src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
        assert "_tool_call_format_example(_native)" in src
        assert '"native_tools"' in src
        # The no_function_tag branch must no longer hardcode the
        # attribute dialect unconditionally.
        branch = src.split('== "no_function_tag"', 1)[1].split("elif", 1)[0]
        assert "_tool_call_format_example" in branch
