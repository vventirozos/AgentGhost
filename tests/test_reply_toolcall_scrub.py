"""Unparsed tool-call markup never reaches the user as prose (§4FS, 2026-09-09).

10 of 2,185 delivered replies since July carried a literal `<tool_call>`
block — most recently the 2026-09-08 schema comparison, whose reply
ended in `<tool_call><function=execute><parameter=command>python3 -c "…`.
A call whose body failed to parse was delivered verbatim, and the file or
command it described never ran. The parser side is chased by the CDATA
hints; the delivery side had no guard at all. Both excerpts below are the
real delivered text.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.reply_smoothing import (
    UNPARSED_TOOL_CALL_NOTE, smooth_reply, strip_unparsed_tool_calls)

LEAK_5E9B9320 = """I'll compare the two schema dump files. Let me write a script to parse and diff the DDL properly.

The initial diff conflated tables and indexes. Let me refine the analysis to produce an accurate report.
<tool_call>
<function=execute>
<parameter=command>
python3 -c "
import re, difflib

def read(p):
    with open(p) as f: return f.read()

dev = read('cerebro_dev.schema.sql')
prod = read('cerebro.schema.sql')
print('DEV lines:', dev.count(chr(10)))
"
</parameter>
</function>
</tool_call>"""

LEAK_FFBCDA9F = """Building `bell-bearings.html` with two checklists (one per bearing type), each with its own progress bar, matching the working `talismans.html` pattern using the shared `EldenStore`.
<tool_call>
<function=file_system>
<parameter=operation>
write
</parameter>
<parameter=path>
projects/feeb0941bd8b/bell-bearings.html
</parameter>
<parameter=content>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bell Bearings</title>
</head>
</parameter>
</function>
</tool_call>

The page is registered in the project manifest."""


def test_the_schema_compare_leak_is_removed_and_declared():
    out = strip_unparsed_tool_calls(LEAK_5E9B9320)
    assert "<tool_call>" not in out and "<function=" not in out and "</parameter>" not in out, out
    assert "python3 -c" not in out
    assert out.count(UNPARSED_TOOL_CALL_NOTE) == 1
    assert "The initial diff conflated tables and indexes." in out


def test_the_html_write_leak_keeps_the_prose_around_it():
    out = strip_unparsed_tool_calls(LEAK_FFBCDA9F)
    assert "<!DOCTYPE html>" not in out and "<tool_call>" not in out
    assert out.startswith("Building `bell-bearings.html`")
    assert "The page is registered in the project manifest." in out
    assert out.count(UNPARSED_TOOL_CALL_NOTE) == 1, "one note, however many blocks"


def test_an_unclosed_block_is_removed_to_the_end():
    """A truncated call (budget ran out mid-write) has no closing tag."""
    text = "Writing the file now.\n\n<tool_call>\n<function=file_system>\n<parameter=content>\nhalf a file"
    out = strip_unparsed_tool_calls(text)
    assert "half a file" not in out and "<tool_call>" not in out
    assert out.startswith("Writing the file now.")
    assert UNPARSED_TOOL_CALL_NOTE in out


def test_two_blocks_one_note():
    text = ("Step one.\n\n<tool_call>\n<function=a>\n</function>\n</tool_call>\n\n"
            "Step two.\n\n<tool_call>\n<function=b>\n</function>\n</tool_call>")
    out = strip_unparsed_tool_calls(text)
    assert out.count(UNPARSED_TOOL_CALL_NOTE) == 1
    assert "Step one." in out and "Step two." in out


def test_a_reply_that_is_only_markup_becomes_the_note():
    out = strip_unparsed_tool_calls("<tool_call>\n<function=execute>\n</function>\n</tool_call>")
    assert out.strip() == UNPARSED_TOOL_CALL_NOTE


def test_fenced_documentation_about_tool_calls_is_content():
    """A fence is atomic here as everywhere else in the module: the model
    explaining the call format to the user is legitimate prose."""
    text = ("The call format is:\n\n```\n<tool_call>\n<function=execute>\n</function>\n</tool_call>\n```\n\n"
            "That is all.")
    assert strip_unparsed_tool_calls(text) == text


def test_a_clean_reply_is_untouched():
    text = "All done.\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert strip_unparsed_tool_calls(text) == text


def test_case_insensitive_and_attribute_bearing_tags():
    text = "Ok.\n\n<TOOL_CALL name=\"x\">\n<Function=execute>\n</Function>\n</TOOL_CALL>\n\nDone."
    out = strip_unparsed_tool_calls(text)
    assert "TOOL_CALL" not in out and "Function=" not in out


def test_the_scrub_composes_with_smoothing():
    """The leaked reply also opened with narration; after the scrub the
    smoother still sees a multi-paragraph reply and does its own job."""
    out = smooth_reply(strip_unparsed_tool_calls(LEAK_5E9B9320))
    assert "<tool_call>" not in out
    assert UNPARSED_TOOL_CALL_NOTE in out
    assert out.startswith("The initial diff conflated"), "the smoother did not do its own job"


def test_a_bare_function_fragment_without_the_wrapper_is_removed():
    """Some leaks carry only the inner tag (the model dropped the wrapper);
    the second pattern exists for exactly that. World where it fails: only
    wrapped blocks are scrubbed and the fragment is delivered."""
    text = "Saving it now.\n\n<function=file_system>\n<parameter=path>\nx.txt\n</parameter>\n</function>\n\nDone."
    out = strip_unparsed_tool_calls(text)
    assert "<function=" not in out and "</function>" not in out and "<parameter=" not in out, out
    assert UNPARSED_TOOL_CALL_NOTE in out and "Done." in out


# --- the delivery site, EXECUTED ----------------------------------------
# A first version pinned source ORDER here and passed while the feature was
# dead: an older scrub upstream had already removed the markup, so the new
# scrub saw nothing and the note never reached the user (review, 2026-09-09).
# `make_fin_agent` / `_fs` drive `_finalize_and_return` for real.

import asyncio
import pytest
from tests.test_finalize_stream_pins import make_fin_agent, _fs


def _finalize(text, tools=()):
    a = make_fin_agent()
    out, _, _ = asyncio.run(a._finalize_and_return(_fs(final_ai_content=text, tools_run_this_turn=list(tools))))
    return out


def test_executed_the_delivered_reply_says_the_call_did_not_run():
    out = _finalize(LEAK_5E9B9320, tools=[])              # zero tools: outside every gate
    assert "<tool_call>" not in out and "<function=" not in out, out
    assert out.count(UNPARSED_TOOL_CALL_NOTE) == 1, out
    assert "The initial diff conflated tables and indexes." in out


def test_executed_the_watchdogs_replan_marker_is_not_a_leak():
    """The cognitive watchdog appends its own synthetic replan call to the
    durable text on purpose; it must not be reported as a failed call."""
    text = "Here is the analysis so far.\n</think>\n<tool_call>\n<function=replan>\n<parameter=reason>\nloop\n</parameter>\n</function>\n</tool_call>"
    out = _finalize(text, tools=[])
    assert UNPARSED_TOOL_CALL_NOTE not in out, out


def test_executed_inline_code_about_the_syntax_is_content():
    text = "The parser expects `<tool_call>` to wrap `<function=execute>` calls.\n\nThat is the whole format, and here is the long explanation you asked for."
    out = _finalize(text, tools=[])
    assert "The parser expects `<tool_call>`" in out, out
    assert UNPARSED_TOOL_CALL_NOTE not in out


def test_executed_a_tool_response_echo_is_not_a_failed_call():
    text = "Result:\n\n<tool_response>\nOK, 3 rows updated\n</tool_response>\n\nAll good."
    out = _finalize(text, tools=[])
    assert UNPARSED_TOOL_CALL_NOTE not in out, out
    assert "All good." in out


# --- the hardened scrub, at the unit --------------------------------------

def test_an_unbalanced_fence_does_not_shield_a_leak():
    text = "Config:\n\n```yaml\nkey: value\n\nNow the call:\n<tool_call>\n<function=execute>\n<parameter=command>rm -rf /tmp/x</parameter>\n</function>\n</tool_call>"
    out = strip_unparsed_tool_calls(text)
    assert "<tool_call>" not in out and "rm -rf" not in out, out
    assert UNPARSED_TOOL_CALL_NOTE in out


def test_inline_code_mentions_are_left_alone():
    text = "The parser expects `<tool_call>` to wrap `<function=execute>` calls.\n\nThat is the whole format."
    assert strip_unparsed_tool_calls(text) == text


def test_the_attribute_dialect_is_removed_too():
    text = "Hi.\n\n<function name=\"file_system\">\n<parameter name=\"path\">x</parameter>\n</function>\n\nBye."
    out = strip_unparsed_tool_calls(text)
    assert "<function" not in out and "Bye." in out and UNPARSED_TOOL_CALL_NOTE in out


def test_a_tool_response_block_is_not_a_call():
    from ghost_agent.core.reply_smoothing import unparsed_call_markup_present
    text = "<tool_response>\nOK 3 rows\n</tool_response>"
    assert not unparsed_call_markup_present(text)
    assert strip_unparsed_tool_calls(text) == text


def test_the_replan_marker_is_preserved_in_the_record():
    from ghost_agent.core.reply_smoothing import unparsed_call_markup_present
    text = "text\n</think>\n<tool_call>\n<function=replan>\n<parameter=reason>\nx\n</parameter>\n</function>\n</tool_call>"
    assert not unparsed_call_markup_present(text)
    assert strip_unparsed_tool_calls(text) == text


def test_whitespace_is_collapsed_only_at_the_seam():
    """The user's own spacing elsewhere is content."""
    text = "Poem:\n\n\n\nline one\n\n\n\nline two\n\n<tool_call>\n<function=x>\n</function>\n</tool_call>"
    out = strip_unparsed_tool_calls(text)
    assert out.startswith("Poem:\n\n\n\nline one\n\n\n\nline two"), out
