"""Pending-action honesty guards + inline `-c` AST rescue (2026-07-14).

Follow-ups from the code-correction failure-chain session (journal
2026-07-14j). Two live losses this closes:

* A mid-repair turn finalized with "…That's what's causing the error. Let me
  fix it." — the turn ended, nothing ran, and the user believed the fix was
  applied. Likewise, `force_final_response` silently DROPPED pending
  `file_system` calls at the finish line (observed 2026-07-12, twice).
  New: `_ends_with_action_promise` backs a once-per-request act-or-admit
  steer, and `_dropped_mutation_note` appends a "NOT been applied" note when
  a mutating tool call is dropped at finalization.

* A 769-char repair one-liner with mixed quotes was SYSTEM-BLOCKED (the
  unescaped delimiter defeats the shlex auto-convert), costing a strike plus
  a ~4-step write-probe detour. New: when the RAW regex-captured body parses
  as valid Python, it is shipped via base64 (bash never sees it) instead of
  blocked.
"""
import base64
import re
import shlex
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.agent import (
    _dropped_mutation_note,
    _ends_with_action_promise,
)
from ghost_agent.tools.execute import tool_execute


# ---------------------------------------------------------------------------
# _ends_with_action_promise
# ---------------------------------------------------------------------------

def test_promise_detected_in_final_sentence():
    text = ("Found it. Line 79 has `====` — a leftover artifact. "
            "That's what's causing the error. Let me fix it.")
    assert _ends_with_action_promise(text) == "Let me fix it."


def test_promise_variants_detected():
    assert _ends_with_action_promise("Diagnosis done.\nI'll restart the service now.")
    assert _ends_with_action_promise("The bug is clear. I will patch app.js next.")
    assert _ends_with_action_promise("OK. I am going to rewrite that block.")


def test_let_me_know_is_not_a_promise():
    assert _ends_with_action_promise(
        "The fix is applied and verified. Let me know if anything else breaks.") == ""


def test_promise_mid_text_followed_by_result_is_fine():
    text = ("Let me fix it. I replaced the broken line and the test passes. "
            "Everything is green.")
    assert _ends_with_action_promise(text) == ""


def test_long_final_sentence_is_ignored():
    text = "Done. " + "I'll now describe the architecture " + "x" * 120 + "."
    assert _ends_with_action_promise(text) == ""


def test_empty_and_none_are_fine():
    assert _ends_with_action_promise("") == ""
    assert _ends_with_action_promise(None) == ""


# ---------------------------------------------------------------------------
# _dropped_mutation_note
# ---------------------------------------------------------------------------

def test_dropped_file_system_gets_note():
    note = _dropped_mutation_note(["file_system"])
    assert "NOT been applied" in note
    assert "file_system" in note


def test_dropped_terminal_tool_stays_silent():
    assert _dropped_mutation_note(["self_play"]) == ""
    assert _dropped_mutation_note([]) == ""
    assert _dropped_mutation_note(None) == ""


def test_dropped_mixed_names_sorted_and_deduped():
    note = _dropped_mutation_note(["self_play", "execute", "file_system", "execute"])
    assert "execute, file_system" in note
    assert "self_play" not in note


# ---------------------------------------------------------------------------
# Inline `-c` AST rescue (tool_execute)
# ---------------------------------------------------------------------------

def _mock_mgr():
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    return mgr


def _ran_command(mgr) -> str:
    return mgr.execute.call_args[0][0]


def _decode_transported_body(ran: str) -> str:
    m = re.search(r'printf %s (.+?) \| base64 -d', ran)
    assert m, f"no base64 transport found in: {ran}"
    b64 = shlex.split(m.group(1))[0]
    return base64.b64decode(b64).decode("utf-8")


# A body with BOTH quote types where the double-quote delimiter appears
# unescaped — `_quote_safe` is False so the shlex path can't run. Valid
# Python, so the AST rescue must auto-convert instead of blocking. Padded
# past the 120-char trigger to mirror the live 769-char repair script.
_MIXED_QUOTE_BODY = (
    "import os\n"
    "d = 'x' * 3\n"
    'print("ok" + d)\n'
    "paths = [p for p in os.listdir('.') if p.endswith('.py')]\n"
    'print("count:", len(paths))'
)


@pytest.mark.asyncio
async def test_mixed_quote_valid_python_rescued(tmp_path):
    assert len(_MIXED_QUOTE_BODY) >= 120
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{_MIXED_QUOTE_BODY}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = _ran_command(mgr)
    assert "base64 -d" in ran
    assert _decode_transported_body(ran) == _MIXED_QUOTE_BODY


@pytest.mark.asyncio
async def test_mixed_quote_invalid_python_still_blocked(tmp_path):
    body = _MIXED_QUOTE_BODY + "\nprint('unclosed"
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{body}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" in result
    mgr.execute.assert_not_called()


@pytest.mark.asyncio
async def test_skill_wrap_not_rescued(tmp_path):
    body = ("from my_skill import my_skill\n"
            'print("run:", my_skill(\'arg\'))\n' + "pad = 1\n" * 12)
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{body}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" in result
    assert "my_skill" in result
    mgr.execute.assert_not_called()


@pytest.mark.asyncio
async def test_trailing_pipe_not_rescued(tmp_path):
    mgr = _mock_mgr()
    result = await tool_execute(
        command=f'python3 -c "{_MIXED_QUOTE_BODY}" | head -5',
        sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" in result
    mgr.execute.assert_not_called()


@pytest.mark.asyncio
async def test_bash_body_not_rescued(tmp_path):
    # bash has no host-side parse — mixed-quote bash bodies stay blocked.
    body = 'for f in *.py; do echo "$f" && grep -c \'def\' "$f"; done; echo done; echo again'
    mgr = _mock_mgr()
    result = await tool_execute(command=f"bash -c '{body}'",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" in result
    mgr.execute.assert_not_called()
