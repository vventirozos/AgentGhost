"""§4EC deletion pass — the invariants that made each deletion safe.

A deleted arm was, by mutation, invisible to every test; these pins do
not test the arm (it is gone) but the PROPERTY that made it redundant,
in the world where that property is load-bearing. If a later change
breaks the property, the pin — not a silent drop — is what fails.
"""
from __future__ import annotations

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.stream_guards import _tail_has_stop_marker
from tests.test_turnloop_r1_fixes import _dispatch_agent, _ts, _call


# ── F9: one collapse gate is enough because the hash carries the name ── #

def _pick_readsafe():
    # any allow-listed name will do: the tool body is the test's own
    return sorted(A._COLLAPSE_READSAFE)[0]


@pytest.mark.asyncio
async def test_identical_args_on_a_read_safe_and_an_unsafe_tool_both_execute():
    """The world the deleted lookup gate would have guarded IF the batch
    hash were name-blind: a mutating call must never be collapsed onto a
    read-safe twin with the same arguments (a dropped write is silent
    state drift)."""
    agent = _dispatch_agent()
    ran = {"safe": 0, "unsafe": 0}
    safe = _pick_readsafe()

    async def safe_tool(**kw):
        ran["safe"] += 1
        return "ok"

    async def pg(**kw):
        ran["unsafe"] += 1
        return "INSERT 0 1"
    agent.available_tools = {safe: safe_tool, "postgres_admin": pg}
    args = {"sql": "INSERT INTO t(x) VALUES (1)"}
    tc = [_call(safe, args), _call("postgres_admin", args),
          _call("postgres_admin", args)]
    for i, c in enumerate(tc):
        c["id"] = f"c{i}"
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=tc))
    assert ran == {"safe": 1, "unsafe": 2}, ran


@pytest.mark.asyncio
async def test_identical_read_safe_calls_still_collapse_to_one_execution():
    """…and the dedup itself still works with the single (registration)
    gate: two byte-identical read-safe calls run once."""
    agent = _dispatch_agent()
    ran = {"n": 0}
    safe = _pick_readsafe()

    async def safe_tool(**kw):
        ran["n"] += 1
        return "ok"
    agent.available_tools = {safe: safe_tool}
    args = {"query": "same"}
    tc = [_call(safe, args), _call(safe, args)]
    tc[0]["id"], tc[1]["id"] = "r0", "r1"
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=tc))
    assert ran["n"] == 1, ran


# ── F13: a marker at the front of the scan window is conservative ────── #

def test_marker_at_the_front_of_the_window_latches_even_behind_a_quote():
    tok = "<tool_call>"
    window = len(tok) + 16
    # the quote sits ONE char before the window, so it is invisible; the
    # marker is at idx 0 of the scanned tail and must still latch
    buf = 'say "' + tok + "x" * (window + 1 - len(tok))
    tail = buf[-(window + 1):]
    assert len(tail) == window + 1 and tail.startswith(tok)
    assert _tail_has_stop_marker(buf, tok) is True
    # …whereas the same quote INSIDE the window is a mention and does not
    buf2 = 'say "' + tok + "x" * (window - len(tok))
    assert buf2[-(window + 1):].startswith('"' + tok)
    assert _tail_has_stop_marker(buf2, tok) is False


# ── F10: the one UI scrub must reach a fixed point ───────────────────── #

def _parse(content):
    from tests.test_4ec_parser_dialects import _agent
    return _agent()._parse_assistant_tool_calls(content, {})


@pytest.mark.parametrize("content,expected", [
    # the reviewer's input: one removal splices the neighbours into a new tag
    ("hello <t<tool_call>x</tool_call>ool_call>y</tool_call> bye", "hello  bye"),
    # three splice levels — leaked even under the OLD two-pass accident
    ("a <t<t<tool_call>x</tool_call>ool_call>y</tool_call>ool_call>z</tool_call> b",
     "a  b"),
    ("<fun<function name=\"a\">z</function>ction name=\"b\">w</function> ok", "ok"),
], ids=["one_splice", "two_splices", "function_splice"])
def test_ui_scrub_reaches_a_fixed_point(content, expected):
    _tcs, ui, _reason = _parse(content)
    assert ui == expected, ui
    assert "<tool_call" not in ui.lower() and "<function" not in ui.lower()
