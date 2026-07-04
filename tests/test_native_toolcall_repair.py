"""Regression tests for native tool_call de-corruption.

Live symptom (2026-07-04): "hey ghost, tell me about yourself" made the
agent call ``introspect`` and get, every turn::

    introspect -> SYSTEM ERROR: 'action' must be one of
    ['narrative', 'recall', 'recent', 'stats', 'summary'].

even though the model's reasoning clearly passed a valid action
('summary' / 'narrative' / 'stats'). Root cause: with ``--native-tools``
on (default for Qwen 3.6+), the UPSTREAM server parses the tool-call XML
itself. On a multi-tool reply that uses the equals-format
``<function=name>`` / ``<parameter=name>`` shape, the server closed the
first ``<parameter>`` but leaked the ENTIRE serialized XML of every
following tool call into the first argument's string value::

    {"action": "summary</parameter>\\n</function>\\n</tool_call>\\n
                <tool_call>\\n<function=list_lessons>\\n<parameter=scope>\\nall",
     "limit": 10}

so ``action`` became a giant invalid string. Our own XML parser handles
this reply correctly, but native tool_calls bypass it — the server hands
us the corrupt structured call directly. ``_repair_native_tool_calls``
detects the CLOSE-then-OPEN leak signature, truncates the value at the
first framing token, and recovers the merged calls as separate
tool_calls.

The exact strings below are copied verbatim from the failing session's
trajectory (session e260ae3e).
"""
from __future__ import annotations

import json

from ghost_agent.core.agent import (
    _repair_native_tool_calls,
    _value_has_leaked_framing,
    _recover_calls_from_tail,
)
from ghost_agent.tools.introspect import _VALID_ACTIONS


AVAIL = ["introspect", "list_lessons", "manage_skills", "self_state", "execute"]


def _tc(name, args):
    return {"id": "call_x", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _args(tc):
    return json.loads(tc["function"]["arguments"])


# ── the exact live corruption ──────────────────────────────────────────

# Verbatim from trajectory session e260ae3e, turn 1.
CORRUPT_SUMMARY = ("summary</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
                   "<function=list_lessons>\n<parameter=scope>\nall")
# Verbatim from trajectory session e260ae3e, turn 2.
CORRUPT_NARRATIVE = ("narrative</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
                     "<function=introspect>\n<parameter=action>\nstats</parameter>\n"
                     "</function>\n</tool_call>\n<tool_call>\n<function=self_state>\n"
                     "<parameter=action>\nlist")


class TestLiveCorruption:
    def test_summary_leak_recovers_valid_action(self):
        out, repaired = _repair_native_tool_calls(
            [_tc("introspect", {"action": CORRUPT_SUMMARY, "limit": 10})], AVAIL)
        assert repaired is True
        # Primary introspect now carries the value the model intended.
        assert _args(out[0])["action"] == "summary"
        # And crucially, it now PASSES the introspect validator.
        assert _args(out[0])["action"] in _VALID_ACTIONS
        # The leaked second call is recovered, not dropped.
        assert out[1]["function"]["name"] == "list_lessons"
        assert _args(out[1])["scope"] == "all"

    def test_narrative_leak_recovers_three_calls(self):
        out, repaired = _repair_native_tool_calls(
            [_tc("introspect", {"action": CORRUPT_NARRATIVE})], AVAIL)
        assert repaired is True
        names = [tc["function"]["name"] for tc in out]
        assert names == ["introspect", "introspect", "self_state"]
        assert _args(out[0])["action"] == "narrative"
        assert _args(out[1])["action"] == "stats"
        assert _args(out[2])["action"] == "list"
        # every recovered introspect action is valid
        for tc in out:
            if tc["function"]["name"] == "introspect":
                assert _args(tc)["action"] in _VALID_ACTIONS

    def test_recovered_ids_are_unique(self):
        out, _ = _repair_native_tool_calls(
            [_tc("introspect", {"action": CORRUPT_NARRATIVE})], AVAIL)
        ids = [tc["id"] for tc in out]
        assert len(ids) == len(set(ids)), "recovered tool_calls must have unique ids"


# ── clean calls are never touched ──────────────────────────────────────

class TestPureFramingPrimary:
    # Live functional-hunt finding (unit 5): a file_system read where the native
    # parser dumped the ENTIRE merged reply into the first argument, so the value
    # STARTS with framing (no clean prefix). The old repair skipped it and
    # dispatched a tool call whose argument was raw tag-soup → a wasted strike.
    LEAK = ("</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=file_system>\n"
            "<parameter=operation>\nread</parameter>\n<parameter=path>\nhunt_notes.txt")

    def test_phantom_primary_dropped_and_real_call_promoted(self):
        out, repaired = _repair_native_tool_calls(
            [_tc("file_system", {"path": self.LEAK})], AVAIL + ["file_system"])
        assert repaired is True
        # Exactly one call, and it is the REAL file_system read — no tag-soup.
        assert len(out) == 1
        assert out[0]["function"]["name"] == "file_system"
        got = _args(out[0])
        assert got.get("operation") == "read"
        assert got.get("path") == "hunt_notes.txt"
        # No argument value contains leaked framing.
        assert "</parameter>" not in out[0]["function"]["arguments"]
        assert "<tool_call>" not in out[0]["function"]["arguments"]


class TestSiblingParamLeak:
    # Live functional-hunt finding (unit 7): manage_tasks stop — the SECOND
    # parameter (task_identifier) of the SAME call leaked into the first param's
    # value: `</parameter>` followed by a sibling `<parameter=…>`, not a new
    # <function>. The old repair only caught close-then-new-CALL, so it missed
    # this and dispatched `action="stop</parameter>…"` → "Unknown action" strike.
    LEAK = "stop</parameter>\n<parameter=task_identifier>\ntask_fd313c3034"

    def test_leaked_sibling_param_folded_into_same_call(self):
        out, repaired = _repair_native_tool_calls(
            [_tc("manage_tasks", {"action": self.LEAK})], AVAIL)
        assert repaired is True
        assert len(out) == 1  # no phantom extra call
        assert out[0]["function"]["name"] == "manage_tasks"
        assert _args(out[0]) == {"action": "stop", "task_identifier": "task_fd313c3034"}

    def test_sibling_param_does_not_clobber_existing_key(self):
        # If the real key already holds a value, the leaked one must not overwrite.
        tc = _tc("manage_tasks", {"action": self.LEAK, "task_identifier": "explicit_id"})
        out, repaired = _repair_native_tool_calls([tc], AVAIL)
        assert repaired is True
        assert _args(out[0])["action"] == "stop"
        assert _args(out[0])["task_identifier"] == "explicit_id"


class TestCleanPassthrough:
    def test_clean_call_unchanged(self):
        original = [_tc("manage_skills", {"action": "list"})]
        out, repaired = _repair_native_tool_calls(original, AVAIL)
        assert repaired is False
        assert out == original

    def test_valid_single_action_untouched(self):
        out, repaired = _repair_native_tool_calls(
            [_tc("introspect", {"action": "summary"})], AVAIL)
        assert repaired is False
        assert _args(out[0])["action"] == "summary"

    def test_empty_and_none(self):
        assert _repair_native_tool_calls([], AVAIL) == ([], False)
        assert _repair_native_tool_calls(None, AVAIL) == ([], False)

    def test_multiple_clean_calls_preserved_in_order(self):
        original = [_tc("introspect", {"action": "stats"}),
                    _tc("manage_skills", {"action": "list"})]
        out, repaired = _repair_native_tool_calls(original, AVAIL)
        assert repaired is False
        assert [t["function"]["name"] for t in out] == ["introspect", "manage_skills"]


# ── false-positive protection ──────────────────────────────────────────

class TestNoFalsePositives:
    def test_code_mentioning_close_tag_only(self):
        # A `</parameter>` in content with NO following tool-call opening is
        # legit text, not a leak.
        args = {"content": "x = '</parameter> is just a string'"}
        out, repaired = _repair_native_tool_calls([_tc("execute", args)], AVAIL)
        assert repaired is False
        assert _args(out[0])["content"] == args["content"]

    def test_close_then_open_of_unknown_tool_not_repaired(self):
        # CLOSE-then-OPEN present, but the opening names no known tool -> we
        # cannot positively identify a leak, so we must NOT mangle the value.
        args = {"content": "a</parameter>\n<function=totally_not_a_tool>\n<parameter=z>1"}
        out, repaired = _repair_native_tool_calls([_tc("execute", args)], AVAIL)
        assert repaired is False
        assert _args(out[0])["content"] == args["content"]

    def test_documentation_example_open_before_close(self):
        # A tool-call EXAMPLE in content is OPEN-then-CLOSE, not
        # CLOSE-then-OPEN, so it is not treated as a leak.
        args = {"content": 'Use <function name="x"><parameter name="y">z</parameter></function>'}
        out, repaired = _repair_native_tool_calls([_tc("execute", args)], AVAIL)
        assert repaired is False
        assert _args(out[0])["content"] == args["content"]


# ── input-shape robustness ─────────────────────────────────────────────

class TestInputShapes:
    def test_arguments_as_dict(self):
        tc = {"id": "c", "type": "function",
              "function": {"name": "introspect", "arguments": {"action": CORRUPT_SUMMARY}}}
        out, repaired = _repair_native_tool_calls([tc], AVAIL)
        assert repaired is True
        assert _args(out[0])["action"] == "summary"

    def test_non_json_arguments_passed_through(self):
        tc = {"id": "c", "type": "function",
              "function": {"name": "execute", "arguments": "not json at all"}}
        out, repaired = _repair_native_tool_calls([tc], AVAIL)
        assert repaired is False
        assert out[0]["function"]["arguments"] == "not json at all"

    def test_missing_function_key(self):
        tc = {"id": "c", "type": "function"}
        out, repaired = _repair_native_tool_calls([tc], AVAIL)
        assert repaired is False
        assert out == [tc]

    def test_no_available_names_still_recovers(self):
        # When available_names is None the recovery has no gate and accepts
        # any function name in the tail.
        out, repaired = _repair_native_tool_calls(
            [_tc("introspect", {"action": CORRUPT_SUMMARY, "limit": 10})], None)
        assert repaired is True
        assert _args(out[0])["action"] == "summary"
        assert out[1]["function"]["name"] == "list_lessons"


# ── unit-level helpers ─────────────────────────────────────────────────

class TestHelpers:
    def test_leak_signature_detection(self):
        assert _value_has_leaked_framing(CORRUPT_SUMMARY) is True
        assert _value_has_leaked_framing("summary") is False
        assert _value_has_leaked_framing("just </parameter> alone") is False
        assert _value_has_leaked_framing(None) is False
        assert _value_has_leaked_framing(12) is False

    def test_recover_from_tail_filters_by_available(self):
        tail = ("</parameter>\n<tool_call>\n<function=list_lessons>\n"
                "<parameter=scope>\nall</parameter>")
        got = _recover_calls_from_tail(tail, ["list_lessons"])
        assert got == [("list_lessons", {"scope": "all"})]
        # gated out when not available
        assert _recover_calls_from_tail(tail, ["introspect"]) == []
