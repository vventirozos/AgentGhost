"""Hippocampus consolidation truncation.

Production failure: the smart-memory consolidation call capped output at
max_tokens=1024, the model's JSON died mid-object (right after
`"profile_update":`), and `extract_json_from_text` dropped the whole
extraction. Two-sided fix: a larger cap + json_object response_format on
the call, and a truncated-JSON repair pass in the extractor so a cap hit
salvages every complete key/value pair instead of losing the memory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect
import re

from ghost_agent.core.agent import (
    GhostAgent,
    _repair_truncated_json,
    extract_json_from_text,
)


# ------------------------------------------------------------ repair helper

def test_repair_production_truncation_shape():
    """The exact shape observed in the production log."""
    text = ('{ "score": 0.8,   "fact": "User has a PostgreSQL table named '
            'web_order_line_options with columns id and header_id",   '
            '"profile_update":')
    res = extract_json_from_text(text, repair_truncated=True)
    assert res.get("score") == 0.8
    assert "web_order_line_options" in res.get("fact", "")
    # the dangling key is completed with null, not dropped silently
    assert "profile_update" in res and res["profile_update"] is None


def test_repair_mid_string_truncation():
    res = _repair_truncated_json('{"score": 0.8, "fact": "User has a Post')
    assert res == {"score": 0.8, "fact": "User has a Post"}


def test_repair_nested_open_structures():
    assert _repair_truncated_json('{"a": {"b": [1, 2') == {"a": {"b": [1, 2]}}


def test_repair_trailing_comma():
    assert _repair_truncated_json('{"a": 1,') == {"a": 1}


def test_repair_partial_key_dropped():
    res = _repair_truncated_json('{"a": 1, "partial_ke')
    assert res == {"a": 1}


def test_repair_garbage_returns_empty():
    assert _repair_truncated_json("hello") == {}
    assert _repair_truncated_json("{not json") == {}
    assert _repair_truncated_json("") == {}


def test_extractor_well_formed_unchanged():
    assert extract_json_from_text('{"a": 1, "b": [2]}') == {"a": 1, "b": [2]}


def test_extractor_no_json_still_empty():
    assert extract_json_from_text("no json here at all") == {}


def test_extractor_escaped_quote_in_truncated_string():
    res = _repair_truncated_json('{"fact": "user said \\"hi')
    assert res == {"fact": 'user said "hi'}


# ----------------------------------------------------- consolidation payload

def test_smart_memory_payload_has_room_and_json_mode():
    """Pin the call-site fix: a 1024 cap reliably truncated real
    consolidations (score + fact + profile_update + graph_triplets)."""
    src = inspect.getsource(GhostAgent.run_smart_memory_task)
    m = re.search(r'"max_tokens":\s*(\d+)', src)
    assert m, "smart memory payload must set an explicit max_tokens"
    assert int(m.group(1)) >= 2048
    assert "json_object" in src
    # and the extraction opts in to truncation salvage
    assert "repair_truncated=True" in src


# ── the PLANNER hits the same wall (2026-08-11, req 6e9efd6a) ──────────
#
# Same defect class as the consolidation truncation above, one subsystem
# over, and it cost 40 minutes of a live request. The planner runs at
# max_tokens=4096 with a `tree_update` that grows with the task list; SEVEN
# times in one request it hit `finish_reason: length` at exactly 4096
# completion tokens, `extract_json_from_text` returned {}, and the WHOLE plan
# was dropped — thought, tree and focus — leaving the log reading
# "No thought provided." / "Plan Updated. Focus: " with nothing naming the
# token cap as the cause. The loop then ran on unguided.
#
# ⚠ AND THE OBVIOUS FIX IS THE DANGEROUS ONE. `_repair_truncated_json`
# closes dangling braces, so a `tree_update` cut mid-object returns
# COMPLETE-LOOKING with tasks silently gone. Measured on those seven
# payloads: one repaired into a root task with ZERO children (the entire
# plan), another into a plausible 8-node tree that was still incomplete.
# Feeding either to `task_tree.load_from_json` adopts it as the plan of
# record. Only the RAW text can say whether the object actually closed.

import logging

from ghost_agent.core import agent as _ag
from ghost_agent.core.agent import (
    _complete_object_for_key, _scan_json_braces, _unclosed_braces,
    _has_bare_json_braces, salvage_truncated_plan,
)

_CUT_TREE = ('{\n  "thought": "Verify the fixes in the browser.",\n'
             '  "tree_update": {\n    "id": "root",\n'
             '    "description": "Fix Jiu Jitsu Calendar')
_INTACT_TREE = ('{\n  "thought": "Read the file.",\n'
                '  "tree_update": {"id": "root", "description": "Fix UI",\n'
                '    "children": [{"id": "task_1", "status": "READY"}]},\n'
                '  "next_action_id": "task_1",\n  "required_tool":')


def test_brace_scan_ignores_braces_INSIDE_strings():
    """A brace in a value must not register, or a complete object reads as
    truncated and a truncated one as complete."""
    assert _scan_json_braces('{"css": "a { z-index: 1 }"}') == 0
    assert _unclosed_braces('{"css": "a { z-index: 1 }"}') == 0
    assert _unclosed_braces('{"a": {"b": 1}') == 1
    assert _unclosed_braces('{"a": "unterminated { ') == 1


def test_prose_with_braces_is_not_a_malformed_JSON_WARNING(caplog):
    """⚠ THE FALSE POSITIVE. A markdown reply quoting a CSS rule logged a
    WARNING about malformed JSON — while the turn's real payload rode
    `tool_calls` and no JSON was ever intended. 2 of the 9 warnings in req
    6e9efd6a were this, and they are what sent a reader hunting a parser bug
    that did not exist."""
    prose = ("**CONFIRMED:**\n- Z-index fix applied: "
             "`#session-modal.modal-overlay { z-index: 1001 }`\n"
             "- The file still has the original function")
    assert not _has_bare_json_braces(prose)
    with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
        assert extract_json_from_text(prose) == {}
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"prose braces must not WARN: {warnings}"


def test_UNQUOTED_keys_still_warn_because_that_is_a_real_malformation(caplog):
    """⚠ THE OVER-NARROWING I SHIPPED FIRST, and the existing suite caught it.
    My initial discriminator demanded a QUOTED key — but a model emitting the
    plan as `{ thought: ..., next_action_id: ... }` is a genuine malformation,
    and it is structurally identical to the CSS snippet above. What separates
    them is the code span, not the quoting."""
    malformed = ("Here is the plan: { thought: missing-quotes, "
                 "next_action_id: !!! }")
    assert _has_bare_json_braces(malformed)
    with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
        assert extract_json_from_text(malformed) == {}
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_FENCED_json_block_that_fails_still_warns():
    """A ``` fence is an explicit claim that JSON follows — masking those
    would mute the clearest signal there is."""
    assert _has_bare_json_braces('```json\n{ bad: !!! }\n```')


def test_truncated_json_is_reported_as_TRUNCATED_not_malformed(caplog):
    """"Malformed" describes a syntax error and points at the parser. These
    were the model being cut off — the token budget is the fault, and the
    message has to say so or the next reader repeats the wrong hunt."""
    with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
        assert extract_json_from_text(_CUT_TREE) == {}
    msg = " ".join(r.getMessage() for r in caplog.records
                   if r.levelno >= logging.WARNING)
    assert "TRUNCATED" in msg and "unclosed" in msg
    assert "malformed" not in msg


def test_genuinely_malformed_json_STILL_warns(caplog):
    """⚠ OVER-SUPPRESSION GUARD. Narrowing the warning must not mute the case
    it was built for: balanced braces, a real key, and still unparseable."""
    with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
        assert extract_json_from_text('{"a": [1, 2,, ], "b" "c"}') == {}
    msg = " ".join(r.getMessage() for r in caplog.records
                   if r.levelno >= logging.WARNING)
    assert "malformed" in msg


def test_complete_object_for_key_tells_CUT_from_CLOSED():
    """The whole discriminator, on the two real shapes."""
    assert _complete_object_for_key(_CUT_TREE, "tree_update") is None
    whole = _complete_object_for_key(_INTACT_TREE, "tree_update")
    assert whole and whole["id"] == "root"
    assert len(whole["children"]) == 1
    # absent key, and a non-object value, are both "nothing to trust"
    assert _complete_object_for_key('{"thought": "x"}', "tree_update") is None
    assert _complete_object_for_key('{"tree_update": "n/a"}',
                                    "tree_update") is None


def test_salvage_keeps_the_thought_and_DROPS_a_cut_tree():
    """⚠ THE LOAD-BEARING PIN, from the live payload. The repaired dict DOES
    contain a `tree_update` — that is exactly the trap. A caller using it
    adopts a root task with no children as the plan of record."""
    salvaged, whole = salvage_truncated_plan(_CUT_TREE)
    assert salvaged.get("thought")
    assert "tree_update" in salvaged, (
        "the repair really does hand back a complete-LOOKING tree — if this "
        "ever stops being true the guard below is testing nothing")
    assert not (salvaged["tree_update"].get("children") or []), (
        "and it is the destructive shape: the task list is gone")
    assert whole is None, "a CUT tree must never be offered to the caller"


def test_salvage_KEEPS_a_tree_that_closed_before_the_cut():
    """The other side: truncation after `tree_update` closed leaves a genuinely
    intact tree, and throwing it away would be pure loss."""
    salvaged, whole = salvage_truncated_plan(_INTACT_TREE)
    assert salvaged.get("thought")
    assert whole is not None and len(whole["children"]) == 1


def test_planner_feeds_tree_update_from_the_VERIFIED_tree_only():
    """⚠ THE SEAM. The helpers can be perfect and the planner still adopt
    `plan_json['tree_update']`; nothing in a unit test of the helpers would
    catch it. Pins the DECISION, not a spelling: on the truncated path the
    tree must come from the verified value."""
    src = inspect.getsource(_ag.GhostAgent.handle_chat)
    assert "salvage_truncated_plan(plan_content)" in src
    assert re.search(r'tree_update\s*=\s*_whole_tree\s+or\s+\{\}', src), (
        "the truncated path must assign tree_update from the VERIFIED tree")
    body = src[src.find("_plan_truncated"):]
    assert 'plan_json.get("tree_update")' not in body.split("if tree_update")[0]


def test_planner_cap_is_raised_and_env_tunable():
    """2026-08-11 (operator): 4096 truncated **26.3% of every planning step**
    (44 of 167 recorded calls, all at exactly the cap).

    ⚠ The completed distribution is CENSORED at the cap — anything wanting
    more lands in the truncated bucket, not in that tail — so its p99 (3835)
    is a FLOOR on the requirement, never an estimate of it. The size comes
    from what the calls produce: ~4.5k tokens of reasoning + ~1.5k of content
    at the observed maximum."""
    assert _ag._PLANNER_MAX_TOKENS >= 8192
    src = inspect.getsource(_ag.GhostAgent.handle_chat)
    assert '"max_tokens": _PLANNER_MAX_TOKENS' in src, (
        "the planner payload must read the constant, or the cap is a literal "
        "again and the env knob is decoration")
    assert '"max_tokens": 4096' not in src.split("PLANNING_SYSTEM_PROMPT")[-1][:2000]


def test_planner_cap_floor_survives_a_junk_env_value():
    """A typo in the env must not hand the planner a cap so small that every
    call truncates — the failure this whole section is about.

    Tested through the pure parser rather than by reloading the module: a
    reload mints a second set of classes and breaks `isinstance` for every
    module already holding the first."""
    f = _ag._planner_cap_from_env
    for bad in ("0", "-1", "12", "", "lots", None):
        assert f(bad) >= 1024, bad
    assert f("16384") == 16384
    assert f(" 12000 ") == 12000
    assert f(None) == _ag._PLANNER_CAP_DEFAULT


def test_planner_no_think_is_ON_by_default_with_a_kill_switch():
    """✅ ENABLED 2026-08-11 (operator) on the measurement: parse 11 better /
    1 worse (p=0.006), truncation 6/0 (p=0.031), 3258→502 tokens and 58.9→9.9s
    on a call that runs every turn.

    DEFAULT rather than a launcher export, mirroring GHOST_VISUAL_NO_THINK: an
    env-only switch is live in production and absent in tests, ablations and
    manual restarts — the prod/dev flag drift this repo keeps paying for."""
    assert _ag._PLANNER_NO_THINK is True, "the measured-better path is default"
    src = inspect.getsource(_ag.GhostAgent.handle_chat)
    assert "/no_think" in src, "soft switch missing"
    assert '"enable_thinking": False' in src, (
        "hard switch missing — the soft one alone is not reliable")
    # both must be under the SAME gate, or one can be live while the other is not
    assert src.count("_PLANNER_NO_THINK") >= 2


def test_planner_no_think_kill_switch_actually_kills():
    """A knob nobody has ever seen turn OFF is not a knob. Pinned on the
    parse, not on the constant, so a renamed env var fails here."""
    import subprocess
    code = ("import sys; sys.path.insert(0, 'src');"
            "from ghost_agent.core.agent import _PLANNER_NO_THINK as f;"
            "print(f)")
    for val, want in (("0", "False"), ("false", "False"), ("no", "False"),
                      ("1", "True"), ("", "True")):
        env = dict(os.environ, GHOST_PLANNER_NO_THINK=val,
                   GHOST_API_KEY="test-key")
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True,
                             cwd=os.path.join(os.path.dirname(__file__), ".."))
        assert out.stdout.strip() == want, f"{val!r} -> {out.stdout!r}"
