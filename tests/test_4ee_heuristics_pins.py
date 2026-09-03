"""§4EE pins for `distill/outcome_heuristics.py` and the corrections
sidecar — survivors of the label-producer battery, each with the world
where the line decides (R4)."""
from __future__ import annotations

import json

import pytest

from ghost_agent.distill import outcome_heuristics as H
from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.distill.collector import TrajectoryCollector


def _traj(tool_calls, final_response="done", user_request="do it", outcome="unknown"):
    return Trajectory(session_id="s", user_request=user_request,
                      tool_calls=tool_calls, final_response=final_response,
                      outcome=outcome)


# ── the repeated-error key is a NORMALISED error text ──────────────────── #

def test_repeated_error_counter_keys_on_the_normalised_text():
    variants = ["Error: ENOENT  no such file", "error: enoent no such file",
                "  ERROR:   ENOENT no  such file  "]
    tcs = [ToolCall(name="execute", result=v) for v in variants]
    v = H.classify_chat_outcome(_traj(tcs))
    assert v.outcome == Outcome.FAILED.value and "same error 3×" in v.reason, v


def test_normalise_tool_error_handles_non_strings_and_whitespace():
    assert H._normalize_tool_error(None) == ""
    assert H._normalize_tool_error(12) == ""
    assert H._normalize_tool_error("  Error:  A   B  ") == H._normalize_tool_error("error: a b")
    assert H._normalize_tool_error("x" * 400) == "x" * 200


# ── structural cause label ─────────────────────────────────────────────── #

def test_structural_cause_squashes_whitespace_and_counts_the_rest():
    one = _traj([ToolCall(name="execute", error="boom\n\n  now")])
    assert H.structural_cause_for_trajectory(one) == "execute: boom now"
    bare = _traj([ToolCall(name="execute", error="x", result="")])
    # a flagged failure with no usable text: the NAME alone, no dangling colon
    bare.tool_calls[0].error = ""
    bare.tool_calls[0].result = "Error:"
    label = H.structural_cause_for_trajectory(bare)
    assert label.startswith("execute") and not label.endswith(": "), label
    two = _traj([ToolCall(name="execute", error="a"), ToolCall(name="web", error="b")])
    assert H.structural_cause_for_trajectory(two) == "execute: a (+1 more)"
    assert H.structural_cause_for_trajectory(_traj([ToolCall(name="ok", result="fine")])) == ""


# ── the unresolved predicate and the text sniffer ──────────────────────── #

def test_unresolved_needs_both_halves_of_the_swarm_sentence():
    assert H.is_unresolved_tool_result("3 still running in the background, they were NOT cancelled") is True
    assert H.is_unresolved_tool_result("3 still running in the background") is False
    assert H.is_unresolved_tool_result("they were NOT cancelled") is False
    assert H.is_unresolved_tool_result("plain success") is False


def test_sniffer_says_False_for_non_strings():
    assert H._looks_like_tool_error(None) is False
    assert H._looks_like_tool_error({"error": "x"}) is False
    assert H.looks_like_tool_error(["Error: x"]) is False


# ── instructed literal + acknowledgement ───────────────────────────────── #

def test_instructed_literal_bounds():
    req = "just reply with exactly the word: NOPE"
    assert H._reply_is_instructed_literal("NOPE", req) is True
    assert H._reply_is_instructed_literal("", req) is False
    long = "N" * (H._INSTRUCTED_LITERAL_MAX_CHARS + 1)
    assert H._reply_is_instructed_literal(long, f"reply with exactly {long}") is False
    assert H._reply_is_instructed_literal("NOPE", "") is False


def test_empty_or_blank_reply_acknowledges_nothing():
    assert H.response_acknowledges_failure("") is False
    assert H.response_acknowledges_failure("   \n") is False
    assert H.response_acknowledges_failure("the file was not found") is True


# ── tool_failure_flags over both shapes ─────────────────────────────────── #

def test_flags_skip_unresolved_and_keep_scanning():
    tools = [{"name": "swarm", "content": "2 still running in the background, they were NOT cancelled"},
             {"name": "execute", "content": "Error: boom"},
             {"name": "read", "content": "ok"}]
    assert H.tool_failure_flags(tools) == [True, False]


def test_flags_read_corpus_objects_by_their_error_flag():
    tcs = [ToolCall(name="execute", error="structured failure", result="weird text"),
           ToolCall(name="read", result="fine")]
    assert H.tool_failure_flags(tcs) == [True, False]


def test_flags_stringify_a_non_string_content():
    tools = [{"name": "x", "content": {"k": "Error: boom"}}]
    # the dict's str form starts with "{'k': 'Error: boom'}" — the sniffer
    # sees `error:` in the head, exactly as it would in the operator stream
    assert H.tool_failure_flags(tools) == [True]
    assert H.tool_failure_flags([{"name": "x", "content": None}]) == [False]


def test_precomputed_flags_bypass_the_tools_argument():
    assert H.unacknowledged_total_failure(tool_failures=[True, True],
                                          final_response="42") is True
    assert H.unacknowledged_total_failure(tool_failures=[True, False],
                                          final_response="42") is False
    assert H.unacknowledged_total_failure(tool_failures=[], final_response="42") is False


def test_trajectory_shape_rule_honours_the_instructed_literal():
    tcs = [ToolCall(name="execute", error="boom"), ToolCall(name="execute", error="boom")]
    plain = _traj(tcs, final_response="NOPE", user_request="what is it?")
    assert H.unacknowledged_total_failure_for_trajectory(plain) is True
    licensed = _traj(tcs, final_response="NOPE",
                     user_request="just reply with exactly the word: NOPE")
    assert H.unacknowledged_total_failure_for_trajectory(licensed) is False


# ── classify_chat_outcome: the browser thrash window ───────────────────── #

def _browser(selector=None, op="click", result="ok", actions=None):
    args = {"operation": op}
    if selector is not None:
        args["selector"] = selector
    if actions is not None:
        args["actions"] = actions
    return ToolCall(name="browser", arguments=args, result=result)


def test_a_non_browser_call_before_the_thrash_does_not_end_the_scan():
    tcs = [ToolCall(name="execute", result="ok")] + [_browser("#go")] * 4
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.FAILED.value


def test_a_successful_navigate_restarts_the_window_and_the_scan_continues():
    tcs = [_browser("#go")] * 3 + [_browser(op="navigate")] + [_browser("#go")] * 3
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.UNKNOWN.value
    tcs = [_browser("#go")] * 2 + [_browser(op="navigate")] + [_browser("#go")] * 4
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.FAILED.value


def test_only_a_goto_action_inside_interact_resets_the_window():
    click = [{"action": "click", "selector": "#go"}]
    tcs = [_browser(op="interact", actions=click)] * 4
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.FAILED.value
    goto = [{"action": "goto", "url": "x"}]
    tcs = ([_browser(op="interact", actions=click)] * 3
           + [_browser(op="interact", actions=goto)]
           + [_browser(op="interact", actions=click)] * 3)
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.UNKNOWN.value


def test_empty_selectors_are_not_counted_as_a_thrash():
    tcs = [_browser("")] * 5 + [_browser(op="interact", actions=[{"action": "click", "selector": ""}])] * 5
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.UNKNOWN.value


def test_the_worst_selector_is_a_running_maximum():
    tcs = [_browser("#a")] * 4 + [_browser("#b")]
    v = H.classify_chat_outcome(_traj(tcs))
    assert v.outcome == Outcome.FAILED.value and "'#a'" in v.reason and "4×" in v.reason, v


def test_string_arguments_do_not_crash_the_classifier():
    tc = ToolCall(name="browser", arguments={"_raw": "not json"}, result="ok")
    tc.arguments = "not a dict"           # a corrupted legacy row
    assert H.classify_chat_outcome(_traj([tc] * 4)).outcome == Outcome.UNKNOWN.value


def test_sequence_aborted_on_a_browser_call_fails_the_turn():
    tcs = [ToolCall(name="execute", result="SEQUENCE ABORTED (not a browser)"),
           _browser(op="interact", result="SEQUENCE ABORTED: goto failed",
                    actions=[{"action": "goto", "url": "x"}])]
    v = H.classify_chat_outcome(_traj(tcs))
    assert v.outcome == Outcome.FAILED.value and "sequence aborted" in v.reason


def test_no_tool_calls_is_never_a_failure():
    assert H.classify_chat_outcome(_traj([])).outcome == Outcome.UNKNOWN.value


# ── corrections sidecar: the cache is dropped with the file ────────────── #

def test_corrections_cache_is_dropped_when_the_sidecar_disappears(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    assert col.update_outcome("t1", "failed", reason="r", source="user_correction") is True
    assert col.latest_correction("t1")["outcome"] == "failed"
    col._corrections_path().unlink()
    assert col.latest_correction("t1") is None
    assert col._load_corrections() == {}


# ── boundaries the battery named ───────────────────────────────────────── #

def test_structural_reason_and_cause_are_capped_at_120():
    r = H.structural_reason("c" * 300)
    assert r == f"{H.STRUCTURAL_FAILURE_REASON}: " + "c" * 120
    assert H.is_structural_reason(r)
    one = _traj([ToolCall(name="execute", error="e" * 300)])
    assert len(H.structural_cause_for_trajectory(one)) == 120


def test_sniffer_head_is_the_first_120_chars():
    assert H._looks_like_tool_error("x" * 114 + "error: late") is True      # ends at 120
    assert H._looks_like_tool_error("x" * 115 + "error: late") is False     # the colon is char 121


def test_instructed_literal_max_is_inclusive():
    lit = "N" * H._INSTRUCTED_LITERAL_MAX_CHARS
    assert H._reply_is_instructed_literal(lit, f"reply with exactly {lit}") is True


def test_selector_maximum_starts_at_zero():
    tcs = [ToolCall(name="execute", result="ok")]
    v = H.classify_chat_outcome(_traj(tcs), repeated_selector_threshold=1)
    assert v.outcome == Outcome.UNKNOWN.value, v


def test_a_successful_call_before_the_repeats_does_not_end_the_count():
    tcs = [ToolCall(name="execute", result="ok")] + \
          [ToolCall(name="execute", result="Error: same") for _ in range(3)]
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.FAILED.value


def test_shape_rule_on_a_corrupt_trajectory_is_False_not_True():
    t = _traj([], final_response="42")
    t.tool_calls = 5                   # not iterable — the sniffer raises inside
    assert H.unacknowledged_total_failure_for_trajectory(t) is False
    assert H.unacknowledged_total_failure_for_trajectory(None) is False


def test_only_browser_calls_feed_the_selector_window():
    tcs = [ToolCall(name="execute", arguments={"selector": "#a"}, result="ok")] * 5
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.UNKNOWN.value


def test_a_failed_navigate_does_not_reset_the_window():
    tcs = [_browser("#go")] * 3 + [_browser(op="navigate", result="Error: timeout")] + [_browser("#go")]
    assert H.classify_chat_outcome(_traj(tcs)).outcome == Outcome.FAILED.value


def test_sequence_aborted_counts_only_on_browser_calls_and_any_of_them():
    only_exec = [ToolCall(name="execute", result="SEQUENCE ABORTED: not a browser")]
    assert H.classify_chat_outcome(_traj(only_exec)).outcome == Outcome.UNKNOWN.value
    first_then_ok = [_browser(op="interact", result="SEQUENCE ABORTED: goto failed",
                              actions=[{"action": "goto", "url": "x"}]),
                     _browser("#later", result="ok")]
    v = H.classify_chat_outcome(_traj(first_then_ok))
    assert v.outcome == Outcome.FAILED.value and "sequence aborted" in v.reason, v


# ── collector: roots and the disabled path ─────────────────────────────── #

def test_default_root_follows_GHOST_HOME(monkeypatch, tmp_path):
    from ghost_agent.distill import collector as C
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
    assert C._default_root() == tmp_path / "home" / "system" / "trajectories"
    monkeypatch.delenv("GHOST_HOME", raising=False)
    assert C._default_root().name == "trajectories" and ".ghost" in str(C._default_root())
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home2"))
    assert TrajectoryCollector().root == C._default_root()


def test_append_many_counts_only_what_was_written(tmp_path):
    col = TrajectoryCollector(root=tmp_path, enabled=False)
    trajs = [Trajectory(session_id="s", user_request="q", final_response="a") for _ in range(3)]
    assert col.append_many(trajs) == 0
    live = TrajectoryCollector(root=tmp_path)
    assert live.append_many(trajs) == 3


# ── collector: the human-label disk scan and the overlay's fields ──────── #

def _sidecar(col, *recs):
    p = col._corrections_path(); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def test_disk_scan_finds_a_human_record_for_THIS_id_past_junk_and_others(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    assert col._scan_for_human_label("t") is False           # no sidecar yet
    _sidecar(col, "", "{not json", {"trajectory_id": "other", "outcome": "passed",
                                    "source": "human_feedback:api"},
             {"trajectory_id": "t", "outcome": "failed", "source": "verifier_late"})
    assert col._scan_for_human_label("t") is False
    _sidecar(col, {"trajectory_id": "t", "outcome": "passed", "source": "human_feedback:slack"})
    assert col._scan_for_human_label("t") is True
    assert col.has_human_label("t") is True and col.has_human_label("other") is True
    assert col.has_human_label("nobody") is False


def test_overlay_carries_the_correction_source_and_a_prior_human_flag(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    t = Trajectory(session_id="s", user_request="q", final_response="a", outcome="unknown")
    col.append(t)
    assert col.update_outcome(t.id, "passed", source="human_feedback:api") is True
    assert col.update_outcome(t.id, "failed", reason="late", source="verifier_late") is True
    row = next(x for x in col.iter_trajectories() if x.id == t.id)
    assert row.outcome == "failed" and row.extra["outcome_source"] == "verifier_late"
    assert row.extra["outcome_source_at"] and row.extra["human_labeled"] is True


def test_overlay_ignores_a_correction_with_an_empty_outcome(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    t = Trajectory(session_id="s", user_request="q", final_response="a", outcome="passed")
    col.append(t)
    _sidecar(col, {"trajectory_id": t.id, "outcome": "", "reason": "x", "source": "operator"})
    row = next(x for x in col.iter_trajectories() if x.id == t.id)
    assert row.outcome == "passed"


def test_update_outcome_on_an_unwritable_root_is_False_and_unreadable_sidecar_is_empty(tmp_path):
    stray = tmp_path / "afile"; stray.write_text("x")
    col = TrajectoryCollector(root=stray)                     # a FILE, not a dir
    assert col.update_outcome("t", "failed", source="user_correction") is False
    col2 = TrajectoryCollector(root=tmp_path / "ok")
    assert col2.update_outcome("t", "failed", source="user_correction") is True
    import os
    p = col2._corrections_path(); os.chmod(p, 0)
    try:
        out = col2._load_corrections()
        assert out == {} and out is not None
    finally:
        os.chmod(p, 0o644)


def test_count_skips_stray_files_and_blank_lines(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    (tmp_path / "stray.txt").write_text("not a day dir")
    day = tmp_path / "2026-09-03"; day.mkdir()
    (day / "session-x.jsonl").write_text('{"a":1}\n\n   \n{"b":2}\n')
    assert col.count() == 2
    assert isinstance(col.redaction, type(TrajectoryCollector(root=tmp_path / "z").redaction))
    from ghost_agent.distill.redact import RedactionConfig
    assert isinstance(col.redaction, RedactionConfig)


# ── collector overlay fields and the bad-line skip ─────────────────────── #

def test_overlay_adds_source_fields_only_when_the_correction_carries_them(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    t = Trajectory(session_id="s", user_request="q", final_response="a", outcome="unknown")
    col.append(t)
    # a correction with an EMPTY source and no timestamp must not stamp keys
    _sidecar(col, {"trajectory_id": t.id, "outcome": "failed", "reason": "r", "source": ""})
    row = next(x for x in col.iter_trajectories() if x.id == t.id)
    assert row.outcome == "failed"
    assert "outcome_source" not in row.extra and "outcome_source_at" not in row.extra
    # a source but no timestamp: the source is stamped, the timestamp is not
    col2 = TrajectoryCollector(root=tmp_path / "b")
    t2 = Trajectory(session_id="s", user_request="q", final_response="a")
    col2.append(t2)
    _sidecar(col2, {"trajectory_id": t2.id, "outcome": "failed", "source": "operator"})
    row2 = next(x for x in col2.iter_trajectories() if x.id == t2.id)
    assert row2.extra["outcome_source"] == "operator" and "outcome_source_at" not in row2.extra


def test_iter_keeps_reading_good_rows_after_a_bad_json_line(tmp_path):
    col = TrajectoryCollector(root=tmp_path)
    good = [Trajectory(session_id="s", user_request=f"q{i}", final_response="a") for i in range(2)]
    for t in good:
        col.append(t)
    # splice a corrupt line between the two good ones
    import json as _json
    day = sorted(p for p in col.root.iterdir() if p.is_dir())[-1]
    f = sorted(day.glob("session-*.jsonl"))[0]
    lines = f.read_text().splitlines()
    f.write_text(lines[0] + "\n{ this is not json\n" + lines[1] + "\n")
    ids = {t.id for t in col.iter_trajectories()}
    assert all(g.id in ids for g in good), "a bad line dropped a good row"
