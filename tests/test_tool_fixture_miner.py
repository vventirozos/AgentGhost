"""§4F Phase 2b — tool-choice fixture miner (optim/tool_fixtures.py).

Pins the audited 2026-08-01 mining contract: era filter in LOCAL time,
structured-tool_calls-only choice records, ground truth joined from
trajectories via iter_trajectories() (corrections overlay applied) on
request_id == Trajectory.session_id, honest-failure exclusion, the
"EXIT CODE: 0 is success" gotcha, and the request_id-hash tier split.
"""
import json
from pathlib import Path

import pytest

from ghost_agent.optim.tool_fixtures import (
    DEFAULT_ERA_CUTOFF_LOCAL,
    ToolChoiceFixture,
    era_cutoff_utc,
    is_choice_record,
    load_fixtures_jsonl,
    mine_fixtures,
    write_fixtures_jsonl,
)
from ghost_agent.optim.trainset import holdout_tier

# Safely inside/outside the era for ANY host timezone (cutoff local
# 2026-07-31T19:15 -> UTC somewhere in [07-31T05:15Z, 08-01T07:15Z]).
TS_OK = "2026-08-01T09:00:00Z"
TS_PRE = "2026-07-30T00:00:00Z"


def _tools(*names):
    return [{"type": "function",
             "function": {"name": n, "description": f"{n} desc",
                          "parameters": {"type": "object", "properties": {}}}}
            for n in names]


def _record(*, ts=TS_OK, request_id="req00001", session_id="123-abc",
            ordinal=1, kind="chat_completion_stream", tools=("file_system",),
            tool_calls=(("file_system", '{"operation": "read"}'),),
            messages=None):
    message = {"role": "assistant", "content": ""}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"t{i}", "type": "function",
             "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(tool_calls)]
    payload = {"model": "m", "messages": messages or [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
    ]}
    if tools:
        payload["tools"] = _tools(*tools)
    return {
        "ts": ts, "session_id": session_id, "ordinal": ordinal,
        "request_id": request_id, "kind": kind, "fingerprint": "f" * 16,
        "payload": payload,
        "response": {"object": "chat.completion.stream-reassembled",
                     "choices": [{"message": message, "finish_reason": "tool_calls"}]},
        "meta": {},
    }


def _write_day_file(tmp_path, records, name="2026-08-01.jsonl"):
    day = tmp_path / "recordings"
    day.mkdir(exist_ok=True)
    path = day / name
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _write_trajectory(tmp_path, *, traj_id="traj1", session_id="req00001",
                      outcome="passed", tool_calls=None, user_request="do the thing",
                      failure_reason=""):
    root = tmp_path / "trajectories"
    day_dir = root / "2026-08-01"
    day_dir.mkdir(parents=True, exist_ok=True)
    traj = {
        "id": traj_id, "timestamp": TS_OK, "session_id": session_id,
        "task_kind": "user_request", "user_request": user_request,
        "tool_calls": tool_calls if tool_calls is not None else [
            {"name": "file_system", "arguments": {}, "result": "ok", "error": ""}],
        "outcome": outcome, "failure_reason": failure_reason,
        "final_response": "done",
    }
    with open(day_dir / "session-test.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(traj) + "\n")
    return root


def _write_correction(root, *, traj_id, outcome, reason="verifier refuted"):
    with open(root / "corrections.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "trajectory_id": traj_id, "outcome": outcome, "reason": reason,
            "source": "verifier_late", "timestamp": TS_OK}) + "\n")


def _mine(tmp_path):
    day = tmp_path / "recordings"
    return mine_fixtures(sorted(day.glob("*.jsonl")), tmp_path / "trajectories")


class TestEraFilter:
    def test_cutoff_is_local_time_converted_to_utc(self, monkeypatch):
        # Pin the actual conversion, not just awareness: under a fixed
        # +03:00 zone the local cutoff must land 3h EARLIER in UTC. The
        # naive-bug (`replace(tzinfo=utc)`) would return 19:15Z and fail.
        import time as _time
        monkeypatch.setenv("TZ", "Etc/GMT-3")  # POSIX sign: GMT-3 == UTC+3
        _time.tzset()
        try:
            cut = era_cutoff_utc("2026-07-31T19:15")
            assert cut.utcoffset().total_seconds() == 0
            assert (cut.year, cut.month, cut.day) == (2026, 7, 31)
            assert (cut.hour, cut.minute) == (16, 15)
        finally:
            monkeypatch.delenv("TZ", raising=False)
            _time.tzset()

    def test_pre_era_records_dropped(self, tmp_path):
        _write_day_file(tmp_path, [_record(ts=TS_PRE)])
        _write_trajectory(tmp_path)
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["pre_era"] == 1


class TestChoiceDetection:
    def test_requires_tools_and_structured_tool_calls(self):
        assert is_choice_record(_record())
        assert not is_choice_record(_record(tools=()))
        assert not is_choice_record(_record(tool_calls=()))
        assert not is_choice_record(_record(kind="route"))

    def test_non_choice_records_counted_not_mined(self, tmp_path):
        _write_day_file(tmp_path, [_record(tool_calls=())])
        _write_trajectory(tmp_path)
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["no_choice"] == 1


class TestLabels:
    def test_clean_passed_is_positive(self, tmp_path):
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path, outcome="passed")
        fixtures, stats = _mine(tmp_path)
        assert len(fixtures) == 1
        assert fixtures[0].label == 1.0
        assert fixtures[0].user_request == "do the thing"
        assert stats["positive"] == 1

    def test_exit_code_zero_is_success_not_failure(self, tmp_path):
        # The pinned gotcha: a bare "EXIT CODE" substring must NOT read as
        # a failed tool call — only a non-zero code does.
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path, outcome="passed", tool_calls=[
            {"name": "execute", "arguments": {},
             "result": "ran fine\nEXIT CODE: 0", "error": ""}])
        fixtures, stats = _mine(tmp_path)
        assert len(fixtures) == 1
        assert fixtures[0].label == 1.0
        assert stats["honest_failure_excluded"] == 0

    def test_failed_is_negative(self, tmp_path):
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path, outcome="failed",
                          failure_reason="verifier refuted")
        fixtures, stats = _mine(tmp_path)
        assert len(fixtures) == 1
        assert fixtures[0].label == 0.0
        assert fixtures[0].failure_reason == "verifier refuted"
        assert stats["negative"] == 1

    def test_honest_failure_excluded(self, tmp_path):
        # PASSED turn with a structurally failed tool call = honest failure:
        # the choice signal is ambiguous and never enters the corpus.
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path, outcome="passed", tool_calls=[
            {"name": "execute", "arguments": {},
             "result": "boom\nEXIT CODE: 1", "error": ""}])
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["honest_failure_excluded"] == 1

    def test_unknown_outcome_excluded(self, tmp_path):
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path, outcome="unknown")
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["unlabeled_outcome"] == 1

    def test_unjoined_record_dropped(self, tmp_path):
        _write_day_file(tmp_path, [_record(request_id="orphan99")])
        _write_trajectory(tmp_path, session_id="req00001")
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["unjoined"] == 1


class TestCorrectionsOverlay:
    def test_verifier_late_correction_flips_label(self, tmp_path):
        # Half the ground-truth labels only exist in the sidecar — the miner
        # MUST read through iter_trajectories(), never the day-files raw.
        _write_day_file(tmp_path, [_record()])
        root = _write_trajectory(tmp_path, traj_id="t-flip", outcome="passed")
        _write_correction(root, traj_id="t-flip", outcome="failed")
        fixtures, _ = _mine(tmp_path)
        assert len(fixtures) == 1
        assert fixtures[0].label == 0.0
        assert fixtures[0].outcome == "failed"


class TestTierSplit:
    def test_tier_shared_per_request_and_deterministic(self, tmp_path):
        recs = [_record(ordinal=1), _record(ordinal=3)]
        _write_day_file(tmp_path, recs)
        _write_trajectory(tmp_path)
        fixtures, _ = _mine(tmp_path)
        assert len(fixtures) == 2
        expected = holdout_tier("toolfx:req00001", private_pct=30)
        assert {f.tier for f in fixtures} == {expected}


class TestResultPairing:
    def test_marker_extraction_strips_the_injected_state_block(self, tmp_path):
        # Live shape: the volatile <system_state_update> block is PREPENDED
        # to the very user message that carries the tool result. The
        # preview must start at the <tool_response marker — a head slice
        # would capture the agent's own injected state (79% of previews in
        # the first live mine).
        follow = _record(
            ordinal=2, tool_calls=(),
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "do the thing"},
                {"role": "assistant", "content": "calling tool"},
                {"role": "user", "content": (
                    "<system_state_update>\n### SKILL PLAYBOOK: secrets\n"
                    "</system_state_update>\n\n"
                    "<tool_response tool='file_system'>file contents here"
                    "</tool_response>")},
            ])
        _write_day_file(tmp_path, [_record(ordinal=1), follow])
        _write_trajectory(tmp_path)
        fixtures, stats = _mine(tmp_path)
        choice = next(f for f in fixtures if f.source["ordinal"] == 1)
        assert len(choice.tool_results) == 1
        assert choice.tool_results[0].startswith("<tool_response")
        assert "SKILL PLAYBOOK" not in choice.tool_results[0]
        assert stats["paired_results"] == 1

    def test_same_request_summarizer_at_next_ordinal_not_mispaired(self, tmp_path):
        # A same-request background call (context-shield summarizer: one
        # bare user message, no marker) can occupy ordinal+1. It must
        # yield NO preview — a lost pair, never a mispair.
        summarizer = _record(
            ordinal=2, kind="chat_completion", tools=(), tool_calls=(),
            messages=[{"role": "user",
                       "content": "Summarize this tool output: ..."}])
        _write_day_file(tmp_path, [_record(ordinal=1), summarizer])
        _write_trajectory(tmp_path)
        fixtures, stats = _mine(tmp_path)
        choice = next(f for f in fixtures if f.source["ordinal"] == 1)
        assert choice.tool_results == []
        assert stats["paired_results"] == 0

    def test_next_ordinal_from_other_request_not_paired(self, tmp_path):
        stranger = _record(ordinal=2, request_id="otherreq", tool_calls=())
        _write_day_file(tmp_path, [_record(ordinal=1), stranger])
        _write_trajectory(tmp_path)
        fixtures, _ = _mine(tmp_path)
        choice = next(f for f in fixtures if f.source["ordinal"] == 1)
        assert choice.tool_results == []


class TestTolerance:
    def test_system_sentinel_request_ids_excluded(self, tmp_path):
        # request_id_context defaults to "SYSTEM" for idle/background work
        # — over 80% of live records. They have no per-request trajectory;
        # joining them would let one background trajectory label the whole
        # bucket.
        _write_day_file(tmp_path, [_record(request_id="SYSTEM")])
        _write_trajectory(tmp_path, session_id="SYSTEM")
        fixtures, stats = _mine(tmp_path)
        assert fixtures == []
        assert stats["no_request_context"] == 1

    def test_missing_ts_counted_separately_from_pre_era(self, tmp_path):
        rec = _record()
        del rec["ts"]
        _write_day_file(tmp_path, [rec])
        _write_trajectory(tmp_path)
        _, stats = _mine(tmp_path)
        assert stats["bad_ts"] == 1
        assert stats["pre_era"] == 0

    def test_structurally_odd_records_do_not_abort_the_mine(self, tmp_path):
        odd = [
            {"ts": TS_OK, "kind": "route", "payload": {"messages": []},
             "response": "worker", "session_id": "s", "ordinal": 7,
             "request_id": "r"},
            _record() | {"response": {"choices": "not-a-list"}},
            _record() | {"payload": "not-a-dict"},
        ]
        _write_day_file(tmp_path, odd + [_record()])
        _write_trajectory(tmp_path)
        fixtures, _ = _mine(tmp_path)
        # The healthy record still mines despite the garbage before it.
        assert len(fixtures) == 1

    def test_load_skips_field_incomplete_lines(self, tmp_path):
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path)
        fixtures, _ = _mine(tmp_path)
        out = tmp_path / "fixtures.jsonl"
        write_fixtures_jsonl(fixtures, out)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write('{"label": 1.0}\n')      # missing required fields
            fh.write("not json at all\n")
        assert len(load_fixtures_jsonl(out)) == 1


class TestRoundTrip:
    def test_write_then_load(self, tmp_path):
        _write_day_file(tmp_path, [_record()])
        _write_trajectory(tmp_path)
        fixtures, _ = _mine(tmp_path)
        out = tmp_path / "out" / "fixtures.jsonl"
        assert write_fixtures_jsonl(fixtures, out) == 1
        loaded = load_fixtures_jsonl(out)
        assert len(loaded) == 1
        assert isinstance(loaded[0], ToolChoiceFixture)
        assert loaded[0].to_dict() == fixtures[0].to_dict()
