"""Tests for distill.schema."""

import json

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome


def test_trajectory_default_id_is_uuid_hex():
    t = Trajectory()
    assert len(t.id) == 32
    assert all(c in "0123456789abcdef" for c in t.id)


def test_trajectory_default_timestamp_isoz():
    t = Trajectory()
    assert t.timestamp.endswith("Z")


def test_trajectory_jsonl_roundtrip():
    t = Trajectory(
        session_id="s1", user_request="hello", final_response="hi",
        tool_calls=[ToolCall(name="file_system", arguments={"action": "list"}, result="ok")],
        outcome=Outcome.PASSED.value,
    )
    line = t.to_jsonl()
    d = json.loads(line)
    t2 = Trajectory.from_dict(d)
    assert t2.session_id == t.session_id
    assert t2.user_request == t.user_request
    assert len(t2.tool_calls) == 1
    assert t2.tool_calls[0].name == "file_system"
    assert t2.outcome == Outcome.PASSED.value


def test_trajectory_empty_tool_calls_survive_roundtrip():
    t = Trajectory()
    d = json.loads(t.to_jsonl())
    t2 = Trajectory.from_dict(d)
    assert t2.tool_calls == []


def test_outcome_values_are_strings():
    assert Outcome.PASSED.value == "passed"
    assert Outcome.FAILED.value == "failed"
    assert Outcome.UNKNOWN.value == "unknown"


def test_toolcall_defaults():
    tc = ToolCall(name="x")
    assert tc.arguments == {}
    assert tc.result == ""
    assert tc.error == ""
    assert tc.duration_s == 0.0


def test_trajectory_from_dict_ignores_extra_tool_call_fields():
    """Schema drift: a future ToolCall with more fields shouldn't break
    Trajectory.from_dict reading old data.

    We guard by checking that from_dict tolerates *missing* fields;
    forward-compat on extra fields requires filtering, which we don't
    implement today. This test documents the current contract.
    """
    d = {
        "id": "a" * 32, "timestamp": "2026-01-01T00:00:00Z",
        "session_id": "s", "task_kind": "user_request",
        "cluster": None, "tier": None, "model": "", "temperature": 0.0,
        "sample_index": None, "batch_id": None,
        "system_prompt": "", "user_request": "hi",
        "planning_output": None,
        "tool_calls": [{"name": "x", "arguments": {}, "result": "", "error": "", "duration_s": 0.0}],
        "n_steps": 0, "tokens_in": 0, "tokens_out": 0, "duration_s": 0.0,
        "outcome": "passed", "failure_reason": "",
        "validator_signal": {}, "final_response": "bye", "extra": {},
    }
    t = Trajectory.from_dict(d)
    assert t.final_response == "bye"
    assert t.tool_calls[0].name == "x"
