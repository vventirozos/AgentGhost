"""Tests for distill.collector TrajectoryCollector."""

import json
from pathlib import Path

import pytest

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory, ToolCall


def _sample(**overrides) -> Trajectory:
    base = dict(
        user_request="hello",
        final_response="world",
        outcome="passed",
    )
    base.update(overrides)
    return Trajectory(**base)


def test_append_creates_day_partitioned_file(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="test-sess")
    path = c.append(_sample())
    assert path is not None
    assert path.exists()
    assert path.parent.name.count("-") == 2  # YYYY-MM-DD
    assert path.name == "session-test-sess.jsonl"


def test_append_writes_one_line_per_trajectory(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="test-sess")
    c.append(_sample(user_request="a"))
    c.append(_sample(user_request="b"))
    c.append(_sample(user_request="c"))
    lines = list(tmp_path.rglob("session-*.jsonl"))
    assert len(lines) == 1
    content = lines[0].read_text().splitlines()
    assert len(content) == 3
    for line in content:
        json.loads(line)  # must parse


def test_append_redacts_secrets(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="test-sess")
    t = _sample(final_response="key=sk-liveABCDEFGHIJKL1234567")
    path = c.append(t)
    content = path.read_text()
    assert "sk-liveABCDEF" not in content


def test_append_when_disabled_returns_none(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="off", enabled=False)
    assert c.append(_sample()) is None
    assert not list(tmp_path.rglob("*.jsonl"))


def test_append_many_returns_count(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="test-sess")
    trajs = [_sample(user_request=f"t{i}") for i in range(5)]
    assert c.append_many(trajs) == 5


def test_iter_trajectories_reads_back(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="rt")
    c.append(_sample(user_request="first"))
    c.append(_sample(user_request="second"))

    c2 = TrajectoryCollector(root=tmp_path, session_id="other")
    read_back = list(c2.iter_trajectories())
    assert len(read_back) == 2
    prompts = {t.user_request for t in read_back}
    assert prompts == {"first", "second"}


def test_iter_filters_by_session(tmp_path):
    c1 = TrajectoryCollector(root=tmp_path, session_id="s1")
    c2 = TrajectoryCollector(root=tmp_path, session_id="s2")
    c1.append(_sample(user_request="one"))
    c2.append(_sample(user_request="two"))
    c_any = TrajectoryCollector(root=tmp_path, session_id="other")
    res = list(c_any.iter_trajectories(session_id="s1"))
    assert len(res) == 1
    assert res[0].user_request == "one"


def test_count_equals_append_count(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="c")
    for i in range(7):
        c.append(_sample(user_request=f"t{i}"))
    assert c.count() == 7


def test_iter_skips_malformed_lines(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="c")
    c.append(_sample(user_request="good"))
    # Manually write a broken line into the same file
    files = list(tmp_path.rglob("session-*.jsonl"))
    assert files
    with files[0].open("a") as f:
        f.write("not json at all\n")
        f.write("{\"bad\": \"missing trajectory fields\"}\n")
    c.append(_sample(user_request="still good"))

    res = list(c.iter_trajectories())
    assert len(res) == 2  # only the two good ones
    assert {t.user_request for t in res} == {"good", "still good"}


def test_append_failure_is_non_fatal(tmp_path, monkeypatch):
    c = TrajectoryCollector(root=tmp_path / "nonwritable" / "nested", session_id="c")

    # Force a write failure and confirm it's swallowed.
    def broken_mkdir(*a, **kw):
        raise PermissionError("simulated")
    monkeypatch.setattr(Path, "mkdir", broken_mkdir)
    result = c.append(_sample())
    assert result is None  # never raises; returns None


def test_iter_returns_nothing_when_root_missing(tmp_path):
    c = TrajectoryCollector(root=tmp_path / "never-created", session_id="c")
    assert list(c.iter_trajectories()) == []
    assert c.count() == 0


def test_roundtrip_preserves_tool_calls(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="c")
    t = _sample(
        tool_calls=[
            ToolCall(name="file_system", arguments={"action": "list"}, result="a\nb"),
            ToolCall(name="execute", arguments={"code": "print(1)"}, result="1"),
        ]
    )
    c.append(t)
    [rt] = list(c.iter_trajectories())
    names = [tc.name for tc in rt.tool_calls]
    assert names == ["file_system", "execute"]
