"""Unit tests for the autobiographical memory writer (proposal item #1)."""

import json
from pathlib import Path

import pytest

from ghost_agent.selfhood.autobiographical import (
    AutobiographicalMemory,
    summarise_turn_first_person,
)
from ghost_agent.selfhood.schema import Experience


def test_summarise_passed_with_tools():
    s = summarise_turn_first_person(
        user_request="count lines in /etc/hosts",
        tool_names=["execute"],
        outcome="passed",
        final_response="There are 12 lines.",
    )
    assert s.startswith('I worked on "count lines in /etc/hosts"')
    assert "reached for execute" in s
    assert "landed" in s


def test_summarise_failed_with_reason():
    s = summarise_turn_first_person(
        user_request="parse the malformed JSON",
        tool_names=["execute", "file_system"],
        outcome="failed",
        final_response="",
        failure_reason="JSONDecodeError on line 3",
    )
    assert "didn't land" in s
    assert "JSONDecodeError" in s


def test_summarise_unknown_outcome_no_tools():
    s = summarise_turn_first_person(
        user_request="what do you think about consciousness?",
        tool_names=[],
        outcome="unknown",
        final_response="A long answer.",
    )
    assert "reasoned through it without tools" in s
    assert "without a verdict either way" in s


def test_summarise_truncates_long_requests():
    long_req = "x" * 500
    s = summarise_turn_first_person(
        user_request=long_req,
        tool_names=[],
        outcome="passed",
        final_response="ok",
    )
    assert "…" in s
    assert len(s) < 400


def test_summarise_pluralises_three_tools():
    s = summarise_turn_first_person(
        user_request="do everything",
        tool_names=["a", "b", "c"],
        outcome="passed",
        final_response="ok",
    )
    assert "strung together" in s
    assert "a, b and c" in s


def test_autobio_append_and_iter(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    e1 = Experience(summary="I tried thing one.", outcome="passed")
    e2 = Experience(summary="I tried thing two.", outcome="failed")
    assert mem.append(e1) is not None
    assert mem.append(e2) is not None

    out = list(mem.iter_experiences())
    assert len(out) == 2
    assert out[0].summary == "I tried thing one."
    assert out[1].outcome == "failed"


def test_autobio_refuses_empty_summary(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    e = Experience(summary="")
    assert mem.append(e) is None
    assert mem.count() == 0


def test_autobio_disabled_is_noop(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path, enabled=False)
    e = Experience(summary="I did a thing.")
    assert mem.append(e) is None
    assert mem.count() == 0


def test_autobio_recent_returns_newest_last(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    for i in range(5):
        mem.append(Experience(summary=f"turn {i}"))
    recent = mem.recent(limit=3)
    assert len(recent) == 3
    assert recent[0].summary == "turn 2"
    assert recent[-1].summary == "turn 4"


def test_autobio_recent_with_zero_limit(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="hi"))
    assert mem.recent(limit=0) == []


def test_autobio_search_by_keyword(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I parsed log files for nginx.", user_first_words="parse logs"))
    mem.append(Experience(summary="I baked a cake.", user_first_words="cake recipe"))
    mem.append(Experience(summary="I parsed JSON for an API.", user_first_words="api"))

    hits = mem.search_my_past("nginx logs", limit=5)
    assert len(hits) >= 1
    assert any("nginx" in h.summary for h in hits)

    hits2 = mem.search_my_past("cake", limit=5)
    assert len(hits2) == 1
    assert "cake" in hits2[0].summary


def test_autobio_search_empty_query_returns_empty(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="I did something."))
    assert mem.search_my_past("", limit=5) == []
    assert mem.search_my_past("ab", limit=5) == []  # tokens length <= 2 filtered


def test_autobio_jsonl_format_per_line(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="A"))
    mem.append(Experience(summary="B"))
    raw = mem.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2
    for line in raw:
        d = json.loads(line)
        assert d["subject"] == "self"


def test_autobio_skips_malformed_lines(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="good"))
    with mem.path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write('{"summary": "second good"}\n')
    out = list(mem.iter_experiences())
    summaries = [e.summary for e in out]
    assert "good" in summaries
    assert "second good" in summaries
    assert "not json" not in summaries


def test_autobio_count_handles_missing_file(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    assert mem.count() == 0


def test_autobio_get_by_ids(tmp_path: Path):
    mem = AutobiographicalMemory(tmp_path)
    e1 = Experience(summary="first thing I did")
    e2 = Experience(summary="second thing I did")
    mem.append(e1)
    mem.append(e2)
    got = mem.get_by_ids([e2.id, "no-such-id"])
    assert [e.id for e in got] == [e2.id]
    assert mem.get_by_ids([]) == []
    assert mem.get_by_ids(["missing"]) == []


def test_summary_writer_never_raises_on_disk_error(tmp_path: Path):
    """The autobio writer's contract is: a failure must NEVER raise.
    Exercise the failure path by pointing at a path under a file (not
    a directory) — mkdir on the parent will raise, the append should
    swallow it and return None."""
    blocker = tmp_path / "i_am_a_file"
    blocker.write_text("not a directory", encoding="utf-8")
    mem = AutobiographicalMemory(blocker / "subdir")  # parent is a file
    result = mem.append(Experience(summary="will fail"))
    assert result is None
