"""Bounded autobiographical.jsonl — compaction + tail reads (#17).

The diary log grew monotonically while every turn did 3-4 full-file O(n) parses
plus one O(n) rewrite — quadratic over the agent's lifetime. It now compacts to
the newest N entries once it passes a byte cap (rolling the dropped ones into a
summary record), and `recent()` reads only the tail instead of the whole file.
"""
import json

import pytest

from ghost_agent.selfhood import autobiographical as AB
from ghost_agent.selfhood.autobiographical import AutobiographicalMemory
from ghost_agent.selfhood.schema import Experience


def test_recent_reads_only_the_tail(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    for i in range(50):
        mem.append(Experience(summary=f"entry {i}"))
    got = mem.recent(3)
    assert [e.summary for e in got] == ["entry 47", "entry 48", "entry 49"]


def test_recent_empty_and_zero(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    assert mem.recent(5) == []          # no file yet
    mem.append(Experience(summary="x"))
    assert mem.recent(0) == []          # zero limit


def test_compaction_keeps_newest_and_writes_summary(tmp_path, monkeypatch):
    # Tighten the caps so the test compacts cheaply.
    monkeypatch.setattr(AB, "_COMPACT_MAX_BYTES", 2000)
    monkeypatch.setattr(AB, "_COMPACT_KEEP_LINES", 10)
    mem = AutobiographicalMemory(tmp_path)

    for i in range(200):
        mem.append(Experience(summary=f"diary line number {i} with some padding text"))

    lines = mem.path.read_text().strip().splitlines()
    # Newest 10 + one summary head record (compaction may run more than once,
    # but the file must be bounded well under the original 200).
    assert len(lines) <= AB._COMPACT_KEEP_LINES + 1
    # The most recent entry survived.
    parsed = [json.loads(l) for l in lines]
    assert any("number 199" in p["summary"] for p in parsed)
    # A consolidation summary record is present at the head after a drop.
    assert any("Consolidated" in p["summary"] for p in parsed)


def test_compaction_invalidates_search_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(AB, "_COMPACT_MAX_BYTES", 2000)
    monkeypatch.setattr(AB, "_COMPACT_KEEP_LINES", 5)
    mem = AutobiographicalMemory(tmp_path)
    mem.append(Experience(summary="findable trapdoor keyword here"))
    # Prime the search cache.
    mem.search_my_past("trapdoor")
    assert mem._search_cache  # populated
    # Grow past the cap → compaction runs and clears the cache.
    for i in range(200):
        mem.append(Experience(summary=f"noise entry {i} padded padded padded"))
    assert mem._search_cache == {}


def test_no_compaction_below_cap(tmp_path):
    mem = AutobiographicalMemory(tmp_path)
    for i in range(20):
        mem.append(Experience(summary=f"entry {i}"))
    lines = mem.path.read_text().strip().splitlines()
    assert len(lines) == 20  # nothing dropped, no summary record
    assert not any("Consolidated" in l for l in lines)
