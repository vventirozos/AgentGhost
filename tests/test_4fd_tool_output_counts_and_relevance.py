"""§4FD pins for the two tool-output changes that feed the evidence gate and
remove a counting failure class.

* `recall`: every chunk carries RELEVANCE, and the header names the best
  match — an all-LOW recall says it is probably unrelated (the codename
  confabulation, trajectories 929115b8 / c8ee1d80).
* `file_system list_files`: the listing states its entry count (19 real
  turns contradicted their own listing by counting; e.g. 13798301 "45 files"
  over 34 entries).
"""
import asyncio
from pathlib import Path

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools import memory as mem


class _Mem:
    def __init__(self, scores):
        self._scores = scores

    def search_advanced(self, query, limit=10):
        return [{"score": s, "text": f"chunk {i}", "metadata": {"source": f"src{i}"}}
                for i, s in enumerate(self._scores)]


def _recall(scores):
    return asyncio.run(mem.tool_recall("project codename", memory_system=_Mem(scores)))


def test_all_low_recall_is_labelled_unrelated_in_the_header_and_per_chunk():
    """World where it fails: LOW rows are still announced as 'highly
    relevant' — the model cannot tell an unrelated hit from a good one."""
    out = _recall([1.20, 1.28, 1.33])
    assert out.startswith("SYSTEM: Found 3 memories (best match: LOW")
    assert "probably UNRELATED" in out
    assert out.count("RELEVANCE: LOW (distance 1.") == 3


def test_best_match_names_the_best_grade_not_the_first_row():
    """World where it fails: the header takes the first row's grade, or
    ranks grades alphabetically (HIGH < LOW < MEDIUM)."""
    out = _recall([1.30, 0.95, 1.25])
    assert "(best match: MEDIUM)" in out
    out2 = _recall([1.30, 0.50])
    assert "(best match: HIGH)" in out2 and "UNRELATED" not in out2


def test_rows_past_the_hard_cut_are_still_dropped():
    out = _recall([1.40, 1.50])
    assert out.startswith("SYSTEM OBSERVATION: Zero high-confidence memories found")


def test_list_files_states_the_entry_count(tmp_path):
    """World where it fails: the count line is missing, or counts the
    truncated listing instead of the total."""
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    out = asyncio.run(fs.tool_list_files(tmp_path))
    lines = out.splitlines()
    count_ix = next(i for i, l in enumerate(lines) if l.startswith("5 entries under the workspace root"))
    first_entry_ix = next(i for i, l in enumerate(lines) if l.strip().startswith("f0.txt"))
    assert count_ix < first_entry_ix
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.py").write_text("def f():\n    pass\n")
    out_sub = asyncio.run(fs.tool_list_files(tmp_path, path="sub"))
    assert any(l.startswith("1 entries under /sub") for l in out_sub.splitlines())


def test_list_files_count_uses_the_total_when_the_listing_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_LIST_MAX_ENTRIES", 3)
    for i in range(7):
        (tmp_path / f"f{i}.txt").write_text("x")
    out = asyncio.run(fs.tool_list_files(tmp_path))
    assert any(l.startswith("7 entries under") for l in out.splitlines())
    assert "4 more files NOT shown (7 total)" in out
