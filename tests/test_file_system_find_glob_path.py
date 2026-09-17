"""`find path="*.md"` means pattern="*.md" (2026-09-15, §4HC).

All 7 corpus rejections for a missing `pattern` had the glob in `path`. A
literal directory is never spelled with `*` or `?`, so the intent is
unambiguous and the rejection was a strike for nothing (live: req
c16679f1, turn 1, "Strike 1/6 (fatal)").
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.tools import file_system as FS


@pytest.fixture(autouse=True)
def _find_stub(monkeypatch):
    """Capture what `tool_find_files` is asked to do; it needs no sandbox."""
    calls = []

    async def _fake(pattern, sandbox_manager, directory, sandbox_dir=None):
        calls.append((pattern, directory))
        return f"FOUND {pattern} in {directory}"

    monkeypatch.setattr(FS, "tool_find_files", _fake)
    return calls


@pytest.mark.asyncio
async def test_a_glob_in_path_becomes_the_pattern(tmp_path, _find_stub):
    """FAILS IF: the live call is still rejected."""
    out = await FS.tool_file_system(operation="find", sandbox_dir=tmp_path, path="*.md")
    assert "MANDATORY" not in str(out)
    assert _find_stub == [("*.md", ".")]


@pytest.mark.asyncio
async def test_an_explicit_pattern_still_wins(tmp_path, _find_stub):
    """FAILS IF: the tolerant read overrides a real pattern + directory."""
    await FS.tool_file_system(operation="find", sandbox_dir=tmp_path, path="docs", pattern="*.md")
    assert _find_stub == [("*.md", "docs")]


@pytest.mark.asyncio
async def test_a_plain_directory_without_a_pattern_is_still_rejected(tmp_path, _find_stub):
    """FAILS IF: the tolerance widens to any path — 'docs' is not a glob and
    guessing a pattern for it would be an invention."""
    out = await FS.tool_file_system(operation="find", sandbox_dir=tmp_path, path="docs")
    assert "MANDATORY" in str(out)
    assert _find_stub == []


@pytest.mark.asyncio
async def test_question_mark_globs_count_too(tmp_path, _find_stub):
    await FS.tool_file_system(operation="find", sandbox_dir=tmp_path, path="report_?.pdf")
    assert _find_stub == [("report_?.pdf", ".")]
