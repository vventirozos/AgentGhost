"""Line-ranged read (IMPROVEMENTS.md #11).

After a failed replace or a too-large-file error, the model's only recovery
used to be re-reading the WHOLE file (up to ~90k tokens) or improvising a `sed`
via execute (a sandbox exec + probe + turn). `operation='read'` now takes
`start_line`/`end_line`: a bounded, line-number-prefixed slice, exempt from the
whole-file size cap, streamed so cost tracks the range not the file size.
"""
from pathlib import Path

import pytest

from ghost_agent.tools.file_system import (
    tool_read_file, tool_file_system, _read_line_range, _LINE_RANGE_MAX_LINES,
)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "big.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 501)) + "\n")
    return tmp_path


async def test_reads_requested_range_with_line_numbers(sandbox):
    out = await tool_read_file("big.py", sandbox, start_line=200, end_line=203)
    assert "lines 200-203" in out
    assert "200\tline 200" in out
    assert "203\tline 203" in out
    assert "line 204" not in out
    assert "line 199" not in out


async def test_range_exempt_from_size_cap(sandbox):
    # A whole-file read at a tiny budget would be refused; a range must not be.
    big = sandbox / "huge.txt"
    big.write_text("\n".join(f"row {i}" for i in range(1, 20001)) + "\n")
    whole = await tool_read_file("huge.txt", sandbox, max_context=2000)
    assert "too large" in whole.lower()
    ranged = await tool_read_file("huge.txt", sandbox, max_context=2000,
                                  start_line=10000, end_line=10005)
    assert "10000\trow 10000" in ranged
    assert "too large" not in ranged.lower()


async def test_end_defaults_to_capped_span(sandbox):
    out = await tool_read_file("big.py", sandbox, start_line=1)
    # Only 500 lines exist, well under the cap → all shown, no cap notice.
    assert "lines 1-500" in out
    assert "cap" not in out.lower()


async def test_span_cap_truncates_and_points_forward(sandbox):
    huge = sandbox / "huge.txt"
    huge.write_text("\n".join(f"row {i}" for i in range(1, 5001)) + "\n")
    out = await tool_read_file("huge.txt", sandbox, start_line=1, end_line=4000)
    assert f"lines 1-{_LINE_RANGE_MAX_LINES}" in out
    assert f"start_line={_LINE_RANGE_MAX_LINES + 1}" in out


async def test_start_past_eof_reports_actual_length(sandbox):
    out = await tool_read_file("big.py", sandbox, start_line=9000, end_line=9010)
    assert "past the end" in out
    assert "500 lines" in out


async def test_end_before_start_rejected(sandbox):
    out = await tool_read_file("big.py", sandbox, start_line=50, end_line=10)
    assert "before start_line" in out


async def test_non_integer_bounds_rejected(sandbox):
    out = await tool_read_file("big.py", sandbox, start_line="abc")
    assert "integers" in out


async def test_dispatch_threads_aliases(sandbox):
    # The router accepts common aliases and forwards them to the range read.
    out = await tool_file_system(operation="read", sandbox_dir=sandbox,
                                 path="big.py", start=100, end=102)
    assert "100\tline 100" in out and "102\tline 102" in out


async def test_dispatch_whole_file_when_no_range(sandbox):
    out = await tool_file_system(operation="read", sandbox_dir=sandbox, path="big.py")
    assert "CONTENTS" in out
    assert "line 1" in out and "line 500" in out


def test_read_line_range_empty_file(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    out = _read_line_range(tmp_path / "empty.txt", "empty.txt", 1, 10)
    assert "empty" in out.lower()
