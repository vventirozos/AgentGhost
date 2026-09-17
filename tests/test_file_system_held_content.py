"""A rejected replace holds its payload for the corrective write (2026-09-15, §4HB).

Live (req 6a7882f5): operation='replace' arrived with the whole 13.7 k-char
report as `content` and no `replace_with`; rejected — correctly, since 8 of
13 such payloads in the corpus were FRAGMENTS and promoting them would
overwrite whole files — and the model then produced the same 13.6 k chars
again as a write. 124 s, one fifth of the turn. The hold lets the model
redeem the payload BY NAME; nothing is written it did not ask to write.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import _HELD_CONTENT_TOKEN, tool_file_system

_REPORT = "# Forensic report\n\n" + ("finding line\n" * 400)


@pytest.fixture(autouse=True)
def _clear_holds():
    FS._HELD_REPLACE_CONTENT.clear()
    yield
    FS._HELD_REPLACE_CONTENT.clear()


@pytest.mark.asyncio
async def test_the_live_shape_is_held_then_redeemed_without_resending(tmp_path):
    """FAILS IF: the rejection does not hold, or the write cannot redeem.

    The exact live sequence, with the payload sent ONCE.
    """
    (tmp_path / "report.md").write_text("old report")
    rej = await tool_file_system(operation="replace", sandbox_dir=tmp_path,
                                 path="report.md", content=_REPORT)
    text = str(rej)
    assert "forgot to specify 'replace_with'" in text
    assert "HELD" in text and _HELD_CONTENT_TOKEN in text
    assert "do NOT resend" in text
    # The file was NOT touched by the rejection — the hold is not a promote.
    assert (tmp_path / "report.md").read_text() == "old report"

    out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                 path="report.md", content=_HELD_CONTENT_TOKEN)
    assert "Error" not in str(out)[:40]
    assert (tmp_path / "report.md").read_text() == _REPORT


@pytest.mark.asyncio
async def test_a_hold_is_redeemed_once(tmp_path):
    """FAILS IF: a second redeem replays stale content."""
    await tool_file_system(operation="replace", sandbox_dir=tmp_path,
                           path="a.md", content=_REPORT)
    await tool_file_system(operation="write", sandbox_dir=tmp_path,
                           path="a.md", content=_HELD_CONTENT_TOKEN)
    again = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                   path="a.md", content=_HELD_CONTENT_TOKEN)
    assert "no held content" in str(again)
    assert (tmp_path / "a.md").read_text() == _REPORT


@pytest.mark.asyncio
async def test_a_hold_is_bound_to_its_path(tmp_path):
    """FAILS IF: content held for one file can be written to another.

    The failure direction that matters: a report landing in `main.py`.
    """
    await tool_file_system(operation="replace", sandbox_dir=tmp_path,
                           path="report.md", content=_REPORT)
    out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                 path="main.py", content=_HELD_CONTENT_TOKEN)
    assert "no held content" in str(out)
    assert not (tmp_path / "main.py").exists()


@pytest.mark.asyncio
async def test_a_hold_expires(tmp_path, monkeypatch):
    """FAILS IF: a ten-minute-old payload is still redeemable."""
    await tool_file_system(operation="replace", sandbox_dir=tmp_path,
                           path="r.md", content=_REPORT)
    key = next(iter(FS._HELD_REPLACE_CONTENT))
    t, c = FS._HELD_REPLACE_CONTENT[key]
    FS._HELD_REPLACE_CONTENT[key] = (t - FS._HELD_CONTENT_TTL_S - 1, c)
    out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                 path="r.md", content=_HELD_CONTENT_TOKEN)
    assert "no held content" in str(out)


@pytest.mark.asyncio
async def test_the_py_auto_promote_path_is_unchanged(tmp_path):
    """FAILS IF: the hold intercepts the branch that already promotes.

    A complete Python module still auto-promotes to a write (the
    pre-existing behaviour) — the hold is only for what that branch rejects.
    """
    # ≥60 chars, ≥4 non-blank lines, a top-level import and a def — the
    # predicate's stated minimum (`_looks_like_complete_python_module`).
    mod = ("import os\nimport sys\n\n\ndef main():\n    print(os.getcwd())\n"
           "    return sys.argv\n\n\nif __name__ == '__main__':\n    main()\n")
    out = await tool_file_system(operation="replace", sandbox_dir=tmp_path,
                                 path="m.py", content=mod)
    assert "auto-promoted" in str(out)
    assert (tmp_path / "m.py").read_text() == mod
    assert not FS._HELD_REPLACE_CONTENT


def test_holds_are_bounded(tmp_path):
    """FAILS IF: a rejection loop keeps an unbounded number of payloads alive."""
    for i in range(FS._HELD_CONTENT_MAX + 5):
        FS._hold_rejected_content(tmp_path, f"f{i}.md", "x" * 10)
    assert len(FS._HELD_REPLACE_CONTENT) == FS._HELD_CONTENT_MAX


@pytest.mark.asyncio
async def test_a_real_write_with_the_token_as_prose_is_not_hijacked(tmp_path):
    """FAILS IF: the token is matched as a substring rather than the whole payload.

    A document that MENTIONS the token must be written verbatim.
    """
    body = f"Notes: the tool uses {_HELD_CONTENT_TOKEN} as a marker.\n"
    out = await tool_file_system(operation="write", sandbox_dir=tmp_path,
                                 path="notes.md", content=body)
    assert "no held content" not in str(out)
    assert (tmp_path / "notes.md").read_text() == body
