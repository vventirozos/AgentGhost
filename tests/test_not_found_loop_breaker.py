"""Tests for the missing-file read loop-breaker.

Regression target: a fresh "create a new Minecraft project" request looped
for many turns reading `minecraft-clone/index.html` — which existed only in
a DIFFERENT (prior) project. Stale workspace-narrative context asserted the
file existed, the scoped file_system said "not found", and nothing
reconciled the contradiction. Two reinforcing defects:

  C. The not-found message was a bare ``Error: '<f>' not found.`` with no
     exit — so the model kept retrying the same path.
  B. The structural strike count decays by 1 on any successful tool turn,
     so the "failed read → sandbox-listing succeeds → failed read" oscillation
     never tripped the strike cap and burned all 40 turns. A repeated-
     identical-failure detector now freezes that decay and redirects.
"""
import pytest

from ghost_agent.core.agent import _note_repeated_failure
from ghost_agent.tools.file_system import (
    _missing_file_message,
    tool_read_file,
    tool_inspect_file,
)


# ── B: repeated-identical-failure detection ──────────────────────────
def test_identical_failures_become_persistent_at_threshold():
    sigs = {}
    err = "Error: 'minecraft-clone/index.html' not found."
    r1 = _note_repeated_failure(sigs, "file_system", err)
    r2 = _note_repeated_failure(sigs, "file_system", err)
    r3 = _note_repeated_failure(sigs, "file_system", err)
    assert (r1[1], r1[2]) == (1, False)
    assert (r2[1], r2[2]) == (2, False)
    assert (r3[1], r3[2]) == (3, True)   # 3rd identical → persistent loop


def test_signature_is_whitespace_stable():
    sigs = {}
    s1, _, _ = _note_repeated_failure(sigs, "file_system", "Error:  'x'   not found.")
    s2, _, _ = _note_repeated_failure(sigs, "file_system", "Error: 'x' not found.")
    assert s1 == s2  # whitespace-normalised → same signature, counts to 2
    assert sigs[s1] == 2


def test_distinct_failures_tracked_separately():
    sigs = {}
    _note_repeated_failure(sigs, "file_system", "Error: 'a' not found.")
    _note_repeated_failure(sigs, "file_system", "Error: 'b' not found.")
    _, _, persist = _note_repeated_failure(sigs, "execute", "Error: 'a' not found.")
    # three different signatures → none has hit the threshold
    assert persist is False
    assert len(sigs) == 3


def test_custom_threshold():
    sigs = {}
    err = "boom"
    assert _note_repeated_failure(sigs, "t", err, threshold=2)[2] is False
    assert _note_repeated_failure(sigs, "t", err, threshold=2)[2] is True


# ── C: loop-breaking not-found message ───────────────────────────────
def test_missing_message_empty_sandbox(tmp_path):
    msg = _missing_file_message("minecraft-clone/index.html", tmp_path)
    assert "does not exist" in msg
    assert "EMPTY" in msg
    assert "AUTHORITATIVE" in msg
    assert "DIFFERENT project" in msg
    assert "operation='write'" in msg
    assert "minecraft-clone/index.html" in msg


def test_missing_message_lists_existing_files(tmp_path):
    (tmp_path / "real.html").write_text("<html></html>")
    (tmp_path / "app.js").write_text("//")
    msg = _missing_file_message("ghost.html", tmp_path)
    assert "real.html" in msg and "app.js" in msg
    assert "EMPTY" not in msg


@pytest.mark.asyncio
async def test_read_missing_returns_loop_breaker_not_bare_error(tmp_path):
    res = await tool_read_file("minecraft-clone/index.html", tmp_path)
    # must NOT be the old bare message that caused the loop
    assert res != "Error: 'minecraft-clone/index.html' not found."
    assert "Do NOT read this path again" in res
    assert "AUTHORITATIVE" in res


@pytest.mark.asyncio
async def test_inspect_missing_returns_loop_breaker(tmp_path):
    res = await tool_inspect_file("nope.txt", tmp_path)
    assert "AUTHORITATIVE" in res
    assert "Do NOT read this path again" in res


@pytest.mark.asyncio
async def test_read_existing_file_still_works(tmp_path):
    (tmp_path / "ok.txt").write_text("hello world")
    res = await tool_read_file("ok.txt", tmp_path)
    assert "hello world" in res
