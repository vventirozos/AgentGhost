"""The ingest not-found message names what the action does NOT do
(§4FR, 2026-09-09).

Request d50a34bd, turn 1: the model called
    knowledge_base(action='transcribe', filename='postgresql-19-A4.pdf')
on a file it had not fetched yet, believing one call downloads and
indexes. The tool answered

    Error: File 'postgresql-19-A4.pdf' not found. Check list_files to see
    the exact name.

— the wrong advice for this case. The file was never there; listing the
sandbox cannot help. The turn cost 40 s and a strike, and the model had to
work out the download route by itself. The message now says the action
never downloads and names the route that does, in the tool's own
vocabulary (`file_system(operation='download', url=…, path=…)`).

The containment refusal (§4DX) is a DIFFERENT branch and keeps its own
wording — a refused file must never be reported as a missing one. Pinned
again here so the two messages cannot be merged back into one.
"""
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.tools.memory import tool_gain_knowledge


class _EmptyLibrary:
    """Only what the not-found path touches before it returns."""

    def get_library(self):
        return []


@pytest.fixture
def sandbox(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "present.txt").write_text("here")
    return sb


def _ingest(sandbox, filename):
    return str(asyncio.run(tool_gain_knowledge(
        filename=filename, sandbox_dir=sandbox, memory_system=_EmptyLibrary())))


def test_a_missing_file_is_told_the_action_never_downloads(sandbox):
    """THE REGRESSION, in the consumer's words: the model reads this text
    and chooses its next call from it."""
    out = _ingest(sandbox, "postgresql-19-A4.pdf")
    low = out.lower()
    assert out.startswith("Error:")
    assert "postgresql-19-A4.pdf" in out
    assert "not found" in low                          # the pins that predate this
    assert "never downloads" in low, out
    assert "file_system(operation='download'" in out, out
    assert "path='postgresql-19-A4.pdf'" in out, "the route must carry the caller's own filename"
    assert "filename='postgresql-19-A4.pdf'" in out, "…and say what to call again with"
    # …and, when the Tor download is refused, the file comes from the USER —
    # never a command that would fetch it cleartext from the sandbox
    # (review, 2026-09-09: the sandbox's egress is not Tor).
    assert "ask the user" in out.lower(), out
    assert "curl" not in out.lower() and "execute(" not in out, out


def test_list_files_is_the_last_resort_not_the_advice(sandbox):
    """The old message's only advice was "Check list_files". It is still
    mentioned — for the case where the file SHOULD exist — but after the
    download route, not instead of it."""
    out = _ingest(sandbox, "notes.pdf")
    assert out.index("file_system(operation='download'") < out.index("list_files"), out


def test_a_containment_refusal_is_still_not_a_missing_file(sandbox):
    """§4DX's pin, restated beside the new message so the two branches stay
    distinct: a refused path names the refusal and never says "not found"."""
    out = _ingest(sandbox, "../quiet.txt")
    low = out.lower()
    assert "security" in low or "refus" in low, out
    assert "not found" not in low, out
    assert "never downloads" not in low, "the download advice leaked into the refusal branch"


def test_a_present_file_does_not_get_the_message(sandbox):
    """The message is for the missing case only; a present file proceeds
    (and may fail later for other reasons, but not with THIS text)."""
    out = _ingest(sandbox, "present.txt")
    assert "never downloads" not in out.lower()


def test_a_pdf_url_keeps_its_own_route_advice(sandbox):
    """The sibling branch — a PDF URL passed as the filename — already
    named the download route. Both branches must agree on it."""
    out = _ingest(sandbox, "https://example.org/manual.pdf")
    assert "file_system(operation='download')" in out, out
    assert "ask the user" in out.lower() and "curl" not in out.lower(), out


def test_the_symlink_fallback_refusal_is_not_a_missing_file_either(sandbox, tmp_path):
    """§4DX round 2's branch: the exact name misses, the STEM match finds a
    symlink, and the link points outside the sandbox. That refusal has its
    own wording and must not collapse into the not-found message — a
    mutant that did exactly that survived the first battery because no
    pin drove this branch."""
    (tmp_path / "OUTSIDE.txt").write_text("CANARY")
    os.symlink(tmp_path / "OUTSIDE.txt", sandbox / "link.txt")
    out = _ingest(sandbox, "link")                       # stem match → the symlink
    low = out.lower()
    assert "security" in low or "refus" in low, out
    assert "not found" not in low, out
    assert "never downloads" not in low, out


def test_no_ingest_message_ever_suggests_a_shell_command(sandbox):
    """Both messages once suggested `curl` from the sandbox — cleartext from
    the host IP. Whatever the filename, no message carries a command."""
    for name in ("it's here.pdf", "plain.pdf", "https://example.org/manual.pdf"):
        out = _ingest(sandbox, name)
        assert "execute(" not in out and "curl" not in out.lower(), out
