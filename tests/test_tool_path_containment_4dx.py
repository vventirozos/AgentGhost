"""§4DX — the tool layer's path-containment boundary.

`knowledge_base(action='ingest_document'|'transcribe', filename=...)` resolved
its LOCAL-file argument with a bare

    clean_name = str(filename).lstrip("/")
    file_path = sandbox_dir / clean_name

That contains an ABSOLUTE path (`/etc/passwd` -> `<sandbox>/etc/passwd`) but not
a RELATIVE one. `../../.ghost_api_key` resolved straight out of the sandbox, and
the tool then READ the file and embedded its contents into durable vector
memory — retrievable by `recall` forever — while returning "SUCCESS". Verified
end-to-end through the real tool on 2026-08-30.

Reachable from prompt injection: fetched web/darkweb content enters the model's
context and the model's next tool call is the payload.

The tests are split in two on purpose:
  * `_get_safe_path` is fuzzed directly — it is the helper 19 call sites rely
    on, so a regression there is a regression everywhere at once;
  * the TOOL is driven end-to-end — a helper that is correct but not CALLED
    protects nothing, which is exactly the bug this section is about.
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

from ghost_agent.tools.file_system import _get_safe_path
from ghost_agent.tools.memory import tool_gain_knowledge


# ── the shared helper ─────────────────────────────────────────────────────

ESCAPE_PAYLOADS = [
    "../OUTSIDE.txt",
    "../../etc/passwd",
    "foo/../../OUTSIDE.txt",
    "./././../OUTSIDE.txt",
    "sub/./../../OUTSIDE.txt",
    "/workspace/../OUTSIDE.txt",
    "workspace/../../OUTSIDE.txt",
    "sandbox/../../OUTSIDE.txt",
]

CONTAINED_BUT_ODD = [
    "/etc/passwd",            # absolute -> re-rooted under the sandbox
    "....//OUTSIDE.txt",      # not a traversal, just a weird dir name
    "%2e%2e/OUTSIDE.txt",     # encoded dots are NOT decoded here
    "..%2fOUTSIDE.txt",
    "..\\OUTSIDE.txt",        # backslash is not a separator on POSIX
    "~/.ghost_api_key",       # `~` is a literal directory name, not $HOME
    "$HOME/.ghost_api_key",
]


@pytest.fixture
def sandbox(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    (sb / "inside.txt").write_text("LEGITIMATE-SANDBOX-CONTENT")
    (tmp_path / "OUTSIDE.txt").write_text("CANARY-CONTENT-OUTSIDE")
    return sb


@pytest.mark.parametrize("payload", ESCAPE_PAYLOADS)
def test_get_safe_path_refuses_every_escape(payload, sandbox):
    with pytest.raises(ValueError):
        _get_safe_path(sandbox, payload)


@pytest.mark.parametrize("payload", CONTAINED_BUT_ODD)
def test_get_safe_path_contains_the_odd_spellings(payload, sandbox):
    """These are NOT rejected — they are re-rooted inside the sandbox. Pinned
    because the distinction is the whole design: the helper refuses what
    escapes and contains what merely looks strange, so tightening it into a
    blanket rejection would break `/workspace/...` paths the model writes
    constantly."""
    resolved = _get_safe_path(sandbox, payload).resolve()
    assert str(resolved).startswith(str(sandbox.resolve())), resolved


def test_get_safe_path_still_resolves_an_ordinary_name(sandbox):
    """The counterweight. A containment helper that refused everything would
    pass every test above."""
    assert _get_safe_path(sandbox, "inside.txt").resolve() == \
        (sandbox / "inside.txt").resolve()


# ── the tool, driven ──────────────────────────────────────────────────────

class _RecordingMemory:
    """Captures whatever the ingest path hands to vector memory."""

    def __init__(self):
        self.writes = []

    def __getattr__(self, name):
        def _call(*a, **k):
            self.writes.append((name, a, k))
            return []
        return _call

    def saw(self, needle):
        return any(needle in str(w) for w in self.writes)


@pytest.mark.parametrize("payload", [
    "../OUTSIDE.txt",
    "sandbox/../../OUTSIDE.txt",
    "../../../../OUTSIDE.txt",
    "a/../../OUTSIDE.txt",
])
def test_ingest_cannot_read_outside_the_sandbox(payload, sandbox):
    """⚠ DRIVEN END TO END, NOT ASSERTED ON THE HELPER.

    The helper was always correct; this branch simply did not call it. A test
    that only fuzzes `_get_safe_path` would have passed throughout the whole
    period the tool was reading arbitrary host files.

    Fails if the resolution goes back to `sandbox_dir / clean_name`: the tool
    then returns "SUCCESS: Ingested ..." and the outside file's contents are
    in `mem.writes`.
    """
    mem = _RecordingMemory()
    result = asyncio.run(tool_gain_knowledge(
        filename=payload, sandbox_dir=sandbox, memory_system=mem))
    assert not mem.saw("CANARY-CONTENT-OUTSIDE"), (
        f"content from outside the sandbox reached vector memory: {payload}")
    assert "SUCCESS" not in str(result), result


def test_ingest_still_works_for_a_real_sandbox_file(sandbox):
    """The counterweight: containment must not break the tool. Fails if the
    guard is tightened into a blanket refusal."""
    mem = _RecordingMemory()
    result = asyncio.run(tool_gain_knowledge(
        filename="inside.txt", sandbox_dir=sandbox, memory_system=mem))
    assert "SUCCESS" in str(result), result
    assert mem.saw("LEGITIMATE-SANDBOX-CONTENT")


def test_ingest_reports_the_refusal_rather_than_failing_silently(sandbox):
    """A refusal the model cannot read is a refusal it will retry forever.
    The message must name the path and say it was refused."""
    mem = _RecordingMemory()
    result = str(asyncio.run(tool_gain_knowledge(
        filename="../OUTSIDE.txt", sandbox_dir=sandbox, memory_system=mem)))
    # The message is the BARE "Security Error: …", matching the ~10
    # `file_system` sites. Prefixing "Error:" made it match the FAILURE
    # regex before the REJECTION one, so the same event had two statuses
    # depending on which tool raised it.
    assert result.startswith("Security Error:"), result
    assert "OUTSIDE.txt" in result


# ══ round two: the pins rewritten in the CONSUMER's vocabulary ════════════
#
# Four mutants survived the first version of this file. Every pin above was
# written in the PRODUCER's terms — it raises `ValueError`, the message
# starts with `Error:`, the message contains the path — and none asked the
# question the turn loop actually asks: *is this result classified as a
# refusal, and does the loop credit it with changing the world?*
#
# That gap was not hypothetical. `Security Error:` matched neither
# `_FAILURE_PREFIX_RE` (anchored on `ERROR\b|Error\b` — this head starts
# with an S) nor `_REJECTION_RE`, so twelve `file_system` handlers returned
# a containment refusal that the loop booked as a WORLD-CHANGING SUCCESS:
# the pre-flight guard was cleared, the loop-breaker's memory wiped, a
# strike decayed, and the project work_log recorded a file never written.

from ghost_agent.tools.outcome import ToolOutcome  # noqa: E402


@pytest.mark.parametrize("payload", ["../OUTSIDE.txt", "/workspace/../OUTSIDE.txt"])
def test_a_containment_refusal_is_not_a_world_change(payload, sandbox):
    """⚠ THE CONSUMER'S QUESTION. Fails if `Security Error` drops out of the
    rejection heads: the loop then credits a traversal refusal — including
    the guard that stops an `rmtree` of the whole workspace — as a
    successful write."""
    from ghost_agent.tools.file_system import tool_write_file
    result = asyncio.run(tool_write_file(
        filename=payload, content="x", sandbox_dir=sandbox))
    outcome = ToolOutcome.coerce(result)
    assert outcome.changed_the_world is False, (
        f"a containment refusal was credited as a world change: {result!r}")
    assert outcome.status == "rejected", outcome.status


def test_a_real_write_is_still_a_world_change(sandbox):
    """The counterweight. Classifying everything as a rejection would pass
    the test above and break the loop in the other direction."""
    from ghost_agent.tools.file_system import tool_write_file
    result = asyncio.run(tool_write_file(
        filename="ok.txt", content="hello", sandbox_dir=sandbox))
    assert ToolOutcome.coerce(result).status != "rejected", result
    assert (sandbox / "ok.txt").read_text() == "hello"


def test_ingest_refusal_reads_as_a_refusal_not_a_missing_file(sandbox):
    """⚠ A mutant that changed the message to "file not found. Use
    file_system(operation='list_files')…" passed the original wording pin —
    `startswith("Error:")` and the filename were both still there — while
    telling the model to go looking for a file that exists and is refused.
    It would then retry spellings forever. The message must name the
    REFUSAL."""
    # ⚠ THE PAYLOAD MUST NOT CONTAIN THE WORDS BEING ASSERTED. The first
    # version used `../OUTSIDE.txt` and accepted `"outside" in result` — so
    # a message that named no refusal at all ("'../OUTSIDE.txt' could not be
    # read") satisfied it, purely from the echoed filename.
    mem = _RecordingMemory()
    result = str(asyncio.run(tool_gain_knowledge(
        filename="../quiet.txt", sandbox_dir=sandbox, memory_system=mem)))
    low = result.lower()
    assert "security" in low or "refus" in low, (
        f"the refusal does not name itself as one: {result}")
    assert "not found" not in low, (
        "a containment refusal is being reported as a missing file")


def test_containment_follows_symlinks_rather_than_matching_dots(sandbox):
    """⚠ RIGHT ANSWER, WRONG REASON. Two mutants replaced resolve-and-contain
    with a lexical `".." in parts` test and passed every traversal payload
    in this file. A lexical check cannot see a SYMLINK, so this is the case
    that separates them: the name contains no `..` at all."""
    outside = sandbox.parent / "OUTSIDE.txt"
    (sandbox / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError):
        _get_safe_path(sandbox, "link.txt")


def test_containment_allows_a_dotdot_that_stays_inside(sandbox):
    """The other half of the same separation, and the reason a lexical check
    is wrong in BOTH directions: `sub/../inside.txt` never leaves the
    sandbox and must resolve normally."""
    (sandbox / "sub").mkdir()
    resolved = _get_safe_path(sandbox, "sub/../inside.txt").resolve()
    assert resolved == (sandbox / "inside.txt").resolve()


def test_containment_is_not_a_string_prefix_test(tmp_path):
    """⚠ THE SIBLING-DIRECTORY ESCAPE. `str(target).startswith(str(root))`
    accepts `<root>_backup/secret` because the root is a string prefix of
    the sibling's path. The codebase already documents this in
    `file_system.py`'s own pre-3.9 fallback ("so a sibling dir can't
    prefix-match"); a mutant removed it from the primary path and survived
    every other test here."""
    root = tmp_path / "sandbox"
    root.mkdir()
    sibling = tmp_path / "sandbox_backup"
    sibling.mkdir()
    (sibling / "secret").write_text("SIBLING-SECRET")
    with pytest.raises(ValueError):
        _get_safe_path(root, "../sandbox_backup/secret")
