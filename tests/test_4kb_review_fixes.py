"""§4KB round 2 — pins for the defects fresh-eye review found in round 1's fixes.

R3: the previous round's fix is the least-reviewed code in the tree. Every test
here reproduces a defect that shipped in the §4KB batch and passed its own
24,231-green suite, six new batteries, and a 21-mutant battery. Each names the
world it fails in, which is the world before the corresponding fix.
"""
import pathlib
import subprocess

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import tool_replace_text
from ghost_agent.utils.edit_ledger import read_ledger


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))
    return h


# ---------------------------------------------------------------------------
# 1. CRITICAL — a `.git` FILE let git operate on any repo on the host
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. The gate is hoisted above the size branches
# ---------------------------------------------------------------------------


async def test_a_failed_write_is_not_recorded_as_applied(sandbox, home):
    """`telem["strategies"]` is set BEFORE the write. When the write raised,
    the generic handler returned a plain "Error: ..." string with no
    `is_rejection`, and the row said `applied: True` over a byte-identical
    file. Fails in that world.
    """
    f = sandbox / "ro.py"
    f.write_text("VALUE = 1\n")
    f.chmod(0o444)
    try:
        await tool_replace_text("ro.py", "VALUE = 1", "VALUE = 2", sandbox)
        assert f.read_text() == "VALUE = 1\n"      # nothing landed
        (row,) = read_ledger(home=home)
        assert row["applied"] is False, row
    finally:
        f.chmod(0o644)


# ---------------------------------------------------------------------------
# 3. Encoding: the receipt is the sha of the BYTES
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. A missing file must not latch the gate off process-wide
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Unreadable existing file: the write gate must fail CLOSED
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Control traffic pays nothing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. A mixed batch must not claim the FILE is unchanged
# ---------------------------------------------------------------------------

async def test_mixed_batch_does_not_claim_the_file_is_unchanged(sandbox, home):
    """One block lands, one is ambiguous — the file IS changed.

    The per-block message said "The file is UNCHANGED" and the log line said
    "file left unchanged", both false, while `UNIQUE_ONE = 2` was on disk.
    The same row also carried `applied=True` WITH `reason="ambiguous_block"`,
    contradicting the ledger's contract that `reason` explains a row that did
    NOT apply. Fails in that world.
    """
    f = sandbox / "m.py"
    f.write_text("UNIQUE_ONE = 1\nDUP = 0\nPAD = 0\nDUP = 0\n")
    res = await tool_replace_text(
        "m.py",
        "<<<< SEARCH\nUNIQUE_ONE = 1\n====\nUNIQUE_ONE = 2\n>>>>\n"
        "<<<< SEARCH\nDUP = 0\n====\nDUP = 9\n>>>>",
        None, sandbox)
    body = f.read_text()
    assert "UNIQUE_ONE = 2" in body          # the unique block DID land
    assert body.count("DUP = 0") == 2        # the ambiguous one did not
    assert "UNCHANGED" not in str(res)
    assert "THIS BLOCK was not applied" in str(res)

    (row,) = read_ledger(home=home)
    assert row["applied"] is True
    assert row["reason"] == "", (
        "a row that APPLIED must not carry a rejection reason")


async def test_ambiguous_verdict_survives_rewording_the_message(sandbox, home,
                                                                monkeypatch):
    """The all-ambiguous verdict used to be chosen by
    `e.startswith("Block REJECTED as AMBIGUOUS")` — a lexical match on this
    module's own prose. Rewording the message silently fell back to the
    generic "None ... matched" header, whose prescribed repair (widen the
    search) is the OPPOSITE of the right one.

    Pinned BEHAVIOURALLY, not by reading the source: the message is replaced
    with text sharing no prefix with the original, and the verdict must be
    unchanged. Fails in the lexical world.
    """
    monkeypatch.setattr(
        fs, "_ambiguous_block_error",
        lambda content, search, n: "totally different wording, no prefix")

    f = sandbox / "m.py"
    f.write_text("DUP = 0\nPAD = 1\nDUP = 0\n")
    res = await tool_replace_text(
        "m.py", "<<<< SEARCH\nDUP = 0\n====\nDUP = 9\n>>>>", None, sandbox)
    assert getattr(res, "is_rejection", False), res
    assert res.reason_code == "ambiguous_block", (
        "the verdict was selected from the message text")
    assert "matched MORE THAN ONCE" in str(res)
    assert f.read_text() == "DUP = 0\nPAD = 1\nDUP = 0\n"


# ---------------------------------------------------------------------------
# 8. git honours RELEASED-project immutability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. The registry fallback must not arm an un-benched gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. Navigation must never claim completeness it does not have
# ---------------------------------------------------------------------------

def test_an_empty_heuristic_outline_still_carries_its_caveat():
    """The caveat lived only in the NON-empty branch, so a heuristic read
    that found nothing returned a bare "(no top-level symbols found)" — the
    one output a model is most likely to read as "this file has no
    functions". Fails in that world.
    """
    from ghost_agent.tools.outline import (METHOD_HEURISTIC, render_outline)
    out = render_outline([], "mystery.js", METHOD_HEURISTIC)
    assert "no top-level symbols found" in out
    assert "NOT proof a symbol is absent" in out


def test_an_empty_ast_outline_does_not_carry_the_heuristic_caveat():
    """`ast` on a genuinely empty module IS exhaustive — claiming doubt there
    would be its own kind of lie."""
    from ghost_agent.tools.outline import METHOD_AST, render_outline
    out = render_outline([], "empty.py", METHOD_AST)
    assert "NOT proof" not in out


def test_symbols_says_so_when_the_index_was_truncated(tmp_path):
    """`build_symbol_index` caps the walk. A definite "No definition of X
    found in this project" printed over a SILENTLY truncated index is the
    same lie `render_outline` already refuses to tell. Fails in that world.
    """
    from ghost_agent.tools.outline import (TRUNCATED_KEY, build_symbol_index,
                                           render_definitions)
    for i in range(12):
        (tmp_path / f"f{i}.py").write_text(f"def fn{i}():\n    pass\n")
    (tmp_path / "needle.py").write_text("def the_needle():\n    pass\n")

    idx = build_symbol_index(tmp_path, cap=5)
    assert TRUNCATED_KEY in idx
    # os.walk order is filesystem-dependent, so ask for a name that is
    # certainly absent — the point is the CAVEAT, not which files were seen.
    out = render_definitions(idx, "certainly_not_defined_anywhere")
    assert "No definition" in out
    assert "PARTLY scanned" in out
    assert "NOT proof" in out

    full = build_symbol_index(tmp_path, cap=1000)
    assert TRUNCATED_KEY not in full
    miss = render_definitions(full, "certainly_not_defined_anywhere")
    assert "No definition" in miss
    assert "PARTLY scanned" not in miss       # a COMPLETE scan says nothing
    assert "needle.py" in render_definitions(full, "the_needle")
    # the sentinel must never leak into a user-facing listing
    assert TRUNCATED_KEY not in render_definitions(idx, "fn1")


# ---------------------------------------------------------------------------
# 11. Two gaps the round-2 mutation battery found (N3, N9)
# ---------------------------------------------------------------------------


