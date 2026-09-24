"""§4KB step 1 — the edit ladder is PRICED: every replace call leaves one row.

`_locate_block` has always computed which rung matched (exact / flexible /
fuzzy / anchor) and the caller threw it away, so the one question that decides
whether the tolerance ladder is an asset or a liability — how often does an
APPLIED edit rest below `exact`, and does it get corrected right afterwards —
could not be asked. These pins hold the ledger to four properties:

  1. **Rung fidelity (R5 table).** One input per rung, one story: the row's
     family is the rung that actually matched. Fails if a rung stops being
     recorded, or is recorded under another rung's name.
  2. **Class closure.** EVERY return path of `tool_replace_text` — success,
     each rejection family, and an exception — leaves exactly one row. This is
     the enumeration that fails when a future return path bypasses the wrapper;
     it is the reason the recording lives in a wrapper and not at the rungs.
  3. **A rolled-back edit is NOT an applied edit.** The marker-leak and
     syntax-regression guards return REJECTED after a rung matched. If
     `applied` were read off the rung alone, every rollback would be counted
     as a tolerance success — the exact inversion that would make the ladder
     look safe because its safety net fired.
  4. **The ledger cannot change the outcome.** An unwritable ledger must cost
     nothing: same text, same status, file still edited.
"""
import json

import pytest

from ghost_agent.tools.file_system import tool_replace_text
from ghost_agent.utils.edit_ledger import (
    TOLERANT_FAMILIES,
    ledger_path,
    read_ledger,
    record_edit,
    rung_family,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated GHOST_HOME. Never let this fixture fall back to the operator's
    live data dir — `run-and-test-setup` records a test that did exactly that
    via a stray monkeypatch.undo()."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))
    return h


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


def rows(home):
    return read_ledger(home=home)


# ---------------------------------------------------------------------------
# 0. rung_family — the only parsing the ledger does
# ---------------------------------------------------------------------------

def test_rung_family_splits_suffix_and_passes_unknowns_through():
    assert rung_family("exact") == "exact"
    assert rung_family("flexible") == "flexible"
    assert rung_family("fuzzy:87%") == "fuzzy"
    assert rung_family("anchor:12-40") == "anchor"
    assert rung_family("") == ""
    assert rung_family(None) == ""
    # An unrecognised rung must surface as ITSELF in the report, not be
    # folded into a known family — otherwise a new rung would be invisible.
    assert rung_family("semantic:0.4") == "semantic"


# ---------------------------------------------------------------------------
# 1. Rung fidelity — one row per rung, correctly named (R5)
# ---------------------------------------------------------------------------

async def test_exact_rung_recorded(home, sandbox):
    (sandbox / "a.txt").write_text("alpha\nbeta\ngamma\n")
    res = await tool_replace_text("a.txt", "beta", "BETA", sandbox)
    assert "SUCCESS" in res
    (row,) = rows(home)
    assert row["families"] == ["exact"]
    assert row["applied"] is True
    assert row["file_len"] == len("alpha\nbeta\ngamma\n")


async def test_flexible_rung_recorded(home, sandbox):
    (sandbox / "a.txt").write_text("alpha\nkeep   the    spacing\ngamma\n")
    # Differs from the file only in whitespace → exact misses, flexible hits.
    res = await tool_replace_text(
        "a.txt", "keep the spacing", "REPLACED", sandbox)
    assert "SUCCESS" in res
    (row,) = rows(home)
    assert row["families"] == ["flexible"]
    assert row["applied"] is True


async def test_fuzzy_rung_recorded(home, sandbox):
    # Fixture lifted from tests/test_file_replace_fuzzy_and_size_guidance.py
    # (read from the tree, not invented — a transcribed fixture has twice
    #  been part-invented in this project's review history).
    (sandbox / "game.js").write_text(
        "const A = 1;\nlet spawnPoint = new V3(0, 30, 0);\nconst B = 2;\n")
    res = await tool_replace_text(
        "game.js",
        "let spawnPont = new V3(0, 30, 0);",      # single-char typo
        "let spawnPoint = new V3(0, 40, 0);",
        sandbox,
    )
    assert "SUCCESS" in res and "Fuzzy" in res
    (row,) = rows(home)
    assert row["families"] == ["fuzzy"]
    # The ratio must survive into the row — a bare "fuzzy" would lose the
    # only number that says HOW tolerant this apply was.
    assert row["strategies"][0].startswith("fuzzy:")
    assert row["applied"] is True


async def test_anchor_rung_recorded(home, sandbox):
    # .txt so the ladder is measured in isolation: no node, no ast, no
    # rollback guard participating in the outcome.
    (sandbox / "mesher.txt").write_text(
        "class Mesher {\n"
        "  addFace(vertices, colors, normals, x, y, z, dir, color) {\n"
        "    const [dx, dy, dz] = dir;\n"
        "    const indices = [];\n"
        "  }\n"
        "  other() { return 1; }\n"
        "}\n"
    )
    drifted = (
        "  addFace(vertices, colors, normals, x, y, z, dir, color) {\n"
        "    TOTALLY DIFFERENT MIDDLE THAT DOES NOT MATCH;\n"
        "  }"
    )
    res = await tool_replace_text(
        "mesher.txt", drifted, "  addFace() { return 0; }", sandbox)
    assert "SUCCESS" in res and "Anchor" in res
    (row,) = rows(home)
    assert row["families"] == ["anchor"]
    assert row["strategies"][0].startswith("anchor:")
    assert row["applied"] is True


async def test_every_tolerant_family_is_reachable(home, sandbox):
    """The three tolerant rungs are not dead code.

    Fails in the world where a ladder rung is unreachable — which would make
    the whole ledger a measurement of two rungs while reporting four.
    """
    seen = set()

    (sandbox / "f1.txt").write_text("alpha\nkeep   the    spacing\ngamma\n")
    await tool_replace_text("f1.txt", "keep the spacing", "X", sandbox)

    (sandbox / "f2.js").write_text(
        "const A = 1;\nlet spawnPoint = new V3(0, 30, 0);\nconst B = 2;\n")
    await tool_replace_text(
        "f2.js", "let spawnPont = new V3(0, 30, 0);",
        "let spawnPoint = new V3(0, 40, 0);", sandbox)

    (sandbox / "f3.txt").write_text(
        "head\nfunction build(chunk) {\n  const a = 1;\n  const b = 2;\n}\ntail\n")
    await tool_replace_text(
        "f3.txt",
        "function build(chunk) {\n  WRONG BODY here;\n}",
        "function build(chunk) { return 0; }",
        sandbox,
    )

    for r in rows(home):
        seen.update(f for f in r["families"] if f in TOLERANT_FAMILIES)
    assert seen == set(TOLERANT_FAMILIES), seen


# ---------------------------------------------------------------------------
# 2. Class closure — every return path leaves exactly one row
# ---------------------------------------------------------------------------

#: One label per RETURN FAMILY of `_replace_text_impl`.
#:
#: ⚠ This table was 8 labels reaching 7 of the function's 30 top-level
#: returns, and it was described in the journal as a class closure. It was
#: not: a pin-quality audit added an early `return` to the WRAPPER for
#: `new_text is None` and all 21 tests stayed green. A hand-written list of
#: paths is a sample; what closes the class is the COUNT assertion below,
#: which walks the function's own AST and fails when a return family grows
#: that no label drives.
RETURN_PATHS = [
    "empty_old_text",
    "identical_args_corruption",
    "missing_replace_with",
    "no_match",
    "exact_success",
    "flexible_success",
    "fuzzy_success",
    "anchor_success",
    "block_form_success",
    "block_no_match",
    "block_partial",
    "multi_edit_envelope_rejected",
    "ambiguous_block",
    "path_escape",
    "file_not_found",
    "binary_file",
    "syntax_rollback",
    "streaming_success",
    "auto_promote_write",
    "too_large",
    "inode_denied",
    "guard_inode_mismatch",
    "streaming_source_swapped",
    "read_changed_underneath",
    "not_a_regular_file",
]


@pytest.mark.parametrize("label", RETURN_PATHS)
async def test_every_return_path_records_exactly_one_row(label, home, sandbox):
    """Each return FAMILY leaves exactly one row — never zero, never two.

    Fails in the world where a return bypasses the wrapper (zero rows) or a
    path records twice. The table is a sample; `test_the_return_path_table_is
    _not_a_sample` is what stops the sample silently shrinking.
    """
    before = len(rows(home))
    await _drive(label, sandbox)
    after = rows(home)
    assert len(after) - before == 1, (
        f"{label}: expected exactly 1 ledger row, got {len(after) - before}. "
        "A return path that bypasses the wrapper is the defect this pin "
        "exists to catch.")
    assert after[-1]["path"]
    # Every row must carry the request id: the corrective-rate proxy's
    # same-request scoping rests entirely on it, and NOTHING pinned it until
    # a pin-quality audit pointed that out.
    assert "req_id" in after[-1]
    assert after[-1]["search_len"] >= 0


async def _drive(label, sandbox):
    (sandbox / "t.txt").write_text("alpha\nbeta\ngamma\n")
    if label == "empty_old_text":
        return await tool_replace_text("t.txt", "", "x", sandbox)
    if label == "identical_args_corruption":
        return await tool_replace_text("t.txt", "beta", "beta", sandbox)
    if label == "missing_replace_with":
        return await tool_replace_text("t.txt", "beta", None, sandbox)
    if label == "no_match":
        return await tool_replace_text(
            "t.txt", "NOTHING_LIKE_THIS_EXISTS_ANYWHERE_42", "x", sandbox)
    if label == "exact_success":
        return await tool_replace_text("t.txt", "beta", "BETA", sandbox)
    if label == "flexible_success":
        (sandbox / "t.txt").write_text("alpha\nkeep   the    spacing\n")
        return await tool_replace_text(
            "t.txt", "keep the spacing", "X", sandbox)
    if label == "fuzzy_success":
        (sandbox / "t.js").write_text(
            "const A = 1;\nlet spawnPoint = new V3(0, 30, 0);\nconst B = 2;\n")
        return await tool_replace_text(
            "t.js", "let spawnPont = new V3(0, 30, 0);",
            "let spawnPoint = new V3(0, 40, 0);", sandbox)
    if label == "anchor_success":
        (sandbox / "t.txt").write_text(
            "head\nfunction build(c) {\n  const a = 1;\n}\ntail\n")
        return await tool_replace_text(
            "t.txt", "function build(c) {\n  WRONG;\n}",
            "function build(c) { return 0; }", sandbox)
    if label == "block_form_success":
        return await tool_replace_text(
            "t.txt", "<<<< SEARCH\nbeta\n====\nBETA\n>>>>", None, sandbox)
    if label == "block_no_match":
        return await tool_replace_text(
            "t.txt", "<<<< SEARCH\nNOPE_NOT_HERE_99\n====\nx\n>>>>",
            None, sandbox)
    if label == "block_partial":
        return await tool_replace_text(
            "t.txt",
            "<<<< SEARCH\nbeta\n====\nBETA\n>>>>\n"
            "<<<< SEARCH\nNOT_THERE_7\n====\nx\n>>>>", None, sandbox)
    if label == "multi_edit_envelope_rejected":
        return await tool_replace_text(
            "t.txt",
            "<<<< SEARCH\nbeta\n====\nBETA\n====\ngamma\n>>>>", None, sandbox)
    if label == "ambiguous_block":
        (sandbox / "t.txt").write_text("DUP = 0\nPAD\nDUP = 0\n")
        return await tool_replace_text(
            "t.txt", "<<<< SEARCH\nDUP = 0\n====\nDUP = 9\n>>>>",
            None, sandbox)
    if label == "path_escape":
        return await tool_replace_text("../../etc/passwd", "a", "b", sandbox)
    if label == "file_not_found":
        return await tool_replace_text("absent_file_xyz.txt", "a", "b", sandbox)
    if label == "binary_file":
        (sandbox / "b.bin").write_bytes(b"\x00\x01\x02binary\x00\xff" * 40)
        return await tool_replace_text("b.bin", "binary", "x", sandbox)
    if label == "syntax_rollback":
        (sandbox / "m.py").write_text("def f():\n    return 1\n")
        return await tool_replace_text(
            "m.py", "    return 1", "    return (1", sandbox)
    if label == "streaming_success":
        big = sandbox / "big.txt"
        big.write_text("HEAD\n" + ("pad\n" * 200_000) + "TARGET\n")
        return await tool_replace_text("big.txt", "TARGET", "HIT", sandbox)
    if label == "auto_promote_write":
        (sandbox / "mod.py").write_text("OLD = 1\n")
        return await tool_replace_text(
            "mod.py", "def f():\n    return 1\n", None, sandbox)
    if label == "too_large":
        import os as _os
        big = sandbox / "huge.txt"
        big.write_bytes(b"")
        _os.truncate(big, 51 * 1024 * 1024)   # sparse: instant
        return await tool_replace_text("huge.txt", "x", "y", sandbox)
    if label == "inode_denied":
        import os as _os
        _os.link(sandbox / "t.txt", sandbox / "twin.txt")
        return await tool_replace_text("t.txt", "beta", "BETA", sandbox)
    if label == "read_changed_underneath":
        # r6: the non-streaming read's identity mismatch is the guard's own
        # rejected outcome (a directory swap between the guard's open and
        # the read's open; triggered from the reader's frame)
        import os as _os
        import sys as _sys
        import ghost_agent.tools.file_system as _fsmod
        (sandbox / "d").mkdir()
        (sandbox / "d2").mkdir()
        (sandbox / "d" / "t.txt").write_text("alpha\nbeta\ngamma\n")
        (sandbox / "d2" / "t.txt").write_text("alpha\nbeta\ngamma\n")
        real_open = _os.open
        state = {"done": False}

        def _swap_for_reader(p, flags, *a, **k):
            if (not state["done"] and str(p).endswith("t.txt")
                    and _sys._getframe(1).f_code.co_name == "_read_text_guarded"):
                state["done"] = True
                _os.rename(sandbox / "d", sandbox / "tmp")
                _os.rename(sandbox / "d2", sandbox / "d")
            return real_open(p, flags, *a, **k)
        _fsmod.os.open = _swap_for_reader
        try:
            return await tool_replace_text("d/t.txt", "beta", "BETA", sandbox)
        finally:
            _fsmod.os.open = real_open
    if label == "streaming_source_swapped":
        # r4: the streaming reader opens its own fd; a name swapped to a
        # different inode between the guard and that open is refused.
        import os as _os
        import ghost_agent.tools.file_system as _fsmod
        (sandbox / "d").mkdir()
        (sandbox / "d2").mkdir()
        body = "HEAD\n" + ("pad\n" * 200_000) + "TARGET\n"
        (sandbox / "d" / "big.txt").write_text(body)
        (sandbox / "d2" / "big.txt").write_text(body)
        import sys as _sys
        real_open = _os.open
        state = {"swapped": False}

        def _swap_for_streaming(p, flags, *a, **k):   # by the caller's frame
            if (not state["swapped"] and str(p).endswith("big.txt")
                    and _sys._getframe(1).f_code.co_name == "_streaming_replace"):
                state["swapped"] = True
                _os.rename(sandbox / "d", sandbox / "tmp")
                _os.rename(sandbox / "d2", sandbox / "d")
            return real_open(p, flags, *a, **k)
        _fsmod.os.open = _swap_for_streaming
        try:
            return await tool_replace_text("d/big.txt", "TARGET", "HIT", sandbox)
        finally:
            _fsmod.os.open = real_open
    if label == "guard_inode_mismatch":
        # §4KC r4: the name is swapped to a clean inode for the guard's own
        # open and swapped back — the pre-read stat and the guarded fstat
        # disagree, and the replace is refused before anything is read.
        import os as _os
        import ghost_agent.tools.file_system as _fsmod
        (sandbox / "d").mkdir()
        (sandbox / "d2").mkdir()
        (sandbox / "d" / "t.txt").write_text("alpha\nbeta\ngamma\n")
        (sandbox / "d2" / "t.txt").write_text("alpha\nbeta\ngamma\n")
        real_open = _os.open
        state = {"done": False}

        def _swap(p, flags, *a, **k):          # directory swap: inode ctimes untouched
            if (not state["done"] and str(p).endswith("d/t.txt")
                    and flags & getattr(_os, "O_NOFOLLOW", 0) and not flags & _os.O_WRONLY):
                state["done"] = True
                _os.rename(sandbox / "d", sandbox / "tmp")
                _os.rename(sandbox / "d2", sandbox / "d")
                fd = real_open(p, flags, *a, **k)
                _os.rename(sandbox / "d", sandbox / "d2")
                _os.rename(sandbox / "tmp", sandbox / "d")
                return fd
            return real_open(p, flags, *a, **k)
        _fsmod.os.open = _swap
        try:
            return await tool_replace_text("d/t.txt", "beta", "BETA", sandbox)
        finally:
            _fsmod.os.open = real_open
    if label == "not_a_regular_file":
        import os as _os
        fifo = sandbox / "pipe.txt"
        _os.mkfifo(fifo)
        try:
            return await tool_replace_text("pipe.txt", "a", "b", sandbox)
        finally:
            _os.unlink(fifo)
    raise AssertionError(f"unmapped path {label}")


def test_the_return_path_table_is_not_a_sample():
    """THE class closure. Everything above is a sample; this is the closure.

    Walks `_replace_text_impl`'s own AST, counts its top-level `return`
    statements, and fails when that count moves. A hand-written path list
    cannot notice a new return; this does, and forces whoever adds one to
    decide whether it needs a label.

    Fails in the world where a return family is added — which is exactly how
    the streaming path went unrecorded through a whole review round.
    """
    import ast
    import inspect
    import ghost_agent.tools.file_system as fsmod

    tree = ast.parse(inspect.getsource(fsmod._replace_text_impl))
    fn = tree.body[0]
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    # 37 = 36 ledger paths + the `return` inside the nested `_inode_check`
    # helper (§4KC r3), which `ast.walk` also visits.
    assert len(returns) == 40, (
        f"`_replace_text_impl` now has {len(returns)} return statements, not "
        f"40. A new return is a new ledger path: give it a RETURN_PATHS label "
        f"and a `_drive` case, then update this number.")


async def test_exception_path_records_then_reraises(home, sandbox, monkeypatch):
    """An exception is an outcome, and the one most worth counting.

    Fails in the world where the wrapper swallows the exception (caller loses
    a failure it is written to see) or drops the row (the loudest failures
    become the invisible ones).
    """
    import ghost_agent.tools.file_system as fs

    boom = RuntimeError("disk on fire")

    async def _explode(*a, **kw):
        raise boom

    monkeypatch.setattr(fs, "_replace_text_impl", _explode)
    with pytest.raises(RuntimeError, match="disk on fire"):
        await fs.tool_replace_text("t.txt", "a", "b", sandbox)
    (row,) = rows(home)
    assert row["applied"] is False
    assert row["reason"] == "exception:RuntimeError"


# ---------------------------------------------------------------------------
# 3. A rolled-back edit is NOT an applied edit
# ---------------------------------------------------------------------------

async def test_syntax_rollback_records_matched_rung_but_not_applied(home, sandbox):
    """The rung matched; the guard rolled it back; `applied` must be False.

    Fails in the world where `applied` is computed from the rung alone — in
    which case every syntax-regression rollback would be tallied as a
    successful tolerant apply, and the ladder's measured safety would be an
    artifact of its safety net firing.
    """
    target = sandbox / "m.py"
    target.write_text("def f():\n    return 1\n")
    res = await tool_replace_text(
        "m.py", "    return 1", "    return (1", sandbox)
    assert getattr(res, "is_rejection", False), res
    assert target.read_text() == "def f():\n    return 1\n"   # unchanged
    (row,) = rows(home)
    assert row["families"] == ["exact"]      # the rung DID match
    assert row["applied"] is False           # but nothing landed


# ---------------------------------------------------------------------------
# 4. The ledger cannot change the outcome
# ---------------------------------------------------------------------------

async def test_unwritable_ledger_costs_nothing(tmp_path, sandbox, monkeypatch):
    """Point GHOST_HOME at a FILE so every mkdir/append raises.

    Fails in the world where a telemetry write can fail a turn.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    monkeypatch.setenv("GHOST_HOME", str(blocker))

    (sandbox / "a.txt").write_text("alpha\nbeta\n")
    res = await tool_replace_text("a.txt", "beta", "BETA", sandbox)
    assert "SUCCESS" in res
    assert (sandbox / "a.txt").read_text() == "alpha\nBETA\n"
    assert read_ledger(home=blocker) == []


def test_record_edit_never_raises_and_reports_failure(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("file")
    assert record_edit(path="x.py", home=blocker) is False


def test_read_ledger_skips_malformed_lines(tmp_path):
    home = tmp_path / "h"
    p = ledger_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"path": "a.py", "applied": True}) + "\n"
        "{ this is not json\n"
        "\n"
        + json.dumps({"path": "b.py", "applied": False}) + "\n"
    )
    got = read_ledger(home=home)
    assert [r["path"] for r in got] == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# 5. Block form — the counts that make "edits per turn" answerable
# ---------------------------------------------------------------------------

async def test_block_form_counts_envelopes_and_applies(home, sandbox):
    (sandbox / "a.txt").write_text("alpha\nbeta\ngamma\n")
    res = await tool_replace_text(
        "a.txt",
        "<<<< SEARCH\nalpha\n====\nALPHA\n>>>>\n"
        "<<<< SEARCH\ngamma\n====\nGAMMA\n>>>>",
        None,
        sandbox,
    )
    assert "SUCCESS" in res
    (row,) = rows(home)
    assert row["op"] == "replace_block"
    assert row["blocks_total"] == 2
    assert row["blocks_applied"] == 2
    assert row["families"] == ["exact", "exact"]


async def test_block_form_partial_records_both_counts(home, sandbox):
    """One envelope lands, one misses: the row must show 2 attempted, 1 applied.

    Fails in the world where blocks_total is derived from blocks_applied —
    which would erase every partial edit from the record, and partials are
    exactly where a tolerant ladder does its damage.
    """
    (sandbox / "a.txt").write_text("alpha\nbeta\ngamma\n")
    await tool_replace_text(
        "a.txt",
        "<<<< SEARCH\nalpha\n====\nALPHA\n>>>>\n"
        "<<<< SEARCH\nNOT_IN_THE_FILE_7\n====\nx\n>>>>",
        None,
        sandbox,
    )
    (row,) = rows(home)
    assert row["blocks_total"] == 2
    assert row["blocks_applied"] == 1
