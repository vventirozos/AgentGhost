"""§4KB step 1 — the ladder report's arithmetic, re-derived independently.

R6: verify the instrument before the code. Every number this report prints is
a decision input for whether the tolerance ladder survives step 3, so each one
is recomputed here by hand from a fixture whose answer is known by
construction — never copied from the reporter's own output.

The corrective-rate is a PROXY (a later edit to the same file may be a
correction or planned work). These pins hold it to the two properties that
make the proxy comparable across rungs at all: it is scoped to the SAME
REQUEST, and it looks only FORWARD.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from edit_ladder_report import build, corrective_index      # noqa: E402


def row(req, path, applied=True, families=("exact",), reason="", ts=0.0):
    fams = list(families)
    return {
        "ts": ts, "req_id": req, "path": path, "op": "replace",
        "applied": applied,
        "strategies": [f if f != "fuzzy" else "fuzzy:93%" for f in fams],
        "families": fams, "reason": reason,
        "search_len": 10, "file_len": 100,
        "blocks_total": 0, "blocks_applied": 0,
    }


# ---------------------------------------------------------------------------
# corrective_index
# ---------------------------------------------------------------------------

def test_corrective_requires_same_request():
    """A later edit to the same file in ANOTHER request is ordinary work.

    Fails in the world where the index ignores req_id — which would inflate
    every rung's rate with unrelated turns and destroy the cross-rung
    comparison the proxy exists for.
    """
    rows = [row("r1", "a.py", families=["fuzzy"]),
            row("r2", "a.py", families=["exact"])]
    assert corrective_index(rows, window=3) == {0: False, 1: False}


def test_corrective_requires_same_path():
    rows = [row("r1", "a.py", families=["fuzzy"]),
            row("r1", "b.py", families=["exact"])]
    assert corrective_index(rows, window=3) == {0: False, 1: False}


def test_corrective_looks_forward_only():
    """The FIRST edit is not corrected by the one before it.

    Fails in the world where the window is symmetric — under which a single
    corrected edit would mark BOTH rows and double the measured rate.
    """
    rows = [row("r1", "a.py", families=["exact"]),
            row("r1", "a.py", families=["fuzzy"])]
    idx = corrective_index(rows, window=3)
    assert idx[0] is True and idx[1] is False


def test_corrective_window_is_bounded():
    """Outside the window, not corrective. Fails if `window` is ignored."""
    rows = [row("r1", "a.py", families=["fuzzy"])]
    rows += [row("r1", f"other{i}.py") for i in range(5)]
    rows += [row("r1", "a.py", families=["exact"])]
    assert corrective_index(rows, window=2)[0] is False
    assert corrective_index(rows, window=9)[0] is True


def test_rejected_rows_are_not_corrective_targets():
    """A REJECTED later edit did not fix anything — nothing landed.

    Fails in the world where the index counts attempts instead of applies,
    which would score a retry storm as a wave of successful corrections.
    """
    rows = [row("r1", "a.py", families=["fuzzy"]),
            row("r1", "a.py", applied=False, families=[],
                reason="no_blocks_matched")]
    assert corrective_index(rows, window=3)[0] is False


# ---------------------------------------------------------------------------
# build — the printed numbers
# ---------------------------------------------------------------------------

def test_build_counts_are_hand_checkable():
    rows = [
        row("r1", "a.py", families=["exact"]),                  # applied
        row("r1", "b.py", families=["fuzzy"]),                   # applied, corrected below
        row("r1", "b.py", families=["exact"]),                   # applied
        row("r2", "c.py", applied=False, families=[],
            reason="no_blocks_matched"),                         # rejected
    ]
    rep = build(rows, window=3)
    assert rep["rows"] == 4
    assert rep["applied"] == 3
    assert rep["rejected"] == 1
    # one of the three applies was non-exact
    assert rep["tolerant_applied"] == 1
    assert rep["tolerant_share"] == pytest.approx(1 / 3)
    # exact: a.py never re-edited, second b.py has nothing after it → 0/2
    assert rep["per_family"]["exact"] == {
        "applied": 2, "corrected": 0, "corrective_rate": 0.0}
    # fuzzy: the b.py fuzzy apply IS followed by a b.py apply → 1/1
    assert rep["per_family"]["fuzzy"] == {
        "applied": 1, "corrected": 1, "corrective_rate": 1.0}
    # COUNTS, not membership: the fixture had exactly one rejected row, so
    # `reasons[k] += 1` and `reasons[k] = 1` were indistinguishable.
    assert rep["rejection_reasons"] == {"no_blocks_matched": 1}
    multi = rows + [row("r3", "d.py", applied=False, families=[],
                        reason="no_blocks_matched"),
                    row("r4", "e.py", applied=False, families=[],
                        reason="ambiguous_block")]
    rep2 = build(multi, window=3)
    assert rep2["rejection_reasons"] == {"no_blocks_matched": 2,
                                         "ambiguous_block": 1}


def test_block_row_with_mixed_rungs_counts_each_family_once():
    """One call applying an exact AND a fuzzy envelope is one row, two rungs.

    Fails in the world where a multi-rung row is attributed to its first rung
    only — which would hide every tolerant apply that travelled beside an
    exact one, and those are the most common shape in the block form.
    """
    rows = [row("r1", "a.py", families=["exact", "fuzzy"])]
    rep = build(rows, window=3)
    assert rep["per_family"]["exact"]["applied"] == 1
    assert rep["per_family"]["fuzzy"]["applied"] == 1
    # still ONE applied edit overall — families are a breakdown, not a count
    assert rep["applied"] == 1
    assert rep["tolerant_applied"] == 1


def test_empty_ledger_reports_zeroes_not_division_error():
    rep = build([], window=3)
    assert rep["rows"] == 0
    assert rep["tolerant_share"] == 0.0
    assert rep["per_family"] == {}


def test_report_runs_end_to_end(tmp_path, capsys, monkeypatch):
    """The script's main() path, not just its helpers.

    Fails in the world where the helpers are right and the CLI is broken —
    which is how an instrument gets trusted without ever having been run.
    """
    import edit_ladder_report as rep_mod

    home = tmp_path / "h"
    p = home / "system" / "edits" / "ladder.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in [
        row("r1", "a.py", families=["exact"]),
        row("r1", "a.py", families=["fuzzy"]),
    ]) + "\n")

    monkeypatch.setattr(sys, "argv",
                        ["edit_ladder_report.py", "--home", str(home),
                         "--json"])
    assert rep_mod.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] == 2
    assert out["tolerant_applied"] == 1
