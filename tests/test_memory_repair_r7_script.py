"""The §4KX r7 repair: exact rows, gate-passing rewrites, and only pruned
OWNER facts are restored."""
import runpy
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4kx_r7.py")
M = runpy.run_path(SCRIPT, run_name="not_main")


def _rows():
    return [{"trigger": t, "task": t, "mistake": "m", "solution": "s"} for t in M["RETRACT"] + list(M["REWRITE"])]


def _arch():
    return [{"reason": "prune_stale_edges", "subject": s, "predicate": p, "object": o} for s, p, o in M["RESTORE_EDGES"]]


def test_the_spec_resolves_and_restores_only_owner_facts():
    assert len(M["check"](_rows(), _arch())) == len(M["RESTORE_EDGES"])


def test_an_edge_that_was_not_pruned_by_decay_is_not_restored():
    arch = _arch()
    arch[0]["reason"] = "delete_by_target"                 # the user forgot it: never restored
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_rows(), arch)


def test_a_missing_row_applies_nothing():
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_rows()[1:], _arch())


def test_no_rewrite_keeps_the_leaked_word():
    assert all("the trigger" not in str(v).lower() for new in M["REWRITE"].values() for v in new.values())
