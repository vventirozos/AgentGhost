"""The §4KX r8 repair: only the five health/companion owner facts
the decay pruned are restored, and a live one is not re-added."""
import runpy
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4kx_r8.py")
M = runpy.run_path(SCRIPT, run_name="not_main")


def _arch():
    return [{"reason": "prune_stale_edges", "subject": s, "predicate": p, "object": o} for s, p, o in M["RESTORE_EDGES"]]


def test_the_spec_restores_the_five_owner_facts_and_never_the_profession():
    assert len(M["check"](_arch(), set())) == 5
    assert ["user", "HAS_PROFESSION", "doctor"] not in M["RESTORE_EDGES"]


def test_a_fact_the_user_forgot_is_never_restored():
    arch = _arch()
    arch[0]["reason"] = "forget_entity"
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](arch, set())


def test_a_live_fact_is_not_added_twice():
    live = {tuple(M["RESTORE_EDGES"][0])}
    assert len(M["check"](_arch(), live)) == 4
