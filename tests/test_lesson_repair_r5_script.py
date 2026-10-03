"""The one-off §4KW r5 playbook repair: exact rows, all-or-nothing, the
duplicate pair loses exactly one row, and rewrites pass the gates."""
import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "lesson_repair_4kw_r5.py")
M = runpy.run_path(SCRIPT, run_name="not_main")


def _rows():
    rows = [{"trigger": t, "task": t, "mistake": "m", "solution": "s"} for t in M["RETRACT"] + list(M["REWRITE"])]
    one = M["RETRACT_ONE"]
    rows += [{"trigger": one["trigger"], "solution": one["solution_prefix"] + " …"},
             {"trigger": one["trigger"], "solution": "a different fix"}]
    return rows


def test_the_spec_resolves():
    M["check"](_rows())


@pytest.mark.parametrize("cut", [0, -1])
def test_a_missing_or_changed_row_applies_nothing(cut):
    rows = _rows()
    del rows[cut]
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](rows)


def test_only_one_of_the_duplicate_pair_is_selected():
    rows = _rows()
    assert sum(1 for x in rows if M["_is_dup_one"](x)) == 1


def test_no_rewrite_names_update_profile_for_reading():
    sol = next(v["solution"] for v in M["REWRITE"].values() if "profile" in v.get("solution", ""))
    assert "never for reading" in sol


def test_restamp_marks_only_scoped_rows_twins(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "plan A", "scope": "request", "source_request": "plan A", "solution": "s", "mistake": "m"},
                      {"trigger": "rule B", "solution": "s", "mistake": "m"}])
    vm = MagicMock()
    vm.collection.get.return_value = {"ids": ["1", "2"], "metadatas": [{"trigger": "plan A", "scope": "general"},
                                                                    {"trigger": "rule B", "scope": "general"}]}
    assert M["restamp_scoped_twins"](sm, vm) == 1
    kw = vm.collection.update.call_args.kwargs
    assert kw["ids"] == ["1"] and kw["metadatas"][0]["scope"] == "request"
