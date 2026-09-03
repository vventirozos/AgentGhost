"""§4EC — `evolve/negative_controls.py` survivors of the §R re-verification of
§4CP (2026-09-02): the no-op control's identity helpers and the runner's result
bookkeeping (what `ok` / `rejected` / `detail` say on each refusal path)."""
from pathlib import Path

import pytest

from ghost_agent.evolve import negative_controls as NC
from ghost_agent.evolve import evaluator as EV

REPO = Path(__file__).resolve().parents[1]


def _tree(root, files):
    for rel, text in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    return root


class TestSemanticIdentity:
    def test_a_comment_is_a_no_op_and_a_deleted_function_is_not(self, tmp_path):
        base = _tree(tmp_path / "base", {"src/m.py": "def f():\n    return 1\n"})
        same = _tree(tmp_path / "same", {"src/m.py": "# note\ndef f():\n    return 1\n"})
        gone = _tree(tmp_path / "gone", {"src/m.py": "x = 1\n"})
        assert NC._semantically_identical(base, same, "src/m.py")[0] is True
        ok, why = NC._semantically_identical(base, gone, "src/m.py")
        assert ok is False and "syntax tree" in why

    def test_missing_unparseable_and_side_changes_are_named(self, tmp_path):
        base = _tree(tmp_path / "base", {"src/m.py": "x = 1\n", "src/o.py": "y = 2\n"})
        missing = _tree(tmp_path / "missing", {"src/o.py": "y = 2\n"})
        broken = _tree(tmp_path / "broken", {"src/m.py": "def (:\n", "src/o.py": "y = 2\n"})
        side = _tree(tmp_path / "side", {"src/m.py": "x = 1\n", "src/o.py": "y = 3\n"})
        assert NC._semantically_identical(base, missing, "src/m.py") == (False, "src/m.py does not exist in the candidate")
        ok, why = NC._semantically_identical(base, broken, "src/m.py"); assert ok is False and "does not parse" in why
        ok, why = NC._semantically_identical(base, side, "src/m.py"); assert ok is False and "also changed 1 other file" in why

    def test_changed_files_scans_every_tree_and_reports_new_and_differing(self, tmp_path):
        base = _tree(tmp_path / "base", {"src/a.py": "1", "tests/t.py": "t", "scripts/s.py": "s"})
        cand = _tree(tmp_path / "cand", {"tests/t.py": "T", "scripts/s.py": "s", "scripts/new.py": "n"})   # no src/ at all
        assert NC._changed_files(base, cand) == ["scripts/new.py", "tests/t.py"]   # `continue` past the missing src/


class TestRunnerBookkeeping:
    def test_edits_a_test_is_ok_only_when_refused_for_diff_scope(self, tmp_path, monkeypatch):
        """L304-305: `ok=(not passed and "diff-scope" in reason)` — refused for
        another reason is rejected but NOT ok; let through is neither."""
        monkeypatch.setattr(EV, "stage0_static", lambda *a, **k: EV.StageResult(EV.STAGE_STATIC, False, "some other objection", {}, 0.0))
        r = [x for x in NC.run_negative_controls(REPO, tmp_path, only=("edits_a_test",)).results if x.name == "edits_a_test"][0]
        assert (r.rejected, r.rejected_at, r.ok) == (True, EV.STAGE_STATIC, False)
        monkeypatch.setattr(EV, "stage0_static", lambda *a, **k: EV.StageResult(EV.STAGE_STATIC, True, "", {}, 0.0))
        r = [x for x in NC.run_negative_controls(REPO, tmp_path / "b", only=("edits_a_test",)).results if x.name == "edits_a_test"][0]
        assert (r.rejected, r.rejected_at, r.ok) == (False, "", False)

    def test_a_guard_control_that_cannot_materialise_says_so(self, tmp_path):
        """L315-316: materialise refused → not ok, detail names the refusal."""
        r = [x for x in NC.run_negative_controls(REPO, tmp_path, only=("deletes_a_guard",),
                                                  materialize=lambda *a, **k: (False, "disk full")).results
             if x.name == "deletes_a_guard"][0]
        assert r.ok is False and r.detail.startswith("could not materialise") and "disk full" in r.detail
