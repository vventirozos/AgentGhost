"""Regression tests for bug-hunt units 19-26.

See BUGHUNT.md. Fixed bugs pinned here:

Unit 19 (prm):
 - derive_step_labels SKIPS any non-PASSED/FAILED outcome (a junk outcome
   string no longer trains the PRM as an all-negative sample); a checkpoint
   whose weight length != current feature count is rejected at load
   (test_prm_corner_cases pins the load-raises side).
Unit 20 (reflection):
 - a transient reflection error un-claims the trajectory so a later tick can
   retry (test_audit_concurrency_fixes pins the run() path); the behavioural
   post-mortem lesson is not routed when its solution is empty; _split_patch_output
   splits on WORD boundaries ("dispatch" no longer triggers the patch marker).
Unit 21 (selfhood):
 - SelfState.from_dict filters unknown keys, so a schema-divergent/hand-edited
   state.json is NOT wiped-and-overwritten; detect_referenced_experiences only
   credits experiences actually present in the wake-up prefix.
Unit 22 (distill):
 - Trajectory.from_dict tolerates an extra (newer-schema) field instead of
   dropping the whole record, but still skips a no-recognized-fields record;
   the conn-URI redactor catches passwords containing '/' or ':'; JSON secret
   values with escaped quotes are fully redacted; the tool-error heuristic
   detects non-zero exit codes.
Unit 23 (eval):
 - a template task whose runner returns a dict WITHOUT a 'passed' verdict is
   scored unverified/fail (not a silent pass on non-empty text); SuiteResult
   recomputes its summary on construction.
Unit 24 (optim):
 - split_train_eval never leaves the train split empty (a 1-example corpus
   goes to train, not eval).
Unit 25 (skills_auto):
 - graduate() returns None when the just-added skill is overflow-evicted.
"""

import types

import pytest


# ══════════════════════════════════════════════════════════════════════
# Unit 19 — prm labels
# ══════════════════════════════════════════════════════════════════════

class TestPrmLabels:
    def _traj(self, outcome):
        from ghost_agent.distill.schema import Trajectory, ToolCall
        return Trajectory(
            user_request="do a thing",
            outcome=outcome,
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
        )

    def test_junk_outcome_skipped(self):
        from ghost_agent.prm.labels import derive_step_labels
        # Not PASSED/FAILED/UNKNOWN → must be skipped, not trained as all-neg.
        assert derive_step_labels(self._traj("error")) == []
        assert derive_step_labels(self._traj("timeout")) == []
        assert derive_step_labels(self._traj("")) == []

    def test_passed_outcome_labelled(self):
        from ghost_agent.prm.labels import derive_step_labels
        labels = derive_step_labels(self._traj("passed"))
        assert len(labels) == 2
        assert labels[-1] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# Unit 20 — reflection
# ══════════════════════════════════════════════════════════════════════

class TestReflectionPostmortem:
    def test_split_patch_word_boundary(self):
        from ghost_agent.reflection.postmortem import _split_patch_output
        # "dispatch" must NOT be read as the "patch" section marker.
        text = "reproducing test\nassert dispatch(x) == 1\n\npatch\n--- a\n+++ b"
        test_part, patch_part = _split_patch_output(text)
        assert "dispatch(x)" in test_part
        assert patch_part.startswith("patch")

    def test_no_marker_returns_empty(self):
        from ghost_agent.reflection.postmortem import _split_patch_output
        assert _split_patch_output("") == ("", "")


# ══════════════════════════════════════════════════════════════════════
# Unit 21 — selfhood
# ══════════════════════════════════════════════════════════════════════

class TestSelfStateSchemaDrift:
    def test_from_dict_tolerates_unknown_keys(self):
        from ghost_agent.selfhood.schema import SelfState
        # A newer/hand-edited file with an extra key on a nested item AND at
        # the top level must load (dropping the extras), not raise — a raise
        # is swallowed into an empty state that then overwrites the good file.
        d = {
            "schema_version": "9.9",
            "open_questions": [{"id": "q1", "text": "why", "priority": "high"}],
            "unfinished_threads": [{"id": "t1", "descriptor": "d", "extra": 1}],
            "mood": {"label": "curious", "evidence": "e", "future_field": True},
            "last_session_at": "2026-07-04T00:00:00Z",
            "brand_new_top_level_key": 123,
        }
        st = SelfState.from_dict(d)
        assert st.open_questions[0].text == "why"
        assert st.unfinished_threads[0].descriptor == "d"
        assert st.mood is not None and st.mood.label == "curious"
        assert st.last_session_at == "2026-07-04T00:00:00Z"


class TestDetectReferencedExperiences:
    def test_only_prefix_experiences_credited(self):
        from ghost_agent.selfhood.autobiographical import detect_referenced_experiences
        exp = types.SimpleNamespace(
            id="e1", summary="debugging the flaky checkout timeout", user_first_words="")
        # Response echoes the experience content, but it was NOT in the prefix
        # → must NOT be credited (the old code ignored prefix_text entirely).
        out = detect_referenced_experiences(
            prefix_text="something totally unrelated about weather",
            response_text="I recall debugging the flaky checkout timeout earlier",
            experiences=[exp],
        )
        assert out == []
        # Same experience, now present in the prefix → credited.
        out2 = detect_referenced_experiences(
            prefix_text="earlier: debugging the flaky checkout timeout",
            response_text="I recall debugging the flaky checkout timeout earlier",
            experiences=[exp],
        )
        assert out2 == ["e1"]


# ══════════════════════════════════════════════════════════════════════
# Unit 22 — distill
# ══════════════════════════════════════════════════════════════════════

class TestTrajectorySchemaDrift:
    def test_extra_field_loads(self):
        from ghost_agent.distill.schema import Trajectory
        d = {"user_request": "hi", "outcome": "passed", "a_future_field": 42,
             "tool_calls": [{"name": "t", "unknown_tc_field": 1}]}
        t = Trajectory.from_dict(d)
        assert t.user_request == "hi"
        assert t.outcome == "passed"
        assert t.tool_calls[0].name == "t"

    def test_no_recognized_fields_raises(self):
        from ghost_agent.distill.schema import Trajectory
        with pytest.raises(Exception):
            Trajectory.from_dict({"only": "garbage", "keys": "here"})


class TestRedactHardening:
    def test_conn_uri_password_with_slash_and_colon(self):
        from ghost_agent.distill.redact import redact_text
        out = redact_text("mongodb://admin:aB/cD3f@host:27017/db")
        assert "aB/cD3f" not in out and "<REDACTED>" in out

    def test_json_secret_with_escaped_quote(self):
        from ghost_agent.distill.redact import redact_text
        out = redact_text(r'{"password": "a\"b_secret"}')
        assert "b_secret" not in out


class TestOutcomeHeuristicExitCode:
    def test_nonzero_exit_code_is_error(self):
        from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
        assert _looks_like_tool_error("stdout...\nEXIT CODE: 127\n") is True
        assert _looks_like_tool_error("EXIT CODE: 0\nall good") is False


# ══════════════════════════════════════════════════════════════════════
# Unit 23 — eval
# ══════════════════════════════════════════════════════════════════════

class TestEvalTemplateVerdict:
    def test_template_dict_without_passed_is_unverified_fail(self):
        from ghost_agent.eval.tasks import ChallengeTemplateTask
        t = ChallengeTemplateTask(task_id="x", category="template", prompt="p", validator="exit 0")
        ok, reason = t.validate({"output": "some non-empty answer text", "steps": 3})
        assert ok is False
        assert "no 'passed' verdict" in reason

    def test_template_dict_with_passed_true(self):
        from ghost_agent.eval.tasks import ChallengeTemplateTask
        t = ChallengeTemplateTask(task_id="x", category="template", prompt="p", validator="exit 0")
        ok, _ = t.validate({"passed": True, "output": "text"})
        assert ok is True


class TestSuiteResultSummary:
    def test_summary_recomputed_on_construction(self):
        from ghost_agent.eval.metrics import SuiteResult, TaskResult
        tr = [
            TaskResult(task_id="a", category="c", cluster="cl", tier="t",
                       passed=True, duration_s=1.0),
            TaskResult(task_id="b", category="c", cluster="cl", tier="t",
                       passed=False, duration_s=1.0),
        ]
        sr = SuiteResult(suite_name="s", timestamp="t", ghost_version="v", results=tr)
        assert sr.summary  # not empty
        assert sr.summary["n"] == 2
        assert sr.summary["pass_rate"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════
# Unit 24 — optim
# ══════════════════════════════════════════════════════════════════════

class TestTrainsetSplit:
    def test_single_example_keeps_train_nonempty(self):
        from ghost_agent.optim.trainset import split_train_eval, TrainExample
        one = [TrainExample(signature_name="s", inputs={"a": "b"}, expected_output={})]
        train, ev = split_train_eval(one)
        assert len(train) == 1
        assert len(ev) == 0

    def test_multi_example_split_unchanged(self):
        from ghost_agent.optim.trainset import split_train_eval, TrainExample
        five = [TrainExample(signature_name="s", inputs={"i": str(i)}, expected_output={})
                for i in range(5)]
        train, ev = split_train_eval(five)
        assert len(train) == 4 and len(ev) == 1


# ══════════════════════════════════════════════════════════════════════
# Unit 25 — skills_auto store overflow
# ══════════════════════════════════════════════════════════════════════

class TestGraduateOverflow:
    def _cand(self, sig, conf):
        return types.SimpleNamespace(
            signature_hash=sig, name=sig, confidence=conf,
            support=5, tool_sequence=("a", "b"), cluster=None,
            trigger_examples=[], exemplar_trajectory_id="",
        )

    def test_overflow_evicted_entry_returns_none(self, tmp_path):
        from ghost_agent.skills_auto.store import GraduatedSkillStore, MAX_SKILLS
        store = GraduatedSkillStore(tmp_path / "auto_skills.json")
        for i in range(MAX_SKILLS):
            assert store.graduate(self._cand(f"hi{i}", 0.9), confidence=0.9) is not None
        # One more, lowest confidence → evicted by the overflow trim → None.
        assert store.graduate(self._cand("low", 0.1), confidence=0.1) is None

    def test_normal_graduation_returns_entry(self, tmp_path):
        from ghost_agent.skills_auto.store import GraduatedSkillStore
        store = GraduatedSkillStore(tmp_path / "auto_skills.json")
        entry = store.graduate(self._cand("s1", 0.8), confidence=0.8)
        assert entry is not None and entry["signature_hash"] == "s1"
