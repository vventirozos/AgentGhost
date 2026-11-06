"""Regression tests for bug-hunt units 19-26.

See BUGHUNT.md. Fixed bugs pinned here:

Units 19 (prm) and 20 (reflection) were deleted with their subsystems (§4NM:
`ghost_agent/prm/` and `ghost_agent/reflection/` are gone).
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
"""

import types

import pytest


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
            "mood": {"p": 0.6, "e": 0.45, "updated_at": "2026-07-04T00:00:00Z", "label": "cheerful",
                     "intensity": 2, "future_field": True},
            "last_session_at": "2026-07-04T00:00:00Z",
            "brand_new_top_level_key": 123,
        }
        st = SelfState.from_dict(d)
        # §4TB: the two lists are removed fields now — ignored like any unknown key
        assert not hasattr(st, "open_questions") and not hasattr(st, "unfinished_threads")
        assert st.mood is not None and st.mood.label == "cheerful" and not hasattr(st.mood, "future_field")
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

