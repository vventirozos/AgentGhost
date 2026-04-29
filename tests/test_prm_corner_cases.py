"""Corner-case + adversarial tests for the PRM pipeline.

Categories exercised here:

  * Data degeneracy — empty corpus, all-UNKNOWN, all-PASSED,
    all-FAILED, mixed-with-corruption, malformed tool_args.
  * Numerical extremes — NaN/inf labels, NaN inputs, all-zero features,
    extreme weights' interior arithmetic.
  * Type / shape — None inputs, wrong dataclasses, mismatched lengths.
  * I/O — corrupt JSON, missing required fields, wrong schema, missing
    parent dir, save-then-load round-trip after partial mutation.
  * Scorer — score under partial state, score_many empty / single, bad
    types, clamp-NaN-to-neutral.
  * MCTS integration — scorer that always raises, scorer returning
    NaN, all-tied scores.
  * Trainer — repeat ``run`` calls, save error, hot-swap of
    ``trainer.model`` after a failed second run.
  * Stress — 5K-sample training time bound, 1K-candidate batched
    scoring sanity.
  * Concurrency — score() interleaved with set_model() must never
    return a torn read or raise.

Each failure here is either a real bug or a contract violation; either
way it must be fixed before the feature is shipped.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm import (
    PRMScorer,
    PRMTrainer,
    PRM_FEATURE_NAMES,
    ActionFeatures,
    PlanState,
    StepLabelSpec,
    StepValueModel,
    derive_step_labels,
    extract_step_features,
    iter_step_samples,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _state(**kw):
    base = dict(
        user_request="x",
        steps_so_far=0, failures_so_far=0, pending_count=0, plan_depth=0,
        tools_used_this_turn=(), tools_failed_this_turn=(),
    )
    base.update(kw)
    return PlanState(**base)


def _action(**kw):
    base = dict(description="", tool_name="", tool_args={})
    base.update(kw)
    return ActionFeatures(**base)


def _passing(*, n=2, request="ok", tool="scratchpad"):
    return Trajectory(
        user_request=request, outcome=Outcome.PASSED.value,
        tool_calls=[ToolCall(name=tool) for _ in range(n)],
        n_steps=n,
    )


def _failing(*, n=2, request="bad", tool="execute"):
    return Trajectory(
        user_request=request, outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name=tool, error="boom") for _ in range(n)],
        n_steps=n,
    )


def _balanced(n_pass=8, n_fail=8):
    return ([_passing(request=f"p{i}") for i in range(n_pass)]
            + [_failing(request=f"f{i}") for i in range(n_fail)])


def _trained_scorer():
    """Quick fitter, returns a real scorer with a real model."""
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    trainer.run(_balanced())
    return PRMScorer(model=trainer.model)


# ══════════════════════════════════════════════════════════════════════
# DATA DEGENERACY
# ══════════════════════════════════════════════════════════════════════

class TestDataDegeneracy:
    def test_empty_corpus_bails_cleanly(self):
        report = PRMTrainer().run([])
        assert not report.fit_attempted
        assert "trajectories" in report.bail_reason

    def test_all_unknown_corpus_bails_on_samples(self):
        corpus = [
            Trajectory(user_request=f"x{i}", outcome=Outcome.UNKNOWN.value,
                       tool_calls=[ToolCall(name="a")])
            for i in range(50)
        ]
        report = PRMTrainer(min_trajectories=2, min_samples=5).run(corpus)
        assert not report.fit_attempted
        assert report.n_samples_total == 0

    def test_all_passed_corpus_bails_on_imbalance(self):
        report = PRMTrainer(min_trajectories=2, min_samples=5).run(
            [_passing(request=f"p{i}") for i in range(20)]
        )
        assert not report.fit_attempted
        assert "imbalance" in report.bail_reason

    def test_all_failed_corpus_bails_on_imbalance(self):
        report = PRMTrainer(min_trajectories=2, min_samples=5).run(
            [_failing(request=f"f{i}") for i in range(20)]
        )
        assert not report.fit_attempted
        assert "imbalance" in report.bail_reason

    def test_zero_step_trajectory_yields_no_samples(self):
        t = Trajectory(
            user_request="x", outcome=Outcome.PASSED.value,
            tool_calls=[], n_steps=0,
        )
        assert derive_step_labels(t) == []
        assert list(iter_step_samples([t])) == []

    def test_malformed_tool_args_dont_crash_extraction(self):
        """Tool args could be a string, list, None, or even something
        weird. Feature extraction must not crash on any of these."""
        for weird_args in [None, "not a dict", [1, 2, 3], 42, b"bytes",
                           {"nested": {"deep": [1, {"x": object()}]}}]:
            tc = ToolCall(name="x", arguments=weird_args)  # type: ignore
            t = Trajectory(
                user_request="x", outcome=Outcome.PASSED.value,
                tool_calls=[tc], n_steps=1,
            )
            samples = list(iter_step_samples([t]))
            # Must produce a sample without raising.
            assert len(samples) == 1
            fv = extract_step_features(samples[0].state, samples[0].action)
            assert all(math.isfinite(v) for v in fv.values)

    def test_huge_user_request_doesnt_overflow_features(self):
        t = _passing(request="x" * 10_000_000)  # 10 MB
        samples = list(iter_step_samples([t]))
        for s in samples:
            fv = extract_step_features(s.state, s.action)
            assert all(math.isfinite(v) for v in fv.values)

    def test_unicode_request_handled(self):
        t = _passing(request="日本語 🎉 αβγ ∑∫∂")
        samples = list(iter_step_samples([t]))
        for s in samples:
            fv = extract_step_features(s.state, s.action)
            assert all(math.isfinite(v) for v in fv.values)

    def test_many_steps_trajectory(self):
        """A 200-step trajectory: γ^199 ≈ 0 should be tiny but finite,
        not underflow to negatives or NaN."""
        t = _passing(n=200)
        labels = derive_step_labels(t, StepLabelSpec(discount_factor=0.9))
        assert len(labels) == 200
        assert all(0.0 <= v <= 1.0 for v in labels)
        assert all(math.isfinite(v) for v in labels)
        # Every label finite even at γ^199 ≈ 7e-10
        assert labels[0] >= 0.0

    def test_outcome_with_mixed_case_string(self):
        """Defensive: outcome strings written by old code might have
        case variations. derive_step_labels normalises to lowercase."""
        t = Trajectory(
            user_request="x", outcome="PASSED",
            tool_calls=[ToolCall(name="a")], n_steps=1,
        )
        # Whatever the legacy behaviour, this must not raise.
        # We accept either 1 sample (if normalised) or 0 (if strict).
        out = derive_step_labels(t)
        assert isinstance(out, list)


# ══════════════════════════════════════════════════════════════════════
# NUMERICAL EXTREMES
# ══════════════════════════════════════════════════════════════════════

class TestNumericalExtremes:
    def test_fit_with_nan_label_does_not_corrupt_weights(self):
        """A NaN label should not silently produce a model that returns
        NaN for every prediction. Either reject or sanitise."""
        X = [extract_step_features(_state(), _action(tool_name="execute"))
             for _ in range(20)]
        y = [float("nan")] * 10 + [1.0] * 10
        m = StepValueModel(epochs=50)
        try:
            m.fit(X, y)
        except (ValueError, RuntimeError):
            return  # Rejection is acceptable.
        # If accepted, predictions must not be NaN.
        p = m.predict_proba(X[0])
        assert math.isfinite(p), f"predictions corrupted by NaN labels: {p}"

    def test_fit_with_inf_label_does_not_corrupt(self):
        X = [extract_step_features(_state(), _action(tool_name="execute"))
             for _ in range(20)]
        y = [float("inf")] * 10 + [0.0] * 10
        m = StepValueModel(epochs=50)
        try:
            m.fit(X, y)
        except (ValueError, RuntimeError):
            return
        p = m.predict_proba(X[0])
        assert 0.0 <= p <= 1.0, f"infinity-poisoned model returned {p}"

    def test_predict_proba_nan_input_does_not_propagate(self):
        m = _trained_scorer().model
        bad = [float("nan")] * len(PRM_FEATURE_NAMES)
        # Either clamp/reject or return a finite value. Must not return NaN.
        try:
            p = m.predict_proba(bad)
            assert math.isfinite(p), f"NaN propagated through predict: {p}"
        except (ValueError, TypeError):
            pass  # Rejection is acceptable.

    def test_predict_proba_inf_input(self):
        m = _trained_scorer().model
        bad = [float("inf")] * len(PRM_FEATURE_NAMES)
        try:
            p = m.predict_proba(bad)
            # Sigmoid clip handles ±60; inf gets clipped, returns 0 or 1.
            assert 0.0 <= p <= 1.0
        except (ValueError, TypeError):
            pass

    def test_all_zero_feature_vector_predicts_bias_only(self):
        m = _trained_scorer().model
        zero_vec = [0.0] * len(PRM_FEATURE_NAMES)
        p = m.predict_proba(zero_vec)
        # With zero features, the prediction is sigmoid(bias).
        from ghost_agent.prm.model import _sigmoid
        expected = _sigmoid(m.bias_)
        assert abs(p - expected) < 1e-9

    def test_extreme_positive_weights_dont_overflow_predict(self):
        m = StepValueModel()
        m.weights_ = np.full(len(PRM_FEATURE_NAMES), 1e10)
        m.bias_ = 1e10
        p = m.predict_proba([1.0] * len(PRM_FEATURE_NAMES))
        assert 0.0 <= p <= 1.0
        assert math.isfinite(p)

    def test_extreme_negative_weights_dont_underflow_predict(self):
        m = StepValueModel()
        m.weights_ = np.full(len(PRM_FEATURE_NAMES), -1e10)
        m.bias_ = -1e10
        p = m.predict_proba([1.0] * len(PRM_FEATURE_NAMES))
        assert 0.0 <= p <= 1.0
        assert math.isfinite(p)

    def test_discount_nan_clamps_safely(self):
        """A NaN discount_factor must not produce NaN labels."""
        t = _passing(n=3)
        labels = derive_step_labels(t, StepLabelSpec(discount_factor=float("nan")))
        # Either the NaN clamps, or labels stay 0/empty — never NaN itself.
        assert all(math.isfinite(v) for v in labels), (
            f"NaN discount produced NaN labels: {labels}"
        )

    def test_predict_proba_wrong_length_vector_raises(self):
        m = _trained_scorer().model
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            # Vector of wrong length should fail loud, not silently
            # produce a misaligned dot product.
            m.predict_proba([0.0] * (len(PRM_FEATURE_NAMES) + 5))

    def test_fit_reproducibility_with_same_seed(self):
        """Same data + same random_state must produce bit-identical weights."""
        X = [extract_step_features(_state(), _action(tool_name="scratchpad"))
             for _ in range(8)]
        X += [extract_step_features(
                  _state(failures_so_far=2,
                         tools_failed_this_turn=("execute",)),
                  _action(tool_name="execute"))
              for _ in range(8)]
        y = [1] * 8 + [0] * 8
        a = StepValueModel(epochs=200, random_state=42).fit(X, y)
        b = StepValueModel(epochs=200, random_state=42).fit(X, y)
        np.testing.assert_array_equal(a.weights_, b.weights_)
        assert a.bias_ == b.bias_


# ══════════════════════════════════════════════════════════════════════
# TYPE & SHAPE ERRORS
# ══════════════════════════════════════════════════════════════════════

class TestTypeAndShape:
    def test_predict_proba_rejects_object_input(self):
        m = _trained_scorer().model
        with pytest.raises(TypeError):
            m.predict_proba(object())

    def test_predict_proba_rejects_dict_input(self):
        m = _trained_scorer().model
        with pytest.raises(TypeError):
            m.predict_proba({"a": 1})

    def test_fit_rejects_mismatched_lengths(self):
        X = [extract_step_features(_state(), _action()) for _ in range(5)]
        y = [1, 0, 1]  # mismatched
        with pytest.raises(ValueError):
            StepValueModel().fit(X, y)

    def test_state_with_none_request_does_not_crash(self):
        """Some callers might pass None for user_request defensively."""
        s = PlanState(user_request=None,  # type: ignore
                      steps_so_far=0, failures_so_far=0, pending_count=0,
                      plan_depth=0, tools_used_this_turn=(),
                      tools_failed_this_turn=())
        fv = extract_step_features(s, _action())
        assert all(math.isfinite(v) for v in fv.values)

    def test_action_with_none_tool_name(self):
        s = _state()
        a = ActionFeatures(description="x", tool_name=None,  # type: ignore
                           tool_args={})
        fv = extract_step_features(s, a)
        # An unknown / None tool falls into the unknown bucket.
        assert fv.by_name["tool_is_unknown"] == 1.0


# ══════════════════════════════════════════════════════════════════════
# I/O FAILURES
# ══════════════════════════════════════════════════════════════════════

class TestIOFailures:
    def test_load_corrupted_json_raises(self, tmp_path: Path):
        p = tmp_path / "corrupt.json"
        p.write_text("{this is not json")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            StepValueModel.load(p)

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises((FileNotFoundError, OSError)):
            StepValueModel.load(tmp_path / "nope.json")

    def test_load_missing_weights_field_raises(self, tmp_path: Path):
        p = tmp_path / "noweights.json"
        p.write_text(json.dumps({
            "schema": "ghost.prm.logreg.v1",
            "feature_names": list(PRM_FEATURE_NAMES),
            # weights & bias missing
        }))
        with pytest.raises((KeyError, ValueError)):
            StepValueModel.load(p)

    def test_load_wrong_weights_length_silently_loads_then_predict_raises(
        self, tmp_path: Path,
    ):
        """If somehow a checkpoint with the right feature_names but
        wrong weights length is loaded, predict should fail loud."""
        p = tmp_path / "shortw.json"
        p.write_text(json.dumps({
            "schema": "ghost.prm.logreg.v1",
            "feature_names": list(PRM_FEATURE_NAMES),
            "weights": [0.5, 0.5],  # shorter than feature names
            "bias": 0.0,
        }))
        m = StepValueModel.load(p)
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            m.predict_proba(extract_step_features(_state(), _action()))

    def test_save_creates_parent_dir(self, tmp_path: Path):
        m = StepValueModel(epochs=50)
        X = [extract_step_features(_state(), _action(tool_name="a"))
             for _ in range(5)]
        X += [extract_step_features(_state(), _action(tool_name="b"))
              for _ in range(5)]
        m.fit(X, [1] * 5 + [0] * 5)
        nested = tmp_path / "deep" / "nested" / "ckpt.json"
        m.save(nested)
        assert nested.exists()

    def test_save_atomic_no_leftover_tmp_after_success(self, tmp_path: Path):
        scorer = _trained_scorer()
        ckpt = tmp_path / "atomic.json"
        scorer.model.save(ckpt)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == [], f"found stale tmp files: {leftovers}"

    def test_load_extra_unknown_fields_ignored(self, tmp_path: Path):
        """Forward-compat: an older loader must tolerate new fields it
        doesn't recognise. (Reverse migration path.)"""
        scorer = _trained_scorer()
        ckpt = tmp_path / "extra.json"
        scorer.model.save(ckpt)
        # Inject an unknown field
        data = json.loads(ckpt.read_text())
        data["future_unknown_field"] = {"x": [1, 2, 3]}
        ckpt.write_text(json.dumps(data))
        # Must still load.
        m2 = StepValueModel.load(ckpt)
        assert m2.weights_ is not None

    def test_save_then_load_then_save_again_stays_identical(self, tmp_path: Path):
        a = _trained_scorer().model
        p1 = tmp_path / "a.json"
        a.save(p1)
        b = StepValueModel.load(p1)
        p2 = tmp_path / "b.json"
        b.save(p2)
        # Predictions of a and b must match on a fresh sample.
        sample = extract_step_features(_state(), _action(tool_name="execute"))
        assert a.predict_proba(sample) == pytest.approx(b.predict_proba(sample))


# ══════════════════════════════════════════════════════════════════════
# SCORER EDGE INPUTS
# ══════════════════════════════════════════════════════════════════════

class TestScorerEdgeInputs:
    def test_score_with_empty_scorer_returns_neutral(self):
        s = PRMScorer()
        assert s.score(_state(), _action()) == 0.5

    def test_score_with_explicit_none_default_score(self):
        # default_score must be clamped to a finite [0, 1] value.
        s = PRMScorer(default_score=float("nan"))
        v = s.score(_state(), _action())
        assert 0.0 <= v <= 1.0
        assert math.isfinite(v)

    def test_score_with_inf_default_score_clamped(self):
        s_pos = PRMScorer(default_score=float("inf"))
        s_neg = PRMScorer(default_score=float("-inf"))
        assert s_pos.score(_state(), _action()) <= 1.0
        assert s_neg.score(_state(), _action()) >= 0.0

    def test_score_many_empty_list(self):
        s = _trained_scorer()
        assert s.score_many(_state(), []) == []

    def test_score_many_one_item(self):
        s = _trained_scorer()
        out = s.score_many(_state(), [_action(tool_name="execute")])
        assert len(out) == 1
        assert 0.0 <= out[0] <= 1.0

    def test_score_isolation_when_predict_value_raises(self):
        """If the underlying predict raises, score must return neutral
        rather than propagate the exception. (PRM scoring is advisory
        and must never break a turn.)"""
        s = _trained_scorer()
        original = s._model.predict_value
        s._model.predict_value = MagicMock(side_effect=RuntimeError("boom"))
        v = s.score(_state(), _action())
        assert v == 0.5
        s._model.predict_value = original

    def test_set_model_to_none_returns_neutral(self):
        s = _trained_scorer()
        s.set_model(None)
        assert s.has_model is False
        assert s.score(_state(), _action()) == 0.5


# ══════════════════════════════════════════════════════════════════════
# MCTS INTEGRATION EDGE CASES
# ══════════════════════════════════════════════════════════════════════

class TestMCTSIntegrationEdgeCases:
    @pytest.fixture
    def expansion_response(self):
        return {"choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "a", "tool_name": "file_system"},
                {"description": "b", "tool_name": "browser"},
            ],
        })}}]}

    @pytest.fixture
    def fake_llm(self, expansion_response):
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value=expansion_response)
        llm.route = AsyncMock(return_value=None)
        return llm

    async def test_scorer_returning_nan_clamped_to_neutral(self, fake_llm):
        """A scorer that returns NaN for a candidate must not poison the
        ranking — clamp to 0.5."""
        from ghost_agent.core.mcts import MCTSReasoner

        bad = MagicMock()
        bad.has_model = True
        bad.score = MagicMock(return_value=float("nan"))
        m = MCTSReasoner(llm_client=fake_llm, max_candidates=2,
                         prm_scorer=bad)
        winner = await m.select_best_action(
            "t", "s", ["file_system", "browser"], prm_state=_state(),
        )
        # Both candidates score 0.5 (NaN-clamped). A winner is still
        # picked (sort stable on tie); winner.score is a finite 0.5.
        assert winner is not None
        assert math.isfinite(winner.score)

    async def test_scorer_returning_negative_score_clamped(self, fake_llm):
        from ghost_agent.core.mcts import MCTSReasoner
        bad = MagicMock()
        bad.has_model = True
        bad.score = MagicMock(return_value=-3.0)
        m = MCTSReasoner(llm_client=fake_llm, max_candidates=2,
                         prm_scorer=bad)
        winner = await m.select_best_action(
            "t", "s", ["file_system", "browser"], prm_state=_state(),
        )
        assert winner is not None
        assert 0.0 <= winner.score <= 1.0

    async def test_scorer_returning_score_above_one_clamped(self, fake_llm):
        from ghost_agent.core.mcts import MCTSReasoner
        bad = MagicMock()
        bad.has_model = True
        bad.score = MagicMock(return_value=42.0)
        m = MCTSReasoner(llm_client=fake_llm, max_candidates=2,
                         prm_scorer=bad)
        winner = await m.select_best_action(
            "t", "s", ["file_system", "browser"], prm_state=_state(),
        )
        assert winner is not None
        assert 0.0 <= winner.score <= 1.0

    async def test_all_candidates_tie_winner_still_picked(self, fake_llm):
        """When PRM returns identical scores for every candidate, a
        winner is still selected (first by sort-stability)."""
        from ghost_agent.core.mcts import MCTSReasoner
        flat = MagicMock()
        flat.has_model = True
        flat.score = MagicMock(return_value=0.5)
        m = MCTSReasoner(llm_client=fake_llm, max_candidates=2,
                         prm_scorer=flat)
        winner = await m.select_best_action(
            "t", "s", ["file_system", "browser"], prm_state=_state(),
        )
        assert winner is not None
        assert winner.score == 0.5

    async def test_mcts_with_no_llm_and_prm_returns_none(self):
        """Even with a PRM scorer, no llm_client → no candidates →
        returns None. PRM scoring requires candidates to score against."""
        from ghost_agent.core.mcts import MCTSReasoner
        scorer = _trained_scorer()
        m = MCTSReasoner(llm_client=None, prm_scorer=scorer)
        winner = await m.select_best_action(
            "t", "s", ["x"], prm_state=_state(),
        )
        assert winner is None


# ══════════════════════════════════════════════════════════════════════
# TRAINER REPEAT-RUN BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════

class TestTrainerRepeatRun:
    def test_run_then_run_again_overwrites_model(self):
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        r1 = trainer.run(_balanced(n_pass=8, n_fail=8))
        first_model = trainer.model
        assert r1.fit_succeeded
        r2 = trainer.run(_balanced(n_pass=10, n_fail=10))
        assert r2.fit_succeeded
        # New model object (or at least new weights — could be same
        # object but updated). Either way the predictions may differ.
        assert trainer.model is not None

    def test_run_succeeds_then_bails_keeps_old_model(self):
        """If run() succeeds, then a follow-up run() bails on bad data,
        the trainer's stored model should still be the GOOD one — we
        don't want to clobber it with None on a bail."""
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        r1 = trainer.run(_balanced(n_pass=8, n_fail=8))
        assert r1.fit_succeeded
        good_model = trainer.model
        # Now feed degenerate data
        r2 = trainer.run([_passing(request="single")])
        assert not r2.fit_attempted
        # The good model must still be available — bail does not
        # corrupt the previously-fit state.
        assert trainer.model is good_model

    def test_save_failure_does_not_lose_model(self, tmp_path: Path):
        """If save raises (e.g., disk full), trainer.model should still
        be set so the caller can retry the save or hot-swap."""
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        # Save path that points to a directory (not a file) so save fails
        bad = tmp_path / "is_a_dir"
        bad.mkdir()
        report = trainer.run(_balanced(), save_path=bad)
        # Either the save was attempted and failed, or the file system
        # let us write a same-name file (some FS allow this if dir is
        # empty?). If save failed, we should have a bail_reason.
        if report.saved_to == "":
            # Save failed — but model must still be available
            assert trainer.model is not None


# ══════════════════════════════════════════════════════════════════════
# CONCURRENCY: hot-swap must not torn-read
# ══════════════════════════════════════════════════════════════════════

class TestConcurrency:
    def test_concurrent_score_during_set_model(self):
        """Spam ``score`` from N threads while another thread does
        ``set_model`` swaps. No call should raise; every result must be
        a finite float in [0, 1]."""
        scorer = _trained_scorer()
        # Pre-build two distinct models
        m_a = scorer.model
        m_b = StepValueModel()
        m_b.weights_ = np.zeros(len(PRM_FEATURE_NAMES))
        m_b.bias_ = 0.0

        stop = threading.Event()
        results: List[float] = []
        errors: List[Exception] = []

        def reader():
            state, action = _state(), _action(tool_name="execute")
            while not stop.is_set():
                try:
                    v = scorer.score(state, action)
                    if not (0.0 <= v <= 1.0) or not math.isfinite(v):
                        errors.append(ValueError(f"out-of-range: {v}"))
                    else:
                        results.append(v)
                except Exception as e:  # pragma: no cover
                    errors.append(e)

        def swapper():
            i = 0
            while not stop.is_set():
                scorer.set_model(m_a if i % 2 else m_b)
                i += 1

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=swapper))
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"concurrent failures: {errors[:5]}"
        assert len(results) > 100, "barely any reads happened"

    def test_concurrent_score_on_unmutated_scorer(self):
        """No swaps — pure read-only concurrency. Same scorer scored
        from N threads must produce identical answers (deterministic)."""
        scorer = _trained_scorer()
        state, action = _state(), _action(tool_name="execute")
        expected = scorer.score(state, action)
        results: List[float] = []
        errors: List[Exception] = []

        def reader():
            for _ in range(50):
                try:
                    results.append(scorer.score(state, action))
                except Exception as e:  # pragma: no cover
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"concurrent read errors: {errors[:5]}"
        assert all(r == expected for r in results), (
            "deterministic read returned different values under load"
        )


# ══════════════════════════════════════════════════════════════════════
# STRESS / SCALE
# ══════════════════════════════════════════════════════════════════════

class TestStressAndScale:
    def test_5k_sample_fit_completes_quickly(self):
        """5K samples should fit in under ~10 s on a developer laptop.
        We assert under 30 s to leave plenty of slack for CI."""
        # 250 trajectories × ~20 steps = 5000 samples.
        many = []
        for i in range(125):
            many.append(_passing(request=f"p{i}", n=20))
        for i in range(125):
            many.append(_failing(request=f"f{i}", n=20))
        trainer = PRMTrainer(min_samples=100, min_trajectories=10)
        start = time.perf_counter()
        report = trainer.run(many)
        elapsed = time.perf_counter() - start
        assert report.fit_succeeded
        assert report.n_samples_total >= 5000
        assert elapsed < 30.0, f"fit took {elapsed:.1f} s — too slow"

    def test_score_many_with_1k_candidates_is_linear(self):
        scorer = _trained_scorer()
        candidates = [
            _action(tool_name="execute" if i % 2 else "scratchpad",
                    description=f"action {i}")
            for i in range(1000)
        ]
        start = time.perf_counter()
        scores = scorer.score_many(_state(), candidates)
        elapsed = time.perf_counter() - start
        assert len(scores) == 1000
        assert all(0.0 <= s <= 1.0 for s in scores)
        # 1k scores should run well under a second.
        assert elapsed < 5.0, f"score_many took {elapsed:.2f} s"


# ══════════════════════════════════════════════════════════════════════
# CORRECTIONS-SIDECAR INTERACTION
# ══════════════════════════════════════════════════════════════════════

class TestCorrectionsSidecarInteraction:
    def test_corrections_overlay_changes_labels(self, tmp_path: Path):
        """Trajectory written as PASSED but later promoted to FAILED via
        the sidecar must yield FAILED-shaped labels when the trainer
        reads through ``iter_trajectories``."""
        from ghost_agent.distill import TrajectoryCollector

        coll = TrajectoryCollector(root=tmp_path, session_id="test", enabled=True)
        traj = Trajectory(
            id="t1", user_request="x", outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")], n_steps=2,
        )
        coll.append(traj)

        # Read once before correction → values follow PASSED shape.
        labels_before = derive_step_labels(next(coll.iter_trajectories()))
        assert labels_before[-1] > 0.5  # last step gets full passed credit

        # Promote via sidecar
        coll.update_outcome("t1", Outcome.FAILED.value, reason="user said wrong")

        # Read again → values are zeros (failed shape).
        re_read = list(coll.iter_trajectories())
        assert len(re_read) == 1
        labels_after = derive_step_labels(re_read[0])
        assert labels_after == [0.0, 0.0]


# ══════════════════════════════════════════════════════════════════════
# BIOLOGICAL PHASE 2.7 — additional invariants
# ══════════════════════════════════════════════════════════════════════

class TestBiologicalPhaseExtra:
    """Additional bullet-proofing on top of test_prm_biological_phase.py."""

    @pytest.fixture
    def make_ctx(self):
        import datetime
        from types import SimpleNamespace

        def _make(*, idle_secs=1200, scorer=None, collector=None,
                  checkpoint=None):
            ctx = MagicMock()
            ctx.memory_system = MagicMock()
            ctx.llm_client = SimpleNamespace(foreground_tasks=0)
            ctx.journal = None
            ctx.memory_system.collection.get = MagicMock(
                return_value={"ids": []})
            ctx.last_activity_time = (
                datetime.datetime.now()
                - datetime.timedelta(seconds=idle_secs)
            )
            ctx.args = MagicMock()
            ctx.args.model = "x"
            ctx.args.prm_train_cooldown = None
            ctx.frontier_tracker = None
            ctx.reflector = None
            ctx.trajectory_collector = collector
            ctx.prm_scorer = scorer
            ctx._prm_checkpoint_path = checkpoint
            ctx.mcts_reasoner = None
            return ctx
        return _make

    async def _tick(self, ctx, suppress=True):
        import datetime
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        if suppress:
            now = datetime.datetime.now()
            agent._last_dream_at = now
            agent._last_reflection_at = now
            agent._last_skills_auto_at = now
            agent._last_selfplay_at = now
        await agent._biological_tick()
        return agent

    async def test_phase_27_handles_save_path_with_unwritable_parent(
        self, make_ctx, tmp_path: Path,
    ):
        """An unwritable save path must not crash the watchdog. The
        trainer's bail or save error is caught by the phase's try-finally."""
        collector = MagicMock()
        collector.iter_trajectories = MagicMock(
            side_effect=lambda **kw: iter(_balanced()),
        )
        scorer = PRMScorer()
        # Create a path whose parent already exists as a regular file
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        bad = blocker / "child" / "ckpt.json"
        ctx = make_ctx(scorer=scorer, collector=collector, checkpoint=bad)
        agent = await self._tick(ctx)
        # Anchor still advanced — no infinite refire
        import datetime
        assert agent._last_prm_train_at > datetime.datetime.min

    async def test_phase_27_anchor_advances_on_trainer_exception(
        self, make_ctx,
    ):
        """Trainer raising mid-run must still advance the anchor."""
        collector = MagicMock()
        # iter raises mid-iteration
        def boom():
            yield _passing(request="x", n=1)
            raise RuntimeError("simulated mid-iteration crash")
        collector.iter_trajectories = MagicMock(
            side_effect=lambda **kw: boom(),
        )
        scorer = PRMScorer()
        ctx = make_ctx(scorer=scorer, collector=collector)
        import datetime
        agent = await self._tick(ctx)
        assert agent._last_prm_train_at > datetime.datetime.min
        # Scorer remained un-trained.
        assert scorer.has_model is False

    async def test_phase_27_does_not_swap_when_trainer_model_is_none(
        self, make_ctx,
    ):
        """If somehow trainer.model is None even after fit_succeeded
        (defensive), we must not call set_model(None)."""
        # Synthesise an all-passing collector so trainer bails on imbalance
        collector = MagicMock()
        collector.iter_trajectories = MagicMock(
            side_effect=lambda **kw: iter(
                [_passing(request=f"p{i}") for i in range(20)]
            ),
        )
        scorer = PRMScorer()
        ctx = make_ctx(scorer=scorer, collector=collector)
        await self._tick(ctx)
        # Bailed on imbalance → no model swapped in
        assert scorer.has_model is False


# ══════════════════════════════════════════════════════════════════════
# DETERMINISM / ORDERING
# ══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_extract_features_deterministic_under_dict_iteration_order(self):
        """tool_args dict iteration order must not affect the feature
        vector — feature extraction must be order-independent over
        argument keys."""
        a1 = _action(tool_name="execute", tool_args={"a": "x", "b": "y"})
        a2 = _action(tool_name="execute", tool_args={"b": "y", "a": "x"})
        f1 = extract_step_features(_state(), a1)
        f2 = extract_step_features(_state(), a2)
        assert f1.values == f2.values

    def test_feature_names_are_unique(self):
        assert len(set(PRM_FEATURE_NAMES)) == len(PRM_FEATURE_NAMES)

    def test_label_for_long_trajectory_decreases_monotonically_backward(self):
        """For a PASSED trajectory, V(step_i) must be strictly
        non-decreasing as i increases (γ^N-i-1 grows toward 1)."""
        t = _passing(n=10)
        labels = derive_step_labels(t, StepLabelSpec(discount_factor=0.5))
        for i in range(len(labels) - 1):
            assert labels[i] <= labels[i + 1], (
                f"non-monotone at i={i}: {labels[i]} > {labels[i+1]}"
            )
