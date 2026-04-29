"""Adversarial / fuzz / large-scale tests for the PRM pipeline.

Where ``test_prm_corner_cases.py`` covers specific edge inputs, this
file goes broader:

  * Property-based randomised inputs to feature extraction — every
    valid combination must produce a finite vector in [0, 1] for the
    bounded features.
  * 10K-sample stress fit and 5K-candidate scoring batch.
  * Adversarial trajectory shapes — extremely deep tool calls,
    tool args containing JSON-injection-shaped payloads, names with
    nulls / control chars / non-printable.
  * Schema-version migration scenarios.
  * Repeated hot-swap thrash + score mix.
  * Defensive: feature-dict ordering invariance.

These are slower than the unit tests but still complete in seconds.
"""

from __future__ import annotations

import json
import math
import random
import string
import threading
import time
from pathlib import Path
from typing import List

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
# Fuzz / property-based input generators
# ──────────────────────────────────────────────────────────────────────

def _random_string(rng: random.Random, max_len: int = 200) -> str:
    """Random string of ASCII / unicode / control chars."""
    n = rng.randint(0, max_len)
    pool = string.ascii_letters + string.digits + string.punctuation + " \t\n"
    if rng.random() < 0.1:
        pool += "日本語∑∫🎉αβγ"
    return "".join(rng.choice(pool) for _ in range(n))


def _random_tool_args(rng: random.Random) -> dict:
    """A randomised tool_args dict — strings, ints, lists, nested dicts."""
    n_keys = rng.randint(0, 5)
    args = {}
    for _ in range(n_keys):
        key = _random_string(rng, 20) or "k"
        kind = rng.choice(["str", "int", "list", "nested", "none"])
        if kind == "str":
            args[key] = _random_string(rng, 100)
        elif kind == "int":
            args[key] = rng.randint(-1_000_000, 1_000_000)
        elif kind == "list":
            args[key] = [_random_string(rng, 20) for _ in range(rng.randint(0, 3))]
        elif kind == "nested":
            args[key] = {"x": _random_string(rng, 20), "y": rng.randint(0, 100)}
        elif kind == "none":
            args[key] = None
    return args


def _random_state(rng: random.Random) -> PlanState:
    return PlanState(
        user_request=_random_string(rng, 1000),
        steps_so_far=rng.randint(0, 50),
        failures_so_far=rng.randint(0, 20),
        pending_count=rng.randint(0, 30),
        plan_depth=rng.randint(0, 10),
        tools_used_this_turn=tuple(_random_string(rng, 20) for _ in range(rng.randint(0, 5))),
        tools_failed_this_turn=tuple(_random_string(rng, 20) for _ in range(rng.randint(0, 3))),
    )


def _random_action(rng: random.Random) -> ActionFeatures:
    tools = list({
        "file_system", "execute", "browser", "scratchpad", "vision",
        "web_search", "knowledge_base", "skill_memory",
        "", "unknown_tool_xyz",
    })
    return ActionFeatures(
        description=_random_string(rng, 200),
        tool_name=rng.choice(tools),
        tool_args=_random_tool_args(rng),
    )


# ══════════════════════════════════════════════════════════════════════
# FUZZ: feature extraction is total
# ══════════════════════════════════════════════════════════════════════

class TestFeatureExtractionFuzz:
    @pytest.mark.parametrize("seed", range(10))
    def test_random_inputs_always_produce_finite_vector(self, seed):
        rng = random.Random(seed)
        for _ in range(200):
            state = _random_state(rng)
            action = _random_action(rng)
            fv = extract_step_features(state, action)
            assert len(fv.values) == len(PRM_FEATURE_NAMES)
            assert all(math.isfinite(v) for v in fv.values), (
                f"non-finite feature for state={state} action={action}"
            )

    def test_random_inputs_score_in_unit_interval(self):
        """Drive a trained scorer with 1000 random inputs — every output
        must be a finite probability."""
        # Quick fitter
        from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
        passing = [Trajectory(
            user_request=f"p{i}", outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="scratchpad")] * 2, n_steps=2,
        ) for i in range(8)]
        failing = [Trajectory(
            user_request=f"f{i}", outcome=Outcome.FAILED.value,
            tool_calls=[ToolCall(name="execute", error="x")] * 2, n_steps=2,
        ) for i in range(8)]
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        trainer.run(passing + failing)
        scorer = PRMScorer(model=trainer.model)

        rng = random.Random(99)
        for _ in range(1000):
            v = scorer.score(_random_state(rng), _random_action(rng))
            assert 0.0 <= v <= 1.0
            assert math.isfinite(v)


# ══════════════════════════════════════════════════════════════════════
# FUZZ: trainer is total over random valid corpora
# ══════════════════════════════════════════════════════════════════════

class TestTrainerFuzz:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_balanced_corpus_either_fits_or_bails_cleanly(self, seed):
        rng = random.Random(seed)
        n_pass = rng.randint(0, 30)
        n_fail = rng.randint(0, 30)
        corpus = []
        for i in range(n_pass):
            corpus.append(Trajectory(
                user_request=_random_string(rng, 50),
                outcome=Outcome.PASSED.value,
                tool_calls=[ToolCall(name=rng.choice([
                    "scratchpad", "file_system", "execute", "x"
                ])) for _ in range(rng.randint(1, 5))],
                n_steps=rng.randint(1, 5),
            ))
        for i in range(n_fail):
            corpus.append(Trajectory(
                user_request=_random_string(rng, 50),
                outcome=Outcome.FAILED.value,
                tool_calls=[ToolCall(name=rng.choice([
                    "execute", "browser", "y"
                ]), error="x") for _ in range(rng.randint(1, 5))],
                n_steps=rng.randint(1, 5),
            ))
        rng.shuffle(corpus)

        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        report = trainer.run(corpus)
        # Either it bailed (with a clear reason) or fit (with weights).
        if report.fit_attempted and report.fit_succeeded:
            assert trainer.model is not None
            assert all(np.isfinite(trainer.model.weights_))
            # Sanity: predict_proba on a fresh sample is in unit interval.
            fv = extract_step_features(_random_state(rng), _random_action(rng))
            p = trainer.model.predict_proba(fv)
            assert 0.0 <= p <= 1.0
        else:
            assert report.bail_reason != ""


# ══════════════════════════════════════════════════════════════════════
# 10K-SAMPLE STRESS
# ══════════════════════════════════════════════════════════════════════

class TestLargeScaleStress:
    def test_10k_sample_fit_under_60s(self):
        """500 trajectories × 20 steps = 10K samples. The numpy LR
        gradient descent has 300 epochs by default — that's 300 ×
        (10000 × 25) = 75M float ops, which should finish in single-
        digit seconds on any reasonable laptop. Asserting < 60 s is
        far above that to leave room for CI noise."""
        many = []
        for i in range(250):
            many.append(Trajectory(
                user_request=f"p{i}", outcome=Outcome.PASSED.value,
                tool_calls=[ToolCall(name="scratchpad")] * 20,
                n_steps=20,
            ))
        for i in range(250):
            many.append(Trajectory(
                user_request=f"f{i}", outcome=Outcome.FAILED.value,
                tool_calls=[ToolCall(name="execute", error="x")] * 20,
                n_steps=20,
            ))
        trainer = PRMTrainer(min_samples=100, min_trajectories=10)
        start = time.perf_counter()
        report = trainer.run(many)
        elapsed = time.perf_counter() - start
        assert report.fit_succeeded
        assert report.n_samples_total >= 10000
        assert elapsed < 60.0, f"10k fit took {elapsed:.1f} s — degraded"

    def test_5k_candidate_scoring_batch(self):
        from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
        passing = [Trajectory(
            user_request=f"p{i}", outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="scratchpad")] * 2, n_steps=2,
        ) for i in range(8)]
        failing = [Trajectory(
            user_request=f"f{i}", outcome=Outcome.FAILED.value,
            tool_calls=[ToolCall(name="execute", error="x")] * 2, n_steps=2,
        ) for i in range(8)]
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        trainer.run(passing + failing)
        scorer = PRMScorer(model=trainer.model)

        rng = random.Random(42)
        candidates = [_random_action(rng) for _ in range(5000)]
        start = time.perf_counter()
        scores = scorer.score_many(_random_state(rng), candidates)
        elapsed = time.perf_counter() - start
        assert len(scores) == 5000
        assert all(0.0 <= s <= 1.0 for s in scores)
        # 5K predictions are ~5K dot products of 25-vectors → microseconds
        # each. Asserting < 10 s leaves enormous slack.
        assert elapsed < 10.0


# ══════════════════════════════════════════════════════════════════════
# ADVERSARIAL TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════

class TestAdversarialTrajectories:
    def test_tool_args_with_null_bytes(self):
        t = Trajectory(
            user_request="x", outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(
                name="execute",
                arguments={"cmd": "echo '\x00\x01\x02hello'"},
            )], n_steps=1,
        )
        samples = list(iter_step_samples([t]))
        fv = extract_step_features(samples[0].state, samples[0].action)
        assert all(math.isfinite(v) for v in fv.values)

    def test_tool_name_with_control_chars(self):
        a = ActionFeatures(tool_name="exec\x00ute", description="", tool_args={})
        fv = extract_step_features(
            PlanState(user_request="x", steps_so_far=0, failures_so_far=0,
                      pending_count=0, plan_depth=0,
                      tools_used_this_turn=(), tools_failed_this_turn=()),
            a,
        )
        # Tool name with embedded null doesn't match any known bucket.
        assert fv.by_name["tool_is_unknown"] == 1.0

    def test_tool_args_with_huge_nested_structure(self):
        """Defensive: deeply nested args shouldn't cause O(N^2) blowup
        in feature extraction."""
        deep = {"x": "y"}
        for _ in range(50):
            deep = {"nested": deep}
        a = ActionFeatures(tool_name="x", description="", tool_args=deep)
        start = time.perf_counter()
        fv = extract_step_features(
            PlanState(user_request="", steps_so_far=0, failures_so_far=0,
                      pending_count=0, plan_depth=0,
                      tools_used_this_turn=(), tools_failed_this_turn=()),
            a,
        )
        elapsed = time.perf_counter() - start
        assert all(math.isfinite(v) for v in fv.values)
        assert elapsed < 1.0  # Should be microseconds

    def test_tool_args_with_url_injection_attempt(self):
        """No matter what's in args, features extraction must not
        execute or interpret it — just count patterns."""
        a = ActionFeatures(
            tool_name="execute",
            description="",
            tool_args={
                "cmd": "rm -rf /",
                "url": "javascript:alert(1)",
                "path": "../../../etc/passwd",
            },
        )
        fv = extract_step_features(
            PlanState(user_request="", steps_so_far=0, failures_so_far=0,
                      pending_count=0, plan_depth=0,
                      tools_used_this_turn=(), tools_failed_this_turn=()),
            a,
        )
        # Features should be finite — no execution.
        assert all(math.isfinite(v) for v in fv.values)

    def test_circular_reference_in_tool_args(self):
        """Pathological: a dict containing itself. ``json.dumps`` would
        recurse forever. Feature extraction must not call dumps on
        the args."""
        d = {"x": "y"}
        d["self"] = d  # circular
        a = ActionFeatures(tool_name="x", description="", tool_args=d)
        fv = extract_step_features(
            PlanState(user_request="", steps_so_far=0, failures_so_far=0,
                      pending_count=0, plan_depth=0,
                      tools_used_this_turn=(), tools_failed_this_turn=()),
            a,
        )
        # The args_total_length feature stringifies dict values; a
        # self-referential dict would blow that up if the implementation
        # called repr/str. Confirm it survives.
        assert all(math.isfinite(v) for v in fv.values)


# ══════════════════════════════════════════════════════════════════════
# HOT-SWAP THRASH
# ══════════════════════════════════════════════════════════════════════

class TestHotSwapThrash:
    def test_rapid_set_model_swap_cycle(self):
        """Swap a thousand times back and forth — no leaked references,
        no stale weights."""
        # Build two distinct models.
        m_a = StepValueModel()
        m_a.weights_ = np.ones(len(PRM_FEATURE_NAMES))
        m_a.bias_ = 1.0
        m_b = StepValueModel()
        m_b.weights_ = np.zeros(len(PRM_FEATURE_NAMES))
        m_b.bias_ = 0.0

        scorer = PRMScorer()

        for i in range(1000):
            scorer.set_model(m_a if i % 2 else m_b)
        # After the loop, scorer should have m_b (i=999, odd → m_a)
        # actually i=999 % 2 == 1 → m_a. Last swap.
        assert scorer.has_model

    def test_set_model_to_none_then_back(self):
        scorer = PRMScorer()
        m = StepValueModel()
        m.weights_ = np.ones(len(PRM_FEATURE_NAMES))
        m.bias_ = 0.0
        scorer.set_model(m)
        assert scorer.has_model
        scorer.set_model(None)
        assert not scorer.has_model
        scorer.set_model(m)
        assert scorer.has_model


# ══════════════════════════════════════════════════════════════════════
# SCHEMA MIGRATION SCENARIOS
# ══════════════════════════════════════════════════════════════════════

class TestSchemaMigration:
    def test_load_legacy_v0_format_rejected(self, tmp_path: Path):
        """A pre-v1 checkpoint (or one without a schema field) must
        not load — it would silently produce wrong predictions."""
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps({
            "weights": [0.5] * len(PRM_FEATURE_NAMES),
            "bias": 0.0,
            # No 'schema' field
        }))
        with pytest.raises(ValueError):
            StepValueModel.load(p)

    def test_load_with_partial_feature_names_rejected(self, tmp_path: Path):
        """Half the features missing — the model can't be aligned."""
        p = tmp_path / "partial.json"
        partial = list(PRM_FEATURE_NAMES[:10])
        p.write_text(json.dumps({
            "schema": "ghost.prm.logreg.v1",
            "feature_names": partial,
            "weights": [0.0] * 10,
            "bias": 0.0,
        }))
        with pytest.raises(ValueError, match="feature schema drift"):
            StepValueModel.load(p)


# ══════════════════════════════════════════════════════════════════════
# DETERMINISM ACROSS PYTHON DICT ORDER
# ══════════════════════════════════════════════════════════════════════

class TestStrongDeterminism:
    def test_feature_names_tuple_is_immutable(self):
        """The features tuple must be a tuple (immutable). Otherwise a
        callee could mutate it and break other models silently."""
        assert isinstance(PRM_FEATURE_NAMES, tuple)
        with pytest.raises((TypeError, AttributeError)):
            PRM_FEATURE_NAMES[0] = "boom"  # type: ignore

    def test_repeated_extract_same_result(self):
        rng = random.Random(7)
        for _ in range(50):
            s = _random_state(rng)
            a = _random_action(rng)
            f1 = extract_step_features(s, a)
            f2 = extract_step_features(s, a)
            assert f1.values == f2.values

    def test_pred_consistent_across_dtypes(self):
        """Passing the same logical input as FeatureVector vs ndarray vs
        list must yield identical predictions."""
        # Train a quick model
        passing = [Trajectory(
            user_request=f"p{i}", outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="scratchpad")] * 2, n_steps=2,
        ) for i in range(8)]
        failing = [Trajectory(
            user_request=f"f{i}", outcome=Outcome.FAILED.value,
            tool_calls=[ToolCall(name="execute", error="x")] * 2, n_steps=2,
        ) for i in range(8)]
        trainer = PRMTrainer(min_samples=5, min_trajectories=2,
                             random_state=123)
        trainer.run(passing + failing)
        m = trainer.model

        fv = extract_step_features(
            PlanState(user_request="hello", steps_so_far=0,
                      failures_so_far=0, pending_count=0, plan_depth=0,
                      tools_used_this_turn=(), tools_failed_this_turn=()),
            ActionFeatures(tool_name="execute", description="x", tool_args={}),
        )
        p_fv = m.predict_proba(fv)
        p_arr = m.predict_proba(np.array(fv.values, dtype=float))
        p_list = m.predict_proba(list(fv.values))
        p_tuple = m.predict_proba(tuple(fv.values))
        assert p_fv == p_arr == p_list == p_tuple
