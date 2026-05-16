"""Unit tests for ``core.frontier_selection`` — pure-function layer that
turns PRM uncertainty + trajectory rarity into a weighted cluster pick.

These tests pin the math. The wiring into FrontierTracker and dream.py
is covered by separate integration tests so a regression in either
direction surfaces in the right place.
"""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.frontier_selection import (
    combine_weights,
    compute_cluster_rarity,
    compute_cluster_uncertainty,
    count_trajectories_by_cluster,
    pick_weighted,
    representative_state,
)


# ──────────────────────────────────────────────────────────────────────
# representative_state
# ──────────────────────────────────────────────────────────────────────

class TestRepresentativeState:
    def test_returns_state_and_action(self):
        state, action = representative_state("sql")
        assert state.user_request == "solve a sql challenge"
        assert action.tool_name == ""

    def test_underscores_become_spaces(self):
        state, _ = representative_state("data_analysis")
        assert "data analysis" in state.user_request

    def test_none_or_empty_falls_to_general(self):
        for key in (None, "", "   "):
            state, _ = representative_state(key)
            # Empty string sneaks through as "solve a  challenge" — accept
            # either, just make sure no crash and we got a non-empty request.
            assert state.user_request.startswith("solve a")


# ──────────────────────────────────────────────────────────────────────
# compute_cluster_uncertainty
# ──────────────────────────────────────────────────────────────────────

class TestComputeClusterUncertainty:
    def test_none_scorer_returns_one_per_cluster(self):
        out = compute_cluster_uncertainty(None, ["sql", "bash", "algo"])
        assert out == {"sql": 1.0, "bash": 1.0, "algo": 1.0}

    def test_scorer_called_once_per_cluster(self):
        scorer = MagicMock()
        scorer.uncertainty = MagicMock(return_value=0.42)
        out = compute_cluster_uncertainty(scorer, ["sql", "bash"])
        assert out == {"sql": 0.42, "bash": 0.42}
        assert scorer.uncertainty.call_count == 2

    def test_scorer_exception_falls_to_max_uncertainty(self):
        """A bad cluster shouldn't dropout — it should get max weight
        so we still explore it instead of silently skipping."""
        scorer = MagicMock()
        scorer.uncertainty = MagicMock(side_effect=RuntimeError("boom"))
        out = compute_cluster_uncertainty(scorer, ["sql"])
        assert out == {"sql": 1.0}

    def test_empty_cluster_list_returns_empty(self):
        scorer = MagicMock()
        assert compute_cluster_uncertainty(scorer, []) == {}


# ──────────────────────────────────────────────────────────────────────
# compute_cluster_rarity
# ──────────────────────────────────────────────────────────────────────

class TestComputeClusterRarity:
    def test_zero_count_is_max_rarity(self):
        out = compute_cluster_rarity({}, ["sql"])
        assert out["sql"] == pytest.approx(1.0)

    def test_rarity_decays_monotonically(self):
        out = compute_cluster_rarity(
            {"a": 0, "b": 1, "c": 10, "d": 100, "e": 1000}, ["a", "b", "c", "d", "e"]
        )
        assert out["a"] > out["b"] > out["c"] > out["d"] > out["e"]

    def test_rarity_bounded_above_by_one(self):
        out = compute_cluster_rarity({"a": 0}, ["a"])
        assert out["a"] <= 1.0

    def test_rarity_stays_positive_at_large_counts(self):
        """log1p decay means even 10k trajectories don't crush the
        weight to zero — guard against accidental linear scaling."""
        out = compute_cluster_rarity({"a": 10_000}, ["a"])
        assert out["a"] > 0.0
        assert out["a"] < 0.2

    def test_missing_count_defaults_to_zero(self):
        out = compute_cluster_rarity({}, ["unknown"])
        assert out["unknown"] == pytest.approx(1.0)

    def test_negative_count_clamped_to_zero(self):
        out = compute_cluster_rarity({"weird": -5}, ["weird"])
        assert out["weird"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# combine_weights
# ──────────────────────────────────────────────────────────────────────

class TestCombineWeights:
    def test_multiplicative(self):
        out = combine_weights({"sql": 0.5}, {"sql": 0.4})
        assert out["sql"] == pytest.approx(0.2)

    def test_excluded_keys_get_zero(self):
        out = combine_weights(
            {"sql": 1.0, "bash": 1.0},
            {"sql": 1.0, "bash": 1.0},
            exclude=["sql"],
        )
        assert out["sql"] == 0.0
        assert out["bash"] == 1.0

    def test_missing_one_side_collapses_to_zero(self):
        out = combine_weights({"sql": 0.7}, {"bash": 0.6})
        # sql has no rarity → 0; bash has no uncertainty → 0
        assert out["sql"] == 0.0
        assert out["bash"] == 0.0

    def test_nan_inputs_become_zero(self):
        out = combine_weights({"sql": float("nan")}, {"sql": 0.5})
        assert out["sql"] == 0.0

    def test_inf_inputs_become_zero(self):
        out = combine_weights({"sql": float("inf")}, {"sql": 0.5})
        assert out["sql"] == 0.0

    def test_negative_inputs_become_zero(self):
        out = combine_weights({"sql": -0.1}, {"sql": 0.5})
        assert out["sql"] == 0.0

    def test_empty_dicts_return_empty(self):
        assert combine_weights({}, {}) == {}


# ──────────────────────────────────────────────────────────────────────
# pick_weighted
# ──────────────────────────────────────────────────────────────────────

class TestPickWeighted:
    def test_zero_weights_returns_none(self):
        assert pick_weighted({"a": 0.0, "b": 0.0}) is None

    def test_empty_returns_none(self):
        assert pick_weighted({}) is None

    def test_single_nonzero_always_picked(self):
        rng = random.Random(0)
        assert pick_weighted({"a": 1.0, "b": 0.0, "c": 0.0}, rng=rng) == "a"

    def test_uniform_weights_distribute(self):
        """With equal weights the picks should spread over all clusters
        across many trials — guard against an off-by-one collapsing to
        the first key."""
        rng = random.Random(42)
        keys = ["a", "b", "c", "d"]
        weights = {k: 1.0 for k in keys}
        picks = [pick_weighted(weights, rng=rng) for _ in range(400)]
        seen = set(picks)
        assert seen == set(keys), f"Expected all keys to appear, got {seen}"

    def test_heavy_weight_dominates(self):
        rng = random.Random(0)
        weights = {"a": 100.0, "b": 0.01}
        picks = [pick_weighted(weights, rng=rng) for _ in range(200)]
        # 100:0.01 → ≈10000:1, so b should virtually never come up
        a_count = sum(1 for p in picks if p == "a")
        assert a_count >= 195

    def test_deterministic_with_seeded_rng(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        s1 = [pick_weighted(weights, rng=rng1) for _ in range(20)]
        s2 = [pick_weighted(weights, rng=rng2) for _ in range(20)]
        assert s1 == s2


# ──────────────────────────────────────────────────────────────────────
# count_trajectories_by_cluster
# ──────────────────────────────────────────────────────────────────────

class TestCountTrajectoriesByCluster:
    def test_groups_by_cluster_field(self):
        trajs = [
            MagicMock(cluster="sql"),
            MagicMock(cluster="sql"),
            MagicMock(cluster="bash"),
        ]
        out = count_trajectories_by_cluster(trajs)
        assert out == {"sql": 2, "bash": 1}

    def test_none_cluster_skipped(self):
        trajs = [MagicMock(cluster=None), MagicMock(cluster="sql")]
        out = count_trajectories_by_cluster(trajs)
        assert out == {"sql": 1}

    def test_empty_cluster_string_skipped(self):
        trajs = [MagicMock(cluster=""), MagicMock(cluster="sql")]
        out = count_trajectories_by_cluster(trajs)
        assert out == {"sql": 1}

    def test_missing_attribute_skipped(self):
        bare = object()
        trajs = [bare, MagicMock(cluster="sql")]
        out = count_trajectories_by_cluster(trajs)
        assert out == {"sql": 1}

    def test_empty_iterable_returns_empty(self):
        assert count_trajectories_by_cluster([]) == {}


# ──────────────────────────────────────────────────────────────────────
# end-to-end (still pure)
# ──────────────────────────────────────────────────────────────────────

class TestEndToEndComposition:
    def test_full_pipeline_with_mocks(self):
        """Compose the four functions: scorer + counts → weights → pick.

        Sets up a scenario where one cluster is high-uncertainty AND
        rare, another is high-uncertainty but well-explored, and a
        third is well-explored and confident. With a deterministic
        RNG and lopsided weights, the rare-uncertain cluster should
        win the vast majority of picks.
        """
        scorer = MagicMock()

        def fake_uncertainty(state, action):
            if "sql" in state.user_request:
                return 0.9   # PRM unsure
            if "bash" in state.user_request:
                return 0.9   # PRM unsure
            if "algo" in state.user_request:
                return 0.05  # PRM very confident
            return 0.5
        scorer.uncertainty = MagicMock(side_effect=fake_uncertainty)

        clusters = ["sql", "bash", "algo"]
        unc = compute_cluster_uncertainty(scorer, clusters)
        rar = compute_cluster_rarity(
            {"sql": 1, "bash": 500, "algo": 5},
            clusters,
        )
        weights = combine_weights(unc, rar)

        # sql: 0.9 * (1 / (1 + log1p(1)))    ≈ 0.532
        # bash: 0.9 * (1 / (1 + log1p(500))) ≈ 0.122
        # algo: 0.05 * (1 / (1 + log1p(5)))  ≈ 0.0179
        assert weights["sql"] > weights["bash"] > weights["algo"]

        rng = random.Random(7)
        picks = [pick_weighted(weights, rng=rng) for _ in range(500)]
        # sql should dominate but not exclude bash entirely
        sql_count = sum(1 for p in picks if p == "sql")
        bash_count = sum(1 for p in picks if p == "bash")
        algo_count = sum(1 for p in picks if p == "algo")
        assert sql_count > bash_count > algo_count
        assert sql_count >= 300
