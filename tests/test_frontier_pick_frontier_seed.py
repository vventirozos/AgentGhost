"""Integration tests for ``FrontierTracker.pick_frontier_seed``.

These exercise the bridge between the pure ``core.frontier_selection``
math and the on-disk frontier state: saturation exclusion, fallback
paths, and the dict-shape contract that ``dream.synthetic_self_play``
depends on.
"""

from __future__ import annotations

import random

import pytest

from ghost_agent.memory.frontier import FrontierTracker


class TestPickFrontierSeedFallbacks:
    def test_empty_signals_fall_back_to_pick_seed(self, tmp_path, monkeypatch):
        ft = FrontierTracker(tmp_path)
        called = {"n": 0}
        orig = ft.pick_seed

        def spy(*a, **kw):
            called["n"] += 1
            return orig(*a, **kw)
        monkeypatch.setattr(ft, "pick_seed", spy)

        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster=None,
            rarity_by_cluster=None,
        )
        assert called["n"] == 1
        assert isinstance(seed, dict)
        assert "mode" in seed

    def test_both_empty_dicts_fall_back(self, tmp_path, monkeypatch):
        ft = FrontierTracker(tmp_path)
        called = {"n": 0}
        orig = ft.pick_seed
        monkeypatch.setattr(
            ft, "pick_seed",
            lambda *a, **kw: (called.update(n=called["n"] + 1), orig(*a, **kw))[1],
        )
        ft.pick_frontier_seed(
            uncertainty_by_cluster={},
            rarity_by_cluster={},
        )
        assert called["n"] == 1

    def test_uniform_sample_floor_tagged(self, tmp_path, monkeypatch):
        """When the uniform-sample dice roll hits, the seed should be
        whatever pick_seed returns BUT tagged with frontier_fallback
        so logs/tests can attribute it."""
        ft = FrontierTracker(tmp_path)
        # Force the dice roll to hit.
        monkeypatch.setattr(random, "random", lambda: 0.0)
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 1.0},
            rarity_by_cluster={"sql": 1.0},
            uniform_sample_prob=0.5,
        )
        assert seed.get("frontier_fallback") == "uniform_sample"

    def test_all_zero_weights_fall_back(self, tmp_path, monkeypatch):
        """If both signals produce zero for every cluster, the picker
        bails to pick_seed rather than picking nothing."""
        ft = FrontierTracker(tmp_path)
        # Force past the uniform-sample roll.
        monkeypatch.setattr(random, "random", lambda: 0.99)
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 0.0, "bash": 0.0},
            rarity_by_cluster={"sql": 0.0, "bash": 0.0},
            uniform_sample_prob=0.2,
        )
        assert seed.get("frontier_fallback") == "no_positive_weight"


class TestPickFrontierSeedHappyPath:
    def test_returns_weighted_pick_with_full_shape(self, tmp_path, monkeypatch):
        ft = FrontierTracker(tmp_path)
        # Force past uniform-sample, force pick_weighted to settle on "sql".
        monkeypatch.setattr(random, "random", lambda: 0.99)
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 0.9, "bash": 0.01},
            rarity_by_cluster={"sql": 0.9, "bash": 0.01},
            uniform_sample_prob=0.2,
        )
        assert seed["mode"] == "frontier_weighted"
        assert seed["cluster_key"] == "sql"
        assert seed["weight"] == pytest.approx(0.81)
        assert seed["uncertainty"] == pytest.approx(0.9)
        assert seed["rarity"] == pytest.approx(0.9)
        assert "difficulty_tier" in seed
        assert "FRONTIER TARGET (PRM-weighted)" in seed["hint"]
        assert "saturated_clusters" in seed

    def test_excludes_saturated_clusters(self, tmp_path, monkeypatch):
        ft = FrontierTracker(tmp_path)
        # Seed sql as saturated: 2 first-try wins, delta=0.
        ft.record_run("sql", "c1", 1, True, 100)
        ft.record_run("sql", "c2", 1, True, 100)
        assert "sql" in ft.list_saturated_clusters()

        monkeypatch.setattr(random, "random", lambda: 0.99)
        # Give sql the strongest signal — saturation should still exclude it.
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 1.0, "bash": 0.5},
            rarity_by_cluster={"sql": 1.0, "bash": 0.5},
            uniform_sample_prob=0.2,
        )
        assert seed["cluster_key"] != "sql"
        assert seed["cluster_key"] == "bash"
        assert "sql" in seed["saturated_clusters"]

    def test_no_positive_weight_after_saturation_falls_back(self, tmp_path, monkeypatch):
        """The only cluster with signal is saturated → fall back."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 100)
        ft.record_run("sql", "c2", 1, True, 100)
        monkeypatch.setattr(random, "random", lambda: 0.99)
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 1.0},
            rarity_by_cluster={"sql": 1.0},
            uniform_sample_prob=0.2,
        )
        assert seed.get("frontier_fallback") == "no_positive_weight"


class TestPickFrontierSeedShapeContract:
    """The dream.py call site assumes a stable dict shape. Pin it so a
    refactor here surfaces as a clear test failure."""

    REQUIRED_KEYS = {"mode", "cluster_key", "hint"}

    def test_frontier_weighted_pick_has_required_keys(self, tmp_path, monkeypatch):
        ft = FrontierTracker(tmp_path)
        monkeypatch.setattr(random, "random", lambda: 0.99)
        seed = ft.pick_frontier_seed(
            uncertainty_by_cluster={"sql": 0.8},
            rarity_by_cluster={"sql": 0.8},
            uniform_sample_prob=0.2,
        )
        assert self.REQUIRED_KEYS.issubset(seed.keys())

    def test_fallback_pick_has_required_keys(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        seed = ft.pick_frontier_seed()
        assert self.REQUIRED_KEYS.issubset(seed.keys())
