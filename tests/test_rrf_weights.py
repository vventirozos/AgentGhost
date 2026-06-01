"""Tests for learned RRF intent→source weights (quick win e):
core/rrf_weights.py (load/fit/save) + the bus integration that uses an
injected matrix while keeping the classmethod back-compatible."""

import pytest

from ghost_agent.core.rrf_weights import (
    DEFAULT_INTENT_WEIGHTS,
    load_intent_weights,
    fit_intent_weights,
    save_intent_weights,
    WEIGHT_MIN,
    WEIGHT_MAX,
)
from ghost_agent.core.bus import MemoryBus


# ──────────────────────────────────────────────────────────────────────
# load / save
# ──────────────────────────────────────────────────────────────────────

def test_load_absent_returns_none(tmp_path):
    assert load_intent_weights(tmp_path / "nope.json") is None


def test_load_wrong_schema_returns_none(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"schema": "bad", "weights": {"factual": {"graph": 2.0}}}')
    assert load_intent_weights(p) is None


def test_load_corrupt_returns_none(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("{ not json")
    assert load_intent_weights(p) is None


def test_save_load_roundtrip_and_clamp(tmp_path):
    p = tmp_path / "w.json"
    save_intent_weights(p, {"factual": {"graph": 99.0, "vector": -5.0}})
    loaded = load_intent_weights(p)
    assert loaded is not None
    # clamped into the sane band on load
    assert loaded["factual"]["graph"] == WEIGHT_MAX
    assert loaded["factual"]["vector"] == WEIGHT_MIN


# ──────────────────────────────────────────────────────────────────────
# fit
# ──────────────────────────────────────────────────────────────────────

def test_fit_upweights_successful_source():
    # Under 'factual', graph correlates with success, skill with failure.
    obs = []
    for _ in range(10):
        obs.append(("factual", "graph", True))
        obs.append(("factual", "skill", False))
    fitted = fit_intent_weights(obs, min_obs_per_cell=3)
    assert fitted["factual"]["graph"] > fitted["factual"]["skill"]
    assert fitted["factual"]["graph"] > 1.0
    assert fitted["factual"]["skill"] < 1.0


def test_fit_thin_cells_keep_base():
    obs = [("procedural", "skill", True)]  # only 1 sample < floor
    fitted = fit_intent_weights(obs, min_obs_per_cell=3)
    assert fitted["procedural"]["skill"] == DEFAULT_INTENT_WEIGHTS["procedural"]["skill"]


def test_fit_returns_full_matrix():
    fitted = fit_intent_weights([], min_obs_per_cell=3)
    assert set(fitted.keys()) == set(DEFAULT_INTENT_WEIGHTS.keys())


# ──────────────────────────────────────────────────────────────────────
# bus integration
# ──────────────────────────────────────────────────────────────────────

def _g(): return [{"source": "graph", "text": "G"}]
def _v(): return [{"source": "vector", "text": "V"}]


def test_fusion_backcompat_uses_defaults():
    # No weight_overrides → factual default (graph 2.0 > vector 1.0) → G first.
    fused = MemoryBus._reciprocal_rank_fusion([_g(), _v()], k=60, intent="factual")
    assert fused[0][0]["source"] == "graph"


def test_fusion_override_flips_ranking():
    override = {"factual": {"graph": 0.1, "vector": 3.0, "skill": 0.5, "episodic": 0.3}}
    fused = MemoryBus._reciprocal_rank_fusion(
        [_g(), _v()], k=60, intent="factual", weight_overrides=override,
    )
    assert fused[0][0]["source"] == "vector"  # learned weights win


def test_bus_instance_threads_learned_weights():
    override = {"factual": {"graph": 0.1, "vector": 3.0, "skill": 0.5, "episodic": 0.3}}
    bus = MemoryBus(intent_weights=override)
    assert bus._intent_weights == override
    # A bus with no override keeps defaults (None).
    assert MemoryBus()._intent_weights is None
