"""Unit tests for ghost_agent.core.confidence — composite confidence."""

import math

import pytest

from ghost_agent.core.confidence import CompositeConfidence, ConfidenceReading


class TestCompositeConfidence:
    def test_defaults_sane(self):
        cc = CompositeConfidence()
        r = cc.score(normalised_entropy=0.0, competence_p_success=1.0,
                     n_observations=100)
        # Pristine signals (no entropy, perfect competence) → ~1.0
        assert r.composite > 0.95

    def test_high_entropy_pulls_composite_down(self):
        cc = CompositeConfidence()
        low = cc.score(normalised_entropy=0.9, competence_p_success=0.8,
                       n_observations=100)
        high = cc.score(normalised_entropy=0.1, competence_p_success=0.8,
                        n_observations=100)
        assert high.composite > low.composite

    def test_low_competence_pulls_composite_down(self):
        cc = CompositeConfidence()
        low = cc.score(normalised_entropy=0.1, competence_p_success=0.1,
                       n_observations=100)
        high = cc.score(normalised_entropy=0.1, competence_p_success=0.9,
                        n_observations=100)
        assert high.composite > low.composite

    def test_below_threshold_flag(self):
        cc = CompositeConfidence(threshold=0.7)
        # Should be below threshold
        r = cc.score(normalised_entropy=0.9, competence_p_success=0.4,
                     n_observations=100)
        assert r.below_threshold is True

    def test_above_threshold_flag(self):
        cc = CompositeConfidence(threshold=0.3)
        r = cc.score(normalised_entropy=0.1, competence_p_success=0.9,
                     n_observations=100)
        assert r.below_threshold is False

    def test_shrinkage_toward_prior_when_few_obs(self):
        cc = CompositeConfidence()
        # With 0 observations the competence component should be 0.5
        r = cc.score(normalised_entropy=0.0, competence_p_success=1.0,
                     n_observations=0)
        # competence_component pulled toward 0.5 → composite ≈ (1 + 0.5)/2 = 0.75
        assert r.composite == pytest.approx(0.75, abs=0.05)
        assert r.competence_component == pytest.approx(0.5, abs=0.05)

    def test_weights_normalise(self):
        # Caller passes weights that don't sum to 1 — should still work
        cc = CompositeConfidence(w_entropy=2.0, w_competence=8.0)
        assert cc.w_entropy + cc.w_competence == pytest.approx(1.0)
        assert cc.w_competence > cc.w_entropy

    def test_zero_weights_default_to_half_half(self):
        cc = CompositeConfidence(w_entropy=0.0, w_competence=0.0)
        assert cc.w_entropy == 0.5
        assert cc.w_competence == 0.5

    def test_nan_input_does_not_propagate(self):
        cc = CompositeConfidence()
        r = cc.score(normalised_entropy=float("nan"),
                     competence_p_success=float("nan"),
                     n_observations=100)
        assert math.isfinite(r.composite)
        assert 0.0 <= r.composite <= 1.0

    def test_out_of_range_inputs_clamped(self):
        cc = CompositeConfidence()
        r = cc.score(normalised_entropy=-1.0, competence_p_success=5.0,
                     n_observations=100)
        assert 0.0 <= r.composite <= 1.0
        assert r.entropy_component == pytest.approx(1.0)  # 1 - 0
        assert r.competence_component == pytest.approx(1.0)  # 1 capped

    def test_threshold_clamped(self):
        cc = CompositeConfidence(threshold=-99.0)
        assert cc.threshold == 0.0
        cc = CompositeConfidence(threshold=99.0)
        assert cc.threshold == 1.0
