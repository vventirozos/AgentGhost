"""Embedder degradation guard (log-audit fix, 2026-06-11).

sentence-transformers does NOT raise when it can't resolve the
all-MiniLM-L6-v2 config (HF unreachable / not forced offline) — it logs
"Creating a new one with mean pooling" and silently builds an UNTRAINED
model that returns 384-d but non-normalised vectors, poisoning retrieval.
VectorMemory now probes the embedder at boot and fails loud. The decision
logic is a pure function so it's testable without a real model.
"""

import math

from ghost_agent.memory.vector import (
    EXPECTED_EMBED_DIM,
    _embedding_degradation_reason,
)


def _unit_vector(dim=EXPECTED_EMBED_DIM):
    # a finite, L2-normalised 384-d vector (looks like the trained model)
    v = [1.0] + [0.0] * (dim - 1)
    return v


def test_trained_embedding_passes():
    assert _embedding_degradation_reason(_unit_vector()) is None


def test_normalised_nontrivial_vector_passes():
    raw = [0.3, -0.4, 0.5] + [0.0] * (EXPECTED_EMBED_DIM - 3)
    n = math.sqrt(sum(x * x for x in raw))
    unit = [x / n for x in raw]
    assert _embedding_degradation_reason(unit) is None


def test_unnormalised_meanpooling_fallback_is_flagged():
    # the degraded fallback returns large-norm vectors (~7.7 observed)
    big = [2.0] * EXPECTED_EMBED_DIM  # norm = 2*sqrt(384) ≈ 39
    reason = _embedding_degradation_reason(big)
    assert reason is not None
    assert "normalis" in reason.lower()


def test_wrong_dimension_is_flagged():
    reason = _embedding_degradation_reason([1.0, 0.0, 0.0])  # 3-d
    assert reason is not None
    assert "dimension" in reason.lower()


def test_none_vector_is_flagged():
    assert _embedding_degradation_reason(None) is not None


def test_non_finite_vector_is_flagged():
    bad = _unit_vector()
    bad[10] = float("inf")
    reason = _embedding_degradation_reason(bad)
    assert reason is not None
    assert "finite" in reason.lower()


def test_non_numeric_vector_is_flagged():
    assert _embedding_degradation_reason(["a", "b"]) is not None


def test_tolerance_allows_small_norm_drift():
    # numerically near-unit (norm ~1.05) should still pass
    raw = _unit_vector()
    raw[0] = 1.05
    assert _embedding_degradation_reason(raw) is None
