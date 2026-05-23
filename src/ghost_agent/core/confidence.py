"""Composite confidence scorer — roadmap phase 2.4.

Fuses the two objective calibration signals the agent now produces:

  * Token-level normalised Shannon entropy   ``e in [0, 1]``
    (1 = maximally uncertain, 0 = pinpoint deterministic).
  * Per-domain capability prior              ``p in [0, 1]``
    (posterior mean p(success) from ``memory.competence``).

The composite is:

    C = w_e * (1 - e) + w_c * p     (default weights 0.5 / 0.5)

with ``w_e + w_c = 1`` enforced and both clamped into [0, 1] so a
caller that passes weird inputs gets a graceful degradation rather
than a NaN cascading into the routing layer.

A typical threshold τ ≈ 0.55–0.6 is what flips the agent over to the
dual-solver arbiter (``core.arbiter``). Values are advisory; the
caller decides the action policy.

Why not just take the geometric mean? Because the two signals are
asymmetric — high entropy means "this generation is uncertain" but
low domain competence means "the model is *historically* unreliable
here". Treating them as weighted additive lets the operator tune the
balance per-deployment (e.g. an edge build with a thin model raises
``w_c`` because token entropy is noisier on small models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConfidenceReading:
    """The composite output. ``below_threshold`` is the precomputed
    decision callers actually use — keeping it on the dataclass means
    the threshold convention lives in one place."""

    composite: float
    entropy_component: float       # (1 - normalised_entropy)
    competence_component: float    # p(success) for the domain
    threshold: float
    below_threshold: bool


class CompositeConfidence:
    """Stateless scorer. Constructed once per agent so the weights
    and threshold can be reconfigured at runtime without rewiring
    every call site (e.g. /config to relax τ during a known-noisy
    fine-tune)."""

    DEFAULT_W_ENTROPY = 0.5
    DEFAULT_W_COMPETENCE = 0.5
    DEFAULT_THRESHOLD = 0.55

    def __init__(
        self,
        *,
        w_entropy: float = DEFAULT_W_ENTROPY,
        w_competence: float = DEFAULT_W_COMPETENCE,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = _clamp_unit(threshold)
        we, wc = _normalise_weights(w_entropy, w_competence)
        self.w_entropy = we
        self.w_competence = wc

    def score(self, *, normalised_entropy: float,
              competence_p_success: float,
              n_observations: int = 0) -> ConfidenceReading:
        """Compute one composite reading.

        ``n_observations`` discounts the competence prior: when the
        per-domain cell has too few samples (<5) the competence
        component is shrunk toward 0.5 so it doesn't dominate before
        the prior has earned its weight. This is the "calibration of
        calibration" check the doc's Level-1 critique calls out.
        """
        e = _clamp_unit(normalised_entropy)
        p = _clamp_unit(competence_p_success)
        if n_observations < 5:
            # Shrinkage toward the neutral prior; coefficient is
            # n/(n+5) so 0 obs → fully neutral, 5 obs → halfway.
            n = max(0, int(n_observations))
            shrink = n / (n + 5.0) if (n + 5.0) > 0 else 0.0
            p = shrink * p + (1.0 - shrink) * 0.5
        entropy_component = 1.0 - e
        composite = self.w_entropy * entropy_component + self.w_competence * p
        composite = _clamp_unit(composite)
        return ConfidenceReading(
            composite=composite,
            entropy_component=entropy_component,
            competence_component=p,
            threshold=self.threshold,
            below_threshold=composite < self.threshold,
        )


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _clamp_unit(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    import math
    if not math.isfinite(v):
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalise_weights(we: float, wc: float) -> tuple:
    we = max(0.0, float(we))
    wc = max(0.0, float(wc))
    total = we + wc
    if total <= 0.0:
        return 0.5, 0.5
    return we / total, wc / total


__all__ = ["CompositeConfidence", "ConfidenceReading"]
