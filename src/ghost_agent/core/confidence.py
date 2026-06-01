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
    the threshold convention lives in one place.

    ``uncertainty_pressure`` is the verbalised-uncertainty signal that
    fed this reading (0 = the agent flagged nothing; → 1 = heavy
    unresolved high-impact unknowns). It is recorded by the calibration
    spine (:mod:`core.calibration`) so the two previously-disjoint
    tracks — objective entropy/competence and the agent's own "I'm not
    sure" flags — are fit together. Defaulted so existing call sites
    and tests that build a reading positionally keep working.
    """

    composite: float
    entropy_component: float       # (1 - normalised_entropy)
    competence_component: float    # p(success) for the domain
    threshold: float
    below_threshold: bool
    uncertainty_pressure: float = 0.0


class CompositeConfidence:
    """Stateless scorer. Constructed once per agent so the weights
    and threshold can be reconfigured at runtime without rewiring
    every call site (e.g. /config to relax τ during a known-noisy
    fine-tune)."""

    DEFAULT_W_ENTROPY = 0.5
    DEFAULT_W_COMPETENCE = 0.5
    DEFAULT_THRESHOLD = 0.55
    # Penalty weight on the verbalised-uncertainty pressure. 0.0 keeps
    # the historical two-signal behaviour byte-for-byte; the calibration
    # spine (:mod:`core.calibration`) raises it only if the agent's own
    # "I'm not sure" flags turn out to predict failure on logged turns.
    DEFAULT_LAMBDA_UNCERTAINTY = 0.0

    def __init__(
        self,
        *,
        w_entropy: float = DEFAULT_W_ENTROPY,
        w_competence: float = DEFAULT_W_COMPETENCE,
        threshold: float = DEFAULT_THRESHOLD,
        lambda_uncertainty: float = DEFAULT_LAMBDA_UNCERTAINTY,
    ):
        self.threshold = _clamp_unit(threshold)
        we, wc = _normalise_weights(w_entropy, w_competence)
        self.w_entropy = we
        self.w_competence = wc
        self.lambda_uncertainty = _clamp_unit(lambda_uncertainty)

    def apply_fitted(self, params) -> None:
        """Hot-swap the threshold/weights/λ from a calibration fit.

        ``params`` is a :class:`core.calibration.FittedParams` (or any
        object exposing the same attributes). Called from the idle
        calibration-refit phase and at startup when a persisted fit
        exists, so a long-running agent's confidence becomes
        empirically calibrated without a restart. Defensive: a malformed
        params object leaves the current settings untouched.
        """
        try:
            we, wc = _normalise_weights(
                float(params.w_entropy), float(params.w_competence)
            )
            self.w_entropy = we
            self.w_competence = wc
            self.threshold = _clamp_unit(float(params.threshold))
            self.lambda_uncertainty = _clamp_unit(
                float(getattr(params, "lambda_uncertainty", 0.0))
            )
        except Exception:  # pragma: no cover — defensive
            pass

    def score(self, *, normalised_entropy: float,
              competence_p_success: float,
              n_observations: int = 0,
              uncertainty_pressure: float = 0.0) -> ConfidenceReading:
        """Compute one composite reading.

        ``n_observations`` discounts the competence prior: when the
        per-domain cell has too few samples (<5) the competence
        component is shrunk toward 0.5 so it doesn't dominate before
        the prior has earned its weight. This is the "calibration of
        calibration" check the doc's Level-1 critique calls out.

        ``uncertainty_pressure`` (0..1) is the verbalised-uncertainty
        signal from :class:`core.uncertainty.UncertaintyTracker`. It
        applies a multiplicative penalty ``× (1 − λ·pressure)`` to the
        composite, fusing the formerly-disjoint "the agent said it was
        unsure" track into the objective score. With the default
        ``λ = 0`` it is a no-op, so this is fully back-compatible until
        the calibration spine fits a positive λ.
        """
        e = _clamp_unit(normalised_entropy)
        p = _clamp_unit(competence_p_success)
        # Shrink the competence component toward the neutral prior (0.5)
        # so it can't dominate before the prior has earned its weight.
        # Coefficient is n/(n+5): 0 obs → fully neutral, 5 obs → halfway,
        # → 1.0 as n grows. Applied for ALL n (no `< 5` cutoff): that cutoff
        # made the coefficient jump discontinuously from 4/9≈0.44 at n=4 to
        # 1.0 at n=5 — right at the calibration threshold the design cares about.
        n = max(0, int(n_observations))
        shrink = n / (n + 5.0)
        p = shrink * p + (1.0 - shrink) * 0.5
        entropy_component = 1.0 - e
        composite = self.w_entropy * entropy_component + self.w_competence * p
        # Fuse the verbalised-uncertainty pressure as a multiplicative
        # penalty (defaults to no-op at λ = 0).
        pressure = _clamp_unit(uncertainty_pressure)
        composite = composite * (1.0 - self.lambda_uncertainty * pressure)
        composite = _clamp_unit(composite)
        return ConfidenceReading(
            composite=composite,
            entropy_component=entropy_component,
            competence_component=p,
            threshold=self.threshold,
            below_threshold=composite < self.threshold,
            uncertainty_pressure=pressure,
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
