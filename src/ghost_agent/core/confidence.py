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

import math
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
    # The composite BEFORE the outcome_penalty multiply — i.e. the model's
    # confidence prediction that does NOT know the turn's outcome. The
    # calibration spine records THIS (calibration measures how well a
    # pre-outcome prediction matches reality; folding the outcome into the
    # stored prediction made Brier/ECE read optimistically on exactly the
    # negative samples the loop exists to catch). The `below_threshold`
    # decision still uses the penalized `composite`. Defaults to composite
    # for positionally-built readings / no-penalty calls.
    pre_penalty_composite: float = -1.0
    # Whether ``entropy_component`` reflects REAL observed token logprobs
    # or is the neutral 0.5 stand-in used when the upstream returned none.
    #
    # This distinction is load-bearing for calibration. llama-server refuses
    # `logprobs` on `tools + stream` payloads, and the turn loop attaches
    # tools on every non-final generation — so the overwhelming majority of
    # turns observe NO logprobs. Recording those as a flat 0.5 made
    # "no signal" indistinguishable from "the model was genuinely
    # 50/50", which is what left 1200/1201 stored samples at exactly 0.5
    # and `w_entropy` permanently unfittable (zero variance → the grid
    # search can only lose by weighting entropy, so it pins w_e = 0).
    # Samples without an observation are now excluded from the entropy
    # weight fit instead of flooding it with fabricated neutrals.
    entropy_observed: bool = False
    # Turn-EFFORT component: 1.0 = a short, non-repetitive turn; → 0 as the
    # turn sprawls or the model hammers the same tool. This is the first
    # input that varies per TURN rather than per domain — competence is a
    # per-domain historical average and so was near-constant within a
    # domain, which is why the composite's leak-free AUC was 0.473 (no
    # discrimination at all). Measured on 296 labelled trajectories:
    # AUC 0.670, separation +0.232 (passed turns averaged 2.4 tool calls,
    # failed turns 11.7).
    effort_component: float = 0.5
    # Whether `effort_component` was actually derived from this turn's tool
    # calls. Legacy records and callers that don't pass turn shape leave it
    # False, and those samples are excluded from the effort-weight fit
    # exactly as unobserved entropy is.
    effort_observed: bool = False

    def __post_init__(self):
        if self.pre_penalty_composite < 0.0:
            object.__setattr__(self, "pre_penalty_composite", self.composite)


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
        # Effort weight starts at 0 so behaviour is unchanged until a fit
        # earns it — same policy as w_entropy. The three weights sum to 1.
        self.w_effort = 0.0
        self.lambda_uncertainty = _clamp_unit(lambda_uncertainty)
        # Platt recalibration of the blended score into a probability, set
        # by `apply_fitted`. The identity pair means "not fitted yet" and
        # the mapping is skipped entirely, so an agent with no calibration
        # history behaves exactly as it did before this stage existed.
        self.platt_a = 1.0
        self.platt_b = 0.0

    @property
    def _has_platt(self) -> bool:
        return not (self.platt_a == 1.0 and self.platt_b == 0.0)

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
            _weff = float(getattr(params, "w_effort", 0.0))
            if math.isfinite(_weff) and 0.0 <= _weff < 1.0:
                # Re-normalise the trio: the fit reports w_entropy/w_effort
                # and leaves competence as the remainder, so honour that
                # rather than letting the pair-normaliser above win.
                self.w_effort = _weff
                _rest = max(0.0, 1.0 - _weff)
                self.w_entropy = we * _rest
                self.w_competence = wc * _rest
            self.threshold = _clamp_unit(float(params.threshold))
            self.lambda_uncertainty = _clamp_unit(
                float(getattr(params, "lambda_uncertainty", 0.0))
            )
            # The fitted threshold lives on the CALIBRATED scale, so these
            # two must move together — adopting one without the other would
            # compare a raw composite against a probability cut.
            _a = float(getattr(params, "platt_a", 1.0))
            _b = float(getattr(params, "platt_b", 0.0))
            if math.isfinite(_a) and math.isfinite(_b):
                self.platt_a, self.platt_b = _a, _b
        except Exception:  # pragma: no cover — defensive
            pass

    def score(self, *, normalised_entropy=None,
              competence_p_success: float,
              n_observations: int = 0,
              uncertainty_pressure: float = 0.0,
              outcome_penalty: float = 0.0,
              entropy_observed=None,
              effort=None) -> ConfidenceReading:
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

        ``outcome_penalty`` (0..1) is a DIRECT, objective negative signal
        about THIS turn's result — set when the verifier REFUTED the
        answer or the turn finalised on an unverified mutation. Unlike the
        two priors above (which are historical/verbalised), this is ground
        truth that the just-produced answer is wrong/unconfirmed, so it
        applies an un-discounted multiplicative penalty ``× (1 − penalty)``.
        It exists because competence + neutral-entropy alone make composite
        ≈ competence, so a domain the agent is historically good at reports
        high confidence even when the verifier just said the answer is
        broken (the "C=0.92 below=no on a REFUTED build" failure). Defaults
        to 0.0 → no-op, fully back-compatible.
        """
        # ``normalised_entropy=None`` means "no logprobs were observed" —
        # score with the neutral 0.5 (unchanged behaviour) but FLAG it, so
        # calibration can exclude it from the entropy-weight fit instead of
        # treating a fabricated neutral as evidence. Callers that pass an
        # explicit float are treated as observations unless they say
        # otherwise via ``entropy_observed``.
        if entropy_observed is None:
            entropy_observed = normalised_entropy is not None
        e = _clamp_unit(0.5 if normalised_entropy is None else normalised_entropy)
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
        # Missing-feature renormalisation. When no logprobs were observed the
        # entropy term carries NO information, so the competence prior takes
        # the full objective weight instead of being diluted by a constant
        # 0.5. Blending in the stand-in would drag every such reading toward
        # 0.5 in exact proportion to w_entropy — which is why w_entropy could
        # only ever be fit to 0 while unobserved turns dominated. With this
        # split the weight can be learned from turns that DO carry entropy
        # without degrading the ones that don't. (w_entropy + w_competence
        # sum to 1 by construction, so `p` alone is the renormalised form.)
        # Generalised over ALL three components: blend the ones actually
        # observed and renormalise by their weights. Competence is always
        # present, so this reduces to the old two-term form when entropy is
        # observed and effort is not, and to plain competence when neither
        # is — byte-identical to the previous behaviour in both cases.
        effort_observed = effort is not None
        eff = _clamp_unit(0.5 if effort is None else effort)
        parts = [(self.w_competence, p)]
        if entropy_observed:
            parts.append((self.w_entropy, entropy_component))
        if effort_observed:
            parts.append((self.w_effort, eff))
        _wsum = sum(w for w, _ in parts)
        composite = (sum(w * v for w, v in parts) / _wsum) if _wsum > 0 else p
        # Fuse the verbalised-uncertainty pressure as a multiplicative
        # penalty (defaults to no-op at λ = 0).
        pressure = _clamp_unit(uncertainty_pressure)
        composite = composite * (1.0 - self.lambda_uncertainty * pressure)
        # Snapshot the prediction BEFORE the outcome penalty — this is what
        # calibration records (a prediction must not know its own outcome).
        pre_penalty = _clamp_unit(composite)
        # Apply the objective outcome penalty LAST and un-discounted: a
        # verifier REFUTED / unverified-mutation verdict is ground truth
        # about this specific answer and must pull the reading below
        # threshold regardless of how strong the historical priors are.
        composite = composite * (1.0 - _clamp_unit(outcome_penalty))
        composite = _clamp_unit(composite)
        # Map the blended score onto a calibrated PROBABILITY as the last
        # step. The weights decide how to combine the signals but cannot set
        # their level: measured live, the raw composite ranked turns well
        # (AUC 0.679) yet scored a worse Brier than always predicting the
        # base rate. Platt is monotonic, so this preserves every ordering
        # and every relative decision — it only fixes the scale. Both
        # outputs are mapped so the recorded prediction and the
        # threshold-compared value live on the same scale as the fitted
        # threshold. Skipped entirely until a fit has run.
        if self._has_platt:
            composite = _apply_platt(composite, self.platt_a, self.platt_b)
            pre_penalty = _apply_platt(pre_penalty, self.platt_a, self.platt_b)
        return ConfidenceReading(
            composite=composite,
            entropy_component=entropy_component,
            competence_component=p,
            threshold=self.threshold,
            below_threshold=composite < self.threshold,
            uncertainty_pressure=pressure,
            pre_penalty_composite=pre_penalty,
            entropy_observed=bool(entropy_observed),
            effort_component=eff,
            effort_observed=bool(effort_observed),
        )


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

# Saturation points for the two turn-effort signals, chosen by measuring
# 296 labelled trajectories: passed turns averaged 2.4 tool calls with a
# longest same-tool run of 1.4; failed turns averaged 11.7 calls and 5.8.
# A turn at or past these values contributes 0 to its half of the term.
_EFFORT_CALLS_SATURATE = 12.0
_EFFORT_REPEAT_SATURATE = 6.0


def effort_component(tool_names) -> float:
    """Turn-shape confidence term in ``[0, 1]`` — 1.0 is a clean, short turn.

    Two struggle signals, averaged:
      * how MANY tool calls the turn needed (sprawl), and
      * the longest run of the SAME tool back to back (spinning).

    Averaged rather than min()'d so one extreme signal cannot slam the term
    to zero on its own; measured AUC is within noise of the best single
    signal (0.670 vs 0.678) while depending on both.

    Deliberately EXCLUDES whether any tool errored. That is the strongest
    raw correlate (AUC 0.229) but it is literally a term of the calibration
    label (`execution_failure_count > 0`), so feeding it back in would
    manufacture a self-fulfilling score that predicts the label from the
    label and adds no information.
    """
    names = [str(n) for n in (tool_names or []) if n]
    n_calls = len(names)
    longest = 0
    run = 0
    prev = None
    for nm in names:
        run = run + 1 if nm == prev else 1
        prev = nm
        longest = max(longest, run)
    sprawl = min(1.0, n_calls / _EFFORT_CALLS_SATURATE)
    spin = min(1.0, longest / _EFFORT_REPEAT_SATURATE)
    return _clamp_unit(1.0 - (sprawl + spin) / 2.0)


def _apply_platt(composite: float, a: float, b: float) -> float:
    """``sigmoid(a·composite + b)``, overflow-safe and clamped.

    Kept local (rather than imported from core.calibration) so the scorer
    has no import dependency on the fitting module — but the arithmetic
    MUST stay identical to `calibration.apply_platt`, or the agent would
    evaluate a different function than the one that was fitted. A test
    pins the two against each other.
    """
    try:
        z = float(a) * float(composite) + float(b)
    except (TypeError, ValueError):
        return _clamp_unit(composite)
    if not math.isfinite(z):
        return _clamp_unit(composite)
    if z >= 0:
        p = 1.0 / (1.0 + math.exp(-z)) if z < 700 else 1.0
    else:
        ez = math.exp(z) if z > -700 else 0.0
        p = ez / (1.0 + ez)
    return _clamp_unit(p)


def _clamp_unit(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
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
