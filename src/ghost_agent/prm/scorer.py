"""Production-facing PRM scorer.

A thin wrapper around ``StepValueModel`` that callers use without
caring whether a checkpoint is loaded. ``PRMScorer.score`` always
returns a finite float in [0, 1]; when the underlying model is
missing or untrained, the scorer returns the ``default_score``
neutral value so callers can keep the same code path on both branches.

The scorer is the boundary between the PRM module and the rest of the
agent (mcts.py, agent.py, future re-rankers). Callers should not
reach into ``StepValueModel`` directly; that lets us swap the model
implementation later (an MLP, a small transformer, an EBM) without
the call sites moving.

Thread-safety: ``score`` reads ``weights_`` / ``bias_`` once per call.
The retrain path (biological phase 2.7) replaces the entire underlying
model object via ``set_model`` rather than mutating the array in place,
so a concurrent ``score`` either sees the old model end-to-end or the
new one — never a torn read.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .features import (
    PRM_FEATURE_NAMES,
    ActionFeatures,
    PlanState,
    extract_step_features,
)
from .model import StepValueModel


logger = logging.getLogger("GhostAgent")


class PRMScorer:
    """Loaded-once, thread-safe scoring wrapper.

    Construction modes:
        PRMScorer()                       - empty, returns default score
        PRMScorer(model=trained_model)    - given an in-memory model
        PRMScorer.load(path)              - load from JSON checkpoint

    All ``score`` calls are safe to invoke concurrently — see module
    docstring for the reload-replace pattern.
    """

    DEFAULT_NEUTRAL_SCORE: float = 0.5

    def __init__(
        self,
        *,
        model: Optional[StepValueModel] = None,
        default_score: float = DEFAULT_NEUTRAL_SCORE,
    ):
        self._model = model
        self._default_score = float(_clamp_unit(default_score))

    # -----------------------------------------------------------------
    # Loading / hot-reload
    # -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        default_score: float = DEFAULT_NEUTRAL_SCORE,
    ) -> "PRMScorer":
        """Load a checkpoint into a new scorer. Raises on schema /
        feature drift — callers should catch and downgrade to a
        no-op scorer rather than crash the agent.
        """
        model = StepValueModel.load(Path(path))
        return cls(model=model, default_score=default_score)

    def set_model(self, model: Optional[StepValueModel]) -> None:
        """Hot-swap the underlying model. Used by the biological
        retrain phase to publish a freshly-fit checkpoint without
        restarting the agent."""
        self._model = model

    @property
    def has_model(self) -> bool:
        return self._model is not None and self._model.weights_ is not None

    @property
    def model(self) -> Optional[StepValueModel]:
        return self._model

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def score(self, state: PlanState, action: ActionFeatures) -> float:
        """Return p(success | state, action) ∈ [0, 1].

        When the scorer has no trained model, returns ``default_score``
        — typically 0.5 (neutral). Callers can use this branch as a
        sentinel: if every candidate scores identically the neutral
        value, MCTS effectively falls through to its existing
        simulation path.

        Never raises: PRM scoring is advisory and must not break a
        plan-selection turn.
        """
        if not self.has_model:
            return self._default_score
        try:
            value = self._model.predict_value(state, action)
        except Exception as exc:
            logger.debug("PRM score failed; returning neutral: %s", exc)
            return self._default_score
        return _clamp_unit(value)

    def score_many(
        self,
        state: PlanState,
        actions: Sequence[ActionFeatures],
    ) -> List[float]:
        """Score a batch of actions against the same state.

        Convenience for MCTS, which scores N candidates per expansion.
        Implemented as a loop because the model is so small that
        vectorising is not worth the array-construction overhead, and
        keeping it as a loop preserves the per-action try/except
        isolation."""
        return [self.score(state, action) for action in actions]

    def uncertainty(self, state: PlanState, action: ActionFeatures) -> float:
        """Return model uncertainty for (state, action) ∈ [0, 1].

        Defined as ``1 - 2·|p − 0.5|``: scores at the decision boundary
        (p=0.5) map to 1.0 (maximally uncertain), scores at the rails
        (0 or 1) map to 0.0 (maximally confident). When no model is
        loaded the scorer returns its neutral 0.5, which by this metric
        IS maximum uncertainty — semantically correct: "we have no
        opinion" and "we are most unsure" are the same posture for an
        un-trained logistic regression.

        Used by frontier-aware self-play to target clusters the PRM is
        unsure about. Never raises; on internal error returns 1.0 to
        bias toward exploration rather than silently dropping the
        cluster.
        """
        try:
            p = self.score(state, action)
        except Exception:
            return 1.0
        # _clamp_unit already pinned p into [0, 1] and neutralised NaN.
        return _clamp_unit(1.0 - 2.0 * abs(p - 0.5))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _clamp_unit(x) -> float:
    """Clamp a float into [0, 1]. NaN / inf become the neutral 0.5 —
    a finite, neutral value is the safe choice for a fail-safe scorer.
    """
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
