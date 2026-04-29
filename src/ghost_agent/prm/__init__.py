"""Ghost Agent Process Reward Model (PRM).

Per-step value estimator that scores `(state, candidate_action)` tuples
without executing the action. Used by `core.mcts.MCTSReasoner` as a fast
path: when a PRM scorer is attached, plan candidates are scored in
microseconds against the learned model instead of paying a worker-LLM
simulation call per candidate.

Design non-negotiables (mirroring router/):

  * Local-only. Pure-numpy logistic regression. No outbound traffic at
    feature time, no embedding service, no hosted scorer.
  * JSON-persisted (not pickle). The schema is human-diffable and safe
    to load — no code-execution risk on `PRMScorer.load`.
  * Fail-safe. The scorer is always advisory: when it's missing or
    untrained, callers fall back to the existing simulation path. The
    PRM can only make plan selection cheaper, never less capable.
  * Training data comes from existing trajectory logs (distill/). Step
    labels are derived from the trajectory's terminal outcome via
    Monte Carlo value backprop, exactly the way AlphaZero distributes
    credit through a winning rollout.

Public API:
    extract_step_features   : (state, action) → FeatureVector
    derive_step_labels      : Trajectory → List[float] (continuous values
                              ∈ [0, 1] per step, MC-discounted from
                              terminal outcome)
    label_step_value_binary : (value, threshold) → 0 | 1 (binary view
                              for logistic regression)
    StepValueModel          : numpy LR + train/predict_value/save/load
    PRMScorer               : production-facing scorer (load-once,
                              score(state, action) → float)
    PRMTrainer              : pipeline (read trajectories → derive
                              step samples → fit → save).
"""

from .features import (
    extract_step_features,
    PlanState,
    ActionFeatures,
    FeatureVector,
    PRM_FEATURE_NAMES,
)
from .labels import (
    derive_step_labels,
    label_step_value_binary,
    iter_step_samples,
    StepSample,
    StepLabelSpec,
)
from .model import StepValueModel, PRMTrainingReport
from .scorer import PRMScorer
from .trainer import PRMTrainer, TrainerReport

__all__ = [
    "extract_step_features",
    "PlanState",
    "ActionFeatures",
    "FeatureVector",
    "PRM_FEATURE_NAMES",
    "derive_step_labels",
    "label_step_value_binary",
    "iter_step_samples",
    "StepSample",
    "StepLabelSpec",
    "StepValueModel",
    "PRMTrainingReport",
    "PRMScorer",
    "PRMTrainer",
    "TrainerReport",
]
