"""Ghost Agent complexity router.

Tiny classifier that sits *in front of* the swarm dispatcher. Predicts
whether an incoming request is "easy" (routine chat, lookup, quick
answer — don't wake the coding/visual pools) or "hard" (planning-
dependent, tool-heavy, code generation — engage the full pipeline).

Design non-negotiables:

  * Local-only. Features and model both run on CPU with no outbound
    traffic. No embedding service, no hosted classifier.
  * Pure-numpy logistic regression under the hood. sentence-transformer
    embeddings can be *added* to the feature vector when the lib is
    available, but the module works without them.
  * Fail-safe: when confidence is below a threshold, the dispatcher
    escalates to the full swarm. The router can only make requests
    CHEAPER, never LESS CAPABLE.
  * Training data comes from trajectory logs (distill/). Labels are
    derived from outcome signals the trajectory already carries
    (step count, tool-call count, tier at resolution, whether a coding
    node was needed).

Public API:
    extract_features        : request-string → dict of float features
    derive_label            : trajectory → "easy" | "hard" | None
    ComplexityClassifier    : numpy LR + train/predict
    ComplexityDispatcher    : production-facing wrapper
"""

from .features import extract_features, FeatureVector, FEATURE_NAMES
from .labels import derive_label, label_trajectories, LabelSpec
from .model import ComplexityClassifier, TrainingReport
from .dispatch import ComplexityDispatcher, RoutingDecision
from .trainer import RouterTrainer, RouterTrainerReport

__all__ = [
    "extract_features",
    "FeatureVector",
    "FEATURE_NAMES",
    "derive_label",
    "label_trajectories",
    "LabelSpec",
    "ComplexityClassifier",
    "TrainingReport",
    "ComplexityDispatcher",
    "RoutingDecision",
    "RouterTrainer",
    "RouterTrainerReport",
]
