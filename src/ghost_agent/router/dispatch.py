"""Complexity-aware dispatcher.

A thin adapter that turns a ComplexityClassifier prediction into a
routing decision the existing swarm dispatcher can consume. Its single
responsibility: emit a RoutingDecision that callers in `core/llm.py`
or `tools/swarm.py` can use to decide whether a request can skip the
heavy pools.

Fail-safe rule enforced here:
    low-confidence prediction   → escalate to full swarm (hard path)
    high-confidence "easy"      → pick the cheap path
    high-confidence "hard"      → pick the full path

The dispatcher never blocks a request. The worst thing it can do is
run a simple request through the full pipeline when it could have been
cheap — i.e. a cost regression, not a capability regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .features import extract_features
from .model import ComplexityClassifier


# Canonical pool names, mirroring Ghost's existing swarm flags.
# Kept loose (strings) so `main.py` can plug them in without a
# dependency on this module.
POOL_UPSTREAM = "upstream"
POOL_WORKER = "worker"
POOL_CODING = "coding"
POOL_VISUAL = "visual"
POOL_SWARM = "swarm"


@dataclass
class RoutingDecision:
    """Returned by ComplexityDispatcher.route.

    `allowed_pools` is the whitelist the caller should respect.
    `label` and `confidence` are observability: the dispatcher records
    them on the trajectory so we can A/B test the router's effect.
    """

    allowed_pools: List[str]
    label: str
    confidence: float
    escalated: bool = False
    reason: str = ""


class ComplexityDispatcher:
    """Wraps a ComplexityClassifier with fail-safe routing logic.

    Usage:
        dispatcher = ComplexityDispatcher(clf, confidence_threshold=0.3)
        decision = dispatcher.route("what's 2+2?")
        if POOL_CODING not in decision.allowed_pools:
            # skip coding swarm
    """

    # Pool sets that each label unlocks. An "easy" request is served
    # by upstream alone; a "hard" request can use any pool; an
    # "escalated" request defaults to full swarm (same as "hard")
    # because escalation = "we aren't sure, go safe".
    EASY_POOLS = (POOL_UPSTREAM,)
    HARD_POOLS = (POOL_UPSTREAM, POOL_WORKER, POOL_SWARM, POOL_CODING, POOL_VISUAL)

    def __init__(
        self,
        classifier: Optional[ComplexityClassifier] = None,
        *,
        confidence_threshold: float = 0.3,
        disabled: bool = False,
    ):
        """`disabled=True` returns HARD_POOLS for every request; used
        as a feature-flag kill switch while validating the router."""
        self.classifier = classifier
        self.confidence_threshold = float(confidence_threshold)
        self.disabled = bool(disabled)

    def route(
        self,
        user_request: str,
        *,
        prior_turn_text: str = "",
    ) -> RoutingDecision:
        if self.disabled or self.classifier is None or self.classifier.weights_ is None:
            return RoutingDecision(
                allowed_pools=list(self.HARD_POOLS),
                label="hard",
                confidence=0.0,
                escalated=True,
                reason="router disabled or untrained",
            )
        fv = extract_features(user_request, prior_turn_text=prior_turn_text)
        label, confidence = self.classifier.predict(fv)

        if confidence < self.confidence_threshold:
            return RoutingDecision(
                allowed_pools=list(self.HARD_POOLS),
                label=label,
                confidence=confidence,
                escalated=True,
                reason=f"confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}",
            )

        if label == "easy":
            return RoutingDecision(
                allowed_pools=list(self.EASY_POOLS),
                label="easy",
                confidence=confidence,
                escalated=False,
                reason=f"high-confidence easy (conf={confidence:.2f})",
            )
        return RoutingDecision(
            allowed_pools=list(self.HARD_POOLS),
            label="hard",
            confidence=confidence,
            escalated=False,
            reason=f"high-confidence hard (conf={confidence:.2f})",
        )
