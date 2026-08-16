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

from .embedding import embed_text
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

    `allowed_pools` is the whitelist a caller COULD respect — but note: no
    production code currently gates swarm pools on it (the swarm dispatch
    never reads `_router_decision.allowed_pools`). Today the router's only
    consumed effect is `label`/`escalated`, read at ONE site (`agent.py`,
    the MCTS lookahead gate: `label == "hard" and not escalated`) — and
    that site is itself behind `_MCTS_TURNSTART_ENABLED = False`, so
    NOTHING acts on a routing decision today; it is recorded only. The
    confidence-thresholded strategic-planner skip that this docstring used
    to name as a second consumer was RETIRED in §4AN — do not re-add one
    without first showing the model is informative in the region that
    clears the bar. Pool-whitelisting is not wired at all. `label` and `confidence` are also recorded on the trajectory
    for A/B-testing the router's effect.
    """

    allowed_pools: List[str]
    label: str
    confidence: float
    escalated: bool = False
    reason: str = ""
    #: WHY this escalated, as a stable machine-readable token rather than
    #: only in the human `reason` string: "" (did not escalate),
    #: "low_confidence", "no_model", "no_embedding", "scoring_error".
    #: Without it a failed embed is recorded as label="hard",
    #: confidence=0.0 — indistinguishable in `turn_facts` from a genuine
    #: low-confidence prediction, so an embedder outage would land in the
    #: 0.0-0.3 bucket of `scripts/router_confidence_backtest.py` as if it
    #: were something the model believed. This flip is justified by a
    #: measurement about confidence; that instrument must not pool the
    #: two.
    escalation_kind: str = ""


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
        embed_fn=None,
    ):
        """`disabled=True` returns HARD_POOLS for every request; used
        as a feature-flag kill switch while validating the router.

        `embed_fn` overrides the process-wide router embedder (tests).
        Production leaves it None and boot registers the vector store's
        embedder once — see `router/embedding.py`.
        """
        self.classifier = classifier
        self.confidence_threshold = float(confidence_threshold)
        self.disabled = bool(disabled)
        self._embed_fn = embed_fn

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
                escalation_kind="no_model",
            )
        # §4BQ flip (vi). An embedding model MUST be handed an embedding;
        # scoring it with a lexical-only vector is the one failure this
        # is built to prevent. If the embedder is missing or returns a
        # wrong-width/non-finite vector, escalate — the fail-safe
        # direction (a cost regression, never a capability one).
        embedding = None
        if getattr(self.classifier, "uses_embeddings_", False):
            embedding = embed_text(user_request, embed_fn=self._embed_fn)
            if embedding is None:
                return RoutingDecision(
                    allowed_pools=list(self.HARD_POOLS),
                    label="hard",
                    confidence=0.0,
                    escalated=True,
                    reason="router needs an embedding but none is available",
                    escalation_kind="no_embedding",
                )
        try:
            fv = extract_features(user_request,
                                  prior_turn_text=prior_turn_text,
                                  embedding=embedding)
            label, confidence = self.classifier.predict(fv)
        except Exception as e:  # noqa: BLE001
            # Width mismatch (FeatureSchemaMismatch) or any other scoring
            # failure. The dispatcher's contract is that it NEVER blocks a
            # request, so degrade to escalate-all rather than propagate.
            return RoutingDecision(
                allowed_pools=list(self.HARD_POOLS),
                label="hard",
                confidence=0.0,
                escalated=True,
                reason=f"router scoring failed ({type(e).__name__}: {e})",
                escalation_kind="scoring_error",
            )

        if confidence < self.confidence_threshold:
            return RoutingDecision(
                allowed_pools=list(self.HARD_POOLS),
                label=label,
                confidence=confidence,
                escalated=True,
                reason=f"confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}",
                escalation_kind="low_confidence",
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
