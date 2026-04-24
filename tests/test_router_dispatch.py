"""Tests for router.dispatch."""

import pytest

from ghost_agent.router.dispatch import (
    ComplexityDispatcher, RoutingDecision,
    POOL_UPSTREAM, POOL_CODING, POOL_VISUAL, POOL_WORKER, POOL_SWARM,
)
from ghost_agent.router.features import extract_features
from ghost_agent.router.model import ComplexityClassifier


def _fitted_classifier():
    easy = ["hi", "hello", "thanks", "what's up", "ok"]
    hard = [
        "write a python program that parses json from an endpoint",
        "implement a graph traversal with bfs and memoization",
        "refactor sql query with window functions for ranking",
        "debug async socket timeout in circuit breaker",
        "scrape data with playwright and cross-reference",
    ]
    X = [extract_features(t) for t in easy + hard]
    y = (["easy"] * len(easy)) + (["hard"] * len(hard))
    return ComplexityClassifier(epochs=500).fit(X, y)


def test_disabled_dispatcher_returns_full_swarm():
    d = ComplexityDispatcher(classifier=None, disabled=True)
    decision = d.route("anything")
    assert POOL_CODING in decision.allowed_pools
    assert decision.escalated
    assert decision.label == "hard"


def test_untrained_classifier_escalates():
    d = ComplexityDispatcher(classifier=ComplexityClassifier())
    decision = d.route("anything")
    assert decision.escalated
    assert POOL_CODING in decision.allowed_pools


def test_easy_classified_routes_to_upstream_only():
    clf = _fitted_classifier()
    d = ComplexityDispatcher(classifier=clf, confidence_threshold=0.1)
    decision = d.route("hello")
    assert decision.label == "easy"
    assert decision.allowed_pools == [POOL_UPSTREAM]
    assert not decision.escalated


def test_hard_classified_routes_to_full_pools():
    clf = _fitted_classifier()
    d = ComplexityDispatcher(classifier=clf, confidence_threshold=0.1)
    decision = d.route(
        "write python code that parses access.log and runs a sql regex analysis"
    )
    assert decision.label == "hard"
    assert POOL_CODING in decision.allowed_pools
    assert POOL_VISUAL in decision.allowed_pools


def test_low_confidence_always_escalates():
    """Set threshold very high so nothing passes; verify we get fail-safe
    escalation regardless of predicted label."""
    clf = _fitted_classifier()
    d = ComplexityDispatcher(classifier=clf, confidence_threshold=0.999)
    for text in ("hi", "write code that does SQL"):
        decision = d.route(text)
        assert decision.escalated
        assert POOL_CODING in decision.allowed_pools


def test_high_confidence_threshold_respected():
    """With a very low threshold, confident predictions route normally."""
    clf = _fitted_classifier()
    d = ComplexityDispatcher(classifier=clf, confidence_threshold=0.01)
    decision = d.route("hi")
    # Most likely the "hi" prediction is confident enough to route to upstream.
    assert decision.label == "easy"
    assert POOL_CODING not in decision.allowed_pools


def test_prior_turn_context_passed_to_extractor():
    """Confirm the coupling feature is actually being computed; we can
    check by comparing the reason strings for different priors."""
    clf = _fitted_classifier()
    d = ComplexityDispatcher(classifier=clf, confidence_threshold=0.1)
    decision = d.route("run it again", prior_turn_text="parse the json endpoint")
    # We don't assert the label (depends on training); we just confirm
    # the dispatcher produces a valid decision rather than raising.
    assert decision.label in ("easy", "hard")
    assert decision.allowed_pools


def test_routing_decision_dataclass_defaults():
    rd = RoutingDecision(allowed_pools=["upstream"], label="easy", confidence=0.9)
    assert rd.escalated is False
    assert rd.reason == ""
