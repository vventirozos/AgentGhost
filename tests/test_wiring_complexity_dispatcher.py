"""Tests for the GhostContext complexity_dispatcher attribute and its
default fail-safe behaviour.

The full chat-path router consultation is verified indirectly by
reading `body['_router_decision']` — but the router itself is tested
in test_router_*.py. This file exists to pin the WIRING: the context
has the attribute, the dispatcher's disabled/no-model fallback
escalates, and the defaults are sane for deployments that never
train a classifier.
"""

from unittest.mock import MagicMock

from ghost_agent.core.agent import GhostContext
from ghost_agent.router.dispatch import (
    ComplexityDispatcher, RoutingDecision,
    POOL_UPSTREAM, POOL_CODING, POOL_VISUAL,
)
from ghost_agent.router.model import ComplexityClassifier


def test_context_defaults_trajectory_collector_none():
    ctx = GhostContext(args=MagicMock(), sandbox_dir=".", memory_dir=".", tor_proxy=None)
    assert ctx.trajectory_collector is None
    assert ctx.reflector is None
    assert ctx.complexity_dispatcher is None


def test_disabled_dispatcher_always_escalates():
    d = ComplexityDispatcher(classifier=None, disabled=True)
    decision = d.route("hi")
    assert decision.escalated is True
    assert decision.label == "hard"
    assert POOL_CODING in decision.allowed_pools
    assert POOL_VISUAL in decision.allowed_pools


def test_missing_model_escalates_fail_safe():
    """A dispatcher built without a trained classifier must never
    downgrade a request — this is the deployment default (main.py
    builds this shape when --router-model isn't passed)."""
    d = ComplexityDispatcher(classifier=None)
    decision = d.route("what is 2+2?")
    assert decision.escalated is True
    assert POOL_CODING in decision.allowed_pools


def test_untrained_classifier_still_escalates():
    """Classifier present but never fit → same fail-safe outcome."""
    clf = ComplexityClassifier()
    d = ComplexityDispatcher(classifier=clf)
    decision = d.route("anything")
    assert decision.escalated is True


def test_routing_decision_contains_observability_fields():
    d = ComplexityDispatcher(classifier=None, disabled=True)
    decision = d.route("hello")
    # These fields let us log/A-B the router's effect.
    assert hasattr(decision, "label")
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "reason")
    assert hasattr(decision, "allowed_pools")


def test_dispatcher_attached_to_context_is_callable():
    """GhostContext is happy to hold the dispatcher; the agent's
    wiring reads it via getattr. A context with no dispatcher
    attribute is still valid."""
    ctx = GhostContext(args=MagicMock(), sandbox_dir=".", memory_dir=".", tor_proxy=None)
    ctx.complexity_dispatcher = ComplexityDispatcher(disabled=True)
    # getattr-style access mirrors the production read path in
    # `GhostAgent.handle_chat`.
    dispatcher = getattr(ctx, "complexity_dispatcher", None)
    assert dispatcher is not None
    decision = dispatcher.route("hi")
    assert POOL_UPSTREAM in decision.allowed_pools
