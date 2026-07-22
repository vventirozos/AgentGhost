"""/api/health `node_health` — circuit-breaker state surfaced per node URL.

`nodes` in the health payload lists only what is CONFIGURED; the
NodeCircuitBreaker tracks what is actually answering. Before 2026-07-22 the
breaker state was never surfaced, so a tripped/dead node was
indistinguishable from a healthy one on the very endpoint the operator uses
to verify node offload. These tests cover: an OPEN node showing up, the
default-empty/guarded paths (missing breaker, raising get_status), the real
NodeCircuitBreaker end-to-end, and that the pre-existing payload shape is
untouched.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(llm_client=None):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)

    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = MagicMock()
    agent.context.args.api_key = "test-key"
    agent.context.args.model = "test-model"
    agent.context.memory_system = MagicMock()
    agent.context.scheduler = None
    agent.context.llm_client = llm_client if llm_client is not None else SimpleNamespace(
        foreground_requests=0, foreground_tasks=0)

    bio = MagicMock()
    bio.done.return_value = False

    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    app.state.biological_task = bio
    app.state.boot_monotonic = 100.0
    app.state.resolved_config = {}
    return app


def _get_health(app):
    client = TestClient(app)
    r = client.get("/api/health", headers={"X-Ghost-Key": "test-key"})
    assert r.status_code == 200
    return r.json()


def test_health_surfaces_open_breaker_node():
    """A tripped node must be visible: url → {state, failures, open_since}."""
    breaker = SimpleNamespace(get_status=lambda: {
        "http://nova:8081": {"failures": 3, "open_since": 1234.5, "state": "open"},
        "http://kilo:8082": {"failures": 0, "open_since": None, "state": "closed"},
    })
    llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0,
                          circuit_breaker=breaker,
                          worker_clients=[{"url": "http://nova:8081", "model": "gemma"}])
    body = _get_health(_make_app(llm))
    assert body["node_health"]["http://nova:8081"] == {
        "state": "open", "failures": 3, "open_since": 1234.5}
    assert body["node_health"]["http://kilo:8082"]["state"] == "closed"
    # The configured-pool listing is untouched by the new key.
    assert body["nodes"]["worker"] == ["http://nova:8081"]


def test_health_node_health_empty_without_breaker():
    """A mock/partial llm_client (no circuit_breaker attr) must not 500 —
    node_health degrades to {}."""
    body = _get_health(_make_app())  # default SimpleNamespace, no breaker
    assert body["status"] == "ok"
    assert body["node_health"] == {}


def test_health_node_health_guards_raising_get_status():
    """health must never raise: a broken get_status() yields {}."""
    def _boom():
        raise RuntimeError("breaker exploded")
    llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0,
                          circuit_breaker=SimpleNamespace(get_status=_boom))
    body = _get_health(_make_app(llm))
    assert body["status"] == "ok"
    assert body["node_health"] == {}


def test_health_node_health_skips_malformed_entries():
    """Non-dict per-url entries are dropped rather than 500ing."""
    breaker = SimpleNamespace(get_status=lambda: {
        "http://ok:1": {"failures": 1, "open_since": None, "state": "closed"},
        "http://bad:2": "not-a-dict",
    })
    llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0,
                          circuit_breaker=breaker)
    body = _get_health(_make_app(llm))
    assert "http://ok:1" in body["node_health"]
    assert "http://bad:2" not in body["node_health"]


def test_health_with_real_circuit_breaker_open_after_threshold():
    """End-to-end with the real NodeCircuitBreaker: 3 consecutive failures
    trip the breaker and /api/health shows state=open for that URL."""
    from ghost_agent.core.llm import NodeCircuitBreaker
    cb = NodeCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    for _ in range(3):
        cb.record_failure("http://nova:8081")
    cb.record_success("http://kilo:8082")
    llm = SimpleNamespace(foreground_requests=0, foreground_tasks=0,
                          circuit_breaker=cb)
    body = _get_health(_make_app(llm))
    nova = body["node_health"]["http://nova:8081"]
    assert nova["state"] == "open"
    assert nova["failures"] == 3
    assert isinstance(nova["open_since"], float)
    assert body["node_health"]["http://kilo:8082"]["state"] == "closed"


def test_health_existing_shape_unchanged():
    """The new key is a SIBLING — every pre-existing key must still be there."""
    body = _get_health(_make_app())
    for key in ("status", "rss_mb", "rss_limit_mb", "uptime_s", "asyncio_tasks",
                "foreground_requests", "foreground_tasks",
                "biological_watchdog_alive", "memory_system_loaded",
                "scheduler_jobs", "nodes", "config"):
        assert key in body, f"pre-existing health key {key!r} went missing"
    assert "node_health" in body
