"""/api/health endpoint + resolved-config dump + RSS watchdog
(IMPROVEMENTS.md #3 + #21).

Behaviour is set by 5 config sources with nothing printing the resolved result,
and nothing watched the process's own RSS on a box the known 270MB→2GB leak can
OOM. This covers: the health endpoint's shape + auth + silent-failure
detectors, the resolved-config builder's redaction/toggle capture, and the RSS
watchdog's opt-in + idle gating.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(memory_loaded=True, bio_done=False):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)

    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = MagicMock()
    agent.context.args.api_key = "test-key"
    agent.context.args.model = "test-model"
    agent.context.memory_system = MagicMock() if memory_loaded else None
    agent.context.scheduler = None
    agent.context.llm_client = SimpleNamespace(foreground_requests=0, foreground_tasks=0)

    bio = MagicMock()
    bio.done.return_value = bio_done

    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    app.state.biological_task = bio
    app.state.boot_monotonic = 100.0
    app.state.resolved_config = {"arg.verifier": True, "arg.api_key": "***set***"}
    return app, agent


def test_health_requires_api_key():
    app, _ = _make_app()
    client = TestClient(app)
    assert client.get("/api/health").status_code == 403
    r = client.get("/api/health", headers={"X-Ghost-Key": "test-key"})
    assert r.status_code == 200


def test_health_reports_core_internals():
    app, _ = _make_app()
    client = TestClient(app)
    body = client.get("/api/health", headers={"X-Ghost-Key": "test-key"}).json()
    assert body["status"] == "ok"
    assert body["memory_system_loaded"] is True
    assert body["biological_watchdog_alive"] is True
    assert body["foreground_requests"] == 0
    assert "config" in body and body["config"]["arg.api_key"] == "***set***"
    # rss_mb is present (float) or None if psutil unavailable; the key must exist.
    assert "rss_mb" in body and "uptime_s" in body and "asyncio_tasks" in body


def test_health_surfaces_degraded_boot():
    """memory_system=None (a failed VectorMemory init disables ALL biological
    phases) and a dead watchdog must be visible, not silent."""
    app, _ = _make_app(memory_loaded=False, bio_done=True)
    client = TestClient(app)
    body = client.get("/api/health", headers={"X-Ghost-Key": "test-key"}).json()
    assert body["memory_system_loaded"] is False
    assert body["biological_watchdog_alive"] is False


def test_health_registered_above_catch_all():
    """The catch-all would proxy /api/health upstream if it were registered
    first — assert the explicit route wins (200, not a proxy attempt)."""
    app, _ = _make_app()
    client = TestClient(app)
    r = client.get("/api/health", headers={"X-Ghost-Key": "test-key"})
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


# ---------------------------------------------------------------- config dump


def test_resolved_config_redacts_key_and_captures_toggles():
    from ghost_agent.main import _build_resolved_config
    args = SimpleNamespace(api_key="secret-value", verifier=True, model="qwen")
    context = SimpleNamespace(memory_system=object(), scheduler=None)
    with patch.dict("os.environ", {"GHOST_CRITIC_ASYNC": "1", "GHOST_MAX_RSS_MB": "1500"}):
        cfg = _build_resolved_config(args, context)
    assert cfg["arg.api_key"] == "***set***"
    assert "secret-value" not in str(cfg)
    assert cfg["arg.verifier"] is True
    assert cfg["env.GHOST_CRITIC_ASYNC"] == "1"
    assert cfg["runtime.memory_system_loaded"] is True
    # The module-constant cognitive toggles must be captured (their only
    # visible surface — no flag controls them).
    assert any(k.startswith("toggle._") for k in cfg)


# ---------------------------------------------------------------- RSS watchdog


def _agent_with_rss(rss_mb, fg_requests=0, fg_tasks=0):
    from ghost_agent.core.agent import GhostAgent
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = SimpleNamespace(
        llm_client=SimpleNamespace(foreground_requests=fg_requests, foreground_tasks=fg_tasks),
        sandbox_manager=None,
    )
    ag._current_rss_mb = lambda: rss_mb
    return ag


def test_rss_watchdog_off_by_default():
    ag = _agent_with_rss(9000)
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("GHOST_MAX_RSS_MB", None)
        with patch("os.execv") as execv:
            ag._rss_watchdog_check()
            execv.assert_not_called()


def test_rss_watchdog_restarts_when_over_and_idle():
    ag = _agent_with_rss(2000)
    with patch.dict("os.environ", {"GHOST_MAX_RSS_MB": "1500"}):
        with patch("os.execv") as execv, patch("sys.stdout.flush"), patch("sys.stderr.flush"):
            ag._rss_watchdog_check()
            execv.assert_called_once()


def test_rss_watchdog_defers_when_foreground_active():
    ag = _agent_with_rss(2000, fg_requests=1)
    with patch.dict("os.environ", {"GHOST_MAX_RSS_MB": "1500"}):
        with patch("os.execv") as execv:
            ag._rss_watchdog_check()
            execv.assert_not_called()


def test_rss_watchdog_noop_under_limit():
    ag = _agent_with_rss(500)
    with patch.dict("os.environ", {"GHOST_MAX_RSS_MB": "1500"}):
        with patch("os.execv") as execv:
            ag._rss_watchdog_check()
            execv.assert_not_called()
