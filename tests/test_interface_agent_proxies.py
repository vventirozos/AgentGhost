"""Agent-API passthrough proxies on the interface server (2026-07-28).

The workspace UI (sessions rail, notifications, status strip, memory
correction) rides an explicit allowlist of proxies added to
interface/server.py. These tests pin:
  * every proxy route declares the verify_interface_key dependency,
  * the forwarded request carries the upstream X-Ghost-Key + query params,
  * upstream status codes propagate (no implicit-200 masking),
  * path params are URL-quoted before being pasted into the upstream URL.
"""
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

import interface.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


PROXY_ROUTES = [
    ("GET", "/api/health"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions/some-id"),
    ("DELETE", "/api/sessions/some-id"),
    ("GET", "/api/turns"),
    ("POST", "/api/turn/cancel"),
    ("GET", "/api/notifications/pending"),
    ("POST", "/api/notifications/ack"),
    ("POST", "/api/memory/correct"),
    ("POST", "/api/memory/delete"),
]


def _fake_upstream(status_code=200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=payload if payload is not None else {"ok": True})
    resp.text = "raw"
    client = MagicMock()
    client.request = AsyncMock(return_value=resp)
    return client


def test_all_proxy_routes_require_auth_unauthenticated_401():
    client = TestClient(server.app)
    for method, path in PROXY_ROUTES:
        r = client.request(method, path)
        assert r.status_code == 401, f"{method} {path} not auth-gated"


def test_all_proxy_routes_declare_auth_dependency_in_source():
    src = Path("interface/server.py").read_text()
    proxies = src.split("Agent API passthrough proxies", 1)[1]
    decorators = [ln for ln in proxies.splitlines() if ln.startswith("@app.")]
    assert len(decorators) >= 11
    for ln in decorators:
        assert "verify_interface_key" in ln, f"proxy decorator missing auth: {ln}"


@pytest.mark.asyncio
async def test_proxy_forwards_key_query_params_and_status():
    fake = _fake_upstream(status_code=200,
                          payload={"enabled": True, "records": []})
    client = TestClient(server.app)
    with patch.object(server, "_get_http_client", return_value=fake):
        r = client.get(
            "/api/notifications/pending?consumer=web-ui&limit=25",
            headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200
    args, kwargs = fake.request.call_args
    assert args[0] == "GET"
    assert args[1].endswith("/api/notifications/pending")
    assert kwargs["params"] == {"consumer": "web-ui", "limit": "25"}
    assert kwargs["headers"]["X-Ghost-Key"] == server.GHOST_API_KEY


@pytest.mark.asyncio
async def test_proxy_propagates_upstream_error_status():
    fake = _fake_upstream(status_code=503,
                          payload={"detail": "sessions are not enabled"})
    client = TestClient(server.app)
    with patch.object(server, "_get_http_client", return_value=fake):
        r = client.get("/api/sessions",
                       headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 503
    assert "sessions" in r.text


@pytest.mark.asyncio
async def test_proxy_forwards_post_body_verbatim():
    fake = _fake_upstream(payload={"cancelled": True})
    client = TestClient(server.app)
    body = b'{"request_id": "req-1", "hard": true}'
    with patch.object(server, "_get_http_client", return_value=fake):
        r = client.post("/api/turn/cancel", content=body,
                        headers={"Content-Type": "application/json",
                                 "X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200
    _, kwargs = fake.request.call_args
    assert kwargs["content"] == body
    assert kwargs["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_valid_session_id_passes_through_verbatim():
    fake = _fake_upstream(payload={"id": "x"})
    client = TestClient(server.app)
    with patch.object(server, "_get_http_client", return_value=fake):
        r = client.get("/api/sessions/web-abc_1.2",
                       headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200
    args, _ = fake.request.call_args
    assert args[1].endswith("/api/sessions/web-abc_1.2")


@pytest.mark.asyncio
async def test_hostile_session_ids_are_rejected():
    """`..` survives quote(safe='') and httpx normalizes dot segments —
    a raw-path client sending /api/sessions/.. would otherwise reach /api
    upstream (the agent's catch-all), escaping the allowlist by one
    segment. Starlette/TestClient normalize dot segments client-side, so
    the validator is exercised directly (defense for --path-as-is style
    clients), plus percent-encoded dots through the full stack."""
    for sid in ("..", ".hidden", "a..b", "x?y", "a/b", ""):
        with pytest.raises(Exception) as exc_info:
            server._safe_session_id(sid)
        assert getattr(exc_info.value, "status_code", None) == 400, f"{sid!r}"
    assert server._safe_session_id("web-abc_1.2") == "web-abc_1.2"

    fake = _fake_upstream()
    client = TestClient(server.app)
    with patch.object(server, "_get_http_client", return_value=fake):
        r = client.get("/api/sessions/%2e%2e",
                       headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 400
    fake.request.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_upstream_connection_failure_is_502():
    client_mock = MagicMock()
    client_mock.request = AsyncMock(side_effect=RuntimeError("connect refused"))
    client = TestClient(server.app)
    with patch.object(server, "_get_http_client", return_value=client_mock):
        r = client.get("/api/health",
                       headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 502
    assert "error" in r.json()
