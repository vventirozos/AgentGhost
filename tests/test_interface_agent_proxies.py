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
    # ⚠ /api/feedback was MISSING from this table (review R1 M12d). It is the
    # route that writes human outcome labels — §4BT made those the primary
    # metric of a live experiment arm — so an unauthenticated or unproxied
    # regression there is the most consequential of the set.
    ("POST", "/api/feedback"),
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
        r = client.get("/api/sessions/web-abc_1-2",
                       headers={"X-Ghost-Key": server.GHOST_API_KEY})
    assert r.status_code == 200
    args, _ = fake.request.call_args
    assert args[1].endswith("/api/sessions/web-abc_1-2")


def test_proxy_id_charset_matches_the_store():
    """The proxy guard and the STORE's guard must be the same set.

    This test used to assert `web-abc_1.2` was ACCEPTED — an id the store's
    ``_ID_RE`` rejects, which makes ``_path()`` return None and every write
    a silent no-op. The lax end of a two-guard pair is where the data goes
    (review R1 M10), so pin equality by SAMPLING both, not by comparing
    pattern text (equal-looking patterns are not proof)."""
    from ghost_agent.core.sessions import _ID_RE
    samples = ["web-abc_1-2", "0123456789abcdef", "a" * 64, "a" * 65,
               "web-abc_1.2", ".hidden", "a..b", "a/b", "x?y", "", "-lead",
               "UPPER_case-9", "a" * 128, "a\n", "a\nb", "a b"]
    # ⚠ 13 hand-picked samples "pinned equality" while `+`, `:`, `~` and a
    # TRAILING NEWLINE all disagreed between the two guards (R2 lens C). A
    # claim of set equality has to be checked over a SET, so sweep every
    # codepoint that could plausibly appear in an id.
    samples += [f"a{chr(cp)}b" for cp in range(0x00, 0x300)]
    samples += [chr(cp) for cp in range(0x20, 0x300)]
    for sid in samples:
        store_ok = bool(_ID_RE.match(sid))
        try:
            server._safe_session_id(sid)
            proxy_ok = True
        except Exception:
            proxy_ok = False
        assert proxy_ok == store_ok, (
            f"{sid!r}: proxy={proxy_ok} store={store_ok} — a proxy-accepted, "
            "store-rejected id persists NOTHING; the reverse breaks a legal "
            "session")


@pytest.mark.asyncio
async def test_hostile_session_ids_are_rejected():
    """`..` survives quote(safe='') and httpx normalizes dot segments —
    a raw-path client sending /api/sessions/.. would otherwise reach /api
    upstream (the agent's catch-all), escaping the allowlist by one
    segment. Starlette/TestClient normalize dot segments client-side, so
    the validator is exercised directly (defense for --path-as-is style
    clients), plus percent-encoded dots through the full stack."""
    for sid in ("..", ".hidden", "a..b", "x?y", "a/b", "", "a.b", "a" * 65):
        with pytest.raises(Exception) as exc_info:
            server._safe_session_id(sid)
        assert getattr(exc_info.value, "status_code", None) == 400, f"{sid!r}"
    assert server._safe_session_id("web-abc_1-2") == "web-abc_1-2"

    # ⚠ BOTH ROUTES. This test probed only GET through the stack, so
    # deleting `_safe_session_id` from the DELETE proxy survived mutation
    # (review R1 M12a) — and DELETE is the destructive one: an escaped path
    # reaching the agent's catch-all would be a DELETE against whatever it
    # resolved to.
    import inspect
    for _name in ("sessions_get_proxy", "sessions_delete_proxy"):
        _src = inspect.getsource(getattr(server, _name))
        assert "_safe_session_id(session_id)" in _src, (
            f"{_name} pastes the raw path param into the upstream URL")

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


def test_the_proxy_is_an_ALLOWLIST_not_a_catch_all():
    """THE design property of this layer, and it was unpinned (review R1
    M12b): adding an `/api/{path:path}` catch-all survived the whole suite,
    because the existing structural test only counts decorators and accepts
    `>= 11` — a catch-all carries the auth dependency too, so it passes.

    The allowlist exists because the agent's API is strictly larger than what
    a browser should be able to reach. A catch-all silently grants the
    console every agent endpoint, including ones that mutate memory, run
    tools, or arm bench drains.
    """
    routes = []
    for r in server.app.routes:
        path = getattr(r, "path", "")
        if path.startswith("/api"):
            routes.append(path)
    wildcards = [p for p in routes if "{path" in p or p.rstrip("/") == "/api"]
    assert not wildcards, (
        f"a catch-all proxy route exists: {wildcards} — the allowlist is the "
        "boundary between the browser and the agent's full API")
    # And no path parameter may be pasted into an UPSTREAM path without a
    # validator. (`/api/chat/resume/{task_id}` is fine: task_id is only ever
    # a local `active_chat_tasks` dict key, never part of a proxied URL — an
    # unknown id simply misses the dict. The distinction that matters is
    # "reaches the upstream URL", not "is a parameter".)
    import inspect
    import re as _re
    for r in server.app.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/api") or "{" not in path:
            continue
        fn = getattr(r, "endpoint", None)
        if fn is None:
            continue
        src = inspect.getsource(fn)
        for param in _re.findall(r"\{(\w+)\}", path):
            # Does this param appear inside an f-string upstream path?
            pasted = _re.search(
                r'f"/api/[^"]*\{[^"]*' + param + r'[^"]*\}', src)
            if pasted:
                assert "_safe_session_id(" in src or "_safe_" in src, (
                    f"{path} pastes {param} into the upstream URL with no "
                    "validator — this is how the allowlist gets escaped by "
                    "one segment")


def test_a_non_allowlisted_agent_endpoint_is_NOT_reachable():
    """Behavioural mirror of the above: the routes that exist on the agent
    but deliberately not here must 404 at the proxy rather than reach it."""
    client = TestClient(server.app)
    headers = {"X-Ghost-Key": server.GHOST_API_KEY}
    for path in ("/api/introspect", "/api/skills", "/api/tasks",
                 "/api/memory/search", "/api/bench/drain", "/api/chat/task/x"):
        r = client.get(path, headers=headers)
        assert r.status_code in (404, 405), (
            f"{path} is reachable through the interface proxy ({r.status_code})")
