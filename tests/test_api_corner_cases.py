"""Corner-case tests for the FastAPI surface.

These exercise the request-validation and error-shape contracts of
the API layer without booting the full agent. We use FastAPI's
TestClient against a minimally-stubbed app, which is enough to catch:

  * malformed JSON bodies → 4xx with a parseable error
  * missing API key → 401/403
  * wrong content-type
  * streaming endpoint resilience to handler exceptions
  * non-streaming exception → 500 with OpenAI-shaped error JSON
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the entire module if FastAPI is not available — the API tests
# rely on TestClient.
fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_test_app(handle_chat_impl=None):
    """Build a minimal FastAPI app that mounts the real api.routes and
    stubs the agent's handle_chat. Returns (app, agent_mock)."""
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)

    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = MagicMock()
    agent.context.args.api_key = "test-key"
    agent.context.args.model = "test-model"
    agent.context.args.no_memory = True
    agent.context.scratchpad = MagicMock()
    agent.context.scratchpad._data = {}
    agent.context.sandbox_dir = MagicMock()
    agent.context.llm_client = MagicMock()

    if handle_chat_impl is None:
        async def _ok(body, bg, request_id=None):
            return ("ok", 1234, "req-1")
        agent.handle_chat = _ok
    else:
        agent.handle_chat = handle_chat_impl

    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    return app, agent


# ──────────────────────────────────────────────────────────────────────
# /api/version (no auth required)
# ──────────────────────────────────────────────────────────────────────

class TestVersionEndpoint:
    def test_version_returns_200(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.get("/api/version")
            assert r.status_code == 200
            data = r.json()
            assert "version" in data

    def test_root_redirects_or_responds(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.get("/")
            # Either 200 with content or a redirect
            assert r.status_code in (200, 301, 302, 307, 308)


# ──────────────────────────────────────────────────────────────────────
# /api/chat — auth + body validation
# ──────────────────────────────────────────────────────────────────────

class TestChatAuth:
    def test_missing_api_key_rejected(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code in (401, 403), (
                f"missing API key got {r.status_code}, expected 401/403"
            )

    def test_wrong_api_key_rejected(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "WRONG"},
                json={"messages": []},
            )
            assert r.status_code in (401, 403)

    def test_correct_api_key_accepted(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
            # 200 (or streaming if stream=True). Definitely NOT 401.
            assert r.status_code != 401
            assert r.status_code != 403


# ──────────────────────────────────────────────────────────────────────
# /api/chat — handler error shape
# ──────────────────────────────────────────────────────────────────────

class TestChatErrorShape:
    def test_handler_exception_returns_500_with_error_json(self):
        async def boom(body, bg, request_id=None):
            raise RuntimeError("simulated handler crash")

        app, _ = _make_test_app(handle_chat_impl=boom)
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                # Must supply a VALID messages list so the request
                # passes input validation and actually reaches the
                # handler (where `boom` raises). `messages: []` would
                # now be rejected with 422 by the request-validation
                # layer before the handler is invoked.
                json={"messages": [{"role": "user", "content": "hi"}],
                      "stream": False},
            )
            assert r.status_code == 500
            data = r.json()
            assert "error" in data, f"500 missing error field: {data}"
            assert "message" in data["error"]
            assert "type" in data["error"]
            # The error type is now the generic "InternalError" — the
            # Python exception class name MUST NOT leak to the wire
            # (security/info-disclosure hardening). An error_id is
            # included for log correlation.
            assert data["error"]["type"] == "InternalError"
            assert "RuntimeError" not in data["error"]["message"]
            assert "error_id=" in data["error"]["message"]

    def test_handler_exception_in_streaming_writes_error_event(self):
        async def boom(body, bg, request_id=None):
            raise RuntimeError("boom in stream")

        app, _ = _make_test_app(handle_chat_impl=boom)
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                json={"messages": [{"role": "user", "content": "hi"}],
                      "stream": True},
            )
            # SSE stream returns 200 + event-stream content
            assert r.status_code == 200
            text = r.text
            # Error event must be in the stream
            assert "event: error" in text or "CRITICAL SERVER ERROR" in text


# ──────────────────────────────────────────────────────────────────────
# /api/chat — body parsing
# ──────────────────────────────────────────────────────────────────────

class TestChatBodyParsing:
    def test_malformed_json_returns_400_or_500(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={
                    "X-Ghost-Key": "test-key",
                    "Content-Type": "application/json",
                },
                content=b"{not valid json",
            )
            # FastAPI should reject malformed JSON with 400 (or
            # 500 with a parseable error JSON). Either is acceptable;
            # a 200 would be a bug.
            assert r.status_code in (400, 422, 500), (
                f"malformed JSON got {r.status_code}"
            )

    def test_empty_body_handled(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                content=b"",
            )
            # Some non-200 is expected (no body to parse)
            assert r.status_code != 200 or r.json().get("error")

    def test_body_without_messages_handled(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                json={},
            )
            # Either 200 (handler accepts it and returns empty), 4xx
            # (validator rejects), or 5xx (handler raises). NOT a crash
            # without a JSON body.
            assert r.status_code in (200, 400, 422, 500)
            try:
                data = r.json()
            except Exception:
                pytest.fail("Response is not JSON")


# ──────────────────────────────────────────────────────────────────────
# /api/show — basic shape
# ──────────────────────────────────────────────────────────────────────

class TestModelMetadataEndpoints:
    def test_api_show_returns_200(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.post("/api/show", json={"name": "test-model"})
            assert r.status_code == 200
            assert "modelfile" in r.json() or "details" in r.json() or r.json()

    def test_api_tags_returns_200(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.get("/api/tags")
            assert r.status_code == 200
            data = r.json()
            assert "models" in data


# ──────────────────────────────────────────────────────────────────────
# Streaming — graceful handling of long-running handlers
# ──────────────────────────────────────────────────────────────────────

class TestStreamingResilience:
    def test_streaming_with_async_iterable_content(self):
        async def streaming_handler(body, bg, request_id=None):
            async def _gen():
                for i in range(3):
                    yield f"data: chunk {i}\n\n".encode()
            return (_gen(), 1234, "req-1")

        app, _ = _make_test_app(handle_chat_impl=streaming_handler)
        with TestClient(app) as c:
            r = c.post(
                "/api/chat",
                headers={"X-Ghost-Key": "test-key"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert r.status_code == 200
            # Got the chunks
            assert "chunk 0" in r.text
            assert "chunk 2" in r.text


# ──────────────────────────────────────────────────────────────────────
# Concurrent request handling — no shared-state corruption
# ──────────────────────────────────────────────────────────────────────

class TestConcurrentRequests:
    def test_concurrent_chat_requests_dont_cross_contaminate(self):
        """Two requests in flight at once must produce independent
        responses tied to their own request_id."""
        import threading

        async def handler(body, bg, request_id=None):
            # Echo the request's first message content back
            if body.get("messages"):
                content = body["messages"][0].get("content", "")
            else:
                content = ""
            return (f"echo: {content}", 1234, request_id or "no-id")

        app, _ = _make_test_app(handle_chat_impl=handler)
        # TestClient is not safe for concurrent use; we simulate by
        # making sequential calls with different bodies and verifying
        # response identity.
        with TestClient(app) as c:
            for i in range(5):
                r = c.post(
                    "/api/chat",
                    headers={"X-Ghost-Key": "test-key"},
                    json={
                        "messages": [{"role": "user", "content": f"req-{i}"}],
                        "stream": False,
                    },
                )
                assert r.status_code == 200
                data = r.json()
                # The response content reflects THIS request, not a
                # prior one
                assert f"echo: req-{i}" in data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────
# /api/version is reachable WITHOUT auth (per implementation)
# ──────────────────────────────────────────────────────────────────────

class TestUnauthenticatedEndpoints:
    """Some endpoints intentionally don't require auth (version probe,
    etc.). Verify they're reachable for monitoring."""

    def test_version_no_auth(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.get("/api/version")
            assert r.status_code == 200

    def test_tags_no_auth(self):
        app, _ = _make_test_app()
        with TestClient(app) as c:
            r = c.get("/api/tags")
            assert r.status_code == 200
