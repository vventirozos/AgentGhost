"""§4VD-B — the web chat forwards the probe mark.

World where each test fails: the interface drops `X-Ghost-Origin: probe` on
/api/chat (a test of web chat runs as the owner's turn: it teaches, moves
the mood, counts in the census), drops the experiment headers that ride a
probe, or forwards an arm/rule header on a request that is NOT a probe.
"""
import json
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

PROBE_SET = {"X-Ghost-Origin": "probe", "X-Ghost-Arm": "look_owned",
             "X-Ghost-Probe-Rule": "rule text", "X-Ghost-Probe-Self-Image": "s",
             "X-Ghost-Probe-Self-Image-Rule": "r"}


def _plain_post(headers):
    """Drive the NON-streamed path; return the headers sent upstream."""
    resp = MagicMock(status_code=200, text="{}")
    resp.json = MagicMock(return_value={"choices": []})
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    with patch.object(server, "_get_http_client", return_value=client):
        r = TestClient(server.app).post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Ghost-Key": server.GHOST_API_KEY, **headers})
    assert r.status_code == 200
    return client.post.call_args.kwargs["headers"]


def _streamed_post(headers):
    """Drive the STREAMED path; return the headers sent upstream."""
    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        async def aiter_bytes(self, chunk_size=None):
            yield b"data: [DONE]\n\n"

    class _Ctx:
        def __init__(self, hdrs):
            seen.update(hdrs)

        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    client = MagicMock()
    client.stream = lambda method, url, json=None, headers=None, timeout=None: _Ctx(headers)
    with patch.object(server, "_get_http_client", return_value=client), \
            patch.object(server, "_schedule_reply_push", lambda *a, **k: None):
        with TestClient(server.app) as tc:
            r = tc.post("/api/chat",
                        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                        headers={"X-Ghost-Key": server.GHOST_API_KEY, **headers})
            assert r.status_code == 200
            assert b"[DONE]" in r.content
    return seen


@pytest.mark.parametrize("drive", [_plain_post, _streamed_post])
def test_a_probe_through_web_chat_reaches_the_agent_as_a_probe(drive):
    sent = drive(PROBE_SET)
    for k, v in PROBE_SET.items():
        assert (sent.get(k).decode("latin-1") if isinstance(sent.get(k), bytes) else sent.get(k)) == v, k
    assert sent["X-Ghost-Key"] == server.GHOST_API_KEY


@pytest.mark.parametrize("drive", [_plain_post, _streamed_post])
def test_an_ordinary_page_turn_carries_no_probe_headers(drive):
    sent = drive({})
    assert not [k for k in sent if k.lower().startswith(("x-ghost-origin", "x-ghost-arm", "x-ghost-probe"))]


@pytest.mark.parametrize("drive", [_plain_post, _streamed_post])
def test_experiment_headers_without_the_mark_are_not_forwarded(drive):
    sent = drive({k: v for k, v in PROBE_SET.items() if k != "X-Ghost-Origin"})
    assert not [k for k in sent if k.lower().startswith(("x-ghost-origin", "x-ghost-arm", "x-ghost-probe"))]


@pytest.mark.parametrize("origin,probe", [("probe", True), ("PROBE ", True), ("owner", False),
                                          ("probe-ish", False), ("", False)])
def test_only_the_exact_mark_counts(origin, probe):
    assert bool(server._probe_headers({"X-Ghost-Origin": origin})) is probe


def test_the_agent_reads_exactly_the_forwarded_set():
    """The forwarded list is the agent's list: a probe header the agent reads
    but the proxy does not carry would vanish again silently."""
    import ast
    import re
    tree = ast.parse((_ROOT / "src" / "ghost_agent" / "api" / "routes.py").read_text(encoding="utf-8"))
    read = {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
            and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "headers"
            and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
            and re.fullmatch(r"X-Ghost-(?:Arm|Probe-[A-Za-z-]+)", n.args[0].value)}
    assert read and read == set(server._PROBE_FOLLOWERS)


def test_a_probe_turn_is_minted_with_the_agents_probe_id():
    """Review r1 m7: the agent renames a probe turn `probe-<id>`; a plain id minted here made Stop
    name a turn the agent does not have."""
    from ghost_agent.utils.logging import PROBE_REQUEST_PREFIX, is_probe_request_id
    assert server.PROBE_REQUEST_PREFIX == PROBE_REQUEST_PREFIX
    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        async def aiter_bytes(self, chunk_size=None):
            yield b"data: [DONE]\n\n"

    class _Ctx:
        def __init__(self, hdrs):
            seen.update(hdrs)

        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    client = MagicMock()
    client.stream = lambda method, url, json=None, headers=None, timeout=None: _Ctx(headers)
    for probe in (True, False):
        seen.clear()
        with patch.object(server, "_get_http_client", return_value=client), \
                patch.object(server, "_schedule_reply_push", lambda *a, **k: None):
            with TestClient(server.app) as tc:
                r = tc.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                            headers={"X-Ghost-Key": server.GHOST_API_KEY,
                                     **({"X-Ghost-Origin": "probe"} if probe else {})})
        assert is_probe_request_id(r.headers["X-Request-ID"]) is probe
        assert seen["X-Request-ID"] == r.headers["X-Request-ID"]


def test_a_non_ascii_probe_rule_reaches_httpx_intact():
    """Review r1 m8: httpx encodes a str header as ASCII — a Greek rule raised UnicodeEncodeError."""
    import httpx
    rule = "Απάντα σύντομα".encode("utf-8").decode("latin-1")     # what starlette hands over
    h = server._probe_headers({"X-Ghost-Origin": "probe", "X-Ghost-Probe-Rule": rule})
    req = httpx.Request("POST", "http://x/api/chat", headers=h)
    raw = {k.lower(): v for k, v in req.headers.raw}
    assert raw[b"x-ghost-probe-rule"] == "Απάντα σύντομα".encode("utf-8")
