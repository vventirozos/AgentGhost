"""Web push + in-flight-turn resume surface on the interface server
(2026-08-01 — the Claude-app mobile model).

Pins: the new routes are auth-gated; the manifest is key-gated and its
start_url carries the key (an installed PWA must open authenticated);
task-state/ack semantics; webpush_notify store round-trip + dead-sub
pruning; and the reply-ready push contract — pushes fire ONLY when live
page JS never acked delivery within the grace window.
"""
import asyncio
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
from interface import webpush_notify  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

KEY = server.GHOST_API_KEY
AUTH = {"X-Ghost-Key": KEY}


@pytest.fixture
def client():
    return TestClient(server.app)


def _real_vapid_json():
    """A REAL keypair in the exact on-disk format. The first shipped
    version passed PEM text straight to pywebpush, which cannot parse it
    (Vapid.from_string b64-decodes) — push was 100% dead while a
    MagicMock'd pywebpush kept every test green. Real key material keeps
    the module honest end-to-end."""
    import base64
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    priv = ec.generate_private_key(ec.SECP256R1())
    pem = priv.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    pub = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return {"private_key_pem": pem,
            "public_key_b64url": base64.urlsafe_b64encode(pub).rstrip(b"=").decode(),
            # py_vapid strictly validates the sub email (TLD required).
            "sub": "mailto:test@example.com"}


@pytest.fixture
def push_files(tmp_path, monkeypatch):
    """Isolate the VAPID + subscription stores and reset BOTH caches."""
    vapid = tmp_path / "vapid.json"
    vapid.write_text(json.dumps(_real_vapid_json()))
    subs = tmp_path / "subs.json"
    monkeypatch.setattr(webpush_notify, "_VAPID_FILE", vapid)
    monkeypatch.setattr(webpush_notify, "_SUBS_FILE", subs)
    monkeypatch.setattr(webpush_notify, "_vapid_cache", None)
    monkeypatch.setattr(webpush_notify, "_vapid_signer", None)
    yield vapid, subs
    webpush_notify._vapid_cache = None
    webpush_notify._vapid_signer = None


def _sub(endpoint="https://push.example/dev1"):
    return {"endpoint": endpoint,
            "keys": {"p256dh": "AAA", "auth": "BBB"}}


def _real_sub(endpoint="https://push.example/real"):
    """Subscription with cryptographically valid p256dh/auth so real
    pywebpush encryption succeeds."""
    import base64
    import os as _os
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    browser_key = ec.generate_private_key(ec.SECP256R1())
    p256dh = browser_key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    b64u = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    return {"endpoint": endpoint,
            "keys": {"p256dh": b64u(p256dh), "auth": b64u(_os.urandom(16))}}


class TestRouteAuth:
    ROUTES = [
        ("GET", "/api/push/vapid"),
        ("POST", "/api/push/subscribe"),
        ("POST", "/api/push/unsubscribe"),
        ("POST", "/api/chat/ack/some-task"),
        ("GET", "/api/chat/task/some-task/state"),
    ]

    def test_unauthenticated_401(self, client):
        for method, path in self.ROUTES:
            r = client.request(method, path)
            assert r.status_code == 401, f"{method} {path} not auth-gated"

    def test_manifest_requires_key(self, client):
        assert client.get("/manifest.webmanifest").status_code == 401
        assert client.get("/manifest.webmanifest?key=wrong").status_code == 401

    def test_manifest_start_url_carries_key(self, client):
        r = client.get(f"/manifest.webmanifest?key={KEY}")
        assert r.status_code == 200
        data = r.json()
        assert data["display"] == "standalone"
        assert KEY in data["start_url"]
        assert data["icons"]

    def test_index_injects_manifest_link(self, client):
        r = client.get(f"/?key={KEY}")
        assert r.status_code == 200
        assert 'rel="manifest"' in r.text
        assert "/manifest.webmanifest?key=" in r.text


class TestTaskStateAndAck:
    def test_unknown_task_state(self, client):
        r = client.get("/api/chat/task/nope/state", headers=AUTH)
        assert r.status_code == 200
        assert r.json() == {"exists": False}

    def test_live_task_state_and_ack(self, client):
        server.active_chat_tasks["t-live"] = {
            "buffer": [b"a", b"b"], "buffer_size": 2, "done": False,
            "error": None, "truncated": False,
        }
        try:
            r = client.get("/api/chat/task/t-live/state", headers=AUTH)
            assert r.json() == {"exists": True, "done": False, "error": None,
                                "truncated": False, "chunks": 2}
            r = client.post("/api/chat/ack/t-live", headers=AUTH)
            assert r.json() == {"ok": True}
            assert server.active_chat_tasks["t-live"]["client_acked"] is True
        finally:
            server.active_chat_tasks.pop("t-live", None)

    def test_ack_unknown_task_404(self, client):
        r = client.post("/api/chat/ack/nope", headers=AUTH)
        assert r.status_code == 404


class TestSubscriptionStore:
    def test_round_trip_and_upsert(self, push_files):
        assert webpush_notify.add_subscription(_sub())
        assert webpush_notify.add_subscription(_sub())  # upsert, not dup
        assert webpush_notify.subscription_count() == 1
        assert webpush_notify.remove_subscription(_sub()["endpoint"])
        assert webpush_notify.subscription_count() == 0

    def test_malformed_rejected(self, push_files):
        assert not webpush_notify.add_subscription({})
        assert not webpush_notify.add_subscription(
            {"endpoint": "http://insecure", "keys": {"p256dh": "x", "auth": "y"}})
        assert not webpush_notify.add_subscription(
            {"endpoint": "https://ok", "keys": {}})
        assert webpush_notify.subscription_count() == 0

    def test_endpoints_store_and_validate(self, client, push_files):
        r = client.post("/api/push/subscribe", headers=AUTH,
                        json={"subscription": _sub()})
        assert r.status_code == 200 and r.json()["count"] == 1
        r = client.post("/api/push/subscribe", headers=AUTH,
                        json={"subscription": {"endpoint": "junk"}})
        assert r.status_code == 400
        r = client.post("/api/push/unsubscribe", headers=AUTH,
                        json={"endpoint": _sub()["endpoint"]})
        assert r.json()["removed"] is True

    def test_vapid_endpoint_reflects_config(self, client, push_files):
        r = client.get("/api/push/vapid", headers=AUTH)
        data = r.json()
        assert data["enabled"] is True
        assert data["key"] == json.loads(push_files[0].read_text())[
            "public_key_b64url"]


class TestBroadcast:
    def test_real_crypto_end_to_end(self, push_files, monkeypatch):
        """REAL pywebpush + py_vapid signing/encryption — only the HTTP
        POST is captured. This is the test that would have caught the
        PEM-format CRITICAL: any key-format mismatch raises inside
        pywebpush before the network layer."""
        import pywebpush as real_pywebpush
        webpush_notify.add_subscription(_real_sub())
        captured = {}

        class FakeResp:
            status_code = 201
            text = ""
            headers = {}

        class FakeSession:
            def post(self, url, *, data=None, headers=None, timeout=None, **kw):
                captured["url"] = url
                captured["auth"] = (headers or {}).get("Authorization", "")
                captured["body_len"] = len(data or b"")
                return FakeResp()

        orig_webpush = real_pywebpush.webpush

        def webpush_with_fake_session(*args, **kwargs):
            kwargs["requests_session"] = FakeSession()
            return orig_webpush(*args, **kwargs)

        monkeypatch.setattr(real_pywebpush, "webpush", webpush_with_fake_session)
        sent = webpush_notify.broadcast("Ghost", "real crypto", url="/x")
        assert sent == 1
        assert captured["url"].startswith("https://push.example/")
        assert captured["auth"].startswith("vapid ")  # real VAPID JWT header
        assert captured["body_len"] > 0               # real encrypted payload

    def test_dead_subscriptions_pruned(self, push_files):
        webpush_notify.add_subscription(_sub("https://push.example/dead"))
        webpush_notify.add_subscription(_sub("https://push.example/alive"))

        class FakeWPE(Exception):
            def __init__(self, status):
                self.response = MagicMock(status_code=status)

        def fake_webpush(subscription_info, **kw):
            if subscription_info["endpoint"].endswith("dead"):
                raise FakeWPE(410)
            return MagicMock()

        fake_mod = MagicMock(webpush=fake_webpush, WebPushException=FakeWPE)
        with patch.dict(sys.modules, {"pywebpush": fake_mod}):
            sent = webpush_notify.broadcast("t", "b", url="/x")
        assert sent == 1
        assert webpush_notify.subscription_count() == 1

    def test_no_vapid_means_disabled(self, push_files, monkeypatch):
        monkeypatch.setattr(webpush_notify, "_VAPID_FILE",
                            Path("/nonexistent/vapid.json"))
        monkeypatch.setattr(webpush_notify, "_vapid_cache", None)
        assert webpush_notify.public_key() is None
        assert webpush_notify.broadcast("t", "b") == 0


class TestReplyReadyPush:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_acked_task_never_pushes(self, monkeypatch):
        server.active_chat_tasks["t-ack"] = {"client_acked": True, "done": True}
        pushed = AsyncMock()
        monkeypatch.setattr(server.webpush_notify, "broadcast_async", pushed)
        monkeypatch.setattr(server, "_push_ack_grace_s", lambda: 0.0)
        try:
            self._run(server._push_if_unacked("t-ack", "hello"))
        finally:
            server.active_chat_tasks.pop("t-ack", None)
        pushed.assert_not_awaited()

    def test_unacked_task_pushes_with_preview(self, monkeypatch):
        server.active_chat_tasks["t-gone"] = {"done": True}
        pushed = AsyncMock(return_value=1)
        monkeypatch.setattr(server.webpush_notify, "broadcast_async", pushed)
        monkeypatch.setattr(server.webpush_notify, "subscription_count", lambda: 1)
        monkeypatch.setattr(server, "_push_ack_grace_s", lambda: 0.0)
        try:
            self._run(server._push_if_unacked("t-gone", "summarize my inbox"))
        finally:
            server.active_chat_tasks.pop("t-gone", None)
        pushed.assert_awaited_once()
        args = pushed.await_args
        assert "summarize my inbox" in args.args[1]
        from urllib.parse import quote as _q
        assert _q(KEY) in args.kwargs["url"]  # url carries the QUOTED key

    def test_cancelled_turn_never_pushes(self, monkeypatch):
        # User hit Stop: a "Reply ready" buzz 12s later is noise, not news.
        server.active_chat_tasks["t-can"] = {"done": True, "cancelled": True}
        pushed = AsyncMock()
        monkeypatch.setattr(server.webpush_notify, "broadcast_async", pushed)
        monkeypatch.setattr(server.webpush_notify, "subscription_count", lambda: 1)
        monkeypatch.setattr(server, "_push_ack_grace_s", lambda: 0.0)
        try:
            self._run(server._push_if_unacked("t-can", "x"))
        finally:
            server.active_chat_tasks.pop("t-can", None)
        pushed.assert_not_awaited()

    def test_cancel_endpoint_sets_the_flag(self, client):
        server.active_chat_tasks["t-flag"] = {
            "done": False, "background_task": None,
            "new_data_event": MagicMock(),
        }
        try:
            r = client.post("/api/chat/cancel/t-flag", headers=AUTH)
            assert r.status_code == 200
            assert server.active_chat_tasks["t-flag"]["cancelled"] is True
        finally:
            server.active_chat_tasks.pop("t-flag", None)

    def test_no_subscribers_no_push(self, monkeypatch):
        server.active_chat_tasks["t-nosub"] = {"done": True}
        pushed = AsyncMock()
        monkeypatch.setattr(server.webpush_notify, "broadcast_async", pushed)
        monkeypatch.setattr(server.webpush_notify, "subscription_count", lambda: 0)
        monkeypatch.setattr(server, "_push_ack_grace_s", lambda: 0.0)
        try:
            self._run(server._push_if_unacked("t-nosub", "x"))
        finally:
            server.active_chat_tasks.pop("t-nosub", None)
        pushed.assert_not_awaited()

    def test_last_user_text_extraction(self):
        assert server._last_user_text({"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"},
        ]}) == "second"
        assert server._last_user_text({}) == ""
        assert server._last_user_text({"messages": [{"role": "user",
                                                     "content": ["odd"]}]}) == ""
