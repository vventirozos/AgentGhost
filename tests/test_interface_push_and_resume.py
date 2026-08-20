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
                                "truncated": False, "cancelled": False,
                                "chunks": 2}
            r = client.post("/api/chat/ack/t-live", headers=AUTH)
            assert r.json() == {"ok": True}
            assert server.active_chat_tasks["t-live"]["client_acked"] is True
        finally:
            server.active_chat_tasks.pop("t-live", None)

    def test_a_cancelled_task_says_so(self, client):
        """The probe omitted `cancelled`, so the silent-resume path saw a
        live task and replayed a stream the user deliberately stopped
        (R2 lens A)."""
        server.active_chat_tasks["t-stopped"] = {
            "buffer": [b"a"], "buffer_size": 1, "done": True, "error": None,
            "truncated": False, "cancelled": True,
        }
        try:
            r = client.get("/api/chat/task/t-stopped/state", headers=AUTH)
            assert r.json()["cancelled"] is True
        finally:
            server.active_chat_tasks.pop("t-stopped", None)

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


class TestPushHealthTellsTheTruth:
    """R2 lens A: the health signal sat on a WEAKER condition than sending."""

    def test_a_corrupt_PEM_reports_push_as_OFF(self, push_files, monkeypatch,
                                               tmp_path):
        import json as _json
        vapid = tmp_path / "vapid.json"
        vapid.write_text(_json.dumps({
            "public_key_b64url": "BOGUSPUBLICKEY",
            "private_key_pem": "-----BEGIN PRIVATE KEY-----\nnot-a-key\n"
                               "-----END PRIVATE KEY-----\n"}))
        monkeypatch.setattr(webpush_notify, "_VAPID_FILE", vapid)
        monkeypatch.setattr(webpush_notify, "_vapid_cache", None)
        monkeypatch.setattr(webpush_notify, "_vapid_signer", None)
        # The weaker predicate still says yes — that is the trap.
        assert webpush_notify.public_key() == "BOGUSPUBLICKEY"
        assert webpush_notify.can_send() is False, (
            "push reports itself usable with a PEM that cannot sign")
        assert webpush_notify.broadcast("t", "b") == 0

    def test_the_endpoint_reports_can_send_not_public_key(self, client,
                                                          monkeypatch):
        monkeypatch.setattr(webpush_notify, "public_key", lambda: "PUB")
        monkeypatch.setattr(webpush_notify, "can_send", lambda: False)
        r = client.get("/api/push/vapid", headers=AUTH)
        assert r.status_code == 200
        assert r.json()["enabled"] is False, (
            "the UI is told push works because a key file parses")


class TestSubscriptionsAreNotDestroyedByABadRead:
    """R2 lens A: `_load_subs` failed open to {}, and the next write
    persisted that emptiness over every registered device."""

    def _write_subs(self, monkeypatch, tmp_path, text):
        p = tmp_path / "push_subs.json"
        p.write_text(text)
        monkeypatch.setattr(webpush_notify, "_SUBS_FILE", p)
        return p

    def test_a_truncated_file_is_not_overwritten_by_a_new_subscribe(
            self, monkeypatch, tmp_path):
        p = self._write_subs(monkeypatch, tmp_path,
                             '{"https://push.example/a": {"endpoint": "a"')
        ok = webpush_notify.add_subscription({
            "endpoint": "https://push.example/new",
            "keys": {"p256dh": "x", "auth": "y"}})
        assert ok is False, "a new device was stored over an unreadable file"
        assert p.read_text().startswith('{"https://push.example/a"'), (
            "the unrepaired file was clobbered — every other device lost")

    def test_a_json_LIST_is_refused_too(self, monkeypatch, tmp_path):
        p = self._write_subs(monkeypatch, tmp_path, '[{"endpoint": "a"}]')
        assert webpush_notify.add_subscription({
            "endpoint": "https://push.example/new",
            "keys": {"p256dh": "x", "auth": "y"}}) is False
        assert p.read_text() == '[{"endpoint": "a"}]'

    def test_an_ABSENT_file_is_still_the_empty_case(self, monkeypatch, tmp_path):
        monkeypatch.setattr(webpush_notify, "_SUBS_FILE",
                            tmp_path / "does-not-exist.json")
        assert webpush_notify.subscription_count() == 0
        assert webpush_notify.add_subscription({
            "endpoint": "https://push.example/new",
            "keys": {"p256dh": "x", "auth": "y"}}) is True

    def test_read_only_callers_never_raise(self, monkeypatch, tmp_path):
        self._write_subs(monkeypatch, tmp_path, "not json at all")
        assert webpush_notify.subscription_count() == 0
        assert webpush_notify.broadcast("t", "b") == 0


class TestEveryPushCarriesATimeout:
    def test_webpush_is_called_with_an_explicit_timeout(self, push_files,
                                                        monkeypatch):
        """pywebpush's own default is `timeout=None`, and it passes that
        INTO `WebPusher.send`, so `kwargs.pop("timeout", 10000)` yields None
        and `requests.post` blocks forever. One unresponsive endpoint then
        parks the notify poller for the life of the process (R2 lens A)."""
        seen = {}

        def _fake_webpush(**kw):
            seen.update(kw)
            return type("R", (), {"status_code": 201})()

        import sys, types
        fake = types.ModuleType("pywebpush")
        fake.webpush = _fake_webpush
        fake.WebPushException = type("WebPushException", (Exception,), {})
        monkeypatch.setitem(sys.modules, "pywebpush", fake)
        monkeypatch.setattr(webpush_notify, "_vapid_key_object",
                            lambda: object())
        monkeypatch.setattr(webpush_notify, "_load_subs_or_empty", lambda: {
            "https://push.example/a": {"endpoint": "https://push.example/a",
                                       "keys": {"p256dh": "x", "auth": "y"}}})
        assert webpush_notify.broadcast("t", "b") == 1
        assert "timeout" in seen, "the push has no timeout — it can hang forever"
        assert 0 < seen["timeout"] <= 60, seen["timeout"]


class TestPushFailuresAreLegible:
    """R3 lens A: the R2 fixes stopped the data loss and left the REPORTS
    wrong."""

    def test_a_corrupt_store_is_a_503_not_a_client_error(self, client,
                                                         monkeypatch, tmp_path):
        p = tmp_path / "push_subs.json"
        p.write_text('{"https://push.example/a": {"endpoint": "a"')
        monkeypatch.setattr(webpush_notify, "_SUBS_FILE", p)
        r = client.post("/api/push/subscribe", headers=AUTH, json={
            "subscription": {"endpoint": "https://push.example/new",
                             "keys": {"p256dh": "x", "auth": "y"}}})
        assert r.status_code == 503, (
            f"a server-side corrupt store is reported to the device as ITS "
            f"fault: {r.status_code} {r.text[:120]}")
        assert "unreadable" in r.text.lower()

    def test_a_healthy_store_still_400s_a_malformed_payload(self, client,
                                                            monkeypatch, tmp_path):
        p = tmp_path / "push_subs.json"
        p.write_text("{}")
        monkeypatch.setattr(webpush_notify, "_SUBS_FILE", p)
        r = client.post("/api/push/subscribe", headers=AUTH,
                        json={"subscription": {"endpoint": "not-https"}})
        assert r.status_code == 400

    def test_a_dead_push_subsystem_says_so_at_every_exit(self, monkeypatch,
                                                          tmp_path, caplog):
        """The "reached 0 of N" alarm could not fire for the two likeliest
        causes — no VAPID config, and an unusable PEM — because both return
        earlier (R3 lens A)."""
        import logging
        monkeypatch.setattr(webpush_notify, "_load_subs_or_empty", lambda: {
            "https://push.example/a": {"endpoint": "https://push.example/a"}})
        monkeypatch.setattr(webpush_notify, "vapid_config", lambda: None)
        with caplog.at_level(logging.WARNING):
            assert webpush_notify.broadcast("t", "b") == 0
        blob = " ".join(r.getMessage() for r in caplog.records)
        assert "push is off" in blob, (
            f"push is entirely dead and nothing said so: {blob!r}")


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

    def test_ack_grace_is_a_real_wait_read_AFTER_the_sleep(self, monkeypatch):
        """Every other test in this class patches the grace to 0.0, so the
        grace was pinned NOWHERE: a 0-second default (or checking the ack
        flag before sleeping) would push to the phone the instant the turn
        finished, while the page that was about to ack was still parsing the
        stream — a buzz for a reply the user is already reading (review R1
        M12e).

        Two things are asserted: the default is long enough for a live page
        to ack, and the flag is read AFTER the wait, not before."""
        assert server._push_ack_grace_s() >= 5.0, (
            "the ack grace is too short for a live page to ack in")
        monkeypatch.setenv("GHOST_PUSH_ACK_GRACE", "3.5")
        assert server._push_ack_grace_s() == 3.5, "the env override is dead"
        monkeypatch.setenv("GHOST_PUSH_ACK_GRACE", "not-a-number")
        assert server._push_ack_grace_s() >= 5.0, (
            "a malformed override collapses the grace instead of falling back")
        monkeypatch.delenv("GHOST_PUSH_ACK_GRACE", raising=False)

        server.active_chat_tasks["t-late-ack"] = {"done": True}
        pushed = AsyncMock(return_value=1)
        monkeypatch.setattr(server.webpush_notify, "broadcast_async", pushed)
        monkeypatch.setattr(server.webpush_notify, "subscription_count", lambda: 1)
        monkeypatch.setattr(server, "_push_ack_grace_s", lambda: 0.15)

        async def _acks_during_the_grace():
            probe = asyncio.ensure_future(
                server._push_if_unacked("t-late-ack", "x"))
            await asyncio.sleep(0.05)      # inside the grace window
            # ⚠ REPLACE the registry entry, do not mutate the dict in place.
            # Mutating it let a probe that looked the task up BEFORE sleeping
            # still observe the ack, so hoisting the lookup above the sleep —
            # the actual defect — survived the mutation.
            server.active_chat_tasks["t-late-ack"] = {"done": True,
                                                      "client_acked": True}
            await probe

        try:
            self._run(_acks_during_the_grace())
        finally:
            server.active_chat_tasks.pop("t-late-ack", None)
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
