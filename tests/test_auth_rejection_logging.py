"""Auth-rejection log noise from the agent's own test suite (2026-07-27).

`functional_live_test.py` deliberately probes `/api/health` and
`/api/game/move` with a missing and a wrong key to prove auth is enforced.
Every run therefore emitted WARNING lines indistinguishable from a real
intruder's — which is precisely how a security signal gets learned-ignored.

The fix re-levels those to INFO. The security-relevant property is that it
can NEVER be used to hide anything:

  * the line is ALWAYS emitted — only its level changes;
  * the 403 is unaffected;
  * the marker is honoured ONLY from loopback, so a remote attacker cannot
    lower their own level by setting a header (headers are attacker-
    controlled; loopback is not).
"""

import pytest

from ghost_agent.api import routes


def _levels(monkeypatch):
    """Capture (level, message) pairs from the auth logger."""
    seen = []

    def _fake(title, msg, **kw):
        seen.append((kw.get("level"), f"{title}: {msg}"))

    monkeypatch.setattr(routes, "pretty_log", _fake)
    return seen


class _Req:
    """Minimal Request stand-in: headers + client host."""

    def __init__(self, ua="", ip="127.0.0.1", path="/api/health"):
        self.headers = {"user-agent": ua} if ua else {}
        self.client = type("C", (), {"host": ip})() if ip else None
        self.url = type("U", (), {"path": path})()


def _call(monkeypatch, req, key="wrong"):
    """Run verify_api_key against a stubbed agent with auth configured."""
    import asyncio

    agent = type("A", (), {})()
    agent.context = type("C", (), {})()
    agent.context.args = type("Ar", (), {"api_key": "real-key"})()
    monkeypatch.setattr(routes, "get_agent", lambda _r: agent)
    from fastapi import HTTPException
    try:
        asyncio.get_event_loop().run_until_complete(
            routes.verify_api_key(req, key))
        return None
    except HTTPException as e:
        return e.status_code


class TestStillRejects:
    """The level change must not touch the DECISION."""

    @pytest.mark.parametrize("ua,ip", [
        ("ghost-functional-test", "127.0.0.1"),
        ("ghost-functional-test", "10.0.0.9"),
        ("curl/8.0", "10.0.0.9"),
        ("", ""),
    ])
    def test_bad_key_is_always_403(self, monkeypatch, ua, ip):
        _levels(monkeypatch)
        assert _call(monkeypatch, _Req(ua=ua, ip=ip)) == 403

    def test_correct_key_is_accepted(self, monkeypatch):
        _levels(monkeypatch)
        assert _call(monkeypatch, _Req(), key="real-key") is None


class TestAlwaysLogged:
    """Nothing may be suppressed — only re-levelled."""

    @pytest.mark.parametrize("ua,ip", [
        ("ghost-functional-test", "127.0.0.1"),
        ("ghost-functional-test", "203.0.113.5"),
        ("curl/8.0", "127.0.0.1"),
    ])
    def test_a_line_is_always_emitted(self, monkeypatch, ua, ip):
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua=ua, ip=ip))
        assert len(seen) == 1, "an auth rejection must never go unlogged"

    def test_ip_and_ua_are_recorded(self, monkeypatch):
        """A real hit must stay identifiable in the log."""
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua="curl/8.0", ip="203.0.113.5"))
        _, msg = seen[0]
        assert "203.0.113.5" in msg and "curl/8.0" in msg

    def test_key_bytes_are_never_logged(self, monkeypatch):
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(), key="sup3r-s3cret")
        assert "sup3r-s3cret" not in seen[0][1]


class TestMarkerCannotBeAbused:
    """The header alone must not lower the level — that would hand an
    attacker a switch to mute their own probes."""

    def test_loopback_plus_marker_is_info(self, monkeypatch):
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua="ghost-functional-test", ip="127.0.0.1"))
        assert seen[0][0] == "INFO"
        assert "own functional suite" in seen[0][1]

    @pytest.mark.parametrize("ip", ["203.0.113.5", "10.0.0.9", "192.168.1.20"])
    def test_marker_from_a_REMOTE_host_stays_warning(self, monkeypatch, ip):
        """THE security property: spoofing the UA off-host changes nothing."""
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua="ghost-functional-test", ip=ip))
        assert seen[0][0] == "WARNING"
        assert "own functional suite" not in seen[0][1]

    def test_loopback_without_the_marker_stays_warning(self, monkeypatch):
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua="curl/8.0", ip="127.0.0.1"))
        assert seen[0][0] == "WARNING"

    def test_missing_client_info_stays_warning(self, monkeypatch):
        """Unknown origin resolves toward the louder level."""
        seen = _levels(monkeypatch)
        _call(monkeypatch, _Req(ua="ghost-functional-test", ip=""))
        assert seen[0][0] == "WARNING"


class TestSuiteSendsTheMarker:
    def test_functional_suite_identifies_itself(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "functional_live_test.py").read_text()
        assert '"User-Agent": "ghost-functional-test"' in src

    def test_marker_matches_on_both_sides(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "functional_live_test.py").read_text()
        assert routes._SELF_TEST_UA in src, (
            "the suite's marker and the server's constant must agree")
