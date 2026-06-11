"""Boot-time security policy: no hardcoded default API key, refuse
non-loopback binds without an explicit key choice, and --mandatory-tor
defaults ON (fail-closed Tor is the default stance, not opt-in)."""

import sys

import pytest

from ghost_agent.main import enforce_api_key_policy, parse_args

# --- parse_args defaults -------------------------------------------------


def test_api_key_has_no_hardcoded_default(monkeypatch):
    monkeypatch.delenv("GHOST_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["main"])
    args = parse_args()
    assert args.api_key is None


def test_api_key_env_var_respected(monkeypatch):
    monkeypatch.setenv("GHOST_API_KEY", "real-secret")
    monkeypatch.setattr(sys, "argv", ["main"])
    assert parse_args().api_key == "real-secret"


def test_mandatory_tor_defaults_on(monkeypatch):
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    monkeypatch.setattr(sys, "argv", ["main"])
    assert parse_args().mandatory_tor is True


def test_mandatory_tor_argv_optout(monkeypatch):
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    monkeypatch.setattr(sys, "argv", ["main", "--no-mandatory-tor"])
    assert parse_args().mandatory_tor is False


def test_mandatory_tor_env_optout(monkeypatch):
    monkeypatch.setenv("GHOST_MANDATORY_TOR", "0")
    monkeypatch.setattr(sys, "argv", ["main"])
    assert parse_args().mandatory_tor is False


def test_mandatory_tor_argv_beats_env_optout(monkeypatch):
    monkeypatch.setenv("GHOST_MANDATORY_TOR", "0")
    monkeypatch.setattr(sys, "argv", ["main", "--mandatory-tor"])
    assert parse_args().mandatory_tor is True


# --- enforce_api_key_policy ----------------------------------------------


def test_refuses_nonloopback_bind_with_unset_key():
    with pytest.raises(SystemExit):
        enforce_api_key_policy(None, "0.0.0.0")


def test_refuses_tailscale_style_bind_with_unset_key():
    with pytest.raises(SystemExit):
        enforce_api_key_policy(None, "100.64.0.7")


def test_allows_loopback_bind_with_unset_key():
    enforce_api_key_policy(None, "127.0.0.1")
    enforce_api_key_policy(None, "localhost")
    enforce_api_key_policy(None, "::1")


def test_allows_explicit_empty_key_on_network_bind_with_warning(capsys):
    # --api-key '' is an informed operator choice (e.g. trusted mesh):
    # boot proceeds, but a loud warning lands on stdout.
    enforce_api_key_policy("", "0.0.0.0")
    assert "DISABLED" in capsys.readouterr().out


def test_allows_real_key_on_network_bind(capsys):
    enforce_api_key_policy("a-real-secret", "0.0.0.0")
    assert "WARNING" not in capsys.readouterr().out


def test_warns_on_old_publicly_known_key(capsys):
    enforce_api_key_policy("ghost-secret-123", "127.0.0.1")
    assert "publicly-known" in capsys.readouterr().out
