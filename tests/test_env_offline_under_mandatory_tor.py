"""--mandatory-tor must force HF offline so the cached embedder loads
without the (guard-blocked) cleartext model-resolution network call."""

import os
import sys

import ghost_agent._env as env

_OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def _clear_offline(monkeypatch):
    for k in _OFFLINE:
        monkeypatch.delenv(k, raising=False)


def test_requested_via_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--mandatory-tor", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    assert env._mandatory_tor_requested() is True


def test_requested_via_envvar(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--port", "8000"])
    monkeypatch.setenv("GHOST_MANDATORY_TOR", "1")
    assert env._mandatory_tor_requested() is True


def test_not_requested(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    assert env._mandatory_tor_requested() is False


def test_offline_forced_under_mandatory_tor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--mandatory-tor"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    _clear_offline(monkeypatch)
    env.ensure_disabled()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"
    # telemetry flags are always set regardless
    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"


def test_offline_not_forced_without_mandatory_tor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    _clear_offline(monkeypatch)
    env.ensure_disabled()
    # cold-install model download must still be possible in normal mode
    assert os.environ.get("HF_HUB_OFFLINE") is None
    # telemetry hardening still applies
    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"


def test_offline_respects_operator_override(monkeypatch):
    # setdefault: an operator who routes HF through the SOCKS proxy can
    # explicitly keep it online.
    monkeypatch.setattr(sys, "argv", ["main", "--mandatory-tor"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    env.ensure_disabled()
    assert os.environ.get("HF_HUB_OFFLINE") == "0"
