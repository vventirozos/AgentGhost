"""Tests for ghost_agent._env (telemetry hardening)."""

import importlib
import os

import pytest

import ghost_agent._env as env_module


def test_import_side_effect_sets_env_vars():
    """Fresh import should leave every required flag set."""
    # Wipe one to prove the import-time application also fires on reload.
    os.environ.pop("ANONYMIZED_TELEMETRY", None)
    importlib.reload(env_module)
    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"


def test_ensure_disabled_is_idempotent():
    env_module.ensure_disabled()
    env_module.ensure_disabled()
    ok, missing = env_module.check_disabled()
    assert ok, f"missing: {missing}"


def test_check_disabled_reports_missing_keys(monkeypatch):
    # Temporarily override a required flag.
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    ok, missing = env_module.check_disabled()
    assert not ok
    assert "HF_HUB_DISABLE_TELEMETRY" in missing


def test_required_flags_cover_known_leakers():
    """Guards against silently dropping a telemetry opt-out for a lib
    we already know phones home."""
    required_keys = set(env_module._REQUIRED_FLAGS.keys())
    for key in (
        "ANONYMIZED_TELEMETRY",   # chromadb
        "POSTHOG_DISABLED",       # posthog-client SDKs
        "CHROMA_TELEMETRY_IMPL",  # chromadb v2
        "HF_HUB_DISABLE_TELEMETRY",  # huggingface-hub
    ):
        assert key in required_keys, f"{key} must be opted out"


def test_probe_passes_after_import():
    """The eval regression probe should go green now that the env
    module is the source of truth."""
    # Re-import to ensure a clean state.
    importlib.reload(env_module)
    from ghost_agent.eval.tasks import _load_regression_probes
    probes = _load_regression_probes()
    probe = next(p for p in probes if "telemetry" in p.task_id)
    ok, reason = probe.validate(None, None)
    assert ok, f"probe failed: {reason}"


def test_probe_fails_loudly_when_flag_missing(monkeypatch):
    """If someone ever removes an env assignment, the probe must
    surface the specific missing keys, not just a vague boolean."""
    monkeypatch.setenv("POSTHOG_DISABLED", "")
    from ghost_agent.eval.tasks import _load_regression_probes
    probes = _load_regression_probes()
    probe = next(p for p in probes if "telemetry" in p.task_id)
    ok, reason = probe.validate(None, None)
    assert not ok
    assert "POSTHOG_DISABLED" in reason
