"""Mandatory-tor (the DEFAULT) must force HF offline so the cached embedder
loads without the (guard-blocked) cleartext model-resolution network call.
Opt-outs: --no-mandatory-tor on argv, or GHOST_MANDATORY_TOR=0."""

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


def test_default_is_on(monkeypatch):
    # Fail-closed by default: nothing on argv, no env var → requested.
    monkeypatch.setattr(sys, "argv", ["main", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    assert env._mandatory_tor_requested() is True


def test_optout_via_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--no-mandatory-tor", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    assert env._mandatory_tor_requested() is False


def test_optout_via_envvar(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--port", "8000"])
    monkeypatch.setenv("GHOST_MANDATORY_TOR", "0")
    assert env._mandatory_tor_requested() is False


def test_explicit_argv_beats_env_optout(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--mandatory-tor"])
    monkeypatch.setenv("GHOST_MANDATORY_TOR", "0")
    assert env._mandatory_tor_requested() is True


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


def test_offline_not_forced_when_opted_out(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--no-mandatory-tor", "--port", "8000"])
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    _clear_offline(monkeypatch)
    env.ensure_disabled()
    # cold-install model download must still be possible when opted out
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


# ------------------------------------------------------------------ #
# The hardening has to land BEFORE the first heavy import            #
# ------------------------------------------------------------------ #

def _fresh(code):
    """Run `code` in a clean interpreter with the flags UNSET, so the
    only thing that can set them is ghost_agent's own import."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _P
    env = dict(os.environ)
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(k, None)
    env["PYTHONPATH"] = str(_P(__file__).resolve().parents[1] / "src")
    return subprocess.run([_sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=180)


def test_importing_any_ghost_module_freezes_hf_offline_before_transformers():
    """MEASURED FAILURE, not a hypothetical. `ghost_agent/__init__.py` was
    empty and `_env` was imported only by `main`, so a script that
    reached `core.agent` (→ `utils.token_counter` → `from transformers
    import AutoTokenizer`) loaded huggingface_hub with the flags unset.
    The library freezes `HF_HUB_OFFLINE` into a module CONSTANT at
    import, and `_OFFLINE_FLAGS` uses `setdefault`, so nothing downstream
    could correct it: a validation harness doing exactly that opened a
    cleartext HTTPS connection to a public CDN from the operator's own IP
    — against this project's hard no-identity-egress rule.

    The pin is the frozen CONSTANT, not the env var: setting the var
    after the import leaves the constant False, which is precisely how
    the fault stayed invisible.
    """
    r = _fresh(
        "import ghost_agent.utils.token_counter\n"
        "import huggingface_hub.constants as C\n"
        "print('OFFLINE=%s' % C.HF_HUB_OFFLINE)\n")
    assert r.returncode == 0, r.stderr[-800:]
    assert "OFFLINE=True" in r.stdout, r.stdout


def test_the_bare_package_import_is_enough():
    r = _fresh("import ghost_agent, os\n"
               "print('VAR=%s' % os.environ.get('HF_HUB_OFFLINE'))\n"
               "print('TEL=%s' % os.environ.get('HF_HUB_DISABLE_TELEMETRY'))")
    assert r.returncode == 0, r.stderr[-800:]
    assert "VAR=1" in r.stdout and "TEL=1" in r.stdout


def test_an_operator_override_still_wins():
    """`_OFFLINE_FLAGS` is setdefault by design — someone who has routed
    HF through the SOCKS proxy can still say so."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _P
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "0"
    env["PYTHONPATH"] = str(_P(__file__).resolve().parents[1] / "src")
    r = subprocess.run(
        [_sys.executable, "-c",
         "import ghost_agent, os; print('VAR=%s' % os.environ['HF_HUB_OFFLINE'])"],
        env=env, capture_output=True, text=True, timeout=180)
    assert "VAR=0" in r.stdout, r.stdout + r.stderr[-400:]


def test_the_LAST_flag_wins_like_argparse_does(monkeypatch):
    """`_env` scanned for `--no-mandatory-tor` first regardless of
    position while `main.py`'s BooleanOptionalAction takes the last, so
    one command line resolved to fail-closed Tor with HF still ONLINE —
    exactly the hole this module exists to close. The launcher passes
    `--mandatory-tor` and then appends "$@", so a reversed pair is one
    operator invocation away."""
    import ghost_agent._env as E
    monkeypatch.delenv("GHOST_MANDATORY_TOR", raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["x", "--no-mandatory-tor", "--mandatory-tor"])
    assert E._mandatory_tor_requested() is True
    monkeypatch.setattr(sys, "argv",
                        ["x", "--mandatory-tor", "--no-mandatory-tor"])
    assert E._mandatory_tor_requested() is False
