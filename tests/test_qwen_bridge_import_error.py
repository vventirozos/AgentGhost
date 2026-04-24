"""Regression test for the qwen_bridge import-error message.

Before the fix, a missing `soundfile` (transitive dep of qwen-agent
pulled in via `qwen_agent.utils.utils`) crashed `import
ghost_agent.tools.qwen_bridge` with a cryptic
``ModuleNotFoundError: No module named 'soundfile'`` coming from
deep inside qwen_agent's internals. Users trying to use the Qwen
agent variant had no breadcrumb pointing at the real fix.

After the fix:
  * `soundfile>=0.12.0` is pinned in requirements.txt so fresh
    installs never hit the bug.
  * The qwen_bridge module wraps the `from qwen_agent.tools.base
    import ...` line in a try/except that re-raises as ImportError
    with an actionable message (how to install, note about libsndfile
    on Linux, clarification that the default agent path doesn't need
    qwen_bridge at all).
"""

import importlib
import sys
from pathlib import Path

import pytest


def test_qwen_bridge_imports_cleanly_in_current_env():
    """Sanity: with soundfile installed (it's in requirements.txt),
    qwen_bridge imports without raising. If this fails, the venv is
    missing a declared dep."""
    # Purge any cached module so we re-run the import path.
    for mod_name in list(sys.modules):
        if mod_name.startswith("ghost_agent.tools.qwen_bridge"):
            del sys.modules[mod_name]
    mod = importlib.import_module("ghost_agent.tools.qwen_bridge")
    assert hasattr(mod, "set_context")
    assert hasattr(mod, "_current_ctx")
    assert hasattr(mod, "GhostFileSystem")


def test_qwen_bridge_source_guards_import_with_helpful_message():
    """Static check: the source of qwen_bridge.py must include the
    try/except wrapper AND a hint pointing at soundfile. A future
    refactor that strips this would silently reintroduce the cryptic
    error when soundfile is missing."""
    qb_path = Path(__file__).resolve().parent.parent / "src" / "ghost_agent" / "tools" / "qwen_bridge.py"
    src = qb_path.read_text()
    assert "try:" in src and "except ModuleNotFoundError" in src, (
        "qwen_bridge must wrap its qwen_agent import in try/except"
    )
    assert "soundfile" in src.lower(), (
        "error message must mention soundfile so the user knows what to install"
    )
    assert "pip install" in src.lower() or "install soundfile" in src.lower(), (
        "error must tell the user how to fix it"
    )


def test_soundfile_pinned_in_requirements():
    """If anyone removes soundfile from requirements.txt, fresh
    installs regress to the cryptic error. Pin the presence here."""
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    req = req_path.read_text()
    assert "soundfile" in req, (
        "requirements.txt must declare soundfile (qwen-agent transitive dep)"
    )


def test_qwen_bridge_import_error_names_default_path_is_fine():
    """The error message must clarify that the default agent path
    does NOT need qwen_bridge, so a user who hit the error on the
    Qwen variant doesn't wrongly conclude their whole install is
    broken."""
    qb_path = Path(__file__).resolve().parent.parent / "src" / "ghost_agent" / "tools" / "qwen_bridge.py"
    src = qb_path.read_text().lower()
    assert "default agent path" in src or "default path" in src or "ghost_agent.main" in src
