"""_get_safe_path must resolve a host-absolute sandbox path the same as
its relative form. The LLM sometimes echoes the full host path of a
sandbox file (e.g. /Users/x/.../sandbox/report.md) back into a
file_system call; that used to join onto sandbox_dir → phantom nested
path → "not found" → wasted strike + retry (the report-regeneration
trace burned ~5 turns on this)."""

import pytest

from ghost_agent.tools.file_system import _get_safe_path


@pytest.fixture
def sandbox(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    return sb


def test_host_absolute_sandbox_path_equals_relative(sandbox):
    rel = _get_safe_path(sandbox, "report.md")
    absform = _get_safe_path(sandbox, str(sandbox / "report.md"))
    assert absform == rel == (sandbox / "report.md").resolve()


def test_host_absolute_nested(sandbox):
    p = _get_safe_path(sandbox, str(sandbox / "sub" / "x.md"))
    assert p == (sandbox / "sub" / "x.md").resolve()


def test_host_absolute_sandbox_root(sandbox):
    assert _get_safe_path(sandbox, str(sandbox)) == sandbox.resolve()


def test_workspace_prefix_still_stripped(sandbox):
    assert _get_safe_path(sandbox, "/workspace/foo.py") == (sandbox / "foo.py").resolve()
    assert _get_safe_path(sandbox, "workspace/foo.py") == (sandbox / "foo.py").resolve()


def test_relative_paths_unchanged(sandbox):
    assert _get_safe_path(sandbox, "foo.py") == (sandbox / "foo.py").resolve()
    assert _get_safe_path(sandbox, "/foo.py") == (sandbox / "foo.py").resolve()


def test_traversal_still_rejected(sandbox):
    with pytest.raises(ValueError):
        _get_safe_path(sandbox, "../etc/passwd")


# --- Tilde-prefixed host paths (2026-07-05) ---------------------------------
# The model echoes the run command it just gave the user
# ("python3 ~/Data/AI/Data/sandbox/chess_client.py") back into file_system
# and used to burn a strike on the unexpanded "~".

def test_tilde_sandbox_path_equals_relative(sandbox, monkeypatch):
    monkeypatch.setenv("HOME", str(sandbox.parent))
    p = _get_safe_path(sandbox, f"~/{sandbox.name}/report.md")
    assert p == (sandbox / "report.md").resolve()


def test_tilde_nested_path(sandbox, monkeypatch):
    monkeypatch.setenv("HOME", str(sandbox.parent))
    p = _get_safe_path(sandbox, f"~/{sandbox.name}/chess/client.py")
    assert p == (sandbox / "chess" / "client.py").resolve()


def test_tilde_outside_sandbox_stays_inside_sandbox(sandbox, monkeypatch):
    # A "~" that expands OUTSIDE the sandbox must still resolve to a path
    # under the sandbox root (traversal safety invariant unchanged).
    monkeypatch.setenv("HOME", "/nonexistent_home_xyz")
    p = _get_safe_path(sandbox, "~/other/file.txt")
    assert str(p).startswith(str(sandbox.resolve()))
