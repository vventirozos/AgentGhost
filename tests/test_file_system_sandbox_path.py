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
