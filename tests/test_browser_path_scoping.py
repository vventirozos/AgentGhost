"""Regression: project-scoped path heal must collapse the host-root-relative
``sandbox/projects/<id>/`` prefix (and any ``<root>/projects/<id>/`` form),
not just ``/workspace/`` and ``projects/<id>/``.

Live bug (webOS build): the agent wrote ``browser-os.html`` (bare), then
navigated to ``file:///workspace/sandbox/projects/<id>/browser-os.html`` — the
``sandbox/`` segment went unhealed and the URL double-nested to
``/workspace/projects/<id>/sandbox/projects/<id>/browser-os.html`` → 404, three
times, until the loop-breaker fired.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.file_system import _get_safe_path
from ghost_agent.tools.browser import _resolve_file_url
from ghost_agent.tools.projects import _workspace_note


PID = "65738e769fb4"


def _scoped(tmp_path, root_name="sandbox"):
    """A project-scoped sandbox dir: <tmp>/<root_name>/projects/<id>."""
    sb = tmp_path / root_name / "projects" / PID
    sb.mkdir(parents=True)
    return sb


# ----------------------------------------------------------------- _get_safe_path

@pytest.mark.parametrize("inp", [
    f"sandbox/projects/{PID}/browser-os.html",
    f"/sandbox/projects/{PID}/browser-os.html",
    f"workspace/sandbox/projects/{PID}/browser-os.html",
    f"/workspace/sandbox/projects/{PID}/browser-os.html",
    f"projects/{PID}/browser-os.html",          # pre-existing heal still works
    f"/workspace/projects/{PID}/browser-os.html",
    "/workspace/browser-os.html",
    "browser-os.html",                           # bare
])
def test_all_routes_collapse_to_project_relative(tmp_path, inp):
    sb = _scoped(tmp_path)
    got = _get_safe_path(sb, inp)
    assert got == (sb / "browser-os.html").resolve(), f"{inp} -> {got}"


def test_nested_sandbox_prefixed_path(tmp_path):
    sb = _scoped(tmp_path)
    got = _get_safe_path(sb, f"sandbox/projects/{PID}/sub/app.js")
    assert got == (sb / "sub" / "app.js").resolve()


def test_heal_uses_actual_root_dir_name_not_hardcoded_sandbox(tmp_path):
    # The sandbox root dir is not always literally "sandbox" — the heal must
    # derive it from sandbox_dir.parent.parent.name.
    sb = _scoped(tmp_path, root_name="myroot")
    got = _get_safe_path(sb, f"myroot/projects/{PID}/x.txt")
    assert got == (sb / "x.txt").resolve()


def test_bare_project_prefix_dir_collapses_to_root(tmp_path):
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, f"sandbox/projects/{PID}") == sb.resolve()
    assert _get_safe_path(sb, f"projects/{PID}") == sb.resolve()


# ----------------------------------------------------------------- guards (no over-strip)

def test_literal_projects_subdir_not_over_stripped(tmp_path):
    # 'projects/notes.txt' is NOT 'projects/<id>/...' — must stay literal.
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, "projects/notes.txt") == (sb / "projects" / "notes.txt").resolve()


def test_literal_sandbox_subdir_not_over_stripped(tmp_path):
    # A real 'sandbox/' subdir inside the project (not the redundant
    # 'sandbox/projects/<id>/' prefix) is honored literally.
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, "sandbox/foo.txt") == (sb / "sandbox" / "foo.txt").resolve()


def test_traversal_still_blocked(tmp_path):
    sb = _scoped(tmp_path)
    with pytest.raises(ValueError):
        _get_safe_path(sb, f"sandbox/projects/{PID}/../../../escape.txt")


# ----------------------------------------------------------------- _resolve_file_url

@pytest.mark.parametrize("url", [
    f"file:///workspace/sandbox/projects/{PID}/browser-os.html",
    f"file:///sandbox/projects/{PID}/browser-os.html",
    f"file:///workspace/projects/{PID}/browser-os.html",
    "file:///workspace/browser-os.html",
    "file://browser-os.html",
])
def test_resolve_file_url_yields_correct_container_path(tmp_path, url):
    sb = _scoped(tmp_path)
    (sb / "browser-os.html").write_text("<html>")
    expected = f"file:///workspace/projects/{PID}/browser-os.html"
    assert _resolve_file_url(sb, url) == expected


def test_resolve_file_url_no_double_nest(tmp_path):
    sb = _scoped(tmp_path)
    (sb / "browser-os.html").write_text("<html>")
    out = _resolve_file_url(sb, f"file:///sandbox/projects/{PID}/browser-os.html")
    assert f"projects/{PID}/sandbox/projects/{PID}" not in out


# ----------------------------------------------------------------- workspace note

def test_workspace_note_warns_against_prefixes_and_drops_cleanup_path():
    note = _workspace_note(PID)
    # no longer leaks the host-root-relative cleanup path the model copied
    assert f"remove sandbox/projects/{PID}" not in note
    # explicitly steers away from the prefixes that double-nest
    assert "sandbox/" in note and "/workspace/" in note
    assert "bare" in note.lower()
