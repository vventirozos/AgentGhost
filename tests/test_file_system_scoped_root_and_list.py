"""Regressions for the 2026-07-14 file_system audit.

Live failure (ghost-agent.log, sessions 1D/77): the agent git-cloned a repo
via ``execute`` using an absolute ``/workspace/analysis/...`` path — the files
landed at the sandbox ROOT while every file_system op stayed scoped to the
project dir. ``list_files`` said the sandbox was EMPTY, reads of
``analysis/agentghost/README.md`` said "does not exist", and the model gave up
on file_system entirely, falling back to raw shell commands.

Covered here:
  * _get_safe_path: existence-aware root anchoring for /workspace/... and
    host-absolute paths under a project-scoped sandbox (heal default kept).
  * read-only root fallback for read / inspect / read_chunked (+ NOTE), and
    the explicit no-silent-edit guidance from replace.
  * tool_list_files: `path` subdirectory scoping, count-aware truncation,
    deterministic order, and the empty-workspace root hint.
  * dispatcher: list aliases + path passthrough, inspect `lines` passthrough.
  * downloader: single (not double) Tor identity rotation per httpx attempt.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path,
    _scoped_root_fallback,
    tool_file_system,
    tool_inspect_file,
    tool_list_files,
    tool_read_file,
    tool_read_document_chunked,
    tool_replace_text,
)

PID = "abc123def456"


def _scoped(tmp_path):
    """A project-scoped sandbox: <tmp>/sandbox/projects/<id>. Returns
    (scoped_dir, outer_root)."""
    root = tmp_path / "sandbox"
    sb = root / "projects" / PID
    sb.mkdir(parents=True)
    return sb, root


# ------------------------------------------------- _get_safe_path root anchor

def test_workspace_path_resolves_to_root_when_file_exists_there(tmp_path):
    sb, root = _scoped(tmp_path)
    target = root / "analysis" / "agentghost" / "README.md"
    target.parent.mkdir(parents=True)
    target.write_text("# readme")
    got = _get_safe_path(sb, "/workspace/analysis/agentghost/README.md")
    assert got == target.resolve()


def test_workspace_path_stays_scoped_when_nothing_exists(tmp_path):
    # The historical heal: /workspace/X for a brand-new file lands in the
    # project dir (write-then-execute symmetry relies on this).
    sb, root = _scoped(tmp_path)
    got = _get_safe_path(sb, "/workspace/new-file.txt")
    assert got == (sb / "new-file.txt").resolve()


def test_workspace_path_prefers_scoped_copy_when_both_exist(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "dup.txt").write_text("root")
    (sb / "dup.txt").write_text("scoped")
    got = _get_safe_path(sb, "/workspace/dup.txt")
    assert got == (sb / "dup.txt").resolve()


def test_workspace_new_file_inside_root_tree_anchors_to_root(tmp_path):
    # A NEW file whose parent dir only exists at the root (an
    # execute-created tree, e.g. a clone) must land inside that tree.
    sb, root = _scoped(tmp_path)
    (root / "analysis" / "repo").mkdir(parents=True)
    got = _get_safe_path(sb, "/workspace/analysis/repo/notes.md")
    assert got == (root / "analysis" / "repo" / "notes.md").resolve()


def test_host_absolute_path_under_outer_root_anchors_to_root(tmp_path):
    sb, root = _scoped(tmp_path)
    target = root / "data.csv"
    target.write_text("a,b\n")
    got = _get_safe_path(sb, str(target))
    assert got == target.resolve()


def test_workspace_traversal_still_blocked_when_root_anchored(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "analysis").mkdir()
    with pytest.raises(ValueError):
        _get_safe_path(sb, "/workspace/analysis/../../../etc/passwd")


def test_destructive_op_on_outer_root_rejected(tmp_path):
    sb, root = _scoped(tmp_path)
    with pytest.raises(ValueError, match="destructive"):
        _get_safe_path(sb, str(root), allow_root=False)
    with pytest.raises(ValueError, match="destructive"):
        _get_safe_path(sb, "/workspace", allow_root=False)


# --------------------------------------------------------- root fallback (RO)

def test_scoped_root_fallback_finds_root_file(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "clone").mkdir()
    (root / "clone" / "main.py").write_text("print('hi')\n")
    fb = _scoped_root_fallback(sb, "clone/main.py")
    assert fb == (root / "clone" / "main.py").resolve()
    assert _scoped_root_fallback(sb, "clone/missing.py") is None


def test_scoped_root_fallback_none_for_unscoped(tmp_path):
    assert _scoped_root_fallback(tmp_path, "anything.txt") is None


@pytest.mark.asyncio
async def test_read_falls_back_to_root_with_note(tmp_path):
    sb, root = _scoped(tmp_path)
    f = root / "analysis" / "README.md"
    f.parent.mkdir(parents=True)
    f.write_text("# hello root")
    out = await tool_read_file("analysis/README.md", sb)
    assert "# hello root" in out
    assert "sandbox ROOT" in out
    assert "/workspace/analysis/README.md" in out


@pytest.mark.asyncio
async def test_read_missing_everywhere_still_errors(tmp_path):
    sb, root = _scoped(tmp_path)
    out = await tool_read_file("nowhere.txt", sb)
    assert out.startswith("Error")


@pytest.mark.asyncio
async def test_inspect_falls_back_to_root(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "peek.txt").write_text("l1\nl2\nl3\n")
    out = await tool_inspect_file("peek.txt", sb, lines=2)
    assert "l1" in out and "l2" in out and "l3" not in out
    assert "sandbox-ROOT" in out


@pytest.mark.asyncio
async def test_inspect_coerces_bad_lines_value(tmp_path):
    (tmp_path / "peek.txt").write_text("\n".join(f"line{i}" for i in range(20)))
    out = await tool_inspect_file("peek.txt", tmp_path, lines=None)
    assert "line9" in out and "line10" not in out  # default 10


@pytest.mark.asyncio
async def test_read_chunked_falls_back_to_root(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "big.txt").write_text("chunked-content " * 10)
    out = await tool_read_document_chunked("big.txt", sb, page=1)
    assert "chunked-content" in out


@pytest.mark.asyncio
async def test_replace_does_not_silently_edit_root_copy(tmp_path):
    sb, root = _scoped(tmp_path)
    f = root / "app.py"
    f.write_text("x = 1\n")
    out = await tool_replace_text("app.py", "x = 1", "x = 2", sb)
    assert out.startswith("Error")
    assert "/workspace/app.py" in out
    assert f.read_text() == "x = 1\n"  # untouched


# ------------------------------------------------------------ tool_list_files

@pytest.mark.asyncio
async def test_list_files_scopes_to_subdirectory(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("x")
    (tmp_path / "top.txt").write_text("x")
    out = await tool_list_files(tmp_path, path="sub")
    assert "sub/inner.txt" in out
    assert "top.txt" not in out


@pytest.mark.asyncio
async def test_list_files_path_missing_returns_guidance(tmp_path):
    out = await tool_list_files(tmp_path, path="ghost-dir")
    assert out.startswith("Error")
    assert "ghost-dir" in out


@pytest.mark.asyncio
async def test_list_files_path_to_file_says_use_read(tmp_path):
    (tmp_path / "solo.txt").write_text("x")
    out = await tool_list_files(tmp_path, path="solo.txt")
    assert "FILE" in out and "operation='read'" in out


@pytest.mark.asyncio
async def test_list_files_truncation_reports_hidden_count(tmp_path):
    for i in range(230):
        (tmp_path / f"f{i:03d}.txt").write_text("x")
    out = await tool_list_files(tmp_path)
    assert "30 more files NOT shown" in out
    assert "230 total" in out
    assert "path='<subdir>'" in out


@pytest.mark.asyncio
async def test_list_files_no_truncation_note_at_exact_cap(tmp_path):
    for i in range(200):
        (tmp_path / f"f{i:03d}.txt").write_text("x")
    out = await tool_list_files(tmp_path)
    assert "more files NOT shown" not in out


@pytest.mark.asyncio
async def test_list_files_empty_scoped_hints_at_root_files(tmp_path):
    sb, root = _scoped(tmp_path)
    (root / "analysis").mkdir()
    (root / "loose.txt").write_text("x")
    out = await tool_list_files(sb)
    assert "[Empty]" in out
    assert "analysis/" in out and "loose.txt" in out
    assert "/workspace/" in out  # teaches the reachable path form


@pytest.mark.asyncio
async def test_list_files_empty_unscoped_has_no_hint(tmp_path):
    out = await tool_list_files(tmp_path)
    assert "[Empty]" in out
    assert "sandbox ROOT" not in out


@pytest.mark.asyncio
async def test_list_files_workspace_path_lists_root_tree_scoped(tmp_path):
    # The exact live sequence: clone under /workspace/analysis via execute,
    # then list /workspace/analysis/agentghost/src from a scoped project.
    sb, root = _scoped(tmp_path)
    src = root / "analysis" / "agentghost" / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text("def run():\n    pass\n")
    out = await tool_list_files(sb, path="/workspace/analysis/agentghost/src")
    assert "main.py" in out
    # Entries carry a path the model can feed straight back into read.
    assert "/workspace/analysis/agentghost/src/main.py" in out


# ---------------------------------------------------------------- dispatcher

@pytest.mark.asyncio
async def test_dispatcher_list_passes_path_through(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("x")
    (tmp_path / "top.txt").write_text("x")
    out = await tool_file_system(operation="list_files", sandbox_dir=tmp_path,
                                 path="sub")
    assert "sub/inner.txt" in out and "top.txt" not in out


@pytest.mark.asyncio
async def test_dispatcher_list_aliases(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    for op in ("list", "ls", "dir", "tree", "list_dir", "list_directory"):
        out = await tool_file_system(operation=op, sandbox_dir=tmp_path)
        assert "a.txt" in out, f"alias {op!r} did not list"


@pytest.mark.asyncio
async def test_dispatcher_inspect_passes_lines(tmp_path):
    (tmp_path / "many.txt").write_text("\n".join(f"L{i}" for i in range(30)))
    out = await tool_file_system(operation="inspect", sandbox_dir=tmp_path,
                                 path="many.txt", lines=5)
    assert "L4" in out and "L5" not in out


# ------------------------------------------------------------------ download

@pytest.mark.asyncio
async def test_httpx_tor_retry_rotates_identity_once_per_attempt(tmp_path, monkeypatch):
    """401/403/503 under TOR used to rotate the Tor identity TWICE per
    attempt (once inside the hop loop, once after it) and sleep 10s instead
    of 5s. Now exactly one rotation per failed attempt."""
    from ghost_agent.tools import file_system as fs

    class _Resp:
        status_code = 503
        headers = {}

    class _StreamCM:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            return _StreamCM()

    monkeypatch.setattr(fs, "curl_requests", None)
    monkeypatch.setattr(fs.httpx, "AsyncClient", _Client)
    rotations = []
    monkeypatch.setattr(fs, "request_new_tor_identity",
                        lambda: rotations.append(1))

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(fs.asyncio, "sleep", _no_sleep)

    out = await fs.tool_download_file(
        "http://example.com/x.bin", tmp_path,
        "socks5://127.0.0.1:9050", "x.bin")
    assert out.startswith("Error")
    assert len(rotations) == 3  # one per attempt — was 6 with the double fire
