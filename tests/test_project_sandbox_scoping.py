"""Project-scoped sandbox working directory.

When a project is active, file ops (file_system, report_pdf) and code
execution (execute) are rooted at ``<sandbox>/projects/<id>/`` instead of
the sandbox root, so a whole project's scratch space cleans up with one
``rm -rf sandbox/projects/<id>``. These tests pin:

  * the path heal that avoids double-nesting the model's ``projects/<id>/``
    paths, while leaving the unscoped root untouched;
  * the container-path translation (bind mount is at the ROOT, so a scoped
    file is reported at ``/workspace/projects/<id>/...``);
  * the write-then-read symmetry between file_system and execute (the
    invariant with multiple prior post-mortems);
  * report_pdf emitting a ROOT-relative download link so the PDF is
    reachable through ``/api/download/<path>``;
  * the registry wiring that activates scoping ONLY for a real string
    project id (never a MagicMock auto-vivified attribute).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path,
    _to_container_path,
    project_scoped_sandbox,
    tool_file_system,
)
from ghost_agent.tools.report_pdf import tool_generate_pdf
from ghost_agent.tools.registry import get_available_tools

PID = "abc123def456"


def _scoped(tmp_path):
    d = tmp_path / "projects" / PID
    d.mkdir(parents=True)
    return d


# --------------------------------------------- shared scoping helper

def test_project_scoped_sandbox_helper(tmp_path):
    from types import SimpleNamespace
    # no project → root, no workdir
    ctx = SimpleNamespace(sandbox_dir=tmp_path, current_project_id=None)
    assert project_scoped_sandbox(ctx) == (tmp_path, None)
    # active project → scoped dir (created) + container workdir, id lowercased
    ctx.current_project_id = PID.upper()
    hd, wd = project_scoped_sandbox(ctx)
    assert hd == tmp_path / "projects" / PID and hd.exists()
    assert wd == f"/workspace/projects/{PID}"
    # stateful opts out
    assert project_scoped_sandbox(ctx, stateful=True) == (tmp_path, None)


def test_project_scoped_sandbox_ignores_magicmock_id(tmp_path):
    m = MagicMock()
    m.sandbox_dir = tmp_path
    assert project_scoped_sandbox(m)[0] == tmp_path  # truthy Mock id → unscoped


def test_project_scoped_sandbox_none_safe():
    from types import SimpleNamespace
    # A context without a sandbox_dir (some unit-test contexts) must not crash.
    ctx = SimpleNamespace(sandbox_dir=None, current_project_id="abc")
    assert project_scoped_sandbox(ctx) == (None, None)


# --------------------------------------------------------------- path heal

def test_bare_name_lands_in_project_dir(tmp_path):
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, "X.md") == (sb / "X.md").resolve()


def test_redundant_project_prefix_not_double_nested(tmp_path):
    sb = _scoped(tmp_path)
    # The model echoes the project-relative path it saw in a listing.
    assert _get_safe_path(sb, f"projects/{PID}/X.md") == (sb / "X.md").resolve()


def test_redundant_prefix_case_insensitive(tmp_path):
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, f"projects/{PID.upper()}/X.md") == (sb / "X.md").resolve()


def test_workspace_project_prefix_collapses(tmp_path):
    sb = _scoped(tmp_path)
    assert _get_safe_path(sb, f"/workspace/projects/{PID}/X.md") == (sb / "X.md").resolve()


def test_unscoped_root_keeps_projects_prefix(tmp_path):
    # The normal (no-project) root must be untouched: a literal
    # "projects/abc/X" is honoured verbatim, not stripped.
    root = tmp_path
    assert _get_safe_path(root, "projects/abc/X.md") == (root / "projects" / "abc" / "X.md").resolve()


# ------------------------------------------------------- container path

def test_container_path_is_root_relative_when_scoped(tmp_path):
    sb = _scoped(tmp_path)
    host = sb / "run.py"
    # Bind mount is at the ROOT, so the scoped file is /workspace/projects/<id>/run.py
    assert _to_container_path(sb, host) == f"/workspace/projects/{PID}/run.py"


def test_container_path_unscoped(tmp_path):
    assert _to_container_path(tmp_path, tmp_path / "run.py") == "/workspace/run.py"


# ------------------------------------------- write-then-read symmetry

@pytest.mark.asyncio
async def test_file_system_write_then_read_symmetry(tmp_path):
    """A file written by file_system at a bare name, then named the same
    way, reads back — and physically lives under projects/<id>/."""
    sb = _scoped(tmp_path)
    await tool_file_system(operation="write", path="notes.md", content="hello",
                           sandbox_dir=sb)
    # physically under the project dir
    assert (sb / "notes.md").read_text() == "hello"
    res = await tool_file_system(operation="read", path="notes.md", sandbox_dir=sb)
    assert "hello" in str(res)
    # and the same file is addressable by the redundant project-prefixed path
    res2 = await tool_file_system(operation="read", path=f"projects/{PID}/notes.md",
                                  sandbox_dir=sb)
    assert "hello" in str(res2)


# --------------------------------------------------- report_pdf link

@pytest.mark.asyncio
async def test_report_pdf_download_link_is_root_relative(tmp_path):
    sb = _scoped(tmp_path)
    out = await tool_generate_pdf(
        title="Scoped Report",
        sections=[{"heading": "A", "body": "body text"}],
        sandbox_dir=sb,
    )
    assert out.startswith("SUCCESS")
    # link carries the projects/<id>/ prefix so /api/download (rooted at the
    # sandbox ROOT) can resolve it
    assert f"/api/download/projects/{PID}/report_" in out
    # the file physically lives in the scoped dir
    assert list(sb.glob("report_*.pdf"))


@pytest.mark.asyncio
async def test_report_pdf_unscoped_link_has_no_prefix(tmp_path):
    out = await tool_generate_pdf(
        title="Root Report",
        sections=[{"heading": "A", "body": "b"}],
        sandbox_dir=tmp_path,
    )
    assert out.startswith("SUCCESS")
    assert "/api/download/report_" in out
    assert "/api/download/projects/" not in out


# ------------------------------------------------- registry wiring

def _ctx(tmp_path):
    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.max_context = 32768
    return ctx


def test_registry_scopes_file_system_when_project_active(tmp_path, monkeypatch):
    import ghost_agent.tools.registry as reg
    captured = {}

    def fake_fs(**kwargs):
        captured["sandbox_dir"] = kwargs.get("sandbox_dir")
        return "ok"

    monkeypatch.setattr(reg, "tool_file_system", fake_fs)
    ctx = _ctx(tmp_path)
    ctx.current_project_id = PID
    tools = get_available_tools(ctx)
    tools["file_system"](action="read", filename="x")
    assert captured["sandbox_dir"] == tmp_path / "projects" / PID
    assert captured["sandbox_dir"].exists()  # mkdir'd on access


def test_registry_scopes_knowledge_base(tmp_path, monkeypatch):
    """knowledge_base's gain_knowledge reads SOURCE files from the sandbox, so
    it must read the project dir to ingest a doc file_system just wrote."""
    import ghost_agent.tools.registry as reg
    cap = {}
    monkeypatch.setattr(reg, "tool_knowledge_base",
                        lambda **kw: cap.update(sandbox_dir=kw.get("sandbox_dir")))
    ctx = _ctx(tmp_path)
    ctx.current_project_id = PID
    tools = get_available_tools(ctx)
    tools["knowledge_base"](action="gain_knowledge", filename="notes.md")
    assert cap["sandbox_dir"] == tmp_path / "projects" / PID


@pytest.mark.asyncio
async def test_upload_route_lands_in_project_dir(tmp_path):
    """A file uploaded while a project is active lands in projects/<id>/ so it
    joins the working set the (scoped) listing shows and file_system reads."""
    from ghost_agent.api.routes import upload_file

    class _UF:
        filename = "data.csv"
        async def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"a,b,c\n1,2,3\n"

    req = _fake_request(tmp_path, PID)
    res = await upload_file(req, _UF())
    assert res["status"] == "success"
    assert (tmp_path / "projects" / PID / "data.csv").exists()
    assert not (tmp_path / "data.csv").exists()  # not at root


@pytest.mark.asyncio
async def test_upload_route_root_when_no_project(tmp_path):
    from ghost_agent.api.routes import upload_file

    class _UF:
        filename = "data.csv"
        async def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"x"

    req = _fake_request(tmp_path, None)
    await upload_file(req, _UF())
    assert (tmp_path / "data.csv").exists()  # root, no project


def test_registry_unscoped_when_no_project(tmp_path, monkeypatch):
    import ghost_agent.tools.registry as reg
    captured = {}
    monkeypatch.setattr(reg, "tool_file_system",
                        lambda **kw: captured.update(sandbox_dir=kw.get("sandbox_dir")))
    ctx = _ctx(tmp_path)
    ctx.current_project_id = None
    tools = get_available_tools(ctx)
    tools["file_system"](action="read", filename="x")
    assert captured["sandbox_dir"] == tmp_path  # root, not scoped


def test_registry_ignores_magicmock_project_id(tmp_path, monkeypatch):
    """A MagicMock context auto-vivifies current_project_id to a truthy
    Mock; scoping must NOT activate off that garbage."""
    import ghost_agent.tools.registry as reg
    captured = {}
    monkeypatch.setattr(reg, "tool_file_system",
                        lambda **kw: captured.update(sandbox_dir=kw.get("sandbox_dir")))
    ctx = _ctx(tmp_path)  # current_project_id left as auto-vivified Mock
    tools = get_available_tools(ctx)
    tools["file_system"](action="read", filename="x")
    assert captured["sandbox_dir"] == tmp_path  # treated as no project


@pytest.mark.asyncio
async def test_registry_execute_threads_container_workdir(tmp_path, monkeypatch):
    import ghost_agent.tools.registry as reg
    captured = {}

    async def fake_exec(**kwargs):
        captured.update(sandbox_dir=kwargs.get("sandbox_dir"),
                        container_workdir=kwargs.get("container_workdir"))
        return "ok"

    monkeypatch.setattr(reg, "tool_execute", fake_exec)
    ctx = _ctx(tmp_path)
    ctx.current_project_id = PID
    tools = get_available_tools(ctx)
    await tools["execute"](command="echo hi")
    assert captured["sandbox_dir"] == tmp_path / "projects" / PID
    assert captured["container_workdir"] == f"/workspace/projects/{PID}"


def test_registry_scopes_vision_and_image_gen(tmp_path, monkeypatch):
    """The execute→vision plot-analysis loop (and image_generation, its
    documented pair) must read/write the SAME scoped dir as file_system,
    or a plot written by the scoped execute is invisible to vision."""
    import ghost_agent.tools.vision as vision_mod
    import ghost_agent.tools.image_gen as imggen_mod
    cap = {}
    # vision_analysis / image_generation are lazy-imported inside
    # get_available_tools, so patch them at the source module.
    monkeypatch.setattr(vision_mod, "tool_vision_analysis",
                        lambda **kw: cap.update(vision=kw.get("sandbox_dir")))
    monkeypatch.setattr(imggen_mod, "tool_generate_image",
                        lambda **kw: cap.update(img=kw.get("sandbox_dir")))
    ctx = _ctx(tmp_path)
    ctx.current_project_id = PID
    ctx.llm_client = MagicMock()
    tools = get_available_tools(ctx)
    tools["vision_analysis"](action="describe", target="x.png")
    if "image_generation" in tools:
        tools["image_generation"](prompt="a cat")
    scoped = tmp_path / "projects" / PID
    assert cap["vision"] == scoped
    if "img" in cap:
        assert cap["img"] == scoped


@pytest.mark.asyncio
async def test_vision_falls_back_to_root_when_scoped(tmp_path):
    """A read-only vision call, while scoped to a project, still finds an
    image a root-only tool (e.g. browser screenshot) wrote at the sandbox
    root — so the browser→vision handoff isn't broken by scoping."""
    from unittest.mock import AsyncMock
    import ghost_agent.tools.vision as vision_mod
    sb = _scoped(tmp_path)
    (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # at ROOT, not scoped
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "a screenshot"}}]})
    out = await vision_mod.tool_vision_analysis(
        action="describe", target="shot.png", llm_client=llm, sandbox_dir=sb)
    assert "not found" not in out.lower()  # found via root fallback
    assert "screenshot" in out

class _RecordingMgr:
    """Returns file-not-found for the scoped run, success for the root retry."""
    def __init__(self, scoped_fails=True):
        self.calls = []
        self.scoped_fails = scoped_fails

    def execute(self, cmd, timeout=300, **kw):
        self.calls.append(kw.get("workdir"))
        if kw.get("workdir") and self.scoped_fails:
            return ("python3: can't open file 'chart.py': [Errno 2] No such file or directory", 1)
        return ("ran ok", 0)


@pytest.mark.asyncio
async def test_execute_command_retries_at_root_on_not_found(tmp_path):
    """A bare-file command that fails in the scoped cwd (file was written at
    root before the project switch took effect) retries from the root."""
    from ghost_agent.tools.execute import tool_execute
    mgr = _RecordingMgr(scoped_fails=True)
    out = await tool_execute(command="python3 chart.py", sandbox_manager=mgr,
                             sandbox_dir=tmp_path / "projects" / PID,
                             container_workdir=f"/workspace/projects/{PID}")
    assert "ran ok" in out
    assert mgr.calls == [f"/workspace/projects/{PID}", None]  # scoped, then root retry


@pytest.mark.asyncio
async def test_execute_command_no_retry_when_scoped_run_succeeds(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    mgr = _RecordingMgr(scoped_fails=False)
    out = await tool_execute(command="python3 chart.py", sandbox_manager=mgr,
                             sandbox_dir=tmp_path / "projects" / PID,
                             container_workdir=f"/workspace/projects/{PID}")
    assert "ran ok" in out
    assert mgr.calls == [f"/workspace/projects/{PID}"]  # single run, no retry


@pytest.mark.asyncio
async def test_execute_command_no_retry_when_unscoped(tmp_path):
    """No project active → no scoped/root distinction → no retry."""
    from ghost_agent.tools.execute import tool_execute
    mgr = _RecordingMgr(scoped_fails=True)
    out = await tool_execute(command="python3 chart.py", sandbox_manager=mgr,
                             sandbox_dir=tmp_path, container_workdir=None)
    assert mgr.calls == [None]  # one run at root, no retry loop


def _fake_request(sandbox_dir, project_id):
    from types import SimpleNamespace
    agent = SimpleNamespace(context=SimpleNamespace(
        sandbox_dir=Path(sandbox_dir), current_project_id=project_id))
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent=agent)))


@pytest.mark.asyncio
async def test_download_route_falls_back_to_active_project_dir(tmp_path):
    """A model-emitted bare link (/api/download/plot.png) to a file that the
    scoped execute/image_generation wrote into projects/<id>/ still resolves
    while that project is active."""
    from ghost_agent.api.routes import download_file
    (tmp_path / "projects" / PID).mkdir(parents=True)
    (tmp_path / "projects" / PID / "plot.png").write_bytes(b"\x89PNG\r\n")
    req = _fake_request(tmp_path, PID)
    resp = await download_file(req, "plot.png")
    assert Path(resp.path) == (tmp_path / "projects" / PID / "plot.png").resolve()


@pytest.mark.asyncio
async def test_download_route_no_fallback_without_project(tmp_path):
    from fastapi import HTTPException
    from ghost_agent.api.routes import download_file
    (tmp_path / "projects" / PID).mkdir(parents=True)
    (tmp_path / "projects" / PID / "plot.png").write_bytes(b"\x89PNG")
    req = _fake_request(tmp_path, None)  # no active project
    with pytest.raises(HTTPException):
        await download_file(req, "plot.png")  # not at root, no fallback → 404


@pytest.mark.asyncio
async def test_download_route_root_file_still_served(tmp_path):
    from ghost_agent.api.routes import download_file
    (tmp_path / "at_root.txt").write_text("hi")
    req = _fake_request(tmp_path, PID)  # project active, but file is at root
    resp = await download_file(req, "at_root.txt")
    assert Path(resp.path) == (tmp_path / "at_root.txt").resolve()


@pytest.mark.asyncio
async def test_registry_execute_stateful_opts_out_of_scoping(tmp_path, monkeypatch):
    """Stateful kernel sessions stay at root (kernel conn file pinned to
    /workspace), so no project scoping for them."""
    import ghost_agent.tools.registry as reg
    captured = {}

    async def fake_exec(**kwargs):
        captured.update(sandbox_dir=kwargs.get("sandbox_dir"),
                        container_workdir=kwargs.get("container_workdir"))
        return "ok"

    monkeypatch.setattr(reg, "tool_execute", fake_exec)
    ctx = _ctx(tmp_path)
    ctx.current_project_id = PID
    tools = get_available_tools(ctx)
    await tools["execute"](filename="x.py", content="print(1)", stateful=True)
    assert captured["sandbox_dir"] == tmp_path  # root
    assert captured["container_workdir"] is None
