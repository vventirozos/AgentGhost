"""Race-free upload/download project scoping (PROJECT_JOURNAL §4B, finding #3).

`current_project_id` is a process-global read by the stateless /api/upload and
/api/download endpoints, which carry no conversation context — so a concurrent
conversation's switch/reconcile can point it at the wrong project, landing an
upload in another conversation's sandbox. #22 (turn serialization) closed the
chat-turn-vs-chat-turn window but NOT the upload/download-vs-chat window. The fix
lets a client pass ?project_id=<id> for a race-free scope; the global stays the
fallback when absent.
"""
from types import SimpleNamespace

import pytest

from ghost_agent.tools.file_system import project_scoped_sandbox


def _ctx(tmp_path, global_pid=None):
    return SimpleNamespace(sandbox_dir=str(tmp_path), current_project_id=global_pid)


def test_explicit_project_id_overrides_global(tmp_path):
    # Global points at project B, but the caller explicitly wants project A.
    ctx = _ctx(tmp_path, global_pid="proj-b")
    host, workdir = project_scoped_sandbox(ctx, explicit_project_id="proj-a")
    assert host.name == "proj-a"
    assert host.parent.name == "projects"
    assert workdir.endswith("/projects/proj-a")


def test_absent_explicit_falls_back_to_global(tmp_path):
    ctx = _ctx(tmp_path, global_pid="proj-b")
    host, _ = project_scoped_sandbox(ctx)  # no explicit id → global
    assert host.name == "proj-b"


def test_explicit_id_normalized(tmp_path):
    ctx = _ctx(tmp_path, global_pid=None)
    host, _ = project_scoped_sandbox(ctx, explicit_project_id="  ProjXYZ  ")
    assert host.name == "projxyz"


def test_empty_explicit_id_scopes_to_root(tmp_path):
    # A client explicitly asking for "" (no project) gets root, and the
    # conversation-binding fallback is NOT applied (explicit intent).
    ctx = _ctx(tmp_path, global_pid=None)
    host, workdir = project_scoped_sandbox(ctx, explicit_project_id="")
    assert host == tmp_path and workdir is None


# ------------------------------------------------- endpoint wiring


fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def _app(tmp_path):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = MagicMock()
    agent.context.args.api_key = "test-key"
    agent.context.sandbox_dir = tmp_path  # download route uses it as a Path
    agent.context.current_project_id = "wrong-project"  # the racy global
    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    return TestClient(app), agent


def test_upload_uses_explicit_project_id(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    client, _ = _app(sb)
    r = client.post("/api/upload?project_id=right-project",
                    headers={"X-Ghost-Key": "test-key"},
                    files={"file": ("note.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    # Landed in the EXPLICIT project, not the racy global "wrong-project".
    assert (sb / "projects" / "right-project" / "note.txt").exists()
    assert not (sb / "projects" / "wrong-project" / "note.txt").exists()


def test_download_bare_path_uses_explicit_project_id(tmp_path):
    sb = tmp_path / "sandbox"
    (sb / "projects" / "right-project").mkdir(parents=True)
    (sb / "projects" / "right-project" / "plot.png").write_bytes(b"PNGDATA")
    client, _ = _app(sb)
    r = client.get("/api/download/plot.png?project_id=right-project",
                   headers={"X-Ghost-Key": "test-key"})
    assert r.status_code == 200
    assert r.content == b"PNGDATA"
