"""§4KW fourth review: the one-off sets the immutable flag on workspaces of
projects already released."""
import os
import runpy
import stat
import sys
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "release_immutable_4kw.py")
_IMM = getattr(stat, "UF_IMMUTABLE", 0) if hasattr(os, "chflags") else 0


@pytest.mark.skipif(not _IMM, reason="BSD file flags only")
def test_apply_flags_every_released_workspace(tmp_path, monkeypatch):
    from ghost_agent.memory import projects as P
    ws = tmp_path / "sandbox" / "projects" / "aaaaaaaaaaaa"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("x")
    (ws / "data.db").write_bytes(b"")
    monkeypatch.setattr(P.ProjectStore, "list_projects",
                        lambda self, status=None: [{"id": "aaaaaaaaaaaa", "workspace_dir": str(ws)}])
    monkeypatch.setattr(P.ProjectStore, "get_project",
                        lambda self, pid: {"id": pid, "workspace_dir": str(ws), "status": "RELEASED"})
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_AGENT_PORT", "1")          # not the live agent
    monkeypatch.setattr(sys, "argv", ["x", "--apply"])
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
        assert all(os.lstat(e).st_flags & _IMM for e in [ws / "src", ws / "src" / "app.py"])
        # the app's runtime state and its folder stay writable (fifth review)
        assert not any(os.lstat(e).st_flags & _IMM for e in [ws, ws / "data.db"])
    finally:
        P.ProjectStore._chmod_tree(ws, False)


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    from ghost_agent.memory import projects as P
    ws = tmp_path / "sandbox" / "projects" / "aaaaaaaaaaaa"
    ws.mkdir(parents=True)
    (ws / "app.py").write_text("x")
    monkeypatch.setattr(P.ProjectStore, "list_projects",
                        lambda self, status=None: [{"id": "aaaaaaaaaaaa", "workspace_dir": str(ws)}])
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x"])
    runpy.run_path(SCRIPT, run_name="__main__")
    assert "0/2 entries immutable" in capsys.readouterr().out
    assert not (os.lstat(ws / "app.py").st_flags & _IMM if _IMM else 0)


def test_an_apply_refuses_while_the_agent_listens(tmp_path, monkeypatch):
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        monkeypatch.setenv("GHOST_AGENT_PORT", str(srv.getsockname()[1]))
        monkeypatch.setattr(sys, "argv", ["x", "--apply"])
        with pytest.raises(SystemExit, match="agent is running"):
            runpy.run_path(SCRIPT, run_name="__main__")
    finally:
        srv.close()
