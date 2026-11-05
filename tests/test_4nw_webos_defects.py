"""§4NW (2026-10-12): the webOS turns 6b3e22a6 / 4304bd63 / 92dfdd7d.
Every test names the world it fails in. (The browser click-effect pins
live in test_4ln_browser.py next to the other interact pins.)"""
import errno
from types import SimpleNamespace

import pytest

from ghost_agent.sandbox import services as S


# ── http.server names its port after its options ─────────────────────────── #

@pytest.mark.parametrize("cmd,port", [
    ("python3 -m http.server --directory webos 8100", 8100),
    ("python3 -m http.server --directory=webos 8101", 8101),
    ("cd x && python3 -m http.server --bind 0.0.0.0 8102", 8102),
    ("python3 -m http.server 8100 -d webos", 8100),
    ("python3 -m http.server -d webos", None),
    ('python3 -c "import http.server"', None),
])
def test_the_port_after_the_options_is_the_port(cmd, port):
    assert S.extract_command_port(cmd) == port


def test_a_command_that_names_its_port_gets_no_second_one():
    """Fails where `--directory webos 8100` became `8100 --directory webos
    8100` and http.server refused it ("unrecognized arguments")."""
    cmd = "python3 -m http.server --directory webos 8100"
    assert S.add_default_port(cmd, 8100) == (cmd, None)
    new, note = S.add_default_port("python3 -m http.server -d webos", 8100)
    assert new == "python3 -m http.server 8100 -d webos" and note


# ── a start over a running service says which step runs the NEW command ─── #

def test_a_different_command_is_told_to_stop_then_start():
    """Fails where it said "use restart to replace it" — restart relaunches
    the STORED command (the old root server: a directory listing)."""
    e = {"pid": 42030, "port": 8100, "command": "python3 -m http.server 8100"}
    msg = S.already_running_message("webos", e, "python3 -m http.server --directory webos 8100")
    assert "DIFFERENT command" in msg and "action='stop'" in msg and "action='start'" in msg
    assert "would relaunch THAT stored command" in msg


@pytest.mark.parametrize("cmd,workdir", [
    ("python3 -m http.server 8100", None),
    ("python3 -m http.server", None),                          # the start added the port
    ("cd webos && python3 -m http.server 8100", "/workspace/webos"),   # the start dropped this cd
])
def test_the_same_command_is_not_called_different(cmd, workdir):
    e = {"pid": 1, "port": 8100, "command": "python3 -m http.server 8100", "workdir": workdir}
    assert "same command" in S.already_running_message("webos", e, cmd)


def test_a_cd_into_another_directory_is_a_different_command():
    e = {"pid": 1, "port": 8100, "command": "python3 -m http.server 8100", "workdir": "/workspace"}
    assert "DIFFERENT" in S.already_running_message("webos", e, "cd webos && python3 -m http.server 8100")


# ── the state block never names a project that does not exist ────────────── #

from tests.focus_helpers import scope_reset  # noqa: F401,E402  (§4SX B0: no scope leaks between tests)


def _ctx(tmp_path, pid_in_store=True):
    """§4SX B0: the focus is a project-store row now (no scratchpad sentinel)."""
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools import projects as tp
    store = ProjectStore(tmp_path / "mem")
    pid = store.create_project("kept")
    store.set_focus("session:conv-1", pid)
    if not pid_in_store:
        with store._connect() as conn:          # gone behind the store's back
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
            conn.commit()
    ctx = SimpleNamespace(project_store=store, scratchpad=Scratchpad(), current_project_id=None,
                          workspace_model=None, conversation_key="")
    return ctx, pid, tp


def _start(ctx, key="session:conv-1"):
    from tests.focus_helpers import start_request
    return start_request(ctx, key)


def test_a_binding_to_a_deleted_project_is_dropped(tmp_path):
    """Fails where a pointer to a deleted project survived every restart and
    the model read it as its current project (manage_projects: no project
    matches 'e6f85dfb94f8')."""
    ctx, pid, tp = _ctx(tmp_path, pid_in_store=False)
    _start(ctx)
    assert ctx.current_project_id is None
    assert ctx.project_store.focus_rows() == []


def test_a_binding_to_a_live_project_is_kept(tmp_path):
    ctx, pid, tp = _ctx(tmp_path)
    _start(ctx)
    assert ctx.current_project_id == pid
    assert ctx.project_store.get_focus("session:conv-1") == pid


def test_a_store_that_cannot_answer_drops_nothing(tmp_path):
    ctx, pid, tp = _ctx(tmp_path)
    real = ctx.project_store

    class _Broken:
        def get_focus(self, k):
            raise RuntimeError("locked")
    ctx.project_store = _Broken()
    _start(ctx)                                  # never raises; this request is unfocused
    assert ctx.current_project_id is None
    assert real.get_focus("session:conv-1") == pid


def test_deleting_a_project_clears_a_persisted_binding(tmp_path):
    """Fails where delete cleared only the LIVE pointer — after a restart it
    was None, so the persisted binding outlived the project."""
    ctx, pid, tp = _ctx(tmp_path)
    tp.forget_project_pointer(ctx, pid)
    assert ctx.project_store.focus_rows() == []
    other, pid2, _ = _ctx(tmp_path / "o")
    tp.forget_project_pointer(other, "some-other-id")         # not the focused one
    assert other.project_store.get_focus("session:conv-1") == pid2


def test_the_runtime_keys_are_not_shown_to_the_model():
    from ghost_agent.memory.scratchpad import Scratchpad
    sp = Scratchpad()
    for k, v in (("__current_project__", "e6f85dfb94f8"), ("_session_sandbox_writes", "4"),
                 ("_promotion_suggested", "1"), ("todo", "build the calculator")):
        sp.set(k, v, namespace=None)
    shown = sp.list_all(hide_internal=True)
    assert shown == "todo: build the calculator"
    assert "__current_project__" in sp.list_all()        # the full view is unchanged


# ── a screenshot can always overwrite its own earlier file ───────────────── #

def test_a_screenshot_replaces_the_file_and_retries_a_busy_rename(tmp_path, monkeypatch):
    """Fails where re-writing a PNG a few seconds after it was created raised
    BlockingIOError (EAGAIN) through the bind mount and the action failed."""
    from ghost_agent.tools import browser_runner as R
    out = tmp_path / "shots" / "calc.png"
    R._write_replacing(str(out), b"one")
    real, calls = R.os.replace, []

    def _flaky(a, b):
        calls.append(a)
        if len(calls) == 1:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return real(a, b)
    monkeypatch.setattr(R.os, "replace", _flaky)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    R._write_replacing(str(out), b"two")
    assert out.read_bytes() == b"two" and len(calls) == 2
    assert sorted(p.name for p in out.parent.iterdir()) == ["calc.png"]   # no temp left


def test_a_real_write_error_is_not_retried(tmp_path, monkeypatch):
    from ghost_agent.tools import browser_runner as R
    calls = []

    def _denied(a, b):
        calls.append(a)
        raise PermissionError(errno.EACCES, "denied")
    monkeypatch.setattr(R.os, "replace", _denied)
    with pytest.raises(PermissionError):
        R._write_replacing(str(tmp_path / "x.png"), b"x")
    assert len(calls) == 1 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path,kind", [("/workspace/a.png", "png"), ("/workspace/a.JPG", "jpeg"),
                                       ("/workspace/a.jpeg", "jpeg"), ("/workspace/a", "png")])
def test_the_screenshot_type_follows_the_extension(path, kind):
    """Fails where the bytes form saved PNG bytes under a .jpg name (the
    path= form inferred the type)."""
    from ghost_agent.tools import browser_runner as R
    assert R._screenshot_type(path) == kind


def test_an_unsupported_screenshot_extension_is_refused():
    from ghost_agent.tools import browser_runner as R
    with pytest.raises(ValueError, match=r"\.png, \.jpg"):
        R._screenshot_type("/workspace/a.webp")


@pytest.mark.parametrize("new", ["python3 -m http.server 9000", "python3 -m http.server 8100"])
def test_a_different_port_is_a_different_command_and_the_message_names_the_live_port(new):
    """Fails where a start on another port was called "the same command —
    nothing to do" (review: the model then told the user the app was on
    9000 while it ran on 8101)."""
    e = {"pid": 1, "port": 8101, "command": "python3 -m http.server 8101"}
    msg = S.already_running_message("webos", e, new)
    assert "DIFFERENT" in msg and "(port 8101)" in msg
    same = S.already_running_message("webos", e, "python3 -m http.server 8101")
    assert "same command on port 8101" in same


def test_a_two_digit_argument_is_not_a_port():
    assert S.extract_command_port("python3 -m http.server --directory site 80") is None


def test_the_screenshot_is_taken_in_the_type_its_name_says(tmp_path):
    """Fails where the bytes form ignored the extension (PNG bytes in a .jpg)."""
    import asyncio
    from ghost_agent.tools import browser_runner as R
    seen = {}

    class _Page:
        async def screenshot(self, full_page=False, type=None):
            seen.update(full_page=full_page, type=type)
            return b"\xff\xd8jpeg"
    asyncio.run(R._save_screenshot(_Page(), str(tmp_path / "shot.jpg"), True))
    assert seen == {"full_page": True, "type": "jpeg"}
    assert (tmp_path / "shot.jpg").read_bytes() == b"\xff\xd8jpeg"


# ── the wiring: the real delete paths, the render, the tool's list ───────── #

async def test_the_tool_delete_clears_a_persisted_binding(tmp_path):
    """Fails where `manage_projects delete` cleared only the live pointer —
    after a restart (live pointer None) the focus row outlived the project."""
    import json as _json
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools import projects as tp
    ctx = SimpleNamespace(project_store=ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb"),
                          scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"), graph_memory=None,
                          current_project_id=None, request_start_project_id=None,
                          last_user_content="", conversation_key="conv-1")
    pid = _json.loads(await tp.tool_manage_projects(ctx, action="create", title="probe", kind="CODING",
                                                    goal="x"))["created"]
    ctx.project_store.set_focus("session:other", pid)
    ctx.current_project_id = None                       # as after a restart
    await tp.tool_manage_projects(ctx, action="delete", project_id=pid)
    assert ctx.project_store.focus_rows() == []


def test_the_api_delete_clears_a_persisted_binding(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api.projects_routes import projects_router
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    sp = Scratchpad(persist_path=tmp_path / "sp.db")
    ctx = SimpleNamespace(args=SimpleNamespace(api_key=""), project_store=store, scratchpad=sp,
                          graph_memory=None, current_project_id=None)
    app = FastAPI()
    app.state.agent = SimpleNamespace(context=ctx)
    app.include_router(projects_router)
    tc = TestClient(app)
    pid = tc.post("/api/projects", json={"title": "A"}).json()["id"]
    store.set_focus("session:x", pid)
    assert tc.delete(f"/api/projects/{pid}").status_code in (200, 204)     # soft (archive)
    assert store.focus_rows() == []
    pid2 = tc.post("/api/projects", json={"title": "B"}).json()["id"]
    store.set_focus("session:x", pid2)
    assert tc.delete(f"/api/projects/{pid2}?hard=true").status_code in (200, 204)
    assert store.focus_rows() == []


async def test_the_scratchpad_tool_list_hides_the_runtime_keys():
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools.memory import tool_scratchpad
    sp = Scratchpad()
    sp.set("__current_project__", "e6f85dfb94f8", namespace=None)
    sp.set("todo", "x", namespace=None)
    out = await tool_scratchpad(action="list", scratchpad=sp)
    assert "todo" in str(out) and "__current_project__" not in str(out)


def test_the_state_block_render_hides_the_runtime_keys():
    """Enumerated over the parsed source: the one scratchpad render in the
    dynamic state passes hide_internal=True (it sits deep inside the turn)."""
    import ast
    from pathlib import Path
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "list_all"]
    assert calls and all(any(k.arg == "hide_internal" and getattr(k.value, "value", None) is True
                             for k in c.keywords) for c in calls)
