"""§4UU — coding mechanics: page check after a web write, the checklist tool, "a run is a run", lab/browser/EPERM
hints, the web-app starter project."""
import asyncio
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from ghost_agent.core import code_claims as C
from ghost_agent.core import env_limits as E
from ghost_agent.core import page_check as P
from ghost_agent.tools import checklist as K
from ghost_agent.tools import scaffold as S

REPO = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


# ── #3: a syntax check / a read is not a run ──────────────────────────

def _w(path):
    return {"name": "file_system", "arguments": {"operation": "write", "path": path},
            "content": f"SUCCESS: Wrote 10 chars to '{path}'. Script-side path (from sandbox cwd): '{path}'."}


def _x(cmd, code=0):
    return {"name": "execute", "arguments": {"command": cmd}, "content": f"EXIT CODE: {code}\n--- STDOUT ---\nok"}


@pytest.mark.parametrize("cmd,kinds", [
    ("node --check a.js", ["check"]),
    ("python3 -m py_compile app.py", ["check"]),
    ("cat app.py | grep foo", ["read", "read"]),
    ("curl -s localhost:8000/app.py", ["read"]),
    ("cd x && python3 app.py", ["read", "run"]),
    ("python3 -m pytest -q", ["run"]),
    ("npm test", ["run"]),
])
def test_segment_kinds(cmd, kinds):
    assert [C.segment_kind(w) for w in C._shell_segments(cmd)] == kinds


@pytest.mark.parametrize("cmd,key", [("node server.js -c config.json", "server.js"), ("bash run.sh -n 5", "run.sh"),
                                     ("cat app.py | python3", "app.py"), ("git bisect run python3 app.py", "app.py"),
                                     ("ruby app.rb -c 3", "app.rb")])
def test_flags_after_the_script_and_stdin_runs_still_count(cmd, key):
    assert C.unbacked_code_writes([_w(key), _x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd", ["node --check app.js", "cat app.js", "grep -n init app.js", "ls -la app.js",
                                 "curl -s http://127.0.0.1:8000/app.js"])
def test_a_check_or_read_does_not_back_a_works_claim(cmd):
    never, failed = C.unbacked_code_writes([_w("app.js"), _x(cmd)])
    assert never == ["app.js"] and failed == []


def test_a_real_run_backs_it_and_a_heredoc_run_counts():
    assert C.unbacked_code_writes([_w("app.py"), _x("python3 app.py")]) == ([], [])
    heredoc = "cat > note.txt <<EOF\nhi\nEOF\npython3 app.py"
    assert C.unbacked_code_writes([_w("app.py"), _x(heredoc)]) == ([], [])


def test_the_never_run_wording_names_the_syntax_check():
    note, failed = C.unbacked_claim_note("I tested it and it works.", [_w("app.js"), _x("node --check app.js")])
    assert failed and "syntax" in note.lower()


# ── #1: the page check note, read back by the claim arm ───────────────

def test_page_check_note_round_trips_into_the_claim_arm():
    note = P.note_for(("web/index.html", ""), True, {"index.html", "app.js"})
    assert note.strip().startswith("[PAGE CHECK] ✓")
    row = _w("web/app.js")
    row["content"] += note
    assert C.unbacked_code_writes([row]) == ([], [])
    bad = _w("web/app.js")
    bad["content"] += P.note_for(("web/index.html", "UNCAUGHT JS EXCEPTIONS (1)\n • x is not defined"), True,
                                 {"index.html", "app.js"})
    assert C.unbacked_code_writes([bad]) == ([], ["web/app.js"])


def test_only_the_first_page_check_line_counts():
    block = "UNCAUGHT JS EXCEPTIONS (1)\n[PAGE CHECK] ✓ loaded 'x' headless: no uncaught JS exception. Ran: game.js."
    text = P.note_for(("index.html", block), True, {"index.html", "app.js"})
    assert P.checks_in(text) == [(False, {"index.html", "app.js"})]


def test_page_check_text_inside_file_content_is_not_a_check():
    # an echoed file line carrying the marker mid-line never reads as the system's check
    assert P.checks_in("x = '[PAGE CHECK] ✓ loaded a. Ran: app.js.'") == []


def test_page_check_not_run_note_only_for_html():
    assert P.note_for(None, False) == ""
    assert "not run" in P.note_for(None, True)


def test_loaded_names_follows_script_tags_and_module_imports(tmp_path):
    (tmp_path / "js").mkdir()
    (tmp_path / "index.html").write_text('<script src="js/main.js"></script>'
                                         '<script src="https://cdn.x/lib.js"></script>')
    (tmp_path / "js" / "main.js").write_text("import { a } from './util.js';")
    (tmp_path / "js" / "util.js").write_text("export const a = 1;")
    assert P.loaded_names(tmp_path / "index.html", tmp_path) == {"index.html", "main.js", "util.js"}


def test_page_check_switch(monkeypatch):
    monkeypatch.setenv("GHOST_PAGE_CHECK", "0")
    assert not P.enabled()
    monkeypatch.delenv("GHOST_PAGE_CHECK")
    assert P.enabled()


def _agent_stub(tmp_path, res):
    from ghost_agent.core.agent import GhostAgent
    calls = []

    async def probe(written, project_id=None):
        calls.append(list(written))
        return res

    a = types.SimpleNamespace(available_tools={"browser": object()}, _execute_web_artifact=probe,
                              _scoped_sandbox_for=lambda pid: tmp_path)
    a._page_check_after_write = GhostAgent._page_check_after_write.__get__(a)
    return a, calls


def test_page_check_after_write_probes_the_last_web_write_only(tmp_path):
    (tmp_path / "index.html").write_text('<script src="app.js"></script>')
    (tmp_path / "app.js").write_text("1;")
    a, calls = _agent_stub(tmp_path, ("index.html", ""))
    text = "SUCCESS: Wrote 10 chars to 'app.js'. Script-side path (from sandbox cwd): 'app.js'."
    later = ("file_system", "id2", "h", True, "x", False, "write", "index.html", False, {})
    ok_later = "SUCCESS: Wrote 9 chars to 'index.html'. Script-side path (from sandbox cwd): 'index.html'."
    assert run(a._page_check_after_write(text, 0, [("file_system",), later], [text, ok_later])) == ""
    assert calls == []
    # a later write that FAILED does not take the probe away (review m1)
    note = run(a._page_check_after_write(text, 0, [("file_system",), later], [text, "Error: old_string not found"]))
    assert calls == [["app.js"]] and "Ran: app.js, index.html." in note
    note = run(a._page_check_after_write(text, 1, [("file_system",), later], [text, ok_later]))
    assert calls == [["app.js"], ["app.js"]]


def test_the_page_probe_does_not_move_the_models_browser(tmp_path):
    from ghost_agent.tools import browser_support as BS
    from ghost_agent.utils.logging import request_id_context
    seen = []
    a, _calls = _agent_stub(tmp_path, ("index.html", ""))

    async def probe(written, project_id=None):
        seen.append((BS._last_url_filename(), BS._last_url_conv_filename()))
        return ("index.html", "")
    a._execute_web_artifact = probe
    tok = request_id_context.set("req4uu")
    try:
        run(a._page_check_after_write("SUCCESS: Wrote 1 chars to 'index.html'. Script-side path (from sandbox "
                                      "cwd): 'index.html'.", 0, []))
        assert seen == [(".last_url.sysprobe", "")]
        assert BS._last_url_filename() == ".last_url.req4uu"
    finally:
        request_id_context.reset(tok)


def test_page_check_after_write_skips_open_parts_and_non_web(tmp_path):
    from ghost_agent.tools.file_system import OPEN_PART_NOTE
    a, calls = _agent_stub(tmp_path, ("index.html", ""))
    text = "SUCCESS: Wrote 10 chars to 'index.html'. Script-side path (from sandbox cwd): 'index.html'."
    assert run(a._page_check_after_write(text + "\n" + OPEN_PART_NOTE, 0, [])) == ""
    py = "SUCCESS: Wrote 10 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'."
    assert run(a._page_check_after_write(py, 0, [])) == ""
    assert calls == []


def test_page_check_after_write_never_for_a_member(tmp_path, monkeypatch):
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "requester_is_member", lambda: True)
    a, calls = _agent_stub(tmp_path, ("index.html", ""))
    text = "SUCCESS: Wrote 10 chars to 'index.html'. Script-side path (from sandbox cwd): 'index.html'."
    assert run(a._page_check_after_write(text, 0, [])) == "" and calls == []


# ── #2: the checklist ─────────────────────────────────────────────────

@pytest.fixture
def ctx(tmp_path):
    from ghost_agent.utils.logging import conversation_key_context
    tok = conversation_key_context.set("conv-4uu")
    yield types.SimpleNamespace(memory_dir=str(tmp_path / "sys" / "memory"), current_project_id=None)
    conversation_key_context.reset(tok)


async def _ex(command):
    code = 0 if "pass" in command else 1
    return f"EXIT CODE: {code}\n--- STDOUT ---\nran {command}"


def test_checklist_only_a_passing_check_ticks(ctx, tmp_path):
    items = [{"title": "a", "check": "python3 pass.py"}, {"title": "b", "check": "python3 fail.py"}]
    run(K.tool_checklist(ctx, action="set", items=items))
    assert "0/2" in K.status_line(ctx)
    out = run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    assert "✓ #1 a" in out and "✗ #2 b" in out and "ran python3 fail.py" in out
    assert "1/2" in K.status_line(ctx) and "#2 b ✗" in K.status_line(ctx)
    # the list lives in the HOST system dir, not the sandbox
    assert list((tmp_path / "sys" / "checklists").glob("conv-*.json"))


def test_checklist_set_keeps_state_of_unchanged_items(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"}]))
    run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    out = run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"},
                                                         {"title": "c", "check": "python3 x.py"}]))
    assert "✓ #1 a" in out and "○ #2 c" in out


@pytest.mark.parametrize("check", ["node --check app.js", "ls", "cat a.js", "grep x a.js", "true", "echo ok",
                                   "curl -s localhost:8000"])
def test_checklist_refuses_a_check_that_runs_nothing(ctx, check):
    res = run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": check}]))
    assert "runs nothing" in str(res)
    assert K.load(ctx)["items"] == []


def test_a_title_cannot_plant_a_check(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "Loop — check: python3 game.py",
                                                    "check": "python3 pass.py"}]))
    out = run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    row = {"name": "checklist", "arguments": {"action": "check"}, "content": out}
    assert C.unbacked_code_writes([_w("game.py"), row]) == (["game.py"], [])
    assert "ONE line" in str(run(K.tool_checklist(ctx, action="add", items=[{"title": "x",
                                                                             "check": "python3 a.py\npython3 b.py"}])))


def test_a_passed_item_goes_stale_when_code_changes(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"}]))
    run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    assert K.status_line(ctx) == ""
    assert K.note_code_changed(ctx, []) == 0
    assert K.note_code_changed(ctx, ["app.py"]) == 1
    assert "↻" in K.status_line(ctx)
    assert "code changed since it passed" in K.checklist_directive("It's done.", [{"name": "checklist"}], ctx)


def test_weak_checks_are_refused(ctx):
    for check in ("exit 0", ":", "sleep 0", "python3 -V", "node --version"):
        assert "runs nothing" in str(run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": check}])))
    res = run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "file:a.txt"}]))
    assert "RUNS something" in str(res)


def test_a_check_honours_the_privilege_memo(ctx, tmp_path):
    from ghost_agent.utils.logging import request_id_context
    E._reset_for_tests()
    tok = request_id_context.set("rq-el")
    try:
        E.note_result("rq-el", "su x", "su: Authentication failure", 1)
        run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "su x -c 'python3 pass.py'"}]))
        out = run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
        assert "✗ #1" in out and "SYSTEM BLOCK" in out
    finally:
        request_id_context.reset(tok)


def test_checklist_and_execute_keep_the_models_order_in_a_batch():
    from ghost_agent.core.agent import _batch_call_depends
    assert _batch_call_depends("checklist", {"action": "check"}, "execute", {"command": "npm run build"})
    assert _batch_call_depends("checklist", {"action": "check"}, "checklist", {"action": "add"})
    assert _batch_call_depends("execute", {"command": "x"}, "checklist", {"action": "check"})
    assert not _batch_call_depends("checklist", {"action": "check"}, "manage_services", {})


def test_checklist_item_needs_a_check(ctx):
    assert "has no check" in str(run(K.tool_checklist(ctx, action="add", items=[{"title": "a"}])))


def test_checklist_file_and_page_checks(ctx, tmp_path):
    sbx = tmp_path / "sbx"
    sbx.mkdir()
    (sbx / "a.txt").write_text("x")
    (sbx / "empty.txt").write_text("")

    async def probe(paths):
        return (paths[0], "") if paths[0] == "ok.html" else (paths[0], "UNCAUGHT JS EXCEPTIONS (1)")

    run(K.tool_checklist(ctx, action="set", items=[
        {"title": "f", "check": "file:a.txt"}, {"title": "g", "check": "file:none.txt"},
        {"title": "p", "check": "page:ok.html"}, {"title": "q", "check": "page:bad.html"},
        {"title": "e", "check": "file:empty.txt"}]))
    out = run(K.tool_checklist(ctx, action="check", page_probe=probe, sandbox_dir=sbx))
    assert "✓ #1 f" in out and "✗ #2 g" in out and "✓ #3 p" in out and "✗ #4 q" in out and "✗ #5 e" in out


def test_checklist_conversation_list_idles_out(ctx):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 a.py"}]))
    p = next((Path(ctx.memory_dir).parent / "checklists").glob("*.json"))
    d = json.loads(p.read_text())
    d["updated"] = time.time() - K.CONVERSATION_TTL_S - 5
    p.write_text(json.dumps(d))
    assert K.load(ctx)["items"] == []


def test_checklist_project_scope(tmp_path):
    ctx = types.SimpleNamespace(memory_dir=str(tmp_path / "memory"), current_project_id="abc123")
    assert K.scope_key(ctx) == "project-abc123"


def test_checklist_directive_only_on_a_done_claim_with_open_items(ctx, tmp_path):
    rows = [{"name": "checklist"}]
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 fail.py"}]))
    assert "1 of 1" in K.checklist_directive("All features are done and working.", rows, ctx)
    assert K.checklist_directive("I built step one; next is the board.", rows, ctx) == ""
    assert K.checklist_directive("It's done.", [], ctx) == ""          # not worked on this request
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"}]))
    run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    assert K.checklist_directive("It's done.", rows, ctx) == ""


def test_a_checklist_check_is_a_run_for_the_claim_arm_but_a_view_is_not():
    content = "Checklist: 1/1 passed.\n✓ #1 runs — check: cd g && python3 app.py"
    check = {"name": "checklist", "arguments": {"action": "check"}, "content": content}
    view = {"name": "checklist", "arguments": {"action": "view"}, "content": content}
    assert C.unbacked_code_writes([_w("g/app.py"), check]) == ([], [])
    assert C.unbacked_code_writes([_w("g/app.py"), view]) == (["g/app.py"], [])
    other = {"name": "checklist", "arguments": {"action": "check", "ids": [2]}, "content": content}
    assert C.unbacked_code_writes([_w("g/app.py"), other]) == (["g/app.py"], [])


def test_checklist_registered_and_contained():
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    assert "checklist" in {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "checklist" in REPLAY_FORBIDDEN_TOOLS


def test_restricted_surface_without_execute_drops_checklist():
    from ghost_agent.core.isolation import restrict_tool_surface
    agent = types.SimpleNamespace(context=types.SimpleNamespace(), disabled_tools=set(),
                                  available_tools={"execute": 1, "checklist": 2, "file_system": 3})
    keep = restrict_tool_surface(agent, forbidden={"execute"})
    assert "checklist" not in keep and "checklist" not in agent.available_tools


def test_status_line_for_batch_only_when_coding_or_touched(ctx):
    from ghost_agent.core.agent import GhostAgent
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 a.py"}]))
    a = types.SimpleNamespace(available_tools={"checklist": 1}, context=ctx)
    f = GhostAgent._checklist_status_for_batch
    assert f(a, [("web_search", "i", "h", False)], []) == ""
    assert "[CHECKLIST 0/1" in f(a, [("file_system", "i", "h", True)], [])
    assert f(a, [("file_system", "i", "h", False)], []) == ""            # a read
    assert "[CHECKLIST" in f(a, [("web_search", "i", "h", False)], [{"name": "checklist"}])


# ── #4: labs, browser, EPERM ──────────────────────────────────────────

def test_lab_templates_enum_matches_the_template_files():
    from ghost_agent.tools.labs import LAB_TEMPLATES, LAB_TOOL_DEFINITION
    stems = {p.stem for p in (REPO / "interface" / "ghost-labs" / "templates").glob("*.yaml")}
    assert set(LAB_TEMPLATES) == stems
    props = LAB_TOOL_DEFINITION["function"]["parameters"]["properties"]
    assert set(props["template"]["enum"]) == stems


def test_lab_read_member_default():
    from ghost_agent.tools.labs import _default_read_member
    assert _default_read_member("http", ["web"], {}) == "web"
    assert _default_read_member("http", ["db", "web"], {"member": "web"}) == "web"
    assert _default_read_member("http", ["db", "web"], {}) == ""
    assert _default_read_member("http", ["db", "web"], {"member": "gone"}) == ""
    assert _default_read_member("exec", ["web"], {}) == ""


@pytest.mark.parametrize("url,host", [("http://lab-1.tail9.ts.net:8080/", "lab-1.tail9.ts.net"),
                                      ("http://[fd7a:115c:a1e0::1]:80/", "fd7a:115c:a1e0::1"),
                                      ("http://evilts.net/", ""),
                                      ("100.101.2.3:3000", "100.101.2.3"), ("http://example.com", ""),
                                      ("http://100.128.0.1/", ""), ("file:///workspace/a.html", "")])
def test_tailnet_host(url, host):
    from ghost_agent.tools.browser_tailnet import tailnet_host
    assert tailnet_host(url) == host


def test_browser_refuses_a_tailnet_url_with_the_lab_hint(tmp_path):
    from ghost_agent.tools.browser import tool_browser
    out = str(run(tool_browser(operation="navigate", url="http://lab-1.tail9.ts.net/", sandbox_dir=tmp_path,
                               sandbox_manager=object())))
    assert "tailnet" in out and "lab http" in out and "if you have the `lab` tool" in out
    out = str(run(tool_browser(operation="interact", url="file:///workspace/a.html", sandbox_dir=tmp_path,
                               sandbox_manager=object(), actions=[{"action": "goto", "url": "http://100.64.1.1/"}])))
    assert "tailnet" in out


def test_env_limits_remembers_a_privilege_refusal_for_the_request():
    E._reset_for_tests()
    assert E.note_result("r1", "su postgres -c 'psql'", "su: Operation not permitted", 1) == ["su"]
    assert "SYSTEM BLOCK" in E.refusal("r1", "cd /x && su - postgres -c 'createdb a'")
    assert E.refusal("r1", "psql -c 'select 1'") == ""
    assert E.refusal("r2", "su postgres") == ""                         # another request
    assert E.note_result("r3", "su x", "ok", 0) == []                    # it worked
    assert E.note_result("r3", "su x", "su: Authentication failure (warning only)", 0) == []   # exit 0 = it ran
    assert E.note_result("r3", "chown a b", "Operation not permitted", 1) == []   # not a privilege program
    assert E.note_result("r3", "sudo ls", "No such file", 1) == []      # not a privilege refusal
    # only the program the refusal names is blamed; a refusal naming none blames a lone privilege program only
    # the refusal names `kill`, not the privilege program: nothing is blamed
    assert E.note_result("r4", "kill -9 1; sudo -n true", "bash: kill: (1) - Operation not permitted", 1) == []
    assert E.note_result("r4b", "chown pg x; sudo -n true", "chown: changing ownership of 'x': Operation not permitted", 1) == []
    assert E.note_result("r4c", "sudo -n true", "sudo: a password is required", 1) == ["sudo"]
    assert E.note_result("r5", "sudo a; runuser b", "Operation not permitted", 1) == []
    assert E.note_result("r6", "su p -c x && sudo ls", "su: Authentication failure", 1) == ["su"]
    assert "postgres_admin" not in E.refusal("r1", "su x")
    E.forget("r1")
    assert E.refusal("r1", "su postgres") == ""


@pytest.mark.parametrize("cmd,progs", [("timeout 30 sudo -u x ls", ["sudo"]),
                                       ("A=1 env -i setpriv --reuid 1 id", ["setpriv"]),
                                       ("echo $(su -c id)", ["echo", "su"]),
                                       ("cat <<EOF > notes.txt\nsu is blocked\nEOF", ["cat"]),
                                       ("cat > s.sh <<'X'\nsudo apt-get install y\nX\nbash s.sh", ["cat", "bash"]),
                                       ('echo "done; sudo is not here"', ["echo"]),
                                       ("command -v sudo || echo none", ["echo"]),
                                       ("which su", ["which"])])
def test_programs_in(cmd, progs):
    assert E.programs_in(cmd) == progs


# ── #5: the web-app starter ───────────────────────────────────────────

def test_scaffold_writes_the_files_and_never_overwrites(tmp_path):
    from ghost_agent.tools.file_system import tool_file_system
    res = str(run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="my-game", template="web-app")))
    from ghost_agent.tools.file_system import _template_written_paths
    assert _template_written_paths(res) == ["my-game/index.html", "my-game/css/style.css", "my-game/js/app.js",
                                    "my-game/js/main.js", "my-game/tests/check_page.py"]
    assert "My Game" in (tmp_path / "my-game" / "index.html").read_text()
    again = str(run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="my-game", template="web-app")))
    assert "never overwrites" in again
    assert "unknown template" in str(run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="z",
                                                         template="react")))
    assert "no content" in str(run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="z",
                                                   template="web-app", content="x")))
    assert "not the workspace root" in str(run(tool_file_system(operation="write", sandbox_dir=tmp_path,
                                                               path=".", template="web-app")))


def test_scaffold_absolute_and_container_paths_resolve(tmp_path):
    from ghost_agent.tools.file_system import _get_safe_path
    ok, msg, written = S.scaffold(tmp_path, str(tmp_path / "abs"), "web-app", _get_safe_path)
    assert ok and written[0][0] == "abs/index.html" and (tmp_path / "abs" / "index.html").is_file()
    ok, msg, written = S.scaffold(tmp_path, "/workspace/wsapp", "web-app", _get_safe_path)
    assert ok and written[0][0] == "wsapp/index.html" and "cd wsapp &&" in msg
    ok, msg, _ = S.scaffold(tmp_path, "repo/.git/x", "web-app", _get_safe_path)
    assert not ok and ".git" in msg


def test_scaffold_files_are_this_requests_own(tmp_path):
    from ghost_agent.tools.file_system import _get_safe_path
    marked = []
    S.scaffold(tmp_path, "app", "web-app", _get_safe_path, marked.append)
    assert len(marked) == 5


def test_scaffold_title_is_escaped(tmp_path):
    files = S.render("web-app", "<b>x'y")
    assert "<b>" not in files["index.html"] and "&lt;b&gt;x&#x27;y" in files["index.html"]
    assert "<h1>&lt;b&gt;x&#x27;y</h1>" in files["js/app.js"]          # no raw quote can end the JS string


def test_scaffold_confirmation_feeds_the_write_ledgers(tmp_path):
    from ghost_agent.tools.file_system import tool_file_system
    msg = str(run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="app", template="web-app")))
    row = {"name": "file_system", "arguments": {"operation": "write", "path": "app", "template": "web-app"},
           "content": msg}
    assert set(C.code_evidence([row])) >= {"index.html", "app.js", "main.js"}


def test_scaffold_runner_passes_headless(tmp_path):
    pytest.importorskip("playwright")
    from ghost_agent.tools.file_system import _get_safe_path
    ok, _msg, _w2 = S.scaffold(tmp_path, "app", "web-app", _get_safe_path)
    try:
        r = subprocess.run([sys.executable, "tests/check_page.py"], cwd=tmp_path / "app", capture_output=True,
                           text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no headless browser here: {e}")
    if "Executable doesn't exist" in r.stderr or "playwright install" in r.stderr:
        pytest.skip("no chromium for playwright")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS app loads" in r.stdout
    (tmp_path / "app" / "js" / "main.js").write_text("throw new Error('boom');")
    r = subprocess.run([sys.executable, "tests/check_page.py"], cwd=tmp_path / "app", capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 1 and "boom" in r.stdout


# ── round-2 review fixes ──────────────────────────────────────────────

def test_a_check_after_the_write_in_one_batch_stays_passed(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"},
                                                   {"title": "b", "check": "python3 pass2.py"}]))
    out = run(K.tool_checklist(ctx, action="check", ids=[1], run_execute=_ex, sandbox_dir=tmp_path))
    assert K.ran_ids_in(out) == {1}                       # #2 was printed with its old state, not run
    run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    assert K.note_code_changed(ctx, ["app.py"], exempt_ids={1, 2}) == 0        # both ran after the write
    assert K.note_code_changed(ctx, ["README.md", "notes.txt", "shot.png"]) == 0   # not code
    assert K.note_code_changed(ctx, ["style.css"], exempt_ids={1}) == 1        # #2 ran BEFORE the write
    assert [i["status"] for i in K.load(ctx)["items"]] == ["passed", "stale"]


def _meta(name, args):
    return (name, "id", "h", True, None, False, None, None, False, args)


def test_the_exemption_follows_batch_order():
    ran = "✓ #1 a — check: python3 a.py\n✓ #2 b — check: python3 b.py (not run in this call)"
    metas = [_meta("checklist", {"action": "check", "ids": [1]}), _meta("execute", {"command": "x"}),
             _meta("checklist", {"operation": "check"}), _meta("checklist", {"action": "view"})]
    results = [ran, "ok", "✓ #3 c — check: python3 c.py", "✓ #4 d — check: python3 d.py"]
    assert K.exempt_ids_after("execute", 1, metas, results) == {3}          # #1 ran BEFORE; view runs nothing
    assert K.exempt_ids_after("file_system", 1, metas, results) == {1, 3}   # phase 1: before every call
    assert K.exempt_ids_after("execute", 1, metas, results[:2]) == set()    # no result yet → nothing


@pytest.mark.parametrize("cmd", ["if [ -f app.js ]; then echo ok; fi", "[[ -s app.js ]]", "! grep -q x app.js",
                                 "(cat app.js)", "while read l; do echo $l; done < app.js", "ls | xargs cat app.js"])
def test_shell_keywords_and_tests_are_not_runs(cmd):
    assert C.unbacked_code_writes([_w("app.js"), _x(cmd)]) == (["app.js"], [])


@pytest.mark.parametrize("cmd", ["if true; then node app.js; fi", "(cd x && node app.js)", "{ node app.js; }"])
def test_a_run_inside_shell_grouping_still_counts(cmd):
    assert C.unbacked_code_writes([_w("app.js"), _x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd,progs", [("for f in a; do sudo rm $f; done", ["for", "sudo"]),
                                       ("if x; then sudo id; fi", ["x", "sudo"]), ("{ sudo id; }", ["sudo"]),
                                       ("! sudo id", ["sudo"]), ("grep x <<< hello\nsudo id", ["grep", "sudo"])])
def test_privileged_runs_inside_shell_grouping_are_seen(cmd, progs):
    assert E.programs_in(cmd) == progs


def test_page_folder_and_asset_scripts():
    from ghost_agent.core.agent import _page_folder
    assert _page_folder("app/js/clock.js") == _page_folder("app/index.html") == "app"
    assert _page_folder("a/index.html") != _page_folder("b/js/x.js")


def test_a_later_write_of_another_page_does_not_take_the_probe(tmp_path):
    a, calls = _agent_stub(tmp_path, ("a/index.html", ""))
    text = "SUCCESS: Wrote 10 chars to 'a/app.js'. Script-side path (from sandbox cwd): 'a/app.js'."
    other = "SUCCESS: Wrote 9 chars to 'b/index.html'. Script-side path (from sandbox cwd): 'b/index.html'."
    same = "SUCCESS: Wrote 9 chars to 'a/js/x.js'. Script-side path (from sandbox cwd): 'a/js/x.js'."
    meta = [("file_system",), ("file_system",)]
    assert run(a._page_check_after_write(text, 0, meta, [text, same])) == "" and calls == []
    run(a._page_check_after_write(text, 0, meta, [text, other]))
    assert calls == [["a/app.js"]]


def test_rows_a_check_did_not_run_are_not_credited():
    content = ("Checklist: 1/2 passed.\n✓ #1 a — check: python3 app.py (not run in this call)\n"
               "✗ #2 b — check: python3 other.py")
    row = {"name": "checklist", "arguments": {"action": "check"}, "content": content}
    assert C.unbacked_code_writes([_w("app.py"), row]) == (["app.py"], [])


@pytest.mark.parametrize("cmd,key", [("case $1 in start) python3 app.py;; esac", "app.py"),
                                     ("for t in a_test.py b_test.py; do python3 $t; done", "a_test.py"),
                                     ("echo $(python3 app.py)", "app.py"), ("python3 app.py 2>&1 | tail", "app.py"),
                                     ("sudo -u x python3 app.py", "app.py")])
def test_round3_runs_count(cmd, key):
    assert C.unbacked_code_writes([_w(key), _x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd", ["sudo -u www cat app.py", "env -i cat app.py", "nice -n 5 cat app.py",
                                 "xargs -n1 cat app.py", "timeout -s KILL 5 cat app.py", "cat app.py 2>&1",
                                 "command -v node"])
def test_round3_prefix_flags_do_not_make_a_read_a_run(cmd):
    assert not K.runs_something(cmd)
    assert C.unbacked_code_writes([_w("app.py"), _x(cmd)]) == (["app.py"], [])


@pytest.mark.parametrize("cmd,progs", [("cat<<EOF\nsudo x\nEOF\necho", ["cat<<EOF", "echo"]),
                                       ("echo $(( 1 << x ))\nsudo id", ["echo", "1", "sudo"])])
def test_round3_heredoc_edges(cmd, progs):
    assert E.programs_in(cmd) == progs


def test_two_pages_in_one_folder_are_both_probed(tmp_path):
    a, calls = _agent_stub(tmp_path, ("site/index.html", ""))
    first = "SUCCESS: Wrote 1 chars to 'site/index.html'. Script-side path (from sandbox cwd): 'site/index.html'."
    second = "SUCCESS: Wrote 1 chars to 'site/about.html'. Script-side path (from sandbox cwd): 'site/about.html'."
    meta = [("file_system",), ("file_system",)]
    run(a._page_check_after_write(first, 0, meta, [first, second]))
    assert calls == [["site/index.html"]]
    # a script of the same page, written later, does take the probe over
    calls.clear()
    js = "SUCCESS: Wrote 1 chars to 'site/js/a.js'. Script-side path (from sandbox cwd): 'site/js/a.js'."
    assert run(a._page_check_after_write(first, 0, meta, [first, js])) == "" and calls == []


def test_a_later_syntax_failed_write_does_not_take_the_probe(tmp_path):
    from ghost_agent.tools.outcome import ToolOutcome
    a, calls = _agent_stub(tmp_path, ("index.html", ""))
    first = "SUCCESS: Wrote 1 chars to 'index.html'. Script-side path (from sandbox cwd): 'index.html'."
    broken = ToolOutcome.partial("SUCCESS: Wrote 1 chars to 'app.js'. Script-side path (from sandbox cwd): 'app.js'.",
                                 world_changed=True, reason_code="written_but_syntax_failed")
    run(a._page_check_after_write(first, 0, [("file_system",), ("file_system",)], [first, broken]))
    assert calls == [["index.html"]]


@pytest.mark.parametrize("cmd", ['x="$(python3 app.py)"; echo "$x"', 'echo "Result: $(python3 app.py)"',
                                 'diff <(python3 app.py) expected.txt', "echo `python3 app.py`",
                                 "cat <<EOF | python3\nimport app\nEOF", "python3 - <<EOF\nimport app\nEOF"])
def test_round4_substitutions_and_fed_heredocs_are_runs(cmd):
    assert C.unbacked_code_writes([_w("app.py"), _x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd,kind", [("watch -d node --check app.js", "check"), ("sudo -s bash -n run.sh", "check")])
def test_round4_value_options_are_per_prefix(cmd, kind):
    assert [C.segment_kind(w) for w in C._shell_segments(cmd)] == [kind]


@pytest.mark.parametrize("cmd", ["timeout 30 python3 - <<'EOF'\nimport app\nEOF",
                                 'python3 - <<EOF\nexec(open("app.py").read())\nEOF', "t() { python3 app.py; }; t",
                                 "echo `python3 app.py && echo"])
def test_round5_heredoc_bodies_and_calls_count(cmd):
    assert C.unbacked_code_writes([_w("app.py"), _x(cmd)]) == ([], [])


def test_round5_a_file_redirect_substitution_is_a_read():
    assert C.unbacked_code_writes([_w("app.py"), _x("x=$(< app.py)")]) == (["app.py"], [])


def test_round4_heredoc_without_space_after_a_digit():
    assert E.programs_in("python3 test2.py<<EOF\nsudo x\nEOF\nls") == ["python3", "ls"]


@pytest.mark.parametrize("cmd,key", [("python3 $(find . -name app.py)", "app.py"), ("python3 `which tool.py`", "tool.py")])
def test_round6_a_substituted_script_is_run(cmd, key):
    assert C.unbacked_code_writes([_w(key), _x(cmd)]) == ([], [])


def test_round6_a_substitution_under_a_read_is_not_a_run():
    assert C.unbacked_code_writes([_w("app.py"), _x("echo $(cat app.py)")]) == (["app.py"], [])


def test_round6_the_operation_alias_counts_as_a_check():
    row = {"name": "checklist", "arguments": {"operation": "check"}, "content": "✓ #1 a — check: python3 app.py"}
    assert C.unbacked_code_writes([_w("app.py"), row]) == ([], [])


def test_a_passed_item_printed_but_not_run_is_not_a_run(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"},
                                                   {"title": "b", "check": "python3 pass2.py"}]))
    run(K.tool_checklist(ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path))
    out = run(K.tool_checklist(ctx, action="check", ids=[1], run_execute=_ex, sandbox_dir=tmp_path))
    assert "✓ #2 b" in out and K.ran_ids_in(out) == {1}


async def test_a_batch_write_stales_only_items_its_later_check_did_not_run(tmp_path):
    """The agent wiring end to end: a write and a `check ids=[1]` in ONE batch — #1 ran after the write and
    keeps its pass, #2 (passed earlier) must be checked again."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.utils.logging import conversation_key_context
    tok = conversation_key_context.set("conv-4uu-batch")
    try:
        agent = H._agent()
        agent.context.memory_dir = str(tmp_path / "sys" / "memory")
        agent.context.sandbox_dir = str(tmp_path / "ws")
        cl_ctx = types.SimpleNamespace(memory_dir=agent.context.memory_dir, current_project_id=None)
        await K.tool_checklist(cl_ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"},
                                                            {"title": "b", "check": "python3 pass2.py"}])
        await K.tool_checklist(cl_ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path)

        async def fs(**kw):
            return "SUCCESS: Wrote 12 chars to 'app.py'."

        async def checklist(**kw):
            return "✓ #1 a — check: python3 pass.py\n✓ #2 b — check: python3 pass2.py (not run in this call)"
        agent.available_tools = {"file_system": fs, "checklist": checklist}
        ts = H._ts([("file_system", {"operation": "write", "path": "app.py", "content": "print('hi')\n"}),
                    ("checklist", {"action": "check", "ids": [1]})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
        assert [i["status"] for i in K.load(cl_ctx)["items"]] == ["passed", "stale"]
    finally:
        conversation_key_context.reset(tok)


@pytest.mark.parametrize("refused,want", [(True, "passed"), (False, "stale")])
async def test_an_execute_the_tool_refused_wrote_nothing(tmp_path, refused, want):
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
    from ghost_agent.utils.logging import conversation_key_context
    tok = conversation_key_context.set(f"conv-4uu-refused-{refused}")
    try:
        agent = H._agent()
        agent.context.memory_dir = str(tmp_path / "sys" / "memory")
        agent.context.sandbox_dir = str(tmp_path / "ws")
        cl_ctx = types.SimpleNamespace(memory_dir=agent.context.memory_dir, current_project_id=None)
        await K.tool_checklist(cl_ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"}])
        await K.tool_checklist(cl_ctx, action="check", run_execute=_ex, sandbox_dir=tmp_path)

        async def execute(**kw):
            if refused:
                return ToolOutcome("SYSTEM INSTRUCTION: command refused.", status=OutcomeStatus.REJECTED,
                                   reason_code="shell_command_rejected")
            return "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n\n"
        agent.available_tools = {"execute": execute}
        ts = H._ts([("execute", {"command": "cat > app.py <<'EOF'\nprint(1)\nEOF"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
        assert [i["status"] for i in K.load(cl_ctx)["items"]] == [want]
    finally:
        conversation_key_context.reset(tok)


@pytest.mark.parametrize("args", [{}, {"content": ""}, {"content": " "}])
async def test_a_template_write_without_content_reaches_the_tool(args):
    """Live probe 568ab528: the turn loop's empty-content guard blocked every template call before the tool saw
    it, while the tool refused one WITH content — the model had no way through and built the skeleton by hand."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fs(**kw):
        seen.append(kw)
        return "SUCCESS: Wrote 10 chars to 'app/index.html'."
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {"operation": "write", "path": "app", "template": "web-app", **args})],
               StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and seen[0].get("template") == "web-app"


def test_whitespace_content_with_a_template_still_scaffolds(tmp_path):
    from ghost_agent.tools.file_system import tool_file_system
    out = run(tool_file_system(operation="write", path="app", template="web-app", content=" ",
                               sandbox_dir=tmp_path))
    assert "SUCCESS: Wrote" in str(out) and (tmp_path / "app" / "index.html").is_file()


# ── gaps closed after the first deploy ──

@pytest.mark.parametrize("cmd,files", [
    ('sed -i "s/a/b/" app.py', ["app.py"]), ("sed -i.bak -e x app.js lib.js", ["app.js", "lib.js"]),
    ("perl -pi -e s/a/b/ app.py", ["app.py"]), ("cp backup/app.py app.py", ["app.py"]),
    ("git checkout -- app.py", ["app.py"]), ("npx prettier --write app.js", ["app.js"]), ("black app.py", ["app.py"]),
    ("ruff format app.py", ["app.py"]),
    ("sed -n 1,5p app.py", []), ("cp app.py app.py.bak", []), ("prettier --check app.js", []),
    ("black --check app.py", []), ("ruff check app.py", []),
])
def test_in_place_edits_are_writes(cmd, files):
    assert C._inplace_written(cmd) == files


def test_an_in_place_edit_after_the_run_needs_a_new_run():
    assert C.unbacked_code_writes([_w("main.py"), _x("python3 main.py"), _x("sed -i s/a/b/ main.py")]) == (["main.py"], [])
    assert C.unbacked_code_writes([_w("main.py"), _x("sed -i s/a/b/ main.py && python3 main.py")]) == ([], [])


@pytest.mark.parametrize("cmd,progs", [
    ("bash <<EOF\nsudo apt-get install x\nEOF", ["bash", "sudo"]), ("cat <<EOF | sh\nsu -c id\nEOF", ["cat", "sh", "su"]),
    ("sudo bash <<EOF\napt-get update\nEOF", ["sudo", "apt-get"]), ("bash -s <<'EOF'\nchown root f\nEOF", ["bash", "chown"]),
    ("cat > run.sh <<EOF\nsudo x\nEOF", ["cat"]),
])
def test_a_heredoc_fed_to_a_shell_is_commands(cmd, progs):
    assert E.programs_in(cmd) == progs


def _page_write(path):
    return {"name": "file_system", "content": f"SUCCESS: Wrote 30 chars to '{path}'.",
            "args": {"operation": "write", "path": path, "content": "<script src='js/app.js'></script>"}}


@pytest.mark.parametrize("row,tested", [
    ({"name": "execute", "content": "--- COMMAND RESULT ---\nEXIT CODE: 0\nPASS clock",
      "args": {"command": "cd tetris && python3 tests/check_page.py clock"}}, True),
    ({"name": "execute", "content": "--- COMMAND RESULT ---\nEXIT CODE: 1\nFAIL clock",
      "args": {"command": "cd tetris && python3 tests/check_page.py clock"}}, False),
    ({"name": "execute", "content": "--- COMMAND RESULT ---\nEXIT CODE: 0\n",
      "args": {"command": "python3 other/tests/check_page.py"}}, False),
    ({"name": "checklist", "content": "✓ #1 clock — check: python3 tetris/tests/check_page.py clock",
      "args": {"action": "check"}}, True),
    ({"name": "checklist", "content": "✓ #1 clock — check: python3 tetris/tests/check_page.py clock (not run in this call)",
      "args": {"action": "check"}}, False),
])
def test_the_templates_click_test_exercises_its_pages(row, tested):
    from ghost_agent.core import task_state as TSt
    out = TSt.untested_pages([_page_write("tetris/index.html"), row])
    assert (out == []) is tested


@pytest.mark.parametrize("cmd,ok", [
    ("curl -s http://127.0.0.1:8080/ | grep ok", True), ("curl -f localhost:5000/api/items", True),
    ("curl -sf http://localhost:3000/health", True), ("wget -q -O- localhost:8000 | grep x", True),
    ("curl http://localhost:8080/", False), ("curl -f https://example.com", False),
    ("curl -s https://example.com | grep ok", False),
])
def test_a_local_service_check_that_fails_on_a_bad_answer_runs_something(cmd, ok):
    assert K.runs_something(cmd) is ok


def test_a_template_path_that_names_a_file_is_refused(tmp_path):
    from ghost_agent.tools.file_system import tool_file_system
    out = run(tool_file_system(operation="write", path="t4/index.html", template="web-app", sandbox_dir=tmp_path))
    assert "FOLDER" in str(out) and not (tmp_path / "t4" / "index.html").exists()


def test_the_never_run_directive_names_the_templates_click_check():
    d = C.unbacked_claim_directive("The app is done and it works.", [_w("app.py")])
    assert "tests/check_page.py" in d and "load it in the browser" not in d


def test_a_capability_query_is_not_a_privilege_switch():
    E._reset_for_tests()
    assert E.note_result("rq", "capsh --user=x -- -c id", "capsh: Operation not permitted", 1) == ["capsh"]
    assert E.refusal("rq", "capsh --print") == "" and E.refusal("rq", "capsh --user=y") != ""


@pytest.mark.parametrize("reply,fires", [
    ("Not all features are working yet.", False), ("complete except for the score board, which still fails", False),
    ("The app is done; 2 items remaining.", False), ("I built step one; next is the board.", False),
    ("Done!", True), ("Your app is ready.", True), ("The Tetris game is ready to play.", True),
    ("I finished the app.", True), ("Everything is working now.", True),
])
def test_done_wording(ctx, reply, fires):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 fail.py"}]))
    assert bool(K.checklist_directive(reply, [{"name": "checklist"}], ctx)) is fires


def test_drop_unknown_ids_and_a_damaged_file(ctx, tmp_path):
    run(K.tool_checklist(ctx, action="set", items=[{"title": "a", "check": "python3 pass.py"}]))
    assert "no item with id [99]" in str(run(K.tool_checklist(ctx, action="drop", ids=[99])))
    p = K._path_for(ctx, K.scope_key(ctx))
    p.write_text('{"items": [{"id": "x", "title": "bad", "check": "c"}, {"id": 2, "title": "no check"},'
                 ' {"id": 3, "title": "ok", "check": "python3 pass.py"}], "updated": "soon"}')
    assert [i["id"] for i in K.load(ctx)["items"]] == [3]
    assert "Checklist cleared" in str(run(K.tool_checklist(ctx, action="clear")))


def test_a_folder_check(ctx, tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "a.js").write_text("1")
    ok, out = run(K._run_one("file:dist", run_execute=None, page_probe=None, sandbox_dir=tmp_path))
    assert ok and "1 item" in out


def test_loaded_names_reads_only_real_scripts(tmp_path):
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "a.js").write_text("import {$} from './b.js?v=2';")
    (tmp_path / "js" / "b.js").write_text("import './c.js';")
    (tmp_path / "js" / "c.js").write_text("import('./d.js');")
    (tmp_path / "js" / "d.js").write_text("1")
    (tmp_path / "js" / "x.js").write_text("1")
    (tmp_path / "js" / "lazy.js").write_text("1")          # exists, but `data-src` does not load it
    page = tmp_path / "index.html"
    page.write_text('<script src=js/x.js></script><script data-src="js/lazy.js"></script>'
                    '<script type="module">import "./js/a.js";</script><script src="js/missing.js"></script>')
    assert P.loaded_names(page, tmp_path) == {"index.html", "x.js", "a.js", "b.js", "c.js", "d.js"}


# ── second verification round: guard interplay (driven through the batch) ──

async def _churn_steers(mode):
    import tests.test_4jj_search_yield_steer as H
    from unittest.mock import MagicMock
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()

    async def checklist(**kw):
        return "Checklist: 0/1 passed.\n✗ #1 counter — check: python3 tests/check_page.py counter\n    FAIL counter"

    async def fs(**kw):
        r = "SUCCESS: Edited 'app/js/counter.js' (1 occurrence)."
        if mode == "page":
            r += "\n\n[PAGE CHECK] ✓ loaded 'app/index.html' headless: no uncaught JS exception. Ran: counter.js, index.html."
        return r
    agent.available_tools = {"checklist": checklist, "file_system": fs}
    rs, strikes, steers = MagicMock(), StrikeLedger(), 0
    for k in range(8):
        b = [("file_system", {"operation": "edit", "path": "app/js/counter.js", "old_string": f"a{k}", "new_string": f"b{k}"})]
        if mode == "checklist":
            b.append(("checklist", {"action": "check"}))
        if mode == "view":
            b.append(("checklist", {"action": "view"}))
        ts = H._ts(b, strikes, set())
        ts.request_state = rs
        await agent._dispatch_and_process_tool_batch(ts)
        steers += sum("edit-churn" in str(m.get("content")) for m in ts.messages if m.get("role") == "user")
    return steers


@pytest.mark.parametrize("mode,steered", [("checklist", False), ("page", False), ("bare", True), ("view", True)])
async def test_a_check_or_a_page_check_verifies_for_the_edit_churn_brake(mode, steered):
    assert bool(await _churn_steers(mode)) is steered


async def test_a_failing_checklist_check_is_debugging_not_a_strike(tmp_path):
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.utils.logging import conversation_key_context
    tok = conversation_key_context.set("conv-4uu-strike")
    try:
        agent = H._agent()
        n = {"k": 0}

        async def checklist(**kw):
            n["k"] += 1
            return f"Checklist: 0/1 passed.\n✗ #1 counter — check: python3 t.py\n    FAIL: expected 5 got {n['k']}"

        async def fs(**kw):
            return "SUCCESS: Edited 'app/js/counter.js' (1 occurrence)."
        agent.available_tools = {"checklist": checklist, "file_system": fs}
        strikes, prev, trail = StrikeLedger(), None, []
        for k in range(6):
            ts = H._ts([("file_system", {"operation": "edit", "path": "app/js/counter.js", "old_string": f"a{k}",
                                         "new_string": f"b{k}"}), ("checklist", {"action": "check"})], strikes, set())
            if prev is not None:
                ts.execution_failure_count = prev.execution_failure_count
            await agent._dispatch_and_process_tool_batch(ts)
            prev = ts
            trail.append(ts.execution_failure_count)
        assert trail == [0] * 6 and not prev.force_final_response
    finally:
        conversation_key_context.reset(tok)


async def test_a_status_question_does_not_start_checks():
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-status-4uu")
    try:
        agent = H._agent()
        agent.context._status_only_req = "req-status-4uu"
        ran = []

        async def checklist(**kw):
            ran.append("checklist")
            return "Checklist: 1/1 passed."
        agent.available_tools = {"checklist": checklist}
        ts = H._ts([("checklist", {"action": "check"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
        assert ran == []
    finally:
        request_id_context.reset(tok)


async def test_checklist_checks_run_through_the_egress_guarded_execute(tmp_path, monkeypatch):
    from ghost_agent.tools import registry as R
    seen = []

    def fake_refusal(name, kwargs, context):
        seen.append(name)
        return "SYSTEM BLOCK — egress" if name == "execute" else None
    monkeypatch.setattr("ghost_agent.memory.egress.content_egress_refusal", fake_refusal)
    from unittest.mock import MagicMock
    ctx = MagicMock()
    ctx.sandbox_dir, ctx.memory_dir, ctx.current_project_id = str(tmp_path), str(tmp_path / "m"), None
    ctx.agent, ctx.memory_system = None, None
    tools = R.get_available_tools(ctx)
    from ghost_agent.utils.logging import conversation_key_context
    tok = conversation_key_context.set("conv-4uu-egress")
    try:
        await tools["checklist"](action="set", items=[{"title": "a", "check": "python3 pass.py"}])
        out = str(await tools["checklist"](action="check"))
    finally:
        conversation_key_context.reset(tok)
    assert "execute" in seen and "egress" in out


@pytest.mark.parametrize("cmd,files", [
    ("python3 - <<'EOF'\ns=open('app.py').read()\nopen('app.py','w').write(s)\nEOF", ["app.py"]),
    ("node -e \"require('fs').writeFileSync('app.js', 'x')\"", ["app.js"]),
])
def test_inline_code_writes_are_writes_not_runs(cmd, files):
    assert C.unbacked_code_writes([_w(files[0]), _x(cmd)]) == (files, [])


@pytest.mark.parametrize("cmd,never", [
    ('python3 -c "\nprint(open(\'page.html\').read())\n"', ["page.html"]),
    ("python3 -c \"exec(open('page.py').read())\"", []),
    ("python3 -c 'import page; page.main()'", []),
])
def test_inline_code_runs_what_it_imports_or_executes_not_what_it_opens(cmd, never):
    f = "page.html" if "html" in cmd else "page.py"
    assert C.unbacked_code_writes([_w(f), _x(cmd)])[0] == never


def _b(args, ok=True):
    return {"name": "browser", "arguments": args, "args": args,
            "content": "--- BROWSER RESULT ---\nSTATUS: " + ("OK" if ok else "FAILED") + "\n"}


@pytest.mark.parametrize("load,never", [
    ({"operation": "navigate", "url": "http://127.0.0.1:8100/"}, []),
    ({"operation": "interact", "actions": [{"action": "goto", "url": "file:///workspace/app/index.html"}]}, []),
    ({"operation": "navigate", "url": "file:///workspace/other/index.html"}, ["app/js/app.js"]),
])
def test_a_clean_page_load_runs_the_scripts_it_serves(load, never):
    rows = [_w("app/index.html"), _w("app/js/app.js"), _b(load)]
    assert [p for p in C.unbacked_code_writes(rows)[0] if p.endswith(".js")] == never


def test_a_failed_page_load_credits_no_script():
    rows = [_w("app/index.html"), _w("app/js/app.js"), _b({"operation": "navigate", "url": "http://127.0.0.1:8100/"}, ok=False)]
    assert "app/js/app.js" in C.unbacked_code_writes(rows)[0]


def test_a_playwright_script_that_loads_the_page_runs_its_scripts():
    pw = ("python3 - <<'EOF'\nfrom playwright.sync_api import sync_playwright\nwith sync_playwright() as p:\n"
          "  pg = p.chromium.launch().new_page(); pg.goto('file:///workspace/app/index.html')\nEOF")
    assert C.unbacked_code_writes([_w("app/index.html"), _w("app/js/app.js"), _x(pw)]) == ([], [])


@pytest.mark.parametrize("rows", [
    [{"name": "file_system", "args": {"operation": "delete", "path": "fix2.py"}, "content": "SUCCESS: deleted fix2.py"}],
    [None],
])
def test_a_deleted_file_is_not_never_run(rows):
    rows = [_w("fix2.py")] + [r if r else _x("rm -f fix2.py") for r in rows]
    assert C.unbacked_code_writes(rows) == ([], [])


@pytest.mark.parametrize("cmd,kind", [("bash -c 'node --check a.js'", "check"), ("sh -c 'cat app.py'", "read"),
                                      ("bash -c true", "read"),
                                      ("python3 -c 'import re,app; app.main(re.compile(\"x\"))'", "run")])
def test_shell_c_and_re_compile(cmd, kind):
    assert [C.segment_kind(w) for w in C._shell_segments(cmd)] == [kind]


def test_a_huge_hostile_line_parses_fast():
    t = time.time()
    C._shell_segments("echo $(" + "(a)" * 20000)
    assert time.time() - t < 1.0


@pytest.mark.parametrize("console,fails", [
    ("CONSOLE (1 error/warning):\n  • [error] file:///w/js/app.js — Failed to load resource: net::ERR_FILE_NOT_FOUND", True),
    ("CONSOLE (1 error/warning):\n  • [error] file:///w/img/logo.png — Failed to load resource: net::ERR_FILE_NOT_FOUND", False),
    ("CONSOLE (1 error/warning):\n  • [error] Access to script at 'file:///w/js/a.js' from origin 'null' has been blocked by CORS policy", True),
    ("CONSOLE (1 error/warning):\n  • [error] https://cdn.x/y.js — Failed to load resource: net::ERR_NAME_NOT_RESOLVED", False),
    ("", False),
])
def test_a_local_file_that_did_not_load_fails_the_page(console, fails):
    from ghost_agent.core.agent import _local_load_failures
    out = _local_load_failures(console)
    assert bool(out) is fails
    if "origin 'null'" in console:
        assert "classic" in out


def _scaffolded(tmp_path, name):
    from ghost_agent.tools.file_system import _get_safe_path
    ok, msg, _ = S.scaffold(tmp_path, name, "web-app", _get_safe_path)
    assert ok, msg
    return tmp_path / name


def _run_runner(app, *words, timeout=60):
    r = subprocess.run([sys.executable, str(app / "tests" / "check_page.py"), *words],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def _has_playwright() -> bool:
    """find_spec raises when a sibling test left a spec-less `playwright` stub in sys.modules (same xdist worker)."""
    try:
        return __import__("importlib").util.find_spec("playwright") is not None
    except (ValueError, ImportError):
        return False


_needs_pw = pytest.mark.skipif(shutil.which("true") is None or not _has_playwright(),
                               reason="playwright not installed")


@_needs_pw
def test_the_runner_names_a_missing_script_and_a_bad_step(tmp_path):
    app = _scaffolded(tmp_path, "app")
    idx = app / "index.html"
    idx.write_text(idx.read_text().replace('<script src="js/main.js">', '<script src="js/nothere.js"></script><script src="js/main.js">'))
    code, out = _run_runner(app)
    assert code == 1 and "nothere.js" in out
    app2 = _scaffolded(tmp_path, "app2")
    cp = app2 / "tests" / "check_page.py"
    cp.write_text(cp.read_text().replace('("text", "#status", "ready")', '("text", "#status")'))
    code, out = _run_runner(app2)
    assert code == 1 and "needs 2 argument(s)" in out
    assert _run_runner(app2, "app", "nothing")[0] == 2                 # every word must match a check's name


@_needs_pw
def test_the_runner_fails_a_frozen_page_instead_of_hanging(tmp_path):
    app = _scaffolded(tmp_path, "app")
    js = app / "js" / "app.js"
    js.write_text(js.read_text() + "\nsetTimeout(function(){while(true){}},300);\n")
    cp = app / "tests" / "check_page.py"
    cp.write_text(cp.read_text().replace("CHECK_TIMEOUT_S = 30", "CHECK_TIMEOUT_S = 3").replace(
        '("app loads", [("visible", "#app"), ("text", "#status", "ready")]),', '("app loads", [("wait", 600), ("eval", "1")]),'))
    t = time.time()
    code, out = _run_runner(app, timeout=40)
    assert code == 1 and "froze" in out and time.time() - t < 30



# ── third verification round ──

async def test_a_status_question_may_still_read_the_checklist():
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-status-4uu-v")
    try:
        agent = H._agent()
        agent.context._status_only_req = "req-status-4uu-v"
        ran = []

        async def checklist(**kw):
            ran.append(kw.get("action"))
            return "Checklist: 1/2 passed."
        agent.available_tools = {"checklist": checklist}
        ts = H._ts([("checklist", {"action": "view"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
        assert ran == ["view"]
    finally:
        request_id_context.reset(tok)


def test_a_page_load_that_threw_credits_no_script():
    rows = [_w("app/index.html"), _w("app/js/app.js"),
            _b({"operation": "navigate", "url": "file:///workspace/app/index.html"})]
    rows[-1]["content"] += "⚠ UNCAUGHT JS EXCEPTIONS (1) — boom"
    assert "app/js/app.js" in C.unbacked_code_writes(rows)[0]


def test_a_localhost_page_credits_front_end_scripts_only():
    rows = [_w("scripts/migrate.js"), _w("web/js/app.js"), _b({"operation": "navigate", "url": "http://localhost:3000/"})]
    assert C.unbacked_code_writes(rows)[0] == ["scripts/migrate.js"]


@pytest.mark.parametrize("cmd", ["isort --check-only app.py", "ruff format --check app.py", "rustfmt --check app.rs",
                                 "perl -Mstrict x.pl", "mv a.py app.py", "prettier -c app.js"])
def test_checks_and_renames_are_not_rewrites(cmd):
    assert C._inplace_written(cmd) == []


@pytest.mark.parametrize("cmd,f", [
    ("python3 -c \"import subprocess; subprocess.run('python3 app.py', shell=True)\"", "app.py"),
    ("python3 -c \"import os; os.system('python3 app.py')\"", "app.py"),
    ("node -e \"require('child_process').execSync('node app.js')\"", "app.js"),
    ("python3 -c \"import subprocess,sys; subprocess.check_call([sys.executable,'app.py'])\"", "app.py"),
    ("python3 -c \"from pathlib import Path; exec(Path('app.py').read_text())\"", "app.py"),
])
def test_inline_code_that_runs_a_file_runs_it(cmd, f):
    assert C.unbacked_code_writes([_w(f), _x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd", [
    "python3 - <<'EOF'\nopen('gen.py','w').write('print(1)')\nimport subprocess; subprocess.run(['python3','gen.py'])\nEOF",
    "python3 - <<'EOF'\nopen('helper.py','w').write('x=1')\nimport helper\nEOF",
])
def test_inline_code_that_writes_then_runs_a_file(cmd):
    assert C.unbacked_code_writes([_x(cmd)]) == ([], [])


@pytest.mark.parametrize("cmd,folders", [
    ("cd app && timeout 60 python3 tests/check_page.py", ["app"]),
    ("cd app && python3 -m pytest && python3 tests/check_page.py", ["app"]),
    ("cd 'my app' && python3 tests/check_page.py x", ["my app"]),
    ("cd app && python3 tests/check_page.py || true", []), ("cd app; python3 tests/check_page.py; true", []),
])
def test_check_page_folders(cmd, folders):
    from ghost_agent.core.task_state import _check_page_folders
    assert _check_page_folders(cmd) == folders


@pytest.mark.parametrize("cmd,ok", [
    ("curl -fsS localhost:8000 && echo ok", True), ("curl -f localhost:8080 | true", False),
    ("echo curl -f localhost", False), ("curl --fail-early -s localhost:3000", False),
    ("curl -s -d '-f x' localhost:3000", False),
])
def test_service_check_reads_the_curls_own_flags(cmd, ok):
    assert K.runs_something(cmd) is ok


@pytest.mark.parametrize("check", ["pytest -q || true", "python3 t.py; exit 0", "curl -s localhost:3000 | grep ok || :"])
def test_a_check_that_can_never_fail_is_refused(check):
    assert "always exits 0" in K._coerce_items([{"title": "a", "check": check}])[1]


@pytest.mark.parametrize("reply,done", [
    ("It's done — no remaining issues", True), ("Done! All tests pass, 0 failed", True),
    ("It's complete; there are no failing tests", True), ("Everything is done. Earlier the score test failed, I fixed it.", True),
    ("I finished the login page. Next I'll build the cart", False), ("Done so far:\n- login\nNext: cart", False),
    ("I completed step 1 of 3", False), ("The server is ready for you to add features", False),
])
def test_finished_versus_progress(reply, done):
    assert K.presents_as_finished(reply, C.claims_code_works) is done


@pytest.mark.parametrize("cmd,progs", [
    ("sudo -u pg bash <<EOF\napt-get x\nEOF", ["sudo", "apt-get"]), ("bash script.sh <<EOF\nsudo y\nEOF", ["bash"]),
    ("capsh --user=root --print -- -c id", ["capsh"]),
])
def test_heredoc_shell_and_capsh_edges(cmd, progs):
    assert E.programs_in(cmd) == progs


async def test_a_page_checked_css_write_resets_only_web_files():
    import tests.test_4jj_search_yield_steer as H
    from unittest.mock import MagicMock
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()

    async def fs(**kw):
        p = kw.get("path")
        r = f"SUCCESS: Edited '{p}' (1 occurrence)."
        if p.endswith(".css"):
            r += "\n\n[PAGE CHECK] ✓ loaded 'index.html' headless: no uncaught JS exception. Ran: index.html."
        return r
    agent.available_tools = {"file_system": fs}
    rs, strikes = MagicMock(), StrikeLedger()
    rs._edit_churn = {"counts": {"server.py": 3, "app.js": 3}, "steers": 0}
    ts = H._ts([("file_system", {"operation": "edit", "path": "style.css", "old_string": "a", "new_string": "b"})],
               strikes, set())
    ts.request_state = rs
    await agent._dispatch_and_process_tool_batch(ts)
    assert rs._edit_churn["counts"] == {"server.py": 3}


# ── fourth verification round ──

@pytest.mark.parametrize("reply", [
    "The app is ready for you to try at localhost:5000.", "It's all done. Now I'll walk you through what changed:",
    "All done. Here is what works so far:", "It's done.\nNext: open the page and click Start.",
])
def test_a_done_claim_with_a_walkthrough_is_still_done(reply):
    assert K.presents_as_finished(reply, C.claims_code_works) is True


@pytest.mark.parametrize("cmd,folders", [
    ("(cd app && python3 tests/check_page.py x)", ["app"]), ("bash -c 'cd app && python3 tests/check_page.py'", ["app"]),
    ("cd app && python3 tests/check_page.py x; echo done", []), ("cd app && python3 tests/check_page.py | tee log", []),
    ("cd app && python3 tests/check_page.py &", []), ("cd app && python3 tests/check_page.py && echo PASS", ["app"]),
])
def test_only_a_check_whose_exit_is_the_commands_counts(cmd, folders):
    from ghost_agent.core.task_state import _check_page_folders
    assert _check_page_folders(cmd) == folders


@pytest.mark.parametrize("check,refused", [
    ("python3 t.py && true", False), ("python3 t.py && :", False), ("pytest -q || true; pytest", False),
    ("curl -s localhost | grep -q ok || echo no", True),
])
def test_swallow_is_the_last_word(check, refused):
    assert ("always exits 0" in K._coerce_items([{"title": "a", "check": check}])[1]) is refused


def test_a_command_string_only_logged_is_not_a_run():
    cmd = "python3 -c \"import logging; logging.info('python3 app.py ready')\""
    assert C.unbacked_code_writes([_w("app.py"), _x(cmd)])[0] == ["app.py"]


@pytest.mark.parametrize("cmd", ["black -l 100 app.py", "isort -l 100 app.py", "eslint -c .eslintrc.json --fix app.js",
                                 "prettier --write -c app.js"])
def test_option_letters_are_per_formatter(cmd):
    assert C._inplace_written(cmd) != []


def test_a_versioned_missing_script_fails_the_load():
    from ghost_agent.core.agent import _local_load_failures
    text = "CONSOLE (1):\n  • [error] file:///w/js/app.js?v=2:1:1 — Failed to load resource: net::ERR_FILE_NOT_FOUND"
    assert _local_load_failures(text)


@pytest.mark.parametrize("cmd,progs", [
    ("bash > out.log <<EOF\nsudo x\nEOF", ["bash", "sudo"]), ("sudo -iu pg bash <<EOF\napt-get x\nEOF", ["sudo", "apt-get"]),
    ("bash -s arg <<EOF\nsu z\nEOF", ["bash", "su"]),
])
def test_shell_fed_heredoc_edges_round4(cmd, progs):
    assert E.programs_in(cmd) == progs



# ── battery 12 survivors ──

def test_inline_code_that_rewrites_a_run_file_makes_it_unrun_again():
    rows = [_w("app.py"), _x("python3 app.py"), _x("python3 -c \"open('app.py','w').write('print(2)')\"")]
    assert C.unbacked_code_writes(rows)[0] == ["app.py"]


async def test_a_page_whose_script_never_loaded_fails_the_page_check(tmp_path, monkeypatch):
    import json
    from tests.test_4ks_log_review_fixes import _web_agent

    def missing_script(url):
        d = {"status": 200, "url": url, "title": "t", "text": "body", "length": 4,
             "console": [{"type": "error", "loc": "file:///workspace/templates/js/app.js",
                          "text": "Failed to load resource: net::ERR_FILE_NOT_FOUND"}]}
        return "[BROWSER_OK] " + json.dumps(d) + "\n", 0
    agent, _ = _web_agent(tmp_path, monkeypatch, {"*": missing_script})
    page_rel, block = await agent._execute_web_artifact(["templates/index.html"])
    assert page_rel == "templates/index.html" and block.startswith("LOCAL FILES THAT DID NOT LOAD")


# ── live functional findings ──

async def test_a_script_outside_any_page_folder_does_not_borrow_another_page(tmp_path, monkeypatch):
    from tests.test_4ks_log_review_fixes import _web_agent, _ok
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)})       # writes a fresh templates/index.html
    assert await agent._execute_web_artifact(["hello/hello.js"]) is None and visited == []
    page_rel, block = await agent._execute_web_artifact(["templates/js/app.js"])
    assert page_rel == "templates/index.html" and block == ""


def test_every_bad_checklist_item_is_named_at_once():
    _, why = K._coerce_items([{"title": "a", "check": "ls x"}, {"title": "b", "check": "cat y"},
                              {"title": "c", "check": "python3 t.py"}])
    assert why.startswith("2 items need a different check") and "'a'" in why and "'b'" in why and "file:<path>" in why


@pytest.mark.parametrize("cmd", ["iptables-legacy -A INPUT -j DROP", "/usr/sbin/ip6tables-nft-restore < r"])
def test_firewall_variants_are_the_same_privilege(cmd):
    assert E.programs_in(cmd)[0] in ("iptables", "ip6tables")


async def test_a_repeated_operation_check_runs_again():
    """`operation=check` is a run like `action=check`: the same check twice (after a fix) is not a duplicate."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    ran = []

    async def checklist(**kw):
        ran.append(kw)
        return "Checklist: 1/1 passed.\n✓ #1 a — check: python3 t.py"
    agent.available_tools = {"checklist": checklist}
    seen, strikes = set(), StrikeLedger()
    for _ in range(3):              # the same check after each fix: a run, never the no-progress read-repeat stop
        ts = H._ts([("checklist", {"operation": "check"})], strikes, seen)
        await agent._dispatch_and_process_tool_batch(ts)
        assert not ts.force_final_response
    assert len(ran) == 3
