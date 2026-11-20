"""§4UY (code subsystem, C2/C3/C4/C20) — the "was it run?" ledger and the claim detector.

§4UX-2 found the two masking each other: the ledger never credited the `run` tool (where the foreground-server
block sends servers), `cargo run`, `make && ./app`, `node .`, `python3 -m pkg`, `bash run.sh` — and it credited
compiles, `--help`, a suite in another folder, a script whose text merely named a file, `|| true`, and keyed
files by BASENAME (`backend/index.js` backed `frontend/index.js`). The claim detector missed the model's real
words ("All 12 tests pass", "verified", "confirmed it's working", "live-tested"). A clean page load credited every
written script.

Every row below fails in the world before the fix (checked against the pre-§4UY module), except the rows marked
"guard", which pin the new rule's own limits (a broader detector must not swallow honest reporting).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import code_claims as C
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome


def _w(path, content=None):
    args = {"operation": "write", "path": path}
    if content is not None:
        args["content"] = content
    return {"name": "file_system", "arguments": args,
            "content": f"SUCCESS: Wrote 10 chars to '{path}'. Script-side path (from sandbox cwd): '{path}'."}


def _x(cmd=None, code=0, *, filename=None, content=None):
    args = {k: v for k, v in (("command", cmd), ("filename", filename), ("content", content)) if v is not None}
    return {"name": "execute", "arguments": args,
            "content": ToolOutcome(f"--- COMMAND RESULT ---\nEXIT CODE: {code}\nSTDOUT/STDERR:\nok",
                                   status=OutcomeStatus.OK if code == 0 else OutcomeStatus.FAILED, call_args=args)}


def _run(action, name, command=None, *, status=OutcomeStatus.OK, text="", lifetime="server", where="workspace"):
    args = {"action": action, "name": name, "where": where, "lifetime": lifetime}
    if command:
        args["command"] = command
    return {"name": "run", "arguments": args, "content": ToolOutcome(text or f"[run {name}]", status=status,
                                                                     call_args=args)}


def _b(url, ok=True, extra=""):
    return {"name": "browser", "arguments": {"operation": "navigate", "url": url},
            "content": "--- BROWSER RESULT ---\nSTATUS: " + ("OK" if ok else "FAILED") + "\n" + extra}


# ── C2: real runs are credited ────────────────────────────────────────

@pytest.mark.parametrize("rows", [
    # the `run` tool: a started server ran its file
    [_w("app.py"), _run("start", "web", "python3 app.py",
                        text="[run web started — server in the workspace, port 8000, r1]")],
    # a job, collected green
    [_w("app.py"), _run("start", "j", "python3 app.py", lifetime="job"),
     _run("collect", "j", text="[run j exited exit=0]\nEXIT CODE: 0\nok", lifetime="job")],
    [_w("src/main.rs"), _x("cargo run")],
    [_w("main.c"), _x("gcc -o app main.c && ./app")],
    [_w("index.js"), _x("node .")],
    [_w("pkg/__main__.py"), _w("pkg/core.py"), _x("python3 -m pkg")],
    [_w("main.go"), _w("util.go"), _x("go run .")],
    [_w("server/app.js"), _x("npm start")],
    # a shell script the request wrote: what its text runs
    [_w("run.sh", "#!/bin/sh\npython3 app.py\n"), _w("app.py"), _x("bash run.sh")],
    [_x("cat > run.sh <<'EOF'\nnode app.js\nEOF"), _w("app.js"), _x("./run.sh")],
])
def test_c2_real_runs_are_credited(rows):
    assert C.unbacked_code_writes(rows) == ([], [])


def test_c2_run_tool_job_failure_and_pending_job():
    start = _run("start", "j", "python3 app.py", lifetime="job")
    red = _run("collect", "j", status=OutcomeStatus.FAILED, text="[run j exited exit=1]\nEXIT CODE: 1\nboom",
               lifetime="job")
    assert C.unbacked_code_writes([_w("app.py"), start, red]) == ([], ["app.py"])
    assert C.unbacked_code_writes([_w("app.py"), start]) == (["app.py"], [])          # started, no result yet
    # a run on a lab machine is not a run of the workspace file
    lab = _run("start", "l", "python3 app.py", where="lab_0123abcd/db",
               text="[run l started — server on lab]")
    assert C.unbacked_code_writes([_w("app.py"), lab]) == (["app.py"], [])


# ── C3: what is NOT a run ─────────────────────────────────────────────

@pytest.mark.parametrize("rows,never", [
    ([_w("main.c"), _x("gcc -o app main.c")], ["main.c"]),                 # compile only
    ([_w("Main.java"), _x("javac Main.java")], ["Main.java"]),
    ([_w("main.go"), _x("go build main.go")], ["main.go"]),
    ([_w("app.py"), _x("python3 app.py --help")], ["app.py"]),            # usage, not a run
    ([_w("cli.js"), _x("node cli.js --version")], ["cli.js"]),
    ([_w("backend/app.py"), _x("cd frontend && npm test")], ["backend/app.py"]),   # a suite elsewhere
    ([_w("backend/app.py"), _x("cd frontend && python3 -m pytest -q")], ["backend/app.py"]),
    ([_w("app.py"), _x(filename="check.sh", content="echo app.py looks fine\n")], ["app.py"]),   # names it only
    ([_w("app.py"), _x(filename="t.py", content="print('see app.py')\n")], ["app.py"]),
    ([_w("app.py"), _x("python3 -m pytest -q || true")], ["app.py"]),    # masked status
    ([_w("app.py"), _x("python3 app.py || echo failed")], ["app.py"]),
    ([_w("frontend/index.js"), _x("node backend/index.js")], ["frontend/index.js"]),   # path, not basename
    ([_w("frontend/index.js"), _x("cd backend && node index.js")], ["frontend/index.js"]),
    ([_w("app.py"), _x("cat > test_x.py <<'EOF'\nimport app\nEOF")], ["app.py", "test_x.py"]),   # body = data
])
def test_c3_not_runs(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


@pytest.mark.parametrize("rows", [
    [_w("frontend/index.js"), _x("cd frontend && node index.js")],
    [_w("frontend/index.js"), _x("node /workspace/projects/0123456789ab/frontend/index.js")],
    [_w("projects/0123456789ab/app.py"), _x("python3 app.py")],
    [_w("backend/app.py"), _x("cd backend && python3 -m pytest -q")],
    [_w("app.py"), _x("python3 -m pytest -q || true"), _x("python3 -m pytest -q")],
    [_w("app.py"), _x("python3 app.py || python3 app.py --retry")],       # the fallback's status IS the run's
    [_w("app.py"), _x(filename="check.sh", content="python3 app.py\n")],
    [_w("app.py"), _x(filename="t.py", content="import app\napp.main()\n")],
    [_w("app.py"), _x("cd $APPDIR && python3 app.py")],                   # unknowable cd: suffix match
], ids=lambda r: str(r[-1]["arguments"])[:60])
def test_c3_guard_paths_that_are_runs(rows):
    """guard: the path keying and masking must not lose real runs."""
    assert C.unbacked_code_writes(rows) == ([], [])


def test_c3_compile_then_run_is_a_run():
    """guard: compiling is a check, but the binary it built then RAN."""
    assert C.unbacked_code_writes([_w("main.c"), _x("gcc -o app main.c && ./app")]) == ([], [])


def test_c3_segment_kinds():
    assert C.segment_kind(["gcc", "-o", "app", "main.c"]) == "check"
    assert C.segment_kind(["cargo", "build"]) == "check"
    assert C.segment_kind(["cargo", "run"]) == "run"
    assert C.segment_kind(["make"]) == "check"
    assert C.segment_kind(["make", "test"]) == "run"
    assert C.segment_kind(["python3", "app.py", "--help"]) == "read"
    assert C.segment_kind(["python3", "-m", "pytest", "-v"]) == "run"        # guard: -v is verbose here


# ── C4: the claim detector reads the model's words ─────────────────────

@pytest.mark.parametrize("text", [
    "12/12 tests green.",
    "Verified.",
    "I confirmed it's working.",
    "Live-tested in the browser: the board renders.",
    "Smoke-tested the CLI.",
    "0 failed, 12 passed",
    "The build succeeded and the page is green.",
])
def test_c4_claims_matched(text):
    assert C.claims_code_works(text)


@pytest.mark.parametrize("text", [
    "I ran the tests and 2 of 5 failed — the parser needs a fix.",
    "Verified that it still fails on empty input.",
    "2 failed, 10 passed",
    "Run it from the working directory.",
    "You pass the path as the first argument.",
    "This works by splitting the input on commas.",
    "I haven't verified it yet.",
    "Once it is tested, deploy it.",
])
def test_c4_guard_honest_and_conditional_text_is_not_a_claim(text):
    assert not C.claims_code_works(text)


def test_c4_and_c2_together_probe_shape():
    """probe-75a957e4: the last run was a 600 s kill, the reply said "verified working" — the note names it."""
    rows = [_w("server.py"), _x("timeout 600 python3 server.py", code=124)]
    note, failed = C.unbacked_claim_note("Done — verified working.", rows)
    assert failed and "server.py" in note
    rows_ok = [_w("server.py"), _run("start", "srv", "python3 server.py",
                                     text="[run srv started — server in the workspace, port 8000, r1]")]
    assert C.unbacked_claim_note("All 12 tests pass.", rows_ok) == ("", False)


# ── C20: a page load runs only the scripts the page loads ─────────────

def test_c20_clean_load_credits_only_loaded_scripts():
    page = _w("web/index.html", '<html><script src="js/a.js"></script></html>')
    rows = [page, _w("web/js/a.js"), _w("web/js/b.js"), _b("file:///workspace/web/index.html")]
    assert C.unbacked_code_writes(rows)[0] == ["web/js/b.js"]


def test_c20_served_page_resolves_root_relative_srcs():
    page = _w("web/index.html", '<script src="/static/app.js?v=2"></script>')
    rows = [page, _w("web/static/app.js"), _w("web/static/admin.js"), _b("http://localhost:8000/")]
    assert C.unbacked_code_writes(rows)[0] == ["web/static/admin.js"]


def test_c20_unknown_page_text_credits_no_script():
    rows = [_w("web/index.html"), _w("web/js/a.js"), _b("file:///workspace/web/index.html")]
    assert C.unbacked_code_writes(rows)[0] == ["web/js/a.js"]


def test_c20_edit_keeps_the_page_text_current():
    page = _w("index.html", '<script src="old.js"></script>')
    edit = {"name": "file_system",
            "arguments": {"operation": "edit", "path": "index.html", "old_string": "old.js", "new_string": "new.js"},
            "content": "SUCCESS: edited — replaced 1 occurrence of old_string (line 1) in 'index.html'."}
    rows = [page, edit, _w("old.js"), _w("new.js"), _b("file:///workspace/index.html")]
    assert C.unbacked_code_writes(rows)[0] == ["old.js"]


def test_c20_another_host_is_not_our_page():
    rows = [_w("index.html", "<p>x</p>"), _b("https://example.org/index.html")]
    assert C.unbacked_code_writes(rows)[0] == ["index.html"]


# ── r2 (independent review): a folder only for a binary this request built; -m against PYTHONPATH / src;
#    a suite in its tests folder tests the project; the claim rule is about code that ran ────────────────

@pytest.mark.parametrize("rows,never", [
    ([_w("server.py"), _w("util.py"), _x(".venv/bin/pip install -r requirements.txt")], ["server.py", "util.py"]),
    ([_w("server.py"), _x("node_modules/.bin/vite build")], ["server.py"]),
    ([_w("calc.py"), _x("./configure --prefix=/tmp")], ["calc.py"]),
    ([_w("main.c"), _x("./app")], ["main.c"]),                           # no build this request: not its binary
    ([_w("main.c"), _x("cd other && make"), _x("./app")], ["main.c"]),     # built elsewhere
    # a build here, then a tool that is no binary of it
    ([_w("main.c"), _x("gcc -o app main.c"), _x(".venv/bin/pip install -r requirements.txt")], ["main.c"]),
    ([_w("main.c"), _x("gcc -o app main.c"), _x("./configure --prefix=/tmp")], ["main.c"]),
])
def test_r2_only_a_built_binary_credits_its_folder(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


@pytest.mark.parametrize("rows", [
    [_w("main.c"), _x("gcc -o app main.c"), _x("./app")],                  # built in an earlier call
    [_w("main.c"), _x("gcc -o bin/app main.c && ./bin/app")],
    [_w("lib/mypkg/cli.py"), _x("PYTHONPATH=lib python -m mypkg.cli")],         # (not src/: that base is implicit)
    [_w("lib/mypkg/cli.py"), _x("export PYTHONPATH=lib; python3 -m mypkg.cli")],
    [_w("lib/mypkg/cli.py"), _x("PYTHONPATH=/tmp:lib python3 -m mypkg.cli")],
    [_w("c/main.c"), _x("gcc -o c/app c/main.c && c/app")],               # built in a subfolder, run from root
    [_w("c/main.c"), _x("cd c && gcc -o app main.c"), _x("c/app")],         # built IN the subfolder, run from root
    [_w("main.c"), _x("bash -c 'gcc -o app main.c && ./app'")],            # the build + run inside bash -c
    [_w("src/mypkg/cli.py"), _x("python3 -m mypkg.cli")],                  # src/ layout
    [_w("app/models.py"), _x("cd tests && python -m pytest -q")],          # the suite tests the project above
    [_w("pkg/__init__.py"), _x(filename="t.py", content="import pkg\npkg.main()\n")],   # a package import
    [_x("cd backend && cat > app.py <<'EOF'\nprint(1)\nEOF"), _x("python3 backend/app.py")],   # the write's cd
])
def test_r2_runs_are_credited(rows):
    assert C.unbacked_code_writes(rows) == ([], [])


@pytest.mark.parametrize("rows,never", [
    ([_w("app.py"), _x("cat > notes.sh <<'EOF'\npython3 app.py\nEOF")], ["app.py", "notes.sh"]),   # body = data
    ([_w("src/app.js"), _x("npm run build")], ["src/app.js"]),             # a build script is no run
    ([_w("src/app.js"), _x("yarn run lint")], ["src/app.js"]),
])
def test_r2_not_runs(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


def test_r2_a_lab_job_collected_later_is_not_a_workspace_run():
    start = _run("start", "j", "python3 app.py", lifetime="job", where="lab_0123abcd/db")
    collect = {"name": "run", "arguments": {"action": "collect", "name": "j"},
               "content": ToolOutcome("[run j exited exit=0]\nEXIT CODE: 0\nok", status=OutcomeStatus.OK,
                                      call_args={"action": "collect", "name": "j"})}
    assert C.unbacked_code_writes([_w("app.py"), start, collect]) == (["app.py"], [])


@pytest.mark.parametrize("text", [
    "I've tested nothing yet.",
    "I verified the config file was saved.",
    "Confirmed: the file is at app.py.",
    "Validated the JSON schema manually.",
])
def test_r2_not_a_run_claim(text):
    assert not C.claims_code_works(text)


@pytest.mark.parametrize("text", ["Verified the app works end to end.", "Done — verified working.",
                                  "I confirmed it runs."])
def test_r2_guard_code_verification_still_claims(text):
    """guard: verify words about the code or its run stay claims."""
    assert C.claims_code_works(text)


# ── r3 (round-2 review): M3 pages read from the workspace, M4 file_system paths, M5 binaries / java /
#    importlib, m2 tools that run none of the request's code ──────────────

def _ed(path):
    return {"name": "file_system", "arguments": {"operation": "edit", "path": path, "old_string": "a",
                                                 "new_string": "b"},
            "content": f"SUCCESS: edited — replaced 1 occurrence of old_string (line 1) in '{path}'."}


def _inter(url, ok=True, extra=""):
    return {"name": "browser", "arguments": {"operation": "interact", "actions": [
                {"action": "goto", "url": url}, {"action": "evaluate", "js": "OS.openApp('notes')"}]},
            "content": "--- BROWSER RESULT ---\nSTATUS: " + ("OK" if ok else "FAILED") + "\nOP: interact\n"
                       "ACTIONS: 2 OK, 0 errors (of 2 total)\n" + extra}


def _sandbox(tmp_path, monkeypatch, files):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    for rel, text in files.items():
        f = tmp_path / "sandbox" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


_PAGE = '<html><script src="js/apps/notes.js"></script></html>'


@pytest.mark.parametrize("load", [_inter("http://127.0.0.1:8100/"), _b("http://localhost:8100/index.html")])
def test_r3_m3_served_page_not_written_credits_its_scripts_from_the_workspace(tmp_path, monkeypatch, load):
    _sandbox(tmp_path, monkeypatch, {"index.html": _PAGE})
    assert C.unbacked_code_writes([_ed("js/apps/notes.js"), load]) == ([], [])


def test_r3_m3_project_page_found_through_the_project_folder_the_request_named(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch, {"projects/0123456789ab/index.html": _PAGE})
    rows = [_w("projects/0123456789ab/js/apps/notes.js"), _inter("http://127.0.0.1:8100/")]
    assert C.unbacked_code_writes(rows) == ([], [])


def test_r3_m3_project_page_found_through_the_request_scope(tmp_path, monkeypatch):
    from ghost_agent.core import request_scope as RS
    _sandbox(tmp_path, monkeypatch, {"projects/0123456789ab/index.html": _PAGE})
    tok = RS.install(RS.RequestScope(project_id="0123456789ab"))
    try:
        assert C.unbacked_code_writes([_ed("js/apps/notes.js"), _inter("http://127.0.0.1:8100/")]) == ([], [])
    finally:
        RS.reset(tok)


def test_r3_m3_served_from_a_subfolder(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch, {"webos/index.html": '<script src="apps/calc.js"></script>'})
    assert C.unbacked_code_writes([_ed("webos/apps/calc.js"), _inter("http://127.0.0.1:8100/")]) == ([], [])


@pytest.mark.parametrize("files,load", [
    ({"index.html": '<script src="js/other.js"></script>'}, _inter("http://127.0.0.1:8100/")),  # not loaded
    ({}, _inter("http://127.0.0.1:8100/")),                                                     # no page
    ({"index.html": _PAGE}, _inter("http://127.0.0.1:8100/", ok=False)),                        # failed load
    ({"index.html": _PAGE}, _inter("http://127.0.0.1:8100/", extra="UNCAUGHT JS EXCEPTIONS:\nx")),  # threw
    ({"index.html": _PAGE}, _inter("https://example.org/")),                                    # not our host
])
def test_r3_m3_guard_page_scripts_need_a_clean_load_of_a_page_that_loads_them(tmp_path, monkeypatch, files,
                                                                             load):
    _sandbox(tmp_path, monkeypatch, files)
    assert C.unbacked_code_writes([_ed("js/apps/notes.js"), load])[0] == ["js/apps/notes.js"]


def test_r3_m3_guard_a_page_key_never_leaves_the_workspace(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch, {})
    (tmp_path / "index.html").write_text(_PAGE)               # beside the sandbox, not in it
    rows = [_ed("js/apps/notes.js"), _b("file:///workspace/../index.html")]
    assert C.unbacked_code_writes(rows)[0] == ["js/apps/notes.js"]


@pytest.mark.parametrize("rows", [
    [_w("/webos/index.html", "<html><script>1</script></html>"), _b("file:///workspace/webos/index.html")],
    [_w("/tools/gen.py"), _x("python3 tools/gen.py")],
    [_w("/workspace/tools/gen.py"), _x("python3 tools/gen.py")],
])
def test_r3_m4_leading_slash_file_system_path_is_workspace_relative(rows):
    assert C.unbacked_code_writes(rows) == ([], [])


def test_r3_m4_keys():
    assert C._fs_norm("/webos/index.html") == "webos/index.html"
    assert C._fs_norm("/etc/profile.d/zz.sh") == "etc/profile.d/zz.sh"
    assert C._fs_norm("/Users/elena/ghost/sandbox/projects/0123456789ab/js/a.js") == "js/a.js"


@pytest.mark.parametrize("rows", [
    [_w("x.cpp"), _x("clang++ x.cpp && ./a.out")],
    [_w("main.c"), _x("gcc -o x.bin main.c"), _x("./x.bin")],
    [_w("main.c"), _x("gcc -o app.exe main.c && ./app.exe")],
    [_w("Main.java"), _x("javac Main.java && java Main")],
    [_w("src/com/kosmas/App.java"), _x("cd src && javac com/kosmas/App.java && java com.kosmas.App")],
    [_w("Main.java"), _x("javac Main.java && java -cp . Main")],
    [_w("svc/app.py"), _x(content='import importlib.util\nspec = importlib.util.spec_from_file_location("a", '
                                  '"svc/app.py")\nm = importlib.util.module_from_spec(spec)\n'
                                  'spec.loader.exec_module(m)\n')],
    [_w("svc/app.py"), _x(content='from importlib.machinery import SourceFileLoader\n'
                                  'SourceFileLoader("a", "svc/app.py").load_module()\n')],
])
def test_r3_m5_runs_are_credited(rows):
    assert C.unbacked_code_writes(rows) == ([], [])


@pytest.mark.parametrize("rows,never", [
    ([_w("main.c"), _x("./a.out")], ["main.c"]),                                # no build in this request
    ([_w("main.c"), _x("gcc -o x.bin main.c"), _w("main.c"), _x("./x.bin")], ["main.c"]),  # edited after
    ([_w("Main.java"), _w("Other.java"), _x("javac *.java && java Main")], ["Other.java"]),
    ([_w("Main.java"), _x("java -jar app.jar")], ["Main.java"]),
])
def test_r3_m5_guard_not_runs(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


@pytest.mark.parametrize("cmd", [
    "python3 -m pip install -r requirements.txt", "python -m venv .venv", "python3 -m ensurepip",
    "python3 -m http.server 8000", "make clean", "npx kill-port 8000", "cargo clean",
])
def test_r3_m2_tools_are_not_runs(cmd):
    assert C.segment_kind(cmd.split()) == "read"
    assert C.unbacked_code_writes([_w("app.py"), _w("src/main.rs"), _x(cmd)])[0] == ["app.py", "src/main.rs"]


@pytest.mark.parametrize("cmd", ["python3 -m pkg.cli", "make test", "npx jest", "python3 -m pipeline"])
def test_r3_m2_guard_runners_by_module_stay_runs(cmd):
    assert C.segment_kind(cmd.split()) == "run"


# ── r3 (live replay): the false notes the 211-row replay showed ─────────

@pytest.mark.parametrize("out", [
    '127.0.0.1 - - [06/Nov/2026 18:35:21] "GET /apps/notes.js HTTP/1.1" 200 -',   # a server log in a run's output
    '"GET /apps/notes.js HTTP/1.1" 304 -',
])
def test_r4_an_access_log_line_credits_nothing(out):
    """r4 (review, conservative credit): a log line cannot say which server, which run, or which version — a
    `cat old_access.log` printed the same line. The §4UY r3 access-log credit is removed."""
    for cmd in ("cat old_access.log", "python3 drive.py"):
        run = _x(cmd)
        run["content"] = ToolOutcome("--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n" + out,
                                     status=OutcomeStatus.OK, call_args={"command": cmd})
        assert C.unbacked_code_writes([_w("webos/apps/notes.js"), run])[0] == ["webos/apps/notes.js"]


@pytest.mark.parametrize("text", [
    'import subprocess\nsubprocess.run(["bash", "ft_kosmas/backup.sh", "a", "b"], check=True)\n',
    'import subprocess, sys\nsubprocess.run([sys.executable, "ft_kosmas/backup.py"])\n',
    'import os\nos.system("bash ft_kosmas/backup.sh src dst")\n',
])
def test_r4_a_harness_credits_an_interpreter_plus_path_in_the_same_call(text):
    rows = [_w("ft_kosmas/backup.sh"), _w("ft_kosmas/backup.py"), _w("ft_kosmas/t.py", text),
            _x("python3 ft_kosmas/t.py")]
    never = C.unbacked_code_writes(rows)[0]
    assert len(never) == 1 and never[0] in ("ft_kosmas/backup.sh", "ft_kosmas/backup.py")


@pytest.mark.parametrize("text", [
    # rev3 repros: a program that quotes a script name it does not run
    "import subprocess\nsubprocess.run(['tar', 'czf', 'b.tgz', 'ft_kosmas/backup.sh'])\n",
    "import os\nos.system('echo deploying')\nFILES = ['ft_kosmas/backup.sh']\n",
    # a path built at run time is unknowable: no credit (when unsure, none)
    'import os, subprocess\nHERE = os.path.dirname(__file__)\n'
    'subprocess.run(["bash", os.path.join(HERE, "backup.sh")])\n',
    'import os\nprint(open(os.path.join("ft_kosmas", "backup.sh")).read())\n',
    # no interpreter in the call: not credited (the coordinator's conservative rule)
    'exec(open("ft_kosmas/backup.sh").read())\n',
])
def test_r4_guard_a_quoted_script_name_is_not_a_run(text):
    rows = [_w("ft_kosmas/backup.sh"), _w("ft_kosmas/t.py", text), _x("python3 ft_kosmas/t.py")]
    assert C.unbacked_code_writes(rows)[0] == ["ft_kosmas/backup.sh"]


@pytest.mark.parametrize("rows,never", [
    # rev3 repros: a build credits the compiled sources of its folder only
    ([_w("app.c"), _w("tool.py"), _x("gcc -o app app.c && ./app")], ["tool.py"]),
    ([_w("c/app.c"), _w("c/gen.py"), _x("cd c && gcc -o app app.c && ./app")], ["c/gen.py"]),
    ([_w("main.go"), _w("web/app.js"), _w("run.sh"), _x("go build -o app . && ./app")], ["web/app.js", "run.sh"]),
])
def test_r4_a_built_binary_credits_compiled_sources_only(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


def test_r3_a_command_the_validator_refused_is_no_run():
    """live 2b1e6eb4: a SYSTEM BLOCK 'was not run' — it is not a failed run of the script."""
    blocked = _x("bash ft_kosmas/backup.sh src dst")
    blocked["content"] = ("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nSYSTEM BLOCK: shell command "
                          "rejected by pre-execution validator: deny-listed pattern. The command was not run.")
    assert C.unbacked_code_writes([_w("ft_kosmas/backup.sh"), blocked]) == (["ft_kosmas/backup.sh"], [])
    rej = _x("bash ft_kosmas/backup.sh")
    rej["content"] = ToolOutcome("refused", status=OutcomeStatus.REJECTED, call_args={})
    assert C.code_evidence([_w("ft_kosmas/backup.sh"), rej])["ft_kosmas/backup.sh"]["run_idx"] is None


def test_r3_guard_a_real_failure_stays_a_failure():
    assert C.unbacked_code_writes([_w("app.py"), _x("python3 app.py", 1)]) == ([], ["app.py"])


@pytest.mark.parametrize("cmd", ["printf 'export PATH=/workspace/.bin:$PATH\\n' > /etc/profile.d/zz.sh",
                                 "echo 'x' > /usr/local/lib/site.py"])
def test_r3_a_file_written_into_the_machines_own_folders_is_config(cmd):
    """live 6aee2a43: /etc/profile.d/*.sh configures login shells — not the request's program."""
    assert C.code_evidence([_x(cmd)]) == {}


def test_r3_guard_tmp_and_workspace_scripts_stay_tracked():
    ev = C.code_evidence([_x("cat > /tmp/job.sh <<'EOF'\necho hi\nEOF\ncat > run.sh <<'EOF'\necho hi\nEOF")])
    assert set(ev) == {"/tmp/job.sh", "run.sh"}


def _partial(*lines):
    return ("[FAILURE BANNER] --- BROWSER RESULT ---\n--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 3 actions "
            "failed)\nOP: interact\n--- PER-ACTION RESULTS ---\n" + "\n".join(lines))


def test_r3_a_stored_partial_whose_only_error_is_the_models_evaluate_is_a_load():
    """live fb512751: browser.py §4RJ now books this OK; a stored older row says PARTIAL."""
    t = {"name": "browser", "content": _partial("  [0] OK goto → http://127.0.0.1:8100/",
                                                "  [1] ERR evaluate: Error: SyntaxError: Unexpected token ':'")}
    assert C._browser_ok(t) is True


@pytest.mark.parametrize("lines", [
    ("  [0] OK goto → http://127.0.0.1:8100/", "  [1] ERR click: timeout"),
    ("  [0] OK goto → http://127.0.0.1:8100/", "  [1] ERR evaluate: evaluate timed out after 5000 ms"),
    ("  [0] ERR evaluate: x",),
])
def test_r3_guard_other_partials_stay_failures(lines):
    assert C._browser_ok({"name": "browser", "content": _partial(*lines)}) is False


def test_r3_an_appended_page_keeps_the_scripts_of_its_first_part():
    """live fb512751: a page written in parts (`append=true`) — its text is every part, not the last."""
    first = _w("web/index.html", '<html><script src="app.js"></script>')
    more = {"name": "file_system", "arguments": {"operation": "write", "path": "web/index.html", "append": True,
                                                 "content": "<p>more</p></html>"},
            "content": "SUCCESS: Appended 18 chars to 'web/index.html'. Script-side path (from sandbox cwd): "
                       "'web/index.html'."}
    rows = [_w("web/app.js"), first, more, _b("file:///workspace/web/index.html")]
    assert C.unbacked_code_writes(rows) == ([], [])


@pytest.mark.parametrize("text", ["NebulaOS is live! It is fully functional.", "Fully functional now.",
                                  "Everything is verified and the Slack notification is already sent.",
                                  "The dashboard has been tested.", "The app is working."])
def test_r3_the_pre_4uy_claim_forms_are_claims(text):
    """replay 598156cf / 945b9726: the earlier detector's forms — losing them dropped two true notes."""
    assert C.claims_code_works(text)


@pytest.mark.parametrize("text", ["It is not fully functional yet.", "If everything is verified, ship it.",
                                  "Nothing is verified yet.", "The working directory is fully working directory."])
def test_r3_guard_negated_and_conditional_forms_are_not(text):
    assert not C.claims_code_works(text)


# ── r3 (coordinator): a bare "working" is progress narration, not a claim ─────────────────────────────────────
@pytest.mark.parametrize("text,claim", [
    ("working\n\nDone — the page is ready.", False), ("Working on it.", False),
    ("I'm working on the layout.", False), ("I am still working on the tests.", False),
    ("It's working.", True), ("Everything is working now.", True), ("The app is working.", True),
    ("Working fine now.", True), ("verified working", True), ("it works", True),
])
def test_r3_bare_working_is_narration(text, claim):
    from ghost_agent.core.code_claims import claims_code_works
    assert bool(claims_code_works(text)) is claim


@pytest.mark.parametrize("text", ["Validated against the sample file.", "Validated.", "Input is validated by the form."])
def test_r4_bare_validated_is_not_a_claim(text):
    """r4 (coordinator): "validated" is what a form does to input as often as what a run does to code."""
    assert not C.claims_code_works(text)


def test_r4_guard_validated_code_subject_stays_a_claim():
    assert C.claims_code_works("The script has been validated.")


def test_r4_host_text_refuses_a_fifo(tmp_path):
    """r4 (review): a FIFO planted at the page path blocked the open forever."""
    import threading
    os.mkfifo(tmp_path / "index.html")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("v", C._host_text(tmp_path, "index.html")), daemon=True)
    t.start()
    t.join(3)
    assert not t.is_alive() and box["v"] is None


def test_r4_host_text_refuses_a_link(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "secret.html").write_text(_PAGE)
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "index.html").symlink_to(tmp_path / "out" / "secret.html")
    (tmp_path / "ws" / "sub").symlink_to(tmp_path / "out", target_is_directory=True)
    assert C._host_text(tmp_path / "ws", "index.html") is None
    assert C._host_text(tmp_path / "ws", "sub/secret.html") is None
    (tmp_path / "ws" / "real.html").write_text(_PAGE)
    assert C._host_text(tmp_path / "ws", "real.html") == _PAGE



# ── r5 (round-4 review): conservative credit — a runner credits its own language, a binary the sources its build
#    named, a run its script (not its arguments); strings / comments / data lists run nothing ─────────────────

@pytest.mark.parametrize("rows,never", [
    # rev4 repros (every row of _r4/cc.py)
    ([_w("app.py"), _w("server.js"), _w("deploy.sh"), _w("tests/test_app.py"), _x("pytest tests/test_app.py")],
     ["server.js", "deploy.sh"]),
    ([_w("main.c"), _w("tool.c"), _x("gcc -o app main.c && ./app")], ["tool.c"]),
    ([_w("lint.py"), _w("main.py"), _x("python3 lint.py main.py")], ["main.py"]),
    ([_w("index.js"), _w("helper.py"), _x("npm test")], ["helper.py"]),
    ([_w("main.py"), _w("other.py"), _x('python3 main.py "$(cat other.py)"')], ["other.py"]),
    ([_w("app.py"), _x("python3 - <<EOF\nprint(open('app.py').read())\nEOF")], ["app.py"]),
    ([_w("Makefile"), _w("main.c"), _w("setup.sh"), _x("make run")], ["main.c", "setup.sh"]),
    ([_w("pkg/__main__.py"), _w("pkg/unused_tool.sh"), _x("python3 -m pkg")], ["pkg/unused_tool.sh"]),
    ([_w("server/index.js"), _w("server/build.py"), _x("node server/")], ["server/build.py"]),
    ([_w("harness.py", "import subprocess\nFILES=['python3','x.py']\nprint(FILES)"), _w("x.py"),
      _x("python3 harness.py")], ["x.py"]),
    ([_w("main.py", "msg = 'source setup.sh first'\nprint(msg)"), _w("setup.sh"), _x("python3 main.py")],
     ["setup.sh"]),
    ([_w("run.sh", "echo 'now run: python3 app.py'"), _w("app.py"), _x("bash run.sh")], ["app.py"]),
    ([_w("app.py"), _x('python3 -c "print(\'import app\')"')], ["app.py"]),
    ([_w("main.js", "// later: require('./util')\nconsole.log(1)"), _w("util.js"), _x("node main.js")], ["util.js"]),
    # more of the same rules
    ([_w("main.c"), _x("make"), _x("./app")], ["main.c"]),                     # make names no source
    ([_w("src/main.rs"), _x("cargo build && ./target/debug/app")], ["src/main.rs"]),
    ([_w("a.c"), _w("b.c"), _x("gcc -o a a.c && gcc -o b b.c"), _x("./a")], ["b.c"]),   # the binary that RAN
    ([_w("app.py"), _w("conf.py"), _x("python3 app.py --config conf.py")], ["conf.py"]),
    ([_w("app.py"), _w("t.py"), _x("python3 -m pytest t.py app.py --rootdir x.py")], []),   # suite: its .py
    ([_w("tests/a.test.js"), _w("tool.py"), _x("npx jest")], ["tool.py"]),
    ([_w("main.go"), _w("gen.py"), _x("go test ./...")], ["gen.py"]),
    ([_w("app.py"), _x("make test")], ["app.py"]),                                # make: no language
])
def test_r5_conservative_credit(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


@pytest.mark.parametrize("rows", [
    [_w("app.py"), _x("python3 -W ignore -u app.py")],
    [_w("app.js"), _x("node -r dotenv/config app.js")],
    [_w("x.ts"), _x("npx tsx x.ts")],
    [_w("x.ts"), _x("deno run --allow-net x.ts")],
    [_w("app.py"), _x("uv run python app.py")],
    [_w("app.py"), _x("poetry run python3 app.py")],
    [_w("main.go"), _w("util.go"), _x("go run main.go util.go")],
    [_w("app.py"), _x("python3 $(find . -name app.py)")],
    [_w("a.py"), _w("b.py"), _x("for t in a.py b.py; do python3 $t; done")],
    [_w("app.py"), _x("function t { python3 app.py; }; t")],
    [_w("src/main.rs"), _x("cargo run")],
    [_w("main.go"), _w("util.go"), _x("go build -o app . && ./app")],             # a Go package folder named
    [_w("cmd/x/main.go"), _x("go build -o bin/x ./cmd/x && ./bin/x")],
    [_w("x.rs"), _x("rustc x.rs && ./x")],
    [_w("harness.py", "import subprocess\nsubprocess.run(['python3', 'x.py'])\n"), _w("x.py"),
     _x("python3 harness.py")],
    [_w("util.js"), _x("node -e \"const u = require('./util')  // the helper\"")],   # inline code: the call
    [_w("app.py"), _x('python3 -c "import app"')],
])
def test_r5_guard_runs_stay_credited(rows):
    assert C.unbacked_code_writes(rows) == ([], [])


@pytest.mark.parametrize("rows,never", [
    ([_w("cmd/x/main.go"), _w("cmd/x/sub/y.go"), _x("go build -o bin/x ./cmd/x && ./bin/x")], ["cmd/x/sub/y.go"]),
    ([_w("main.go"), _x("go build ./... && ./app")], ["main.go"]),                # no binary named
    ([_w("a.py"), _w("b.py"), _x("for t in a.py b.py; do echo $t; python3 c.py; done")], ["a.py", "b.py"]),
    ([_w("a.py"), _w("b.py"), _x("for t in a.py b.py; do python3 c.py $t; done")], ["a.py", "b.py"]),  # an arg
    ([_w("main.go"), _w("util.go"), _w("other.go"), _x("go run main.go util.go")], ["other.go"]),
    ([_w("main.go"), _w("util.go"), _x("go build -o app main.go && ./app")], ["util.go"]),
    ([_w("main.go"), _w("cmd/x/main.go"), _x("go build -o bin/x ./cmd/x && ./bin/x")], ["main.go"]),
    ([_w("main.c"), _w("notes.py"), _x("gcc -o app main.c notes.py && ./app")], ["notes.py"]),   # not a source
    ([_w("main.py"), _w("data.py"), _x("python3 - data.py < main.py")], ["main.py", "data.py"]),  # stdin script
    ([_w("notes.py"), _x("python3 $(head -1 notes.py)")], ["notes.py"]),        # a reader's output: text
    ([_w("app.py"), _x("python3 - <<EOF\n# import app\nprint(1)\nEOF")], ["app.py"]),           # a comment
    ([_w("x.py"), _x("python3 - <<EOF\n# runpy.run_path('x.py')\nprint(1)\nEOF")], ["x.py"]),
])
def test_r5_guard_go_packages_and_loops(rows, never):
    assert C.unbacked_code_writes(rows)[0] == never


def test_seam_a_host_page_is_read_blocking(tmp_path, monkeypatch):
    """§4SD: `_host_text` switches its O_NONBLOCK fd back to blocking once it is a regular file."""
    import os as _os
    from ghost_agent.core import code_claims as _C
    (tmp_path / "index.html").write_text("<html></html>")
    modes, real = [], _os.read
    monkeypatch.setattr(_os, "read", lambda fd, n: (modes.append(_os.get_blocking(fd)), real(fd, n))[1])
    assert _C._host_text(tmp_path, "index.html") == "<html></html>"
    assert modes == [True]
