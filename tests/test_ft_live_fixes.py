"""Fixes from the 11-18 live functional-test run (scenarios against :8000)."""
import pytest
from pathlib import Path

from ghost_agent.tools.execute import _foreground_server_block


@pytest.mark.parametrize("cmd", [
    "python3 -m http.server 8099 --bind 0.0.0.0", "cd app && python3 -m http.server", "npm start", "npm run dev",
    "flask run --port 5001", "uvicorn main:app --port 8000", "PORT=3000 npm start", "python3 -m http.server | tee log",
])
def test_a_foreground_server_is_refused_before_it_burns_the_budget(cmd):
    assert "run action='start'" in _foreground_server_block(cmd)


@pytest.mark.parametrize("cmd", [
    "timeout 10 python3 -m http.server 8000", "python3 -m http.server 8000 &", "npm install && npm test",
    "python3 app.py", 'echo "npm start"', "python3 -m http.server --help", "npm run build",
    'git commit -m "run flask run; ok"', "python3 -m http.server 9 & sleep 1; curl localhost:9; kill %1",
    "cat <<EOF > run.sh\nnpm start\nEOF",
])
def test_bounded_backgrounded_or_non_server_commands_pass(cmd):
    assert _foreground_server_block(cmd) is None


# §4UY C9/C10: the `$?`-echo rule (`_status_echo_ends_the_run`) is gone — the real exit codes decide; the
# d6bff7a8 negative-test shape is pinned against a real bash in tests/test_execute_self_reported_exit.py.


@pytest.mark.parametrize("cmd", ["npm run dev:build", "npm run dev-build", "npm run start:test", "yarn dev:build",
                                 "npx vite build", "npx serve-handler", "gunicorn --check-config app:app",
                                 "flask run --help", "npm run build"])
def test_review_r1_builds_and_checks_are_not_servers(cmd):
    assert _foreground_server_block(cmd) is None


@pytest.mark.parametrize("cmd", ["df -h; npm start", "pip --version; npm start", "( npm start )", "time npm start",
                                 "bash -c 'python3 -m http.server'", "pnpm dev", "yarn run dev", "bun run dev",
                                 "npm run-script dev", "npm --prefix app start", "next dev", "jupyter notebook",
                                 "npx vite", "npx serve -l 3000"])
def test_review_r1_more_servers_and_per_segment_exemptions(cmd):
    assert _foreground_server_block(cmd)


from ghost_agent.core import env_limits as _envl

_EP = "--- COMMAND RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\ndate: cannot set date: Operation not permitted\n"


@pytest.mark.parametrize("cmd,out,code,fires", [
    ('date -s "2030-01-01"', _EP, 1, True),
    ("bash settime.sh", _EP, 1, True),                       # live w2_privilege: the op ran inside the script
    ("sudo ip link set lo mtu 1400", "RTNETLINK answers: Operation not permitted", 2, True),
    ("mount -t tmpfs none /mnt", "mount: /mnt: permission denied (you must be root)", 32, True),
    ("sysctl -w net.ipv4.ip_forward=1", 'sysctl: setting key "net.ipv4.ip_forward": Operation not permitted', 1, True),
    ("date", _EP, 0, False),                                 # succeeded
    ("chown x f", "chown: Operation not permitted", 1, False),  # not a capability-wall op
    ("cat /proc/x", "Operation not permitted", 1, False),
])
def test_capability_wall_note(cmd, out, code, fires):
    _envl._reset_for_tests()
    note = _envl.capability_note("rq", cmd, out, code)
    assert bool(note) is fires
    if fires:
        assert "Do NOT simulate" in note
        assert _envl.capability_note("rq", cmd, out, code) == ""          # once per request
        assert _envl.refusal("rq", "echo hi") == ""                         # the cap key never reads as a program


def test_capability_wall_note_is_wired_at_the_execute_sink():
    """R1 enumeration: agent.py imports capability_note and CALLS it, appending to the result it hands on."""
    import ast
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text())
    imported = any(isinstance(n, ast.ImportFrom) and n.module == "env_limits"
                   and any(a.name == "capability_note" and a.asname == "_cap_note" for a in n.names)
                   for n in ast.walk(tree))
    called = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_cap_note"
                 for n in ast.walk(tree))
    # battery §4UV F3l: the note must reach the result — an `if _cn:` that appends it to str_res
    appended = any(isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "_cn"
                   and any(isinstance(a, ast.Assign) and any(isinstance(tg, ast.Name) and tg.id == "str_res"
                                                             for tg in a.targets) for a in ast.walk(n))
                   for n in ast.walk(tree))
    assert imported and called and appended


from ghost_agent.tools.validators import _SHELL_DENY


@pytest.mark.parametrize("cmd,blocked", [
    ("rm -rf /tmp/x", False), ('rm -rf "/tmp/a b"', False), ("rm -rf /workspace/tb_src", True),
    ("rm -rf /workspace/projects", True), ("rm -rf /tmp/../etc", True),
    ("rm -rf ./build", False),
    ("rm -rf /workspace", True), ("rm -rf /workspace/", True), ("rm -rf /workspace/*", True), ("rm -rf /tmp", True),
    ("rm -rf /workspace/../etc", True), ("rm -rf /workspace/a/../../etc", True), ('rm -rf "/workspace"', True),
    ("rm -rf /etc", True), ("rm -rf /", True), ("rm -rf ~/x", True), ("rm -rf /workspace/a /etc", True),
])
def test_rm_rf_of_own_scratch_is_not_a_disk_wipe(cmd, blocked):
    """ft live w2_backup: /tmp scratch may be removed; absolute /workspace paths stay denied (owner data lives there).
    §4UY C8: the rm rule is judged per command (`validators._rm_denied`), no longer `_SHELL_DENY[0]`."""
    from ghost_agent.tools.validators import _rm_denied
    assert _rm_denied(cmd) is blocked


@pytest.mark.parametrize("cmd", ['rm -rf "/tmp/x"/../../etc', "rm -rf '/tmp/x/'..'/..'", r"rm -rf /tmp/x/.\./.\./etc"])
def test_review_r2_quoting_cannot_smuggle_dotdot_past_the_tmp_exemption(cmd):
    from ghost_agent.tools.validators import _rm_denied
    assert _rm_denied(cmd)


@pytest.mark.parametrize("cmd,out", [
    ("git commit -m 'fix mount points here'", "Operation not permitted"), ("grep -rn iptables /proc/sys", "Operation not permitted"),
    ("pip install iptables-parser", "Operation not permitted"), ("ifconfig eth0 | grep inet", "Operation not permitted"),
    ("python -m pytest -q", _EP), ("strace -e settimeofday ls", "Operation not permitted"),
])
def test_review_r2_capability_note_ignores_words_in_arguments_and_test_output(cmd, out):
    _envl._reset_for_tests()
    assert _envl.capability_note("rq", cmd, out, 1) == ""


def test_review_r2_capability_note_needs_a_request():
    _envl._reset_for_tests()
    assert _envl.capability_note("", 'date -s "2030"', _EP, 1) == ""


from ghost_agent.core.agent import is_suite_command
from ghost_agent.core import code_claims as _cc


@pytest.mark.parametrize("cmd,suite", [
    ("for i in 1 2; do python -m pytest -q test_tags.py; done", True), ("if true; then pytest -q; fi", True),
    ("time pytest -q", True), ("while true; do npm test; done", True),
    ("echo do pytest", False), ('grep -r "then pytest" .', False), ("pytest --collect-only", False),
])
def test_a_suite_run_behind_a_shell_keyword_is_a_suite_run(cmd, suite):
    assert is_suite_command(cmd) is suite


def test_live_w4_flaky_loop_run_backs_the_claim():
    """req 5d003717: 10 green pytest runs in a `for` loop, then "never run" → a needless auto-repair round."""
    w = {"name": "file_system", "arguments": {"operation": "edit", "path": "ft_w4_flaky/tags.py"},
         "content": "SUCCESS: edited — replaced 1 occurrence of old_string (line 2) in 'ft_w4_flaky/tags.py'."}
    r = {"name": "execute", "arguments": {"command": "cd ft_w4_flaky && for i in 1 2 3; do python -m pytest -q "
                                                      "test_tags.py 2>&1 | tail -1; done"},
         "content": "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n1 passed in 0.00s\n"}
    ev = _cc.code_evidence([w, r])["ft_w4_flaky/tags.py"]          # §4UY C3: keyed by path
    assert ev["run_ok"] is True
    assert _cc.unbacked_claim("Fixed and verified — the test passes.", [w, r])[0] == ""


@pytest.mark.parametrize("cmd,suite", [
    ("if ! pytest -q; then echo bad; fi", True), ("until pytest; do sleep 1; done", True), ("{ pytest -q; }", True),
    ("cat > N.md <<EOF\nThen pytest -q verifies it\nEOF", False),       # prose in a heredoc (review r1)
    ("cat > N.md <<EOF\nDo python -m pytest before you push\nEOF", False), ("cat > g.txt <<EOF\nTime go test it\nEOF", False),
])
def test_review_r1_keyword_heads_are_lower_case_shell_only(cmd, suite):
    assert is_suite_command(cmd) is suite


def test_live_w4_dash_skipped_page_check_names_the_local_data_fetch(tmp_path):
    """req 0de1c73c: a double-click page fetched sales.json; the check was skipped (fetch-backed) with a note that
    sent the model to an http server — which cannot show the file:// failure the owner then hits."""
    import asyncio, types
    from ghost_agent.core.agent import GhostAgent
    (tmp_path / "index.html").write_text("<canvas></canvas><script>fetch('sales.json').then(r => r.json())</script>")
    (tmp_path / "api.html").write_text("<script>fetch('/api/items')</script>")

    async def probe(written, project_id=None):
        return None                                  # skipped: fetch-backed, no running service
    a = types.SimpleNamespace(available_tools={"browser": object()}, _execute_web_artifact=probe,
                              _scoped_sandbox_for=lambda pid: tmp_path)
    a._page_check_after_write = GhostAgent._page_check_after_write.__get__(a)
    ok = "SUCCESS: Wrote 70 chars to '{}'. Script-side path (from sandbox cwd): '{}'."
    note = asyncio.run(a._page_check_after_write(ok.format("index.html", "index.html"), 0, []))
    assert "not run" in note and "'sales.json'" in note and "double-click" in note
    note = asyncio.run(a._page_check_after_write(ok.format("api.html", "api.html"), 0, []))
    assert "not run" in note and "double-click" not in note       # an API page keeps the serve advice only


def test_review_r2_local_fetch_note_resolves_paths_like_the_write_and_does_not_compete(tmp_path):
    import asyncio, types
    from ghost_agent.core.agent import GhostAgent
    (tmp_path / "index.html").write_text("<script>// fetch('old.json')\nfetch('sales.json')</script>")

    async def probe(written, project_id=None):
        return None
    a = types.SimpleNamespace(available_tools={"browser": object()}, _execute_web_artifact=probe,
                              _scoped_sandbox_for=lambda pid: tmp_path)
    a._page_check_after_write = GhostAgent._page_check_after_write.__get__(a)
    for spelled in ("/workspace/index.html", "./index.html"):
        ok = f"SUCCESS: Wrote 70 chars to '{spelled}'. Script-side path (from sandbox cwd): '{spelled}'."
        note = asyncio.run(a._page_check_after_write(ok, 0, []))
        assert "'sales.json'" in note and "'old.json'" not in note, spelled
        assert "If it is meant to be SERVED" in note and "Start it as a server" not in note


def test_review_r2_suite_prefix_is_linear_and_reads_elif():
    import time
    assert is_suite_command("if false; then :; elif pytest -q; then :; fi")
    t = time.time()
    is_suite_command("\nenv " * 8000)
    is_suite_command("do\n" * 6000)
    assert time.time() - t < 1.0


def test_execute_description_says_data_paths_are_relative_to_the_script():
    """ft live w3_logs + w6_migrate: `open('ft_x/access.log')` worked from /workspace only."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    text = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")["function"]["description"]
    assert "Path(__file__).parent" in text


from ghost_agent.tools.execute import _self_loading_overwrite

_APP = "from flask import Flask\napp = Flask(__name__)\n"


@pytest.mark.parametrize("content,blocked", [
    ('import importlib.util\nspec = importlib.util.spec_from_file_location("app", "ft_w6_rest/app.py")\n', True),
    ("from app import app\nc = app.test_client()\n", True), ("import app\n", True),
    ("import runpy\nrunpy.run_path('app.py')\n", True),
    ("from flask import Flask\napp = Flask(__name__)\n@app.get('/')\ndef h(): return 'x'\n", False),   # a rewrite
    (_APP, False),                                                                                     # unchanged
    ("import application\n", False), ("print('happy.py')\n", False),
])
def test_execute_refuses_overwriting_a_file_with_a_script_that_loads_it(tmp_path, content, blocked):
    """req probe-1b (w6_rest): execute(filename='ft_w6_rest/app.py', content=<its test>) replaced the app."""
    f = tmp_path / "ft_w6_rest" / "app.py"
    f.parent.mkdir()
    f.write_text(_APP)
    msg = _self_loading_overwrite("ft_w6_rest/app.py", content, f)
    assert bool(msg) is blocked
    if blocked:
        assert "Nothing was written" in msg and "test_app.py" in msg


def test_self_loading_overwrite_ignores_new_and_empty_files(tmp_path):
    assert _self_loading_overwrite("x.py", "import x\n", tmp_path / "x.py") == ""
    (tmp_path / "y.py").write_text("")
    assert _self_loading_overwrite("y.py", "import y\n", tmp_path / "y.py") == ""


def test_live_shape_through_tool_execute(tmp_path):
    import asyncio
    from ghost_agent.tools.execute import tool_execute
    f = tmp_path / "ft_w6_rest" / "app.py"
    f.parent.mkdir()
    f.write_text(_APP)
    out = asyncio.run(tool_execute(filename="ft_w6_rest/app.py",
                                   content='import importlib.util\nspec = importlib.util.spec_from_file_location('
                                           '"app", "ft_w6_rest/app.py")\n',
                                   sandbox_dir=tmp_path, sandbox_manager=object()))
    assert "would OVERWRITE" in str(out) and f.read_text() == _APP


def _hit_the_cap(tmp_path):
    from ghost_agent.sandbox.services import MAX_SERVICES, ServiceSupervisor
    from tests.test_sandbox_services import FakeSandbox, happy_handler
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    for i in range(MAX_SERVICES):
        assert "RUNNING" in sup.start(f"s{i}", "cmd")
    return sup.start("one-too-many", "cmd")


def test_service_cap_message_names_the_services_and_a_route_that_needs_no_slot(tmp_path):
    """ft live w6_rest: the cap said "stop one first (action='status' to list)" — the slots were other
    conversations' services, which that status does not list."""
    from ghost_agent.tools.execute import _daemonized_server_block, _foreground_server_block
    out = _hit_the_cap(tmp_path)
    assert "already running (s0, s1" in out and "app.test_client()" in out and "Stop one you own" in out
    suggested = "timeout 20 python3 app.py & sleep 2; curl -s localhost:5000/"
    assert not _daemonized_server_block(suggested) and not _foreground_server_block(suggested)


def test_review_r1_self_overwrite_is_a_rejection_and_test_files_may_name_themselves(tmp_path):
    import asyncio
    from ghost_agent.tools.execute import tool_execute
    from ghost_agent.tools.outcome import ToolOutcome
    (tmp_path / "app.py").write_text(_APP)
    out = asyncio.run(tool_execute(filename="app.py", content="import app\n", sandbox_dir=tmp_path,
                                   sandbox_manager=object()))
    assert ToolOutcome.coerce(out).is_rejection
    (tmp_path / "test_app.py").write_text("def test_a(): assert 1\n")
    assert _self_loading_overwrite("test_app.py", "import pytest\ndef test_a(): assert 2\n"
                                   "if __name__ == '__main__': pytest.main(['-q', 'test_app.py'])\n",
                                   tmp_path / "test_app.py") == ""


def test_review_r1_cap_message_hides_service_names_from_a_member(tmp_path, monkeypatch):
    import ghost_agent.utils.logging as L
    monkeypatch.setattr(L, "requester_is_member", lambda: True)
    out = _hit_the_cap(tmp_path)
    assert "already running (…)" in out and "s0" not in out


# ── final review round (2026-11-18) ──────────────────────────────────────

@pytest.mark.parametrize("cmd,blocked", [
    ('git commit -m "fix\nserve static files"', False), ('printf "Steps:\nnpm install\nnpm run dev\n" > README.txt', False),
    ("python3 - <<'EOF'\nserve = make_server()\nvite = 1\nEOF", False), ("vite optimize", False),
    ("flask run -h 0.0.0.0", True), ("python3 -m http.server 8000; kill %1", True), ("sudo -u x npm start", True),
    ("npm -C web run dev", True), ("python3 -m http.server 8000 & sleep 1; curl x; kill %1", False),
])
def test_final_foreground_guard(cmd, blocked):
    assert bool(_foreground_server_block(cmd)) is blocked


@pytest.mark.parametrize("cmd,failed,adopted", [
    ('python3 a.py; echo "exit=$?"; rm -f tmp.txt', "python3 a.py", True),
    ('cd p && python3 a.py; echo "exit=$?"; deactivate', "python3 a.py", True),
    ('python3 a.py; echo "exit=$?"; cat out.log', "python3 a.py", True),
    ('python3 a.py; echo "exit=${?}"', "python3 a.py", True),
    ('python3 inv.py remove z 99; echo "exit=$?"; python3 inv.py list; ls', "python3 inv.py remove z 99", False),
    ('x; echo "exit=$?"; pytest -q', "x", False),
    ('make; echo "exit=$?"; timeout -s KILL 5 python3 t.py', "make", False),
    ('make; echo "exit=$?"; timeout --foreground 5 python3 t.py', "make", False),
])
def test_final_status_echo_is_ended_only_by_a_later_runner(cmd, failed, adopted):
    """§4UY r3: a command that reads `$?` KEEPS the trap (bash restores `$?` after it). A failure whose status
    the very next member reads is the model's own probe — never adopted, whatever ran after it (`adopted`
    records the r0 rule this replaced); a LATER runner's failure still counts."""
    from ghost_agent.tools.execute import _bash_c, _normalise_exit, _reads_own_status
    assert not _reads_own_status(cmd) and " ERR" in _bash_c(cmd)
    assert _normalise_exit(cmd, 0, "", [(1, [1], failed)]) == 0


@pytest.mark.parametrize("content,blocked", [
    ('import argparse\np = argparse.ArgumentParser(prog="app.py")\n', False), ("print('app.py')\n", False),
    ('import os\nos.environ["FLASK_APP"] = "app.py"\n', False), ("import runpy\nrunpy.run_path('app.py')\n", True),
])
def test_final_self_overwrite_needs_a_loader_call(tmp_path, content, blocked):
    f = tmp_path / "app.py"
    f.write_text(_APP)
    assert bool(_self_loading_overwrite("app.py", content, f)) is blocked


@pytest.mark.parametrize("cmd,ok", [
    ("rm -rf /tmp/b\npython3 /workspace/run.py", True), ("rm -rf /tmp/x > /dev/null 2>&1", True),
    ("rm -rf /tmp/x\nrm -rf /etc", False), ("rm -rf / > /dev/null", False),
])
def test_final_rm_rule_stops_at_a_newline_and_skips_redirect_targets(cmd, ok):
    from ghost_agent.tools.validators import validate_shell
    assert validate_shell(cmd)[0] is ok


@pytest.mark.parametrize("cmd,out,fires", [
    ("mount | grep tmpfs; chown root x", "Operation not permitted", False), ("tc qdisc show; chown root x", "Operation not permitted", False),
    ("bash -n x.sh", "RTNETLINK answers: Operation not permitted", False),
    ("bash check.sh", "sysctl: permission denied on key 'kernel.x'", False),
    ("ip l set eth0 mtu 1400", "RTNETLINK answers: Operation not permitted", True),
    ("/bin/date -s 2030", _EP, True), ("env TZ=x date -s 2030", _EP, True), ("timeout 5 ip link set eth0 up", "RTNETLINK answers: Operation not permitted", True),
])
def test_final_capability_note(cmd, out, fires):
    _envl._reset_for_tests()
    assert bool(_envl.capability_note("rq", cmd, out, 1)) is fires


@pytest.mark.parametrize("blob,suite", [
    ("python3 - <<'EOF'\nx = 1\nif pytest:\n    pass\nEOF", False), ("echo 'x; pytest'", False),
    ("cat > t.sh <<'EOF'\npytest -q\nEOF\nbash t.sh", True),     ("for t in a b; do python -m pytest -q; done", True),
])
def test_final_suite_ignores_data_heredocs_and_quotes(blob, suite):
    from ghost_agent.core.agent import is_suite_command
    assert is_suite_command(blob) is suite


def test_final_local_fetch_keeps_code_after_a_double_slash():
    from ghost_agent.core.page_check import local_fetch_note
    page = "const u = 'a // b'; fetch('sales.json').then(r => r.json());\n// fetch('old.json')\n"
    note = local_fetch_note([page])
    assert "sales.json" in note and "old.json" not in note
    import time
    t = time.time(); local_fetch_note(["<!--" * 50000]); assert time.time() - t < 2


def test_final_cap_suggestion_passes_both_server_guards():
    from ghost_agent.tools.execute import _daemonized_server_block, _foreground_server_block
    cmd = "timeout 20 python3 app.py & sleep 2; curl -s localhost:5000/; kill %1"
    assert not _daemonized_server_block(cmd) and not _foreground_server_block(cmd)


# ── §4UV mutation battery survivors (2026-11-18) ─────────────────────────

async def test_battery_foreground_server_refused_by_tool_execute(tmp_path):
    from unittest.mock import MagicMock
    from ghost_agent.tools.execute import tool_execute
    from ghost_agent.tools.outcome import OutcomeStatus
    mgr = MagicMock(); mgr.execute = MagicMock(return_value=("", 0))
    res = await tool_execute(command="python3 -m http.server 8000", sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert getattr(res, "status", None) == OutcomeStatus.REJECTED and "FOREGROUND" in str(res)
    mgr.execute.assert_not_called()


def test_battery_status_echo_heredoc_closer_named_like_a_runner():
    cmd = 'python3 a.py; echo "exit=$?"; cat > notes.txt <<PY\nhello\nPY'
    from ghost_agent.tools.execute import _normalise_exit
    # §4UY r3: the next member reads this status (`echo "exit=$?"`): the model's own probe, not adopted
    assert _normalise_exit(cmd, 0, "exit=1\n", [(1, "python3 a.py")]) == 0


@pytest.mark.parametrize("name,old,content,blocked", [
    ("test_app.py", "def test_x():\n    assert 1\n", "import test_app\n", True),        # imports ITSELF
    ("app.py", None, "import app\n", False),                                           # no file yet
    ("app.py", "import app\n", "import app\n", False),                                 # same content
    ("app.js", "module.exports = {}\n", "const a = require('./app');\n", True),        # node require
    ("./.hidden.py", "x = 1\n", "import runpy\nrunpy.run_path('.hidden.py')\n", True),   # ./ prefix, not chars
])
def test_battery_self_overwrite_edges(tmp_path, name, old, content, blocked):
    f = tmp_path / name[2:] if name.startswith("./") else tmp_path / name
    if old is not None:
        f.write_text(old)
    assert bool(_self_loading_overwrite(name, content, f)) is blocked


@pytest.mark.parametrize("cmd", ["rm -rf /tmp/x/../../etc", 'rm -rf "/tmp/x/../../etc"'])
def test_battery_rm_tmp_dotdot_stays_blocked(cmd):
    from ghost_agent.tools.validators import validate_shell
    assert validate_shell(cmd)[0] is False


def test_battery_rm_refusal_names_the_route():
    from ghost_agent.tools.validators import validate_shell
    ok, why = validate_shell("rm -rf /etc")
    assert not ok and "rm -rf /tmp/x" in why


def test_battery_capability_note_needs_a_failure_and_ignores_quoted_ops():
    _envl._reset_for_tests()
    assert _envl.capability_note("rq", "date -s 2030", _EP, 0) == ""
    assert _envl.capability_note("rq", "echo 'note; date -s 2030 is impossible'; chown a b", _EP, 1) == ""
    assert _envl.capability_note("rq", "date -s 2030", _EP, 1)


def test_battery_suite_heads_are_lower_case_only():
    from ghost_agent.core.agent import is_suite_command
    assert not is_suite_command("notes:\nThen python -m pytest -q is what we run")
    assert is_suite_command("if true; then python -m pytest -q; fi")


def test_battery_python_script_text_is_not_a_suite_run():
    from ghost_agent.core.code_claims import _exec_run_keys
    py = _exec_run_keys({}, {"filename": "t.py", "content": "import pytest\nx = 1\nif pytest:\n    pass\n"})
    sh = _exec_run_keys({}, {"filename": "run.sh", "content": "#!/bin/sh\npython -m pytest -q\n"})
    assert py[1] is False and sh[1] is True


def test_battery_local_fetch_ignores_absolute_urls():
    from ghost_agent.core.page_check import local_fetch_note
    assert local_fetch_note(["fetch('https://example.org/a.json')"]) == ""
    assert "a.json" in local_fetch_note(["fetch('a.json')"])
