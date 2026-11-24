"""The owner's `lab` client (client/lab): no network, no ssh — the Api opener and the runner are faked."""
import importlib.machinery
import importlib.util
import io
import json
import shlex
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

_P = Path(__file__).resolve().parents[2] / "client" / "lab"
_L = importlib.machinery.SourceFileLoader("labcli", str(_P))
_S = importlib.util.spec_from_loader("labcli", _L)
C = importlib.util.module_from_spec(_S)
_L.exec_module(C)

PG = {"lab_id": "lab_1a2b3c4d", "template": "pg-single", "state": "ready", "owner": C.OWNER, "created_at": 2,
      "members": [{"name": "pg1", "image": "ghost-lab/pg:19", "status": "running"}]}
PYPG = {"lab_id": "lab_99999999", "members": [{"name": "app", "image": "ghost-lab/python-app:1"},
                                              {"name": "db", "image": "ghost-lab/pg:19"}]}


class FakeApi:
    def __init__(self, labs=(), detail=None, fail=None):
        self.labs, self.detail, self.fail, self.calls = list(labs), detail or PG, fail, []

    def call(self, method, path, body=None, timeout=60):
        self.calls.append((method, path, body))
        if self.fail:
            raise C.LabCliError(self.fail)
        if method == "GET" and path.startswith("/v1/labs?"):
            return {"labs": self.labs}
        if method == "GET":
            return self.detail
        if method == "POST" and path == "/v1/labs":
            return self.detail
        return {"app": {"address": "host.ts.net:20001"}}


def test_host_comes_from_env_or_the_launcher_never_a_literal(tmp_path):
    assert C.labs_base({"GHOST_LABS_HOST": "10.0.0.1:8700"}) == "http://10.0.0.1:8700"
    assert C.labs_base({"GHOST_LABS_HOST": "https://x:1/"}) == "https://x:1"
    f = tmp_path / "launch.sh"
    f.write_text('# x\nexport GHOST_LABS_HOST="10.9.9.9:8700"\n')
    assert C.labs_base({}, f) == "http://10.9.9.9:8700"
    with pytest.raises(C.LabCliError, match="no labs host"):
        C.labs_base({}, tmp_path / "missing.sh")


def test_key_file_missing_or_empty_is_an_error(tmp_path):
    with pytest.raises(C.LabCliError, match="cannot read"):
        C.labs_key({"GHOST_LABS_KEY_FILE": str(tmp_path / "nope")})
    (tmp_path / "k").write_text("  \n")
    with pytest.raises(C.LabCliError, match="empty"):
        C.labs_key({"GHOST_LABS_KEY_FILE": str(tmp_path / "k")})
    (tmp_path / "k").write_text("abc\n")
    assert C.labs_key({"GHOST_LABS_KEY_FILE": str(tmp_path / "k")}) == "abc"


def test_api_sends_key_and_owner_origin_and_reads_service_errors():
    seen = {}

    def ok(req, timeout):
        seen.update(dict(req.header_items()), url=req.full_url, method=req.get_method())
        return io.BytesIO(b'{"ok": 1}')

    assert C.Api("http://h:1", "K", ok).call("GET", "/v1/labs") == {"ok": 1}
    assert seen["X-labs-key"] == "K" and seen["X-labs-origin"] == "owner" and seen["url"] == "http://h:1/v1/labs"

    def bad(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 409, "x", {}, io.BytesIO(
            json.dumps({"error": "busy", "message": "lab is busy", "hint": "retry"}).encode()))

    with pytest.raises(C.LabCliError, match=r"409 busy lab is busy \(retry\)"):
        C.Api("http://h:1", "K", bad).call("POST", "/v1/labs", {})


def test_only_this_clients_labs_and_resolution():
    other = dict(PG, lab_id="lab_aaaa0000", owner="project:c40d12d2bef7")
    newer = dict(PG, lab_id="lab_1a2b9999", created_at=5)
    api = FakeApi([newer, PG, other])
    assert [x["lab_id"] for x in C.my_labs(api)] == ["lab_1a2b3c4d", "lab_1a2b9999"]
    assert api.calls[0][1] == "/v1/labs?owner=project%3Alabcli"
    assert C.resolve(api, None) == "lab_1a2b9999"                     # newest
    assert C.resolve(api, "1a2b3c4d") == "lab_1a2b3c4d"
    assert C.resolve(api, "lab_1a2b9") == "lab_1a2b9999"
    with pytest.raises(C.LabCliError, match="matches 2 labs"):
        C.resolve(api, "1a2b")
    with pytest.raises(C.LabCliError, match="no lab of yours"):
        C.resolve(api, "aaaa")                                        # the agent's lab is never reachable
    with pytest.raises(C.LabCliError, match="no labs"):
        C.resolve(FakeApi([]), None)


def test_member_choice():
    assert C.pick_member(PYPG, None) == "app"
    assert C.pick_member(PYPG, None, postgres=True) == "db"
    assert C.pick_member(PYPG, "app", postgres=True) == "app"
    with pytest.raises(C.LabCliError, match="no member 'x'"):
        C.pick_member(PYPG, "x")
    with pytest.raises(C.LabCliError, match="no postgres member"):
        C.pick_member({"lab_id": "l", "members": [{"name": "a", "image": "ghost-lab/base:1"}]}, None, postgres=True)


def test_exec_argv_quotes_every_remote_word():
    argv = C.exec_argv("disorder", "lab_1a2b3c4d", "pg1", ["bash", "-lc", 'exec psql "$@"', "psql", "-c", "select 'a; b'"])
    assert argv[:3] == ["ssh", "-t", "disorder"]
    remote = shlex.split(argv[3])
    assert remote[:4] == ["sudo", "docker", "exec", "-it"]
    assert "ghost-lab-lab_1a2b3c4d-pg1" in remote
    assert remote[-6:] == ["bash", "-lc", 'exec psql "$@"', "psql", "-c", "select 'a; b'"]


def test_keepalive_touches_while_the_shell_runs_and_stops_after():
    api = FakeApi()

    def runner(argv):
        time.sleep(0.35)
        return 7

    assert C.run_interactive(api, "lab_1a2b3c4d", ["x"], runner, interval=0.1) == 7
    n = len(api.calls)
    assert n >= 2 and all(c[:2] == ("GET", "/v1/labs/lab_1a2b3c4d") for c in api.calls)
    time.sleep(0.25)
    assert len(api.calls) == n                                        # stopped with the shell


def test_new_creates_as_labcli_and_opens_the_postgres_member():
    api, ran, out = FakeApi(detail=PYPG), [], []
    args = SimpleNamespace(template="python-pg", label=None, no_shell=False)
    assert C.cmd_new(api, args, out.append, lambda a: ran.append(a) or 0) == 0
    body = api.calls[0][2]
    assert body["owner"] == "project:labcli" and body["template"] == "python-pg"
    assert "ghost-lab-lab_99999999-db" in ran[0][3]


def test_keep_uses_its_own_lineage_and_prints_the_address():
    api, out = FakeApi([PG]), []
    assert C.cmd_keep(api, SimpleNamespace(lab="1a2b3c4d", name="mydb"), out.append) == 0
    m, path, body = api.calls[-1]
    assert path == "/v1/labs/lab_1a2b3c4d/keep" and body == {"app": {"name": "mydb"}, "lineage": "labcli-mydb"}
    assert any("psql -h host.ts.net -p 20001" in line for line in out)
    with pytest.raises(C.LabCliError, match="NAME"):
        C.cmd_keep(api, SimpleNamespace(lab="1a2b3c4d", name="My_DB"), out.append)


def test_rm_kept_needs_flag_and_typed_id():
    kept = dict(PG, lifetime="kept")
    api = FakeApi([kept], detail=kept)
    with pytest.raises(C.LabCliError, match="--kept"):
        C.cmd_rm(api, SimpleNamespace(lab="1a2b3c4d", kept=False), print, lambda p: "1a2b3c4d")
    assert C.cmd_rm(api, SimpleNamespace(lab="1a2b3c4d", kept=True), print, lambda p: "yes") == 1
    assert not any(c[0] == "DELETE" for c in api.calls)
    assert C.cmd_rm(api, SimpleNamespace(lab="1a2b3c4d", kept=True), print, lambda p: "1a2b3c4d") == 0
    assert api.calls[-1][:2] == ("DELETE", "/v1/labs/lab_1a2b3c4d?include_kept=1&drop_data=1")
    api2 = FakeApi([PG])
    C.cmd_rm(api2, SimpleNamespace(lab="1a2b3c4d", kept=False), print, lambda p: "")
    assert api2.calls[-1][:2] == ("DELETE", "/v1/labs/lab_1a2b3c4d")


def test_ls_table():
    now = 1000.0
    s = C.format_labs([dict(PG, expires_at=now + 3700, label="t"), dict(PG, lab_id="lab_k", lifetime="kept")], now)
    lines = s.splitlines()
    assert lines[0].split()[:5] == ["LAB", "TEMPLATE", "STATE", "KIND", "IDLE-LEFT"]
    assert "1h01m" in lines[1] and "disposable" in lines[1] and lines[2].split()[3:5] == ["kept", "-"]


def test_main_reports_errors_without_a_traceback(capsys):
    assert C.main(["ls"], api=FakeApi(fail="labs service unreachable")) == 1
    assert "lab: labs service unreachable" in capsys.readouterr().err


def test_config_file_under_the_environment(tmp_path):
    f = tmp_path / "config"
    f.write_text('# lab\nGHOST_LABS_HOST=10.1.1.1:8700\nexport GHOST_LABS_SSH="hostx"\nGHOST_LABS_KEY_FILE=\'/k\'\n')
    s = C.settings({}, f)
    assert s == {"GHOST_LABS_HOST": "10.1.1.1:8700", "GHOST_LABS_SSH": "hostx", "GHOST_LABS_KEY_FILE": "/k"}
    assert C.settings({"GHOST_LABS_SSH": "envhost", "HOME": "/x"}, f)["GHOST_LABS_SSH"] == "envhost"
    assert C.labs_base(s) == "http://10.1.1.1:8700" and C.ssh_host(s) == "hostx"
    assert C.settings({}, tmp_path / "none") == {}
