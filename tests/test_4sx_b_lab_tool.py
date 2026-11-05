"""§4SX Phase B: the `lab` tool (src/ghost_agent/tools/labs.py) and its wiring.

The labs service is replaced by a scripted fake at the `_call` seam (every
HTTP request the tool makes goes through it); `_call` itself is checked
against an httpx MockTransport for its headers and its unreachable mapping.
"""
from __future__ import annotations

import ast
import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import request_scope as RS
from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import OutcomeStatus

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


# ------------------------------------------------------------------ fixtures

def _ctx(tmp_path, *, host="127.0.0.1:8700", key="k" * 64, pid=None, conv="conv1"):
    kf = tmp_path / "labs.key"
    if key is not None:
        kf.write_text(key)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    memory = tmp_path / "memory"
    memory.mkdir(exist_ok=True)
    return SimpleNamespace(
        args=SimpleNamespace(labs_host=host, labs_key_file=str(kf)),
        sandbox_dir=sandbox, memory_dir=memory, tor_proxy="socks5://127.0.0.1:9050",
        current_project_id=pid, conversation_key=conv)


@pytest.fixture
def scope():
    """Install a request scope for the test's context; returns a setter."""
    toks = []

    def _set(ctx, origin="owner"):
        toks.append(RS.install(RS.RequestScope(conversation_key=ctx.conversation_key,
                                               project_id=ctx.current_project_id, origin=origin, context=ctx)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


class FakeLabs:
    """Scripted responses keyed on (method, path); records every call."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def on(self, method, path, *responses):
        self.routes[(method, path)] = list(responses)

    async def __call__(self, context, method, path, *, origin, json_body=None, params=None, timeout=30.0):
        self.calls.append(SimpleNamespace(method=method, path=path, origin=origin, json=json_body,
                                          params=params, timeout=timeout))
        rs = self.routes.get((method, path))
        if not rs:
            return 404, {"error": "not_routed", "message": f"{method} {path}"}
        r = rs.pop(0) if len(rs) > 1 else rs[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


LAB = "lab_0123abcd"


def _detail(state="ready", **kw):
    d = {"lab_id": LAB, "template": "pg-single", "state": state, "owner": "conversation:conv1", "origin": "owner",
         "lifetime": "disposable", "expires_at": 1_800_000_000, "ram_mb": 512,
         "members": [{"name": "db", "status": "running", "ip": "10.77.0.2", "image": "ghost-lab/pg:18",
                      "ports": [{"container": 5432, "host": 20001, "url": "tcp://labs.example:20001"}],
                      "isolated": False}]}
    d.update(kw)
    return d


# ------------------------------------------------------------------ configuration

def test_configured_needs_a_host_and_a_readable_non_empty_key(tmp_path):
    assert L.labs_configured(_ctx(tmp_path))
    assert not L.labs_configured(_ctx(tmp_path, host=""))
    assert not L.labs_configured(_ctx(tmp_path, key=""))
    c = _ctx(tmp_path, key=None)
    c.args.labs_key_file = str(tmp_path / "nope.key")
    assert not L.labs_configured(c)
    assert not L.labs_configured(None)


def test_only_real_strings_configure_it(tmp_path):
    """A MagicMock args (the suite's mock_context) is 'unset', never a host."""
    from unittest.mock import MagicMock
    c = _ctx(tmp_path)
    c.args = MagicMock()
    assert L.labs_settings(c) == ("", "")
    assert not L.labs_configured(c)


@pytest.mark.parametrize("host,bare", [("100.1.2.3:8700", "100.1.2.3"), ("http://h.example:8700/", "h.example"),
                                       ("h.example", "h.example"), ("[fd00::1]:8700", "fd00::1")])
def test_host_part(host, bare):
    assert L.host_part(host) == bare


def test_base_url_prepends_a_scheme_only_when_missing():
    assert L._base_url("1.2.3.4:8700") == "http://1.2.3.4:8700"
    assert L._base_url("https://x:1/") == "https://x:1"


def test_registry_hides_it_unconfigured_and_lists_it_configured(tmp_path):
    from unittest.mock import MagicMock
    from ghost_agent.tools import registry as R

    def names(sn):
        c = MagicMock()
        c.args = MagicMock()
        c.args.labs_host, c.args.labs_key_file = sn.args.labs_host, sn.args.labs_key_file
        c.memory_dir, c.sandbox_dir = sn.memory_dir, sn.sandbox_dir
        return {t["function"]["name"] for t in R.get_active_tool_definitions(c, None, serve_tuned=False)}
    assert "lab" in names(_ctx(tmp_path))
    assert "lab" not in names(_ctx(tmp_path, host=""))
    assert "lab" not in names(_ctx(tmp_path, key=""))


def test_main_args_default_from_env(monkeypatch):
    import ghost_agent.main as M
    monkeypatch.setenv("GHOST_LABS_HOST", "10.0.0.9:8700")
    monkeypatch.setenv("GHOST_LABS_KEY_FILE", "/k/file")
    src = Path(M.__file__).read_text()
    tree = ast.parse(src)
    flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "add_argument" for a in n.args[:1] if isinstance(a, ast.Constant)}
    assert {"--labs-host", "--labs-key-file"} <= flags
    # the key itself is never an argument (args are recorded in last_config.json)
    assert not any(re.fullmatch(r"--labs-key", f) for f in flags)


async def test_unconfigured_call_is_refused_without_a_request(tmp_path, fake, scope):
    c = _ctx(tmp_path, host="")
    scope(c)
    r = await L.tool_lab(c, action="list")
    assert r.status is OutcomeStatus.REJECTED and "not_configured" in r
    assert fake.calls == []


# ------------------------------------------------------------------ who is asking

@pytest.mark.parametrize("origin,header", [("owner", "owner"), ("probe", "probe"),
                                           ("unattended", "autopilot"), ("leaf", "autopilot")])
async def test_origin_header_follows_the_request_scope(tmp_path, fake, scope, origin, header):
    c = _ctx(tmp_path)
    scope(c, origin)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    await L.tool_lab(c, action="list")
    assert fake.calls[0].origin == header


async def test_no_scope_is_autopilot(tmp_path, fake):
    c = _ctx(tmp_path)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    await L.tool_lab(c, action="list")
    assert fake.calls[0].origin == "autopilot"


@pytest.mark.parametrize("origin", ["member", "replay"])
async def test_members_and_replays_never_reach_a_lab(tmp_path, fake, scope, origin):
    c = _ctx(tmp_path)
    scope(c, origin)
    r = await L.tool_lab(c, action="templates")
    assert r.status is OutcomeStatus.REJECTED and "not_allowed" in r
    assert fake.calls == []


async def test_owner_key_is_the_project_else_the_conversation(tmp_path, fake, scope):
    c = _ctx(tmp_path, pid="41f745517fe4")
    scope(c)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    lists = lambda: [x for x in fake.calls if x.path == "/v1/labs"]
    await L.tool_lab(c, action="list")
    assert lists()[-1].params == {"owner": "project:41f745517fe4"}
    c2 = _ctx(tmp_path, conv="abc")
    scope(c2)
    await L.tool_lab(c2, action="list")
    assert lists()[-1].params == {"owner": "conversation:abc"}
    await L.tool_lab(c2, action="list", all_owners=True)
    assert lists()[-1].params is None
    await L.tool_lab(c2, action="list", all_owners="false")          # a model's string "false"
    assert lists()[-1].params == {"owner": "conversation:abc"}


async def test_a_bad_lab_id_is_refused_before_any_request(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    for bad in ("", "lab_xyz", "../v1/gc", "lab_0123abcd/../../gc"):
        r = await L.tool_lab(c, action="status", lab_id=bad)
        assert r.status is OutcomeStatus.REJECTED, bad
    assert fake.calls == []


async def test_unknown_action_is_refused(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    r = await L.tool_lab(c, action="teleport")          # (deploy became a real action in §4SX D3)
    assert r.status is OutcomeStatus.REJECTED and "action must be one of" in r
    assert fake.calls == []


# ------------------------------------------------------------------ exec

async def test_exec_result_carries_the_header_and_the_exit_code_line(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 0, "stdout": "PostgreSQL 18.1\n", "stderr": "",
                                                   "duration_s": 0.4, "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="psql -c 'select version()'")
    assert r.status is OutcomeStatus.OK
    lines = str(r).splitlines()
    assert lines[0] == f"[{LAB}/db exit=0 0.4s]"
    assert "EXIT CODE: 0" in lines
    assert "PostgreSQL 18.1" in r
    assert fake.calls[0].json == {"member": "db", "cmd": "psql -c 'select version()'", "timeout_s": 120}


async def test_a_failing_command_is_failed_with_its_code(tmp_path, fake, scope):
    from ghost_agent.core.strikes import error_line
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 2, "stdout": "", "stderr": "psql: error: boom\n",
                                                   "duration_s": 0.1, "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="psql")
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "lab_exit_nonzero"
    assert "EXIT CODE: 2" in r
    assert "boom" in error_line(r, tool="lab")


async def test_a_timeout_says_timed_out_and_has_no_exit_code(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": None, "stdout": "partial", "stderr": "",
                                                   "duration_s": 5, "timed_out": True}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="sleep 99", timeout_s=5)
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "lab_timed_out"
    assert "TIMED OUT" in r and "EXIT CODE" not in r and "exit=" not in r


async def test_output_keeps_the_tail_and_says_it_was_cut(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    big = "x" * (L.OUT_CAP + 500) + "THE-END"
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 0, "stdout": big, "stderr": "", "duration_s": 1,
                                                   "timed_out": False, "stdout_truncated": True}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="yes")
    assert "THE-END" in r and "stdout truncated" in r and "the labs host also cut" in r
    assert len(r) < L.OUT_CAP + 1000


async def test_exec_timeout_is_clamped_to_the_turn(tmp_path, fake, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    monkeypatch.setattr(L, "_time_left_s", lambda: L.TURN_RESERVE_S + 60.0)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 0, "stdout": "", "stderr": "", "duration_s": 1,
                                                   "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x", timeout_s=600)
    sent = fake.calls[0].json["timeout_s"]
    assert sent == 60
    assert "timeout cut to" in r
    assert fake.calls[0].timeout > sent      # the HTTP wait outlives the command


async def test_no_turn_time_left_runs_nothing(tmp_path, fake, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    monkeypatch.setattr(L, "_time_left_s", lambda: L.TURN_RESERVE_S + 2)
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert r.reason_code == "lab_no_time" and fake.calls == []


async def test_exec_keeps_a_transcript_per_owner(tmp_path, fake, scope):
    c = _ctx(tmp_path, pid="p1")
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 3, "stdout": "o", "stderr": "e", "duration_s": 1,
                                                   "timed_out": False}))
    await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="false")
    f = c.memory_dir / "lab_transcripts" / "project_p1" / f"{LAB}.jsonl"
    row = json.loads(f.read_text().splitlines()[-1])
    assert row["exit_code"] == 3 and row["cmd"] == "false" and row["member"] == "db"


async def test_time_left_reads_the_request_and_the_leaf_deadline(monkeypatch):
    import time as _t
    from ghost_agent.core import coding_executor as ce
    monkeypatch.setattr(ce, "_request_left_s", lambda: 500.0)
    tok = ce.leaf_deadline_context.set(_t.monotonic() + 100)
    try:
        assert 90 < L._time_left_s() <= 100
    finally:
        ce.leaf_deadline_context.reset(tok)
    assert L._time_left_s() == 500.0
    monkeypatch.setattr(ce, "_request_left_s", lambda: None)
    assert L._time_left_s() is None


# ------------------------------------------------------------------ create

async def test_create_sends_owner_and_no_wait_then_polls_to_ready(tmp_path, fake, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    fake.on("POST", "/v1/labs", (201, _detail("creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "host-abc"}))
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail("creating")), (200, _detail("ready")))
    r = await L.tool_lab(c, action="create", template="pg-single", params='{"version": "18"}', purpose="test")
    assert r.status is OutcomeStatus.OK and "ready" in r and "tcp://labs.example:20001" in r
    body = fake.calls[0].json
    assert body["owner"] == "conversation:conv1" and body["wait"] is False and body["params"] == {"version": "18"}
    from ghost_agent.memory.lab_inventory import inventory_for
    row = inventory_for(c).get(LAB)
    assert row["host_id"] == "host-abc" and row["owner"] == "conversation:conv1" and row["origin"] == "owner"


async def _no_sleep(*_a, **_k):
    return None


async def test_create_out_of_turn_time_returns_creating(tmp_path, fake, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    monkeypatch.setattr(L, "_time_left_s", lambda: L.TURN_RESERVE_S)   # nothing left to wait
    fake.on("POST", "/v1/labs", (201, _detail("creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h"}))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert r.status is OutcomeStatus.PARTIAL and "check with `lab status`" in r
    assert not any(x.path == f"/v1/labs/{LAB}" for x in fake.calls)


async def test_create_that_fails_says_so(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", "/v1/labs", (201, _detail("failed")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h"}))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "lab_not_ready"


async def test_create_without_template_or_with_bad_params(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    assert (await L.tool_lab(c, action="create")).status is OutcomeStatus.REJECTED
    assert (await L.tool_lab(c, action="create", template="x", params="[1]")).status is OutcomeStatus.REJECTED
    assert (await L.tool_lab(c, action="create", template="x", params="{bad")).status is OutcomeStatus.REJECTED
    assert fake.calls == []


async def test_create_names_what_was_evicted(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", "/v1/labs", (201, _detail(evicted=[{"lab_id": "lab_99999999", "owner": "project:x"}])))
    fake.on("GET", "/v1/health", (200, {"host_id": "h"}))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert "lab_99999999 of project:x" in r


# ------------------------------------------------------------------ errors

async def test_service_error_renders_message_and_hint(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", "/v1/labs", (409, {"error": "budget_exceeded", "message": "no RAM", "hint": "destroy one"}))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert str(r).startswith("LAB ERROR (budget_exceeded): no RAM") and "HINT: destroy one" in r
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "lab_budget_exceeded"


async def test_unknown_lab_says_who_destroyed_it(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_destroy(LAB, by="project:other")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    r = await L.tool_lab(c, action="status", lab_id=LAB)
    assert "by project:other" in r and "Create a new lab" in r and r.status is OutcomeStatus.REJECTED


async def test_unreachable_host_is_labs_unreachable(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("GET", "/v1/templates", L.LabsUnreachable("ConnectError: refused"))
    r = await L.tool_lab(c, action="templates")
    assert r.status is OutcomeStatus.FAILED and r.reason_code == "lab_labs_unreachable"
    assert "127.0.0.1:8700" in r and "nothing ran" in r


# ------------------------------------------------------------------ destroy / list

async def test_destroy_records_it_and_names_another_owner(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _ctx(tmp_path)
    scope(c)
    inventory_for(c).record_create(LAB, owner="project:p9", origin="owner", template="pg-single")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:p9")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB, "state": "destroyed"}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.OK and "belonged to project:p9" in r
    row = inventory_for(c).get(LAB)
    assert row["destroyed_at"] and row["destroyed_by"] == "conversation:conv1"


async def test_destroy_of_my_own_lab_names_no_one(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail()))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert "belonged" not in r


async def test_list_names_my_labs_destroyed_by_someone_else_and_marks_expired(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_aaaaaaaa", owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_destroy("lab_aaaaaaaa", by="project:z")
    inv.record_create("lab_bbbbbbbb", owner="conversation:conv1", origin="owner", template="pg-single")
    fake.on("GET", "/v1/labs", (200, {"labs": [_detail()]}))
    # review 11-05 M2: an unlisted row is ASKED about before it is retired
    fake.on("GET", "/v1/labs/lab_bbbbbbbb", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", "/v1/labs/lab_bbbbbbbb", (404, {"error": "unknown_lab"}))
    r = await L.tool_lab(c, action="list")
    assert "lab_aaaaaaaa (pg-single) — destroyed" in r and "by project:z" in r
    assert "lab_bbbbbbbb (pg-single) — gone from the labs host" in r
    assert inv.get("lab_bbbbbbbb")["destroyed_by"] == "expired"
    assert LAB in r


# ------------------------------------------------------------------ files

async def test_put_reads_from_the_project_workspace(tmp_path, fake, scope):
    c = _ctx(tmp_path, pid="p1")
    scope(c)
    d = c.sandbox_dir / "projects" / "p1"
    d.mkdir(parents=True)
    (d / "init.sql").write_text("select 1;")
    fake.on("PUT", f"/v1/labs/{LAB}/files", (200, {"bytes": 9}))
    r = await L.tool_lab(c, action="put", lab_id=LAB, member="db", sandbox_path="init.sql", path="/root/init.sql")
    assert r.status is OutcomeStatus.OK
    assert base64.b64decode(fake.calls[0].json["content_b64"]) == b"select 1;"


async def test_put_and_get_stay_inside_the_workspace(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    (tmp_path / "secret.txt").write_text("s")
    for act in ("put", "get"):
        r = await L.tool_lab(c, action=act, lab_id=LAB, member="db", sandbox_path="../secret.txt", path="/x")
        assert r.status is OutcomeStatus.REJECTED and "outside the workspace" in r, act
    assert fake.calls == []


async def test_get_writes_the_file_and_workspace_prefix_is_the_project_root(tmp_path, fake, scope):
    c = _ctx(tmp_path, pid="p1")
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"LOG").decode(), "bytes": 3}))
    r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/var/log/x", sandbox_path="/workspace/out/x.log")
    assert r.status is OutcomeStatus.OK
    # r14: `/workspace/<x>` in a project turn is the project's <x>, as file_system resolves it
    assert (c.sandbox_dir / "projects" / "p1" / "out" / "x.log").read_bytes() == b"LOG"
    await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/var/log/x", sandbox_path="x.log")
    assert (c.sandbox_dir / "projects" / "p1" / "x.log").read_bytes() == b"LOG"


async def test_member_op_must_be_known(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    r = await L.tool_lab(c, action="member", lab_id=LAB, member="db", op="rm")
    assert r.status is OutcomeStatus.REJECTED and fake.calls == []
    fake.on("POST", f"/v1/labs/{LAB}/members/db/isolate", (200, {"status": "running", "isolated": True}))
    r = await L.tool_lab(c, action="member", lab_id=LAB, member="db", op="isolate")
    assert "isolated" in r and r.world_changed is True


async def test_http_without_an_answer_is_failed(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/http", (200, {"member": "web", "port": 80, "path": "/", "status": None,
                                                   "error": "ConnectionRefusedError"}))
    r = await L.tool_lab(c, action="http", lab_id=LAB, member="web")
    assert r.status is OutcomeStatus.FAILED and "ConnectionRefusedError" in r


# ------------------------------------------------------------------ the HTTP seam

async def test_call_sends_key_and_origin_and_maps_transport_errors(tmp_path, monkeypatch):
    import httpx
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("X-Labs-Key")
        seen["origin"] = request.headers.get("X-Labs-Origin")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    real = httpx.AsyncClient

    def client(**kw):
        seen["proxy"] = kw.get("proxy")
        kw.pop("proxy", None)
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    c = _ctx(tmp_path, host="100.94.0.1:8700")
    st, body = await L._call(c, "GET", "/v1/health", origin="probe")
    assert (st, body) == (200, {"ok": True})
    assert seen["key"] == "k" * 64 and seen["origin"] == "probe"
    assert seen["url"] == "http://100.94.0.1:8700/v1/health"
    assert seen["proxy"] is None            # a tailnet address is never sent over Tor

    def boom(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(boom),
                                                                **{k: v for k, v in kw.items() if k != "proxy"}))
    with pytest.raises(L.LabsUnreachable):
        await L._call(c, "GET", "/v1/health", origin="probe")


async def test_boot_line_never_raises(tmp_path, monkeypatch):
    async def bad(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(L, "_call", bad)
    await L.labs_boot_line(_ctx(tmp_path))
    await L.labs_boot_line(_ctx(tmp_path, host=""))


# ------------------------------------------------------------------ the workspace cannot reach it

def test_execute_or_browser_naming_the_labs_host_is_refused(tmp_path):
    c = _ctx(tmp_path, host="100.94.0.1:8700")
    r = L.lab_reach_refusal(c, {"command": "curl http://100.94.0.1:8700/v1/labs"}, "execute")
    assert r.status is OutcomeStatus.REJECTED and r.reason_code == "lab_host_from_workspace"
    assert L.lab_reach_refusal(c, {"command": "curl http://100.94.0.10/"}, "execute") is None
    assert L.lab_reach_refusal(c, {"command": "ping 1100.94.0.1"}, "execute") is None
    assert L.lab_reach_refusal(_ctx(tmp_path, host=""), {"command": "x"}, "execute") is None


def test_a_lab_id_in_workspace_code_gets_a_note_and_keeps_its_outcome():
    from ghost_agent.tools.outcome import ToolOutcome
    res = ToolOutcome("EXIT CODE: 1", status=OutcomeStatus.FAILED, reason_code="x")
    out = L.with_lab_id_note({"command": "ssh lab_0123abcd"}, res)
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "x"
    assert "lab_0123abcd is a lab machine" in out
    assert L.with_lab_id_note({"command": "ls"}, res) is res


async def test_the_registry_execute_and_browser_apply_the_refusal(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from ghost_agent.tools import registry as R
    c = MagicMock()
    c.args = SimpleNamespace(labs_host="100.94.0.1:8700", labs_key_file="", anonymous=True)
    c.sandbox_dir = tmp_path
    tools = R.get_available_tools(c)

    async def must_not_run(**kw):
        raise AssertionError("ran")
    monkeypatch.setattr(R, "tool_execute", must_not_run)
    monkeypatch.setattr(R, "tool_browser", must_not_run)
    r = await tools["execute"](command="curl 100.94.0.1:8700/v1/health")
    assert r.reason_code == "lab_host_from_workspace"
    r = await tools["browser"](action="navigate", url="http://100.94.0.1:8700/")
    assert r.reason_code == "lab_host_from_workspace"
    assert "lab" in tools


# ------------------------------------------------------------------ wiring

def test_no_labs_host_literal_in_src():
    """UNIFIED §11: addresses come from configuration and from the API,
    never from code. A literal of today's hosts in src is a composed address."""
    bad = re.compile(r"100\.94\.162\.61|100\.90\.234\.23|100\.122\.46\.101|\bdisorder\b")
    hits = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py")
            for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1) if bad.search(ln)]
    assert hits == []


def test_redaction_removes_the_labs_key(tmp_path, monkeypatch):
    from ghost_agent.distill import redact as D
    kf = tmp_path / "k"
    kf.write_text("Z" * 40 + "\n")
    monkeypatch.setenv("GHOST_LABS_KEY_FILE", str(kf))
    out = D.redact_text("headers {'X-Labs-Key': '" + "Z" * 40 + "'}")
    assert "Z" * 40 not in out and "<REDACTED_LABS_KEY>" in out
    kf.write_text("short")
    assert D._labs_key_rule() is None


def test_egress_treats_lab_as_code():
    from ghost_agent.memory import egress as E
    assert E.OUTBOUND_TOOLS["lab"] == "url_literals"
    assert "lab" in E.CODE_TOOLS


def test_replays_cannot_call_it():
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    assert "lab" in REPLAY_FORBIDDEN_TOOLS


def test_members_and_leaves_are_not_given_it():
    from ghost_agent.core import agent as A
    assert "lab" not in A._MEMBER_ALLOWED_TOOLS
    from ghost_agent.tools.composed_skills import MACRO_FORBIDDEN_STEP_TOOLS
    assert "lab" in MACRO_FORBIDDEN_STEP_TOOLS


def test_loop_sets_know_it_is_program_output_and_work():
    from ghost_agent.core import agent as A, strikes as S, evidence_gate as G, claim_binding as CB, task_state as TS
    assert "lab" not in S.PROGRAM_OUTPUT_TOOLS and S.is_readwrite_loop_exempt("lab")
    for s in (A._EDIT_CHURN_VERIFY_TOOLS, A._EXTERNAL_EVIDENCE_TOOLS, A._WORK_TOOLS, A._BATCH_WALL_TOOLS,
              A._MUTATING_TOOLS_FOR_DROP_NOTE, G.EVIDENCE_TOOLS, TS.EVIDENCE_TOOLS, CB.SOURCE_EXCLUDED_TOOLS):
        assert "lab" in s


def test_schema_actions_match_the_dispatch():
    props = L.LAB_TOOL_DEFINITION["function"]["parameters"]["properties"]
    assert props["action"]["enum"] == list(L.ACTIONS)
    assert props["op"]["enum"] == list(L.MEMBER_OPS)
    assert "detach" not in json.dumps(L.LAB_TOOL_DEFINITION)      # no detached exec in Phase B



# ------------------------------------------------------------------ review r1 fixes

@pytest.mark.parametrize("action", ["exec", "put", "get", "member", "logs", "http"])
@pytest.mark.parametrize("bad", ["../app/stop#", "../../../gc#", "db/../x", "DB", "", "a" * 40, "db?x=1"])
async def test_a_member_that_is_not_a_name_never_reaches_a_url(tmp_path, fake, scope, action, bad):
    """r1 CRIT: `member` went into the URL path raw — `../app/stop#` became
    POST /v1/labs/<id>/app/stop (a kept app's owner-only route)."""
    c = _ctx(tmp_path)
    scope(c)
    r = await L.tool_lab(c, action=action, lab_id=LAB, member=bad, cmd="x", op="stop",
                         sandbox_path="f", path="/f")
    assert r.status is OutcomeStatus.REJECTED and "needs member" in r
    assert fake.calls == []


def test_only_an_exec_result_is_program_output():
    """r1 MAJOR 2: a Postgres log's ERROR line is the log's subject."""
    from ghost_agent.core.strikes import error_line, is_program_output
    from ghost_agent.tools.outcome import ToolOutcome
    logs = ToolOutcome(f"[{LAB}/db logs]\nERROR:  relation \"x\" does not exist", status=OutcomeStatus.OK)
    assert not is_program_output("lab", logs) and error_line(logs, tool="lab") == ""
    http = ToolOutcome(f"[{LAB}/web:80/] HTTP 200\n<h1>Not Found page title</h1>", status=OutcomeStatus.OK)
    assert error_line(http, tool="lab") == ""
    ok_exec = ToolOutcome(f"[{LAB}/db exit=0 0.2s]\nEXIT CODE: 0\nSTDOUT:\nTraceback (most recent call last)\n"
                          "ValueError: caught and printed", status=OutcomeStatus.OK)
    assert is_program_output("lab", ok_exec) and "ValueError" in error_line(ok_exec, tool="lab")
    assert is_program_output("lab", f"[FAILURE BANNER] x\n[{LAB}/db TIMED OUT after 5 s]")
    assert is_program_output("execute") and is_program_output(None) and not is_program_output("browser")
    err = ToolOutcome("LAB ERROR (unknown_member): no member x\nHINT: lab status", status=OutcomeStatus.REJECTED)
    assert error_line(err, tool="lab").startswith("LAB ERROR (unknown_member)")


def test_evidence_takes_the_tail_of_a_lab_exec():
    from ghost_agent.core import task_state as TS
    from ghost_agent.tools.outcome import ToolOutcome
    out = ToolOutcome(f"[{LAB}/db exit=0 0.2s]\nEXIT CODE: 0\nSTDOUT:\n" + "row\n" * 400 + "THE-RESULT 42",
                      status=OutcomeStatus.OK)
    heads = TS.evidence_heads([{"role": "tool", "name": "lab", "content": out}])
    assert "THE-RESULT 42" in heads


def test_a_status_question_may_still_read_a_lab():
    """r1 MAJOR 6: on a status question during live work only reads run."""
    from ghost_agent.core.agent import _lab_is_read_call
    for act in ("status", "list", "logs", "templates", "spec", "http"):
        assert _lab_is_read_call(json.dumps({"action": act})) and _lab_is_read_call({"action": act.upper()})
    for act in ("create", "exec", "put", "get", "member", "extend", "destroy", ""):
        assert not _lab_is_read_call({"action": act})
    assert not _lab_is_read_call("{bad") and not _lab_is_read_call(None)


def test_a_host_name_is_refused_only_in_a_network_shape(tmp_path):
    """r1 MAJOR 4: with a MagicDNS name, "bipolar disorder study" is a study."""
    c = _ctx(tmp_path, host="labhost:8700")
    for fine in ('echo "bipolar labhost study"', "python analyze.py --label labhost", "ls /workspace/labhost/"):
        assert L.lab_reach_refusal(c, {"command": fine}, "execute") is None, fine
    for bad in ("curl http://labhost:8700/v1/labs", "curl labhost:8700", "ssh root@labhost", "curl http://LABHOST/"):
        assert L.lab_reach_refusal(c, {"command": bad}, "execute") is not None, bad


async def test_no_answer_after_sending_is_not_nothing_ran(tmp_path, fake, scope):
    """r1 MAJOR 3: a read timeout on create — the lab may exist."""
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", "/v1/labs", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert r.reason_code == "lab_no_answer" and "nothing ran" not in r and "lab list" in r
    fake.on("POST", f"/v1/labs/{LAB}/exec", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert "may have run" in r


async def test_create_post_timeout_is_clamped_to_the_turn(tmp_path, fake, scope, monkeypatch):
    c = _ctx(tmp_path)
    scope(c)
    monkeypatch.setattr(L, "_time_left_s", lambda: L.TURN_RESERVE_S + 40)
    fake.on("POST", "/v1/labs", (201, _detail("creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h"}))
    await L.tool_lab(c, action="create", template="pg-single")
    assert fake.calls[0].timeout == 40


async def test_call_maps_connect_and_read_failures_apart(tmp_path, monkeypatch):
    import httpx
    real = httpx.AsyncClient
    c = _ctx(tmp_path, host="100.94.0.1:8700")
    for exc, want in ((httpx.ConnectError("refused"), L.LabsUnreachable),
                      (httpx.ReadTimeout("slow"), L.LabsNoAnswer),
                      (httpx.RemoteProtocolError("dropped"), L.LabsNoAnswer)):
        def handler(request, exc=exc):
            raise exc
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler),
                                                                    **{k: v for k, v in kw.items() if k != "proxy"}))
        with pytest.raises(want):
            await L._call(c, "GET", "/v1/health", origin="probe")


async def test_error_extras_and_fastapi_detail_are_shown(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}/logs", (404, {"error": "unknown_member", "message": "no member web",
                                                  "members": ["db", "replica"]}))
    r = await L.tool_lab(c, action="logs", lab_id=LAB, member="web")
    assert 'members: ["db", "replica"]' in r
    fake.on("POST", f"/v1/labs/{LAB}/http", (422, {"detail": [{"loc": ["body", "port"], "msg": "bad"}]}))
    r = await L.tool_lab(c, action="http", lab_id=LAB, member="web")
    assert "LAB ERROR (http_422):" in r and '"port"' in r


async def test_list_says_host_changed_and_skips_rows_newer_than_the_ask(tmp_path, fake, scope, monkeypatch):
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_cccccccc", owner="conversation:conv1", origin="owner", template="pg-single",
                      host_id="old-host")
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))

    async def list_then_create(*a, **k):        # a create in the same batch lands while listing
        inv.record_create("lab_dddddddd", owner="conversation:conv1", origin="owner", template="pg-single",
                          host_id="new-host")
        return 200, {"labs": []}
    fake.on("GET", "/v1/labs", None)
    real = fake.__call__

    async def routed(context, method, path, **kw):
        if (method, path) == ("GET", "/v1/labs"):
            return await list_then_create()
        return await real(context, method, path, **kw)
    monkeypatch.setattr(L, "_call", routed)
    r = await L.tool_lab(c, action="list")
    assert "lab_cccccccc (pg-single) — not on this labs host's list: it was made on labs host old-host" in r
    assert inv.get("lab_cccccccc")["destroyed_at"] is None     # review 11-05 M2: NOT retired (teardown still sees it)
    assert "lab_dddddddd" not in r and inv.get("lab_dddddddd")["destroyed_at"] is None


async def test_get_refuses_a_folder_and_says_when_it_replaced(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    (c.sandbox_dir / "out").mkdir()
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"x").decode()}))
    for folder in ("out", "/workspace/", "."):
        r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x", sandbox_path=folder)
        assert r.status is OutcomeStatus.REJECTED, folder
    (c.sandbox_dir / "f.txt").write_text("old")
    r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="f.txt")
    assert "REPLACED" in r


async def test_extend_needs_a_ttl_and_a_null_exit_code_is_not_a_code(tmp_path, fake, scope):
    c = _ctx(tmp_path)
    scope(c)
    r = await L.tool_lab(c, action="extend", lab_id=LAB)
    assert r.status is OutcomeStatus.REJECTED and fake.calls == []
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": None, "stdout": "", "stderr": "", "duration_s": 1,
                                                   "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert "EXIT CODE" not in r and "None" not in r and r.status is OutcomeStatus.UNRESOLVED


async def test_time_left_prefers_the_deadline_in_force(monkeypatch):
    from ghost_agent.core import coding_executor as ce
    from ghost_agent.utils import logging as G
    monkeypatch.setattr(ce, "_request_left_s", lambda: 5000.0)
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 300.0 if rid == "r1" else None)
    tok = G.request_id_context.set("r1")
    try:
        assert L._time_left_s() == 300.0
    finally:
        G.request_id_context.reset(tok)
    assert L._time_left_s() == 5000.0


def test_the_key_file_flag_reaches_the_redactor(tmp_path, monkeypatch):
    """r1 MAJOR 5: `--labs-key-file` alone left the key unredacted."""
    import ghost_agent.main as M
    from ghost_agent.distill import redact as D
    kf = tmp_path / "other.key"
    kf.write_text("Q" * 48)
    monkeypatch.delenv("GHOST_LABS_KEY_FILE", raising=False)
    monkeypatch.setattr("sys.argv", ["ghost", "--labs-key-file", str(kf)])
    M.parse_args()
    assert D.redact_text("k=" + "Q" * 48) == "k=<REDACTED_LABS_KEY>"


def test_lab_put_of_a_private_file_after_outside_content_is_refused():
    """r1 minor: `lab put` is itself the network step — no curl needed."""
    from ghost_agent.memory.egress import content_egress_refusal
    from ghost_agent.utils import provenance as P
    from ghost_agent.utils.logging import request_id_context

    class _PM:
        def load(self):
            return {"root": {"name": "Alex Papadakis"}}
    P._reset_for_tests()
    tok = request_id_context.set("own-lab")
    try:
        P.note_user_message("set up a postgres lab", "own-lab")
        ctx = SimpleNamespace(egress_profile=_PM())
        put = {"action": "put", "lab_id": LAB, "member": "db", "sandbox_path": "uploads/tax.pdf", "path": "/x"}
        assert content_egress_refusal("lab", put, ctx) is None          # no outside content yet
        P.note_tool_result("web_search", {}, req_id="own-lab")
        assert content_egress_refusal("lab", put, ctx) is not None
        assert content_egress_refusal("lab", {**put, "sandbox_path": "init.sql"}, ctx) is None
        assert content_egress_refusal("lab", {"action": "get", "sandbox_path": "uploads/x", "path": "/x"}, ctx) is None
    finally:
        request_id_context.reset(tok)
        P._reset_for_tests()


# ------------------------------------------------------------------ review r2 fixes

def test_the_lab_id_note_is_not_an_error_line():
    """r2 MAJOR A: "cannot" in the note made every exit-0 execute naming a lab
    id a same-error strike."""
    from ghost_agent.core.strikes import error_line, error_line_fingerprint
    from ghost_agent.tools.outcome import ToolOutcome
    res = ToolOutcome("EXIT CODE: 0\n{\"ok\": true}", status=OutcomeStatus.OK)
    out = L.with_lab_id_note({"command": f"curl http://{LAB}:8080/"}, res)
    assert "has no route" in out
    assert error_line(out, tool="execute") == "" and error_line_fingerprint(out, tool="execute") == ""
    # r3 minor 7: a file named after a lab is not an attempt to reach it; a refusal ran nothing
    assert L.with_lab_id_note({"command": f"cat results/{LAB}.json"}, res) is res
    rej = ToolOutcome("Not run", status=OutcomeStatus.REJECTED)
    assert L.with_lab_id_note({"command": f"ssh {LAB}"}, rej) is rej


@pytest.mark.parametrize("name,args,prev,pargs,dep", [
    ("lab", {"action": "put", "lab_id": LAB}, "execute", {"command": "tar czf app.tgz src"}, True),   # r3 M2
    ("lab", {"action": "exec", "lab_id": LAB}, "execute", {"command": "x"}, False),
    ("execute", {"command": "cat x.log"}, "lab", {"action": "get", "lab_id": LAB}, True),
    ("execute", {"command": "x"}, "lab", {"action": "exec", "lab_id": LAB}, False),
    ("execute", {"command": "b"}, "execute", {"command": "a"}, True),                                   # §4LO
    ("lab", {"action": "exec", "lab_id": LAB}, "lab", {"action": "put", "lab_id": LAB}, True),
    ("lab", {"action": "destroy", "lab_id": LAB}, "lab", {"action": "exec", "lab_id": LAB}, True),
    ("lab", {"action": "exec", "lab_id": "lab_11111111"}, "lab", {"action": "exec", "lab_id": LAB}, False),  # r3 M3
    ("lab", {"action": "list"}, "lab", {"action": "destroy", "lab_id": LAB}, True),
    ("lab", {"action": "list"}, "lab", {"action": "create"}, True),
    # r4 MAJOR 1: files move in written order across labs
    ("lab", {"action": "put", "lab_id": "lab_11111111"}, "lab", {"action": "get", "lab_id": LAB}, True),
    ("lab", {"action": "get", "lab_id": "lab_11111111"}, "lab", {"action": "put", "lab_id": LAB}, True),
    ("lab", {"action": "get", "lab_id": LAB}, "execute", {"command": "rm -rf out"}, True),
    # r4 MAJOR 2 / r6: a create waits only for a destroy, and no call naming a lab waits for it
    ("lab", {"action": "exec", "lab_id": LAB}, "lab", {"action": "create"}, False),
    ("lab", {"action": "create"}, "lab", {"action": "exec", "lab_id": LAB}, False),
    ("lab", {"action": "create"}, "lab", {"action": "create"}, False),
    # r5 M1/M2
    ("execute", {"command": "tar czf app.tgz src"}, "lab", {"action": "put", "lab_id": LAB}, True),
    ("lab", {"action": "create"}, "lab", {"action": "destroy", "lab_id": LAB}, True),
    ("lab", {"action": "create"}, "lab", {"action": "status", "lab_id": LAB}, False),
    ("lab", {"action": "list"}, "lab", {"action": "status", "lab_id": LAB}, False),
    ("browser", {"url": "x"}, "lab", {"action": "get", "lab_id": LAB}, False),
])
def test_batch_dependencies(name, args, prev, pargs, dep):
    """r2 MAJOR B, redone after r3: written order where it matters, side by side
    where it does not — and every call stays in the cancellable phase."""
    from ghost_agent.core.agent import _batch_call_depends
    assert _batch_call_depends(name, args, prev, pargs) is dep


def test_lab_is_not_in_the_uncuttable_file_phase_and_the_waits_are_built():
    """r3 M1: phase 1 awaits calls plainly (a running call cannot be stopped);
    lab stays in phase 2, where the batch's stop cancels it — and the
    dispatcher builds its waits with `_wire_batch_waits` (tested above), the
    shield with `_shield_exit_banner`, the status refusal with `_starts_new_work`."""
    from ghost_agent.core import agent as A
    import ast
    assert not hasattr(A, "_IN_ORDER_PHASE1_TOOLS")
    tree = ast.parse(Path(A.__file__).read_text())
    helpers = {"_wire_batch_waits", "_shield_exit_banner", "_starts_new_work"}
    used = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name not in helpers:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") in helpers:
                    used.setdefault(n.func.id, set()).add(fn.name)
    assert set(used) == helpers and all(len(v) == 1 for v in used.values()), used
    # the waits' result is what the batch gathers
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "id", "") == "_wire_batch_waits"]
    assert len(assigns) == 1 and assigns[0].targets[0].id == "exec_coros"


def _meta(name, args):
    return (name, None, None, None, None, None, None, None, None, args)


async def test_dependency_waits_order_a_put_after_its_build_and_run_two_labs_together():
    """The dispatcher's own wiring (`_wire_batch_waits`) on real coroutines."""
    import asyncio
    from ghost_agent.core.agent import _wire_batch_waits
    order = []

    async def call(tag, delay):
        order.append(("start", tag))
        await asyncio.sleep(delay)
        order.append(("end", tag))
        return tag
    meta = [_meta("execute", {"command": "tar czf app.tgz"}), _meta("lab", {"action": "put", "lab_id": LAB}),
            _meta("lab", {"action": "exec", "lab_id": "lab_11111111"}), _meta("lab", {"action": "exec", "lab_id": LAB}),
            _meta("browser", {"url": "x"})]
    coros = [(0, call("build", 0.05)), (1, call("put", 0)), (2, call("other-lab", 0.05)), (3, call("exec", 0)),
             (4, call("browser", 0))]
    wired = _wire_batch_waits(coros, meta)
    assert wired[4][1] is coros[4][1] and wired[2][1] is coros[2][1]      # independent: untouched
    assert [i for i, _ in wired] == [0, 1, 2, 3, 4]
    res = await asyncio.gather(*(c for _, c in wired))
    assert res == ["build", "put", "other-lab", "exec", "browser"]
    assert order.index(("end", "build")) < order.index(("start", "put")) < order.index(("start", "exec"))
    assert order.index(("start", "other-lab")) < order.index(("end", "build"))      # side by side


async def test_a_stopped_batch_cancels_a_waiting_lab_call_and_closes_it_unstarted():
    import asyncio
    from ghost_agent.core.agent import _wire_batch_waits
    started = []

    async def call(tag, delay):
        started.append(tag)
        await asyncio.sleep(delay)
    meta = [_meta("lab", {"action": "exec", "lab_id": LAB}), _meta("lab", {"action": "destroy", "lab_id": LAB})]
    second = call("destroy", 0)
    wired = _wire_batch_waits([(0, call("exec", 30)), (1, second)], meta)
    tasks = [asyncio.ensure_future(c) for _, c in wired]
    await asyncio.sleep(0.05)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert started == ["exec"] and second.cr_frame is None          # closed, never started


def test_a_status_question_refuses_new_work_but_not_a_lab_read():
    """r1 MAJOR 6, through the dispatcher's own predicate."""
    from ghost_agent.core.agent import _starts_new_work
    assert not _starts_new_work("lab", json.dumps({"action": "status", "lab_id": LAB}))
    assert _starts_new_work("lab", json.dumps({"action": "exec", "lab_id": LAB}))
    assert _starts_new_work("execute", "{}") and not _starts_new_work("file_system", "{}")


def test_the_shield_banner_brings_back_a_buried_exit_line():
    from ghost_agent.core.agent import _shield_exit_banner
    full = f"[{LAB}/db exit=3 1s]\nEXIT CODE: 3\nSTDOUT:\n" + "x" * 100
    assert _shield_exit_banner("lab", full, "x" * 50).startswith("[FAILURE BANNER] EXIT CODE: 3\n")
    assert _shield_exit_banner("lab", full, full) == full
    assert _shield_exit_banner("lab", "no exit here", "short") == "short"
    assert _shield_exit_banner("execute", "EXIT CODE: 1\nboom", "boom").startswith("[FAILURE BANNER] EXIT CODE: 1")


@pytest.mark.parametrize("text,failed", [
    (f"[{LAB}/db logs]\nERROR:  relation \"t\" does not exist", False),
    (f"[{LAB}/web:80/] HTTP 500\nTraceback (most recent call last):", False),
    (f"[{LAB}/db exit=0 0.2s]\nEXIT CODE: 0\nSTDOUT:\nerror: printed", False),
    (f"[{LAB}/db exit=3 0.2s]\nEXIT CODE: 3\n", True),
    (f"[{LAB}/db TIMED OUT after 5 s]\nTIMED OUT: the command was killed", True),
    ("LAB ERROR (unknown_lab): no such lab", True),
    ("LAB ERROR (no_answer): the labs host …", False),
    (f"[{LAB}/db no exit code 1s]\nThe labs host reported no exit code", False),
])
def test_offline_readers_judge_a_stored_lab_row(text, failed):
    """r2 MAJOR C (+ minor 6, the evidence gate): the same reading offline."""
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    from ghost_agent.core.evidence_gate import _empty_reason
    assert _looks_like_tool_error(text, "lab") is failed
    assert (_empty_reason("lab", {}, text, False) == "lab: error") is failed


@pytest.mark.parametrize("args,untrusted", [
    ({"action": "exec", "cmd": "curl https://example.org"}, True),
    ({"action": "exec", "cmd": "psql -c 'select 1'"}, False),
    ({"action": "exec", "cmd": "bash", "stdin": "wget http://x"}, True),
    ({"action": "http", "member": "web"}, True),
    ({"action": "get", "member": "db", "path": "/x"}, True),
    ({"action": "status"}, False), ({"action": "logs"}, False), ({"action": "list"}, False),
])
def test_lab_output_from_outside_is_outside_content(args, untrusted):
    """r2 MAJOR D: a lab may have Tor egress."""
    from ghost_agent.utils.provenance import is_untrusted_source
    assert is_untrusted_source("lab", args) is untrusted


def test_lab_mutations_reset_the_preflight_guard_and_reads_do_not():
    """r2 MAJOR E."""
    from ghost_agent.core.agent import _call_mutated_world
    for act in ("create", "member", "put", "get", "extend", "destroy"):
        assert _call_mutated_world("lab", {"action": act}, True)
    # r3 minor 2: an exec resets it only when the command changes things, like execute
    assert not _call_mutated_world("lab", {"action": "exec", "cmd": "cat /etc/os-release"}, True)
    assert _call_mutated_world("lab", {"action": "exec", "cmd": "systemctl restart nginx"}, True) == \
        __import__("ghost_agent.core.agent", fromlist=["x"]).looks_mutating_command("systemctl restart nginx")
    for act in ("status", "list", "logs", "http", "spec", "templates"):
        assert not _call_mutated_world("lab", {"action": act}, False)


async def test_a_starting_lab_is_in_flux_not_a_repeatable_failure(tmp_path, fake, scope):
    """r2 MAJOR E: polling a booting app / a not_ready lab never arms the guard."""
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/http", (200, {"member": "web", "port": 80, "path": "/", "status": None,
                                                   "error": "ConnectionRefusedError"}))
    r = await L.tool_lab(c, action="http", lab_id=LAB, member="web")
    assert r.status is OutcomeStatus.FAILED and r.changed_the_world
    fake.on("POST", f"/v1/labs/{LAB}/exec", (409, {"error": "not_ready", "message": "member db is created",
                                                   "state": "created"}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert r.changed_the_world
    # r3 minor 3: an EXITED member stays not-ready until started — a repeatable failure
    fake.on("POST", f"/v1/labs/{LAB}/exec", (409, {"error": "not_ready", "message": "member db is exited",
                                                   "state": "exited"}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert not r.changed_the_world
    fake.on("POST", "/v1/labs", (409, {"error": "budget_exceeded", "message": "no RAM"}))
    r = await L.tool_lab(c, action="create", template="pg-single")
    assert not r.changed_the_world


def test_a_dropped_lab_read_is_not_unapplied_work_and_a_lab_drop_keeps_the_lab():
    """r2 MAJOR F."""
    from ghost_agent.core.agent import _dropped_mutating_names, _dropped_mutation_note
    call = lambda act: {"function": {"name": "lab", "arguments": json.dumps({"action": act, "lab_id": LAB})}}
    assert _dropped_mutating_names([call("status"), call("list")]) == []
    names = _dropped_mutating_names([call("exec")])
    assert names == ["lab"]
    note = _dropped_mutation_note(names)
    assert "Only the results shown above happened" in note and "was actually written" not in note
    assert "Only the results shown above happened" in _dropped_mutation_note(["execute", "lab"])     # r3 minor 6
    mixed = _dropped_mutation_note(["lab", "file_system"], paths=["a.txt"], creating=["a.txt"])
    assert "a.txt" in mixed


async def test_both_streams_fit_under_the_context_shield(tmp_path, fake, scope):
    """r2 MAJOR G: the shield summarised from the first 15,000 chars."""
    c = _ctx(tmp_path)
    scope(c)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 1, "stdout": "o" * 300_000, "stderr": "e" * 300_000
                                                   + "\nFATAL: the real error", "duration_s": 1, "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert len(r) < 16000 and "FATAL: the real error" in r and "EXIT CODE: 1" in r


def test_the_shield_reinjects_the_lab_exit_line():
    """r2 MAJOR G: the shield's EXIT CODE re-injection has a `lab` branch that
    reads through the one anchored helper."""
    import ast
    from ghost_agent.core import agent as A
    tree = ast.parse(Path(A.__file__).read_text())
    hits = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "fname" and any(getattr(c, "value", None) == "lab" for c in n.test.comparators)):
            body = ast.unparse(n)
            if "exec_exit_code" in body and "[FAILURE BANNER] EXIT CODE:" in body:
                hits.append(n.lineno)
    assert len(hits) == 1, hits


async def test_a_read_without_an_answer_is_retryable_and_destroy_preread_is_a_read(tmp_path, fake, scope):
    """r2 minor 1."""
    c = _ctx(tmp_path)
    scope(c)
    fake.on("GET", f"/v1/labs/{LAB}", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(c, action="status", lab_id=LAB)
    assert r.status is OutcomeStatus.FAILED and r.world_changed is False
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.FAILED and "may be gone" not in r
    assert not any(x.method == "DELETE" for x in fake.calls)


async def test_list_says_each_retirement_once(tmp_path, fake, scope):
    """r2 minor 2."""
    import time as _t
    from ghost_agent.memory.lab_inventory import inventory_for
    c = _ctx(tmp_path)
    scope(c)
    inv = inventory_for(c)
    inv.record_create("lab_eeeeeeee", owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_create("lab_ffffffff", owner="conversation:conv1", origin="owner", template="pg-single",
                      host_id="old-host")
    inv.record_create("lab_abababab", owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_destroy("lab_abababab", by="project:z")
    with inv._connect() as con:          # destroyed two days ago
        con.execute("UPDATE labs SET destroyed_at=? WHERE lab_id='lab_abababab'", (_t.time() - 2 * 86400,))
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    fake.on("GET", "/v1/labs/lab_eeeeeeee", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", "/v1/labs/lab_eeeeeeee", (404, {"error": "unknown_lab"}))
    first = await L.tool_lab(c, action="list")
    assert "lab_eeeeeeee (pg-single) — gone" in first and "lab_ffffffff (pg-single) — not on this labs host" in first
    assert "lab_abababab" not in first
    second = await L.tool_lab(c, action="list")
    # the retired row is said once; the other-host row is NOT retired, so it is said on every list (review 11-05 M2)
    assert "lab_eeeeeeee" not in second and "lab_ffffffff" in second


def test_status_question_parse_tolerates_control_characters_and_all_owners_strings():
    """r2 minors 4 + 5."""
    assert L.is_read_call('{"action": "status", "note": "a\tb"}')
    assert L._truthy("false") is False and L._truthy("true") is True and L._truthy(True) is True


async def test_a_probe_destroys_only_its_own_labs(tmp_path, fake, scope):
    """r2 minor 7."""
    c = _ctx(tmp_path)
    scope(c, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:owner-work")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {}))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.REJECTED and not any(x.method == "DELETE" for x in fake.calls)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:conv1")))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.REJECTED          # r13 R2: its key, but an OWNER made it — not the probe's
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="conversation:conv1", origin="probe")))
    r = await L.tool_lab(c, action="destroy", lab_id=LAB)
    assert r.status is OutcomeStatus.OK



async def test_r3_minors(tmp_path, fake, scope):
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.core.evidence_gate import _empty_reason
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    c = _ctx(tmp_path)
    scope(c)
    # 1: a lab the list retired as expired is "gone", not "destroyed by expired"
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_destroy(LAB, by="expired")
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab", "message": "no such lab"}))
    r = await L.tool_lab(c, action="status", lab_id=LAB)
    assert "gone from the labs host" in r and "by expired" not in r
    # 4: a get with no answer wrote nothing
    fake.on("GET", f"/v1/labs/{LAB}/files", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(c, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="x")
    assert r.status is OutcomeStatus.FAILED and r.world_changed is False and "nothing was written" in r
    # 2: a failing exec can be seen twice by the repeat guard
    fake.on("POST", f"/v1/labs/{LAB}/exec", (200, {"exit_code": 1, "stdout": "", "stderr": "x", "duration_s": 1,
                                                   "timed_out": False}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="false")
    assert not r.changed_the_world
    # 5: no answer is not evidence
    assert _empty_reason("lab", {}, "LAB ERROR (no_answer): x", False) == "lab: no answer"
    assert _empty_reason("lab", {}, f"[{LAB}/db no exit code 1s]\nx", False) == "lab: no answer"
    # 8: the two non-exec failure shapes, stored as text
    assert _looks_like_tool_error(f"{LAB} pg-single — failed\nThe lab is failed: read `lab logs` for a member", "lab")
    assert _looks_like_tool_error(f"[{LAB}/web:80/] no answer yet: ConnectionRefusedError", "lab")



async def test_r4_pins(tmp_path, fake, scope, monkeypatch):
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    from ghost_agent.core.evidence_gate import _empty_reason
    c = _ctx(tmp_path)
    scope(c)
    # a second destroy record does not overwrite who destroyed it first
    inv = inventory_for(c)
    inv.record_create(LAB, owner="conversation:conv1", origin="owner", template="pg-single")
    inv.record_destroy(LAB, by="conversation:conv1")
    inv.record_destroy(LAB, by="expired")
    assert inv.get(LAB)["destroyed_by"] == "conversation:conv1"
    # put refuses a file over the cap, before any request
    monkeypatch.setattr(L, "PUT_MAX_BYTES", 4)
    (c.sandbox_dir / "big.bin").write_bytes(b"12345")
    n = len(fake.calls)
    r = await L.tool_lab(c, action="put", lab_id=LAB, member="db", sandbox_path="big.bin", path="/x")
    assert r.status is OutcomeStatus.REJECTED and "max 4" in r and len(fake.calls) == n
    # busy is in flux (the guard must not freeze a retry)
    fake.on("POST", f"/v1/labs/{LAB}/exec", (409, {"error": "busy", "message": "lab is busy"}))
    r = await L.tool_lab(c, action="exec", lab_id=LAB, member="db", cmd="x")
    assert r.changed_the_world
    # r4 minor 3: a stored "no exit code" row is unresolved, whatever its stdout prints
    row = f"[{LAB}/db no exit code 1s]\nSTDOUT:\nEXIT CODE: 1\n"
    assert not _looks_like_tool_error(row, "lab")
    assert _empty_reason("lab", {}, row, False) == "lab: no answer"



def test_r5_a_lab_drop_note_names_the_file_that_did_land():
    from ghost_agent.core.agent import _dropped_mutation_note
    n = _dropped_mutation_note(["lab"], landed=["/workspace/app.py"])
    assert "`/workspace/app.py` DID land" in n and "Only the results shown above happened" in n
    assert "DID land" not in _dropped_mutation_note(["lab"])
