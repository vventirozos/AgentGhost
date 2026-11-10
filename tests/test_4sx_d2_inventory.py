"""§4SX Phase D2: a project owns its runs, services, jobs and labs — one teardown for the tool and the console
(docs/reference/labs/PHASE_D.md D2 + design review r1 H1/H2/H3/M4/M5/M8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import request_scope as RS
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools import labs as L
from ghost_agent.tools import projects as P
from ghost_agent.tools import runs as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_b_lab_tool import FakeLabs  # noqa: E402
from test_4sx_c_run_battery import FakeJobs, FakeSvc  # noqa: E402

LAB_D, LAB_K, LAB_K2 = "lab_0000d001", "lab_0000a002", "lab_0000a003"


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


@pytest.fixture
def env(tmp_path, monkeypatch):
    kf = tmp_path / "labs.key"
    kf.write_text("k" * 64)
    mem = tmp_path / "memory"
    mem.mkdir()
    store = ProjectStore(mem, sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=None, contradiction_log=None, current_project_id=None,
                          conversation_key="conv1", memory_dir=mem, sandbox_manager=None,
                          args=SimpleNamespace(labs_host="127.0.0.1:8700", labs_key_file=str(kf)))
    sups = SimpleNamespace(svc=FakeSvc(), jobs=FakeJobs())
    monkeypatch.setattr(R, "_supervisors", lambda context: (sups.svc, sups.jobs))
    pid = store.create_project("Thing", kind="CODING", goal="g")
    return SimpleNamespace(ctx=ctx, store=store, sups=sups, pid=pid)


def _pid(env):
    return env.pid


@pytest.fixture
def scope():
    toks = []

    def _set(ctx, origin="owner", pid=None):
        toks.append(RS.install(RS.RequestScope(conversation_key=ctx.conversation_key, project_id=pid,
                                               origin=origin, context=ctx)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


def _reg(ctx):
    from ghost_agent.memory.run_registry import registry_for
    return registry_for(ctx)


def _inv(ctx):
    from ghost_agent.memory.lab_inventory import inventory_for
    return inventory_for(ctx)


def _ws_server(env, pid, name="web", lifetime="server"):
    key = f"{pid}:{name}"
    env.sups.svc.ents[key] = {"key": key, "pid": 10, "token": "tok-" + name, "port": 8100}
    return _reg(env.ctx).add(name=name, place="workspace", lifetime=lifetime, command="x", port=8100,
                             owner=f"project:{pid}", origin="owner", backend="services", backend_ref=key,
                             backend_token="tok-" + name)


def _labs(env, fake, pid, labs):
    owner = f"project:{pid}"
    for lab in labs:
        if lab.get("owner", owner) == owner:
            _inv(env.ctx).record_create(lab["lab_id"], owner=owner, origin="owner", template="t", label="",
                                        purpose="", host_id="")
    mine = [dict(lab, owner=lab.get("owner", owner)) for lab in labs]
    fake.on("GET", "/v1/labs", (200, {"labs": mine}))
    for lab in labs:
        fake.on("DELETE", f"/v1/labs/{lab['lab_id']}", (200, {"lab_id": lab["lab_id"], "destroyed": True}))


def _deletes(fake):
    return {c.path.rsplit("/", 1)[-1]: (c.params or {}, c.origin) for c in fake.calls if c.method == "DELETE"}


def _j(out):
    try:
        return json.loads(out)
    except (TypeError, ValueError):
        return {"error": str(out)}


async def _delete(env, pid):
    return _j(await P.tool_manage_projects(env.ctx, action="delete", project_id=pid))


# ------------------------------------------------------------------ delete

async def test_delete_stops_runs_and_destroys_every_lab_of_the_project(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"},
                           {"lab_id": LAB_K, "lifetime": "kept", "lineage": "chess"}])
    r = await _delete(env, pid)
    assert r.get("deleted") is True, r
    d = _deletes(fake)
    assert d[LAB_D] == ({}, "owner")
    assert d[LAB_K] == ({"include_kept": "1", "drop_data": "1"}, "owner")      # alone in its lineage
    assert env.store.get_project(pid) is None
    assert _reg(env.ctx).rows(f"project:{pid}", live_only=True) == []
    assert _inv(env.ctx).rows(f"project:{pid}", live_only=True) == []
    assert "web" in r["teardown"] and LAB_K in r["teardown"]


async def test_a_kept_lab_whose_lineage_another_version_uses_keeps_its_data(env, fake, scope):
    """review r1 H3: deleting v4 must not wipe the data v3 serves."""
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "chess"}])
    everyone = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "chess", "owner": f"project:{pid}"},
                {"lab_id": LAB_K2, "lifetime": "kept", "lineage": "chess", "owner": "project:other"}]
    fake.on("GET", "/v1/labs", (200, {"labs": everyone[:1]}), (200, {"labs": everyone}))
    r = await _delete(env, pid)
    assert r.get("deleted") is True, r
    assert _deletes(fake)[LAB_K][0] == {"include_kept": "1"}
    assert "data volume is KEPT" in r["teardown"] and LAB_K2 in r["teardown"]


async def test_a_lab_that_cannot_be_destroyed_blocks_the_delete(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x"}])
    fake.on("DELETE", f"/v1/labs/{LAB_K}", (403, {"error": "owner_only"}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "NOT deleted" in json.dumps(r) and LAB_K in json.dumps(r)
    assert env.store.get_project(pid) is not None


async def test_labs_host_down_blocks_only_when_the_project_has_labs(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", L.LabsUnreachable("ConnectError"))
    r = await _delete(env, pid)                                     # no lab on record: proceeds with a note
    assert r.get("deleted") is True and "did not answer" in r["teardown"]
    pid2 = env.store.create_project("Other", kind="CODING", goal="g")
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid2}", origin="owner", template="t", label="",
                                purpose="", host_id="")
    r = await _delete(env, pid2)
    assert r.get("deleted") is not True and LAB_D in json.dumps(r)
    assert env.store.get_project(pid2) is not None


async def test_a_run_the_sandbox_cannot_reach_is_recorded_lost_not_blocking(env, fake, scope, monkeypatch):
    pid = _pid(env)
    scope(env.ctx)
    rid = _ws_server(env, pid)
    monkeypatch.setattr(R, "_supervisors", lambda context: (None, None))
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    r = await _delete(env, pid)
    assert r.get("deleted") is True, r
    assert _reg(env.ctx).get(rid)["state"] == "lost"


async def test_a_run_that_will_not_stop_blocks_the_delete(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    env.sups.svc.stop_removes = False
    env.sups.svc.alive = [True]
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "run web" in json.dumps(r)
    assert env.store.get_project(pid) is not None


async def test_a_caller_with_no_request_scope_sends_autopilot_to_the_labs_host(env, fake):
    pid = _pid(env)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    await _delete(env, pid)
    assert _deletes(fake)[LAB_D][1] == "autopilot"


async def test_running_project_jobs_are_cancelled_and_a_refused_cancel_blocks(env, fake, scope, monkeypatch):
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    jobs = FakeJobs()
    jobs.ents = {"j1": {"id": "j1", "state": "running", "project_id": pid},
                 "j2": {"id": "j2", "state": "running", "project_id": "someone-else"}}
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: jobs)
    jobs.cancel = lambda jid: (False, "could not kill — left RUNNING")
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "job j1" in json.dumps(r)
    jobs.cancel = lambda jid: (jobs.ents[jid].update(state="cancelled") or True, "cancelled")
    r = await _delete(env, pid)
    assert r.get("deleted") is True and jobs.ents["j1"]["state"] == "cancelled"
    assert jobs.ents["j2"]["state"] == "running"                     # another project's job untouched


# ------------------------------------------------------------------ archive

async def test_archive_keeps_production_up_and_says_so(env, fake, scope):
    """review r1 H2: app runs and kept labs are production."""
    pid = _pid(env)
    scope(env.ctx)
    srv = _ws_server(env, pid, "web")
    app = _reg(env.ctx).add(name="shop", place=f"{LAB_K}/web", lifetime="app", command="x", port=80,
                            owner=f"project:{pid}", origin="owner", backend="labs_app", backend_ref=f"{LAB_K}/shop")
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"},
                           {"lab_id": LAB_K, "lifetime": "kept", "lineage": "shop"}])
    r = json.loads(await P.tool_manage_projects(env.ctx, action="archive", project_id=pid))
    assert r.get("archived") is True, r
    assert set(_deletes(fake)) == {LAB_D}
    assert _reg(env.ctx).get(srv)["state"] == "stopped" and _reg(env.ctx).get(app)["state"] == "running"
    assert "STILL RUNNING" in r["note"] and "shop" in r["teardown"] and LAB_K in r["teardown"]


# ------------------------------------------------------------------ console routes

class _Req:
    def __init__(self, ctx, origin=None):
        self.headers = {"X-Ghost-Origin": origin} if origin else {}
        self.app = SimpleNamespace(state=SimpleNamespace(context=ctx))


async def test_the_console_delete_is_the_same_teardown_with_the_callers_origin(env, fake, monkeypatch):
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    legs = []
    monkeypatch.setattr(P, "_forget_project_episodes", lambda c, p, t: legs.append("episodes") or 0)
    monkeypatch.setattr(P, "_remove_project_changelog", lambda c, p: legs.append("changelog") or True)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x"}])
    with pytest.raises(Exception) as ei:
        await PR.delete_project(pid, _Req(env.ctx, origin="probe"), hard=True)
    assert getattr(ei.value, "status_code", None) == 403 and fake.calls == []
    resp = await PR.delete_project(pid, _Req(env.ctx), hard=True)
    assert resp.status_code == 204
    assert _deletes(fake)[LAB_K][1] == "owner"                       # review r1 H1: not `autopilot`
    assert legs == ["episodes", "changelog"] and env.store.get_project(pid) is None


async def test_the_console_delete_refuses_while_something_is_left_running(env, fake, monkeypatch):
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (500, {"error": "internal"}))
    resp = await PR.delete_project(pid, _Req(env.ctx), hard=True)
    assert resp.status_code == 409 and LAB_D in json.loads(resp.body)["live_left"][0]
    assert env.store.get_project(pid) is not None


async def test_the_console_archive_by_patch_tears_down_too(env, fake, monkeypatch):
    """review r1 M4."""
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    srv = _ws_server(env, pid)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    await PR.update_project(pid, PR.ProjectUpdate(status="ARCHIVED"), _Req(env.ctx))
    assert set(_deletes(fake)) == {LAB_D} and _reg(env.ctx).get(srv)["state"] == "stopped"
    assert str(env.store.get_project(pid)["status"]).upper() == "ARCHIVED"


# ------------------------------------------------------------------ preview + status

async def test_the_preview_lists_the_inventory_and_a_grown_inventory_needs_a_new_yes(env, fake, scope, monkeypatch):
    """review r1 M8."""
    from ghost_agent.tools import memory as M
    monkeypatch.setattr(M, "_confirm_allowed", lambda plan: "")   # the owner's yes is not what is tested
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    out = json.loads(P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx))
    assert out["confirmation_needed"] is True and "run" in out["would_also_stop"] and "web" in out["would_also_stop"]
    tok = out["confirm_token"]
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t", label="",
                                purpose="", host_id="")
    host = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x", "owner": f"project:{pid}"}]
    again = P._lifecycle_confirmation(env.store, "delete", pid, tok, env.ctx, host)
    assert again is not None and "gained" in str(again) and f"kept:{LAB_K}" in str(again)
    assert f"lab:{LAB_D}" not in str(again)


async def test_status_shows_the_inventory(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    r = json.loads(await P.tool_manage_projects(env.ctx, action="status", project_id=pid))
    assert "web (server @ workspace)" in r["inventory"] and "kept ones" not in r["inventory"]


# ------------------------------------------------------------------ review r2

async def test_two_kept_versions_of_one_lineage_the_last_one_drops_the_data(env, fake, scope):
    """r2 #1: the every-owner list must forget a lab once it is destroyed."""
    pid = _pid(env)
    scope(env.ctx)
    both = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "shop"},
            {"lab_id": LAB_K2, "lifetime": "kept", "lineage": "shop"}]
    _labs(env, fake, pid, both)
    fake.on("DELETE", f"/v1/labs/{LAB_K2}", (200, {"lab_id": LAB_K2, "data_volumes_removed": ["v"]}))
    r = await _delete(env, pid)
    d = _deletes(fake)
    assert d[LAB_K][0] == {"include_kept": "1"} and d[LAB_K2][0] == {"include_kept": "1", "drop_data": "1"}
    assert "with its data (1 volume" in r["teardown"]


async def test_the_host_keeping_the_data_is_reported_not_claimed_dropped(env, fake, scope):
    """r2 #8."""
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x"}])
    fake.on("DELETE", f"/v1/labs/{LAB_K}", (200, {"lab_id": LAB_K, "data_volumes_removed": [],
                                                   "note": "this version had no data mounted; data volumes kept"}))
    r = await _delete(env, pid)
    assert "data was KEPT (this version had no data mounted" in r["teardown"] and "with its data" not in r["teardown"]


@pytest.mark.parametrize("act", ["archive", "delete"])
@pytest.mark.parametrize("origin", ["probe", "unattended", "leaf"])
async def test_only_the_owner_retires_an_owner_project(env, fake, scope, act, origin):
    """r2 #2: the tool path had no origin check (the routes did)."""
    pid = _pid(env)
    scope(env.ctx, origin=origin)
    srv = _ws_server(env, pid)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    r = _j(await P.tool_manage_projects(env.ctx, action=act, project_id=pid))
    assert "nothing was changed" in json.dumps(r), r
    assert fake.calls == [] and _reg(env.ctx).get(srv)["state"] == "running"
    assert str(env.store.get_project(pid)["status"]).upper() != "ARCHIVED"


async def test_a_probe_retires_its_own_project(env, fake, scope):
    pid = env.store.create_project("Probe thing", kind="CODING", goal="g", metadata={"probe_created": True})
    scope(env.ctx, origin="probe")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    r = _j(await P.tool_manage_projects(env.ctx, action="archive", project_id=pid))
    assert r.get("archived") is True, r


async def test_a_job_that_ended_or_was_lost_before_the_cancel_does_not_block(env, fake, scope, monkeypatch):
    """r2 #4: a recreated container marks the job lost inside cancel → nothing runs."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    jobs = FakeJobs()
    jobs.ents = {"j1": {"id": "j1", "state": "running", "project_id": pid}}
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: jobs)

    def _cancel(jid):
        jobs.ents[jid]["state"] = "lost"
        return False, "gone — container was recreated. Marked lost"
    jobs.cancel = _cancel
    jobs.get = lambda jid: jobs.ents.get(jid)
    r = await _delete(env, pid)
    assert r.get("deleted") is True and "Marked lost" in r["teardown"]


async def test_one_raising_cancel_does_not_skip_the_next_and_blocks(env, fake, scope, monkeypatch):
    """r2 #6."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    jobs = FakeJobs()
    jobs.ents = {"j1": {"id": "j1", "state": "running", "project_id": pid},
                 "j2": {"id": "j2", "state": "running", "project_id": pid}}
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: jobs)

    def _cancel(jid):
        if jid == "j1":
            raise RuntimeError("docker exec failed")
        jobs.ents[jid]["state"] = "cancelled"
        return True, "cancelled"
    jobs.cancel = _cancel
    jobs.get = lambda jid: jobs.ents.get(jid)
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "job j1" in json.dumps(r)
    assert jobs.ents["j2"]["state"] == "cancelled"


async def test_an_unexpected_labs_error_counts_the_labs_live(env, fake, scope):
    """r2 #6: a non-Labs exception from _call escaped the whole teardown."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t", label="",
                                purpose="", host_id="")
    fake.on("GET", "/v1/labs", ValueError("bad url"))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and LAB_D in json.dumps(r)


async def test_a_raising_service_state_counts_the_service_live(env, fake, scope, monkeypatch):
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    monkeypatch.setattr(P, "_stop_project_services", lambda c, p, h: 0)
    monkeypatch.setattr(P, "_project_service_entries", lambda c, p: [{"key": f"{p}:web"}])
    import ghost_agent.sandbox.services as SS

    class _Sup:
        def _entry_state(self, e):
            raise RuntimeError("docker down")
    monkeypatch.setattr(SS, "get_service_supervisor", lambda sm: _Sup())
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and f"service {pid}:web" in json.dumps(r)


async def test_the_preview_names_kept_labs_and_a_lab_turning_kept_needs_a_new_yes(env, fake, scope, monkeypatch):
    """r2 #3a/#3b: the digest carries what the HOST says; a lab only the host knows is listed."""
    from ghost_agent.tools import memory as M
    monkeypatch.setattr(M, "_confirm_allowed", lambda plan: "")   # the owner's yes is not what is tested
    pid = _pid(env)
    scope(env.ctx)
    host = [{"lab_id": LAB_D, "lifetime": "disposable", "owner": f"project:{pid}"},
            {"lab_id": LAB_K2, "lifetime": "kept", "lineage": "shop", "owner": f"project:{pid}"}]   # no local row
    out = json.loads(P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx, host))
    assert LAB_K2 in out["would_also_stop"] and "KEPT, lineage shop" in out["would_also_stop"]
    tok = out["confirm_token"]
    host2 = [dict(host[0], lifetime="kept", lineage="x"), host[1]]
    again = P._lifecycle_confirmation(env.store, "delete", pid, tok, env.ctx, host2)
    assert again is not None and f"kept:{LAB_D}" in str(again)


async def test_the_tokenless_yes_also_re_previews_a_grown_inventory(env, fake, scope, monkeypatch):
    pid = _pid(env)
    scope(env.ctx)
    from ghost_agent.tools import memory as M
    plan = {"kind": "project_delete", "project_id": pid, "inventory": []}
    dropped = []
    monkeypatch.setattr(M, "confirmed_earlier_preview", lambda kind, match: ("t0", plan))
    monkeypatch.setattr(M, "_drop_plan", lambda t: dropped.append(t))
    _ws_server(env, pid)                                               # a new server run alone: no new yes
    assert P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx) is None
    host = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x", "owner": f"project:{pid}"}]
    out = P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx, host)
    assert out is not None and "confirmation_needed" in str(out) and dropped == ["t0", "t0"]
    plan["inventory"] = P._inventory_ids(P._project_inventory(env.ctx, pid, host))
    assert P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx, host) is None


async def test_the_console_archive_reports_what_it_left_up(env, fake, monkeypatch):
    """r2 #7."""
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x"}])
    out = await PR.update_project(pid, PR.ProjectUpdate(status="ARCHIVED"), _Req(env.ctx))
    assert out["kept_up"] == [f"kept lab {LAB_K}"] and "LEFT RUNNING" in out["teardown"]


# ------------------------------------------------------------------ review r3

async def test_new_background_work_or_a_disposable_lab_needs_no_new_yes(env, fake, scope):
    """r3 #2: only a kept lab or an app run re-asks (D6 M8); churn must not loop the owner."""
    pid = _pid(env)
    scope(env.ctx)
    out = json.loads(P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx, []))
    _ws_server(env, pid)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t", label="",
                                purpose="", host_id="")
    host = [{"lab_id": LAB_D, "lifetime": "disposable", "owner": f"project:{pid}"}]
    assert P._grew_since_preview(env.ctx, "delete", pid, {"inventory": []}, host) == []
    app = {"run_id": "r1", "name": "shop", "lifetime": "app", "place": "lab"}
    assert P._inventory_ids({"runs": [app]}) == ["app:r1", "run:r1"]
    assert out["confirmation_needed"] is True


async def test_a_lab_only_the_host_knew_is_recorded_so_a_later_unreachable_host_refuses(env, fake, scope):
    """r3 #1."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x",
                                                "owner": f"project:{pid}", "template": "web"}]}))
    host = await P._host_project_labs(env.ctx, pid)
    assert [h["lab_id"] for h in host] == [LAB_K]
    assert [r["lab_id"] for r in _inv(env.ctx).rows(f"project:{pid}", live_only=True)] == [LAB_K]
    fake.on("GET", "/v1/labs", L.LabsNoAnswer("timeout"))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and LAB_K in json.dumps(r)


async def test_a_job_pruned_before_the_cancel_does_not_block(env, fake, scope, monkeypatch):
    """r3 m1."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    jobs = FakeJobs()
    jobs.ents = {"j1": {"id": "j1", "state": "running", "project_id": pid}}
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: jobs)
    jobs.cancel = lambda jid: (jobs.ents.pop(jid) and False, "Error: no sandbox job")
    jobs.get = lambda jid: jobs.ents.get(jid)
    r = await _delete(env, pid)
    assert r.get("deleted") is True, r


async def test_an_unreadable_run_registry_blocks_the_delete(env, fake, scope, monkeypatch):
    """r3 m2."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    reg = _reg(env.ctx)

    def _broken(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(reg, "_connect", _broken)
    assert reg.rows(f"project:{pid}") == []                         # the default stays forgiving
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "run registry could not be read" in json.dumps(r)


async def test_host_leftovers_block_the_delete(env, fake, scope):
    """r3 m4."""
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["ctr-x"]}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "leftovers" in json.dumps(r)


async def test_a_bad_patch_tears_nothing_down(env, fake, monkeypatch):
    """r3 m6."""
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    srv = _ws_server(env, pid)

    def _bad(*a, **k):
        raise ValueError("bad kind")
    monkeypatch.setattr(env.store, "update_project", _bad)
    with pytest.raises(Exception) as ei:
        await PR.update_project(pid, PR.ProjectUpdate(status="ARCHIVED", kind="NOPE"), _Req(env.ctx))
    assert getattr(ei.value, "status_code", None) == 400
    assert _reg(env.ctx).get(srv)["state"] == "running" and fake.calls == []


async def test_a_probe_update_to_archived_changes_no_field(env, fake, scope):
    """r3 m6 (tool path)."""
    pid = _pid(env)
    scope(env.ctx, origin="probe")
    r = _j(await P.tool_manage_projects(env.ctx, action="update", project_id=pid, status="ARCHIVED", goal="hijacked"))
    assert "nothing was changed" in json.dumps(r) and env.store.get_project(pid)["goal"] == "g"


# ------------------------------------------------------------------ review r4 (round-3 findings)

async def test_a_retry_after_leftovers_still_refuses_until_the_host_is_clean(env, fake, scope):
    """r3 #1: the host forgets the lab after a leftover destroy; the retry must ask again, not read 'gone'."""
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["container:x"]}))
    assert (await _delete(env, pid)).get("deleted") is not True
    fake.on("GET", "/v1/labs", (200, {"labs": []}))                  # the host no longer lists it
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "leftovers" in json.dumps(r)
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "destroyed": True}))   # operator cleared
    r = await _delete(env, pid)
    assert r.get("deleted") is True and _inv(env.ctx).rows(f"project:{pid}", live_only=True) == []


async def test_a_lab_made_on_another_labs_host_blocks(env, fake, scope):
    """r3 m1."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t", host_id="old-host")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/health", (200, {"host_id": "new-host"}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "made on labs host old-host" in json.dumps(r)
    assert not [c for c in fake.calls if c.method == "DELETE"]


async def test_archive_says_what_it_could_not_stop(env, fake, scope):
    """r3 #2: live_left was dropped from the summary."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    fake.on("GET", "/v1/labs", L.LabsUnreachable("ConnectError"))
    r = _j(await P.tool_manage_projects(env.ctx, action="archive", project_id=pid))
    assert r.get("archived") is True and "STILL RUNNING (could not be stopped)" in r["teardown"]
    assert LAB_D in r["teardown"] and "could NOT be stopped" in r["note"]


async def test_the_console_soft_delete_reports_what_it_left_up(env, fake, monkeypatch):
    from ghost_agent.api import projects_routes as PR
    pid = _pid(env)
    monkeypatch.setattr(PR, "_store", lambda request: env.store)
    monkeypatch.setattr(PR, "_context", lambda request: env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x"}])
    resp = await PR.delete_project(pid, _Req(env.ctx), hard=False)
    assert resp.status_code == 200 and json.loads(resp.body)["kept_up"] == [f"kept lab {LAB_K}"]
    pid2 = env.store.create_project("Empty", kind="CODING", goal="g")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    assert (await PR.delete_project(pid2, _Req(env.ctx), hard=False)).status_code == 204


async def test_a_yes_while_the_host_is_silent_never_covers_the_labs(env, fake, scope):
    """r3 m2: a lab may have turned kept; with no answer it cannot be checked."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    plan = {"inventory": P._inventory_ids(P._project_inventory(env.ctx, pid))}
    assert P._grew_since_preview(env.ctx, "delete", pid, plan, None) != []
    assert P._grew_since_preview(env.ctx, "delete", pid, plan, []) == []
    _inv(env.ctx).record_destroy(LAB_D, by="t")
    assert P._grew_since_preview(env.ctx, "delete", pid, plan, None) == []   # no labs: nothing to check


async def test_a_replaced_run_is_stopped_by_the_services_step(env, fake, scope, monkeypatch):
    """r3 m3: the services step must really stop the replacement (the earlier test never ran it)."""
    pid = _pid(env)
    scope(env.ctx)
    rid = _ws_server(env, pid)
    key = f"{pid}:web"
    env.sups.svc.ents[key]["token"] = "someone-else"
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    svc = env.sups.svc
    monkeypatch.setattr(P, "_project_service_entries", lambda c, p: [e for e in svc.list_entries()
                                                                     if e["key"].startswith(f"{p}:")])

    def _stop_all(c, p, purge=False):
        n = 0
        for e in list(svc.list_entries()):
            if e["key"].startswith(f"{p}:"):
                svc.stop(e["key"]); n += 1
        return n
    monkeypatch.setattr(P, "_stop_project_services", _stop_all)
    import ghost_agent.sandbox.services as SS
    monkeypatch.setattr(SS, "get_service_supervisor", lambda sm: svc)
    r = await _delete(env, pid)
    assert r.get("deleted") is True and key in svc.stops and key not in svc.ents, r
    assert _reg(env.ctx).get(rid)["state"] == "replaced"
    svc.ents[key] = {"key": key, "pid": 9, "token": "x", "port": 8100}       # a survivor blocks
    svc.stop_removes = False
    pid2 = env.store.create_project("Two", kind="CODING", goal="g")
    svc.ents[f"{pid2}:web"] = {"key": f"{pid2}:web", "pid": 9, "token": "x", "port": 8100}
    r = await _delete(env, pid2)
    assert r.get("deleted") is not True and f"service {pid2}:web" in json.dumps(r)


# ------------------------------------------------------------------ review r5 (round-4 findings)

async def test_a_same_turn_token_keeps_the_plan_for_the_owners_real_yes(env, fake, scope):
    """r4 m2: the growth check ran before the owner's-yes check and dropped the plan."""
    pid = _pid(env)
    scope(env.ctx)
    out = json.loads(P._lifecycle_confirmation(env.store, "delete", pid, None, env.ctx, []))
    host = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "x", "owner": f"project:{pid}"}]
    r = P._lifecycle_confirmation(env.store, "delete", pid, out["confirm_token"], env.ctx, host)
    assert "gained" not in str(r)                                    # refused as not-the-owner first
    from ghost_agent.tools.memory import _resolve_plan
    assert _resolve_plan(out["confirm_token"], "project_delete", match=lambda p: True)[1] is not None


async def test_a_lab_on_record_blocks_while_the_host_identity_is_unknown(env, fake, scope):
    """r4 #2."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t", host_id="h1")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("GET", "/v1/health", L.LabsNoAnswer("timeout"))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "not answering /v1/health" in json.dumps(r)
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (404, {"error": "unknown_lab"}))
    assert (await _delete(env, pid)).get("deleted") is True          # same host, it is gone


async def test_a_host_only_lab_is_recorded_with_the_host_id(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB_D, "lifetime": "disposable",
                                                "owner": f"project:{pid}"}]}))
    fake.on("GET", "/v1/health", (200, {"host_id": "h9"}))
    await P._host_project_labs(env.ctx, pid)
    assert _inv(env.ctx).get(LAB_D)["host_id"] == "h9"


async def test_an_unopenable_store_blocks(env, fake, scope, monkeypatch):
    """r4 m3: registry_for/inventory_for answering None with a memory dir is not 'nothing'."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    import ghost_agent.memory.run_registry as RRm
    monkeypatch.setattr(RRm, "registry_for", lambda c: None)
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "run registry could not be read" in json.dumps(r)
    monkeypatch.undo()
    monkeypatch.setattr(R, "_supervisors", lambda context: (env.sups.svc, env.sups.jobs))
    monkeypatch.setattr(L, "_call", fake)
    import ghost_agent.memory.lab_inventory as LIm
    monkeypatch.setattr(LIm, "inventory_for", lambda c: None)
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "labs" in json.dumps(r).lower()


async def test_the_lab_tool_keeps_a_lab_with_leftovers_on_record(env, fake, scope):
    """r4 m4."""
    scope(env.ctx)
    env.ctx.current_project_id = None
    _inv(env.ctx).record_create(LAB_D, owner="conversation:conv1", origin="owner", template="t")
    fake.on("GET", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "owner": "conversation:conv1", "template": "t"}))
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["container:x"]}))
    out = await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)
    assert "could not remove" in str(out) and _inv(env.ctx).get(LAB_D)["destroyed_at"] is None


# ------------------------------------------------------------------ review r6 (round-5 findings)

async def test_a_lab_with_leftovers_survives_lab_list_and_the_retry_asks_the_host(env, fake, scope):
    """r5: `lab list` retired the kept row as 'expired'; the tool's retry stopped at the GET 404."""
    scope(env.ctx)
    env.ctx.current_project_id = None
    own = "conversation:conv1"
    _inv(env.ctx).record_create(LAB_D, owner=own, origin="owner", template="t")
    fake.on("GET", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "owner": own, "template": "t"}))
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["container:x"]}))
    await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)
    assert _inv(env.ctx).get(LAB_D)["leftovers"] == "container:x"
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    out = await L.tool_lab(env.ctx, action="list")
    assert "still holds container:x" in str(out) and _inv(env.ctx).get(LAB_D)["destroyed_at"] is None
    fake.on("GET", f"/v1/labs/{LAB_D}", (404, {"error": "unknown_lab", "message": "gone"}))
    out = await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)              # still there
    assert "still holds" in str(out) and _inv(env.ctx).get(LAB_D)["destroyed_at"] is None
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "destroyed": True}))
    out = await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)              # operator cleared
    assert "fully gone" in str(out) and _inv(env.ctx).get(LAB_D)["destroyed_at"] is not None


async def test_the_teardown_marks_leftovers_on_the_row(env, fake, scope):
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["network:n"]}))
    await _delete(env, pid)
    assert _inv(env.ctx).get(LAB_D)["leftovers"] == "network:n"
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (200, {"lab_id": LAB_D, "leftovers": ["volume:v"]}))
    await _delete(env, pid)
    assert _inv(env.ctx).get(LAB_D)["leftovers"] == "volume:v"


def test_an_inventory_from_before_the_leftovers_column_is_migrated(tmp_path):
    import sqlite3
    from ghost_agent.memory.lab_inventory import LabInventory
    c = sqlite3.connect(str(tmp_path / "lab_inventory.db"))
    c.execute("CREATE TABLE labs (lab_id TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL "
              "DEFAULT '', template TEXT NOT NULL DEFAULT '', label TEXT NOT NULL DEFAULT '', purpose TEXT NOT NULL "
              "DEFAULT '', host_id TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, destroyed_at REAL, "
              "destroyed_by TEXT NOT NULL DEFAULT '')")
    c.execute("INSERT INTO labs (lab_id, created_at) VALUES ('lab_00000001', 1)")
    c.commit(); c.close()
    inv = LabInventory(tmp_path)
    inv.mark_leftovers("lab_00000001", "container:z")
    assert inv.get("lab_00000001")["leftovers"] == "container:z"


async def test_a_probe_never_retries_another_owners_leftover_lab(env, fake, scope):
    scope(env.ctx, origin="probe")
    env.ctx.current_project_id = None
    _inv(env.ctx).record_create(LAB_D, owner="project:owners", origin="owner", template="t")
    _inv(env.ctx).mark_leftovers(LAB_D, "container:x")
    fake.on("GET", f"/v1/labs/{LAB_D}", (404, {"error": "unknown_lab", "message": "gone"}))
    await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)
    assert not [c for c in fake.calls if c.method == "DELETE"]


async def test_a_leftover_lab_of_another_host_is_not_cleared_by_this_hosts_404(env, fake, scope):
    """r6."""
    scope(env.ctx)
    env.ctx.current_project_id = None
    _inv(env.ctx).record_create(LAB_D, owner="conversation:conv1", origin="owner", template="t", host_id="A")
    _inv(env.ctx).mark_leftovers(LAB_D, "container:x")
    fake.on("GET", f"/v1/labs/{LAB_D}", (404, {"error": "unknown_lab", "message": "no lab"}))
    fake.on("GET", "/v1/health", (200, {"host_id": "B"}))
    out = await L.tool_lab(env.ctx, action="destroy", lab_id=LAB_D)
    assert "made on labs host A" in str(out) and _inv(env.ctx).get(LAB_D)["destroyed_at"] is None
    assert not [c for c in fake.calls if c.method == "DELETE"]


# ------------------------------------------------------------------ battery survivors (round 1)

def test_the_inventory_and_digest_take_only_this_projects_things(env, monkeypatch):
    """I3/I4."""
    pid = env.pid
    jobs = FakeJobs()
    jobs.ents = {"j1": {"id": "j1", "state": "running", "project_id": pid},
                 "j2": {"id": "j2", "state": "running", "project_id": "other"},
                 "j3": {"id": "j3", "state": "done", "project_id": pid}}
    import ghost_agent.sandbox.jobs as SJ
    monkeypatch.setattr(SJ, "get_job_supervisor", lambda sm: jobs)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    inv = P._project_inventory(env.ctx, pid)
    assert [e["id"] for e in inv["jobs"]] == ["j1"]
    assert P._inventory_ids(inv) == [f"job:j1", f"lab:{LAB_D}"]


async def test_the_runs_are_stopped_with_the_callers_origin(env, fake, scope, monkeypatch):
    """R6."""
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    seen = []
    real = R._stop

    async def _spy(context, reg, row, origin):
        seen.append(origin)
        return await real(context, reg, row, origin)
    monkeypatch.setattr(R, "_stop", _spy)
    await P._teardown_project(env.ctx, pid, hard=True, origin="probe")
    assert seen == ["probe"]


async def test_no_labs_host_configured_with_labs_on_record_blocks(env, fake, scope):
    """L2."""
    pid = _pid(env)
    scope(env.ctx)
    env.ctx.args.labs_host = None
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "no labs host is configured" in json.dumps(r)


async def test_another_owners_lab_in_the_host_reply_is_never_destroyed(env, fake, scope):
    """L4."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB_K2, "lifetime": "disposable", "owner": "project:x"}]}))
    r = await _delete(env, pid)
    assert r.get("deleted") is True and not [c for c in fake.calls if c.method == "DELETE"]


async def test_a_kept_lab_of_another_lineage_does_not_keep_the_data(env, fake, scope):
    """L10."""
    pid = _pid(env)
    scope(env.ctx)
    mine = [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "chess", "owner": f"project:{pid}"}]
    other = {"lab_id": LAB_K2, "lifetime": "kept", "lineage": "shop", "owner": "project:other"}
    _labs(env, fake, pid, mine)
    fake.on("GET", "/v1/labs", (200, {"labs": mine}), (200, {"labs": mine + [other]}))
    await _delete(env, pid)
    assert _deletes(fake)[LAB_K][0] == {"include_kept": "1", "drop_data": "1"}


@pytest.mark.parametrize("reply", [(404, {"error": "not_routed"}), (500, {"error": "internal"}),
                                   ValueError("bad url")])
async def test_a_failed_lab_delete_counts_live_and_the_next_lab_is_still_tried(env, fake, scope, reply):
    """L12/N17."""
    pid = _pid(env)
    scope(env.ctx)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"},
                           {"lab_id": LAB_K2, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", reply)
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and f"lab {LAB_D}" in json.dumps(r)
    assert LAB_K2 in _deletes(fake)


async def test_a_non_200_lab_list_with_labs_on_record_blocks(env, fake, scope):
    """L15."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    fake.on("GET", "/v1/labs", (500, {"error": "internal"}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and LAB_D in json.dumps(r) and "HTTP 500" in json.dumps(r)


async def test_the_model_facing_preview_reads_the_labs_host(env, fake, scope):
    """N11: the wrapper passes its own host read to the confirmation."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB_K, "lifetime": "kept", "lineage": "chess",
                                                "owner": f"project:{pid}"}]}))
    out = _j(await P.tool_manage_projects_for_model(env.ctx, action="delete", project_id=pid))
    assert "KEPT, lineage chess" in json.dumps(out), out


async def test_the_refusal_says_what_was_already_stopped(env, fake, scope):
    """N19."""
    pid = _pid(env)
    scope(env.ctx)
    _ws_server(env, pid)
    _labs(env, fake, pid, [{"lab_id": LAB_D, "lifetime": "disposable"}])
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (500, {"error": "internal"}))
    r = await _delete(env, pid)
    assert "already is" in r["error"] and "stopped/destroyed: run web" in r["error"]


async def test_an_unlisted_lab_whose_retry_fails_counts_live(env, fake, scope):
    """Q8."""
    pid = _pid(env)
    scope(env.ctx)
    _inv(env.ctx).record_create(LAB_D, owner=f"project:{pid}", origin="owner", template="t")
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    fake.on("DELETE", f"/v1/labs/{LAB_D}", (500, {"error": "internal"}))
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and LAB_D in json.dumps(r)
    assert _inv(env.ctx).get(LAB_D)["destroyed_at"] is None


async def test_an_unreadable_lab_inventory_blocks(env, fake, scope, monkeypatch):
    """M11."""
    pid = _pid(env)
    scope(env.ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    inv = _inv(env.ctx)

    def _broken(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(inv, "_connect", _broken)
    r = await _delete(env, pid)
    assert r.get("deleted") is not True and "labs (could not be checked)" in json.dumps(r)
