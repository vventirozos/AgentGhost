"""§4SX Phase D3: kept machines through the `lab` tool, `app` runs, the release gate, the down notice and the
operator secrets route (docs/labs/PHASE_D.md D3-design + its review r1)."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import request_scope as RS
from ghost_agent.tools import labs as L
from ghost_agent.tools import runs as R
from ghost_agent.tools.outcome import OutcomeStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_b_lab_tool import FakeLabs  # noqa: E402

PID = "p0000000a1b2"
OWN = f"project:{PID}"
LAB, LAB2 = "lab_0000c001", "lab_0000c002"


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


@pytest.fixture
def ctx(tmp_path):
    kf = tmp_path / "labs.key"
    kf.write_text("k" * 64)
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "memory").mkdir()
    return SimpleNamespace(args=SimpleNamespace(labs_host="127.0.0.1:8700", labs_key_file=str(kf)),
                           sandbox_dir=tmp_path / "sandbox", memory_dir=tmp_path / "memory", tor_proxy=None,
                           current_project_id=PID, conversation_key="conv1", sandbox_manager=None)


@pytest.fixture
def scope():
    toks = []

    def _set(c, origin="owner"):
        toks.append(RS.install(RS.RequestScope(conversation_key=c.conversation_key, project_id=c.current_project_id,
                                               origin=origin, context=c)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


def _reg(c):
    from ghost_agent.memory.run_registry import registry_for
    return registry_for(c)


def _app(state="up", mounted=True, name="shop", **kw):
    return {"name": name, "lineage": PID, "member": "web", "port": 8080, "cmd": "python3 app.py", "state": state,
            "mounted": mounted, "url": "https://labs.ts.net:20001/", "restarts": 0, "last_error": None,
            "secrets": [], **kw}


def _det(lab=LAB, owner=OWN, lifetime="kept", app=None, **kw):
    d = {"lab_id": lab, "owner": owner, "origin": "owner", "lifetime": lifetime, "lineage": PID, "template": "web"}
    if app is not None:
        d["app"] = app
    d.update(kw)
    return d


def _calls(fake, method=None):
    return [(c.method, c.path, c.json, c.origin) for c in fake.calls if method is None or c.method == method]


async def _lab(c, **kw):
    return await L.tool_lab(c, **kw)


# ------------------------------------------------------------------ keep

async def test_keep_sends_the_projects_lineage_and_records_one_app_run(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable")))
    fake.on("POST", f"/v1/labs/{LAB}/keep", (200, _det(app=_app(state="stopped", secrets=["API_KEY"]))))
    out = await _lab(ctx, action="keep", lab_id=LAB,
                     params={"app": {"name": "shop", "member": "web", "port": 8080, "cmd": "python3 app.py",
                                     "junk": 1}, "secrets": ["API_KEY"]})
    assert out.status is OutcomeStatus.OK, str(out)
    sent = [c for c in fake.calls if c.path.endswith("/keep")][0].json
    assert sent["lineage"] == PID and "junk" not in sent["app"] and sent["secrets"] == ["API_KEY"]
    assert "API_KEY hold generated values" in str(out)
    rows = _reg(ctx).rows(OWN, live_only=True)
    assert [(r["name"], r["lifetime"], r["backend"], r["backend_ref"]) for r in rows] == \
        [("shop", "app", "labs_app", f"{PID}/shop")]
    # a second version of the same app: no second row
    fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, lifetime="disposable")))
    fake.on("POST", f"/v1/labs/{LAB2}/keep", (200, _det(lab=LAB2, app=_app(state="staged", mounted=False))))
    out = await _lab(ctx, action="keep", lab_id=LAB2, params={"app": {"name": "shop", "member": "web", "port": 8080}})
    assert "STAGED next version" in str(out) and len(_reg(ctx).rows(OWN, live_only=True)) == 1


@pytest.mark.parametrize("params,why", [
    ({"app": {"name": "shop", "member": "web", "port": 1}, "lineage": "other"}, "lineage is set by the system"),
    ({"app": {"name": "shop", "member": "web", "port": 1}, "secret_values": {"A": "x"}}, "never passed"),
    ({"app": {"name": "shop"}}, "keep needs params.app"),
    ("not json", "params must be a JSON object"),
])
async def test_keep_refuses_bad_inputs_before_the_host(ctx, fake, scope, params, why):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable")))
    out = await _lab(ctx, action="keep", lab_id=LAB, params=params)
    assert why in str(out) and not [c for c in fake.calls if c.method == "POST"]


async def test_keep_of_a_conversation_lab_says_make_it_in_a_project(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(owner="conversation:conv1", lifetime="disposable")))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "a", "member": "web", "port": 80}})
    assert "create the lab there first" in str(out)


async def test_keep_refuses_a_name_a_non_app_run_holds(ctx, fake, scope):
    scope(ctx)
    _reg(ctx).add(name="shop", place="workspace", lifetime="server", command="x", port=8000, owner=OWN,
                  origin="owner", backend="services", backend_ref=f"{PID}:shop")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable")))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "member": "web", "port": 80}})
    assert "already running" in str(out) and not [c for c in fake.calls if c.method == "POST"]


@pytest.mark.parametrize("origin", ["unattended", "leaf"])
@pytest.mark.parametrize("action", sorted(L.KEPT_ACTIONS))
async def test_unattended_work_never_reaches_kept_actions(ctx, fake, scope, origin, action):
    scope(ctx, origin=origin)
    out = await _lab(ctx, action=action, lab_id=LAB, op="start", params={"from_lab": LAB2})
    assert "not available to unattended work" in str(out) and fake.calls == []


class _Store:
    def __init__(self, probe_created):
        self.md = {"probe_created": True} if probe_created else {}

    def get_project(self, pid):
        return {"id": pid, "metadata": self.md}


async def test_a_probe_is_passed_to_the_host_only_in_a_probe_project(ctx, fake, scope):
    scope(ctx, origin="probe")
    ctx.project_store = _Store(False)                       # code r1 #1: the owner's project
    out = await _lab(ctx, action="app", lab_id=LAB, op="stop")
    assert "only in a project a probe created" in str(out) and fake.calls == []
    ctx.project_store = _Store(True)
    fake.on("POST", f"/v1/labs/{LAB}/app/stop", (403, {"error": "owner_only", "message": "app_stop is an owner action"}))
    out = await _lab(ctx, action="app", lab_id=LAB, op="stop")
    assert _calls(fake)[0][3] == "probe" and "owner_only" in str(out)


# ------------------------------------------------------------------ deploy

def _names(archive_b64):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64)), mode="r:gz") as t:
        return sorted(t.getnames())


async def test_deploy_tars_the_folder_without_links_secrets_or_caches(ctx, fake, scope, tmp_path):
    scope(ctx)
    app = ctx.sandbox_dir / "projects" / PID / "site"
    (app / "static").mkdir(parents=True)
    (app / "app.py").write_text("print(1)")
    (app / "static" / "a.css").write_text("x")
    (app / ".env").write_text("SECRET=1")
    (app / ".env.local").write_text("SECRET=2")
    (app / "node_modules").mkdir()
    (app / "node_modules" / "big.js").write_text("x")
    (app / ".git").mkdir()
    (app / ".git" / "HEAD").write_text("x")
    outside = tmp_path / "host_secret.txt"
    outside.write_text("HOST")
    os.symlink(outside, app / "leak.txt")
    os.symlink(tmp_path, app / "leakdir")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site",
                                                               "requirements": {"pip": ["flask"]}})
    assert out.status is OutcomeStatus.OK, str(out)
    body = [c for c in fake.calls if c.path.endswith("/deploy")][0].json
    assert _names(body["archive_b64"]) == ["app.py", "static/a.css"]
    assert body["requirements"] == {"pip": ["flask"], "apt": []}
    assert "leak.txt" in str(out) and "leakdir" in str(out)


async def test_deploy_size_limit_and_failed_probe_and_no_answer(ctx, fake, scope, monkeypatch):
    scope(ctx)
    d = ctx.sandbox_dir / "site"
    d.mkdir()
    (d / "a.bin").write_bytes(os.urandom(2048))
    monkeypatch.setattr(L, "DEPLOY_MAX_BYTES", 1024)
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": "/workspace/site"})
    assert "over" in str(out) and fake.calls == []
    monkeypatch.setattr(L, "DEPLOY_MAX_BYTES", 20 * 1024 * 1024)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(state="crashloop",
                                                                                last_error="exit 1"),
                                                     "probe": {"ok": False, "error": "connection refused"}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": "/workspace/site"})
    assert out.status is OutcomeStatus.FAILED and str(out).startswith("LAB ERROR (app_not_up)")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", L.LabsNoAnswer("timeout"))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": "/workspace/site"})
    assert out.status is OutcomeStatus.UNRESOLVED and "UNKNOWN" in str(out)


async def test_deploy_refuses_a_folder_outside_the_workspace(ctx, fake, scope):
    scope(ctx)
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": "/workspace/../../etc"})
    assert "outside the workspace" in str(out) and fake.calls == []


def test_the_tar_never_follows_a_symlinked_root_child(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "ok.txt").write_text("1")
    os.symlink("/etc/passwd", root / "pw")
    data, skipped, err = L.tar_folder(root)
    assert err == "" and skipped == ["pw"] and "pw" not in _names(base64.b64encode(data))


# ------------------------------------------------------------------ switch / rollback / backup / restore / rehearse / app

async def test_switch(ctx, fake, scope):
    scope(ctx)
    out = await _lab(ctx, action="switch", lab_id=LAB2, params={})
    assert "from_lab" in str(out) and fake.calls == []
    fake.on("POST", f"/v1/labs/{LAB2}/switch", (200, {"switched": False, "reversed": True, "error": "probe failed"}))
    out = await _lab(ctx, action="switch", lab_id=LAB2, params={"from_lab": LAB})
    assert str(out).startswith("LAB ERROR (switch_failed)") and "old version serves again" in str(out)
    assert [c.json for c in fake.calls if c.path.endswith("/switch")][-1] == {"from_lab": LAB}
    fake.on("POST", f"/v1/labs/{LAB2}/switch", (200, {"switched": True, "from_lab": LAB, "app": _app(),
                                                      "probe": {"ok": True}, "rollback_until": 1_900_000_000}))
    out = await _lab(ctx, action="switch", lab_id=LAB2, params={"from_lab": LAB})
    assert out.status is OutcomeStatus.OK and "serves now" in str(out) and "lab rollback" in str(out)


async def test_rollback_restore_backup_backups(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB2}/rollback", (200, {"lab_id": LAB, "rolled_back_from": LAB2, "app": _app(),
                                                        "probe": {"ok": True}}))
    out = await _lab(ctx, action="rollback", lab_id=LAB2, params={"restore_snapshot": "true"})
    assert f"{LAB} serves again" in str(out)
    assert [c.json for c in fake.calls if c.path.endswith("/rollback")][-1] == {"restore_snapshot": True}
    fake.on("POST", f"/v1/labs/{LAB2}/rollback", (200, {"rolled_back": False, "error": "no data"}))
    assert str(await _lab(ctx, action="rollback", lab_id=LAB2)).startswith("LAB ERROR (rollback_failed)")
    assert "backup_id" in str(await _lab(ctx, action="restore", lab_id=LAB, params={}))
    fake.on("POST", f"/v1/labs/{LAB}/restore", (200, {"restored": "b1", "pre_restore": {"backup_id": "b0"},
                                                      "app": _app(), "probe": {"ok": True}}))
    assert "pre-restore backup b0" in str(await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "b1"}))
    fake.on("POST", f"/v1/labs/{LAB}/backups", (200, {"backup_id": "b2", "kind": "manual", "bytes": 10}))
    assert "backup b2" in str(await _lab(ctx, action="backup", lab_id=LAB))
    fake.on("GET", f"/v1/labs/{LAB}/backups", (200, {"backups": [{"backup_id": "b2", "kind": "manual",
                                                                  "member": "web", "ts": 1_800_000_000}]}))
    out = await _lab(ctx, action="backups", lab_id=LAB)
    assert "1 backup(s)" in str(out) and out.world_changed is False


async def test_rehearse_and_app_ops(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/rehearse", (200, {"twin": {"probe": {"ok": False, "error": "500"}}, "app": _app()}))
    assert str(await _lab(ctx, action="rehearse", lab_id=LAB)).startswith("LAB ERROR (rehearsal_failed)")
    fake.on("POST", f"/v1/labs/{LAB}/rehearse", (200, {"twin": {"probe": {"ok": True}}, "app": _app()}))
    assert "rehearsal PASSED" in str(await _lab(ctx, action="rehearse", lab_id=LAB))
    assert "op one of" in str(await _lab(ctx, action="app", lab_id=LAB, op="kill"))
    fake.on("POST", f"/v1/labs/{LAB}/app/start", (200, {"app": _app(state="crashloop"), "probe": {"ok": False}}))
    assert str(await _lab(ctx, action="app", lab_id=LAB, op="start")).startswith("LAB ERROR (app_not_up)")
    fake.on("POST", f"/v1/labs/{LAB}/app/stop", (200, {"app": _app(state="stopped"), "probe": {"ok": False}}))
    assert (await _lab(ctx, action="app", lab_id=LAB, op="stop")).status is OutcomeStatus.OK


def test_kept_failures_read_as_failures_in_stored_history():
    from ghost_agent.distill.outcome_heuristics import _lab_result_failed
    for t in ("LAB ERROR (switch_failed): lab_x: the switch FAILED", "LAB ERROR (app_not_up): lab_x: deployed"):
        assert _lab_result_failed(t)


# ------------------------------------------------------------------ `app` runs

def _app_row(c, origin="owner"):
    return _reg(c).add(name="shop", place=LAB, lifetime="app", command="x", port=8080, owner=OWN, origin=origin,
                       backend="labs_app", backend_ref=f"{PID}/shop")


def _serving(fake, serving_state="up", staged=False, **kw):
    labs = [{"lab_id": LAB, "lifetime": "kept", "lineage": PID, "owner": OWN},
            {"lab_id": "lab_0000dd01", "lifetime": "disposable", "owner": OWN}]
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(state=serving_state, **kw))))
    if staged:
        labs.append({"lab_id": LAB2, "lifetime": "kept", "lineage": PID, "owner": OWN})
        fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, app=_app(state="staged", mounted=False))))
    fake.on("GET", "/v1/labs", (200, {"labs": labs}))


@pytest.mark.parametrize("host,state", [("up", "running"), ("restarting", "running"), ("stopped", "app_stopped"),
                                        ("crashloop", "crashloop")])
async def test_an_app_run_is_probed_on_its_serving_lab(ctx, fake, host, state):
    rid = _app_row(ctx)
    _serving(fake, host, staged=True)
    st, note, det = await R._probe(ctx, _reg(ctx).get(rid), "owner")
    assert st == state and det["lab_id"] == LAB and f"next version {LAB2} staged" in note
    assert "lab_0000dd01" not in [c.path.rsplit("/", 1)[-1] for c in fake.calls if c.path != "/v1/labs"]


async def test_no_serving_version_is_staged_and_no_lab_is_lost_and_silence_is_unknown(ctx, fake):
    rid = _app_row(ctx)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB2, "lifetime": "kept", "lineage": PID}]}))
    fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, app=_app(state="staged", mounted=False))))
    assert (await R._probe(ctx, _reg(ctx).get(rid), "owner"))[0] == "staged"
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    assert (await R._probe(ctx, _reg(ctx).get(rid), "owner"))[0] == "lost"
    fake.on("GET", "/v1/labs", L.LabsNoAnswer("t"))
    assert (await R._probe(ctx, _reg(ctx).get(rid), "owner"))[0] == "unknown"


async def test_run_stop_start_restart_url_logs_of_an_app(ctx, fake, scope):
    scope(ctx)
    rid = _app_row(ctx)
    _serving(fake, "up")
    fake.on("POST", f"/v1/labs/{LAB}/app/stop", (200, {"app": _app(state="stopped"), "probe": {"ok": False}}))
    out = await R.tool_run(ctx, action="stop", name="shop")
    assert out.status is OutcomeStatus.OK, str(out)
    row = _reg(ctx).get(rid)
    assert row["state"] == "app_stopped" and row["ended_at"] is None             # review r1 H4
    _serving(fake, "stopped")
    fake.on("POST", f"/v1/labs/{LAB}/app/start", (200, {"app": _app(state="up"), "probe": {"ok": True}}))
    out = await R.tool_run(ctx, action="start", name="shop")
    assert out.status is OutcomeStatus.OK and _reg(ctx).get(rid)["state"] == "running", str(out)
    _serving(fake, "up")
    fake.on("POST", f"/v1/labs/{LAB}/app/restart", (200, {"app": _app(state="crashloop"), "probe": {"ok": False}}))
    out = await R.tool_run(ctx, action="restart", name="shop")
    assert out.status is OutcomeStatus.FAILED and str(out).startswith("RUN ERROR (app_not_up)")
    _serving(fake, "up")
    assert "https://labs.ts.net:20001/" in str(await R.tool_run(ctx, action="url", name="shop"))
    fake.on("GET", f"/v1/labs/{LAB}/logs", (200, {"logs": "GET / 200"}))
    out = await R.tool_run(ctx, action="logs", name="shop")
    assert "GET / 200" in str(out) and [c.params for c in fake.calls if c.path.endswith("/logs")][-1]["member"] == "web"
    status = await R.tool_run(ctx, action="status")
    assert "https://labs.ts.net:20001/" in str(status)


async def test_run_cannot_create_an_app_and_a_probe_cannot_drive_the_owners(ctx, fake, scope):
    scope(ctx)
    out = await R.tool_run(ctx, action="start", name="new", command="x", where="workspace", lifetime="app")
    assert "made by `lab keep`" in str(out)
    _app_row(ctx, origin="owner")
    RS.reset(RS.install(RS.RequestScope(conversation_key="conv1", project_id=PID, origin="probe", context=ctx)))
    scope(ctx, origin="probe")
    for action in ("start", "stop", "restart"):
        out = await R.tool_run(ctx, action=action, name="shop")
        assert "a probe changes only its own runs" in str(out), action
    assert not [c for c in fake.calls if c.method == "POST"]


async def test_destroying_the_kept_lab_ends_its_app_run(ctx, fake, scope):
    scope(ctx)
    rid = _app_row(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB, "destroyed": True}))
    fake.on("GET", "/v1/labs", (200, {"labs": []}))
    await _lab(ctx, action="destroy", lab_id=LAB)
    assert _reg(ctx).get(rid)["state"] == "lost"


# ------------------------------------------------------------------ release gate

async def test_the_release_gate_is_every_app_up(ctx, fake, scope):
    from ghost_agent.tools import projects as P
    scope(ctx)
    assert await P._release_app_gate(ctx, PID) is None                    # no app: the old rehearsal applies
    _app_row(ctx)
    _serving(fake, "up")
    g = await P._release_app_gate(ctx, PID)
    assert g["ok"] is True and g["apps"][0]["url"] == "https://labs.ts.net:20001/" and g["apps"][0]["lab_id"] == LAB
    assert not [c for c in fake.calls if c.path.endswith("/rehearse")]   # design r1 H5
    _serving(fake, "crashloop", last_error="exit 1")
    g = await P._release_app_gate(ctx, PID)
    assert g["ok"] is False and "must be UP" in g["detail"]


# ------------------------------------------------------------------ down notice

async def test_the_down_notice_needs_two_polls_and_fires_once_per_period(ctx, fake, monkeypatch):
    sent = []
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: sent.append(msg) or True)
    rid = _app_row(ctx)
    _serving(fake, "crashloop", last_error="exit 1")
    await R.app_down_sweep(ctx)
    assert sent == []                                                     # one poll is not enough
    await R.app_down_sweep(ctx)
    assert len(sent) == 1 and "shop" in sent[0] and "crashloop" in sent[0]
    await R.app_down_sweep(ctx)
    assert len(sent) == 1                                                 # same period
    _serving(fake, "up")
    await R.app_down_sweep(ctx)
    assert _reg(ctx).meta_of(_reg(ctx).get(rid)) == {}
    _serving(fake, "crashloop")
    await R.app_down_sweep(ctx)
    await R.app_down_sweep(ctx)
    assert len(sent) == 2                                                 # a new period


async def test_the_down_notice_is_marked_only_when_written_and_ignores_silence_and_owner_stops(ctx, fake, monkeypatch):
    sent = []
    ok = {"v": False}
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: sent.append(msg) or ok["v"])
    rid = _app_row(ctx)
    _serving(fake, "crashloop")
    await R.app_down_sweep(ctx)
    await R.app_down_sweep(ctx)
    assert len(sent) == 1 and not _reg(ctx).meta_of(_reg(ctx).get(rid)).get("down_notified")
    ok["v"] = True
    await R.app_down_sweep(ctx)
    assert len(sent) == 2 and _reg(ctx).meta_of(_reg(ctx).get(rid)).get("down_notified")
    fake.on("GET", "/v1/labs", L.LabsNoAnswer("t"))
    before = _reg(ctx).get(rid)["meta"]
    out = await R.app_down_sweep(ctx)
    assert out["unknown"] == 1 and _reg(ctx).get(rid)["meta"] == before
    rid2 = _reg(ctx).add(name="b", place=LAB, lifetime="app", command="x", port=1, owner="project:other",
                         origin="owner", backend="labs_app", backend_ref="other/b")
    assert R._app_is_down("app_stopped", {"last_error": "x", "stopped_by_owner": True}) is False
    assert R._app_is_down("app_stopped", {"last_error": None}) is False
    assert R._app_is_down("app_stopped", {"last_error": "member down"}) is True
    assert rid2


async def test_one_notice_names_every_app_newly_down(ctx, fake, monkeypatch):
    sent = []
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: sent.append((msg, list(ids))) or True)
    _app_row(ctx)
    _reg(ctx).add(name="blog", place=LAB, lifetime="app", command="x", port=1, owner=OWN, origin="owner",
                  backend="labs_app", backend_ref=f"{PID}/blog")
    labs = [{"lab_id": LAB, "lifetime": "kept", "lineage": PID}, {"lab_id": LAB2, "lifetime": "kept", "lineage": PID}]
    fake.on("GET", "/v1/labs", (200, {"labs": labs}))
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(state="crashloop"))))
    fake.on("GET", f"/v1/labs/{LAB2}", (200, _det(lab=LAB2, app=_app(state="crashloop", name="blog"))))
    await R.app_down_sweep(ctx)
    await R.app_down_sweep(ctx)
    assert len(sent) == 1 and "shop" in sent[0][0] and "blog" in sent[0][0] and len(sent[0][1]) == 2


def test_the_registry_meta_column_migrates_an_old_db(tmp_path):
    import sqlite3
    from ghost_agent.memory.run_registry import RunRegistry
    c = sqlite3.connect(str(tmp_path / "runs.db"))
    c.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, name TEXT NOT NULL, place TEXT NOT NULL, lifetime TEXT NOT "
              "NULL, command TEXT NOT NULL DEFAULT '', port INTEGER, owner TEXT NOT NULL DEFAULT '', origin TEXT NOT "
              "NULL DEFAULT '', backend TEXT NOT NULL, backend_ref TEXT NOT NULL DEFAULT '', backend_token TEXT NOT "
              "NULL DEFAULT '', started_at REAL NOT NULL, state TEXT NOT NULL DEFAULT 'running', state_note TEXT NOT "
              "NULL DEFAULT '', last_probe_at REAL, ended_at REAL)")
    c.execute("INSERT INTO runs (run_id, name, place, lifetime, backend, started_at) VALUES ('r1','a','w','job','jobs',1)")
    c.commit(); c.close()
    reg = RunRegistry(tmp_path)
    assert reg.set_meta("r1", x=1) and reg.meta_of(reg.get("r1")) == {"x": 1}
    assert reg.set_meta("r1", x=None) and reg.meta_of(reg.get("r1")) == {}
    assert reg.set_meta("nope", x=1) is False


# ------------------------------------------------------------------ operator secrets route

class _Req:
    def __init__(self, body, origin=None):
        self._b = body
        self.headers = {"x-ghost-origin": origin} if origin else {}

    async def json(self):
        if isinstance(self._b, Exception):
            raise self._b
        return self._b


async def test_the_secrets_route_passes_values_through_and_answers_names(ctx, fake, monkeypatch):
    from ghost_agent.api import routes as RT
    monkeypatch.setattr(RT, "get_agent", lambda request: SimpleNamespace(context=ctx))
    fake.on("PUT", f"/v1/labs/{LAB}/secrets", (200, {"lab_id": LAB, "secrets": ["API_KEY"], "app": {"x": 1}}))
    r = await RT.operator_lab_secrets(LAB, _Req({"secret_values": {"API_KEY": "s3cr3t"}}))
    assert r.status_code == 200 and json.loads(r.body) == {"lab_id": LAB, "secrets": ["API_KEY"]}
    c = [c for c in fake.calls if c.method == "PUT"][0]
    assert c.json == {"secret_values": {"API_KEY": "s3cr3t"}} and c.origin == "owner"
    await RT.operator_lab_secrets(LAB, _Req({"secret_values": {"A": "b"}}, origin="probe"))
    assert [c for c in fake.calls if c.method == "PUT"][-1].origin == "probe"
    for bad in ({"secret_values": {}}, {"secret_values": {"A": 1}}, ValueError("x"), {"x": 1}):
        assert (await RT.operator_lab_secrets(LAB, _Req(bad))).status_code == 400
    assert (await RT.operator_lab_secrets("../x", _Req({"secret_values": {"A": "b"}}))).status_code == 400


# ------------------------------------------------------------------ code review r1

async def test_keep_of_an_already_kept_lab_records_its_app(ctx, fake, scope):
    """r1 #3: a keep whose answer was lost — the retry records the row instead of failing with already_kept."""
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "member": "web", "port": 8080}})
    assert "already KEPT" in str(out) and not [c for c in fake.calls if c.method == "POST"]
    assert [r["backend_ref"] for r in _reg(ctx).rows(OWN, live_only=True)] == [f"{PID}/shop"]


async def test_a_keep_with_no_answer_that_landed_is_recorded(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable")), (200, _det(app=_app())))
    fake.on("POST", f"/v1/labs/{LAB}/keep", L.LabsNoAnswer("timeout"))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "member": "web", "port": 8080}})
    assert out.status is OutcomeStatus.OK and "KEPT" in str(out), str(out)
    assert len(_reg(ctx).rows(OWN, live_only=True)) == 1


async def test_a_keep_whose_first_read_gets_no_answer_says_nothing_changed(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", L.LabsNoAnswer("t"))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "member": "web", "port": 8080}})
    assert "nothing was changed" in str(out) and out.world_changed is False


@pytest.mark.parametrize("secrets", [{"API_KEY": "v"}, ["api key"], "API_KEY", [1]])
async def test_keep_secrets_must_be_a_list_of_names(ctx, fake, scope, secrets):
    """r1 #13: a dict of values would already sit in the tool args."""
    scope(ctx)
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "a", "member": "web", "port": 1},
                                                             "secrets": secrets})
    assert "list of NAMES" in str(out) and fake.calls == []


async def test_the_sweep_adopts_a_kept_app_with_no_row_once(ctx, fake, monkeypatch):
    monkeypatch.setattr(R, "_ADOPT_SEEN", set())
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: True)
    _serving(fake, "up")
    await R.app_down_sweep(ctx)
    rows = _reg(ctx).rows(OWN, live_only=True)
    assert [(r["name"], r["backend"]) for r in rows] == [("shop", "labs_app")]
    _reg(ctx).set_state(rows[0]["run_id"], "lost", "x")             # ended: a re-adopt would need a new read
    await R.app_down_sweep(ctx)
    assert len(_reg(ctx).rows(OWN, live_only=True)) == 0             # read once per process


async def test_the_sweep_counts_no_serving_version_and_a_slow_restart_as_down_but_not_a_probes_app(ctx, fake,
                                                                                                  monkeypatch):
    """r1 #9 / #10."""
    monkeypatch.setattr(R, "_ADOPT_SEEN", {LAB, LAB2})
    sent = []
    monkeypatch.setattr(R, "_notify_owner", lambda c, msg, ids: sent.append(msg) or True)
    assert R._app_is_down("staged", {}) and R._app_is_down("running", {"app_state": "restarting"})
    assert not R._app_is_down("running", {"app_state": "up"})
    _app_row(ctx, origin="probe")
    _serving(fake, "crashloop")
    await R.app_down_sweep(ctx)
    await R.app_down_sweep(ctx)
    assert sent == []


async def test_an_app_op_with_the_host_unreachable_is_a_run_error(ctx, fake, scope, monkeypatch):
    """r1 #12."""
    scope(ctx)
    _app_row(ctx)
    _serving(fake, "up")
    fake.on("POST", f"/v1/labs/{LAB}/app/restart", L.LabsUnreachable("down"))
    out = await R.tool_run(ctx, action="restart", name="shop")
    assert str(out).startswith("RUN ERROR (labs_unreachable)"), str(out)


async def test_long_kept_calls_need_turn_time_and_are_bounded_in_total(ctx, fake, scope, monkeypatch):
    """r1 #11."""
    scope(ctx)
    monkeypatch.setattr(L, "_time_left_s", lambda: 50.0)
    out = await _lab(ctx, action="switch", lab_id=LAB2, params={"from_lab": LAB})
    assert "no_time" in str(out) and fake.calls == []
    monkeypatch.setattr(L, "_time_left_s", lambda: 100.0)
    assert L._long_timeout() == 85.0
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    assert L._long_timeout() == 600.0


async def test_destroying_a_kept_lab_sends_include_kept_and_drops_data_only_for_the_last_version(ctx, fake, scope):
    """r1 #7."""
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB, "lifetime": "kept", "lineage": PID},
                                               {"lab_id": LAB2, "lifetime": "kept", "lineage": PID}]}))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB, "destroyed": True}))
    out = await _lab(ctx, action="destroy", lab_id=LAB)
    d = [c for c in fake.calls if c.method == "DELETE"][-1]
    assert d.params == {"include_kept": "1"} and "data volume is KEPT" in str(out)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB, "lifetime": "kept", "lineage": PID}]}))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB, "destroyed": True,
                                               "data_volumes_removed": ["v1"], "final_backup": "b9"}))
    out = await _lab(ctx, action="destroy", lab_id=LAB)
    d = [c for c in fake.calls if c.method == "DELETE"][-1]
    assert d.params == {"include_kept": "1", "drop_data": "1"} and "final backup b9" in str(out)


async def test_unattended_destroy_of_a_kept_lab_is_refused(ctx, fake, scope):
    scope(ctx, origin="unattended")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    out = await _lab(ctx, action="destroy", lab_id=LAB)
    assert "KEPT (production)" in str(out) and not [c for c in fake.calls if c.method == "DELETE"]


async def test_the_teardown_destroys_the_serving_version_last(ctx, fake):
    """r1 #8 — through the real teardown: disposable first, then the staged version, the mounted one LAST — and
    it alone gets drop_data (the other version was gone by then)."""
    from ghost_agent.tools import projects as P
    own = f"project:{PID}"
    labs = [{"lab_id": "lab_000000aa", "owner": own, "lifetime": "kept", "lineage": PID, "app_mounted": True},
            {"lab_id": "lab_000000bb", "owner": own, "lifetime": "kept", "lineage": PID, "app_mounted": False},
            {"lab_id": "lab_000000cc", "owner": own, "lifetime": "disposable"}]
    fake.on("GET", "/v1/labs", (200, {"labs": labs}))
    fake.on("GET", "/v1/health", (200, {"ok": True, "host_id": "h"}))
    for x in labs:
        fake.on("DELETE", f"/v1/labs/{x['lab_id']}", (200, {"destroyed": True, "data_volumes_removed": ["v"]}))
    rep = {"stopped": [], "kept_up": [], "failed": [], "live_left": [], "notes": []}
    await P._teardown_labs(ctx, PID, own, hard=True, origin="owner", rep=rep)
    dels = [c for c in fake.calls if c.method == "DELETE"]
    assert [c.path.rsplit("/", 1)[1] for c in dels] == ["lab_000000cc", "lab_000000bb", "lab_000000aa"]
    assert [bool((c.params or {}).get("drop_data")) for c in dels] == [False, False, True]
    assert rep["live_left"] == []


def test_the_tar_opens_the_folder_from_the_workspace_without_following_a_swapped_parent(tmp_path):
    """r1 #4: a parent swapped for a symlink after resolve() cannot redirect the walk."""
    ws = tmp_path / "ws"
    (ws / "p" / "site").mkdir(parents=True)
    (ws / "p" / "site" / "a.txt").write_text("1")
    outside = tmp_path / "host" / "site"
    outside.mkdir(parents=True)
    (outside / "secret.txt").write_text("HOST")
    root = (ws / "p" / "site").resolve()
    data, _, err = L.tar_folder(root, ws.resolve())
    assert err == "" and _names(base64.b64encode(data)) == ["a.txt"]
    import shutil
    shutil.rmtree(ws / "p")
    os.symlink(tmp_path / "host", ws / "p")                     # the swap
    with pytest.raises(OSError):
        L.tar_folder(root, ws.resolve())


def test_the_tar_skips_a_fifo_and_never_blocks(tmp_path):
    """r1 #5."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "ok.txt").write_text("1")
    os.mkfifo(root / "pipe")
    data, skipped, err = L.tar_folder(root, tmp_path.resolve())
    assert err == "" and skipped == ["pipe"] and _names(base64.b64encode(data)) == ["ok.txt"]


def test_execute_after_a_deploy_waits_for_it():
    """r1 #13."""
    from ghost_agent.core.agent import _batch_call_depends
    assert _batch_call_depends("execute", {}, "lab", {"action": "deploy"})
