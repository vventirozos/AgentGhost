"""§4UL pins: the per-deploy build step, in-place deploy with automatic revert, re-keep config update,
template app defaults (static-site, node-app, python-pg, keepable pg-single) and the new images."""
from pathlib import Path

import pytest

from ghost_labs import cli
from ghost_labs import templates as T
from ghost_labs.errors import LabError
from ghost_labs.kept import AUTO_BUILD, DEPS_PREFIX, Kept, _build_of, build_command
from ghost_labs.store import loads

from .test_core import ROOT, env  # noqa: F401
from .test_kept_fixes import _deploy_stubs, app_of, make_kept


@pytest.fixture
def kept(env, tmp_path):  # noqa: F811
    k = Kept(env)
    k.archive_root = tmp_path / "archives"
    k.backup_root = tmp_path / "backups"
    env.kept = k
    return k


# ---------------------------------------------------------------- build setting
def test_build_setting_parse():
    assert _build_of(None) is None and _build_of("auto") is None and _build_of(" AUTO ") is None
    for off in ("", "none", "off", "false", False):
        assert _build_of(off) == ""
    assert _build_of("make dist") == "make dist"
    with pytest.raises(LabError):
        _build_of("a\nb")
    assert build_command(None) == AUTO_BUILD and build_command("") == "" and build_command("x") == "x"


def test_auto_build_installs_into_the_folder_and_never_needs_a_proxy():
    assert "--target .deps" in AUTO_BUILD and "-r requirements.txt" in AUTO_BUILD
    assert "npm ci" in AUTO_BUILD and "npm install" in AUTO_BUILD and "npm run build --if-present" in AUTO_BUILD
    assert "proxy" not in AUTO_BUILD
    # a package.json in a lab without npm is skipped, not a failed deploy
    assert "command -v npm" in AUTO_BUILD
    assert ".deps" in DEPS_PREFIX and "node_modules/.bin" in DEPS_PREFIX


# ---------------------------------------------------------------- build in the NEW folder, before the swap
def test_build_runs_in_the_new_folder_before_the_swap(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    must = _deploy_stubs(kept, env, monkeypatch)
    order = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: order.append(cmd) or
                        {"exit_code": 0 if "ghost-seeded" not in cmd else 1, "stdout": "built ok", "stderr": ""})
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: order.append(cmd) or must.append(cmd)
                        or {"exit_code": 0})
    res = kept._deploy_into("lab_a", app, b"t", {}, [])
    build = [i for i, c in enumerate(order) if "requirements.txt" in c]
    swap = [i for i, c in enumerate(order) if "mv /app.new /app" in c]
    assert build and swap and build[0] < swap[0]
    assert order[build[0]].startswith("cd /app.new && ")
    assert res["build_log"].endswith("built ok")


def test_failed_build_leaves_the_running_version(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up", build="npm run build")
    _deploy_stubs(kept, env, monkeypatch)
    cmds = []

    def sh(lab, m, cmd, *a, **k):
        cmds.append(cmd)
        if "npm run build" in cmd:
            return {"exit_code": 2, "stdout": "", "stderr": "ERR! missing script: build"}
        return {"exit_code": 1 if "ghost-seeded" in cmd else 0, "stdout": "", "stderr": ""}
    monkeypatch.setattr(kept, "_sh", sh)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: pytest.fail("the running app must not stop"))
    with pytest.raises(LabError) as e:
        kept._deploy_into("lab_a", app, b"t", {}, [])
    assert e.value.code == "build_failed" and "missing script" in e.value.message and "untouched" in e.value.message
    assert any("rm -rf /app.new" in c for c in cmds)


def test_requirements_build_overrides_the_app(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up", build="make")
    _deploy_stubs(kept, env, monkeypatch)
    cmds = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or
                        {"exit_code": 1 if "ghost-seeded" in cmd else 0, "stdout": "", "stderr": ""})
    kept._deploy_into("lab_a", app, b"t", {"build": "none"}, [])
    assert not any(c.startswith("cd /app.new && ") for c in cmds)


def test_start_sees_the_folder_deps(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="stopped")
    seen = {}

    class C:
        def exec_run(self, cmd, environment=None, **k):
            seen.update(environment or {})
            from types import SimpleNamespace
            return SimpleNamespace(exit_code=0, output=b"")
    monkeypatch.setattr(env, "container", lambda lab, m: C())
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    kept._start_app("lab_a", app, count=False)
    assert seen["GHOST_APP_CMD"] == DEPS_PREFIX + "python3 app.py"
    assert seen["PORT"] == "8080" and seen["DATA_DIR"] == "/data"


# ---------------------------------------------------------------- in-place deploy reverts on failure
def _deploy_world(kept, env, monkeypatch, results):  # noqa: F811
    make_kept(env, "lab_a", state="up", last_backup_at=0)
    prev_ts = kept._keep_archive("p", "chess", b"OLD", {"pip": ["x"]}, [])
    a = app_of(env, "lab_a")
    a["archive_ts"] = prev_ts
    kept._save("lab_a", a)
    calls = []

    def dep(lab, app, data, req, seed, start=True, built=None):
        calls.append((data, req))
        return results.pop(0)
    monkeypatch.setattr(kept, "_deploy_into", dep)
    monkeypatch.setattr(kept, "_save_built", lambda *a: False)
    # no <workdir>.prev in the member: the revert deploys the previous archive (a version from before §4UL)
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {"exit_code": 1 if ".prev" in cmd else 0,
                                                                   "stdout": "", "stderr": ""})
    return prev_ts, calls


def test_crashed_new_version_is_rolled_back(kept, env, monkeypatch):  # noqa: F811
    prev_ts, calls = _deploy_world(kept, env, monkeypatch, [
        {"ok": False, "alive": False, "why": "the app process is not running"}, {"ok": True, "alive": True}])
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})          # "NEW"
    assert out["rolled_back"].startswith("the app process is not running")
    assert out["probe"]["ok"] is True and out["failed_probe"]["ok"] is False
    assert calls == [(b"NEW", {}), (b"OLD", {"pip": ["x"]})]
    assert app_of(env, "lab_a")["archive_ts"] == prev_ts            # the old version stays the live archive
    assert len(env.store.q("SELECT * FROM archives WHERE app='chess'")) == 2   # the new one is kept for a look


def test_health_error_status_is_rolled_back(kept, env, monkeypatch):  # noqa: F811
    _deploy_world(kept, env, monkeypatch, [
        {"ok": False, "alive": True, "http_status": 500, "why": "the health path answered HTTP 500"},
        {"ok": True, "alive": True}])
    assert "HTTP 500" in kept.deploy("lab_a", {"archive_b64": "TkVX"})["rolled_back"]


def test_slow_start_is_still_starting_not_rolled_back(kept, env, monkeypatch):  # noqa: F811
    prev_ts, calls = _deploy_world(kept, env, monkeypatch, [
        {"ok": False, "alive": True, "error": "ConnectionRefusedError"}])
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert out.get("still_starting") is True and "rolled_back" not in out and len(calls) == 1
    assert app_of(env, "lab_a")["archive_ts"] != prev_ts


def test_no_revert_when_the_old_version_was_not_up(kept, env, monkeypatch):  # noqa: F811
    prev_ts, calls = _deploy_world(kept, env, monkeypatch, [{"ok": False, "alive": False, "why": "x"}])
    a = app_of(env, "lab_a")
    a["state"] = "crashloop"
    kept._save("lab_a", a)
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert "rolled_back" not in out and len(calls) == 1


def test_revert_that_fails_is_reported(kept, env, monkeypatch):  # noqa: F811
    _deploy_world(kept, env, monkeypatch, [{"ok": False, "alive": False, "why": "crashed"},
                                           {"ok": False, "alive": False, "why": "also crashed"}])
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert out["rolled_back"] == "crashed" and "also crashed" in out["rollback_error"]


# ---------------------------------------------------------------- re-keep: update the run config
def _upd(kept, env, monkeypatch, **app_extra):  # noqa: F811
    make_kept(env, "lab_a", state="up", **app_extra)
    seen = {"start": 0, "lease": 0, "stop": 0}
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: seen.__setitem__("stop", seen["stop"] + 1))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: seen.__setitem__("start", seen["start"] + 1))
    monkeypatch.setattr(kept, "_lease", lambda lab, app, move: seen.__setitem__("lease", seen["lease"] + 1))
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(kept, "_has_code", lambda *a: True)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: seen.setdefault("must", []).append(a[2]) or {"exit_code": 0})
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: seen.setdefault("sh", []).append(a[2]) or
                        {"exit_code": 0, "stdout": "", "stderr": ""})
    return seen


def test_update_cmd_restarts(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    out = kept.update_app("lab_a", {"cmd": "gunicorn -b 0.0.0.0:8080 app:app", "port": 8080})
    assert out["changed"] == ["cmd"] and seen["start"] == 1 and seen["lease"] == 0
    assert app_of(env, "lab_a")["cmd"].startswith("gunicorn")


def test_update_port_moves_the_lease(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    out = kept.update_app("lab_a", {"port": 3000, "health": {"http": "/healthz"}})
    assert out["changed"] == ["health", "port"] and seen["lease"] == 1 and seen["start"] == 1
    a = app_of(env, "lab_a")
    assert a["port"] == 3000 and a["health"] == {"http": "/healthz"}


def test_update_workdir_moves_the_folder(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    kept.update_app("lab_a", {"workdir": "/srv/site"})
    assert any("mv /app /srv/site" in c for c in seen["must"])
    assert app_of(env, "lab_a")["workdir"] == "/srv/site"


def test_update_same_config_is_a_no_op(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    out = kept.update_app("lab_a", {"cmd": "python3 app.py", "port": 8080})
    assert out["changed"] == [] and seen["start"] == 0 and seen["stop"] == 0


def test_update_refuses_what_it_cannot_change(kept, env, monkeypatch):  # noqa: F811
    _upd(kept, env, monkeypatch)
    for body, field in (({"name": "x"}, "app"), ({"port": 0}, "app.port"), ({"workdir": "/data"}, "app.workdir"),
                        ({"health": {"http": "nope"}}, "app.health"), ({"cmd": "a\nb"}, "app.cmd")):
        with pytest.raises(LabError) as e:
            kept.update_app("lab_a", body)
        assert e.value.extra["field"] == field


def test_update_does_not_start_an_app_the_owner_stopped(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch, stopped_by_owner=True)
    kept.update_app("lab_a", {"cmd": "node server.js"})
    assert seen["start"] == 0 and app_of(env, "lab_a")["cmd"] == "node server.js"


def test_update_on_a_disposable_lab_is_refused(kept, env):  # noqa: F811
    from .test_core import add_lab
    add_lab(env, "lab_d")
    with pytest.raises(LabError) as e:
        kept.update_app("lab_d", {"cmd": "x"})
    assert e.value.code == "not_kept"


def test_failed_port_lease_keeps_the_old_port(kept, env, monkeypatch):  # noqa: F811
    _upd(kept, env, monkeypatch)

    def busy(*a, **k):
        raise LabError("busy", "port held")
    monkeypatch.setattr(kept, "_lease", busy)
    with pytest.raises(LabError):
        kept.update_app("lab_a", {"port": 3000})
    assert app_of(env, "lab_a")["port"] == 8080


# ---------------------------------------------------------------- templates and images
def test_shipped_templates_carry_app_defaults():
    t = T.load_dir(ROOT / "templates")
    assert {"static-site", "node-app", "python-pg"} <= set(t)
    ss = t["static-site"]
    assert ss["members"][0]["image"] == "ghost-lab/static:1" and ss["members"][0]["mem_mb"] <= 128
    assert ss["app"]["port"] == 8080 and ss["app"]["build"] == "none" and "nginx" in ss["app"]["cmd"]
    assert t["node-app"]["members"][0]["image"] == "ghost-lab/node-app:1" and t["node-app"]["app"]["cmd"] == "npm start"
    pp = t["python-pg"]
    assert {m["name"] for m in pp["members"]} == {"app", "db"}
    assert pp["members"][0]["env"]["DATABASE_URL"] == "postgresql://postgres:lab@db:5432/postgres"
    pg = t["pg-single"]["app"]
    assert pg["service"] is True and pg["port"] == 5432 and pg["health"] == {"http": None, "tcp": True}
    assert "pgrep -x initdb" in pg["cmd"]


def test_app_defaults_validated():
    base = {"name": "x", "members": [{"name": "m", "image": "ghost-lab/base:1", "mem_mb": 128}]}
    for bad in ({"member": "nope"}, {"port": 0}, {"cmd": "a\nb"}, {"health": "/"}, {"bogus": 1}):
        with pytest.raises(LabError):
            T.parse({**base, "app": bad})
    assert T.parse({**base, "app": {"member": "m", "port": 80}})["app"] == {"member": "m", "port": 80}
    assert "app" not in T.parse(base)


def test_new_images_are_built_after_base():
    names = [n for n, _ in cli.IMAGES]
    assert names.index("base") < names.index("static") and names.index("base") < names.index("node-app")
    for d in ("static", "node-app"):
        df = (ROOT / "images" / d / "Dockerfile").read_text()
        assert df.startswith("FROM ghost-lab/base:1")
    conf = (ROOT / "images" / "static" / "default.conf").read_text()
    assert "autoindex off" in conf and "listen 8080" in conf and "root /app" in conf and r"location ~ /\." in conf


def test_keep_fills_template_app_defaults(kept, env, monkeypatch):  # noqa: F811
    from .test_core import add_lab
    add_lab(env, "lab_s", owner="project:p")
    env.store.update_lab("lab_s", template="static-site")
    env.store.x("UPDATE members SET name='site' WHERE lab_id='lab_s'")
    monkeypatch.setattr(kept, "_recreate", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_lease", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_install_cron", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_has_code", lambda *a: False)
    kept.keep("lab_s", {"app": {"name": "site1"}, "project": "p"})
    a = loads(env.store.lab("lab_s")["app_json"])
    assert a["member"] == "site" and a["port"] == 8080 and "nginx" in a["cmd"] and a["build"] == ""
    assert a["health"] == {"http": "/"}


def test_service_app_starts_with_nothing_deployed(kept, env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: pytest.fail("no folder check for a service"))
    assert kept._has_code("lab_x", {"service": True, "member": "m1", "workdir": "/app"}) is True


def test_build_failed_is_a_known_error_code():
    assert LabError("build_failed", "x").status == 422


def test_app_config_route_is_wired():
    src = (Path(__file__).resolve().parents[2] / "ghost_labs" / "api.py").read_text()
    assert '@app.put("/v1/labs/{lab_id}/app/config")' in src and "svc.kept.update_app(lab_id, body)" in src


def test_subdir_and_build_apply_from_the_next_deploy(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    out = kept.update_app("lab_a", {"subdir": "dist/", "build": "npm run build"})
    assert out["changed"] == ["build", "subdir"] and seen["start"] == 0 and seen["stop"] == 0
    assert "next deploy" in out["probe"]["note"]
    a = app_of(env, "lab_a")
    assert a["subdir"] == "dist" and a["build"] == "npm run build" and a["state"] == "up"


def test_subdir_must_stay_inside_the_folder(kept, env, monkeypatch):  # noqa: F811
    _upd(kept, env, monkeypatch)
    for bad in ("../x", "a/../b", "./x", "a//b"):
        with pytest.raises(LabError) as e:
            kept.update_app("lab_a", {"subdir": bad})
        assert e.value.extra["field"] == "app.subdir"


# ---------------------------------------------------------------- owner composition on create
def test_overrides_change_a_template(env):  # noqa: F811
    spec = env.resolve_spec({"template": "python-pg", "overrides": {
        "mem_mb": {"app": 1024}, "ports": {"app": [8080, 5000]}, "egress": "none", "cap_add": ["NET_ADMIN"]}})
    by = {m["name"]: m for m in spec["members"]}
    assert by["app"]["mem_mb"] == 1024 and by["db"]["mem_mb"] == 512 and by["app"]["ports"] == [8080, 5000]
    assert spec["egress"] == "none" and spec["template"] == "python-pg"
    assert by["db"]["cap_add"] == ["NET_ADMIN"] and by["app"]["env"]["DATABASE_URL"].endswith("@db:5432/postgres")


@pytest.mark.parametrize("ov", [{"mem_mb": 10}, {"cap_add": ["SYS_ADMIN"]}, {"egress": "clearnet"},
                                {"mem_mb": {"nope": 512}}, {"image": "x"}, {"ports": [70000]}, "big"])
def test_bad_overrides_refused(env, ov):  # noqa: F811
    with pytest.raises(LabError) as e:
        env.resolve_spec({"template": "python-app", "overrides": ov})
    assert e.value.code == "bad_request"


def test_overrides_need_a_template(env):  # noqa: F811
    with pytest.raises(LabError):
        env.resolve_spec({"overrides": {"mem_mb": 256}})


@pytest.mark.parametrize("origin", ["probe", "autopilot", "member"])
def test_composition_is_the_owners(env, origin):  # noqa: F811
    from ghost_labs.labs import ORIGINS
    if origin not in ORIGINS:
        pytest.skip(f"{origin} is not an origin here")
    for body in ({"template": "python-app", "overrides": {"mem_mb": 1024}},
                 {"members": [{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 256}]}):
        with pytest.raises(LabError) as e:
            env.create({**body, "owner": "conversation:x"}, origin)
        assert e.value.code == "owner_only"
