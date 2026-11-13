"""§4UL labs review fixes: the automatic revert after a backed-up app, no rebuild on a recreate (saved built folder,
<workdir>.prev), postgres stopped for backups/restores and every member backed up, readable deploys, TCP apps,
the twin's env, update_app ordering, template budgets, wait_for/watch."""
import io
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_labs import templates as T
from ghost_labs.errors import LabError
from ghost_labs.kept import ISOLATED_PY_RE, Kept, is_tcp_app, wait_for_cmd
from ghost_labs.labs import _kept_tcp_lease
from ghost_labs.store import dumps

from .fakes import TAR  # noqa: E402
from .test_core import ROOT, env  # noqa: F401
from .test_kept_fixes import _deploy_stubs, app_of, make_kept


@pytest.fixture
def kept(env, tmp_path):  # noqa: F811
    k = Kept(env)
    k.archive_root = tmp_path / "archives"
    k.backup_root = tmp_path / "backups"
    env.kept = k
    return k


def _ok(stdout=""):
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


# ---------------------------------------------------------------- MAJOR 1: the revert fires after a backup
def test_revert_fires_for_an_app_with_a_prior_backup(kept, env, monkeypatch):  # noqa: F811
    """The reviewer's repro: a kept app whose daily backup ran (last_backup_at set, no hook) — the REAL
    _backup_locked stops and starts the app (state "starting"); the crashed new version must still be reverted."""
    make_kept(env, "lab_a", state="up", last_backup_at=time.time() - 3600)
    prev_ts = kept._keep_archive("p", "chess", b"OLD", {}, [])
    a = app_of(env, "lab_a")
    a["archive_ts"] = prev_ts
    kept._save("lab_a", a)
    kept.backup_root.mkdir(parents=True, exist_ok=True)
    (kept.backup_root / "x.tar").write_bytes(b"x")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": "bk_x", "member": mem, "path": str(kept.backup_root / "x.tar"), "bytes": 1})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda *a, **k: SimpleNamespace(exit_code=0, output=b"")))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "_save_built", lambda *a: False)
    calls = []

    def dep(lab, app, data, req, seed, start=True, built=None):
        calls.append(data)
        return {"ok": False, "alive": False, "why": "crashed"}
    monkeypatch.setattr(kept, "_deploy_into", dep)
    reverted = []
    monkeypatch.setattr(kept, "_revert_in_place", lambda lab, app, prev, failed_req=None: reverted.append(prev[0]) or
                        {"ok": True, "alive": True})
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert out["rolled_back"] == "crashed" and reverted == [b"OLD"] and calls == [b"NEW"]
    assert app_of(env, "lab_a")["archive_ts"] == prev_ts


# ---------------------------------------------------------------- MAJOR 4: no rebuild on a revert or a recreate
def test_revert_moves_the_previous_folder_back_without_a_build(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    cmds = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_install_cron", lambda *a: None)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: pytest.fail("no deploy (no build) when .prev exists"))
    res = kept._revert_in_place("lab_a", app, (b"OLD", {}, [], None))
    assert res["ok"] and any("mv /app.prev /app" in c for c in cmds)


def test_revert_without_prev_uses_the_saved_build(kept, env, monkeypatch, tmp_path):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": 1, "stdout": "", "stderr": ""})
    seen = {}
    monkeypatch.setattr(kept, "_deploy_into", lambda lab, app, data, req, seed, start=True, built=None:
                        seen.update(built=built) or {"ok": True})
    kept._revert_in_place("lab_a", app, (b"OLD", {}, [], "/x.built.tar"))
    assert seen["built"] == "/x.built.tar"


def test_swap_keeps_the_previous_folder_and_unpack_makes_files_readable(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up", build="")
    cmds = _deploy_stubs(kept, env, monkeypatch)
    kept._deploy_into("lab_a", app, b"t", {}, [])
    unpack = [c for c in cmds if "tar -xzf" in c][0]
    swap = [c for c in cmds if "mv /app.new /app" in c][0]
    assert "chmod -R a+rX /app.new" in unpack
    assert "mv /app /app.prev" in swap and swap.index("rm -rf /app.prev") < swap.index("mv /app /app.prev")


def test_deploy_from_the_saved_build_skips_the_build(kept, env, monkeypatch, tmp_path):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")      # build None = the automatic build
    cmds = _deploy_stubs(kept, env, monkeypatch, marker="saves\n")
    app["data_seed"] = ["saves"]
    put = []
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        put_archive=lambda path, f: put.append((path, f.read())) or True))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or
                        (_ok("saves\n") if "ghost-seeded" in cmd else _ok()))
    bt = tmp_path / "a.tgz.built.tar"
    bt.write_bytes(b"TAR")
    res = kept._deploy_into("lab_a", app, b"t", {}, [], built=str(bt))
    assert put == [("/app.x", b"TAR")] and res.get("from_saved_build") is True
    assert not [c for c in cmds if "requirements.txt" in c or "tar -xzf" in c]          # no build, no source unpack
    # re-review MAJOR 1: whatever the single top folder is called (a renamed workdir) — real-shell tests in
    # test_4ul_rereview_kept.py
    unpack = [c for c in cmds if 'mv /app.x/"$top" /app.new' in c][0]
    assert "if [ -L /app.new/saves ]; then rm -f /app.new/saves; fi" in unpack and "chmod -R a+rX /app.new" in unpack


def test_built_folder_is_saved_with_the_archive_and_pruned_with_it(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as t:
        t.addfile(tarfile.TarInfo("app"))
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        get_archive=lambda p: (iter([data.getvalue()]), {})))
    a = app_of(env, "lab_a")
    ts = kept._keep_archive("p", "chess", b"v1", {}, [], built_from=("lab_a", a))
    a["archive_ts"] = ts
    kept._save("lab_a", a)
    arch = kept._archive_for("lab_a")
    assert arch[0] == b"v1" and arch[3] and Path(arch[3]).read_bytes() == data.getvalue()
    for i in range(6):
        kept._keep_archive("p", "chess", b"v%d" % i, {}, [], built_from=("lab_a", a))
    built = list((kept.archive_root / "p" / "chess").glob("*.built.tar"))
    assert len(built) == 6 and Path(arch[3]).exists()          # 5 newest + the pinned live one


def test_too_big_built_folder_saves_nothing(kept, env, monkeypatch, tmp_path):  # noqa: F811
    import ghost_labs.kept as K
    make_kept(env, "lab_a", state="up")
    monkeypatch.setattr(K, "BUILT_MAX", 4)
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(get_archive=lambda p: (iter([b"12345"]), {})))
    dst = tmp_path / "x.built.tar"
    assert kept._save_built("lab_a", app_of(env, "lab_a"), dst) is False and not dst.exists()
    assert not list(tmp_path.glob("*.part"))


def test_switch_mount_unpacks_the_saved_build(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="staged", mounted=False)
    monkeypatch.setattr(kept, "_recreate", lambda *a, **k: None)
    seen = {}
    monkeypatch.setattr(kept, "_deploy_into", lambda lab, app, data, req, seed, start=True, built=None:
                        seen.update(built=built) or {"ok": True})
    kept._mount("lab_a", (b"x", {}, [], "/b.built.tar"), [])
    assert seen["built"] == "/b.built.tar"
    kept._mount("lab_a", (b"x", {}), [])            # an old-style tuple: built again
    assert seen["built"] is None


# ---------------------------------------------------------------- MAJOR 2: postgres-safe backups and restores
def _pg_lab(env, **extra):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="app", **extra)
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_a'")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})


def test_backup_stops_postgres_for_the_copy_and_starts_it_after(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_lab(env, watch=[{"member": "db", "cmd": "keeper"}])
    order = []
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: order.append("stop app"))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: order.append("start app"))
    monkeypatch.setattr(kept, "_stop_watch", lambda lab, app, only=None: order.append(f"stop watch {only}"))
    monkeypatch.setattr(kept, "_start_watch", lambda lab, app, only=None: order.append(f"start watch {only}"))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: order.append(
        f"{m}: {'pg stop' if 'pg_ctl' in cmd else 'pg start' if '10-postgres.sh' in cmd else cmd}") or _ok("was-running"))
    (tmp_path / "b.tar").write_bytes(b"x")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: order.append(f"copy {mem}") or {
        "backup_id": f"bk_{mem}", "member": mem, "path": str(tmp_path / "b.tar"), "bytes": 1})
    out = kept._backup_locked("lab_a", "daily")
    assert [m["member"] for m in out["members"]] == ["app", "db"]
    assert order == ["stop app", "stop watch ['db']", "db: pg stop", "copy app", "copy db", "db: pg start",
                     "start watch ['db']", "start app"]


def test_backup_copies_nothing_when_postgres_does_not_stop(kept, env, monkeypatch):  # noqa: F811
    _pg_lab(env)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": 1, "stdout": "", "stderr": "timeout"})
    monkeypatch.setattr(kept, "_copy_data", lambda *a: pytest.fail("no copy of a running cluster"))
    with pytest.raises(LabError) as e:
        kept._backup_locked("lab_a", "daily")
    assert "did not stop" in e.value.message and not env.store.q("SELECT * FROM backups")


def test_backup_hook_runs_and_postgres_still_stops_for_the_copy(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Re-review MAJOR 2 (was: "a hook owns its consistency" — the files copied after the hook were a live cluster)."""
    _pg_lab(env, backup={"cmd": "pg_dumpall > /data/d.sql", "restore_cmd": None})
    sh = []
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, *a, **k: sh.append(f"hook {cmd}") or _ok())
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: sh.append(cmd) or _ok("was-running"))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: sh.append("stop app"))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: sh.append("start app"))
    (tmp_path / "b.tar").write_bytes(b"x")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: sh.append(f"copy {mem}") or {
        "backup_id": f"bk_{mem}", "member": mem, "path": str(tmp_path / "b.tar"), "bytes": 1})
    kept._backup_locked("lab_a", "manual")
    stop = [i for i, c in enumerate(sh) if "pg_ctl" in c]
    assert sh[0].startswith("hook pg_dumpall") and stop and stop[0] < sh.index("copy db")
    assert sh.index("stop app") < stop[0] and sh[-1] == "start app"


def test_restore_of_a_db_member_stops_restores_starts_and_checks(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_lab(env)
    bk = tmp_path / "db.tar"
    bk.write_bytes(TAR)
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_db", "p", "chess", "db", 1.0, str(bk), "daily"))
    order = []
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: order.append("stop app"))
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, *a, **k: order.append(
        f"{m}: {'mark' if cmd.startswith('touch') else 'unmark' if cmd.startswith('rm -f') else 'check script' if 'sha256sum' in cmd else 'quiesce' if 'pkill' in cmd else 'clear'}") or _ok())
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        put_archive=lambda p, f: order.append(f"{m}: put") or True))
    monkeypatch.setattr(env, "_put_bytes", lambda c, path, data, mode: order.append(f"install {path}"))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: order.append(
        f"{m}: {'pg stop' if 'pg_ctl' in cmd else 'pg start'}") or _ok())
    kept._restore_locked("lab_a", "bk_db", start=False)
    # §4UL round 5: the restore marker goes in before the clear and out once postgres answers
    # round 6: the marker-aware start script is installed and checked first, and any running copy of it is killed
    # around the stop
    assert order == ["stop app", "install /etc/lab/start.d/10-postgres.sh", "db: check script", "db: quiesce",
                     "db: pg stop", "db: quiesce", "db: mark", "db: clear", "db: put", "db: pg start", "db: unmark"]


def test_restored_database_that_does_not_start_is_held(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_lab(env)
    bk = tmp_path / "db.tar"
    bk.write_bytes(TAR)
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_db", "p", "chess", "db", 1.0, str(bk), "daily"))
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: _ok())
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {"exit_code": 1 if "10-postgres.sh" in cmd else 0,
                                                                  "stdout": "FATAL: bad", "stderr": ""})
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert "does not start" in e.value.message and a["stopped_by_owner"] and a["state"] == "stopped"


def test_restore_of_a_set_restores_every_member_app_last(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_lab(env)
    for bid, mem in (("bk_app", "app"), ("bk_db", "db")):
        (tmp_path / f"{bid}.tar").write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", mem, 5.0,
                                                                   str(tmp_path / f"{bid}.tar"), "daily"))
    done = []
    monkeypatch.setattr(kept, "_restore_locked", lambda lab, sid, snapshot=True, start=True, pg_restored=False, protect=None, fh=None:
                        done.append((sid, start)) or {"pre_restore": f"pre_{sid}", "probe": {"ok": True}})
    out = kept.restore("lab_a", "bk_app")
    assert done == [("bk_db", False), ("bk_app", True)] and out["restored_set"] == ["bk_db", "bk_app"]


# ---------------------------------------------------------------- MAJOR 6: a database is published as TCP
def test_tcp_app_is_published_as_raw_tcp_with_an_address(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", port=5432, health={"http": None, "tcp": True})
    env.cfg.ts_services = True
    a = app_of(env, "lab_a")
    kept._lease("lab_a", a, move=True)
    calls = [c for c in env.ports.run.calls if c[:2] == ["tailscale", "serve"]]
    assert any(f"--tcp={a['host_port']}" in c and f"tcp://127.0.0.1:{a['host_port']}" in c for c in calls)
    assert not any("--service=svc:chess" in " ".join(c) for c in calls)
    assert a["url"] is None and a["address"] == f"disorder.tail1234.ts.net:{a['host_port']}"
    assert kept._view(a)["address"] == a["address"]


def test_http_app_keeps_its_https_url(kept, env):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    a = app_of(env, "lab_a")
    kept._lease("lab_a", a, move=True)
    assert a["url"].startswith("https://") and "address" not in a and not is_tcp_app(a)


def test_ports_unpublish_and_reconcile_know_tcp(env):  # noqa: F811
    env.ports.unpublish(20005)
    calls = env.ports.run.calls
    assert ["tailscale", "serve", "--https=20005", "off"] in calls and ["tailscale", "serve", "--tcp=20005", "off"] in calls
    lo = env.cfg.port_range()[0]
    env.ports.run = lambda cmd, **k: (calls.append(cmd), SimpleNamespace(
        stdout=dumps({"TCP": {str(lo): {"TCPForward": "127.0.0.1:1"}, str(lo + 1): {"HTTPS": True}}})))[1]
    env.ports.reconcile_serve([])
    assert ["tailscale", "serve", f"--tcp={lo}", "off"] in calls and ["tailscale", "serve", f"--https={lo + 1}", "off"] in calls


def test_restart_reconcile_republishes_a_tcp_lease_as_tcp():
    row = {"app_json": dumps({"member": "pg1", "port": 5432, "health": {"http": None, "tcp": True}})}
    assert _kept_tcp_lease(row, {"lease_key": "p/db", "member": "pg1", "cport": 5432}) is True
    assert _kept_tcp_lease(row, {"lease_key": "", "member": "pg1", "cport": 5432}) is False
    web = {"app_json": dumps({"member": "app", "port": 8080, "health": {"http": "/"}})}
    assert _kept_tcp_lease(web, {"lease_key": "p/w", "member": "app", "cport": 8080}) is False


# ---------------------------------------------------------------- MINORs
def test_twin_carries_member_env_ready_and_ports(kept, env, monkeypatch):  # noqa: F811
    spec = {"members": [{"name": "app", "image": "ghost-lab/python-app:1", "mem_mb": 256, "cpus": 1.0, "cap_add": [],
                         "env": {"DATABASE_URL": "postgresql://postgres:lab@db:5432/postgres"}, "ports": [8080]},
                        {"name": "db", "image": "ghost-lab/pg:18", "mem_mb": 512, "cpus": 1.0, "cap_add": [],
                         "ready": "pg_isready -q"}], "egress": "tor", "readiness": None}
    seen = {}
    monkeypatch.setattr(env, "create", lambda body, origin, pool=None: seen.update(body=body) or
                        (_ for _ in ()).throw(LabError("busy", "stop here")))
    with pytest.raises(LabError):
        kept._twin({"lab_id": "lab_a", "owner": "project:p"}, spec, {"secrets": []}, b"", {})
    m = {x["name"]: x for x in seen["body"]["members"]}
    assert m["app"]["env"]["DATABASE_URL"].endswith("@db:5432/postgres") and m["app"]["ports"] == [8080]
    assert m["db"]["ready"] == "pg_isready -q"


def _upd(kept, env, monkeypatch, sh_exit=0):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    seen = {"stop": 0, "must": [], "lease": []}
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: seen.__setitem__("stop", seen["stop"] + 1))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(kept, "_has_code", lambda *a: True)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: seen["must"].append(a[2]) or _ok())
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": sh_exit, "stdout": "", "stderr": ""})
    return seen


@pytest.mark.parametrize("wd", ["/app/sub", "/srv/x"])
def test_workdir_move_refused_leaves_the_app_untouched(kept, env, monkeypatch, wd):  # noqa: F811
    seen = _upd(kept, env, monkeypatch, sh_exit=1 if wd == "/srv/x" else 0)    # /srv/x: already exists
    with pytest.raises(LabError) as e:
        kept.update_app("lab_a", {"workdir": wd})
    assert e.value.extra["field"] == "app.workdir" and seen["stop"] == 0 and not seen["must"]
    assert app_of(env, "lab_a")["workdir"] == "/app"


def test_workdir_moves_before_the_app_stops(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    order = []
    monkeypatch.setattr(kept, "_must", lambda *a, **k: order.append("move") or _ok())
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: order.append("stop"))
    kept.update_app("lab_a", {"workdir": "/srv/site"})
    assert order[:2] == ["move", "stop"] and seen is not None


def test_failed_port_move_puts_the_old_lease_back(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)

    def lease(lab, app, move):
        seen["lease"].append(app["port"])
        if app["port"] == 3000:
            raise LabError("busy", "serve timed out")
    monkeypatch.setattr(kept, "_lease", lease)
    with pytest.raises(LabError):
        kept.update_app("lab_a", {"port": 3000, "cmd": "node s.js"})
    a = app_of(env, "lab_a")
    assert seen["lease"] == [3000, 8080] and seen["stop"] == 0
    assert a["port"] == 8080 and a["cmd"] == "python3 app.py"


def test_health_change_to_tcp_moves_the_publish(kept, env, monkeypatch):  # noqa: F811
    seen = _upd(kept, env, monkeypatch)
    monkeypatch.setattr(kept, "_lease", lambda lab, app, move: seen["lease"].append(is_tcp_app(app)))
    kept.update_app("lab_a", {"health": {"http": None, "tcp": True}})
    assert seen["lease"] == [True]


def test_build_sees_port_and_the_app_secrets(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up", secrets=["API_KEY"])
    env.store.set_secret("p", "chess", "API_KEY", "s3cret")
    _deploy_stubs(kept, env, monkeypatch)
    envs = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, env=None, **k: envs.append((cmd, env)) or
                        {"exit_code": 1 if "ghost-seeded" in cmd else 0, "stdout": "", "stderr": ""})
    kept._deploy_into("lab_a", app, b"t", {}, [])
    build_env = [e for c, e in envs if c.startswith("cd /app.new && ")][0]
    assert build_env["PORT"] == "8080" and build_env["API_KEY"] == "s3cret" and build_env["DATA_DIR"] == "/data"


def test_isolated_python_is_named_in_the_deploy(kept, env, monkeypatch):  # noqa: F811
    for cmd in ("python3 -I -m http.server", "cd x && python -I app.py", "python3.12 -sI x.py"):
        assert ISOLATED_PY_RE.search(cmd), cmd
    for cmd in ("python3 app.py", "gunicorn -b 0.0.0.0:8080 app:app", "pythonista -I"):
        assert not ISOLATED_PY_RE.search(cmd), cmd
    make_kept(env, "lab_a", state="up", cmd="python3 -I -m http.server 8080", last_backup_at=0)
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": True, "alive": True})
    monkeypatch.setattr(kept, "_save_built", lambda *a: False)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())      # the folder has requirements.txt / .deps
    assert "ignores PYTHONPATH" in kept.deploy("lab_a", {"archive_b64": "TkVX"})["note"]


def test_wait_for_and_watch_start_with_the_app(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="stopped", wait_for="db:5432", watch=[{"member": "db", "cmd": "keeper"}])
    seen = []

    class C:
        def __init__(self, m):
            self.m = m

        def exec_run(self, cmd, environment=None, **k):
            seen.append((self.m, cmd[-1], environment or {}))
            return SimpleNamespace(exit_code=0, output=b"")
    monkeypatch.setattr(env, "container", lambda lab, m: C(m))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    kept._start_app("lab_a", app, count=False)
    start = [e for m, c, e in seen if "GHOST_APP_CMD" in e][0]["GHOST_APP_CMD"]
    assert start.startswith(wait_for_cmd("db:5432")) and "/dev/tcp/db/5432" in start
    watch = [(m, c, e) for m, c, e in seen if e.get("GHOST_WATCH_CMD")]
    assert watch and watch[0][0] == "db" and watch[0][2]["GHOST_WATCH_CMD"] == "keeper" and "watch.pid" in watch[0][1]


def test_keep_takes_wait_for_and_watch_from_the_template(kept, env, monkeypatch):  # noqa: F811
    from .test_core import add_lab
    add_lab(env, "lab_p", owner="project:p", mem=256)
    env.store.update_lab("lab_p", template="python-pg")
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_p'")
    env.store.put_member({"lab_id": "lab_p", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    for f in ("_recreate", "_lease", "_install_cron"):
        monkeypatch.setattr(kept, f, lambda *a, **k: None)
    monkeypatch.setattr(kept, "_has_code", lambda *a: False)
    kept.keep("lab_p", {"app": {"name": "shop", "cmd": "gunicorn -b 0.0.0.0:8080 app:app"}})
    a = app_of(env, "lab_p")
    assert a["wait_for"] == "db:5432" and a["watch"][0]["member"] == "db" and "pgrep -x initdb" in a["watch"][0]["cmd"]


def test_template_watch_and_wait_for_are_validated():
    base = {"name": "x", "members": [{"name": "m", "image": "ghost-lab/base:1", "mem_mb": 128},
                                     {"name": "d", "image": "ghost-lab/pg:18", "mem_mb": 128}]}
    for bad in ({"wait_for": "db"}, {"wait_for": "db:5432; rm -rf /"}, {"watch": [{"member": "nope", "cmd": "x"}]},
                {"watch": [{"member": "d", "cmd": "a\nb"}]}, {"watch": "x"}):
        with pytest.raises(LabError):
            T.parse({**base, "app": bad})
    ok = T.parse({**base, "app": {"wait_for": "d:5432", "watch": [{"member": "d", "cmd": "keep"}]}})["app"]
    assert ok == {"wait_for": "d:5432", "watch": [{"member": "d", "cmd": "keep"}]}


# ---------------------------------------------------------------- MAJOR 5: the shipped budgets
def test_python_pg_fits_beside_a_512_mb_app_in_the_kept_pool():
    t = T.load_dir(ROOT / "templates")
    mem = {n: sum(m["mem_mb"] for m in t[n]["members"]) for n in t}
    assert mem["python-pg"] <= 1536 - 512 and mem["pg-single"] <= 512 and mem["static-site"] <= 128
    assert mem["python-app"] <= 384
    assert "backed up together" in t["python-pg"]["description"] and "stopped" in t["pg-single"]["description"]


def test_apps_page_shows_a_tcp_address_as_text(env, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient
    from ghost_labs import api
    svc = SimpleNamespace(kept=SimpleNamespace(apps_health=lambda: [
        {"lab_id": "lab_a", "app": "db", "state": "up", "url": None, "address": "disorder.x.ts.net:20007",
         "name_url": None, "lan_url": "192.168.1.50:20007"}]), store=env.store)
    html = TestClient(api.create_apps_page(svc)).get("/").text
    assert "<code>disorder.x.ts.net:20007</code> <small>TCP</small>" in html
    assert 'href="192.168.1.50:20007"' not in html and "192.168.1.50:20007" in html
