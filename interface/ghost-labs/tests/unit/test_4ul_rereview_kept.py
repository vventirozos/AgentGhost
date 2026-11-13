"""§4UL labs RE-review fixes: a saved build unpacks whatever its single top folder is called (workdir renamed),
postgres stops for every /data copy (hook or not) and a restore runs the hook after postgres is back, the revert's
pip reinstall and data note, the build-not-saved note, the db keeper after restarts, HTTP→TCP unserves the name,
set restores name the failed member, the twin seeds its own /data, the -I note."""
import io
import os
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_labs.errors import LabError
from ghost_labs.kept import ISOLATED_PY_RE, Kept
from ghost_labs.store import dumps

from .fakes import TAR  # noqa: E402
from .test_core import ROOT, env  # noqa: F401
from .test_kept_fixes import app_of, make_kept


@pytest.fixture
def kept(env, tmp_path):  # noqa: F811
    k = Kept(env)
    k.archive_root = tmp_path / "archives"
    k.backup_root = tmp_path / "backups"
    env.kept = k
    return k


def _ok(stdout=""):
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


def _bash(cmd):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
    return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def _real_shell(kept, env, monkeypatch, root: Path):
    """The member is the local machine under ``root``: _sh/_must run real bash, put_archive really extracts."""
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: _bash(cmd))

    def must(lab, m, cmd, what, *a, **k):
        r = _bash(cmd)
        if r["exit_code"] != 0:
            raise LabError("internal", f"{what} failed: {r['stderr']}")
        return r

    def put_archive(path, f):
        with tarfile.open(fileobj=io.BytesIO(f.read())) as t:
            t.extractall(path)
        return True
    monkeypatch.setattr(kept, "_must", must)
    return SimpleNamespace(put_archive=put_archive)


def _tar_of(folder: Path, arcname: str) -> bytes:
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as t:
        t.add(folder, arcname=arcname)
    return b.getvalue()


# ---------------------------------------------------------------- MAJOR 1: the saved build after a workdir rename
def test_saved_build_unpacks_under_a_renamed_workdir(kept, env, monkeypatch, tmp_path):  # noqa: F811
    src = tmp_path / "src" / "chess"                       # saved while the workdir was /…/chess
    (src / ".deps").mkdir(parents=True)
    (src / "app.py").write_text("print(1)")
    (src / "saves").symlink_to("/data/saves")              # a seed link the saved folder carries
    built = tmp_path / "x.built.tar"
    built.write_bytes(_tar_of(src, "chess"))
    wd = tmp_path / "member" / "srv" / "game"              # update_app moved the workdir since
    wd.parent.mkdir(parents=True)
    c = _real_shell(kept, env, monkeypatch, tmp_path)
    app = {"member": "m1", "workdir": str(wd)}
    assert kept._unpack_built("lab_a", app, c, str(built), ["saves"]) is True
    new = Path(str(wd) + ".new")
    assert (new / "app.py").read_text() == "print(1)" and (new / ".deps").is_dir()
    assert not os.path.lexists(new / "saves")              # the old link is dropped (linked again after the swap)
    assert not Path(str(wd) + ".x").exists()


def test_saved_build_of_unexpected_shape_falls_back_to_the_archive(kept, env, monkeypatch, tmp_path):  # noqa: F811
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as t:          # two top entries: not a saved folder
        for n in ("a.txt", "b.txt"):
            ti = tarfile.TarInfo(n)
            ti.size = 1
            t.addfile(ti, io.BytesIO(b"x"))
    built = tmp_path / "x.built.tar"
    built.write_bytes(b.getvalue())
    wd = tmp_path / "member" / "app"
    wd.parent.mkdir(parents=True)
    c = _real_shell(kept, env, monkeypatch, tmp_path)
    assert kept._unpack_built("lab_a", {"member": "m1", "workdir": str(wd)}, c, str(built), []) is False
    assert not Path(str(wd) + ".new").exists() and not Path(str(wd) + ".x").exists()


def test_deploy_into_builds_from_the_archive_when_the_saved_build_is_unusable(kept, env, monkeypatch, tmp_path):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    built = tmp_path / "x.built.tar"
    built.write_bytes(TAR)
    cmds = []
    monkeypatch.setattr(kept, "_unpack_built", lambda *a: False)
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    for n in ("_stop_proc", "_install_cron", "_start_app"):
        monkeypatch.setattr(kept, n, lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    res = kept._deploy_into("lab_a", app, b"tgz", {}, [], built=str(built))
    assert any("tar -xzf /tmp/ghost-deploy.tgz" in c for c in cmds) and not res.get("from_saved_build")


# ---------------------------------------------------------------- MAJOR 2: restore order with a hook on a pg member
def _pg_single(env, **extra):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="db", **extra)
    env.store.x("UPDATE members SET name='db', image='ghost-lab/pg:18' WHERE lab_id='lab_a'")
    if not env.store.q("SELECT * FROM members WHERE lab_id='lab_a' AND name='db'"):
        env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                              "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})


def _bk(env, tmp_path, bid="bk_db", member="db"):  # noqa: F811
    bk = tmp_path / f"{bid}.tar"
    bk.write_bytes(TAR)
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", member, 1.0, str(bk), "daily"))


def _step_of(cmd):
    return ("hook" if cmd == "pg_restore_it" else "mark" if cmd.startswith("touch /data/.ghost")
            else "unmark" if cmd.startswith("rm -f /data/.ghost") else "check script" if "sha256sum" in cmd
            else "quiesce" if "pkill" in cmd else "clear")


def _restore_stubs(kept, env, monkeypatch, order, must=None):  # noqa: F811
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: order.append("stop app"))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: order.append("start app"))
    monkeypatch.setattr(kept, "_stop_watch", lambda *a, **k: order.append("stop watch"))
    monkeypatch.setattr(kept, "_start_watch", lambda *a, **k: order.append("start watch"))
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(kept, "_must", must or (lambda lab, m, cmd, *a, **k: order.append(_step_of(cmd)) or _ok()))
    # round 6: the script install is mandatory — the stub takes it (and the check above passes), so every
    # restore test runs the success path
    monkeypatch.setattr(env, "_put_bytes", lambda c, path, data, mode: order.append("install"))
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: order.append("put") or True))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: order.append(
        "pg stop" if "pg_ctl" in cmd else "pg start") or _ok())


def test_restore_of_postgres_files_skips_the_older_hook_dump(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Round-3 MAJOR 1: the stopped file copy is newer than the hook's live dump — running restore_cmd after it
    rolled the database back to the dump (or failed on "already exists" and held a good app)."""
    _pg_single(env, backup={"cmd": "dump", "restore_cmd": "pg_restore_it"})
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    out = kept._restore_locked("lab_a", "bk_db")
    assert "hook" not in order and order[:10] == ["stop app", "install", "check script", "stop watch",
                                                "quiesce", "pg stop", "quiesce", "mark", "clear", "put"]
    assert "hook restore skipped" in out["hook_note"] and "dump is in /data" in out["hook_note"]


def test_set_restore_skips_the_hook_once_a_db_member_was_restored(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="app", backup={"cmd": "dump", "restore_cmd": "pg_restore_it"})
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_a'")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    for bid, mem in (("bk_app", "app"), ("bk_db", "db")):
        (tmp_path / f"{bid}.tar").write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", mem, 5.0,
                                                                   str(tmp_path / f"{bid}.tar"), "daily"))
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    out = kept.restore("lab_a", "bk_app")
    assert "hook" not in order and "hook restore skipped" in out["hook_note"]
    assert out["pre_restore_by_member"] == {"db": "bk_pre", "app": "bk_pre"}


def test_restore_of_an_app_without_postgres_runs_the_hook(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", backup={"cmd": "dump", "restore_cmd": "pg_restore_it"})
    _bk(env, tmp_path, "bk_m1", "m1")
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    out = kept._restore_locked("lab_a", "bk_m1")
    assert order[-4:] == ["put", "unmark", "hook", "start app"] and "hook_note" not in out


def test_a_failed_restore_hook_names_its_step(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", backup={"cmd": "dump", "restore_cmd": "pg_restore_it"})
    _bk(env, tmp_path, "bk_m1", "m1")

    def must(lab, m, cmd, what, *a, **k):
        if cmd == "pg_restore_it":
            raise LabError("internal", "restore hook failed (exit 1)")
        return _ok()
    _restore_stubs(kept, env, monkeypatch, [], must=must)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_m1")
    assert e.value.extra["step"] == "restore hook" and e.value.message.startswith("[restore hook]")
    assert e.value.extra["touched"] is True and e.value.extra["pre_restore"] == "bk_pre"
    a = app_of(env, "lab_a")
    assert "[restore hook]" in a["last_error"] and a["stopped_by_owner"]


def test_postgres_that_does_not_stop_leaves_the_app_as_it_was(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Round-3 MAJOR 2: nothing in /data was touched — not held, the app and its keeper start again."""
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": 1, "stdout": "", "stderr": "timeout"})
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["step"] == "stop postgres"
    assert a["state"] == "up" and not a["stopped_by_owner"] and order[-2:] == ["start watch", "start app"]


def test_a_database_that_did_not_come_back_holds_its_keeper(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    _restore_stubs(kept, env, monkeypatch, [])
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {
        "exit_code": 1 if "10-postgres.sh" in cmd else 0, "stdout": "", "stderr": ""})
    with pytest.raises(LabError):
        kept._restore_locked("lab_a", "bk_db")
    assert app_of(env, "lab_a")["watch_held"] is True


# ---------------------------------------------------------------- MINOR 1 + 2 + 3: the revert and the deploy result
def test_revert_reinstalls_the_previous_pip_list(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    cmds = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or _ok())
    for n in ("_stop_proc", "_install_cron", "_start_app"):
        monkeypatch.setattr(kept, n, lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept._revert_in_place("lab_a", app, (b"OLD", {"pip": ["flask==2.3.3"]}, [], None))
    mv = next(i for i, c in enumerate(cmds) if "mv /app.prev /app" in c)
    pip = next(i for i, c in enumerate(cmds) if "/opt/venv/bin/pip install -q flask==2.3.3" in c)
    assert mv < pip


def test_revert_says_the_failed_versions_packages_stay(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    monkeypatch.setattr(kept, "_must", lambda *a, **k: _ok())
    for n in ("_stop_proc", "_install_cron", "_start_app"):
        monkeypatch.setattr(kept, n, lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    res = kept._revert_in_place("lab_a", app, (b"OLD", {}, [], None), failed_req={"pip": ["numpy==2.1"]})
    assert "numpy==2.1" in res["pip_note"] and "stay in /opt/venv" in res["pip_note"]
    assert "pip_note" not in kept._revert_in_place("lab_a", app, (b"OLD", {}, [], None), failed_req={})


def test_rolled_back_deploy_names_the_pre_deploy_backup_and_the_data(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", last_backup_at=time.time() - 60)
    a = app_of(env, "lab_a")
    a["archive_ts"] = kept._keep_archive("p", "chess", b"OLD", {}, [])
    kept._save("lab_a", a)
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_before"})
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": False, "alive": False, "why": "crashed"})
    monkeypatch.setattr(kept, "_revert_in_place", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(kept, "_save_built", lambda *a: False)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert out["rolled_back"] == "crashed" and out["pre_deploy_backup"] == "bk_before"
    assert "as the failed version left it" in out["data_note"] and "bk_before" in out["data_note"]
    assert "builds it again" in out["built_not_saved"]


def test_deploy_says_nothing_about_the_build_when_it_was_saved(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", last_backup_at=None)
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": True, "alive": True})
    monkeypatch.setattr(kept, "_save_built", lambda *a: True)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert "built_not_saved" not in out and "data_note" not in out and "pre_deploy_backup" not in out


# ---------------------------------------------------------------- MINOR 4: the db keeper after restarts
def test_watch_pid_counts_only_while_it_is_the_keeper(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", watch=[{"member": "db", "cmd": "keeper"}])
    seen = []
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda cmd, **k: seen.append(cmd[-1]) or SimpleNamespace(exit_code=0, output=b"")))
    kept._start_watch("lab_a", app)
    assert "/proc/$p/environ" in seen[0] and "GHOST_WATCH_CMD=" in seen[0]
    entry = (ROOT / "images" / "base" / "lab-entrypoint").read_text()
    assert "/run/ghost/watch.pid" in entry


def test_a_watched_member_restarting_alone_gets_its_keeper_back(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", member="app", watch=[{"member": "db", "cmd": "keeper"}])
    got = []
    monkeypatch.setattr(kept, "_start_watch", lambda lab, app, only=None: got.append(only))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: pytest.fail("the app is not restarted for its db"))
    kept.on_member_start("lab_a", "db")
    assert got == [["db"]]
    a = app_of(env, "lab_a")
    a.update(stopped_by_owner=True, state="stopped")        # round-3 MINOR 9: an owner stop never stops the keeper
    kept._save("lab_a", a)
    kept.on_member_start("lab_a", "db")
    assert got == [["db"], ["db"]]
    a["watch_held"] = True                                  # a failed restore of the database holds it
    kept._save("lab_a", a)
    kept.on_member_start("lab_a", "db")
    assert got == [["db"], ["db"]]


# ---------------------------------------------------------------- MINOR 5: HTTP → TCP
def test_http_app_turned_tcp_unserves_its_name(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", port=5432)
    env.cfg.ts_services = True
    a = app_of(env, "lab_a")
    monkeypatch.setattr(kept, "_verify_name", lambda *a, **k: True)
    kept._lease("lab_a", a, move=True)
    assert a["url"]
    gone = []
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: gone.append(name) or True)
    a.update(name_hint="define the Service", health={"http": None, "tcp": True})
    kept._lease("lab_a", a, move=True)
    assert gone == ["chess"] and a["url"] is None and a["name_url"] is None and "name_hint" not in a
    kept._lease("lab_a", a, move=True)
    assert gone == ["chess"]                                # an app that is already TCP: nothing to unserve


def test_failed_unserve_keeps_the_name_and_is_retried(kept, env, monkeypatch):  # noqa: F811
    """Round-4 M5: only the unserve is retried (with backoff) — the lease is never re-published for it."""
    make_kept(env, "lab_a", state="up", port=5432, name_url="https://chess.x/", name_state="live")
    env.cfg.ts_services = True
    a = app_of(env, "lab_a")
    a["health"] = {"http": None, "tcp": True}
    res = iter([False, True])
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: next(res))
    kept._lease("lab_a", a, move=True)
    assert a["name_url"] == "https://chess.x/" and a["name_unserve_pending"] and "could not unserve" in a["last_error"]
    assert kept._view(a)["name_url"] is None and kept._view(a)["name_state"] == "removing name"
    kept._save("lab_a", a)
    monkeypatch.setattr(kept, "_lease", lambda *a, **k: pytest.fail("the lease did not move: never re-published"))
    kept.ensure_leases()                                    # before its retry time: nothing
    assert app_of(env, "lab_a")["name_unserve_pending"]
    b = app_of(env, "lab_a")
    b["unserve_retry_at"] = 0
    kept._save("lab_a", b)
    kept.ensure_leases()
    b = app_of(env, "lab_a")
    assert not b.get("name_unserve_pending") and b["name_url"] is None and b["last_error"] is None


def test_unserve_retry_backs_off_and_gives_up(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", port=5432, health={"http": None, "tcp": True}, name_unserve_pending=True,
              unserve_tries=0, unserve_retry_at=0, address="disorder:20001")
    env.store.x("INSERT INTO ports (port, lab_id, member, cport, lease_key) VALUES (?,?,?,?,?)",
                (20001, "lab_a", "m1", 5432, "p/chess"))            # the lease is where it belongs
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: False)
    monkeypatch.setattr(kept, "_lease", lambda *a, **k: pytest.fail("never re-published"))
    waits = []
    for _ in range(6):
        b = app_of(env, "lab_a")
        b["unserve_retry_at"] = 0
        kept._save("lab_a", b)
        t0 = time.time()
        kept.ensure_leases()
        b = app_of(env, "lab_a")
        if b.get("unserve_retry_at"):
            waits.append(round((b["unserve_retry_at"] - t0) / 60))
    assert waits == [2, 4, 8, 16, 32]           # after the first retry (1 min after the failed unserve)
    assert not b.get("name_unserve_pending") and "after 6 tries" in b["last_error"]


def test_unserve_with_nothing_to_clear_counts_as_done(env):  # noqa: F811
    import subprocess
    env.ports.run = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "error: service svc:chess not found")
    assert env.ports.unserve_service("chess") is True
    env.ports.run = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "error: permission denied")
    assert env.ports.unserve_service("chess") is False


def test_name_fields_only_when_the_service_was_served(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    env.cfg.ts_services = True
    monkeypatch.setattr(env.ports, "serve_service", lambda name, port: False)
    monkeypatch.setattr(kept, "_verify_name", lambda *a, **k: pytest.fail("no name to check"))
    a = app_of(env, "lab_a")
    kept._lease("lab_a", a, move=True)
    assert a["url"] and a["name_url"] is None and a["name_state"] == "serve_failed"


def test_switch_to_a_tcp_version_unserves_the_live_versions_name(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_old", state="up", url="https://disorder:20001/", name_url="https://chess.x/",
              name_state="live", mounted=False)
    env.store.x("INSERT INTO ports (port, lab_id, member, cport, lease_key) VALUES (?,?,?,?,?)",
                (20001, "lab_old", "m1", 8080, "p/chess")) if not env.store.q(
        "SELECT * FROM ports WHERE lease_key='p/chess'") else None
    make_kept(env, "lab_new", state="staged", port=5432, health={"http": None, "tcp": True}, url=None)
    env.cfg.ts_services = True
    gone = []
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: gone.append(name) or True)
    a = app_of(env, "lab_new")
    assert kept._served_as_http("lab_new", a)
    kept._lease("lab_new", a, move=True)
    assert gone == ["chess"]


def test_tcp_to_http_turns_the_tcp_serve_off(env):  # noqa: F811
    env.ports.publish(20007, "10.0.0.2", 8080, tcp=False)
    calls = env.ports.run.calls
    off = calls.index(["tailscale", "serve", "--tcp=20007", "off"])
    on = next(i for i, c in enumerate(calls) if "--https=20007" in c and "http://127.0.0.1:20007" in c)
    assert off < on


def test_name_check_never_marks_a_tcp_app_live(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", port=5432, health={"http": None, "tcp": True}, name_url=None)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: SimpleNamespace(close=lambda: None))
    kept._verify_name("lab_a", "chess", "https://chess.tail1234.ts.net/", tries=1, every=0)
    assert app_of(env, "lab_a").get("name_state") != "live" and not app_of(env, "lab_a").get("name_url")


# ---------------------------------------------------------------- MINOR 6: a set restore failing at a member
def test_set_restore_failing_with_a_docker_error_names_the_member(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="app")
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_a'")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    for bid, mem in (("bk_app", "app"), ("bk_db", "db")):
        (tmp_path / f"{bid}.tar").write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", mem, 5.0,
                                                                   str(tmp_path / f"{bid}.tar"), "daily"))

    def boom(lab, sid, snapshot=True, start=True, **k):
        raise RuntimeError("docker: container db is not running")
    monkeypatch.setattr(kept, "_restore_locked", boom)
    with pytest.raises(LabError) as e:
        kept.restore("lab_a", "bk_app")
    assert e.value.extra["failed_member"] == "db" and e.value.extra["restored_before"] == []
    a = app_of(env, "lab_a")                    # round-3 MAJOR 2: nothing was touched — a running app is left alone
    assert a["state"] == "up" and not a.get("stopped_by_owner") and not a["last_error"]

    def second_fails(lab, sid, snapshot=True, start=True, **k):
        if sid == "bk_app":
            raise RuntimeError("docker: container app is not running")
        return {"pre_restore": "bk_pre_db", "probe": {"ok": None}}
    monkeypatch.setattr(kept, "_restore_locked", second_fails)
    with pytest.raises(LabError) as e:
        kept.restore("lab_a", "bk_app")
    assert e.value.extra["failed_member"] == "app" and e.value.extra["restored_before"] == ["bk_db"]
    assert e.value.extra["pre_restore_by_member"] == {"db": "bk_pre_db"}
    a = app_of(env, "lab_a")                    # the db was already put back: held, and it says where things are
    assert "member 'app'" in a["last_error"] and "bk_pre_db" in a["last_error"] and a["stopped_by_owner"]


# ---------------------------------------------------------------- MINOR 7: the twin's own /data
def test_twin_seeds_its_own_data(kept, env, monkeypatch):  # noqa: F811
    cmds = []
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    monkeypatch.setattr(env, "_put_bytes", lambda c, path, data, mode: cmds.append(f"put {path}"))
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    notes = kept._twin_seeds("lab_t", {"member": "m1", "workdir": "/app"}, b"tgz", ["saves"])
    assert notes == []
    link = next(c for c in cmds if "ln -s" in c)
    assert link.index("rm -f /app/saves") < link.index("cp -a /tmp/twin-seed/saves /app/saves") < link.index("ln -s")
    assert "put /tmp/twin-seed.tgz" in cmds and cmds[-1].startswith("rm -rf /tmp/twin-seed")


def test_twin_lists_a_seed_it_could_not_make(kept, env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: _ok())
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {
        "exit_code": 8 if "ln -s" in cmd else 0, "stdout": "", "stderr": "seed path is in neither"})
    assert kept._twin_seeds("lab_t", {"member": "m1", "workdir": "/app"}, b"t", ["saves"])[0].startswith("saves:")


# ---------------------------------------------------------------- the -I note
@pytest.mark.parametrize("cmd,hit", [("python3 -I -m http.server", True), ("/opt/venv/bin/python3 -I app.py", True),
                                     ("python -u -I x.py", True), ("python3 -X dev -I a.py", True),
                                     ("python3 app.py -I", False), ("python3 -m foo -I", False), ("node x", False)])
def test_isolated_python_regex(cmd, hit):
    assert bool(ISOLATED_PY_RE.search(cmd)) is hit


@pytest.mark.parametrize("has_deps", [True, False])
def test_isolated_note_only_when_the_folder_has_packages(kept, env, monkeypatch, has_deps):  # noqa: F811
    make_kept(env, "lab_a", state="up", cmd="/opt/venv/bin/python3 -I app.py", last_backup_at=None)
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": True, "alive": True})
    monkeypatch.setattr(kept, "_save_built", lambda *a: True)
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {
        "exit_code": 0 if has_deps or "requirements.txt" not in cmd else 1, "stdout": "", "stderr": ""})
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert ("note" in out) is has_deps


# ---------------------------------------------------------------- round 3: smaller fixes
def test_unpack_built_falls_back_when_docker_raises(kept, env, monkeypatch, tmp_path):  # noqa: F811
    built = tmp_path / "x.built.tar"
    built.write_bytes(TAR)
    cmds = []
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())

    def put_archive(path, f):
        raise RuntimeError("docker: 500 Server Error")
    ok = kept._unpack_built("lab_a", {"member": "m1", "workdir": "/app"}, SimpleNamespace(put_archive=put_archive),
                            str(built), [])
    assert ok is False and cmds[-1] == "rm -rf /app.new /app.x"


def test_isolated_note_on_keep_update_and_deploy(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", stopped_by_owner=True)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.update_app("lab_a", {"cmd": "python3 -E app.py"})
    assert "-I or -E" in out["note"]
    assert "note" not in kept.update_app("lab_a", {"cmd": "python3 app.py"})


def test_failed_backup_saves_what_its_cleanup_set(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="app")
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_a'")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_stop_watch", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_start_watch", lambda *a, **k: None)
    monkeypatch.setattr(kept, "_start_app", lambda lab, app, count: app.update(state="starting"))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: (
        {"exit_code": 1, "stdout": "", "stderr": "no"} if "10-postgres.sh" in cmd else _ok("was-running")))

    def copy(*a):
        raise RuntimeError("docker: get_archive failed")
    monkeypatch.setattr(kept, "_copy_data", copy)
    with pytest.raises(RuntimeError):
        kept._backup_locked("lab_a", "daily")
    a = app_of(env, "lab_a")
    assert a["state"] == "starting" and "after the backup" in a["last_error"]


# ---------------------------------------------------------------- round 4
def _app_and_db(env, lab="lab_a"):  # noqa: F811
    make_kept(env, lab, state="up", member="app", backup={"cmd": "dump", "restore_cmd": "pg_restore_it"})
    env.store.x("UPDATE members SET name='app' WHERE lab_id=?", (lab,))
    env.store.put_member({"lab_id": lab, "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})


def test_undo_of_an_app_member_snapshot_never_runs_the_hook(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Round-4 M1 (the reviewer's probe): the app member's own pre-restore snapshot, restored alone in a lab with a
    postgres member, ran restore_cmd — any postgres member now means the hook never restores."""
    _app_and_db(env)
    p = tmp_path / "bk_preapp.tar"
    p.write_bytes(TAR)
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_preapp", "p", "chess", "app", 9.0, str(p), "pre_restore"))
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    out = kept.restore("lab_a", "bk_preapp")
    assert "hook" not in order and "hook restore skipped" in out["hook_note"]


def test_rollback_with_a_hook_never_runs_it_and_undoes_db_first(kept, env, monkeypatch, tmp_path):  # noqa: F811
    from .test_kept_fixes import _two_versions
    _two_versions(env, kept, monkeypatch, tmp_path)
    a1 = app_of(env, "lab_v1")
    a1["backup"] = {"cmd": "dump", "restore_cmd": "pg_restore_it"}
    kept._save("lab_v1", a1)
    env.store.put_member({"lab_id": "lab_v1", "name": "db", "ip": "10.89.9.12", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    for bid, mem in (("bk_sdb", "db"), ("bk_sapp", "m1")):
        (tmp_path / f"{bid}.tar").write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", mem, 1.0,
                                                                   str(tmp_path / f"{bid}.tar"), "pre_switch"))
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_sapp",
             switch_snapshots=["bk_sdb", "bk_sapp"])
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    out = kept.rollback("lab_v2", restore_snapshot=True)
    assert out["lab_id"] == "lab_v1" and "hook" not in order

    # the undo after a failed rollback restores in the restore order: db first, the app member last
    seen = []

    def restore(lab, bid, snapshot=True, start=True, **k):
        seen.append(bid)
        if bid == "bk_sapp":
            raise LabError("internal", "put failed", pre_restore="bk_pre_app")
        return {"pre_restore": "bk_pre_db" if snapshot else None, "probe": {"ok": None}}
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_sapp",
             switch_snapshots=["bk_sdb", "bk_sapp"])
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_restore_locked", restore)
    monkeypatch.setattr(kept, "_reverse", lambda *a: {"ok": True})
    kept.rollback("lab_v2", restore_snapshot=True)
    assert seen == ["bk_sdb", "bk_sapp", "bk_pre_db", "bk_pre_app"]


def test_snapshot_of_a_non_app_member_runs_no_hook(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _app_and_db(env)
    ran = []
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, *a, **k: ran.append(cmd) or _ok())
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    for n in ("_stop_proc", "_start_app", "_stop_watch", "_start_watch"):
        monkeypatch.setattr(kept, n, lambda *a, **k: None)
    (tmp_path / "b.tar").write_bytes(b"x")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": f"bk_{mem}", "member": mem, "path": str(tmp_path / "b.tar"), "bytes": 1})
    kept._backup_locked("lab_a", "pre_restore", only_member="db")
    assert "dump" not in ran


def test_restoring_the_oldest_snapshot_keeps_its_own_target(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Round-4 M2 (the reviewer's probe): the pre-restore snapshot's rotation deleted the very backup being
    restored, after which /data was cleared and the open failed."""
    make_kept(env, "lab_a", state="stopped")
    for i in range(7):
        p = tmp_path / f"bk_p{i}.tar"
        p.write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (f"bk_p{i}", "p", "chess", "m1", float(i + 1),
                                                                   str(p), "pre_restore"))
    n = [0]

    def copy(lab, lin, a, mem, kind):
        n[0] += 1
        q = tmp_path / f"bk_new{n[0]}.tar"
        q.write_bytes(b"n")
        return {"backup_id": f"bk_new{n[0]}", "member": mem, "path": str(q), "bytes": 1}
    monkeypatch.setattr(kept, "_copy_data", copy)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    monkeypatch.undo()                                       # the REAL _backup_locked (the rotation)
    monkeypatch.setattr(kept, "_copy_data", copy)
    for k_, v in (("_stop_proc", lambda *a: None), ("_start_app", lambda *a, **k: None),
                  ("wait_up", lambda *a, **k: {"ok": True}), ("_sh", lambda *a, **k: _ok()),
                  ("_must", lambda lab, m, cmd, *a, **k: order.append(cmd) or _ok())):
        monkeypatch.setattr(kept, k_, v)
    got = []
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        put_archive=lambda p, f: got.append(f.read()) or True))
    out = kept.restore("lab_a", "bk_p0")
    assert got == [TAR] and out["restored"] == "bk_p0" and (tmp_path / "bk_p0.tar").exists()


def test_a_held_app_gets_no_keeper_and_no_daily_backup(kept, env, monkeypatch):  # noqa: F811
    """Round-4 M3 (the reviewer's probe): a backup of a held app restarted the keeper, which initdb'd an empty
    cluster; later dailies rotated the good copies out."""
    _app_and_db(env)
    a = app_of(env, "lab_a")
    a.update(state="stopped", stopped_by_owner=True, held=True, watch_held=True, watch=[{"member": "db", "cmd": "k"}],
             last_backup_at=0)
    kept._save("lab_a", a)
    execs = []
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda *a, **k: execs.append(a) or SimpleNamespace(exit_code=0, output=b"")))
    kept._start_watch("lab_a", a)
    assert execs == []
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("no daily backup of a held app"))
    kept._daily_backup("lab_a")
    spawned = []
    monkeypatch.setattr(kept, "_spawn_daily_backup", lambda lab: spawned.append(lab))
    monkeypatch.setattr(kept, "_alive", lambda *a: False)
    kept._last_daily = time.time()
    kept.tick()
    assert spawned == []


def test_starting_the_app_releases_the_holds_and_the_keeper(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="stopped", held=True, watch_held=True, watch=[{"member": "db", "cmd": "k"}])
    execs = []
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda cmd, **k: execs.append(cmd[-1]) or SimpleNamespace(exit_code=0, output=b"")))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    kept._start_app("lab_a", app, count=False)
    assert "held" not in app and "watch_held" not in app and any("GHOST_WATCH_CMD" in c for c in execs)


def test_a_failing_hook_never_blocks_a_restore(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """Round-4 M4: the pre-restore snapshot takes the file copy anyway and notes the hook."""
    make_kept(env, "lab_a", state="up", backup={"cmd": "broken_dump", "restore_cmd": None})
    _bk(env, tmp_path, "bk_m1", "m1")

    def must(lab, m, cmd, what, *a, **k):
        if cmd == "broken_dump":
            raise LabError("internal", "backup hook failed (exit 2)")
        return _ok()
    monkeypatch.setattr(kept, "_must", must)
    copies = []
    (tmp_path / "c.tar").write_bytes(b"c")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: copies.append(kind) or {
        "backup_id": f"bk_pre{len(copies)}", "member": mem, "path": str(tmp_path / "c.tar"), "bytes": 1})
    stops = []
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: stops.append(1))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    out = kept.restore("lab_a", "bk_m1")
    assert out["restored"] == "bk_m1" and out["pre_restore"] == "bk_pre1" and copies == ["pre_restore"]
    assert stops                                             # copied like an app with no hook: stopped for it
    r = kept._backup_locked("lab_a", "daily")
    assert "hook failed" in r["hook_error"] and "hook failed" in app_of(env, "lab_a")["last_error"]


def test_untouched_failure_whose_restart_fails_is_not_left_in_progress(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    _restore_stubs(kept, env, monkeypatch, [])
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": 1, "stdout": "", "stderr": "timeout"})

    def boom(*a, **k):
        raise RuntimeError("docker: exec failed")
    monkeypatch.setattr(kept, "_start_app", boom)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["restarted"] is False
    assert "in progress" not in a["last_error"] and "did not start again" in a["last_error"]
    assert not a.get("stopped_by_owner") and not a.get("held")


# ---------------------------------------------------------------- round 5
def _run_pg_script(tmp_path, marker: bool, restoring: bool = False, cluster: bool = False):
    """Run images/pg/10-postgres.sh with sudo/chown/initdb/postgres faked: what would it have run?"""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    for n in ("sudo", "chown", "chmod"):
        (fake / n).write_text(f'#!/bin/bash\necho "{n} $*" >> {log}\n')
        (fake / n).chmod(0o755)
    pgdata = tmp_path / "pg"
    pgdata.mkdir(exist_ok=True)
    if cluster:
        (pgdata / "PG_VERSION").write_text("18")
    mark = tmp_path / ".ghost-restore-incomplete"
    if marker:
        mark.write_text("")
    env = {"PATH": f"{fake}:/usr/bin:/bin", "PGDATA": str(pgdata), "GHOST_RESTORE_MARKER": str(mark)}
    if restoring:
        env["GHOST_PG_RESTORING"] = "1"
    r = subprocess.run(["bash", str(ROOT / "images" / "pg" / "10-postgres.sh")], env=env, capture_output=True,
                       text=True, timeout=30)
    return r, (log.read_text() if log.exists() else "")


def test_pg_script_never_initialises_or_starts_behind_the_marker(tmp_path):
    """Round-5 M1: any container start (reboot, OOM, a recreate) with the marker: no initdb, no postgres."""
    r, calls = _run_pg_script(tmp_path, marker=True)
    assert r.returncode == 0 and "restore of /data did not complete" in r.stderr
    assert "initdb" not in calls and "postgres" not in calls and "chown" not in calls


def test_pg_script_restoring_starts_the_files_but_never_initdb(tmp_path):
    r, calls = _run_pg_script(tmp_path, marker=True, restoring=True)          # no cluster in the restored files
    assert r.returncode == 1 and "initdb" not in calls and "no cluster" in r.stderr
    r, calls = _run_pg_script(tmp_path, marker=True, restoring=True, cluster=True)
    assert "initdb" not in calls and "/postgres -D" in calls


def test_pg_script_without_the_marker_initialises_as_before(tmp_path):
    r, calls = _run_pg_script(tmp_path, marker=False)
    assert "initdb" in calls


def test_restore_writes_the_marker_before_the_clear_and_records_it_first(kept, env, monkeypatch, tmp_path):
    _pg_single(env)
    _bk(env, tmp_path)
    seen = []

    def must(lab, m, cmd, what, *a, **k):
        seen.append((cmd.split()[0], app_of(env, "lab_a").get("restore_incomplete"), app_of(env, "lab_a").get("watch_held")))
        return _ok()
    _restore_stubs(kept, env, monkeypatch, [], must=must)
    kept._restore_locked("lab_a", "bk_db")
    seen = [(c if c != "pgrep" else "find", *rest) for c, *rest in seen]   # round 7: the clear checks postgres first
    seen = [x for x in seen if x[0] in ("touch", "find", "rm")]      # (the script check and quiesce: round 6)
    assert [s[0] for s in seen] == ["touch", "find", "rm"] and seen[0][1] == ["db"] and seen[0][2] is True
    a = app_of(env, "lab_a")                    # (the app start, stubbed here, releases `held`)
    assert "restore_incomplete" not in a and "watch_held" not in a


def test_the_clear_spares_the_marker(kept, env, monkeypatch, tmp_path):
    _pg_single(env)
    _bk(env, tmp_path)
    cmds = []
    _restore_stubs(kept, env, monkeypatch, [], must=lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    kept._restore_locked("lab_a", "bk_db")
    clear = next(c for c in cmds if "find /data" in c)
    assert clear.startswith("pgrep -x postgres >/dev/null && exit 3; ")    # round 7: nothing goes while it runs
    assert "! -name .ghost-restore-incomplete" in clear


def _held_incomplete(env, lab="lab_a", **extra):  # noqa: F811
    make_kept(env, lab, state="stopped", stopped_by_owner=True, held=True, watch_held=True,
              restore_incomplete=["m1"], last_error="restore of bk_x failed: [replace /data] …; held", **extra)


def test_a_failed_restore_is_not_released_by_a_plain_start(kept, env, monkeypatch):  # noqa: F811
    _held_incomplete(env)
    with pytest.raises(LabError) as e:
        kept.app_action("lab_a", "start")
    assert "reinit" in e.value.hint and e.value.extra["restore_incomplete"] == ["m1"]
    a = app_of(env, "lab_a")
    kept._start_app("lab_a", a, count=False)                # the backstop: no start, nothing released
    assert a["held"] and a["restore_incomplete"] == ["m1"] and "not started" in a["last_error"]


def test_start_with_reinit_accepts_the_data(kept, env, monkeypatch):  # noqa: F811
    _held_incomplete(env)
    cmds = []
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, *a, **k: cmds.append(cmd) or _ok())
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda *a, **k: SimpleNamespace(exit_code=0, output=b"")))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept.app_action("lab_a", "start", reinit=True)
    a = app_of(env, "lab_a")
    assert cmds == ["rm -f /data/.ghost-restore-incomplete"]
    assert not a.get("restore_incomplete") and not a.get("held") and not a.get("watch_held")


def test_deploy_on_a_held_app_installs_without_starting(kept, env, monkeypatch):  # noqa: F811
    _held_incomplete(env, last_backup_at=None)
    seen = []
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: seen.append(k.get("start")) or {"ok": None})
    monkeypatch.setattr(kept, "_save_built", lambda *a: True)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert seen == [False] and "not started" in out["held"] and "reinit" in out["held"]


def test_rehearse_and_switch_refuse_a_held_app(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _held_incomplete(env)
    a = app_of(env, "lab_a")
    a["archive_ts"] = kept._keep_archive("p", "chess", b"OLD", {}, [])
    kept._save("lab_a", a)
    with pytest.raises(LabError) as e:
        kept.rehearse("lab_a")
    assert "held" in e.value.message and "reinit" in e.value.hint
    make_kept(env, "lab_b", state="staged", mounted=False, data="shared")
    with pytest.raises(LabError) as e:
        kept.switch("lab_b", "lab_a")
    assert "held" in e.value.message


def test_a_failed_final_backup_puts_every_service_back(kept, env, monkeypatch):  # noqa: F811
    """Round-5 M2 (the reviewer's probe, inverted): a delete's final backup that fails restarts postgres, its
    keeper and the app, and says so."""
    _pg_single(env)
    calls = []
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: calls.append("stop app"))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: calls.append("start app"))
    monkeypatch.setattr(kept, "_stop_watch", lambda *a, **k: calls.append("stop watch"))
    monkeypatch.setattr(kept, "_start_watch", lambda *a, **k: calls.append("start watch"))
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: calls.append(
        "pg stop" if "pg_ctl" in cmd else "pg start") or _ok("was-running"))

    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(kept, "_copy_data", boom)
    with pytest.raises(OSError):
        kept._backup_locked("lab_a", "final", restart=False)
    a = app_of(env, "lab_a")
    assert calls[-3:] == ["pg start", "start watch", "start app"]
    assert "final backup failed" in a["last_error"] and "nothing was deleted" in a["last_error"]
    assert a["state"] != "stopped" or calls[-1] == "start app"


def test_delete_reports_a_failed_final_backup(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up")

    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(kept, "_backup_locked", boom)
    with pytest.raises(LabError) as e:
        kept._destroy_with_data("lab_a", "owner")
    assert "nothing was deleted" in e.value.message and env.store.lab("lab_a")


def test_back_to_http_drops_a_pending_unserve(kept, env, monkeypatch):  # noqa: F811
    """Round-5 MINOR 1 (the reviewer's probe, inverted): no unserve of a name that is served again."""
    make_kept(env, "lab_a", state="up", port=5432, name_url="https://chess.x/", name_state="live")
    env.cfg.ts_services = True
    a = app_of(env, "lab_a")
    a["health"] = {"http": None, "tcp": True}
    calls = []
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: calls.append("unserve") or False)
    monkeypatch.setattr(env.ports, "serve_service", lambda name, port: calls.append("serve") or True)
    monkeypatch.setattr(kept, "_verify_name", lambda *a, **k: True)
    kept._lease("lab_a", a, move=True)
    a["health"] = {"http": "/", "tcp": True}
    kept._lease("lab_a", a, move=True)
    assert not a.get("name_unserve_pending") and calls == ["unserve", "serve"]
    a.update(name_unserve_pending=True, unserve_retry_at=0)     # a stale flag on an HTTP app
    kept._save("lab_a", a)
    kept._retry_unserve("lab_a")
    assert calls == ["unserve", "serve"] and not app_of(env, "lab_a").get("name_unserve_pending")


def test_a_restore_of_a_cmdless_app_releases_its_hold(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", cmd="", service=True)
    _bk(env, tmp_path, "bk_m1", "m1")
    _restore_stubs(kept, env, monkeypatch, [])
    monkeypatch.setattr(kept, "_start_app", Kept._start_app.__get__(kept))
    kept._restore_locked("lab_a", "bk_m1")
    assert not app_of(env, "lab_a").get("held")


def test_a_backup_of_a_held_app_keeps_the_holds_explanation(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", stopped_by_owner=True, held=True,
              backup={"cmd": "dump", "restore_cmd": None}, protected_backups=["bk_bbbbbbbbbb"],
              last_error="restore of bk_aaaaaaaaaa failed (pre-restore snapshot bk_bbbbbbbbbb); held")

    def must(*a, **k):
        raise LabError("internal", "backup hook failed (exit 2)")
    monkeypatch.setattr(kept, "_must", must)
    (tmp_path / "c.tar").write_bytes(b"c")
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": "bk_cccccccccc", "member": mem, "path": str(tmp_path / "c.tar"), "bytes": 1})
    kept.backup("lab_a")
    a = app_of(env, "lab_a")
    assert a["last_error"].startswith("restore of bk_aaaaaaaaaa") and "hook failed" in a["last_error"]


def test_protected_snapshots_survive_rotation_across_failed_restores(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", protected_backups=["bk_p0"], last_error="something else")
    for i in range(8):
        p = tmp_path / f"bk_p{i}.tar"
        p.write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (f"bk_p{i}", "p", "chess", "m1", float(i + 1),
                                                                   str(p), "manual"))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": "bk_new", "member": mem, "path": str(tmp_path / "bk_new.tar"), "bytes": 1})
    (tmp_path / "bk_new.tar").write_bytes(TAR)
    kept._backup_locked("lab_a", "manual")
    assert (tmp_path / "bk_p0.tar").exists() and not (tmp_path / "bk_p1.tar").exists()


def test_set_restore_checks_every_file_before_touching_anything(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up", member="app")
    env.store.x("UPDATE members SET name='app' WHERE lab_id='lab_a'")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18",
                          "mem_mb": 512, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    for bid, mem, body in (("bk_app", "app", b"not a tar at all"), ("bk_db", "db", TAR)):
        (tmp_path / f"{bid}.tar").write_bytes(body)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (bid, "p", "chess", mem, 5.0,
                                                                   str(tmp_path / f"{bid}.tar"), "daily"))
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("nothing is snapshotted"))
    monkeypatch.setattr(kept, "_must", lambda *a, **k: pytest.fail("nothing is cleared"))
    with pytest.raises(LabError) as e:
        kept.restore("lab_a", "bk_db")
    assert e.value.code == "unknown_backup" and "bk_app" in e.value.message
    assert app_of(env, "lab_a")["state"] == "up"


def test_first_unserve_retry_is_one_minute_after_the_failure(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", port=5432, name_url="https://chess.x/", name_state="live")
    env.cfg.ts_services = True
    a = app_of(env, "lab_a")
    a["health"] = {"http": None, "tcp": True}
    monkeypatch.setattr(env.ports, "unserve_service", lambda name: False)
    t0 = time.time()
    kept._lease("lab_a", a, move=True)
    assert round((a["unserve_retry_at"] - t0) / 60) == 1
