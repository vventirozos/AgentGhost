"""§4UL round-6 review: the marker-aware start script is installed and CHECKED before a restore changes anything (a
failed install leaves the app as it was), a start script the keeper launched is killed before the marker and the
clear, the script checks the marker again right before initdb and before postgres, mysql honours the marker, and the
hold keeps its own explanation (``hold_reason``) and its protected snapshots."""
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_labs import kept as K
from ghost_labs.errors import LabError
from ghost_labs.kept import PG_QUIESCE, Kept

from .fakes import TAR  # noqa: E402
from .test_core import ROOT, env  # noqa: F401
from .test_kept_fixes import app_of, make_kept
from .test_4ul_rereview_kept import (_bk, _held_incomplete, _ok, _pg_single, _restore_stubs, _run_pg_script,  # noqa: F401
                                     kept)


def _bash(cmd, env=None):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60, env=env)
    return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


# ---------------------------------------------------------------- M1: the script install is mandatory and checked
@pytest.mark.parametrize("how", ["put fails", "check fails", "no script on the host"])
def test_a_failed_script_install_leaves_the_app_and_data_untouched(kept, env, monkeypatch, tmp_path, how):  # noqa: F811
    _pg_single(env, protected_backups=["bk_old"])
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    if how == "put fails":
        def put(*a):
            raise OSError("docker: no such container")
        monkeypatch.setattr(env, "_put_bytes", put)
    elif how == "check fails":
        def must(lab, m, cmd, what, *a, **k):
            order.append("check script" if "sha256sum" in cmd else cmd.split()[0])
            if "sha256sum" in cmd:
                raise LabError("internal", f"{what} failed (exit 1)")
            return _ok()
        monkeypatch.setattr(kept, "_must", must)
    else:
        monkeypatch.setattr(K, "PG_SCRIPT", tmp_path / "missing.sh")
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["step"] == "install start script"
    # nothing was stopped, marked or cleared; the app runs again; no hold and no protection left behind
    assert not {"pg stop", "mark", "clear", "put", "stop watch"} & set(order)
    assert a["state"] == "up" and not a.get("held") and not a.get("restore_incomplete") and not a.get("watch_held")
    assert a["protected_backups"] == ["bk_old"] and not a.get("restore_running")


def test_the_install_check_is_a_real_checksum_of_this_script(kept, env, monkeypatch, tmp_path):  # noqa: F811
    """The check command run for real: the file put in place passes, a different (older) one does not."""
    dest = tmp_path / "start.d" / "10-postgres.sh"
    dest.parent.mkdir()
    monkeypatch.setattr(K, "PG_SCRIPT_DEST", str(dest))

    def must(lab, m, cmd, what, *a, **k):
        r = _bash(cmd)
        if r["exit_code"] != 0:
            raise LabError("internal", f"{what} failed (exit {r['exit_code']})")
        return r
    monkeypatch.setattr(kept, "_must", must)
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    def put(c, path, data, mode):
        Path(path).write_bytes(data)
        os.chmod(path, mode)
    monkeypatch.setattr(env, "_put_bytes", put)
    kept._install_pg_script("lab_a", "db")
    assert dest.read_bytes() == K.PG_SCRIPT.read_bytes()
    monkeypatch.setattr(env, "_put_bytes", lambda c, path, data, mode: Path(path).write_text("#!/bin/bash\ninitdb\n"))
    with pytest.raises(LabError) as e:
        kept._install_pg_script("lab_a", "db")
    assert "check the installed postgres start script" in e.value.message


# ---------------------------------------------------------------- M2: a running start script is killed first
def test_a_running_start_script_is_killed_before_the_marker_and_the_clear(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    kept._restore_locked("lab_a", "bk_db")
    assert order.index("quiesce") < order.index("pg stop") < len(order) - 1 - order[::-1].index("quiesce") \
        < order.index("mark") < order.index("clear")


def test_a_start_script_that_will_not_die_fails_the_restore_untouched(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)

    def must(lab, m, cmd, what, *a, **k):
        if "pkill" in cmd:
            raise LabError("internal", f"{what} failed (exit 1): still running: 77 bash start.d/10-postgres.sh")
        order.append(cmd.split()[0])
        return _ok()
    monkeypatch.setattr(kept, "_must", must)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["step"] == "stop postgres"
    assert "touch" not in order and "find" not in order and not a.get("restore_incomplete") and a["state"] == "up"


def test_quiesce_kills_a_real_start_script_and_its_initdb(tmp_path):
    """PG_QUIESCE run for real against a process whose command line is `bash …/start.d/10-postgres.sh` (sitting in a
    long chown -R, say) and a `…/bin/initdb` child — both are gone after it; the quiesce shell itself survives."""
    sd = tmp_path / "start.d"
    sd.mkdir()
    (sd / "10-postgres.sh").write_text("sleep 60\n")
    bd = tmp_path / "bin"
    bd.mkdir()
    (bd / "initdb").write_text("#!/bin/bash\nsleep 60\n")
    (bd / "initdb").chmod(0o755)
    p1 = subprocess.Popen(["bash", str(sd / "10-postgres.sh")])
    p2 = subprocess.Popen([str(bd / "initdb"), "-D", "x"])
    try:
        time.sleep(0.3)
        r = _bash(PG_QUIESCE)
        assert r["exit_code"] == 0, r
        assert p1.wait(timeout=10) is not None and p2.wait(timeout=10) is not None
    finally:
        for p in (p1, p2):
            if p.poll() is None:
                p.kill()


def test_quiesce_with_nothing_running_is_quick_and_ok():
    t = time.time()
    assert _bash(PG_QUIESCE)["exit_code"] == 0 and time.time() - t < 5


# ---------------------------------------------------------------- M2: the script checks the marker again
def _run_with_marker_mid_chown(tmp_path, cluster: bool):
    """The marker appears while the script sits in chown -R (a restore started meanwhile)."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    mark = tmp_path / ".ghost-restore-incomplete"
    for n in ("sudo", "chmod"):
        (fake / n).write_text(f'#!/bin/bash\necho "{n} $*" >> {log}\n')
        (fake / n).chmod(0o755)
    (fake / "chown").write_text(f'#!/bin/bash\necho "chown $*" >> {log}\ntouch {mark}\n')
    (fake / "chown").chmod(0o755)
    pgdata = tmp_path / "pg"
    pgdata.mkdir(exist_ok=True)
    if cluster:
        (pgdata / "PG_VERSION").write_text("18")
    env = {"PATH": f"{fake}:/usr/bin:/bin", "PGDATA": str(pgdata), "GHOST_RESTORE_MARKER": str(mark)}
    r = subprocess.run(["bash", str(ROOT / "images" / "pg" / "10-postgres.sh")], env=env, capture_output=True,
                       text=True, timeout=30)
    return r, (log.read_text() if log.exists() else "")


def test_pg_script_rechecks_the_marker_before_initdb(tmp_path):
    r, calls = _run_with_marker_mid_chown(tmp_path, cluster=False)
    assert r.returncode == 0 and "before initdb" in r.stderr and "initdb" not in calls


def test_pg_script_rechecks_the_marker_before_postgres(tmp_path):
    r, calls = _run_with_marker_mid_chown(tmp_path, cluster=True)
    assert r.returncode == 0 and "before postgres" in r.stderr and "/postgres -D" not in calls


def test_mysql_script_honours_the_marker(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    log = tmp_path / "calls.log"
    (fake / "docker-entrypoint.sh").write_text(f'#!/bin/bash\necho "entrypoint $*" >> {log}\n')
    (fake / "docker-entrypoint.sh").chmod(0o755)
    mark = tmp_path / ".ghost-restore-incomplete"
    env = {"PATH": f"{fake}:/usr/bin:/bin", "GHOST_RESTORE_MARKER": str(mark)}
    script = str(ROOT / "images" / "mysql" / "10-mysql.sh")
    mark.write_text("")
    r = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "mysqld NOT started" in r.stderr and not log.exists()
    mark.unlink()
    subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=30)
    assert "entrypoint mysqld" in log.read_text()


# ---------------------------------------------------------------- MINORs
def test_rollback_of_a_held_app_needs_restore_snapshot(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1", state="stopped", mounted=False)
    make_kept(env, "lab_v2", state="stopped", held=True, stopped_by_owner=True, switched_from="lab_v1")
    monkeypatch.setattr(kept, "_archive_for", lambda lab: (b"x", {}, [], None))
    monkeypatch.setattr(kept, "_move_data", lambda *a: pytest.fail("nothing moves"))
    with pytest.raises(LabError) as e:
        kept.rollback("lab_v2", restore_snapshot=False)
    assert e.value.code == "not_ready" and "restore_snapshot" in e.value.message


def test_switch_rechecks_the_hold_inside_the_locks(kept, env, monkeypatch):  # noqa: F811
    """A restore that failed (and held) while the switch waited for the locks stops it there."""
    import contextlib
    make_kept(env, "lab_a", state="up", data="shared")
    make_kept(env, "lab_b", state="staged", mounted=False, data="shared")
    monkeypatch.setattr(kept, "_archive_for", lambda lab: (b"x", {}, [], None))
    real = kept._app_locked

    @contextlib.contextmanager
    def locked(name, *a, **k):
        a_ = app_of(env, "lab_a")
        a_.update(held=True, hold_reason="restore of bk_x into m1 failed")
        kept._save("lab_a", a_)
        with real(name, *a, **k):
            yield
    monkeypatch.setattr(kept, "_app_locked", locked)
    monkeypatch.setattr(kept, "_switch_locked", lambda *a: pytest.fail("no switch on a held app"))
    with pytest.raises(LabError) as e:
        kept.switch("lab_b", "lab_a")
    assert "held" in e.value.message and e.value.extra["hold_reason"].startswith("restore of bk_x")


def test_a_held_app_rotates_no_backups(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", held=True, stopped_by_owner=True)
    for i in range(8):
        p = tmp_path / f"bk_p{i}.tar"
        p.write_bytes(TAR)
        env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", (f"bk_p{i}", "p", "chess", "m1", float(i + 1),
                                                                   str(p), "pre_deploy"))
    (tmp_path / "bk_new.tar").write_bytes(TAR)
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": "bk_new", "member": mem, "path": str(tmp_path / "bk_new.tar"), "bytes": 1})
    kept._backup_locked("lab_a", "pre_deploy")
    assert all((tmp_path / f"bk_p{i}.tar").exists() for i in range(8))


def test_the_hold_reason_is_its_own_field_and_survives_notes(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: False))
    with pytest.raises(LabError):
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    reason = a["hold_reason"]
    assert "bk_db" in reason and "docker refused" in reason and "bk_pre" in reason
    Kept._note(a, "x" * 5000)                             # any later note: last_error may be anything
    assert a["hold_reason"] == reason and kept._view(a)["hold_reason"] == reason
    assert kept._view(a)["restore_incomplete"] == ["db"]


def test_a_member_restored_while_another_is_incomplete_keeps_the_hold(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    a = app_of(env, "lab_a")
    a.update(held=True, stopped_by_owner=True, state="stopped", restore_incomplete=["app"],
             hold_reason="restore of bk_app into app failed")
    kept._save("lab_a", a)
    _bk(env, tmp_path)
    _restore_stubs(kept, env, monkeypatch, [])
    monkeypatch.setattr(kept, "_start_app", Kept._start_app.__get__(kept))
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: pytest.fail("no 90 s wait on an app that cannot start"))
    out = kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert a["held"] and a["restore_incomplete"] == ["app"] and a["hold_reason"] == "restore of bk_app into app failed"
    assert "still held" in a["last_error"] and out["probe"] == {"ok": None}


def test_a_failed_marker_touch_is_reported_untouched(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    cmds = []
    _restore_stubs(kept, env, monkeypatch, [])

    def must(lab, m, cmd, what, *a, **k):
        cmds.append(cmd)
        if cmd.startswith("touch"):
            raise LabError("internal", "write the restore marker failed (exit 1): read-only file system")
        return _ok()
    monkeypatch.setattr(kept, "_must", must)
    sh = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: sh.append(cmd) or _ok())
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["step"] == "write the marker"
    assert not any(c.startswith("find /data") for c in cmds) and not a.get("restore_incomplete")
    assert any(c.startswith("rm -f /data/.ghost") for c in sh) and a["state"] == "up"


def test_an_untouched_failure_drops_the_ids_it_protected(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    _restore_stubs(kept, env, monkeypatch, [])
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: {"exit_code": 1, "stdout": "", "stderr": "timeout"})
    with pytest.raises(LabError):
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert not a.get("protected_backups") and not a.get("held") and not a.get("hold_reason")


def test_reinit_keeps_the_good_copies_protected_for_a_week(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _held_incomplete(env, protected_backups=["bk_good"])
    monkeypatch.setattr(kept, "_must", lambda *a, **k: _ok())
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(
        exec_run=lambda *a, **k: SimpleNamespace(exit_code=0, output=b"")))
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept.app_action("lab_a", "start", reinit=True)
    a = app_of(env, "lab_a")
    assert not a.get("held") and a["protected_backups"] == ["bk_good"]
    assert a["protected_until"] > time.time() + 6 * 86400
    # a daily backup inside the week keeps them; the first one after it releases them
    (tmp_path / "bk_d.tar").write_bytes(TAR)
    monkeypatch.setattr(kept, "_copy_data", lambda lab, lin, app, mem, kind: {
        "backup_id": f"bk_d{time.time_ns()}", "member": mem, "path": str(tmp_path / "bk_d.tar"), "bytes": 1})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    kept._backup_locked("lab_a", "daily")
    assert app_of(env, "lab_a")["protected_backups"] == ["bk_good"]
    a = app_of(env, "lab_a")
    a["protected_until"] = time.time() - 1
    kept._save("lab_a", a)
    kept._backup_locked("lab_a", "daily")
    a = app_of(env, "lab_a")
    assert not a.get("protected_backups") and not a.get("protected_until")


def test_reinit_wording_is_what_happens(kept, env):  # noqa: F811
    _held_incomplete(env)
    with pytest.raises(LabError) as e:
        kept.app_action("lab_a", "start")
    assert "whatever /data holds now" in e.value.hint and "only if the restore never got the files in" in e.value.hint


def test_a_held_deploy_keeps_its_build_log_and_says_there_is_no_revert(kept, env, monkeypatch):  # noqa: F811
    _held_incomplete(env, last_backup_at=None)
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": None, "build_log": "pip: installed flask"})
    monkeypatch.setattr(kept, "_save_built", lambda *a: True)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert out["probe"]["build_log"] == "pip: installed flask" and "No automatic revert" in out["held"]


# ---------------------------------------------------------------- round 7
def test_a_postgres_back_before_the_clear_fails_the_restore_untouched(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    sh = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: sh.append(cmd) or _ok())

    def must(lab, m, cmd, what, *a, allow=(), **k):
        order.append(cmd.split()[0])
        if "find /data" in cmd:
            assert cmd.startswith("pgrep -x postgres") and 3 in allow
            return {"exit_code": 3, "stdout": "", "stderr": ""}     # postgres came back: the shell cleared nothing
        return _ok()
    monkeypatch.setattr(kept, "_must", must)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert e.value.extra["touched"] is False and e.value.extra["step"] == "check postgres is stopped"
    assert "running again" in e.value.message and "put" not in order
    assert any(c.startswith("rm -f /data/.ghost") for c in sh)       # the marker goes again
    assert not a.get("restore_incomplete") and not a.get("held") and a["state"] == "up"


def test_a_good_restore_is_recorded_complete_before_a_later_error(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    _restore_stubs(kept, env, monkeypatch, [])

    stored = []

    def watch(*a, **k):
        stored.append(app_of(env, "lab_a").get("restore_incomplete"))    # what the DB says at this moment
        raise RuntimeError("docker went away")
    monkeypatch.setattr(kept, "_start_watch", watch)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    a = app_of(env, "lab_a")
    assert stored == [None]                     # saved complete BEFORE the next thing that can fail
    assert e.value.extra["step"] == "finish" and not a.get("restore_incomplete") and not a.get("watch_held")


def test_an_old_script_on_the_host_fails_the_restore_untouched(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    old = tmp_path / "10-postgres.sh"
    old.write_text("#!/bin/bash\ninitdb -D \"$PGDATA\"\nexec postgres\n")
    monkeypatch.setattr(K, "PG_SCRIPT", old)
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_db")
    assert "old 10-postgres.sh" in e.value.message and "redeploy images/" in e.value.message
    assert e.value.extra["touched"] is False and "install" not in order and "mark" not in order
    assert app_of(env, "lab_a")["state"] == "up"


def test_a_held_deploy_takes_no_pre_deploy_copy(kept, env, monkeypatch):  # noqa: F811
    _held_incomplete(env, last_backup_at=1.0)
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("no copy of a held app's /data"))
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": None})
    monkeypatch.setattr(kept, "_save_built", lambda *a: True)
    monkeypatch.setattr(kept, "_sh", lambda *a, **k: _ok())
    out = kept.deploy("lab_a", {"archive_b64": "TkVX"})
    assert "no pre-deploy copy" in out["pre_deploy_note"] and "pre_deploy_backup" not in out


def test_a_restore_retry_while_held_reuses_the_first_pre_restore_copy(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _pg_single(env)
    _bk(env, tmp_path)
    order = []
    _restore_stubs(kept, env, monkeypatch, order)
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: False))
    with pytest.raises(LabError):
        kept._restore_locked("lab_a", "bk_db")                     # the first attempt: snapshot bk_pre, held
    assert app_of(env, "lab_a")["hold_pre"] == {"db": "bk_pre"}
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("no copy of the half-restored /data"))
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    out = kept._restore_locked("lab_a", "bk_db")
    assert out["pre_restore"] == "bk_pre"
