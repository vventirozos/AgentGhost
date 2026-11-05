"""Pins for the kept-layer review r1 fixes (K1–K11 + minors). Docker is faked; the
kept steps that need a real container are monkeypatched to record what they were asked."""
import io
import tarfile
import time
from types import SimpleNamespace

import pytest

from ghost_labs.errors import LabError
from ghost_labs.kept import Kept, _docker_ts, valid_workdir
from ghost_labs.store import dumps, loads

from .test_core import add_lab, env  # noqa: F401


@pytest.fixture
def kept(env, tmp_path):  # noqa: F811
    k = Kept(env)
    k.archive_root = tmp_path / "archives"
    k.backup_root = tmp_path / "backups"
    env.kept = k
    return k


def make_kept(m, lab_id, lineage="p", name="chess", member="m1", mounted=True, state="up", **extra):
    add_lab(m, lab_id, lifetime="kept", owner=f"project:{lineage}")
    app = {"name": name, "lineage": lineage, "member": member, "cmd": "python3 app.py", "port": 8080,
           "health": {"http": "/"}, "workdir": "/app", "data": "shared", "secrets": [], "cron": [], "backup": None,
           "lan": False, "state": state, "mounted": mounted, "restarts": 0, "fails": 0, "crash_times": [],
           "last_probe_at": time.time(), "last_error": None, "started_at": time.time(), "next_restart_at": None,
           "data_seed": [], "lease_key": f"{lineage}/{name}", "switched_from": None, "last_backup_at": time.time(),
           "last_backup_attempt_at": 0, **extra}
    m.docker.containers.run("img", name=m.container_name(lab_id, member), labels={"ghost.labs": "1", "ghost.lab.id": lab_id})
    m.store.update_lab(lab_id, lineage=lineage, app_json=dumps(app),
                       spec_json=dumps({"members": [{"name": member, "image": "ghost-lab/base:1", "mem_mb": 512,
                                                     "cpus": 1.0, "cap_add": []}], "egress": "none", "readiness": None}))
    return app


def app_of(m, lab_id):
    return loads(m.store.lab(lab_id)["app_json"])


# ---------------------------------------------------------------- K8 workdir
@pytest.mark.parametrize("wd", ["/app", "/srv/chess", "/opt/chess", "/home/u/app", "/root/app"])
def test_workdir_ok(wd):
    assert valid_workdir(wd)


@pytest.mark.parametrize("wd", ["/", "app", "/data", "/data/app", "/etc", "/usr/local/app", "/var/www", "/opt",
                                "/opt/venv", "/opt/venv/x", "/root", "/home", "/app/../etc", "/app/./x", "/app/",
                                "/app dir", "/tmp/x", "/run/ghost", "/proc/1"])
def test_workdir_refused(wd):
    assert not valid_workdir(wd)


def test_keep_refuses_bad_workdir(kept, env):  # noqa: F811
    add_lab(env, "lab_w", owner="project:p")
    with pytest.raises(LabError) as e:
        kept.keep("lab_w", {"app": {"name": "a", "member": "m1", "port": 80, "workdir": "/data"}})
    assert e.value.extra["field"] == "app.workdir"


# ---------------------------------------------------------------- K3 second version
def _stub_keep_steps(kept, monkeypatch):
    calls = {"lease": [], "recreate": []}
    monkeypatch.setattr(kept, "_recreate", lambda lab, m, mount: calls["recreate"].append((lab, m, mount)))
    monkeypatch.setattr(kept, "_lease", lambda lab, app, move: calls["lease"].append((lab, move)))
    monkeypatch.setattr(kept, "_has_code", lambda *a: False)
    monkeypatch.setattr(kept, "_install_cron", lambda *a: None)
    return calls


def test_second_version_fresh_refused(kept, env, monkeypatch):  # noqa: F811
    _stub_keep_steps(kept, monkeypatch)
    make_kept(env, "lab_live")
    add_lab(env, "lab_v2", owner="project:p")
    with pytest.raises(LabError) as e:
        kept.keep("lab_v2", {"app": {"name": "chess", "member": "m1", "port": 8080}, "data": "fresh"})
    assert e.value.extra["field"] == "data" and e.value.extra["live_lab"] == "lab_live"


def test_second_version_other_member_refused(kept, env, monkeypatch):  # noqa: F811
    _stub_keep_steps(kept, monkeypatch)
    make_kept(env, "lab_live", member="web")
    add_lab(env, "lab_v2", owner="project:p")
    with pytest.raises(LabError) as e:
        kept.keep("lab_v2", {"app": {"name": "chess", "member": "m1", "port": 8080}})
    assert e.value.extra["field"] == "app.member"


def test_second_version_staged_even_without_volume(kept, env, monkeypatch):  # noqa: F811
    calls = _stub_keep_steps(kept, monkeypatch)
    make_kept(env, "lab_live")
    add_lab(env, "lab_v2", owner="project:p")
    kept.keep("lab_v2", {"app": {"name": "chess", "member": "m1", "port": 8080}})
    a = app_of(env, "lab_v2")
    assert a["state"] == "staged" and a["mounted"] is False
    assert calls["lease"] == [("lab_v2", False)]
    assert all(mount is False for _, _, mount in calls["recreate"])


def test_first_version_mounted(kept, env, monkeypatch):  # noqa: F811
    calls = _stub_keep_steps(kept, monkeypatch)
    add_lab(env, "lab_v1", owner="project:p")
    kept.keep("lab_v1", {"app": {"name": "chess", "member": "m1", "port": 8080}, "data": "fresh"})
    assert app_of(env, "lab_v1")["mounted"] is True and calls["lease"] == [("lab_v1", True)]


def test_lease_never_taken_from_a_live_lab(kept, env):  # noqa: F811
    make_kept(env, "lab_live")
    env.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20001, "lab_live", "m1", 8080, "p/chess", 0))
    app = make_kept(env, "lab_v2", mounted=False, state="staged")
    with pytest.raises(LabError) as e:
        kept._lease("lab_v2", app, move=True)
    assert e.value.code == "busy" and e.value.extra["live_lab"] == "lab_live"
    assert env.store.q("SELECT lab_id FROM ports WHERE port=20001")[0]["lab_id"] == "lab_live"


# ---------------------------------------------------------------- K4 start_all
def test_start_all_continues_after_a_failing_lab(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", name="a")
    make_kept(env, "lab_b", name="b")
    seen = []

    def boom(lab_id, member):
        seen.append(lab_id)
        if lab_id == "lab_a":
            raise RuntimeError("docker hiccup")
    monkeypatch.setattr(kept, "on_member_start", boom)
    kept.start_all("test")
    assert sorted(seen) == ["lab_a", "lab_b"]


# ---------------------------------------------------------------- on_member_start re-check
def test_on_member_start_rechecks_under_the_lock(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a")
    started = []
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: started.append(1))
    monkeypatch.setattr(kept, "_has_code", lambda *a: True)
    monkeypatch.setattr(kept, "_alive", lambda *a: False)
    real = env.locked

    def locked(lab_id, timeout=30):
        # the state changes while the caller waits for the lock
        a = app_of(env, lab_id)
        a.update(state="staged", mounted=False)
        env.store.update_lab(lab_id, app_json=dumps(a))
        return real(lab_id, timeout)
    monkeypatch.setattr(env, "locked", locked)
    kept.on_member_start("lab_a", "m1")
    assert started == []


# ---------------------------------------------------------------- container restart is not a crash
def test_docker_ts():
    assert _docker_ts("2026-11-04T10:00:00.5Z") == pytest.approx(1793786400.5, abs=1e-3)
    assert _docker_ts("2026-11-04T10:00:00.123456789Z") == pytest.approx(1793786400.123456, abs=1e-3)
    assert _docker_ts("2026-11-04T10:00:00Z") == pytest.approx(1793786400, abs=1e-3)


def test_exited_after_container_restart_restarts_without_counting(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", started_at=1000.0)
    c = SimpleNamespace(attrs={"State": {"StartedAt": "2026-01-01T10:00:00Z"}})
    monkeypatch.setattr(env, "container", lambda lab, m: c)
    monkeypatch.setattr(env, "apply_route", lambda *a: None)
    got = []
    monkeypatch.setattr(kept, "_start_app", lambda lab, a, count: got.append(count))
    monkeypatch.setattr(kept, "_record_crash", lambda *a: got.append("crash"))
    kept._exited("lab_a", app, "app exited")
    assert got == [False]
    app["started_at"] = time.time()
    kept._exited("lab_a", app, "app exited")
    assert got == [False, "crash"]


# ---------------------------------------------------------------- K5 archives
def test_archive_prune_keeps_archives_in_use(kept, env):  # noqa: F811
    first = kept._keep_archive("p", "chess", b"v1", {}, [])
    make_kept(env, "lab_old", archive_ts=first, mounted=False, state="staged")
    for i in range(7):                       # same millisecond on purpose: names must still be unique
        kept._keep_archive("p", "chess", f"v{i + 2}".encode(), {}, [])
    ts = {r["ts"] for r in env.store.q("SELECT ts FROM archives WHERE app='chess'")}
    assert first in ts and len(ts) == 6
    assert kept._archive_for("lab_old")[0] == b"v1"


# ---------------------------------------------------------------- K6 backups
class _Stream:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = chunks, fail_after

    def __iter__(self):
        for i, ch in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise OSError("stream broke")
            yield ch


def test_failed_backup_leaves_no_partial_file_or_row(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="stopped")
    c = SimpleNamespace(get_archive=lambda p: (_Stream([b"a" * 10, b"b" * 10], fail_after=1), {}))
    monkeypatch.setattr(env, "container", lambda lab, m: c)
    with pytest.raises(OSError):
        kept._backup_locked("lab_a", "manual")
    assert list((kept.backup_root / "p" / "chess").glob("*")) == []
    assert env.store.q("SELECT * FROM backups") == []


def test_daily_backup_retried_after_an_hour_not_every_tick(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", last_backup_at=0, last_probe_at=time.time())
    monkeypatch.setattr(kept, "_alive", lambda *a: True)
    tries = []

    def fail(lab_id, kind, **kw):
        tries.append(kind)
        raise OSError("disk")
    monkeypatch.setattr(kept, "_backup_locked", fail)
    kept._last_daily = time.time()
    kept.tick()
    kept.tick()
    assert tries == ["daily"]
    a = app_of(env, "lab_a")
    a["last_backup_attempt_at"] = time.time() - 3700
    env.store.update_lab("lab_a", app_json=dumps(a))
    kept.tick()
    assert tries == ["daily", "daily"]


# ---------------------------------------------------------------- K9 final backup covers every member
def test_final_backup_covers_every_member(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="stopped")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18", "mem_mb": 256,
                          "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    asked = []

    def cont(lab, m):
        asked.append(m)
        return SimpleNamespace(get_archive=lambda p: (_Stream([m.encode()]), {}))
    monkeypatch.setattr(env, "container", cont)
    out = kept._backup_locked("lab_a", "final", all_members=True)
    assert asked == ["m1", "db"] and [x["member"] for x in out["members"]] == ["m1", "db"]
    assert {r["member"] for r in env.store.q("SELECT member FROM backups WHERE kind='final'")} == {"m1", "db"}
    asked.clear()
    kept._backup_locked("lab_a", "manual")
    assert asked == ["m1"]


def test_drop_data_final_backup_covers_every_member(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a")
    asked = []
    monkeypatch.setattr(kept, "_backup_locked", lambda lab, kind, all_members=False, restart=True, **k:
                        asked.append((kind, all_members, restart)) or {"backup_id": "bk_f", "members": []})
    monkeypatch.setattr(env, "destroy", lambda lab, **k: {"lab_id": lab, "destroyed": True})
    out = kept.destroy_with_data("lab_a", "owner")
    assert asked == [("final", True, False)] and out["final_backup"] == "bk_f"


# ---------------------------------------------------------------- K10 restore
def test_restore_refuses_unmounted(kept, env):  # noqa: F811
    make_kept(env, "lab_a", mounted=False, state="staged")
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_x")
    assert e.value.code == "not_ready"


def test_restore_failure_snapshots_first_and_marks_stopped(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    bk = tmp_path / "bk.tar"
    bk.write_bytes(b"tar")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_1", "p", "chess", "m1", time.time(), str(bk), "manual"))
    monkeypatch.setattr(kept, "_backup_locked", lambda lab, kind, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    sent = []
    c = SimpleNamespace(put_archive=lambda path, f: sent.append(f.read()) or False)
    monkeypatch.setattr(env, "container", lambda lab, m: c)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    with pytest.raises(LabError) as e:
        kept._restore_locked("lab_a", "bk_1")
    assert e.value.extra["pre_restore"] == "bk_pre"
    a = app_of(env, "lab_a")
    assert sent == [b"tar"]
    assert a["state"] == "stopped" and "bk_pre" in a["last_error"] and "failed" in a["last_error"]


# ---------------------------------------------------------------- K11 bounded member read
def _tar_of(name, data):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_read_raw_bounds(env, monkeypatch):  # noqa: F811
    add_lab(env, "lab_a")
    raw = _tar_of("x.tgz", b"z" * 100)
    c = SimpleNamespace(get_archive=lambda p: ([raw], {"mode": 0o644, "size": 100}))
    monkeypatch.setattr(env, "container", lambda lab, m: c)
    assert env.read_raw("lab_a", "m1", "/x.tgz", 1000, "cap") == b"z" * 100
    with pytest.raises(LabError):
        env.read_raw("lab_a", "m1", "/x.tgz", 50, "cap")
    c2 = SimpleNamespace(get_archive=lambda p: ([b"x" * 4096] * 100, {"mode": 0o644, "size": 1}))   # lying size
    monkeypatch.setattr(env, "container", lambda lab, m: c2)
    with pytest.raises(LabError):
        env.read_raw("lab_a", "m1", "/x.tgz", 1000, "cap")
    c3 = SimpleNamespace(get_archive=lambda p: ([raw], {"mode": 1 << 31, "size": 0}))
    monkeypatch.setattr(env, "container", lambda lab, m: c3)
    with pytest.raises(LabError):
        env.read_raw("lab_a", "m1", "/dir", 1000, "cap")


def test_deploy_rejects_bad_base64(kept, env):  # noqa: F811
    make_kept(env, "lab_a")
    with pytest.raises(LabError) as e:
        kept.deploy("lab_a", {"archive_b64": "aGVsbG8=!!"})      # lenient decoding would accept this
    assert e.value.extra["field"] == "archive_b64"


# ---------------------------------------------------------------- K7 seeds never replace data
def test_seed_never_removes_existing_data_and_marker_is_read_back(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", data_seed=[], state="stopped")
    cmds, puts = [], []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {"exit_code": 0, "stdout": "saves\n", "stderr": ""})
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or {"exit_code": 0})
    monkeypatch.setattr(env, "_put_bytes", lambda c, path, data, mode: puts.append(path))
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_install_cron", lambda *a: None)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept._deploy_into("lab_a", app, b"tgz", {}, [])
    seed_cmds = [c for c in cmds if "/data/saves" in c]
    assert seed_cmds and not any("rm -rf /data" in c for c in seed_cmds)
    assert any("[ ! -e /data/saves ]" in c for c in seed_cmds)
    assert app["data_seed"] == ["saves"]                    # known from the marker on /data
    assert "/data/.ghost-seeded" not in puts                 # marker already lists it
    with pytest.raises(LabError) as e:
        kept._deploy_into("lab_a", app, b"tgz", {}, ["other"])
    assert e.value.code == "data_exists"


# ---------------------------------------------------------------- K1/K2 switch, rollback, rehearse
def _two_versions(env, kept, monkeypatch, tmp_path):
    make_kept(env, "lab_v1", archive_ts=1.0)
    make_kept(env, "lab_v2", mounted=False, state="staged", archive_ts=2.0)
    env.store.update_lab("lab_v2", retire_at=time.time() + 50)
    for ts, name in ((1.0, "v1"), (2.0, "v2")):
        p = tmp_path / f"{name}.tgz"
        p.write_bytes(name.encode())
        env.store.x("INSERT INTO archives VALUES (?,?,?,?,?,?)", ("p", "chess", ts, str(p), "{}", "[]"))
    monkeypatch.setattr(kept, "_backup_locked", lambda lab, kind, **k: {"backup_id": "bk_snap",
                                                                        "members": [{"backup_id": "bk_snap"}]})
    monkeypatch.setattr(kept, "_lease", lambda *a, **k: None)


def test_switch_success_clears_retire_on_new(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)

    def move(to, fr, archive):
        a = app_of(env, to)
        a.update(mounted=True, state="up")
        env.store.update_lab(to, app_json=dumps(a))
        return {"ok": True}, a
    monkeypatch.setattr(kept, "_move_data", move)
    out = kept.switch("lab_v2", "lab_v1")
    assert out["switched"] is True
    assert env.store.lab("lab_v2")["retire_at"] is None and env.store.lab("lab_v1")["retire_at"]
    assert env.store.lab("lab_v2")["state"] == "ready"


def test_switch_raise_reverses_and_resets_state(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    moves = []

    def move(to, fr, archive):
        moves.append((to, archive[0]))
        if to == "lab_v2":
            raise LabError("internal", "pip install failed")
        a = app_of(env, to)
        a.update(mounted=True, state="up")
        env.store.update_lab(to, app_json=dumps(a))
        return {"ok": True}, a
    monkeypatch.setattr(kept, "_move_data", move)
    out = kept.switch("lab_v2", "lab_v1")
    assert out["switched"] is False and out["reversed"] is True and "pip" in out["error"]
    assert moves == [("lab_v2", b"v2"), ("lab_v1", b"v1")]
    assert env.store.lab("lab_v2")["state"] == "ready"
    assert env.store.lab("lab_v1")["retire_at"] is None


def test_switch_refuses_when_live_has_no_archive(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    env.store.x("DELETE FROM archives WHERE ts=1.0")
    with pytest.raises(LabError) as e:
        kept.switch("lab_v2", "lab_v1")
    assert e.value.code == "not_ready"


def test_rollback_prechecks(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_gone")
    env.store.update_lab("lab_v2", app_json=dumps(a))
    with pytest.raises(LabError) as e:
        kept.rollback("lab_v2", restore_snapshot=True)
    assert e.value.code == "unknown_backup"
    env.store.x("DELETE FROM archives WHERE ts=1.0")
    with pytest.raises(LabError) as e:
        kept.rollback("lab_v2", restore_snapshot=False)
    assert e.value.code == "no_rollback"


def test_rollback_failure_puts_new_back(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1")
    env.store.update_lab("lab_v2", app_json=dumps(a))
    moves = []

    def move(to, fr, archive):
        moves.append(to)
        if to == "lab_v1":
            return {"ok": False, "error": "probe"}, app_of(env, to)
        b = app_of(env, to)
        b.update(mounted=True, state="up")
        env.store.update_lab(to, app_json=dumps(b))
        return {"ok": True}, b
    monkeypatch.setattr(kept, "_move_data", move)
    out = kept.rollback("lab_v2", restore_snapshot=False)
    assert out["rolled_back"] is False and out["reversed"] is True and moves == ["lab_v1", "lab_v2"]
    assert app_of(env, "lab_v2")["switched_from"] == "lab_v1"
    assert env.store.lab("lab_v2")["state"] == "ready"


def test_rehearse_failure_records_error_and_resets_state(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_recreate", lambda *a, **k: None)

    def bad(*a, **k):
        raise LabError("internal", "apt failed")
    monkeypatch.setattr(kept, "_deploy_into", bad)
    with pytest.raises(LabError):
        kept.rehearse("lab_v1")
    a = app_of(env, "lab_v1")
    assert env.store.lab("lab_v1")["state"] == "ready"
    assert a["state"] == "stopped" and "apt failed" in a["last_error"] and "bk_snap" in a["last_error"]


# ---------------------------------------------------------------- twin
def test_twin_gets_fresh_secrets_and_no_cron(kept, env, monkeypatch):  # noqa: F811
    env.store.set_secret("p", "chess", "APP_KEY", "real-secret")
    app = make_kept(env, "lab_a", secrets=["APP_KEY"], cron=["* * * * * root echo hi"])
    add_lab(env, "lab_t")
    monkeypatch.setattr(env, "create", lambda body, origin, pool: {"lab_id": "lab_t"})
    destroyed = []
    monkeypatch.setattr(env, "destroy", lambda lab, **k: destroyed.append(lab))
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    seen = {}

    def dep(tid, tapp, *a, **k):
        seen["env"] = kept._env(tid, tapp)
        seen["cron"] = tapp["cron"]
        return {"ok": None}
    monkeypatch.setattr(kept, "_deploy_into", dep)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept._twin(env.store.lab("lab_a"), loads(env.store.lab("lab_a")["spec_json"]), app, b"x", {})
    assert seen["cron"] == [] and seen["env"]["APP_KEY"] != "real-secret" and len(seen["env"]["APP_KEY"]) > 20
    assert env.store.q("SELECT * FROM secrets WHERE lineage LIKE 'twin-%'") == []
    assert destroyed == ["lab_t"]


# ---------------------------------------------------------------- sidecar rule
def test_sidecar_rule_dropped_only_when_unused(kept, env):  # noqa: F811
    add_lab(env, "lab_a")
    add_lab(env, "lab_b")
    url = "http://100.90.234.23:8555"
    env.store.update_lab("lab_a", agent_api_url=url)
    env.store.update_lab("lab_b", agent_api_url=url)
    n0 = len(env.run.calls)
    kept.drop_sidecar_rule("lab_a")
    assert len(env.run.calls) == n0                           # lab_b still uses it
    env.store.update_lab("lab_b", agent_api_url=None)
    kept.drop_sidecar_rule("lab_a")
    assert any(c[:4] == ["iptables", "-t", "raw", "-C"] for c in env.run.calls[n0:])


# ================================================================ review r2 pins
from ghost_labs.kept import MEMBER_DOWN, check_seeds  # noqa: E402


@pytest.mark.parametrize("bad", [["saves", "saves/slot1"], ["a//b"], ["./a"], ["a/"], ["a/./b"], ["../x"],
                                 [".ghost-seeded"], ["/abs"]])
def test_seed_lists_refused(bad):
    with pytest.raises(LabError):
        check_seeds(bad)


def test_seed_lists_ok():
    assert check_seeds(["b", "a", "a"]) == ["a", "b"]
    assert check_seeds(["saves", "savesx/y"]) == ["saves", "savesx/y"]


def _deploy_stubs(kept, env, monkeypatch, marker=""):
    cmds = []
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {
        "exit_code": 0 if (marker or "ghost-seeded" not in cmd) else 1,
        "stdout": marker if "ghost-seeded" in cmd else "", "stderr": ""})
    monkeypatch.setattr(kept, "_must", lambda lab, m, cmd, what, *a, **k: cmds.append(cmd) or {"exit_code": 0})
    monkeypatch.setattr(env, "_put_bytes", lambda *a: None)
    monkeypatch.setattr(env, "container", lambda lab, m: None)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_install_cron", lambda *a: None)
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    return cmds


def test_link_refuses_a_parent_outside_the_workdir(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="stopped")
    cmds = _deploy_stubs(kept, env, monkeypatch)
    kept._deploy_into("lab_a", app, b"t", {}, ["saves"])
    link = [c for c in cmds if "ln -s" in c][0]
    assert "readlink -f" in link and "exit 9" in link and link.index("exit 9") < link.index("rm -rf")
    assert link.index("exit 9") < link.index("mv ")                                  # guard before the seed move


def test_archive_for_returns_seed_list(kept, env, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", archive_ts=5.0)
    p = tmp_path / "a.tgz"
    p.write_bytes(b"x")
    env.store.x("INSERT INTO archives VALUES (?,?,?,?,?,?)", ("p", "chess", 5.0, str(p), "{}", '["saves"]'))
    assert kept._archive_for("lab_a")[2] == ["saves"]


def test_deploy_failure_after_swap_marks_stopped(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up")
    _deploy_stubs(kept, env, monkeypatch)

    def must(lab, m, cmd, what, *a, **k):
        if what == "swap":
            raise LabError("internal", "mv failed")
        return {"exit_code": 0}
    monkeypatch.setattr(kept, "_must", must)
    with pytest.raises(LabError):
        kept._deploy_into("lab_a", app, b"t", {}, [])
    a = app_of(env, "lab_a")
    assert a["state"] == "stopped" and "after the swap" in a["last_error"]


def test_destroy_unmounted_version_keeps_volumes_and_takes_no_backup(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v2", mounted=False, state="staged")
    env.docker.volumes.create("ghost-labdata-p-chess-m1", labels={"ghost.labs": "1"})
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: pytest.fail("no backup of an unmounted version"))
    out = kept.destroy_with_data("lab_v2", "owner")
    assert out["data_volumes_removed"] == [] and "ghost-labdata-p-chess-m1" in env.docker._volumes and out["note"]


def test_destroy_mounted_removes_volumes_inside_one_lock(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a")
    env.docker.volumes.create("ghost-labdata-p-chess-m1", labels={"ghost.labs": "1"})
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_f", "members": []})
    out = kept.destroy_with_data("lab_a", "owner")
    assert out["data_volumes_removed"] == ["ghost-labdata-p-chess-m1"] and not env.store.lab("lab_a")


def test_switch_refused_from_a_fresh_live_version(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a["data"] = "fresh"
    env.store.update_lab("lab_v1", app_json=dumps(a))
    with pytest.raises(LabError) as e:
        kept.switch("lab_v2", "lab_v1")
    assert e.value.extra["field"] == "from_lab"


def test_second_version_refused_when_live_is_fresh(kept, env, monkeypatch):  # noqa: F811
    _stub_keep_steps(kept, monkeypatch)
    make_kept(env, "lab_live", data="fresh")
    add_lab(env, "lab_v2", owner="project:p")
    with pytest.raises(LabError) as e:
        kept.keep("lab_v2", {"app": {"name": "chess", "member": "m1", "port": 8080}})
    assert e.value.extra["field"] == "data"


def test_app_name_unique_across_projects(kept, env, monkeypatch):  # noqa: F811
    _stub_keep_steps(kept, monkeypatch)
    make_kept(env, "lab_p1", lineage="p1")
    add_lab(env, "lab_p2", owner="project:p2")
    with pytest.raises(LabError) as e:
        kept.keep("lab_p2", {"app": {"name": "chess", "member": "m1", "port": 8080}})
    assert e.value.extra["field"] == "app.name"


def test_app_lock_serialises(kept):
    import threading as th
    held, release = th.Event(), th.Event()

    def holder():
        with kept._app_locked("chess"):
            held.set()
            release.wait(5)
    t = th.Thread(target=holder)
    t.start()
    held.wait(5)
    with pytest.raises(LabError) as e:
        with kept._app_locked("chess", timeout=0.1):
            pass
    assert e.value.code == "busy"
    with kept._app_locked("other", timeout=0.1):
        pass
    release.set()
    t.join()


def test_keep_switch_rollback_destroy_take_the_app_lock(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    names = []
    real = kept._app_locked

    def spy(name, timeout=120.0):
        names.append(name)
        return real(name, timeout)
    monkeypatch.setattr(kept, "_app_locked", spy)
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, a: ({"ok": False}, app_of(env, to)))
    monkeypatch.setattr(kept, "_reverse", lambda *a: {"ok": True})
    kept.switch("lab_v2", "lab_v1")
    a = app_of(env, "lab_v1")
    a["switched_from"] = "lab_v2"
    env.store.update_lab("lab_v1", app_json=dumps(a))
    kept.rollback("lab_v1", restore_snapshot=False)
    with pytest.raises(LabError):
        kept.keep("lab_v1", {"app": {"name": "chess2", "member": "m1", "port": 1}})   # already kept
    kept.destroy_with_data("lab_v2", "owner")
    assert names == ["chess", "chess", "chess2", "chess"]


def test_move_marker_set_then_cleared(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    seen = []
    monkeypatch.setattr(kept, "_recreate", lambda lab, m, mount: seen.append(app_of(env, "lab_v1").get("moving_to")))
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: {"ok": True})
    kept._move_data("lab_v2", "lab_v1", (b"v2", {}, []))
    assert seen[0] == "lab_v2" and app_of(env, "lab_v1")["moving_to"] == "lab_v2"   # until the lease moved
    kept._clear_marks("lab_v1", "lab_v2")
    assert "moving_to" not in app_of(env, "lab_v1")
    assert app_of(env, "lab_v2")["mounted"] and not app_of(env, "lab_v1")["mounted"]


def test_start_all_runs_recovery_first(kept, monkeypatch):
    order = []
    monkeypatch.setattr(kept, "recover_moves", lambda: order.append("recover") or [])
    kept.start_all("t")
    assert order == ["recover"]


def test_reconcile_resets_interrupted_switch_state(env):  # noqa: F811
    add_lab(env, "lab_s", lifetime="kept", state="switching")
    env.docker.containers.run("img", name="c", labels={"ghost.labs": "1", "ghost.lab.id": "lab_s", "ghost.lab.member": "m1"})
    env.reconcile()
    assert env.store.lab("lab_s")["state"] == "ready"


def test_rollback_restore_failure_undoes_the_restore_then_reverses(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    bk = tmp_path / "s.tar"
    bk.write_bytes(b"s")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_snap", "p", "chess", "m1", 1.0, str(bk), "pre_switch"))
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_snap", switch_snapshots=["bk_snap"])
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))
    restores = []

    def restore(lab, bid, snapshot=True, start=True):
        restores.append((lab, bid, snapshot))
        if bid == "bk_snap":
            raise LabError("internal", "put_archive failed", pre_restore="bk_pre")
        return {"pre_restore": None, "probe": {"ok": None}}
    monkeypatch.setattr(kept, "_restore_locked", restore)
    rev = []
    monkeypatch.setattr(kept, "_reverse", lambda live, other, arch, why: rev.append(live) or {"ok": True})
    out = kept.rollback("lab_v2", restore_snapshot=True)
    assert restores == [("lab_v1", "bk_snap", True), ("lab_v1", "bk_pre", False)]
    assert rev == ["lab_v2"] and out["reversed"] and "undone" in out["data_note"]
    assert "undone" in app_of(env, "lab_v2")["last_error"]


def test_snapshot_ids_put_the_app_member_last():
    assert Kept._snapshot_ids(None, {"switch_snapshot": "a", "switch_snapshots": ["a", "b", "c"]}) == ["b", "c", "a"]
    assert Kept._snapshot_ids(None, {"switch_snapshot": "a"}) == ["a"]


def test_switch_stores_every_member_snapshot(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    kinds = []
    monkeypatch.setattr(kept, "_backup_locked", lambda lab, kind, **k: kinds.append(k.get("all_members")) or
                        {"backup_id": "bk_a", "members": [{"backup_id": "bk_a"}, {"backup_id": "bk_db"}]})
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, a: ({"ok": True}, app_of(env, to)))
    kept.switch("lab_v2", "lab_v1")
    assert kinds == [True] and app_of(env, "lab_v2")["switch_snapshots"] == ["bk_a", "bk_db"]


def test_restore_snapshots_the_restored_member(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="stopped")
    env.store.put_member({"lab_id": "lab_a", "name": "db", "ip": "10.89.9.11", "image": "ghost-lab/pg:18", "mem_mb": 256,
                          "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    bk = tmp_path / "db.tar"
    bk.write_bytes(b"t")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_db", "p", "chess", "db", 1.0, str(bk), "final"))
    asked = []
    monkeypatch.setattr(kept, "_backup_locked", lambda lab, kind, **k: asked.append(k.get("only_member")) or {"backup_id": "x"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    kept._restore_locked("lab_a", "bk_db", start=False)
    assert asked == ["db"]


def test_tick_with_member_container_stopped(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    c = env.docker._containers[env.container_name("lab_a", "m1")]
    c.status = "exited"
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: pytest.fail("no start while the member is down"))
    kept._last_daily = time.time()
    kept.tick()
    a = app_of(env, "lab_a")
    assert a["state"] == "stopped" and a["last_error"] == MEMBER_DOWN
    c.status = "running"
    started = []
    monkeypatch.setattr(kept, "_start_app", lambda lab, app, count: started.append(count))
    monkeypatch.setattr(kept, "_has_code", lambda *a: True)
    monkeypatch.setattr(env, "apply_route", lambda *a: None)
    kept.tick()
    assert started == [False] and app_of(env, "lab_a")["last_error"] is None


def test_destroy_with_caller_holding_the_lock(env):  # noqa: F811
    add_lab(env, "lab_x")
    with env.locked("lab_x"):
        assert env.destroy("lab_x", _have_lock=True)["destroyed"]
    assert not env.store.lab("lab_x")


def test_tick_leaves_containers_missing_alone(kept, env):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", last_error="containers_missing")
    env.docker._containers.pop(env.container_name("lab_a", "m1"))
    kept._last_daily = time.time()
    kept.tick()
    assert app_of(env, "lab_a")["last_error"] == "containers_missing"


def test_backup_without_restart_leaves_app_stopped(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(get_archive=lambda p: (_Stream([b"x"]), {})))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    started = []
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: started.append(1))
    kept._backup_locked("lab_a", "final", restart=False)
    assert started == []
    kept._backup_locked("lab_a", "manual")
    assert started == [1]



# ================================================================ review r3 pins
def test_failed_reverse_leaves_no_marker(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)

    def move(to, fr, archive):
        a = app_of(env, fr)
        a.update(mounted=False, state="staged", moving_to=to)
        env.store.update_lab(fr, app_json=dumps(a))
        raise LabError("internal", "pip over tor failed")
    monkeypatch.setattr(kept, "_move_data", move)
    out = kept.switch("lab_v2", "lab_v1")
    assert out["reversed"] is False
    assert "moving_to" not in app_of(env, "lab_v1") and "moving_to" not in app_of(env, "lab_v2")
    assert kept.recover_moves() == []


def test_rollback_undo_failure_does_not_reverse(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    bk = tmp_path / "s.tar"
    bk.write_bytes(b"s")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_snap", "p", "chess", "m1", 1.0, str(bk), "pre_switch"))
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_snap", switch_snapshots=["bk_snap"])
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)

    def restore(lab, bid, snapshot=True, start=True):
        raise LabError("internal", "disk full", pre_restore="bk_pre" if snapshot else None)
    monkeypatch.setattr(kept, "_restore_locked", restore)
    monkeypatch.setattr(kept, "_reverse", lambda *a: pytest.fail("must not reverse onto wiped data"))
    out = kept.rollback("lab_v2", restore_snapshot=True)
    assert out["reversed"] is False and out["pre_restore"] == ["bk_pre"]
    v1 = app_of(env, "lab_v1")
    assert v1["state"] == "stopped" and "bk_pre" in v1["last_error"]
    assert env.store.lab("lab_v2")["state"] == "ready"


def test_staged_post_swap_failure_stays_staged(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", mounted=False, state="staged")
    _deploy_stubs(kept, env, monkeypatch)

    def must(lab, m, cmd, what, *a, **k):
        if what == "swap":
            raise LabError("internal", "mv failed")
        return {"exit_code": 0}
    monkeypatch.setattr(kept, "_must", must)
    with pytest.raises(LabError):
        kept._deploy_into("lab_a", app, b"t", {}, [])
    assert app_of(env, "lab_a")["state"] == "staged"


def test_restore_without_start_marks_stopped(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    bk = tmp_path / "b.tar"
    bk.write_bytes(b"t")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_1", "p", "chess", "m1", 1.0, str(bk), "manual"))
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "x"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    kept._restore_locked("lab_a", "bk_1", start=False)
    assert app_of(env, "lab_a")["state"] == "stopped"


def test_final_backup_with_hook_stops_the_app(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_a", state="up", backup={"cmd": "dump", "restore_cmd": None})
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(get_archive=lambda p: (_Stream([b"x"]), {})))
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    stops = []
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: stops.append(1))
    kept._backup_locked("lab_a", "daily")
    assert stops == []
    kept._backup_locked("lab_a", "final", restart=False)
    assert stops == [1]


def test_tick_keeps_a_stopped_apps_diagnosis(kept, env):  # noqa: F811
    make_kept(env, "lab_a", state="stopped", last_error="restore of bk_1 failed (pre-restore snapshot bk_2)")
    env.docker._containers[env.container_name("lab_a", "m1")].status = "exited"
    kept._last_daily = time.time()
    kept.tick()
    assert "bk_2" in app_of(env, "lab_a")["last_error"]


def test_staged_deploy_installs_no_cron(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", mounted=False, state="staged", cron=["* * * * * root x"])
    _deploy_stubs(kept, env, monkeypatch)
    cron = []
    monkeypatch.setattr(kept, "_install_cron", lambda *a: cron.append(1))
    kept._deploy_into("lab_a", app, b"t", {}, [])
    assert cron == []


def test_staged_seed_union_checked(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", mounted=False, state="staged", data_seed=["saves"])
    _deploy_stubs(kept, env, monkeypatch)
    with pytest.raises(LabError):
        kept._deploy_into("lab_a", app, b"t", {}, ["saves/slot1"])


def test_destroy_with_data_on_adopted_row(kept, env):  # noqa: F811
    add_lab(env, "lab_ad", lifetime="kept", owner="project:p")
    env.store.update_lab("lab_ad", app_json=dumps({"state": "stopped", "last_error": "secrets_lost"}))
    env.docker.volumes.create("ghost-labdata-p-x-m1", labels={"ghost.labs": "1"})
    out = kept.destroy_with_data("lab_ad", "owner")
    assert out["destroyed"] and out["data_volumes_removed"] == [] and "ghost-labdata-p-x-m1" in env.docker._volumes


def test_reverse_failure_itself_clears_markers(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)

    def move(to, fr, archive):
        a = app_of(env, fr)
        a["moving_to"] = to
        env.store.update_lab(fr, app_json=dumps(a))
        raise LabError("internal", "boom")
    monkeypatch.setattr(kept, "_move_data", move)
    assert kept._reverse("lab_v1", "lab_v2", (b"v1", {}, []), "why")["ok"] is False
    assert "moving_to" not in app_of(env, "lab_v2")


def test_switch_clears_markers_even_if_reverse_does_not(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)

    def move(to, fr, archive):
        a = app_of(env, fr)
        a["moving_to"] = to
        env.store.update_lab(fr, app_json=dumps(a))
        raise LabError("internal", "boom")
    monkeypatch.setattr(kept, "_move_data", move)
    monkeypatch.setattr(kept, "_reverse", lambda *a: {"ok": False})
    kept.switch("lab_v2", "lab_v1")
    assert "moving_to" not in app_of(env, "lab_v1")



# ================================================================ review r4 pins
def test_recover_only_reports_and_moves_nothing(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    for lab, mark in (("lab_v1", "lab_v2"), ("lab_v2", "lab_v1")):        # markers on both labs
        a = app_of(env, lab)
        a.update(mounted=False, state="staged", moving_to=mark)
        env.store.update_lab(lab, app_json=dumps(a))
    monkeypatch.setattr(kept, "_recreate", lambda *a, **k: pytest.fail("nothing may move at start"))
    monkeypatch.setattr(kept, "_deploy_into", lambda *a, **k: pytest.fail("nothing may move at start"))
    out = kept.recover_moves()
    assert out and kept.recover_moves() == []
    for lab in ("lab_v1", "lab_v2"):
        a = app_of(env, lab)
        assert "moving_to" not in a and "interrupted" in a["last_error"] and "switch" in a["last_error"]


def test_switch_brings_a_version_back_when_none_is_live(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    moved = []
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: moved.append(to) or ({"ok": True}, app_of(env, to)))
    assert kept.switch("lab_v2", "lab_v1")["switched"] is True and moved == ["lab_v2"]


def test_switch_from_unmounted_refused_while_another_is_live(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    make_kept(env, "lab_v3")                                    # the live one
    with pytest.raises(LabError) as e:
        kept.switch("lab_v2", "lab_v1")
    assert e.value.extra["field"] == "from_lab"


def test_switch_refuses_a_different_seed_list(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    for lab, seeds in (("lab_v1", ["saves"]), ("lab_v2", ["saves/slot1"])):
        a = app_of(env, lab)
        a["data_seed"] = seeds
        env.store.update_lab(lab, app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda *a: pytest.fail("must refuse before moving"))
    with pytest.raises(LabError) as e:
        kept.switch("lab_v2", "lab_v1")
    assert e.value.extra["field"] == "data_seed"


def test_staged_seed_must_equal_the_live_list(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1", data_seed=["saves"])
    app = make_kept(env, "lab_v2", mounted=False, state="staged")
    _deploy_stubs(kept, env, monkeypatch)
    with pytest.raises(LabError) as e:
        kept._deploy_into("lab_v2", app, b"t", {}, ["logs"])
    assert e.value.code == "data_exists"
    kept._deploy_into("lab_v2", app, b"t", {}, ["saves"])
    assert app["data_seed"] == ["saves"]


def test_mount_keeps_its_own_seed_list(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_v1", mounted=False, state="staged", data_seed=["saves"])
    monkeypatch.setattr(kept, "_recreate", lambda *a, **k: None)
    got = {}
    monkeypatch.setattr(kept, "_deploy_into", lambda lab, a, *x, **k: got.setdefault("seed", list(a["data_seed"])) and {"ok": True})
    kept._mount("lab_v1", (b"t", {}, []), ["other"])
    assert got["seed"] == ["saves"]


def test_failed_restore_is_held_against_restarts(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    bk = tmp_path / "b.tar"
    bk.write_bytes(b"t")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_1", "p", "chess", "m1", 1.0, str(bk), "manual"))
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    monkeypatch.setattr(kept, "_must", lambda *a, **k: {"exit_code": 0})
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: False))
    with pytest.raises(LabError):
        kept._restore_locked("lab_a", "bk_1")
    assert not Kept._startable(app_of(env, "lab_a"), "m1")


def test_rollback_undo_failure_is_held(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    bk = tmp_path / "s.tar"
    bk.write_bytes(b"s")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_snap", "p", "chess", "m1", 1.0, str(bk), "pre_switch"))
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1", switch_snapshot="bk_snap", switch_snapshots=["bk_snap"])
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)

    def restore(lab, bid, snapshot=True, start=True):
        raise LabError("internal", "disk full", pre_restore="bk_pre" if snapshot else None)
    monkeypatch.setattr(kept, "_restore_locked", restore)
    kept.rollback("lab_v2", restore_snapshot=True)
    v1 = app_of(env, "lab_v1")
    v1["mounted"] = True
    assert not Kept._startable(v1, "m1")


def test_lease_failure_keeps_a_finished_switch(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, a: ({"ok": True}, app_of(env, to)))

    def bad_lease(*a, **k):
        raise TimeoutError("tailscale serve hung")
    monkeypatch.setattr(kept, "_lease", bad_lease)
    out = kept.switch("lab_v2", "lab_v1")
    assert out["switched"] is True and "hung" in out["lease_error"]
    v2 = app_of(env, "lab_v2")
    assert v2["switched_from"] == "lab_v1" and "did not move" in v2["last_error"]
    assert env.store.lab("lab_v1")["retire_at"]


def test_lease_failure_keeps_a_finished_rollback(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v2")
    a.update(mounted=True, state="up", switched_from="lab_v1")
    env.store.update_lab("lab_v2", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))

    def bad_lease(*a, **k):
        raise TimeoutError("systemctl hung")
    monkeypatch.setattr(kept, "_lease", bad_lease)
    out = kept.rollback("lab_v2", restore_snapshot=False)
    assert out["rolled_back_from"] == "lab_v2" and "hung" in out["lease_error"]
    assert app_of(env, "lab_v2")["switched_from"] is None and env.store.lab("lab_v2")["retire_at"]


def test_seed_command_refuses_symlinks_and_missing_paths(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="stopped")
    cmds = _deploy_stubs(kept, env, monkeypatch)
    kept._deploy_into("lab_a", app, b"t", {}, ["saves"])
    c = [x for x in cmds if "ln -s" in x][0]
    assert "readlink -m" in c and "[ -L /app/saves ]" in c and "exit 8" in c
    assert c.index("exit 9") < c.index("mkdir -p")


def test_recover_tells_the_other_version_too(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged", moving_to="lab_v2")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    kept.recover_moves()
    assert "interrupted" in app_of(env, "lab_v2")["last_error"]


def test_switch_records_itself_before_the_lease_moves(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, a: ({"ok": True}, app_of(env, to)))
    seen = []
    monkeypatch.setattr(kept, "_lease", lambda lab, app, move: seen.append(
        (app_of(env, "lab_v2").get("switched_from"), env.store.lab("lab_v1")["retire_at"] is not None)))
    kept.switch("lab_v2", "lab_v1")
    assert seen == [("lab_v1", True)]



# ================================================================ review r5 pins
def test_recovery_switch_works_with_nothing_mounted(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    real_backup = Kept._backup_locked
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: real_backup(kept, *a, **k))   # the REAL one refuses unmounted
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": True}, app_of(env, to)))
    out = kept.switch("lab_v2", "lab_v1")
    assert out["switched"] is True and out["snapshot"] is None


def test_restore_hold_is_saved_before_data_is_cleared(kept, env, monkeypatch, tmp_path):  # noqa: F811
    make_kept(env, "lab_a", state="up")
    bk = tmp_path / "b.tar"
    bk.write_bytes(b"t")
    env.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_1", "p", "chess", "m1", 1.0, str(bk), "manual"))
    monkeypatch.setattr(kept, "_backup_locked", lambda *a, **k: {"backup_id": "bk_pre"})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    seen = []

    def must(lab, m, cmd, what, *a, **k):
        if what == "clear /data":
            seen.append(Kept._startable(app_of(env, "lab_a"), "m1"))
        return {"exit_code": 0}
    monkeypatch.setattr(kept, "_must", must)
    monkeypatch.setattr(env, "container", lambda lab, m: SimpleNamespace(put_archive=lambda p, f: True))
    monkeypatch.setattr(kept, "_start_app", lambda *a, **k: None)
    monkeypatch.setattr(kept, "wait_up", lambda *a, **k: {"ok": True})
    kept._restore_locked("lab_a", "bk_1")
    assert seen == [False]
    a = app_of(env, "lab_a")
    assert a["stopped_by_owner"] is False and a["last_error"] is None


def test_ensure_leases_repoints_the_port_to_the_live_version(kept, env, monkeypatch):  # noqa: F811
    make_kept(env, "lab_new")
    make_kept(env, "lab_old", mounted=False, state="staged")
    env.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20005, "lab_old", "m1", 8080, "p/chess", 0))
    moved = []
    monkeypatch.setattr(kept, "_lease", lambda lab, app, move: moved.append(lab))
    assert kept.ensure_leases() and moved == ["lab_new"]
    env.store.x("UPDATE ports SET lab_id='lab_new' WHERE port=20005")
    moved.clear()
    assert kept.ensure_leases() == [] and moved == []


def test_tick_runs_ensure_leases(kept, monkeypatch):
    calls = []
    monkeypatch.setattr(kept, "ensure_leases", lambda: calls.append(1) or [])
    kept._last_daily = time.time()
    kept.tick()
    assert calls == [1]


def test_recover_unmounts_stops_and_unretires_both(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged", moving_to="lab_v2")
    env.store.update_lab("lab_v1", app_json=dumps(a), retire_at=time.time() + 100)
    b = app_of(env, "lab_v2")
    b.update(mounted=True, state="up")
    env.store.update_lab("lab_v2", app_json=dumps(b))
    stopped = []
    monkeypatch.setattr(kept, "_stop_proc", lambda lab, app: stopped.append(lab))
    kept.recover_moves()
    assert sorted(stopped) == ["lab_v1", "lab_v2"]
    for lab in ("lab_v1", "lab_v2"):
        x = app_of(env, lab)
        assert x["mounted"] is False and x["state"] == "staged" and env.store.lab(lab)["retire_at"] is None


def test_missing_seed_refused_before_the_app_stops(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_a", state="up", data_seed=["saves"])
    _deploy_stubs(kept, env, monkeypatch)
    monkeypatch.setattr(kept, "_sh", lambda lab, m, cmd, *a, **k: {"exit_code": 1, "stdout": "", "stderr": ""})
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: pytest.fail("the running app must not stop"))
    with pytest.raises(LabError) as e:
        kept._deploy_into("lab_a", app, b"t", {}, [])
    assert e.value.extra["field"] == "data_seed"


# ================================================================ review r6 pins
def test_partial_lease_failure_is_retried(kept, env, monkeypatch):  # noqa: F811
    app = make_kept(env, "lab_new")
    env.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20006, "lab_new", "m1", 8080, "p/chess", 0))  # row moved

    def half(lab, a, move):
        raise TimeoutError("systemctl hung")
    monkeypatch.setattr(kept, "_lease", half)
    assert kept._lease_or_note("lab_new", app, "switched")
    assert app_of(env, "lab_new")["lease_pending"] is True
    kept.ensure_leases()                                   # still failing: throttled, reason on the app
    a = app_of(env, "lab_new")
    assert a["lease_retry_at"] > time.time() + 50 and "did not move" in a["last_error"]
    calls = []
    monkeypatch.setattr(kept, "_lease", lambda lab, a, move: calls.append(lab))
    kept.ensure_leases()
    assert calls == []                                     # within the minute
    a["lease_retry_at"] = 0
    env.store.update_lab("lab_new", app_json=dumps(a))
    kept.ensure_leases()
    a = app_of(env, "lab_new")
    assert calls == ["lab_new"] and "lease_pending" not in a and a["last_error"] is None


def test_recover_keeps_a_restore_diagnosis(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(moving_to="lab_v2", last_error="restore of bk_s in progress (pre-restore snapshot bk_pre)")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    monkeypatch.setattr(kept, "_stop_proc", lambda *a: None)
    kept.recover_moves()
    e = app_of(env, "lab_v1")["last_error"]
    assert "interrupted" in e and "bk_pre" in e


def test_failed_recovery_switch_does_not_reverse(kept, env, monkeypatch, tmp_path):  # noqa: F811
    _two_versions(env, kept, monkeypatch, tmp_path)
    a = app_of(env, "lab_v1")
    a.update(mounted=False, state="staged")
    env.store.update_lab("lab_v1", app_json=dumps(a))
    monkeypatch.setattr(kept, "_move_data", lambda to, fr, arch: ({"ok": False, "error": "probe"}, app_of(env, to)))
    monkeypatch.setattr(kept, "_reverse", lambda *a: pytest.fail("no automatic choice in a recovery"))
    out = kept.switch("lab_v2", "lab_v1")
    assert out["switched"] is False and out["reversed"] is False and "no version is live" in app_of(env, "lab_v2")["last_error"]
