"""Pins for the host review of 2026-11-05 (round 11): CRIT 1-3, MAJOR 4-8, MINOR 9-17; r12 F1-F5, r13, r14 at the end.

Each test fails on the code before the fix. Docker is the in-memory fake (tests/unit/fakes.py); a step is
monkeypatched only when it is not the behaviour the test is about."""
import base64
import http.server
import os
import shlex
import socketserver
import stat
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from ghost_labs import config as C
from ghost_labs import net, redact
from ghost_labs import templates as T
from ghost_labs.errors import LabError
from ghost_labs.kept import data_volume_name
from ghost_labs.store import dumps, loads

from .fakes import FakeDocker, runner_ok
from .test_core import ROOT, add_lab, env  # noqa: F401
from .test_kept_fixes import _stub_keep_steps, app_of, kept, make_kept  # noqa: F401

SPEC_M1 = {"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 512, "cpus": 1.0, "cap_add": []}],
           "egress": "none", "readiness": None}


def _spec(m, lab_id, spec=None):
    m.store.update_lab(lab_id, spec_json=dumps(spec or SPEC_M1))


def _no_ready(m, monkeypatch):
    monkeypatch.setattr(m, "wait_ready", lambda *a, **k: None)


# ================================================================== CRIT 1: reconcile adopts a kept lab with IPs
def test_reconcile_adopts_a_kept_lab_whose_member_has_a_real_ip(env):  # noqa: F811
    m = env
    c = m.docker.containers.run("img", name="ghost-lab-lab_k-app", labels={
        "ghost.labs": "1", "ghost.lab.id": "lab_k", "ghost.lab.member": "app",
        "ghost.lab.lifetime": "kept", "ghost.lab.owner": "project:p"})
    c.attrs["NetworkSettings"] = {"Networks": {"ghost-lab-lab_k": {"IPAddress": "10.89.3.10"}}}
    c.attrs["HostConfig"] = {"Memory": 512 * 1024 * 1024}
    c.attrs["Config"] = {"Image": "ghost-lab/base:1"}
    lines = m.reconcile()                                   # raised TypeError (store.lab None) before
    row = m.store.lab("lab_k")
    assert row["lifetime"] == "kept" and row["subnet"] == "10.89.3.0/24"
    assert [(x["name"], x["ip"], x["mem_mb"]) for x in m.store.members("lab_k")] == [("app", "10.89.3.10", 512)]
    assert any("adopt kept lab_k" in ln for ln in lines)


# ================================================================== CRIT 2: the probe namespace
def _owner_left_volume(m, lineage="shop", app="web", member="m1"):
    vol = data_volume_name(lineage, app, member)
    m.docker.volumes.create(vol, labels={"ghost.labs": "1", "ghost.data.lineage": lineage, "ghost.data.app": app,
                                         "ghost.data.member": member})
    return vol


def test_a_probe_cannot_mount_or_delete_the_data_an_owner_left_behind(kept, env, monkeypatch):  # noqa: F811
    """The reviewer's scenario: the owner destroyed kept shop/web keeping its data; a probe makes a lab for
    project:shop and keeps it as web. It must be refused, and nothing of the owner's may go."""
    m = env
    vol = _owner_left_volume(m)
    m.store.set_secret("shop", "web", "DB_PASSWORD", "owner-secret-123")
    add_lab(m, "lab_pr", owner="project:shop", origin="probe")
    _spec(m, "lab_pr")
    _stub_keep_steps(kept, monkeypatch)
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "web", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only" and "data volumes" in e.value.message
    row = m.store.lab("lab_pr")
    assert row["lifetime"] == "disposable" and not row["app_json"]          # nothing was kept
    assert vol in m.docker._volumes and m.store.secrets("shop", "web") == {"DB_PASSWORD": "owner-secret-123"}


@pytest.mark.parametrize("left", ["secrets", "lease", "backup", "archive"])
def test_a_probe_cannot_take_any_other_thing_an_owner_left_behind(kept, env, monkeypatch, left):  # noqa: F811
    m = env
    if left == "secrets":
        m.store.set_secret("shop", "web", "TOKEN", "owner-token-123")
    elif left == "lease":
        m.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20050, "", "m1", 80, "shop/web", 0))
    elif left == "backup":
        m.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)", ("bk_1", "shop", "web", "m1", time.time(), "/x", "final"))
    else:
        m.store.x("INSERT INTO archives VALUES (?,?,?,?,?,?)", ("shop", "web", time.time(), "/x", "{}", "[]"))
    add_lab(m, "lab_pr", owner="project:shop", origin="probe")
    _spec(m, "lab_pr")
    _stub_keep_steps(kept, monkeypatch)
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "web", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only"


def test_a_probe_cannot_take_the_service_name_an_owner_left_behind(kept, env, monkeypatch):  # noqa: F811
    """svc:<app> is tailnet-wide: the owner's destroyed shop/web still holds svc:web through its kept lease."""
    m = env
    m.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20051, "", "m1", 80, "shop/web", 0))
    add_lab(m, "lab_pr", owner="project:probeproj", origin="probe")
    _spec(m, "lab_pr")
    _stub_keep_steps(kept, monkeypatch)
    with pytest.raises(LabError) as e:
        kept.keep("lab_pr", {"app": {"name": "web", "member": "m1", "port": 80}}, "probe")
    assert e.value.code == "owner_only" and "svc:web" in e.value.message


def test_a_probe_keeps_into_its_own_namespace_and_retires_exactly_it(kept, env, monkeypatch):  # noqa: F811
    m = env
    _stub_keep_steps(kept, monkeypatch)
    add_lab(m, "lab_p1", owner="project:pp", origin="probe")
    _spec(m, "lab_p1")
    kept.keep("lab_p1", {"app": {"name": "site", "member": "m1", "port": 80}, "secret_values": {"K": "probe-value-1"}},
              "probe")
    vol = _owner_left_volume(m, "pp", "site")                # what its /data would be (made by the probe's keep)
    add_lab(m, "lab_p2", owner="project:pp", origin="probe")
    _spec(m, "lab_p2")
    other = _owner_left_volume(m, "pp", "other")             # another app of the lineage: not the probe's
    m.store.set_secret("pp", "other", "X", "owner-other-1")
    # a second probe version of its own app is allowed (the namespace is the probe's)
    kept.keep("lab_p2", {"app": {"name": "site", "member": "m1", "port": 80}}, "probe")
    m.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (20052, "lab_p1", "m1", 80, "pp/site", 0))
    for lid in ("lab_p1", "lab_p2"):
        m.store.update_lab(lid, retire_at=time.time() - 1)
    kept.retire_probe_lab("lab_p2")
    out = kept.retire_probe_lab("lab_p1")
    assert vol in out["data_volumes_removed"] and vol not in m.docker._volumes
    assert m.store.secrets("pp", "site") == {} and m.store.q("SELECT 1 FROM ports WHERE lease_key='pp/site'") == []
    assert other in m.docker._volumes and m.store.secrets("pp", "other") == {"X": "owner-other-1"}


def test_probe_retirement_keeps_a_namespace_the_owner_took_over(kept, env, monkeypatch):  # noqa: F811
    """A probe made pp/site; then the OWNER kept a version of it: from then on no probe retirement deletes it."""
    m = env
    _stub_keep_steps(kept, monkeypatch)
    add_lab(m, "lab_p1", owner="project:pp", origin="probe")
    _spec(m, "lab_p1")
    kept.keep("lab_p1", {"app": {"name": "site", "member": "m1", "port": 80}}, "probe")
    vol = _owner_left_volume(m, "pp", "site")
    add_lab(m, "lab_o1", owner="project:pp", origin="owner")
    _spec(m, "lab_o1")
    kept.keep("lab_o1", {"app": {"name": "site", "member": "m1", "port": 80}}, "owner")
    m.destroy("lab_o1", include_kept=True)                   # the owner's version goes, its data stays (§11.3)
    m.store.update_lab("lab_p1", retire_at=time.time() - 1)
    out = kept.retire_probe_lab("lab_p1")
    assert out["data_volumes_removed"] == [] and vol in m.docker._volumes
    assert out.get("kept_back")


def test_probe_retirement_never_deletes_an_unclaimed_namespace(kept, env):  # noqa: F811
    """The reviewer's last step: a probe kept lab in shop/web whose data a probe did not make (it came from
    before this fix, or was attached some other way) retires without deleting shop/web's volumes."""
    m = env
    vol = _owner_left_volume(m, "shop", "web")
    make_kept(m, "lab_pr", lineage="shop", name="web")
    m.store.update_lab("lab_pr", origin="probe", retire_at=time.time() - 1)
    m.store.set_secret("shop", "web", "DB_PASSWORD", "owner-secret-123")
    out = kept.retire_probe_lab("lab_pr")
    assert vol not in out["data_volumes_removed"] and vol in m.docker._volumes
    assert m.store.secrets("shop", "web") == {"DB_PASSWORD": "owner-secret-123"}


# ================================================================== CRIT 3: anonymous volumes
def _mysql_lab(m, monkeypatch):
    m.docker.images.tags.append("ghost-lab/mysql:8")
    _no_ready(m, monkeypatch)
    out = m.create({"members": [{"name": "db1", "image": "ghost-lab/mysql:8", "mem_mb": 512}], "egress": "none",
                    "owner": "project:p"}, "owner")
    lid = out["lab_id"]
    c = m.docker._containers[m.container_name(lid, "db1")]
    assert len(c.anon_volumes) == 1 and c.anon_volumes[0] in m.docker._volumes
    return lid, c


def test_destroy_removes_the_anonymous_volume_of_an_image_volume_line(env, monkeypatch):  # noqa: F811
    m = env
    lid, c = _mysql_lab(m, monkeypatch)
    anon = c.anon_volumes[0]
    m.destroy(lid)
    assert c.removed_with_volumes is True and anon not in m.docker._volumes


def test_a_kept_mysql_recreate_drops_its_anonymous_volume_and_keeps_the_datadir_on_data(env, kept, monkeypatch):  # noqa: F811
    m = env
    lid, c = _mysql_lab(m, monkeypatch)
    anon = c.anon_volumes[0]
    app = {"name": "db", "lineage": "p", "member": "db1", "data": "shared"}
    m.store.update_lab(lid, lifetime="kept", lineage="p", app_json=dumps(app))
    kept._recreate(lid, "db1", mount=True)
    assert anon not in m.docker._volumes
    new = m.docker._containers[m.container_name(lid, "db1")]
    hc = new.create_kwargs["host_config"]
    # the database lives on the named data volume, not in the (now removed) anonymous one
    assert hc["binds"] == {data_volume_name("p", "db", "db1"): {"bind": "/data", "mode": "rw"}}
    assert new.create_kwargs["environment"]["MYSQL_DATADIR"] == "/data/mysql"


def test_gc_removes_only_the_dangling_anonymous_volumes_lab_containers_had(env, monkeypatch):  # noqa: F811
    m = env
    lid, c = _mysql_lab(m, monkeypatch)
    anon = c.anon_volumes[0]
    c.remove(force=True)                     # a removal that left its anonymous volume behind
    foreign = "f" * 64                       # another program's anonymous volume
    m.docker.volumes.create(foreign, labels={"com.docker.volume.anonymous": ""})
    lid2, c2 = _mysql_lab(m, monkeypatch)    # a running lab's anonymous volume: in use, never removed
    out = m.gc()
    assert out["anon_volumes_removed"] == [anon]
    assert anon not in m.docker._volumes and foreign in m.docker._volumes and c2.anon_volumes[0] in m.docker._volumes


# ================================================================== MAJOR 4: the rehearsal twin's pool
def test_the_twin_counts_in_the_kept_pool_not_the_disposable_one(env, monkeypatch):  # noqa: F811
    m = env
    for i in range(3):
        add_lab(m, f"lab_d{i}", mem=800)                     # disposable pool (2500) nearly full, 3 of 4 labs
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    body = {"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 800}], "egress": "none",
            "owner": "project:p"}
    out = m.create(body, "owner", pool="kept")
    used = m._used()
    assert used == {"kept": 800, "disposable": 2400}
    assert out["lab_id"] not in [l["lab_id"] for l in m._disposable_labs()]      # not toward max_labs
    add_lab(m, "lab_d3", mem=64)                             # the 4th disposable lab still fits max_labs=4
    assert len(m._disposable_labs()) == 4


def test_the_twin_needs_host_memory(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 600})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    with pytest.raises(LabError) as e:
        m.create({"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 800}], "egress": "none",
                  "owner": "project:p"}, "owner", pool="kept")
    assert e.value.code == "budget_exceeded" and e.value.extra["pool"] == "kept"
    assert m.store.labs() == []


def test_a_twin_left_by_a_dead_process_goes_to_the_reaper(env):  # noqa: F811
    m = env
    add_lab(m, "lab_tw", owner="project:p")
    m.store.update_lab("lab_tw", pool="kept")
    m.docker.containers.run("img", name=m.container_name("lab_tw", "m1"), labels={"ghost.labs": "1", "ghost.lab.id": "lab_tw"})
    m.reconcile()
    assert m.store.lab("lab_tw")["state"] == "destroying"
    assert {"lab_id": "lab_tw", "reason": "create_failed"} in m.reap_once()


def test_an_old_database_gets_the_pool_column(tmp_path):
    import sqlite3
    from ghost_labs.store import Store
    db = sqlite3.connect(str(tmp_path / "old.sqlite"))
    db.execute("CREATE TABLE labs (lab_id TEXT PRIMARY KEY, lifetime TEXT)")
    db.execute("INSERT INTO labs VALUES ('lab_a', 'kept')")
    db.commit()
    db.close()
    s = Store(tmp_path / "old.sqlite")
    assert s.q("SELECT pool FROM labs WHERE lab_id='lab_a'") == [{"pool": ""}]


# ================================================================== MAJOR 5: firewall by interface, no NET_RAW
def test_the_firewall_matches_lab_bridges_by_input_interface_and_blocks_routing_out(env):  # noqa: F811
    rules = net.expected_fw_rules("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24")
    for ifc in ("glab-+", "glabs-gw0", "glabs-api0"):
        assert ["filter", f"-A GHOST-LABS-IN -i {ifc} -m conntrack ! --ctstate RELATED,ESTABLISHED -j DROP"] in rules
    assert ["mangle", "-A FORWARD -i glab-+ ! -o glab-+ -j DROP"] in rules

    def no_mangle(cmd, **kw):
        out = ""
        if cmd[:1] == ["iptables"] and "-S" in cmd:
            table = cmd[cmd.index("-t") + 1]
            out = "\n".join(r for t, r in rules if t == table and t != "mangle")
        return subprocess.CompletedProcess(cmd, 0, out, "")
    env.run = no_mangle
    assert env.check_firewall() == ["mangle: -A FORWARD -i glab-+ ! -o glab-+ -j DROP"]
    assert env.firewall_state == "missing"


def test_the_fw_script_installs_exactly_the_rules_the_check_expects(tmp_path):
    """bin/ghost-labs-fw run against a recording iptables: what it adds is what net.expected_fw_rules checks."""
    script = (ROOT / "bin" / "ghost-labs-fw").read_text()
    conf = tmp_path / "fw.env"
    conf.write_text("POOL=10.89.0.0/16\nGW_SUBNET=10.88.0.0/24\nAPI_SUBNET=10.88.1.0/24\n")
    script = script.replace("CONF=/etc/ghost-labs/fw.env", f"CONF={conf}")
    (tmp_path / "fw").write_text(script)
    log = tmp_path / "calls"
    fake = tmp_path / "bin" / "iptables"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/bash\necho "$*" >> {log}\ncase " $* " in *" -C "*) exit 1;; esac\nexit 0\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    r = subprocess.run(["bash", str(tmp_path / "fw")], env={**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    added = set()
    for line in log.read_text().splitlines():
        a = line.split()
        table = "filter"
        if a[:2] == ["-t", "raw"] or a[:2] == ["-t", "mangle"]:
            table, a = a[1], a[2:]
        if a[0] == "-I":                                     # -I CHAIN 1 rule… → -A CHAIN rule…
            a = ["-A", a[1]] + a[3:]
        if a[0] == "-A":
            added.add((table, a[1], frozenset(" ".join(a[2:]).replace("! ", "!").split())))
    for table, rule in net.expected_fw_rules("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24"):
        a = rule.split()
        assert (table, a[1], frozenset(" ".join(a[2:]).replace("! ", "!").split())) in added, (table, rule)


def test_a_lab_network_gets_the_fixed_glab_bridge(env, monkeypatch):  # noqa: F811
    m = env
    _no_ready(m, monkeypatch)
    lid = m.create({"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 256}], "egress": "none",
                    "owner": "project:p"}, "owner")["lab_id"]
    br = m.docker._networks[net.lab_network_name(lid)].attrs["Options"][net.BRIDGE_OPT]
    assert br == "glab-" + lid[4:] and len(br) <= 15
    assert m.legacy_lab_networks() == []
    m.docker.networks.create("ghost-lab-lab_old", labels={"ghost.labs": "1", "ghost.lab.id": "lab_old"})
    assert m.legacy_lab_networks() == ["lab_old"]


def test_members_have_no_raw_sockets_unless_they_ask(env, monkeypatch):  # noqa: F811
    m = env
    _no_ready(m, monkeypatch)
    lid = m.create({"members": [{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 256},
                                {"name": "b", "image": "ghost-lab/base:1", "mem_mb": 256, "cap_add": ["NET_RAW"]},
                                {"name": "c", "image": "ghost-lab/base:1", "mem_mb": 256, "cap_add": ["NET_ADMIN"]}],
                    "egress": "none", "owner": "project:p"}, "owner")["lab_id"]
    hc = {n: m.docker._containers[m.container_name(lid, n)].create_kwargs["host_config"] for n in "abc"}
    assert hc["a"]["cap_drop"] == ["NET_RAW"] and hc["c"]["cap_drop"] == ["NET_RAW"]
    assert hc["b"]["cap_drop"] is None and hc["b"]["cap_add"] == ["NET_RAW"]
    assert hc["a"]["sysctls"]["net.ipv4.ping_group_range"] == "0 2147483647"     # ping keeps working
    with pytest.raises(LabError):
        T.parse({"name": "x", "members": [{"name": "a", "image": "i", "mem_mb": 64, "cap_add": ["SYS_ADMIN"]}]})


# ================================================================== MAJOR 6: concurrent /keep
def test_a_second_keep_of_the_same_lab_is_refused_under_the_lock(kept, env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_s", owner="project:p")
    _spec(m, "lab_s")
    _stub_keep_steps(kept, monkeypatch)
    real_get = m.get
    stale = dict(real_get("lab_s"))            # the row both keeps read before either took the lab lock
    monkeypatch.setattr(m, "get", lambda lab_id: stale if lab_id == "lab_s" else real_get(lab_id))
    kept.keep("lab_s", {"app": {"name": "one", "member": "m1", "port": 80}, "secret_values": {"A": "x" * 10}}, "owner")
    with pytest.raises(LabError) as e:
        kept.keep("lab_s", {"app": {"name": "two", "member": "m1", "port": 80}}, "owner")
    assert e.value.code == "already_kept"
    assert loads(m.store.lab("lab_s")["app_json"])["name"] == "one"


def test_parallel_keeps_of_two_labs_cannot_overfill_the_kept_pool(kept, env, monkeypatch):  # noqa: F811
    """The first keep is held right before it flips to kept (the old race window: its budget check was done
    and released); the second keep must not pass its check meanwhile."""
    m = env
    for lid in ("lab_a", "lab_b"):
        add_lab(m, lid, mem=1000, owner="project:p")         # kept pool is 1500: only one fits
        _spec(m, lid)
    _stub_keep_steps(kept, monkeypatch)
    paused, go = threading.Event(), threading.Event()
    real_update = m.store.update_lab

    def update(lab_id, **f):
        if lab_id == "lab_a" and f.get("lifetime") == "kept":
            paused.set()
            go.wait(5)
        return real_update(lab_id, **f)
    monkeypatch.setattr(m.store, "update_lab", update)
    errors = []

    def keep(lid, name):
        try:
            kept.keep(lid, {"app": {"name": name, "member": "m1", "port": 80}}, "owner")
        except LabError as e:
            errors.append((lid, e.code))
    ta = threading.Thread(target=keep, args=("lab_a", "aa"))
    tb = threading.Thread(target=keep, args=("lab_b", "bb"))
    ta.start()
    assert paused.wait(5)
    tb.start()
    tb.join(0.5)
    go.set()
    ta.join(10)
    tb.join(10)
    assert {l["lab_id"]: l["lifetime"] for l in m.store.labs()} == {"lab_a": "kept", "lab_b": "disposable"}
    assert errors == [("lab_b", "budget_exceeded")] and m._used()["kept"] == 1000


# ================================================================== MAJOR 7: job kill on a kept lab
def _kept_job(m):
    add_lab(m, "lab_k", lifetime="kept", owner="project:p", origin="owner")
    m.docker.containers.run("img", name=m.container_name("lab_k", "m1"), labels={"ghost.labs": "1", "ghost.lab.id": "lab_k"})
    m.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_1", "lab_k", "m1", "migrate", time.time(), None, "running", None))
    return m.docker._containers[m.container_name("lab_k", "m1")]


def test_autopilot_cannot_kill_a_job_in_a_kept_lab_over_the_api(env):  # noqa: F811
    from fastapi.testclient import TestClient
    from ghost_labs.api import create_app
    from .test_api import Svc
    m = env
    c = _kept_job(m)
    cl = TestClient(create_app(Svc(m), "k" * 64))
    for origin in ("autopilot", "probe", None):
        h = {"X-Labs-Key": "k" * 64, **({"X-Labs-Origin": origin} if origin else {})}
        r = cl.delete("/v1/labs/lab_k/jobs/job_1", headers=h)
        assert r.status_code == 403 and r.json()["error"] == "owner_only"
    assert not any("kill -TERM" in " ".join(x) for x in c.exec_log)
    r = cl.delete("/v1/labs/lab_k/jobs/job_1", headers={"X-Labs-Key": "k" * 64, "X-Labs-Origin": "owner"})
    assert r.status_code == 200 and r.json()["killed"]
    assert any("kill -TERM" in " ".join(x) for x in c.exec_log)


def test_a_job_kill_without_an_origin_is_refused_on_a_kept_lab(env):  # noqa: F811
    m = env
    _kept_job(m)
    with pytest.raises(LabError) as e:
        m.job("lab_k", "job_1", kill=True)
    assert e.value.code == "owner_only"
    assert m.job("lab_k", "job_1")["state"]            # reading a job stays open to every origin


# ================================================================== MAJOR 8 + MINOR 10: file redaction
def test_a_file_that_is_not_utf8_is_redacted_byte_wise(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_f")
    m.store.set_secret("p", "app", "DB_PASSWORD", "hunter2-very-secret")
    monkeypatch.setattr(m, "read_raw", lambda *a, **k: b"password=hunter2-very-secret\n\xff\xfe" +
                        base64.b64encode(b"hunter2-very-secret"))
    out = base64.b64decode(m.get_file("lab_f", "m1", "/x", "owner")["content_b64"])
    assert b"hunter2" not in out and out.count(b"<secret:DB_PASSWORD>") == 2 and b"\xff\xfe" in out


def test_secrets_env_with_a_quote_in_a_value_is_redacted_when_read(kept, env, monkeypatch):  # noqa: F811
    m = env
    make_kept(m, "lab_q", lineage="p", name="chess")
    value = "it's-a-secret-value"
    m.store.set_secret("p", "chess", "PASS", value)
    written = {}
    monkeypatch.setattr(m, "_put_bytes", lambda c, path, data, mode: written.update({path: data}))
    kept._write_secrets_file("lab_q", app_of(m, "lab_q"))
    body = written["/run/ghost/secrets.env"]
    assert body == f"export PASS={shlex.quote(value)}\n".encode()
    monkeypatch.setattr(m, "read_raw", lambda *a, **k: body)
    out = base64.b64decode(m.get_file("lab_q", "m1", "/run/ghost/secrets.env", "owner")["content_b64"])
    assert out == b"export PASS=<secret:PASS>\n"
    assert "secret-value" not in redact.apply(body.decode(), m.redactor())


def test_one_secret_name_in_two_apps_redacts_both_values(env):  # noqa: F811
    m = env
    m.store.set_secret("p", "a", "DB_PASSWORD", "first-value-111")
    m.store.set_secret("q", "b", "DB_PASSWORD", "second-value-222")
    assert m.redact_text("first-value-111 second-value-222") == "<secret:DB_PASSWORD> <secret:DB_PASSWORD>"


# ================================================================== MINOR 9: redact before the cut
def test_a_job_tail_is_redacted_before_it_is_cut(env):  # noqa: F811
    m = env
    add_lab(m, "lab_j")
    secret = "s3cr3t-token-abcdef"
    m.store.set_secret("p", "app", "TOK", secret)
    c = m.docker.containers.run("img", name=m.container_name("lab_j", "m1"), labels={"ghost.labs": "1", "ghost.lab.id": "lab_j"})
    log = ("x" * 5000 + secret + "y" * 50).encode()

    def exec_run(cmd, **kw):
        s = " ".join(cmd)
        if "tail -c" in s:
            n = int(s.split("tail -c ")[1].split()[0])
            return SimpleNamespace(exit_code=0, output=log[-n:])
        if "kill -0" in s:
            return SimpleNamespace(exit_code=0, output=b"no")
        return SimpleNamespace(exit_code=0, output=b"0")
    c.exec_run = exec_run
    m.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", ("job_2", "lab_j", "m1", "x", time.time(), None, "running", None))
    out = m.job("lab_j", "job_2", tail_bytes=60)["output_tail"]           # the cut falls inside the secret
    assert "<secret:TOK>" in out and secret[-10:] not in out and out.endswith("y" * 50)
    log = (secret + "y" * 30).encode()                                     # shorter than the margin: whole file
    out = m.job("lab_j", "job_2", tail_bytes=40)["output_tail"]
    assert out == "<secret:TOK>" + "y" * 30


class _Body(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        pass


def test_the_http_body_head_is_redacted_before_it_is_cut(env):  # noqa: F811
    m = env
    add_lab(m, "lab_h")
    m.store.x("UPDATE members SET ip='127.0.0.1' WHERE lab_id='lab_h'")
    secret = "s3cr3t-token-abcdef"
    m.store.set_secret("p", "app", "TOK", secret)
    _Body.body = b"a" * (16384 - 7) + secret.encode() + b"tail"
    srv = socketserver.TCPServer(("127.0.0.1", 0), _Body)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = m.http("lab_h", {"member": "m1", "port": srv.server_address[1], "path": "/"}, "owner")
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["status"] == 200
    assert out["body_head"].endswith("<secret:TOK>") and "s3cr3t" not in out["body_head"]


# ================================================================== MINOR 11: an empty key fails closed
def test_an_empty_api_key_refuses_to_serve(env, tmp_path, monkeypatch):  # noqa: F811
    from ghost_labs.api import create_app
    from .test_api import Svc
    for key in ("", "   ", "short"):
        with pytest.raises(ValueError):
            create_app(Svc(env), key)
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path)
    (tmp_path / "api_key").write_text("\n")
    with pytest.raises(RuntimeError):
        C.api_key()


# ================================================================== MINOR 13: no eviction that can't help
def test_an_owner_create_does_not_evict_when_host_memory_would_still_be_short(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_auto", mem=300, origin="autopilot", owner_use=0)
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 400})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    with pytest.raises(LabError) as e:
        m.create({"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 800}], "egress": "none",
                  "owner": "project:p"}, "owner")
    assert e.value.code == "budget_exceeded"
    assert m.store.lab("lab_auto") is not None                 # 400 + 300 freed < 800 + 512: nobody evicted


# ================================================================== MINOR 14: build under the lab lock
def test_a_destroy_during_the_build_waits_and_leaves_no_port_behind(env, monkeypatch):  # noqa: F811
    m = env
    _no_ready(m, monkeypatch)
    reached, go_on = threading.Event(), threading.Event()
    real_lease = m.ports.lease
    seen = {}

    def slow_lease(lab_id, *a, **k):
        seen["lab"] = lab_id
        reached.set()
        go_on.wait(5)
        return real_lease(lab_id, *a, **k)
    monkeypatch.setattr(m.ports, "lease", slow_lease)
    res = {}
    t_create = threading.Thread(target=lambda: res.update(c=m.create(
        {"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": 256, "ports": [80]}], "egress": "none",
         "owner": "project:p"}, "owner")))
    t_create.start()
    assert reached.wait(5)
    lid = seen["lab"]
    t_destroy = threading.Thread(target=lambda: res.update(d=m.destroy(lid)))
    t_destroy.start()
    t_destroy.join(0.5)
    blocked = t_destroy.is_alive()
    go_on.set()
    t_create.join(5)
    t_destroy.join(5)
    assert blocked                                             # it waited for the build
    assert m.store.lab(lid) is None and m.store.q("SELECT * FROM ports WHERE lab_id=?", (lid,)) == []


# ================================================================== MINOR 16: backups off the supervisor
def test_a_long_daily_backup_does_not_stall_supervision(kept, env, monkeypatch):  # noqa: F811
    m = env
    make_kept(m, "lab_a", name="a", last_backup_at=0)                       # its daily backup is due
    make_kept(m, "lab_b", name="b")
    kept._last_daily = time.time()
    release, started = threading.Event(), threading.Event()

    def long_hook(lab_id, kind, **kw):
        started.set()
        release.wait(10)
        return {}
    monkeypatch.setattr(kept, "_backup_locked", long_hook)
    monkeypatch.setattr(kept, "_alive", lambda *a: True)
    looked = []
    real_running = kept._member_running
    monkeypatch.setattr(kept, "_member_running", lambda lid, app: looked.append(lid) or real_running(lid, app))
    t = threading.Thread(target=kept.tick)
    t.start()
    t.join(3)
    try:
        assert not t.is_alive()                                             # the tick is not held by the backup
        assert started.wait(3) and "lab_b" in looked
        looked.clear()
        kept.tick()                                                         # the next tick runs too
        assert looked == ["lab_b"]                                          # lab_a's lock is the backup's meanwhile
    finally:
        release.set()
        t.join(5)
        getattr(kept, "join_backups", lambda: None)()
    assert not m.lock("lab_a").locked()


# ================================================================== MINOR 17: the cmd text is redacted
def test_the_exec_cmd_is_redacted_in_results_the_jobs_list_and_actions_log(env, tmp_path):  # noqa: F811
    from fastapi.testclient import TestClient
    from ghost_labs.api import create_app
    from ghost_labs.service import Service
    svc = Service(env.cfg, FakeDocker(), ROOT / "templates", tmp_path, runner=runner_ok(), start_threads=False)
    m = svc.mgr
    add_lab(m, "lab_x")
    m.docker.containers.run("img", name=m.container_name("lab_x", "m1"), labels={"ghost.labs": "1", "ghost.lab.id": "lab_x"})
    secret = "pg-pass-7777777"
    m.store.set_secret("p", "app", "PGPASS", secret)
    cl = TestClient(create_app(svc, "k" * 64))
    h = {"X-Labs-Key": "k" * 64, "X-Labs-Origin": "owner"}
    r = cl.post("/v1/labs/lab_x/exec", json={"member": "m1", "cmd": f"psql postgres://u:{secret}@db", "detach": True},
                headers=h)
    assert r.status_code == 202 and secret not in r.text and "<secret:PGPASS>" in r.json()["cmd"]
    job_id = r.json()["job_id"]
    assert secret not in cl.get("/v1/labs/lab_x/jobs", headers=h).text
    assert secret not in cl.get(f"/v1/labs/lab_x/jobs/{job_id}", headers=h).text
    logged = (tmp_path / "actions.log").read_text()
    assert "<secret:PGPASS>" in logged and secret not in logged


# ################################################################## r12 (fresh-eye review of the r11 diff)
def _owner_create(m, mem, egress="none"):
    return m.create({"members": [{"name": "m1", "image": "ghost-lab/base:1", "mem_mb": mem}], "egress": egress,
                     "owner": "project:p"}, "owner")


def _running(m, lab_id, usage_mb=0):
    c = m.docker.containers.run("img", name=m.container_name(lab_id, "m1"),
                                labels={"ghost.labs": "1", "ghost.lab.id": lab_id, "ghost.lab.member": "m1"})
    c.mem_usage = usage_mb * 1024 * 1024
    return c


# ================================================================== r12 F1: eviction counts what victims USE
def test_r12_an_idle_autopilot_lab_is_not_evicted_for_memory_it_does_not_use(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_auto", mem=1000, origin="autopilot", owner_use=0)
    _running(m, "lab_auto", usage_mb=30)                       # limit 1000 MB, uses 30 MB
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 600})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    with pytest.raises(LabError) as e:
        _owner_create(m, 200)                                  # 600 + 30 < 200 + 512
    assert e.value.code == "budget_exceeded"
    assert m.store.lab("lab_auto") is not None                 # nobody evicted for nothing


def test_r12_an_autopilot_lab_whose_use_makes_room_is_evicted(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_auto", mem=1000, origin="autopilot", owner_use=0)
    _running(m, "lab_auto", usage_mb=900)
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000,
                                               "MemAvailable": 600 + (0 if m.store.lab("lab_auto") else 900)})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    out = _owner_create(m, 200)
    assert [e["lab_id"] for e in out["evicted"]] == ["lab_auto"]


def test_r12_eviction_stops_once_the_rest_cannot_make_room(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_a", mem=500, origin="autopilot", owner_use=0, age=100)
    add_lab(m, "lab_b", mem=500, origin="autopilot", owner_use=0, age=10)
    m.store.put_member({"lab_id": "lab_b", "name": "m1", "ip": "10.89.9.11", "image": "ghost-lab/base:1",
                        "mem_mb": 500, "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})
    _running(m, "lab_a", usage_mb=300)
    _running(m, "lab_b", usage_mb=250)
    # 400 + 300 + 250 >= 712: the plan is feasible; but lab_a gave back only 20 MB, and 420 + 250 < 712
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000,
                                               "MemAvailable": 400 + (0 if m.store.lab("lab_a") else 20)})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    with pytest.raises(LabError) as e:
        _owner_create(m, 200)
    assert e.value.code == "budget_exceeded"
    assert m.store.lab("lab_a") is None and m.store.lab("lab_b") is not None     # lab_b not evicted for nothing
    assert [x["lab_id"] for x in e.value.extra["evicted"]] == ["lab_a"]


# ================================================================== r12 F2: creating at restart
def test_r12_a_kept_lab_creating_at_restart_is_unready_not_destroying(env):  # noqa: F811
    m = env
    make_kept(m, "lab_k", state="up")
    m.store.update_lab("lab_k", state="creating")
    m.reconcile()
    assert m.store.lab("lab_k")["state"] == "unready"
    assert m.reap_once() == [] and m.store.lab("lab_k")


def test_r12_a_kept_lab_left_destroying_by_r11_is_repaired(env):  # noqa: F811
    m = env
    make_kept(m, "lab_k", state="up")
    m.store.update_lab("lab_k", state="destroying")
    m.reconcile()
    assert m.store.lab("lab_k")["state"] == "unready"


def test_r12_a_fully_built_disposable_lab_creating_at_restart_is_unready(env):  # noqa: F811
    m = env
    add_lab(m, "lab_c", state="creating")
    _spec(m, "lab_c")
    _running(m, "lab_c")
    m.reconcile()
    assert m.store.lab("lab_c")["state"] == "unready"
    assert m.reap_once() == []


def test_r12_a_half_built_disposable_lab_creating_at_restart_is_reaped(env):  # noqa: F811
    m = env
    two = {**SPEC_M1, "members": SPEC_M1["members"] + [{**SPEC_M1["members"][0], "name": "m2"}]}
    add_lab(m, "lab_c", state="creating")
    _spec(m, "lab_c", two)
    _running(m, "lab_c")                                       # m2 has no member row and no container
    add_lab(m, "lab_p", state="creating")
    _spec(m, "lab_p", {**SPEC_M1, "members": [{**SPEC_M1["members"][0], "ports": [80]}]})
    m.docker.containers.run("img", name=m.container_name("lab_p", "m1"),
                            labels={"ghost.labs": "1", "ghost.lab.id": "lab_p", "ghost.lab.member": "m1"})
    m.reconcile()                                              # lab_p: its port 80 has no lease
    assert m.store.lab("lab_c")["state"] == "destroying" and m.store.lab("lab_p")["state"] == "destroying"
    assert {r["lab_id"] for r in m.reap_once()} == {"lab_c", "lab_p"}


# ================================================================== r12 F4: lock order
def test_r12_an_eviction_skips_a_locked_victim_instead_of_waiting(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_x", mem=2400, origin="autopilot", owner_use=0, owner="project:p")
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    held, release = threading.Event(), threading.Event()

    def keep_like():
        with m.locked("lab_x", timeout=60):        # what /keep holds while it waits for create_lock
            held.set()
            release.wait(10)
    t = threading.Thread(target=keep_like)
    t.start()
    assert held.wait(2)
    try:
        t0 = time.time()
        with pytest.raises(LabError) as e:
            _owner_create(m, 400)
        assert e.value.code == "budget_exceeded" and time.time() - t0 < 5     # no 30 s wait, no "busy"
        assert m.store.lab("lab_x") is not None
        assert not m.create_lock.locked()
    finally:
        release.set()
        t.join(5)


def test_r12_an_eviction_takes_the_unlocked_victim(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_x", mem=1200, origin="autopilot", owner_use=0, age=100)
    add_lab(m, "lab_y", mem=1200, origin="autopilot", owner_use=0, age=10)
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    lk = m.lock("lab_x")
    lk.acquire()
    try:
        out = _owner_create(m, 400)
    finally:
        lk.release()
    assert [e["lab_id"] for e in out["evicted"]] == ["lab_y"] and m.store.lab("lab_x")
    assert not m.lock("lab_y").locked()


def test_r12_create_waits_for_tor_without_the_create_lock(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    seen = []
    monkeypatch.setattr(m, "wait_tor", lambda *a, **k: seen.append(m.create_lock.locked()))
    _owner_create(m, 256, egress="tor")
    assert seen == [False]


# ================================================================== r12 F5: redact, then cut
def test_r12_a_secret_straddling_the_cut_is_not_logged(kept, env, tmp_path):  # noqa: F811
    from ghost_labs.service import ActionLog
    m = env
    app = make_kept(m, "lab_k")
    secret = "straddling-secret-0123456789"
    m.store.set_secret("p", "chess", "TOKEN", secret)
    m.actions = ActionLog(tmp_path / "actions.log", redactor=m.redact_text)
    why = "x" * 190 + secret                                   # the cut at 200 falls inside the secret
    kept._record_crash("lab_k", app, why)
    kept._exited("lab_k", app, why)
    logged = (tmp_path / "actions.log").read_text()
    assert secret[:10] not in logged and logged.count("<secret:TO") == 2


# ################################################################## r13 (fresh-eye review of the r12 diff)
# ================================================================== r13 A: hot page cache is not "freed"
def test_r13_eviction_does_not_count_active_page_cache_as_freed(env, monkeypatch):  # noqa: F811
    """Each victim: 100 MB anon + 200 MB ACTIVE page cache, which MemAvailable already counts as available.
    400 + 100 + 100 < 712: nobody is evicted (r12 measured 300 each and evicted both for nothing)."""
    m = env
    for lid, age in (("lab_a", 100), ("lab_b", 10)):
        add_lab(m, lid, mem=500, origin="autopilot", owner_use=0, age=age)
        _running(m, lid, usage_mb=300).mem_active_file = 200 * 1024 * 1024
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 400
                                               + (0 if m.store.lab("lab_a") else 100)
                                               + (0 if m.store.lab("lab_b") else 100)})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    with pytest.raises(LabError) as e:
        _owner_create(m, 200)
    assert e.value.code == "budget_exceeded" and "evicted" not in e.value.extra
    assert m.store.lab("lab_a") and m.store.lab("lab_b")


@pytest.mark.parametrize("stats,want", [
    ({"active_file": 200, "inactive_file": 50}, 50),                       # cgroup v2
    ({"total_active_file": 200, "total_inactive_file": 50,                 # cgroup v1: the hierarchical totals
      "active_file": 1, "inactive_file": 1}, 50),
    ({"active_file": 400, "inactive_file": 400}, 0),                       # clamped at 0
])
def test_r13_mem_in_use_subtracts_all_file_cache_v1_and_v2(env, monkeypatch, stats, want):  # noqa: F811
    m = env
    add_lab(m, "lab_a", origin="autopilot")
    c = _running(m, "lab_a")
    mb = 1024 * 1024
    monkeypatch.setattr(c, "stats", lambda **k: {"memory_stats": {"usage": 300 * mb,
                                                                  "stats": {k2: v * mb for k2, v in stats.items()}}})
    assert m._mem_in_use_mb("lab_a") == want


# ================================================================== r13 B: memory-only shortfall picks useful victims
def test_r13_an_idle_victim_is_not_evicted_when_only_memory_is_short(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_old", mem=300, origin="autopilot", owner_use=0, age=100)
    _running(m, "lab_old", usage_mb=0)
    add_lab(m, "lab_busy", mem=600, origin="autopilot", owner_use=0, age=10)
    _running(m, "lab_busy", usage_mb=600)
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000,
                                               "MemAvailable": 400 + (0 if m.store.lab("lab_busy") else 600)})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    out = _owner_create(m, 200)
    assert [x["lab_id"] for x in out["evicted"]] == ["lab_busy"] and m.store.lab("lab_old")


def test_r13_memory_only_shortfall_evicts_the_largest_user_first(env, monkeypatch):  # noqa: F811
    m = env
    frees = {"lab_a": 100, "lab_b": 400, "lab_c": 200}                   # oldest first
    for i, (lid, mb) in enumerate(frees.items()):
        add_lab(m, lid, mem=500, origin="autopilot", owner_use=0, age=100 - i)
        _running(m, lid, usage_mb=mb)
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 400 + sum(
        mb for lid, mb in frees.items() if not m.store.lab(lid))})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    out = _owner_create(m, 200)                                           # needs 312 more: lab_b alone
    assert [x["lab_id"] for x in out["evicted"]] == ["lab_b"]
    assert m.store.lab("lab_a") and m.store.lab("lab_c")


def test_r13_a_budget_shortfall_still_evicts_the_oldest_first(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_old", mem=1300, origin="autopilot", owner_use=0, age=100)
    _running(m, "lab_old", usage_mb=0)
    add_lab(m, "lab_new", mem=1200, origin="autopilot", owner_use=0, age=10)
    _running(m, "lab_new", usage_mb=900)
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    out = _owner_create(m, 400)                                           # 2500 + 400 > 2500 disposable
    assert [x["lab_id"] for x in out["evicted"]] == ["lab_old"] and m.store.lab("lab_new")


# ================================================================== r13 C: only a ready/unready lab can be kept
@pytest.mark.parametrize("state", ["destroying", "creating", "failed"])
def test_r13_keep_refuses_a_lab_that_is_not_ready(kept, env, monkeypatch, state):  # noqa: F811
    m = env
    add_lab(m, "lab_d", owner="project:p", state=state)
    m.store.update_lab("lab_d", spec_json=dumps(SPEC_M1))
    _running(m, "lab_d")
    calls = _stub_keep_steps(kept, monkeypatch)
    with pytest.raises(LabError) as e:
        kept.keep("lab_d", {"app": {"name": "web", "member": "m1", "port": 80}}, "owner")
    assert e.value.code == "not_ready" and e.value.status == 409 and e.value.extra["state"] == state
    assert m.store.lab("lab_d")["lifetime"] == "disposable" and not calls["recreate"]


def test_r13_keep_takes_an_unready_lab(kept, env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_u", owner="project:p", state="unready")
    m.store.update_lab("lab_u", spec_json=dumps(SPEC_M1))
    _running(m, "lab_u")
    _stub_keep_steps(kept, monkeypatch)
    kept.keep("lab_u", {"app": {"name": "web", "member": "m1", "port": 80}}, "owner")
    assert m.store.lab("lab_u")["lifetime"] == "kept"


# ================================================================== r13 E: one gateway create at a time
def test_r13_concurrent_creates_make_one_gateway(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(m, "ensure_infra", lambda: None)
    runs = []
    real_run = m._run_gateway

    def slow_run():
        runs.append(1)
        time.sleep(0.2)                    # the window two lock-free creates both fell into
        if m.docker._containers.get(net.GW_CONTAINER):
            raise RuntimeError("409 Conflict: name in use")
        return real_run()
    monkeypatch.setattr(m, "_run_gateway", slow_run)
    errs = []

    def go():
        try:
            m.gateway(start=True)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert not errs and runs == [1]


def test_r13_a_gateway_name_conflict_reads_the_one_made_elsewhere(env, monkeypatch):  # noqa: F811
    m = env
    monkeypatch.setattr(m, "ensure_infra", lambda: None)
    other = {}

    def conflict():                        # another process (e.g. the CLI) made it between get and run
        other["c"] = m.docker.containers.run("ghost-lab/tor-gw:1", name=net.GW_CONTAINER)
        raise type("APIError", (Exception,), {"status_code": 409})("Conflict")
    monkeypatch.setattr(m, "_run_gateway", conflict)
    assert m.gateway(start=True) is other["c"]

    def boom():
        raise type("APIError", (Exception,), {"status_code": 500})("server error")
    m.docker._containers.pop(net.GW_CONTAINER)
    monkeypatch.setattr(m, "_run_gateway", boom)
    with pytest.raises(Exception, match="server error"):
        m.gateway(start=True)


# ################################################################## r14: eviction's docker stats are bounded
def _hanging_stats(c, release):
    def stats(**k):
        release.wait(10)                   # a wedged docker: the client's own timeout is 300 s
        return {"memory_stats": {"usage": 900 * 1024 * 1024, "stats": {}}}
    c.stats = stats


def test_r14_a_hanging_stats_call_counts_zero_within_the_deadline(env, monkeypatch):  # noqa: F811
    from ghost_labs import labs as L
    m = env
    monkeypatch.setattr(L, "STATS_DEADLINE_S", 0.2)
    add_lab(m, "lab_h", origin="autopilot")
    add_lab(m, "lab_ok", origin="autopilot")
    release = threading.Event()
    _hanging_stats(_running(m, "lab_h"), release)
    _running(m, "lab_ok", usage_mb=300)
    try:
        t0 = time.monotonic()
        assert m._mem_in_use_many(["lab_h", "lab_ok"]) == {"lab_h": 0, "lab_ok": 300}
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()
    time.sleep(0.05)
    assert m._mem_in_use_mb("lab_h") == 900           # a late answer never leaks into a later measurement


def test_r14_a_raising_stats_call_counts_zero(env, monkeypatch):  # noqa: F811
    m = env
    add_lab(m, "lab_r", origin="autopilot")
    add_lab(m, "lab_ok", origin="autopilot")

    def boom(**k):
        raise RuntimeError("docker said no")
    monkeypatch.setattr(_running(m, "lab_r"), "stats", boom)
    _running(m, "lab_ok", usage_mb=200)
    assert m._mem_in_use_many(["lab_r", "lab_ok"]) == {"lab_r": 0, "lab_ok": 200}
    m.docker._containers.pop(m.container_name("lab_ok", "m1"))     # gone: 0 too
    assert m._mem_in_use_mb("lab_ok") == 0


def test_r14_a_wedged_stats_call_does_not_block_other_creates(env, monkeypatch):  # noqa: F811
    """Memory is short and the only victim's stats hang: each create waits at most ~one deadline for it, counts
    it as 0 MB (so nothing is evicted) and fails budget_exceeded; a concurrent second create is not stuck
    behind the first's create_lock for docker's 300 s."""
    from ghost_labs import labs as L
    m = env
    deadline = 0.3
    monkeypatch.setattr(L, "STATS_DEADLINE_S", deadline)
    add_lab(m, "lab_v", mem=500, origin="autopilot", owner_use=0)
    release = threading.Event()
    _hanging_stats(_running(m, "lab_v"), release)
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000,
                                               "MemAvailable": 400 + (0 if m.store.lab("lab_v") else 900)})
    monkeypatch.setattr(m, "_build", lambda *a, **k: None)
    _no_ready(m, monkeypatch)
    errs, took = [], []

    def create():
        t0 = time.monotonic()
        try:
            _owner_create(m, 200)
        except LabError as e:
            errs.append(e.code)
        took.append(time.monotonic() - t0)
    try:
        ts = [threading.Thread(target=create) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(5)
        assert errs == ["budget_exceeded", "budget_exceeded"]
        assert max(took) < 2 * deadline + 1.0             # the second waited one deadline, not 300 s
        assert m.store.lab("lab_v") is not None and not m.create_lock.locked()
        assert not m.lock("lab_v").locked()
    finally:
        release.set()


def test_r15_a_stats_call_in_flight_is_not_asked_again(env, monkeypatch):  # noqa: F811
    """review r15: each create started one more wedged stats thread per member (up to docker's 300 s timeout)."""
    import threading as _th
    from ghost_labs import labs as LB
    m = env
    add_lab(m, "lab_h", mem=512)
    gate = _th.Event()
    calls = []

    def wedged(lid, member):
        calls.append((lid, member))
        gate.wait(5)
        return 300 * 1024 * 1024
    monkeypatch.setattr(LB, "STATS_DEADLINE_S", 0.2)
    monkeypatch.setattr(m, "_container_mem_bytes", wedged)
    try:
        for _ in range(4):
            assert m._mem_in_use_many(["lab_h"]) == {"lab_h": 0}
        assert len(calls) == len(m.store.members("lab_h")) >= 1       # asked once, not once per measurement
    finally:
        gate.set()
    deadline = time.time() + 3
    while m._stats_inflight and time.time() < deadline:
        time.sleep(0.02)
    assert not m._stats_inflight                                     # the answer clears the mark
    monkeypatch.setattr(m, "_container_mem_bytes", lambda lid, member: 100 * 1024 * 1024)
    assert m._mem_in_use_many(["lab_h"])["lab_h"] == 100 * len(m.store.members("lab_h"))


def test_r16_a_thread_that_cannot_be_made_leaves_no_mark(env, monkeypatch):  # noqa: F811
    """review r16: the in-flight mark was set before the Thread was built, outside the try that clears it."""
    import threading as _th
    import pytest as _pt
    m = env
    add_lab(m, "lab_t", mem=512)

    def boom(*a, **k):
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(_th, "Thread", boom)
    with _pt.raises(RuntimeError):
        m._mem_in_use_many(["lab_t"])
    assert not m._stats_inflight
