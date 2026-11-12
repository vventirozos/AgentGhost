import base64
import time
from collections import deque
from pathlib import Path

import pytest

from ghost_labs import config as C
from ghost_labs import net, redact
from ghost_labs import templates as T
from ghost_labs.errors import STATUS, LabError
from ghost_labs.labs import Manager, _patroni_three_streaming, _tail_keep
from ghost_labs.ports import Ports
from ghost_labs.store import Store, dumps

from .fakes import FakeDocker, runner_ok

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "meminfo", lambda: {"MemTotal": 8000, "MemAvailable": 7000})
    monkeypatch.setattr(C, "lan_ipv4", lambda: "192.168.1.50")
    cfg = C.Config(ram_budget_mb=4000, kept_mb=1500, max_labs=4)
    cfg.resolved_dns_name = "disorder.tail1234.ts.net"
    store = Store(tmp_path / "s.sqlite")
    run = runner_ok()
    d = FakeDocker()
    ports = Ports(cfg, store, run, fwd_dir=tmp_path / "fwd")
    tpl = T.load_dir(ROOT / "templates")
    m = Manager(cfg, store, d, tpl, runner=run, ports=ports)
    return m


def add_lab(m, lab_id, mem=512, lifetime="disposable", origin="owner", owner="conversation:x", age=0, idle=0,
            ttl_idle=14400, ttl_max=86400, state="ready", owner_use=None):
    now = time.time()
    m.store.put_lab({"lab_id": lab_id, "template": "t", "label": "", "state": state, "owner": owner, "origin": origin,
                     "lineage": "", "lifetime": lifetime, "purpose": "", "ttl_class": "adhoc", "ttl_idle_s": ttl_idle,
                     "ttl_max_s": ttl_max, "egress": "none", "subnet": "10.89.9.0/24", "created_at": now - age,
                     "last_used_at": now - idle, "last_owner_use_at": now if owner_use is None else owner_use,
                     "retire_at": None, "spec_json": dumps({}), "readiness_json": dumps({}), "app_json": None,
                     "agent_api_url": None, "lan": 0})
    m.store.put_member({"lab_id": lab_id, "name": "m1", "ip": "10.89.9.10", "image": "ghost-lab/base:1", "mem_mb": mem,
                        "cpus": 1.0, "cap_add": "[]", "ports_json": "[]", "isolated": 0})


# ---------------------------------------------------------------- errors
def test_every_error_code_has_shape():
    for code, status in STATUS.items():
        e = LabError(code, "msg", hint="h", extra=1)
        assert e.status == status
        assert e.body() == {"error": code, "message": "msg", "hint": "h", "extra": 1}
    with pytest.raises(ValueError):
        LabError("nope", "x")


# ---------------------------------------------------------------- templates
def test_shipped_templates_parse():
    tpl = T.load_dir(ROOT / "templates")
    assert {"blank-dev", "pg-single", "patroni-3", "web", "mysql-single", "python-app"} <= set(tpl)
    assert tpl["pg-single"]["egress"] == "none" and tpl["patroni-3"]["egress"] == "none"
    assert tpl["web"]["egress"] == "tor" and tpl["blank-dev"]["egress"] == "tor"
    assert tpl["patroni-3"]["readiness"]["check"] == "patroni_three_streaming"


def test_params_substitute_and_unknown():
    t = T.parse({"name": "x", "members": [{"name": "a", "image": "i", "mem_mb": 64, "env": {"V": "${p}"}}],
                 "env": {"W": "${p}-${q}"}, "params": {"p": "one", "q": "two"}})
    s = T.substitute(t, {"p": "zz"})
    assert s["env"]["W"] == "zz-two" and s["members"][0]["env"]["V"] == "zz"
    with pytest.raises(LabError) as e:
        T.substitute(t, {"nope": "1"})
    assert e.value.code == "bad_request"


@pytest.mark.parametrize("member,field", [
    ({"name": "a", "image": "i", "mem_mb": 64, "cap_add": ["SYS_ADMIN"]}, "cap_add"),
    ({"name": "a", "image": "i", "mem_mb": 10}, "mem_mb"),
    ({"name": "A!", "image": "i", "mem_mb": 64}, "members"),
    ({"name": "a", "image": "i", "mem_mb": 64, "privileged": True}, "members"),
])
def test_template_rejects(member, field):
    with pytest.raises(LabError) as e:
        T.parse({"name": "x", "members": [member]})
    assert e.value.extra["field"] == field


def test_egress_only_tor_or_none():
    with pytest.raises(LabError):
        T.parse({"name": "x", "egress": "proxy", "members": [{"name": "a", "image": "i", "mem_mb": 64}]})


# ---------------------------------------------------------------- net
def test_subnet_allocation_and_addresses():
    pool = C.Config().pool()
    s = net.allocate_subnet(pool, ["10.89.0.0/24", "10.89.1.0/24"])
    assert s == "10.89.2.0/24"
    assert net.gateway_ip(s) == "10.89.2.2"
    assert net.ip_range(s) == "10.89.2.128/25"
    assert net.next_member_ip(s, []) == "10.89.2.10"
    assert net.next_member_ip(s, ["10.89.2.10"]) == "10.89.2.11"
    # a Docker network overlapping a /24 makes that /24 unusable
    assert net.allocate_subnet(pool, ["10.89.0.0/23"]) == "10.89.2.0/24"


def test_firewall_missing_detects():
    ok = runner_ok(fw_present=True)
    assert net.firewall_missing("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24", runner=ok) == []
    bad = runner_ok(fw_present=False)
    miss = net.firewall_missing("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24", runner=bad)
    # jump + 3 source + 3 interface (review 11-05 #5) + raw + the mangle FORWARD rule
    assert len(miss) == 1 + 3 + 3 + 2 * len(net.PRIVATE_DSTS) + 1


def test_firewall_jump_must_be_first():
    """tailscaled's ts-input above our jump ACCEPTs udp 41641 from the labs (r1 core #19)."""
    import subprocess as sp
    rules = net.expected_fw_rules("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24")

    def run(cmd, **kw):
        t = cmd[cmd.index("-t") + 1]
        out = [r for tb, r in rules if tb == t]
        if t == "filter":
            out = ["-A INPUT -j ts-input"] + out
        return sp.CompletedProcess(cmd, 0, "\n".join(out), "")
    miss = net.firewall_missing("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24", runner=run)
    assert len(miss) == 1 and "not INPUT's first rule" in miss[0]


# ---------------------------------------------------------------- redaction / truncation / readiness
def test_redaction_forms():
    rules = redact.build({"TOKEN": "s3cr3t-value"})
    v = "s3cr3t-value"
    text = " | ".join([v, base64.b64encode(v.encode()).decode(), v.encode().hex(), "s3cr3t\n-value",
                       "s3cr3t%2Dvalue".replace("%2D", "-")])
    out = redact.apply(text, rules)
    assert v not in out and "s3cr3t" not in out
    assert out.count("<secret:TOKEN>") == 5
    assert redact.build({"SHORT": "abc"}) == []


def test_truncation_keeps_tail():
    data, tr = _tail_keep(deque([b"a" * 10, b"b" * 10]), 12)
    assert tr and data == b"a" * 2 + b"b" * 10


def test_patroni_check():
    good = '[{"Role":"Leader","State":"running"},{"Role":"Replica","State":"streaming"},{"Role":"Sync Standby","State":"streaming"}]'
    assert _patroni_three_streaming(good)
    assert not _patroni_three_streaming('[{"Role":"Leader"},{"Role":"Replica","State":"starting"},{"Role":"Replica","State":"streaming"}]')
    assert not _patroni_three_streaming("not json")


# ---------------------------------------------------------------- budget / pre-emption
def test_budget_counts_disposable_pool_only(env):
    m = env
    assert m.budgets() == {"total": 4000, "kept": 1500, "disposable": 2500}
    add_lab(m, "lab_k", mem=1400, lifetime="kept")
    add_lab(m, "lab_a", mem=2000)
    with pytest.raises(LabError) as e:
        m._check_capacity(600, "autopilot")
    assert e.value.code == "budget_exceeded" and e.value.extra["pool"] == "disposable"
    assert e.value.extra["available_mb"] == 500
    assert m._check_capacity(500, "autopilot") == []


def test_max_labs_counts_disposable_only(env):
    m = env
    for i in range(4):
        add_lab(m, f"lab_{i}", mem=100, lifetime="kept" if i == 0 else "disposable")
    assert m._check_capacity(100, "autopilot") == []
    add_lab(m, "lab_x", mem=100)
    with pytest.raises(LabError) as e:
        m._check_capacity(100, "autopilot")
    assert e.value.code == "lab_limit"


def test_owner_preempts_oldest_autopilot_only(env):
    m = env
    add_lab(m, "lab_auto_old", mem=1000, origin="autopilot", age=100)
    add_lab(m, "lab_auto_new", mem=1000, origin="autopilot", age=10)
    add_lab(m, "lab_owner", mem=400, origin="owner", age=200)
    with pytest.raises(LabError):
        m._check_capacity(800, "autopilot")          # autopilot never pre-empts
    ev = m._check_capacity(800, "owner")
    assert [e["lab_id"] for e in ev] == ["lab_auto_old"]
    assert m.store.lab("lab_owner") and m.store.lab("lab_auto_new")
    assert m.store.destroyed("lab_auto_old", 60)["reason"] == "evicted"


def test_owner_cannot_preempt_owner_labs(env):
    m = env
    add_lab(m, "lab_o1", mem=2400, origin="owner")
    with pytest.raises(LabError) as e:
        m._check_capacity(500, "owner")
    assert e.value.code == "budget_exceeded"
    assert m.store.lab("lab_o1")


def test_kept_pool(env):
    m = env
    add_lab(m, "lab_k", mem=1000, lifetime="kept")
    assert m.check_kept_capacity(500) == []
    with pytest.raises(LabError) as e:
        m.check_kept_capacity(501)
    assert e.value.extra["pool"] == "kept"


# ---------------------------------------------------------------- reaper
def test_reaper_decisions(env):
    m = env
    add_lab(m, "lab_idle", idle=20000)
    add_lab(m, "lab_old", age=90000)
    add_lab(m, "lab_stuck", state="creating", age=1000)
    add_lab(m, "lab_fresh")
    add_lab(m, "lab_kept", lifetime="kept", idle=10 ** 7, age=10 ** 7, ttl_idle=0, ttl_max=0)
    acted = {a["lab_id"]: a["reason"] for a in m.reap_once()}
    assert acted == {"lab_idle": "idle_ttl", "lab_old": "max_lifetime", "lab_stuck": "create_failed"}
    assert m.store.lab("lab_kept") and m.store.lab("lab_fresh")


def test_reaper_retires_switched_out_kept(env):
    m = env
    add_lab(m, "lab_v1", lifetime="kept", ttl_idle=0, ttl_max=0)
    m.store.update_lab("lab_v1", retire_at=time.time() - 1)
    assert m.reap_once() == [{"lab_id": "lab_v1", "reason": "retired"}]


# ---------------------------------------------------------------- destroy guard / memory
def test_destroy_guard(env):
    m = env
    add_lab(m, "lab_busy", origin="owner", owner_use=time.time() - 60)
    add_lab(m, "lab_quiet", origin="owner", owner_use=time.time() - 4000)
    add_lab(m, "lab_k", lifetime="kept")
    with pytest.raises(LabError) as e:
        m.destroy("lab_busy", origin="autopilot", by="autopilot")
    assert e.value.code == "owner_active"
    assert m.destroy("lab_quiet", origin="autopilot", by="autopilot")["destroyed"]
    with pytest.raises(LabError) as e:
        m.destroy("lab_k", origin="autopilot", include_kept=True)
    assert e.value.code == "owner_only"
    with pytest.raises(LabError) as e:
        m.destroy("lab_k", origin="owner")
    assert e.value.code == "bad_request"
    assert m.destroy("lab_k", origin="owner", include_kept=True)["destroyed"]


def test_destroyed_memory_and_idempotent(env):
    m = env
    add_lab(m, "lab_gone")
    m.destroy("lab_gone")
    with pytest.raises(LabError) as e:
        m.get("lab_gone")
    assert e.value.code == "unknown_lab" and "was destroyed at" in e.value.message and "deleted" in e.value.message
    again = m.destroy("lab_gone")
    assert again["removed"] == {"containers": 0, "networks": 0, "volumes": 0}
    with pytest.raises(LabError) as e:
        m.get("lab_never")
    assert e.value.code == "unknown_lab" and "was destroyed" not in e.value.message


def test_a_repeat_destroy_still_names_the_leftovers(env):
    """agent §4SX D2 r3: the agent re-asks after a leftover destroy; 'zero' must not hide them."""
    m = env
    add_lab(m, "lab_lo")
    m.destroy("lab_lo")
    assert "leftovers" not in m.destroy("lab_lo")
    m.docker.containers.run("img", name="ghost-lab-lab_lo-m1", labels={"ghost.labs": "1", "ghost.lab.id": "lab_lo"})
    again = m.destroy("lab_lo")
    assert again["destroyed"] is True and again["leftovers"] == ["container:ghost-lab-lab_lo-m1"]


def test_leftovers_are_named_after_the_destroyed_memory_expired(env):
    """agent §4SX D2 r4: a retry a day later must not read 'unknown lab' while its containers remain."""
    m = env
    m.docker.containers.run("img", name="ghost-lab-lab_old-m1", labels={"ghost.labs": "1", "ghost.lab.id": "lab_old"})
    out = m.destroy("lab_old")                       # no row, no destroyed memory
    assert out["leftovers"] == ["container:ghost-lab-lab_old-m1"]
    with pytest.raises(LabError) as e:
        m.destroy("lab_never")
    assert e.value.code == "unknown_lab"


def test_destroy_owner_skips_kept(env):
    m = env
    add_lab(m, "lab_a", owner="project:p1")
    add_lab(m, "lab_b", owner="project:p1", lifetime="kept")
    add_lab(m, "lab_c", owner="project:p2")
    out = m.destroy_owner("project:p1", "owner")
    assert out["destroyed"] == ["lab_a"]
    assert m.store.lab("lab_b") and m.store.lab("lab_c")


# ---------------------------------------------------------------- teardown never removes data volumes
def test_teardown_spares_data_volumes(env):
    m = env
    d = m.docker
    add_lab(m, "lab_t")
    d.volumes.create("labvol", labels={"ghost.labs": "1", "ghost.lab.id": "lab_t"})
    d.volumes.create("ghost-labdata-p-app-m1", labels={"ghost.labs": "1", "ghost.data.lineage": "p"})
    d.containers.run("img", name="ghost-lab-lab_t-m1", labels={"ghost.labs": "1", "ghost.lab.id": "lab_t"})
    d.networks.create("ghost-lab-lab_t", labels={"ghost.labs": "1", "ghost.lab.id": "lab_t"})
    out = m.destroy("lab_t")
    assert out["removed"] == {"containers": 1, "networks": 1, "volumes": 1}
    assert "ghost-labdata-p-app-m1" in d._volumes and "labvol" not in d._volumes
    assert "leftovers" not in out


# ---------------------------------------------------------------- reconcile
def test_reconcile(env):
    m = env
    d = m.docker
    d.containers.run("img", name="orphan", labels={"ghost.labs": "1", "ghost.lab.id": "lab_orph", "ghost.lab.member": "a"})
    d.containers.run("img", name="kept", labels={"ghost.labs": "1", "ghost.lab.id": "lab_kept", "ghost.lab.member": "a",
                                                 "ghost.lab.lifetime": "kept", "ghost.lab.owner": "project:p"})
    d.containers.run("img", name="foreign", labels={"other": "1"})
    d.volumes.create("ghost-labdata-p-a-a", labels={"ghost.labs": "1", "ghost.data.lineage": "p"})
    d.networks.create("ghost-lab-lab_dead", labels={"ghost.labs": "1", "ghost.lab.id": "lab_dead"})
    add_lab(m, "lab_rowonly")
    lines = m.reconcile()
    assert any("destroy orphan lab_orph" in l for l in lines)
    assert any("adopt kept lab_kept" in l for l in lines)
    assert any("drop row lab_rowonly" in l for l in lines)
    assert "orphan" not in d._containers and "foreign" in d._containers and "kept" in d._containers
    assert "ghost-labdata-p-a-a" in d._volumes
    assert "ghost-lab-lab_dead" not in d._networks
    kept = m.store.lab("lab_kept")
    assert kept["lifetime"] == "kept" and kept["purpose"] == "adopted_unreadable"


# ---------------------------------------------------------------- create + rollback
def test_create_rolls_back(env, monkeypatch):
    m = env
    monkeypatch.setattr(m, "tor_state", lambda: "ok")
    monkeypatch.setattr(m, "gateway", lambda start=True: m.docker.containers.run("tor", name="ghost-labs-tor", labels={}))

    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(m, "_create_member", boom)
    with pytest.raises(LabError) as e:
        m.create({"template": "web", "owner": "conversation:c1"}, "owner")
    assert e.value.code == "internal" and "rolled back" in e.value.message
    assert m.store.labs() == []
    assert not [n for n in m.docker._networks if n.startswith("ghost-lab-lab_")]
    assert m.store.q("SELECT * FROM subnets") == [] and m.store.q("SELECT * FROM ports") == []


def test_create_ok_static_ips_ports_ttl_class(env, monkeypatch):
    m = env
    monkeypatch.setattr(m, "tor_state", lambda: "ok")
    gw = m.docker.containers.run("tor", name="ghost-labs-tor", labels={})
    monkeypatch.setattr(m, "gateway", lambda start=True: gw)
    monkeypatch.setattr(m, "apply_route", lambda lab_id, member: True)
    monkeypatch.setattr(m, "_exec", lambda *a, **k: {"exit_code": 0, "stdout": "", "stderr": ""})
    out = m.create({"template": "web", "owner": "project:p1"}, "owner")
    lab = m.store.lab(out["lab_id"])
    assert lab["ttl_class"] == "project" and lab["ttl_idle_s"] == 43200 and lab["state"] == "ready"
    mem = m.store.members(out["lab_id"])[0]
    assert mem["ip"] == net.host_in(lab["subnet"], 10)
    netw = m.docker._networks[net.lab_network_name(out["lab_id"])]
    assert netw.connected[0] == ("ghost-labs-tor", {"ipv4_address": net.gateway_ip(lab["subnet"])})
    p = out["members"][0]["ports"][0]
    assert p["container"] == 80 and p["url"] == f"https://disorder.tail1234.ts.net:{p['host']}/"
    env_file = (m.ports.fwd_dir / f"{p['host']}.env").read_text()
    assert "BIND=127.0.0.1" in env_file and f"TARGET={mem['ip']}:80" in env_file
    # the member was created with gateway DNS, no ipv6, no host binds, not privileged
    c = m.docker._containers[f"ghost-lab-{out['lab_id']}-web1"]
    hc = c.create_kwargs["host_config"]
    assert hc["dns"] == [net.gateway_ip(lab["subnet"])] and hc["binds"] is None and "privileged" not in hc
    assert c.labels["ghost.lab.origin"] == "owner" and c.labels["ghost.lab.owner"] == "project:p1"


def test_create_refused_without_firewall(env):
    m = env
    m.run = runner_ok(fw_present=False)
    with pytest.raises(LabError) as e:
        m.create({"template": "pg-single"}, "owner")
    assert e.value.code == "firewall_missing"


def test_ttl_caps(env, monkeypatch):
    # §4UD G3: create clamps to the class maximum as extend does (it refused; the test pinned the refusal)
    m = env
    monkeypatch.setattr(m, "wait_ready", lambda *a, **k: None)
    out = m.create({"template": "pg-single", "owner": "conversation:c", "ttl_idle_s": 90000}, "owner")
    assert out["clamped_to_s"] == 86400 and m.store.lab(out["lab_id"])["ttl_idle_s"] == 86400
    with pytest.raises(LabError) as e:
        m.create({"template": "pg-single", "owner": "conversation:c", "ttl_idle_s": 0}, "owner")
    assert e.value.extra["field"] == "ttl_idle_s"


def test_bad_owner_and_adhoc_images(env):
    m = env
    with pytest.raises(LabError):
        m.create({"template": "pg-single", "owner": "someone"}, "owner")
    with pytest.raises(LabError) as e:
        m.create({"members": [{"name": "a", "image": "evil/latest", "mem_mb": 128}]}, "owner")
    assert "catalog" in e.value.message


# ---------------------------------------------------------------- review r1 core pins
def test_owner_preemption_infeasible_evicts_nothing(env):
    m = env
    add_lab(m, "lab_auto", mem=500, origin="autopilot")
    add_lab(m, "lab_own", mem=1900, origin="owner")
    with pytest.raises(LabError) as e:
        m._check_capacity(700, "owner")          # 2400-500+700 > 2500 even after evicting all autopilot labs
    assert e.value.code == "budget_exceeded"
    assert m.store.lab("lab_auto")


def test_extend_requires_positive_and_clamps(env):
    m = env
    add_lab(m, "lab_e", ttl_max=1000)
    for bad in ({}, {"ttl_idle_s": 0}, {"ttl_idle_s": -5}, {"ttl_idle_s": "x"}):
        with pytest.raises(LabError):
            m.extend("lab_e", bad, "owner")
    out = m.extend("lab_e", {"ttl_idle_s": 5000}, "owner")
    assert out["clamped_to_s"] == 1000 and m.store.lab("lab_e")["ttl_idle_s"] == 1000
    add_lab(m, "lab_k", lifetime="kept")
    with pytest.raises(LabError):
        m.extend("lab_k", {"ttl_idle_s": 10}, "owner")


def test_reconcile_keeps_kept_rows_and_reaps_half_built_creating(env):
    m = env
    add_lab(m, "lab_k", lifetime="kept")
    m.store.update_lab("lab_k", app_json=dumps({"name": "a", "state": "up"}))
    add_lab(m, "lab_c", state="creating")
    m.docker.containers.run("img", name="c1", labels={"ghost.labs": "1", "ghost.lab.id": "lab_c", "ghost.lab.member": "m1"})
    m.reconcile()
    k = m.store.lab("lab_k")
    assert k and '"containers_missing"' in k["app_json"]
    assert m.store.lab("lab_c")["state"] == "destroying"          # review 11-05 #15
    m.reap_once()
    assert m.store.lab("lab_c") is None


def test_destroy_owner_continues_past_a_refusal(env):
    m = env
    add_lab(m, "lab_a", owner="project:p1", owner_use=time.time())
    add_lab(m, "lab_b", owner="project:p1", owner_use=0)
    out = m.destroy_owner("project:p1", "autopilot")
    assert out["destroyed"] == ["lab_b"] and out["failed"][0]["lab_id"] == "lab_a"
    assert out["failed"][0]["error"] == "owner_active"


def test_retire_refused_while_app_mounted(env):
    m = env
    add_lab(m, "lab_v", lifetime="kept", ttl_idle=0, ttl_max=0)
    m.store.update_lab("lab_v", app_json=dumps({"name": "a", "mounted": True}), retire_at=time.time() - 1)
    with pytest.raises(LabError):
        m.destroy("lab_v", reason="retired", by="system", origin="owner")
    assert m.store.lab("lab_v")


# ---------------------------------------------------------------- §4SX-C detached jobs
def test_start_job_launches_job_script_in_its_workdir_with_the_command_in_env(env, monkeypatch):
    from types import SimpleNamespace
    from ghost_labs.labs import job_script
    add_lab(env, "lab_0000000a")
    seen = {}

    class Ctr:
        status = "running"

        def reload(self):
            pass

        def exec_run(self, cmd, environment=None, **kw):
            seen["cmd"], seen["env"] = cmd, environment
            return SimpleNamespace(exit_code=0, output=b"")
    monkeypatch.setattr(env, "container", lambda lab_id, member: Ctr())
    j = env.start_job("lab_0000000a", "m1", "pgbench -T 5", {"workdir": "/srv/x y", "env": {"A": 1}})
    d = f"/var/log/lab/jobs/{j['job_id']}"
    assert seen["cmd"] == ["sh", "-c", job_script(d, "/srv/x y")]
    assert seen["env"] == {"A": "1", "GHOST_CMD": "pgbench -T 5"}
    assert "cd '/srv/x y'" in seen["cmd"][2]
