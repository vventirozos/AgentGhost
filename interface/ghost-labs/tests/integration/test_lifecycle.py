"""v1 §11 integration tests + v2 §10 lifecycle rows, against the real service."""
import base64
import json
import subprocess
import time

import pytest

from .labs_client import Labs, labs, leftovers, wait_for  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dev(labs):  # noqa: F811
    lab = labs.create(members=[{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 256}], egress="none", label="itest-exec")
    yield lab["lab_id"]
    labs.destroy(lab["lab_id"])


def test_exec_exit_codes_stdin_workdir_env(labs, dev):  # noqa: F811
    assert labs.sh(dev, "a", "exit 7")["exit_code"] == 7
    r = labs.sh(dev, "a", "cat; pwd; echo $FOO", stdin="in-data\n", workdir="/tmp", env={"FOO": "bar"})
    assert r["exit_code"] == 0 and r["stdout"] == "in-data\n/tmp\nbar\n", r
    r = labs.sh(dev, "a", "echo err >&2")
    assert r["stderr"] == "err\n" and r["stdout"] == ""


def test_exec_timeout_kills_process_group(labs, dev):  # noqa: F811
    r = labs.sh(dev, "a", "sleep 300 & sleep 301; echo never", timeout_s=3)
    assert r["timed_out"] is True and r["exit_code"] is None and "never" not in r["stdout"]
    time.sleep(1)
    ps = labs.sh(dev, "a", "pgrep -fa 'sleep 30[01]' || true")
    assert ps["stdout"].strip() == "", ps


def test_output_cap_keeps_tail(labs, dev):  # noqa: F811
    r = labs.sh(dev, "a", "seq 1 200000; echo LAST")
    assert r["stdout_truncated"] is True and r["stdout"].endswith("LAST\n") and len(r["stdout"]) <= 262144


def test_detached_job(labs, dev):  # noqa: F811
    st, j = labs.call("POST", f"/v1/labs/{dev}/exec", {"member": "a", "cmd": "echo started; sleep 2; echo done; exit 4",
                                                         "detach": True})
    assert st == 202 and j["job_id"].startswith("job_")
    out = wait_for(lambda: (lambda x: x if x["state"] != "running" else None)(
        labs.ok("GET", f"/v1/labs/{dev}/jobs/{j['job_id']}")), 30, 1, "job end")
    assert out["state"] == "exited" and out["exit_code"] == 4 and "done" in out["output_tail"]
    st, j = labs.call("POST", f"/v1/labs/{dev}/exec", {"member": "a", "cmd": "sleep 600", "detach": True})
    k = labs.ok("DELETE", f"/v1/labs/{dev}/jobs/{j['job_id']}")
    assert k["state"] in ("killed", "exited"), k
    assert labs.sh(dev, "a", "pgrep -f '[s]leep 600' || true")["stdout"].strip() == ""
    jobs = labs.ok("GET", f"/v1/labs/{dev}/jobs")["jobs"]
    assert len(jobs) == 2


def test_a_detached_job_ends_with_its_whole_group(labs, dev):  # noqa: F811
    """§4SX-C live 11-04: a pipeline / backgrounded child outlived its job (liveness reads the leader
    only) — it held a port while the job read `exited`, and no stop reached it."""
    st, j = labs.call("POST", f"/v1/labs/{dev}/exec", {"member": "a", "detach": True,
                                                         "cmd": "(sleep 611 &); (trap '' TERM; exec sleep 612) & "
                                                                "sleep 1 | cat; exit 3"})
    assert st == 202
    out = wait_for(lambda: (lambda x: x if x["state"] != "running" else None)(
        labs.ok("GET", f"/v1/labs/{dev}/jobs/{j['job_id']}")), 30, 1, "job end")
    assert out["state"] == "exited" and out["exit_code"] == 3, out
    assert labs.sh(dev, "a", "pgrep -f '[s]leep 61[12]' || true")["stdout"].strip() == ""     # r2: TERM-deaf too


def test_files_put_get_and_limits(labs, dev):  # noqa: F811
    labs.put(dev, "a", "/srv/x/y/file.txt", b"hello\n", mode="0600")
    r = labs.sh(dev, "a", "stat -c %a /srv/x/y/file.txt; cat /srv/x/y/file.txt")
    assert r["stdout"] == "600\nhello\n"
    g = labs.ok("GET", f"/v1/labs/{dev}/files?member=a&path=/srv/x/y/file.txt")
    assert base64.b64decode(g["content_b64"]) == b"hello\n" and g["bytes"] == 6
    st, out = labs.call("GET", f"/v1/labs/{dev}/files?member=a&path=/nope")
    assert st == 400
    st, out = labs.call("PUT", f"/v1/labs/{dev}/files", {"member": "a", "path": "rel", "content_b64": ""})
    assert st == 400 and out["field"] == "path"


def test_logs_and_unknown_member(labs, dev):  # noqa: F811
    out = labs.ok("GET", f"/v1/labs/{dev}/logs?member=a&tail=20")
    assert "== container stdout/stderr ==" in out["logs"]
    st, out = labs.call("GET", f"/v1/labs/{dev}/logs?member=zz")
    assert st == 404 and out["error"] == "unknown_member" and out["members"] == ["a"]


def test_member_stop_start_keeps_ip_and_route(labs):  # noqa: F811
    lab = labs.create(template="blank-dev", label="itest-restart")
    lid = lab["lab_id"]
    try:
        ip = lab["members"][0]["ip"]
        labs.ok("POST", f"/v1/labs/{lid}/members/dev/stop")
        assert labs.ok("GET", f"/v1/labs/{lid}")["members"][0]["status"] == "exited"
        st, out = labs.call("POST", f"/v1/labs/{lid}/exec", {"member": "dev", "cmd": "true"})
        assert st == 409 and out["error"] == "not_ready"
        labs.ok("POST", f"/v1/labs/{lid}/members/dev/start")
        r = labs.sh(lid, "dev", "ip -4 route show default; hostname -I")
        assert "via" in r["stdout"] and ip in r["stdout"], r
        # a restart through Docker itself (not the API) gets its route back from the events listener
        subprocess.run(["docker", "restart", f"ghost-lab-{lid}-dev"], check=True, capture_output=True)
        wait_for(lambda: "via" in labs.sh(lid, "dev", "ip -4 route show default")["stdout"], 20, 1, "route after docker restart")
    finally:
        labs.destroy(lid)


def test_pg_single_version_18_and_no_leftovers(labs):  # noqa: F811
    lab = labs.create(template="pg-single", label="itest-pg")
    lid = lab["lab_id"]
    assert lab["state"] == "ready", lab["readiness"]
    r = labs.sh(lid, "pg1", "psql -U postgres -h localhost -tAc 'select version()'", env={"PGPASSWORD": "lab"})
    assert "PostgreSQL 18" in r["stdout"], r
    labs.destroy(lid)
    assert leftovers(lid) == []


def test_patroni_failover(labs):  # noqa: F811
    lab = labs.create(template="patroni-3", label="itest-patroni")
    lid = lab["lab_id"]
    try:
        assert lab["state"] == "ready", lab["readiness"]

        def members():
            r = labs.sh(lid, "pg1", "patronictl -c /etc/patroni.yml list --format json", timeout_s=30)
            try:
                return json.loads(r["stdout"])
            except ValueError:
                r = labs.sh(lid, "pg2", "patronictl -c /etc/patroni.yml list --format json", timeout_s=30)
                return json.loads(r["stdout"] or "[]")
        leader = [m["Member"] for m in members() if m["Role"] == "Leader"][0]
        labs.ok("POST", f"/v1/labs/{lid}/members/{leader}/kill")
        other = [m for m in ("pg1", "pg2", "pg3") if m != leader][0]

        def new_leader():
            r = labs.sh(lid, other, "patronictl -c /etc/patroni.yml list --format json", timeout_s=30)
            try:
                rows = json.loads(r["stdout"])
            except ValueError:
                return None
            ls = [m["Member"] for m in rows if m["Role"] == "Leader" and m["State"] == "running"]
            return ls[0] if ls and ls[0] != leader else None
        nl = wait_for(new_leader, 90, 3, "new leader")
        labs.ok("POST", f"/v1/labs/{lid}/members/{leader}/start")

        def rejoined():
            r = labs.sh(lid, nl, "patronictl -c /etc/patroni.yml list --format json", timeout_s=30)
            try:
                rows = json.loads(r["stdout"])
            except ValueError:
                return False
            return any(m["Member"] == leader and m["Role"] in ("Replica", "Sync Standby") and m["State"] == "streaming"
                       for m in rows)
        wait_for(rejoined, 180, 5, "old leader rejoins as replica")
    finally:
        labs.destroy(lid)
    assert leftovers(lid) == []


def test_mysql_and_web_and_http_probe(labs):  # noqa: F811
    db = labs.create(template="mysql-single", label="itest-mysql")
    try:
        assert db["state"] == "ready", db["readiness"]
        r = labs.sh(db["lab_id"], "db1", "mysql -uroot -plab -Nse 'select version()'")
        assert r["stdout"].startswith("8."), r
    finally:
        labs.destroy(db["lab_id"])
    web = labs.create(template="web", label="itest-web2")
    try:
        h = labs.ok("POST", f"/v1/labs/{web['lab_id']}/http", {"member": "web1", "port": 80, "path": "/"})
        assert h["status"] == 200 and "ghost lab web" in h["body_head"]
        st, out = labs.call("POST", f"/v1/labs/{web['lab_id']}/http", {"member": "web1", "path": "x"})
        assert st == 400
        url = web["members"][0]["ports"][0]["url"]
        r = subprocess.run(["curl", "-sS", "-m", "20", url], capture_output=True, text=True)
        assert "ghost lab web" in r.stdout, r
        shot = labs.ok("POST", f"/v1/labs/{web['lab_id']}/screenshot", {"member": "web1", "port": 80, "path": "/"})
        png = base64.b64decode(shot["png_b64"])
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000
        assert subprocess.run("docker ps -aq --filter label=ghost.labs.role=helper", shell=True, capture_output=True,
                              text=True).stdout.strip() == ""
    finally:
        labs.destroy(web["lab_id"])


def test_budget_exceeded_numbers(labs):  # noqa: F811
    h = labs.ok("GET", "/v1/health")
    free = h["ram_budget_mb"] - h["kept_mb"] - h["disposable_mb_used"]
    st, out = labs.call("POST", "/v1/labs", {"members": [{"name": "a", "image": "ghost-lab/base:1", "mem_mb": free + 1}],
                                             "egress": "none", "owner": "conversation:itest"}, origin="autopilot")
    assert st == 409 and out["error"] == "budget_exceeded" and out["pool"] == "disposable"
    assert out["needed_mb"] == free + 1 and out["available_mb"] == free


def test_reaper_gc(labs):  # noqa: F811
    lab = labs.create(template="blank-dev", ttl_idle_s=5, label="itest-ttl")
    time.sleep(7)
    out = labs.ok("POST", "/v1/gc")
    assert {"lab_id": lab["lab_id"], "reason": "idle_ttl"} in out["reaped"]
    assert leftovers(lab["lab_id"]) == []
    st, err = labs.call("GET", f"/v1/labs/{lab['lab_id']}")
    assert st == 404 and "idle_ttl" in err["message"]


def test_owner_active_and_preemption(labs):  # noqa: F811
    mine = labs.create(members=[{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 128}], egress="none")
    try:
        st, out = labs.call("DELETE", f"/v1/labs/{mine['lab_id']}", origin="autopilot")
        assert st == 409 and out["error"] == "owner_active"
    finally:
        labs.destroy(mine["lab_id"])
    h = labs.ok("GET", "/v1/health")
    free = h["ram_budget_mb"] - h["kept_mb"] - h["disposable_mb_used"]
    auto = labs.ok("POST", "/v1/labs", {"members": [{"name": "a", "image": "ghost-lab/base:1", "mem_mb": free - 200}],
                                        "egress": "none", "owner": "conversation:auto"}, origin="autopilot")
    try:
        own = labs.create(members=[{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 400}], egress="none")
        assert own["evicted"] == [{"lab_id": auto["lab_id"], "owner": "conversation:auto"}]
        st, err = labs.call("GET", f"/v1/labs/{auto['lab_id']}")
        assert "evicted" in err["message"]
        labs.destroy(own["lab_id"])
    finally:
        labs.destroy(auto["lab_id"])


def test_firewall_missing_refuses_creates(labs):  # noqa: F811
    rule = ["-i", "glabs-gw0", "-d", "10.0.0.0/8", "-j", "DROP"]
    subprocess.run(["iptables", "-t", "raw", "-D", "PREROUTING", *rule], check=True)
    try:
        st, out = labs.call("POST", "/v1/labs", {"template": "blank-dev", "owner": "conversation:itest"})
        assert st == 503 and out["error"] == "firewall_missing"
        assert labs.ok("GET", "/v1/health")["firewall"] == "missing"
    finally:
        subprocess.run(["systemctl", "restart", "ghost-labs-fw"], check=True)
    labs.ok("POST", "/v1/gc")
    assert labs.ok("GET", "/v1/health")["firewall"] == "ok"
