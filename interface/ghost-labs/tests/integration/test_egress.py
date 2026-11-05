"""PLAN §3b rows + contract v2 §10/§11.12 egress and isolation, all reading data."""
import json
import subprocess

import pytest

from .labs_client import Labs, labs, leftovers  # noqa: F401

pytestmark = pytest.mark.integration
TOR_API = "https://check.torproject.org/api/ip"


def _host_ips():
    lan = subprocess.run(["ip", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True).stdout.split()
    lan_ip = lan[lan.index("src") + 1]
    ts_ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()[0]
    gw = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True).stdout.split()[2]
    return lan_ip, ts_ip, gw


@pytest.fixture(scope="module")
def torlab(labs):  # noqa: F811
    lab = labs.create(template="blank-dev", label="itest-tor")
    yield lab
    labs.destroy(lab["lab_id"])


@pytest.fixture(scope="module")
def nonelab(labs):  # noqa: F811
    lab = labs.create(members=[{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 256},
                               {"name": "b", "image": "ghost-lab/base:1", "mem_mb": 256}], egress="none", label="itest-none")
    yield lab
    labs.destroy(lab["lab_id"])


def test_health(labs):  # noqa: F811
    h = labs.ok("GET", "/v1/health")
    assert h["docker"] == "ok" and h["firewall"] == "ok" and h["tor"] == "ok", h
    assert h["host_id"] and h["url_host"].endswith(".ts.net")


def test_tor_lab_is_tor_without_proxy(labs, torlab):  # noqa: F811
    lid = torlab["lab_id"]
    r = labs.sh(lid, "dev", "env | grep -ci '^[a-z_]*proxy='")
    assert r["stdout"].strip() == "0", r                       # no proxy env at all (v2 §11.9)
    r = labs.sh(lid, "dev", "env | grep -c GHOST_CMD")
    assert r["stdout"].strip() == "0", r
    r = labs.sh(lid, "dev", f"curl -sS -m 60 {TOR_API}")
    assert json.loads(r["stdout"])["IsTor"] is True, r
    r = labs.sh(lid, "dev", f"curl -sS -m 60 --noproxy '*' {TOR_API}")
    assert json.loads(r["stdout"])["IsTor"] is True
    r = labs.sh(lid, "dev", "getent hosts example.com && ip route show default && cat /etc/resolv.conf")
    assert r["exit_code"] == 0 and "via" in r["stdout"] and "ts.net" not in r["stdout"]


def test_tor_lab_cannot_reach_host_lan_tailnet(labs, torlab):  # noqa: F811
    lid = torlab["lab_id"]
    lan_ip, ts_ip, router = _host_ips()
    sub = labs.ok("GET", f"/v1/labs/{lid}")
    bridge_ip = sub["members"][0]["ip"].rsplit(".", 1)[0] + ".1"
    # In a tor lab every TCP connect lands on Tor's TransPort (so the handshake may succeed);
    # what matters is that no byte from the real service ever comes back.
    for host in (bridge_ip, lan_ip, ts_ip, "10.88.0.1"):
        r = labs.sh(lid, "dev", f"timeout 8 bash -c 'exec 3<>/dev/tcp/{host}/22 && head -c 20 <&3'")
        assert r["stdout"] == "" and "SSH" not in r["stdout"], (host, r)
    for url in (f"http://{ts_ip}:8700/v1/health", f"http://{router}/", f"http://{bridge_ip}:8700/"):
        r = labs.sh(lid, "dev", f"curl -sS -m 8 {url}")
        assert r["exit_code"] != 0 and "unauthorized" not in r["stdout"], (url, r)


def test_none_lab_no_dns_no_tcp_but_members_talk(labs, nonelab):  # noqa: F811
    lid = nonelab["lab_id"]
    r = labs.sh(lid, "a", "getent hosts example.com")
    assert r["exit_code"] != 0
    r = labs.sh(lid, "a", "curl -sS -m 8 https://1.1.1.1/ -o /dev/null")
    assert r["exit_code"] != 0
    r = labs.sh(lid, "a", "ping -c1 -W2 b")
    assert r["exit_code"] == 0, r


def test_partition_isolate_rejoin(labs, nonelab):  # noqa: F811
    lid = nonelab["lab_id"]
    before = labs.ok("GET", f"/v1/labs/{lid}")
    ip_b = [m for m in before["members"] if m["name"] == "b"][0]["ip"]
    out = labs.ok("POST", f"/v1/labs/{lid}/members/b/isolate")
    assert out["isolated"] is True
    assert labs.ok("POST", f"/v1/labs/{lid}/members/b/isolate")["isolated"] is True   # idempotent
    assert labs.sh(lid, "a", "ping -c1 -W1 b")["exit_code"] != 0
    out = labs.ok("POST", f"/v1/labs/{lid}/members/b/rejoin")
    assert out["isolated"] is False
    assert labs.sh(lid, "a", "ping -c1 -W2 b")["exit_code"] == 0
    after = labs.ok("GET", f"/v1/labs/{lid}")
    assert [m for m in after["members"] if m["name"] == "b"][0]["ip"] == ip_b


def test_lab_a_cannot_reach_lab_b(labs, torlab, nonelab):  # noqa: F811
    tor_ip = torlab["members"][0]["ip"]
    labs.ok("POST", f"/v1/labs/{torlab['lab_id']}/exec", {"member": "dev", "cmd": "python3 -m http.server 8088", "detach": True})
    r = labs.sh(nonelab["lab_id"], "a", f"curl -sS -m 6 http://{tor_ip}:8088/")
    assert r["exit_code"] != 0
    other = labs.create(template="blank-dev", label="itest-tor2")
    try:
        r = labs.sh(other["lab_id"], "dev", f"curl -sS -m 10 http://{tor_ip}:8088/")
        assert r["exit_code"] != 0, r
        r = labs.sh(torlab["lab_id"], "dev", "curl -sS -m 5 http://127.0.0.1:8088/ -o /dev/null -w '%{http_code}'")
        assert r["stdout"] == "200"
    finally:
        labs.destroy(other["lab_id"])


def test_gateway_netns_cannot_reach_private(labs, torlab):  # noqa: F811
    pid = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", "ghost-labs-tor"], capture_output=True, text=True).stdout.strip()
    lan_ip, ts_ip, router = _host_ips()
    for host, port in ((lan_ip, 22), (ts_ip, 22), ("10.88.0.1", 22), (router, 80)):
        r = subprocess.run(["nsenter", "-t", pid, "-n", "timeout", "6", "bash", "-c", f"exec 3<>/dev/tcp/{host}/{port}"],
                           capture_output=True, text=True)
        assert r.returncode != 0, (host, port)


def test_gateway_restart_stays_tor_only(labs, torlab):  # noqa: F811
    subprocess.run(["docker", "restart", "ghost-labs-tor"], check=True, capture_output=True)
    lid = torlab["lab_id"]
    import time
    deadline = time.time() + 180
    while time.time() < deadline:
        if labs.ok("GET", "/v1/health")["tor"] == "ok":
            break
        time.sleep(3)
    r = labs.sh(lid, "dev", f"curl -sS -m 60 {TOR_API}")
    assert json.loads(r["stdout"])["IsTor"] is True
    lan_ip, _ts, _r = _host_ips()
    r = labs.sh(lid, "dev", f"curl -sS -m 6 http://{lan_ip}:22/")
    assert r["exit_code"] != 0


def test_destroy_leaves_nothing(labs):  # noqa: F811
    lab = labs.create(template="web", label="itest-web")
    lid = lab["lab_id"]
    port = lab["members"][0]["ports"][0]["host"]
    st, out = labs.destroy(lid)
    assert st == 200 and out["destroyed"] and "leftovers" not in out
    assert leftovers(lid) == []
    r = subprocess.run(["systemctl", "is-active", f"ghost-labs-fwd@{port}"], capture_output=True, text=True)
    assert r.stdout.strip() != "active"
    serve = subprocess.run(["tailscale", "serve", "status", "--json"], capture_output=True, text=True).stdout
    assert f'"{port}"' not in serve
