"""Host checks for the review of 2026-11-05 (round 11): what the unit fakes can't prove.

- CRIT 3: a mysql:8 member's anonymous volume (its image's VOLUME /var/lib/mysql) goes with the lab.
- MAJOR 5: lab bridges are glab-<id>; members have no NET_RAW (ping still works); a forged source address
  is dropped by the INPUT interface rule; a NET_ADMIN member routing via the bridge's .1 reaches no tailnet
  peer (the mangle FORWARD rule) and no host service.
"""
import json
import subprocess

import pytest

from .labs_client import labs  # noqa: F401

pytestmark = pytest.mark.integration


def _sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _counter(table, chain, needle):
    """Packets matched so far by the first rule of table/chain whose text contains needle."""
    out = _sh(f"iptables -t {table} -L {chain} -v -x -n").stdout.splitlines()
    for ln in out[2:]:
        if needle in ln:
            return int(ln.split()[0])
    raise AssertionError(f"no {needle!r} rule in {table}/{chain}:\n" + "\n".join(out))


def _peer_ip():
    st = json.loads(_sh("tailscale status --json").stdout or "{}")
    for p in (st.get("Peer") or {}).values():
        if p.get("Online"):
            ips = [ip for ip in p.get("TailscaleIPs") or [] if ":" not in ip]
            if ips:
                return ips[0]
    return None


def test_mysql_destroy_removes_its_anonymous_volume(labs):  # noqa: F811
    db = labs.create(template="mysql-single", label="itest-r11-mysql")
    lid = db["lab_id"]
    try:
        fmt = '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}} {{end}}{{end}}'
        anon = _sh(f"docker inspect -f '{fmt}' ghost-lab-{lid}-db1").stdout.split()
        assert anon, "mysql:8 declares VOLUME /var/lib/mysql: expected one anonymous volume"
    finally:
        st, out = labs.destroy(lid)
    assert st == 200, out
    for v in anon:
        assert _sh(f"docker volume inspect {v}").returncode != 0, f"anonymous volume {v} leaked"


@pytest.fixture(scope="module")
def netadmin(labs):  # noqa: F811
    lab = labs.create(members=[{"name": "a", "image": "ghost-lab/base:1", "mem_mb": 256, "cap_add": ["NET_ADMIN"]},
                               {"name": "b", "image": "ghost-lab/base:1", "mem_mb": 256}],
                      egress="none", label="itest-r11-netadmin")
    yield lab
    labs.destroy(lab["lab_id"])


def test_lab_bridge_is_named_and_listed_by_no_legacy_check(labs, netadmin):  # noqa: F811
    lid = netadmin["lab_id"]
    assert _sh(f"ip link show glab-{lid[4:]}").returncode == 0
    assert lid not in labs.ok("GET", "/v1/health")["legacy_lab_networks"]


def test_members_have_no_net_raw_but_ping_works(labs, netadmin):  # noqa: F811
    lid = netadmin["lab_id"]
    r = labs.sh(lid, "b", "awk '/^CapBnd/ {print $2}' /proc/1/status")
    assert int(r["stdout"].strip(), 16) & (1 << 13) == 0, r          # CAP_NET_RAW is bit 13
    assert labs.sh(lid, "b", "ping -c1 -W2 a")["exit_code"] == 0


def test_a_forged_source_is_dropped_by_the_input_interface_rule(labs, netadmin):  # noqa: F811
    lid = netadmin["lab_id"]
    bridge_ip = netadmin["members"][0]["ip"].rsplit(".", 1)[0] + ".1"
    before = _counter("filter", "GHOST-LABS-IN", "glab-+")
    # a source outside the pool (what -s <pool> could not see), sent at the host's sshd
    labs.sh(lid, "a", "ip addr add 192.0.2.77/32 dev eth0; "
                      f"curl -sS -m 4 --interface 192.0.2.77 telnet://{bridge_ip}:22 </dev/null; "
                      "ip addr del 192.0.2.77/32 dev eth0; true", timeout_s=30)
    assert _counter("filter", "GHOST-LABS-IN", "glab-+") > before


def test_a_net_admin_member_cannot_route_via_the_bridge(labs, netadmin):  # noqa: F811
    lid = netadmin["lab_id"]
    bridge_ip = netadmin["members"][0]["ip"].rsplit(".", 1)[0] + ".1"
    peer = _peer_ip()
    if not peer:
        pytest.skip("no online tailnet peer to aim at")
    before = _counter("mangle", "FORWARD", "glab-+")
    r = labs.sh(lid, "a", f"ip route replace default via {bridge_ip} && ping -c2 -W3 {peer}", timeout_s=30)
    try:
        assert r["exit_code"] != 0 and " 0% packet loss" not in r["stdout"], r
        assert _counter("mangle", "FORWARD", "glab-+") > before
    finally:
        labs.sh(lid, "a", "ip route del default 2>/dev/null; true")
