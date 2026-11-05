"""Lab networks, the shared Tor gateway, member routes and the host firewall
check (contract v2 §2, §11.1, §11.4, §11.5, §11.10; proven in PLAN §3b)."""
from __future__ import annotations

import ipaddress
import subprocess
from typing import Iterable, List, Optional, Set

GW_CONTAINER = "ghost-labs-tor"
GW_NETWORK = "ghost-labs-gw"
API_NETWORK = "ghost-labs-api"
GW_BRIDGE = "glabs-gw0"
API_BRIDGE = "glabs-api0"
GW_HOST_OCTET = 2           # the gateway's fixed address on every lab network
MEMBER_FIRST, MEMBER_LAST = 10, 127   # static member addresses; .128/25 is Docker's dynamic range
PRIVATE_DSTS = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16"]


def lab_network_name(lab_id: str) -> str:
    return f"ghost-lab-{lab_id}"


def allocate_subnet(pool: ipaddress.IPv4Network, used: Iterable[str]) -> str:
    """First free /24 of the pool, never one already in use (sqlite or Docker)."""
    taken: Set[ipaddress.IPv4Network] = set()
    for u in used:
        try:
            taken.add(ipaddress.ip_network(u, strict=False))
        except ValueError:
            continue
    for sub in pool.subnets(new_prefix=24):
        if not any(sub.overlaps(t) for t in taken):
            return str(sub)
    raise RuntimeError("address pool exhausted")


def host_in(subnet: str, octet: int) -> str:
    net = ipaddress.ip_network(subnet)
    return str(net.network_address + octet)


def gateway_ip(subnet: str) -> str:
    return host_in(subnet, GW_HOST_OCTET)


def ip_range(subnet: str) -> str:
    net = ipaddress.ip_network(subnet)
    return str(ipaddress.ip_network(f"{net.network_address + 128}/25"))


def next_member_ip(subnet: str, used: Iterable[str]) -> str:
    u = set(used)
    for o in range(MEMBER_FIRST, MEMBER_LAST + 1):
        ip = host_in(subnet, o)
        if ip not in u:
            return ip
    raise RuntimeError("no free member address in lab subnet")


def nsenter_route(pid: int, gw: Optional[str], runner=subprocess.run) -> None:
    """Point a member's default route at the gateway (gw=None: remove any default route)."""
    if gw:
        cmd = ["nsenter", "-t", str(pid), "-n", "ip", "route", "replace", "default", "via", gw]
    else:
        cmd = ["nsenter", "-t", str(pid), "-n", "ip", "route", "del", "default"]
    r = runner(cmd, capture_output=True, text=True, timeout=10)
    if gw and r.returncode != 0:
        raise RuntimeError(f"route via {gw} failed: {r.stderr.strip()[:200]}")


def default_route(pid: int, runner=subprocess.run) -> str:
    r = runner(["nsenter", "-t", str(pid), "-n", "ip", "-4", "route", "show", "default"],
               capture_output=True, text=True, timeout=10)
    return r.stdout.strip()


FW_CHAIN = "GHOST-LABS-IN"


def expected_fw_rules(pool: str, gw_subnet: str, api_subnet: str) -> List[List[str]]:
    """The rules ghost-labs-fw installs (iptables -S form, table first)."""
    rules: List[List[str]] = [["filter", f"-A INPUT -j {FW_CHAIN}"]]
    for src in (pool, gw_subnet, api_subnet):
        rules.append(["filter", f"-A {FW_CHAIN} -s {src} -m conntrack ! --ctstate RELATED,ESTABLISHED -j DROP"])
    for br in (GW_BRIDGE, API_BRIDGE):
        for dst in PRIVATE_DSTS:
            rules.append(["raw", f"-A PREROUTING -d {dst} -i {br} -j DROP"])  # iptables -S order
    return rules


def firewall_missing(pool: str, gw_subnet: str, api_subnet: str, runner=subprocess.run) -> List[str]:
    """Expected rules that are NOT present, plus the jump if it is not INPUT's FIRST rule
    (tailscaled puts ts-input on top, which ACCEPTs udp 41641 from anywhere). Empty = ok."""
    lines = {}
    for table in ("filter", "raw"):
        r = runner(["iptables", "-t", table, "-S"], capture_output=True, text=True, timeout=10)
        lines[table] = r.stdout.splitlines() if r.returncode == 0 else []
    missing = []
    for table, rule in expected_fw_rules(pool, gw_subnet, api_subnet):
        if rule not in lines[table]:
            missing.append(f"{table}: {rule}")
    first_input = next((l for l in lines["filter"] if l.startswith("-A INPUT ")), "")
    if first_input != f"-A INPUT -j {FW_CHAIN}" and f"filter: -A INPUT -j {FW_CHAIN}" not in missing:
        missing.append(f"filter: -A INPUT -j {FW_CHAIN} is not INPUT's first rule (first: {first_input or 'none'})")
    return missing
