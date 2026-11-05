"""Configuration: /etc/ghost-labs/config.yaml, every key optional.

Nothing host-specific is hard-coded (UNIFIED_DESIGN §11): the bind address is
found at runtime from Tailscale, the LAN address from the routing table, and
the RAM budget from /proc/meminfo.
"""
from __future__ import annotations

import ipaddress
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CONFIG_DIR = Path(os.environ.get("GHOST_LABS_CONFIG_DIR", "/etc/ghost-labs"))
STATE_DIR = Path(os.environ.get("GHOST_LABS_STATE_DIR", "/var/lib/ghost-labs"))
OPT_DIR = Path(os.environ.get("GHOST_LABS_OPT_DIR", "/opt/ghost-labs"))

DEFAULT_TTL = {
    "adhoc": {"idle_s": 14400, "max_s": 86400},
    "project": {"idle_s": 43200, "max_s": 259200},
    "probe": {"idle_s": 1800, "max_s": 7200},
}


@dataclass
class Config:
    bind_host: str = "auto"
    bind_port: int = 8700
    ram_budget_mb: Any = "auto"
    kept_mb: int = 1536
    max_labs: int = 4
    reaper_interval_s: int = 60
    published_port_range: List[int] = field(default_factory=lambda: [20000, 20999])
    exec_default_timeout_s: int = 120
    exec_max_timeout_s: int = 1800
    output_cap_bytes: int = 262144
    file_max_bytes: int = 20971520
    address_pool: str = "10.89.0.0/16"
    gw_subnet: str = "10.88.0.0/24"      # ghost-labs-gw (bridge glabs-gw0): the Tor gateway's egress
    api_subnet: str = "10.88.1.0/24"     # ghost-labs-api (bridge glabs-api0): agent-api sidecars' egress
    disk_floor_mb: int = 10240
    ttl: Dict[str, Dict[str, int]] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_TTL)))
    backup_keep: int = 7
    crashloop: Dict[str, int] = field(default_factory=lambda: {"max": 5, "window_s": 600})
    health_interval_s: int = 60
    owner_grace_s: int = 1800
    ts_services: bool = True
    stuck_creating_s: int = 900
    destroyed_memory_s: int = 86400
    # resolved at runtime
    resolved_bind_host: str = ""
    resolved_dns_name: str = ""

    def pool(self) -> ipaddress.IPv4Network:
        return ipaddress.ip_network(self.address_pool)

    def port_range(self) -> range:
        lo, hi = int(self.published_port_range[0]), int(self.published_port_range[1])
        return range(lo, hi + 1)


def load(path: Optional[Path] = None) -> Config:
    path = path or (CONFIG_DIR / "config.yaml")
    data: Dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    cfg = Config()
    for k, v in data.items():
        if not hasattr(cfg, k) or k.startswith("resolved_"):
            raise ValueError(f"unknown config key: {k}")
        if k == "ttl":
            merged = json.loads(json.dumps(DEFAULT_TTL))
            for cls, vals in (v or {}).items():
                if cls not in merged:
                    raise ValueError(f"unknown ttl class: {cls}")
                merged[cls].update(vals or {})
            v = merged
        setattr(cfg, k, v)
    return cfg


def tailscale_status() -> Dict[str, Any]:
    out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"tailscale status failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


def resolve_bind(cfg: Config, status: Optional[Dict[str, Any]] = None) -> None:
    """Fill resolved_bind_host / resolved_dns_name. Never falls back to 0.0.0.0."""
    if cfg.bind_host and cfg.bind_host != "auto":
        cfg.resolved_bind_host = cfg.bind_host
    else:
        status = status if status is not None else tailscale_status()
        ips = [ip for ip in (status.get("Self", {}).get("TailscaleIPs") or []) if ":" not in ip]
        if not ips:
            raise RuntimeError("no Tailscale IPv4 address on this host; refusing to start (never 0.0.0.0)")
        cfg.resolved_bind_host = ips[0]
    if status is None:
        try:
            status = tailscale_status()
        except Exception:  # noqa: BLE001
            status = {}
    cfg.resolved_dns_name = str(status.get("Self", {}).get("DNSName") or "").rstrip(".")
    if cfg.resolved_bind_host in ("0.0.0.0", "::", ""):
        raise RuntimeError("bind_host resolved to a wildcard address; refusing to start")


def lan_ipv4() -> Optional[str]:
    """The host's primary LAN IPv4 (the source address of the default route)."""
    out = subprocess.run(["ip", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True, timeout=5)
    parts = out.stdout.split()
    if "src" in parts:
        ip = parts[parts.index("src") + 1]
        try:
            if ipaddress.ip_address(ip).is_private:
                return ip
        except ValueError:
            return None
    return None


def meminfo() -> Dict[str, int]:
    vals: Dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.strip().split()[0]) // 1024  # MB
    return vals


def ram_budget(cfg: Config, mi: Optional[Dict[str, int]] = None) -> int:
    if cfg.ram_budget_mb != "auto":
        return int(cfg.ram_budget_mb)
    mi = mi or meminfo()
    return max(0, int(mi["MemTotal"] * 0.75) - 1024)


def host_id() -> str:
    p = CONFIG_DIR / "host_id"
    return p.read_text().strip() if p.exists() else ""


def api_key() -> str:
    key = (CONFIG_DIR / "api_key").read_text().strip()
    if not key:
        # review 11-05 #11: an empty key must not open the API
        raise RuntimeError(f"{CONFIG_DIR / 'api_key'} is empty; refusing to start")
    return key
