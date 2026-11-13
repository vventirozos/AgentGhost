"""Published ports: host forwarders + Tailscale serve (contract v2 §4, §11.2, §12).

A forwarder is a systemd instance `ghost-labs-fwd@<instance>` that reads
`/etc/ghost-labs/fwd/<instance>.env` (BIND, PORT, TARGET). The loopback
instance is named `<port>`; the optional LAN instance `<port>-lan`. The tailnet
side is `tailscale serve --https=<port>` onto 127.0.0.1:<port>, plus, for kept
apps with a name, `--service=svc:<name>` on 443.
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config as C

Runner = Callable[..., subprocess.CompletedProcess]


class Ports:
    def __init__(self, cfg: C.Config, store, runner: Runner = subprocess.run,
                 fwd_dir: Optional[Path] = None):
        self.cfg, self.store, self.run = cfg, store, runner
        self.fwd_dir = Path(fwd_dir or (C.CONFIG_DIR / "fwd"))
        self._lease_lock = threading.Lock()   # SELECT-then-INSERT must not race (r1 core #1)

    # -- leases ------------------------------------------------------------
    def lease(self, lab_id: str, member: str, cport: int, lease_key: str = "", lan: bool = False) -> int:
        """Return the host port for (lab, member, cport). A non-empty lease_key
        (kept apps: lineage+app) keeps the same port across labs."""
        with self._lease_lock:
            return self._lease(lab_id, member, cport, lease_key, lan)

    def _lease(self, lab_id: str, member: str, cport: int, lease_key: str, lan: bool) -> int:
        if lease_key:
            r = self.store.q("SELECT port FROM ports WHERE lease_key=?", (lease_key,))
            if r:
                port = r[0]["port"]
                self.store.x("UPDATE ports SET lab_id=?, member=?, cport=?, lan=? WHERE port=?",
                             (lab_id, member, cport, int(lan), port))
                return port
        used = {r["port"] for r in self.store.q("SELECT port FROM ports")}
        for port in self.cfg.port_range():
            if port not in used:
                self.store.x("INSERT INTO ports VALUES (?,?,?,?,?,?)", (port, lab_id, member, cport, lease_key, int(lan)))
                return port
        from .errors import LabError
        raise LabError("ports_exhausted", "no free published port", hint="destroy a lab with published ports")

    def leases(self, lab_id: str) -> List[Dict]:
        return self.store.q("SELECT * FROM ports WHERE lab_id=? ORDER BY port", (lab_id,))

    def release_lab(self, lab_id: str, keep_keyed: bool = False) -> int:
        n = 0
        for row in self.leases(lab_id):
            self.unpublish(row["port"])
            if keep_keyed and row["lease_key"]:
                self.store.x("UPDATE ports SET lab_id='' WHERE port=?", (row["port"],))
            else:
                self.store.x("DELETE FROM ports WHERE port=?", (row["port"],))
            n += 1
        return n

    # -- publishing ----------------------------------------------------------
    def _systemctl(self, *args: str) -> subprocess.CompletedProcess:
        return self.run(["systemctl", *args], capture_output=True, text=True, timeout=30)

    def _write_env(self, instance: str, bind: str, port: int, target: str) -> None:
        self.fwd_dir.mkdir(parents=True, exist_ok=True)
        p = self.fwd_dir / f"{instance}.env"
        p.write_text(f"BIND={bind}\nPORT={port}\nTARGET={target}\n")

    def publish(self, port: int, member_ip: str, cport: int, lan: bool = False, tcp: bool = False) -> None:
        """tcp (§4UL: a kept database, any app with no HTTP health path): the tailnet side is a raw TCP forward
        (`tailscale serve --tcp`), reached as host:port — an HTTPS proxy in front of postgres answers nothing."""
        target = f"{member_ip}:{cport}"
        self._write_env(str(port), "127.0.0.1", port, target)
        self._systemctl("enable", "--now", f"ghost-labs-fwd@{port}.service")
        self._systemctl("restart", f"ghost-labs-fwd@{port}.service")
        if lan:
            ip = C.lan_ipv4()
            if ip:
                self._write_env(f"{port}-lan", ip, port, target)
                self._systemctl("enable", "--now", f"ghost-labs-fwd@{port}-lan.service")
                self._systemctl("restart", f"ghost-labs-fwd@{port}-lan.service")
        if tcp:
            # an HTTPS entry left on the port (the app's health changed) would take the port first
            self.run(["tailscale", "serve", f"--https={port}", "off"], capture_output=True, text=True, timeout=30)
            self.run(["tailscale", "serve", "--bg", "--yes", f"--tcp={port}", f"tcp://127.0.0.1:{port}"],
                     capture_output=True, text=True, timeout=30)
            return
        # round 3: a raw TCP entry left on the port (an app whose health went TCP → HTTP, or across a switch)
        self.run(["tailscale", "serve", f"--tcp={port}", "off"], capture_output=True, text=True, timeout=30)
        self.run(["tailscale", "serve", "--bg", "--yes", f"--https={port}", f"http://127.0.0.1:{port}"],
                 capture_output=True, text=True, timeout=30)

    def unpublish(self, port: int) -> None:
        self.run(["tailscale", "serve", f"--https={port}", "off"], capture_output=True, text=True, timeout=30)
        self.run(["tailscale", "serve", f"--tcp={port}", "off"], capture_output=True, text=True, timeout=30)
        for inst in (str(port), f"{port}-lan"):
            self._systemctl("disable", "--now", f"ghost-labs-fwd@{inst}.service")
            try:
                (self.fwd_dir / f"{inst}.env").unlink()
            except FileNotFoundError:
                pass

    def serve_service(self, name: str, port: int) -> bool:
        """True when tailscale took it (round 4: the name fields are set only then)."""
        r = self.run(["tailscale", "serve", f"--service=svc:{name}", "--https=443", "--yes", f"http://127.0.0.1:{port}"],
                     capture_output=True, text=True, timeout=30)
        return getattr(r, "returncode", 0) == 0

    def unserve_service(self, name: str) -> bool:
        """True when the name is no longer served: tailscale took it, or there was nothing to clear (round 4)."""
        r = self.run(["tailscale", "serve", "clear", f"svc:{name}"], capture_output=True, text=True, timeout=30)
        if getattr(r, "returncode", 0) == 0:
            return True
        said = f"{getattr(r, 'stdout', '') or ''} {getattr(r, 'stderr', '') or ''}".lower()
        return any(w in said for w in ("not found", "no such", "does not exist", "not exist", "nothing to"))

    def serve_page(self, local_port: int) -> None:
        """The Apps page on the host's own 443 (v2 §12)."""
        self.run(["tailscale", "serve", "--bg", "--yes", "--https=443", f"http://127.0.0.1:{local_port}"],
                 capture_output=True, text=True, timeout=30)

    def reconcile_serve(self, wanted: List[int]) -> List[str]:
        """Turn off serve entries for ports in our range that no lease wants.
        Only published_port_range is ever touched (443 is the Apps page)."""
        st = self.serve_status()
        rng = self.cfg.port_range()
        removed = []
        for hp, h in (st.get("TCP") or {}).items():
            try:
                port = int(hp)
            except ValueError:
                continue
            if port in rng and port not in wanted:
                flag = f"--tcp={port}" if isinstance(h, dict) and h.get("TCPForward") else f"--https={port}"
                self.run(["tailscale", "serve", flag, "off"], capture_output=True, text=True, timeout=30)
                removed.append(f"serve off {port}")
        return removed

    def url(self, port: int) -> str:
        host = self.cfg.resolved_dns_name or self.cfg.resolved_bind_host
        return f"https://{host}:{port}/"

    def address(self, port: int) -> str:
        """A TCP app's tailnet address (host:port; no scheme — a client such as psql takes it as is)."""
        return f"{self.cfg.resolved_dns_name or self.cfg.resolved_bind_host}:{port}"

    def lan_url(self, port: int) -> Optional[str]:
        ip = C.lan_ipv4()
        return f"http://{ip}:{port}/" if ip else None

    def serve_status(self) -> Dict:
        r = self.run(["tailscale", "serve", "status", "--json"], capture_output=True, text=True, timeout=15)
        try:
            return json.loads(r.stdout or "{}")
        except ValueError:
            return {}
