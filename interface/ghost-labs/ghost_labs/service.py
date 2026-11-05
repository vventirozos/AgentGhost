"""The running service: one Manager plus its background threads (reaper,
Docker events listener, app supervisor). Built once by api.create_app()."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

from . import config as C
from . import templates as T
from .errors import LabError
from .labs import Manager
from .store import Store

log = logging.getLogger("ghost_labs")


class ActionLog:
    """JSON lines in /var/lib/ghost-labs/actions.log (contract v1 §10). Never output, never keys."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, lab_id: str, action: str, outcome: str, duration_ms: int = 0, member: str = "",
              origin: str = "", **extra: Any) -> None:
        row = {"ts": round(time.time(), 3), "lab_id": lab_id, "action": action, "member": member,
               "outcome": outcome, "duration_ms": duration_ms, "origin": origin}
        for k, v in extra.items():
            row[k] = (v[:200] if isinstance(v, str) else v)
        line = json.dumps(row, sort_keys=True)
        log.info(line)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


class Service:
    def __init__(self, cfg: C.Config, client, templates_dir: Path, state_dir: Path, runner=None,
                 start_threads: bool = True):
        import subprocess
        self.cfg = cfg
        self.store = Store(Path(state_dir) / "labs.sqlite")
        self.templates = T.load_dir(templates_dir)
        self.mgr = Manager(cfg, self.store, client, self.templates, runner=runner or subprocess.run)
        self.actions = ActionLog(Path(state_dir) / "actions.log")
        self.mgr.actions = self.actions
        self._stop = threading.Event()
        self.threads: Dict[str, threading.Thread] = {}
        self.kept = None
        try:
            from .kept import Kept
            self.kept = Kept(self.mgr)
            self.mgr.kept = self.kept
        except ImportError:
            pass
        if start_threads:
            self.start()

    # -- startup ---------------------------------------------------------
    def start(self) -> None:
        missing = self.mgr.check_firewall()
        if missing:
            raise RuntimeError("firewall rules missing; refusing to start: " + "; ".join(missing[:3]))
        self.mgr.ensure_infra()
        self.mgr.gateway(start=True)
        for line in self.mgr.reconcile():
            self.actions.write("", "reconcile", line)
        if self.kept:
            self.kept.start_all(reason="service_start")
        self._spawn("reaper", self._reaper_loop)
        self._spawn("events", self._events_loop)
        if self.kept:
            self._spawn("apps", self._apps_loop)

    def _spawn(self, name: str, fn) -> None:
        t = threading.Thread(target=fn, name=f"ghost-labs-{name}", daemon=True)
        t.start()
        self.threads[name] = t

    def stop(self) -> None:
        self._stop.set()

    # -- loops -----------------------------------------------------------
    def _reaper_loop(self) -> None:
        while not self._stop.wait(int(self.cfg.reaper_interval_s)):
            try:
                for a in self.mgr.reap_once():
                    self.actions.write(a["lab_id"], "reap", a["reason"])
                if self.mgr.firewall_state != "ok":
                    log.error("ghost-labs firewall rules are MISSING; creates are refused; re-applying")
                    self.actions.write("", "firewall_missing", "reapply")
                    self.mgr.run([str(C.OPT_DIR / "bin" / "ghost-labs-fw")], capture_output=True, text=True, timeout=60)
                    self.mgr.check_firewall()
            except Exception:  # noqa: BLE001
                log.exception("reaper tick failed")

    def _events_loop(self) -> None:
        """Re-apply routes on container start and network connect (contract v2 §11.10).
        The sweep in the reaper covers events missed while dockerd restarts."""
        while not self._stop.is_set():
            try:
                for ev in self.mgr.docker.events(decode=True, filters={"type": ["container", "network"]}):
                    if self._stop.is_set():
                        return
                    self._on_event(ev)
            except Exception:  # noqa: BLE001
                log.exception("docker events stream broke; reconnecting")
                time.sleep(5)

    def _on_event(self, ev: Dict[str, Any]) -> None:
        action = ev.get("Action") or ev.get("status") or ""
        attrs = (ev.get("Actor") or {}).get("Attributes") or {}
        if ev.get("Type") == "container" and action == "start":
            lab_id, member = attrs.get("ghost.lab.id"), attrs.get("ghost.lab.member")
        elif ev.get("Type") == "network" and action == "connect":
            cid = attrs.get("container", "")
            try:
                c = self.mgr.docker.containers.get(cid)
            except Exception:  # noqa: BLE001
                return
            lab_id, member = (c.labels or {}).get("ghost.lab.id"), (c.labels or {}).get("ghost.lab.member")
        else:
            return
        if not lab_id or not member or not self.store.lab(lab_id):
            return
        try:
            self.mgr.apply_route(lab_id, member)
            if action == "start" and self.kept:
                # its own thread: starting an app waits on the lab lock (up to 60 s) and must not stall events
                threading.Thread(target=self._member_started, args=(lab_id, member), daemon=True,
                                 name=f"ghost-labs-start-{lab_id}").start()
        except LabError as e:
            log.info("event %s for %s/%s deferred: %s", action, lab_id, member, e.message)
        except Exception:  # noqa: BLE001
            log.exception("route apply on event failed for %s/%s", lab_id, member)

    def _member_started(self, lab_id: str, member: str) -> None:
        try:
            self.kept.on_member_start(lab_id, member)
        except LabError as e:
            log.info("app start for %s/%s deferred: %s", lab_id, member, e.message)
        except Exception:  # noqa: BLE001
            log.exception("app start on event failed for %s/%s", lab_id, member)

    def _apps_loop(self) -> None:
        while not self._stop.wait(5):
            try:
                self.kept.tick()
            except Exception:  # noqa: BLE001
                log.exception("app supervisor tick failed")

    # -- health ----------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        from . import __version__
        b = self.mgr.budgets()
        used = self.mgr._used()
        try:
            self.mgr.docker.ping()
            docker = "ok"
        except Exception as e:  # noqa: BLE001
            docker = f"{type(e).__name__}: {e}"[:200]
        mi = C.meminfo()
        self.mgr.check_firewall()   # live, not the last reaper tick
        return {"ok": docker == "ok" and self.mgr.firewall_state == "ok", "version": __version__,
                "host_id": C.host_id(), "ram_budget_mb": b["total"], "ram_used_mb": used["kept"] + used["disposable"],
                "kept_mb": b["kept"], "kept_mb_used": used["kept"], "disposable_mb_used": used["disposable"],
                "mem_available_mb": mi.get("MemAvailable", 0), "disk_free_mb": self.mgr.disk_free_mb(),
                "disk_floor_mb": int(self.cfg.disk_floor_mb),   # agent §4SX D4 r3 #1: what a create needs
                "labs": len(self.mgr._disposable_labs()), "kept_labs": len(self.store.labs()) - len(self.mgr._disposable_labs()),
                "max_labs": self.cfg.max_labs, "docker": docker, "tor": self.mgr.tor_state(),
                "firewall": self.mgr.firewall_state,
                "apps": self.kept.apps_health() if self.kept else [],
                "url_host": self.cfg.resolved_dns_name}
