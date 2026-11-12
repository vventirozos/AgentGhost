"""Kept machines and their apps (contract v2 §5, §6, §11.3/7/8/13/14/16/18, §12).

A kept lab is a disposable lab promoted with /keep. Its members are recreated
with a per-member data volume at /data (labels ghost.data.*, never
ghost.lab.id, so no destroy path removes it). One app per kept lab, run by
the manager inside `app.member`, probed, restarted with backoff, and stopped
as `crashloop` after crashloop.max failures in window_s.
"""
from __future__ import annotations

import base64
import contextlib
import os
import logging
import re
import secrets as _secrets
import shlex
import threading
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C
from . import net
from .errors import LabError
from .store import dumps, loads

log = logging.getLogger("ghost_labs")

APP_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
LINEAGE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SEED_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./-]{0,200}$")
START_GRACE_S = 30
FAILS_TO_RESTART = 3
SIDECAR_OCTET = 3
SIDECAR_PORT = 8000
RUN = "/run/ghost"
ARCHIVE_MEMBER_MAX = 200 * 1024 * 1024
BACKUP_RETRY_S = 3600
MEMBER_DOWN = "the app member's container is not running"
WORKDIR_RE = re.compile(r"^(/[A-Za-z0-9_.-]+)+$")
# workdir is rm -rf'd and replaced on every deploy: never a system tree, /data, or an ancestor of manager paths
WORKDIR_DENY_TREES = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/proc", "/run",
                      "/sbin", "/sys", "/usr", "/var", "/data", "/tmp")
WORKDIR_DENY_EXACT = ("/opt", "/home", "/root", "/srv", "/mnt", "/media", "/opt/venv")


def check_seeds(paths: List[str]) -> List[str]:
    """data_seed paths: relative, normalised (no '', '.', '..' parts), and no path inside another
    (a nested one would be removed through the parent's /data symlink — review r2 #1)."""
    out = sorted(set(str(p) for p in paths))
    for p in out:
        parts = p.split("/")
        if not SEED_RE.match(p) or any(x in ("", ".", "..") for x in parts) or p == ".ghost-seeded":
            raise LabError("bad_request", f"bad data_seed path {p!r} (relative to the app dir, normalised)",
                           field="data_seed")
    for a in out:
        for b in out:
            if a != b and b.startswith(a + "/"):
                raise LabError("bad_request", f"data_seed {b!r} is inside {a!r}; list only {a!r}", field="data_seed")
    return out


def seed_link_cmd(workdir: str, p: str) -> str:
    """Shell for one seed path: the link's parent must resolve inside the workdir (never mv/rm through a
    symlink into /data, checked before anything is created); a symlink in the archive is refused; data
    already in /data is never replaced (the archive's copy only seeds an absent path); a path in neither
    place is an error. Then <workdir>/<p> becomes a link to /data/<p>."""
    q = shlex.quote
    wd, ap, dp = q(workdir), q(f"{workdir}/{p}"), q(f"/data/{p}")
    return (f"w=$(readlink -f {wd}); d=$(readlink -m $(dirname {ap})); "
            f"case \"$d/\" in \"$w\"/*) ;; *) echo \"unsafe link parent $d\" >&2; exit 9;; esac; "
            f"if [ -L {ap} ]; then echo 'seed path is a symlink in the archive' >&2; exit 9; fi; "
            f"mkdir -p $(dirname {ap}) $(dirname {dp}); "
            f"if [ ! -e {dp} ]; then if [ -e {ap} ]; then mv {ap} {dp}; else "
            f"echo 'seed path is in neither the archive nor /data' >&2; exit 8; fi; fi; "
            f"rm -rf {ap}; ln -s {dp} {ap}")


def valid_workdir(wd: str) -> bool:
    if not WORKDIR_RE.match(wd) or any(p in (".", "..") for p in wd.split("/")):
        return False
    if wd in WORKDIR_DENY_EXACT or wd.startswith("/opt/venv/"):
        return False
    return not any(wd == t or wd.startswith(t + "/") for t in WORKDIR_DENY_TREES)


KEEPABLE_STATES = ("ready", "unready")


def data_volume_name(lineage: str, app: str, member: str, lab_id: str = "") -> str:
    base = f"ghost-labdata-{lineage}-{app}-{member}"
    return f"{base}-{lab_id}" if lab_id else base


def _docker_ts(s: str) -> float:
    """Docker's RFC3339 nanosecond time ("2026-11-04T10:00:00.123456789Z") → epoch seconds."""
    from datetime import datetime, timezone
    s = s.strip().rstrip("Z")
    frac = 0.0
    if "." in s:
        s, f = s.split(".", 1)
        f = re.split(r"[+-]", f)[0]
        frac = float("0." + f) if f.isdigit() else 0.0
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() + frac


def _msg(e: BaseException) -> str:
    return e.message if isinstance(e, LabError) else f"{type(e).__name__}: {e}"


def _backoff(n: int) -> int:
    return min(300, 5 * (3 ** max(0, n - 1)))


class Kept:
    def __init__(self, mgr):
        self.mgr = mgr
        self.store = mgr.store
        self.cfg = mgr.cfg
        self.docker = mgr.docker
        self.backup_root = C.STATE_DIR / "backups"
        self.archive_root = C.STATE_DIR / "archives"
        self._last_daily = 0.0
        self._app_locks: Dict[str, threading.Lock] = {}
        self._app_locks_guard = threading.Lock()
        self._backups_running: set = set()
        self._backup_threads: Dict[str, threading.Thread] = {}
        self._backups_guard = threading.Lock()

    @contextlib.contextmanager
    def _app_locked(self, name: str, timeout: float = 120.0):
        """One lock per app NAME (the Tailscale Service name is tailnet-wide): keep, switch, rollback
        and destroy_with_data of any version hold it, so two versions can't both mount (review r2 #8).
        Order: app lock first, then lab locks."""
        with self._app_locks_guard:
            lk = self._app_locks.setdefault(name, threading.Lock())
        if not lk.acquire(timeout=timeout):
            raise LabError("busy", f"another operation on app {name} is running", hint="retry shortly", retry_after_s=10)
        try:
            yield
        finally:
            lk.release()

    # ------------------------------------------------------------ helpers
    def _app(self, lab_id: str) -> Dict[str, Any]:
        row = self.mgr.get(lab_id)
        if row["lifetime"] != "kept" or not row["app_json"]:
            raise LabError("not_kept", f"lab {lab_id} is not kept",
                           hint="keep it first (lab keep with params.app {name, member, port, cmd})")
        return loads(row["app_json"])

    def _log_text(self, text: str, n: int = 200) -> str:
        """actions.log text: redacted FIRST, then cut (review r12 F5: cut first, a secret straddling the cut
        left its head unmatched by the redactor and it was logged as typed)."""
        try:
            return self.mgr.redact_text(str(text))[:n]
        except Exception:  # noqa: BLE001 — a broken redactor must not let the text through
            return "<unredactable>"

    def _save(self, lab_id: str, app: Dict[str, Any]) -> None:
        self.store.update_lab(lab_id, app_json=dumps(app))

    def _env(self, lab_id: str, app: Dict[str, Any]) -> Dict[str, str]:
        row = self.store.lab(lab_id)
        env = dict(self.store.secrets(app.get("lineage") or row["lineage"], app["name"]))
        if row["agent_api_url"]:
            env["GHOST_API"] = f"http://agent-api:{SIDECAR_PORT}"
        return env

    def _sh(self, lab_id: str, member: str, cmd: str, timeout_s: int = 120, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return self.mgr._exec(lab_id, member, cmd, timeout_s=timeout_s, env=env)

    def _must(self, lab_id: str, member: str, cmd: str, what: str, timeout_s: int = 300,
              env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        r = self._sh(lab_id, member, cmd, timeout_s, env)
        if r["exit_code"] != 0:
            raise LabError("internal", f"{what} failed (exit {r['exit_code']}): {(r['stderr'] or r['stdout'])[-600:]}",
                           hint="read the output, fix the archive or requirements, and retry")
        return r

    # ------------------------------------------------------------ keep
    def keep(self, lab_id: str, body: Dict[str, Any], origin: str = "owner") -> Dict[str, Any]:
        name = str((body.get("app") or {}).get("name") or "") if isinstance(body.get("app"), dict) else ""
        with self._app_locked(name):
            return self._keep(lab_id, body, origin)

    def _keep(self, lab_id: str, body: Dict[str, Any], origin: str = "owner") -> Dict[str, Any]:
        row = self.mgr.get(lab_id)
        if row["lifetime"] == "kept":
            raise LabError("already_kept", f"lab {lab_id} is already kept", hint="use deploy / app actions")
        if not row["owner"].startswith("project:"):
            raise LabError("bad_request", "only a project's lab can be kept", field="owner")
        lineage = str(body.get("lineage") or row["lineage"] or row["owner"].split(":", 1)[1])
        if not LINEAGE_RE.match(lineage):
            raise LabError("bad_request", "bad lineage", field="lineage")
        own = row["owner"].split(":", 1)[1]
        if origin != "owner" and lineage != own:
            # agent §4SX D3 r1 H1: only the owner may join another project's lineage (secrets/data/port live there)
            raise LabError("owner_only", f"lineage {lineage!r} is not this lab's project ({own})", field="lineage")
        for l in self.store.labs():
            # agent §4SX D3 code r1 #1: a non-owner joins only a lineage made by its own kind — an owner's kept lab
            # in the lineage (same project or not) means production lives there (data, secrets, the stable URL)
            if (l["lifetime"] == "kept" and l["lineage"] == lineage and origin != "owner"
                    and (l["owner"] != row["owner"] or not self.mgr.may_touch_kept(l, origin))):
                raise LabError("owner_only", f"lineage {lineage!r} holds a kept lab this caller may not touch",
                               field="lineage")
        a = dict(body.get("app") or {})
        name = str(a.get("name") or "")
        if not APP_NAME_RE.match(name):
            raise LabError("bad_request", "app.name must match ^[a-z][a-z0-9-]{0,30}$", field="app.name")
        members = {m["name"]: m for m in self.store.members(lab_id)}
        member = str(a.get("member") or "")
        if member not in members:
            raise LabError("bad_request", "app.member must be one of the lab's members", field="app.member",
                           members=sorted(members))
        port = int(a.get("port") or 0)
        if not (1 <= port <= 65535):
            raise LabError("bad_request", "app.port is required", field="app.port")
        health = dict(a.get("health") or {"http": "/", "tcp": True})
        if health.get("http") is not None and not str(health["http"]).startswith("/"):
            raise LabError("bad_request", "app.health.http must be a path or null", field="app.health")
        data = str(body.get("data") or "shared")
        if data not in ("shared", "fresh"):
            raise LabError("bad_request", "data must be shared|fresh", field="data")
        names = [str(n) for n in (body.get("secrets") or [])]
        values = {str(k): str(v) for k, v in (body.get("secret_values") or {}).items()}
        for n in names + list(values):
            if not SECRET_NAME_RE.match(n):
                raise LabError("bad_request", f"bad secret name {n!r}", field="secrets")
        cron = [str(c) for c in (body.get("cron") or [])]
        for c in cron:
            if "\n" in c or len(c.split()) < 6:
                raise LabError("bad_request", f"bad cron line {c!r}", field="cron")
        backup = body.get("backup")
        if isinstance(backup, str):
            backup = {"cmd": backup, "restore_cmd": None}
        if backup is not None and not (isinstance(backup, dict) and backup.get("cmd")):
            raise LabError("bad_request", "backup must be {cmd, restore_cmd} or null", field="backup")
        agent_api_url = str(body.get("agent_api_url") or row["agent_api_url"] or "")
        if agent_api_url:
            self._validate_agent_api(agent_api_url)
        lan = bool(body.get("lan"))
        workdir = str(a.get("workdir") or "/app")
        if not valid_workdir(workdir):
            raise LabError("bad_request", "app.workdir must be an absolute path outside system dirs and /data "
                           "(it is replaced on every deploy)", field="app.workdir", hint="use e.g. /app or /srv/<name>")
        for l in self.store.labs():
            if l["lifetime"] == "kept" and l["lineage"] != lineage and l["app_json"] \
                    and loads(l["app_json"]).get("name") == name:
                raise LabError("bad_request", f"app name {name!r} is used by project {l['lineage']} (Service names are "
                               "tailnet-wide)", field="app.name", hint="pick another app name")
        other = self._live_holder(lineage, name, lab_id)
        if other:
            oapp = loads(self.store.lab(other)["app_json"])
            if oapp.get("data") != "shared":
                raise LabError("bad_request", f"the live version in {other} keeps its data in a fresh volume, so a "
                               "second version could not take it over", field="data", live_lab=other,
                               hint="destroy or re-keep the live version with data: shared")
            if data != "shared":
                raise LabError("bad_request", f"{name} already runs live in {other}; a second version must use "
                               "data: shared (it is staged, then switched in)", field="data", live_lab=other)
            if member != oapp.get("member"):
                raise LabError("bad_request", f"the live version runs in member {oapp.get('member')!r}; a second "
                               "version must use the same app.member", field="app.member", live_lab=other)
        need = sum(m["mem_mb"] for m in members.values())
        probe_tier = origin == "probe"
        with self.mgr.locked(lab_id, timeout=60):
            # re-read INSIDE the lab lock (review 11-05 #6: two /keep calls on one lab both passed the check
            # made before it, and the second overwrote the first's app and orphaned its secrets)
            cur = self.store.lab(lab_id)
            if cur is None:
                self.mgr.get(lab_id)          # raises unknown_lab
            if cur["lifetime"] == "kept":
                raise LabError("already_kept", f"lab {lab_id} is already kept", hint="use deploy / app actions")
            if cur["state"] not in KEEPABLE_STATES:
                # review r13 C: a lab being torn down (destroying), still building (creating) or failed became
                # production — the reaper never reaps a kept lab, and the next restart called it "unready"
                raise LabError("not_ready", f"lab {lab_id} is {cur['state']}; only a ready or unready lab can be kept",
                               hint="wait until it is ready, or create a new lab", state=cur["state"])
            if probe_tier:
                self._guard_probe_namespace(lineage, name, lab_id)
            if self._live_holder(lineage, name, lab_id) != other:
                raise LabError("busy", "another version of this app changed state meanwhile", hint="retry")
            staged = bool(other)
            app = {"name": name, "lineage": lineage, "member": member, "cmd": str(a.get("cmd") or ""), "port": port, "health": health,
                   "workdir": workdir, "data": data, "secrets": sorted(set(names) | set(values)),
                   "cron": cron, "backup": backup, "lan": lan, "state": "staged" if staged else "stopped",
                   "mounted": not staged, "restarts": 0, "fails": 0, "crash_times": [], "last_probe_at": None,
                   "last_error": None, "started_at": None, "next_restart_at": None, "data_seed": [],
                   "lease_key": f"{lineage}/{name}", "switched_from": None, "last_backup_at": 0,
                   "last_backup_attempt_at": 0}
            # the kept-pool check and the flip to kept under ONE create lock (review 11-05 #6: parallel keeps of
            # different labs each passed the check before either counted, and overfilled the pool)
            # lab lock held, create_lock waited for: allowed by the lock order in Manager.__init__ — a create_lock
            # holder never blocks on a lab lock (eviction only tries it, review r12 F4)
            if not self.mgr.create_lock.acquire(timeout=60):
                raise LabError("busy", "a create or keep holds the budget lock", hint="retry shortly", retry_after_s=10)
            try:
                self.mgr.check_kept_capacity(need, new_containers=False)
                self.store.update_lab(lab_id, lifetime="kept", lineage=lineage, ttl_idle_s=0, ttl_max_s=0, lan=int(lan),
                                      agent_api_url=agent_api_url or None, app_json=dumps(app),
                                      # a probe's kept lab is a test fixture: the reaper retires it (§4SX D3 r1 H2)
                                      **({"retire_at": time.time() + 86400} if str(row.get("origin") or "") == "probe"
                                         else {}))
            finally:
                self.mgr.create_lock.release()
            if probe_tier:
                # what this probe makes in the namespace is its own: its retirement removes exactly that
                self._claim("app", f"{lineage}/{name}", lab_id)
                self._claim("svc", name, lab_id)
            else:
                # a non-probe keep owns the namespace from now on: no probe retirement may delete it (CRIT 2)
                self._drop_claims(lineage, name)
            for n in names:
                if n not in values and n not in self.store.secrets(lineage, name):
                    values[n] = base64.b64encode(_secrets.token_bytes(32)).decode()
            for n, v in values.items():
                self.store.set_secret(lineage, name, n, v)
            if agent_api_url:
                self._ensure_sidecar(lab_id)
            for m in members:
                self._recreate(lab_id, m, mount=app["mounted"])
            self._lease(lab_id, app, move=not staged)
            if app["mounted"]:
                self._install_cron(lab_id, app)
                if app["cmd"] and self._has_code(lab_id, app):
                    self._start_app(lab_id, app, count=False)
            self._save(lab_id, app)
        return self.mgr.detail(lab_id)

    # ------------------------------------------------------------ the probe namespace (review 11-05 CRIT 2)
    def _claim(self, kind: str, key: str, lab_id: str) -> None:
        self.store.x("INSERT OR IGNORE INTO probe_claims VALUES (?,?,?,?)", (kind, key, lab_id, time.time()))

    def _claimed(self, kind: str, key: str) -> bool:
        return bool(self.store.q("SELECT 1 FROM probe_claims WHERE kind=? AND key=?", (kind, key)))

    def _drop_claims(self, lineage: str, app: str) -> None:
        self.store.x("DELETE FROM probe_claims WHERE kind='app' AND key=?", (f"{lineage}/{app}",))
        self.store.x("DELETE FROM probe_claims WHERE kind='svc' AND key=?", (app,))

    def _left_behind(self, lineage: str, app: str, lab_id: str) -> List[str]:
        """What already exists for (lineage, app) and for the Service name svc:<app>, that no probe claimed:
        data volumes, secrets, the stable port lease, backups, archives, another kept version — and, for the
        tailnet-wide name, any lease or kept app of that name in any lineage. A destroy without drop_data
        leaves the data, secrets and lease behind on purpose (contract v2 §11.3), with no live kept lab to say
        whose they are; this is what says it."""
        found: List[str] = []
        key = f"{lineage}/{app}"
        if not self._claimed("app", key):
            try:
                if self.docker.volumes.list(filters={"label": [f"ghost.data.lineage={lineage}", f"ghost.data.app={app}"]}):
                    found.append("data volumes")
            except Exception:  # noqa: BLE001 — can't tell: refuse (fail closed)
                found.append("data volumes (could not be listed)")
            if self.store.secrets(lineage, app):
                found.append("secrets")
            if self.store.q("SELECT 1 FROM ports WHERE lease_key=?", (key,)):
                found.append("the app's port lease")
            if self.store.q("SELECT 1 FROM backups WHERE lineage=? AND app=?", (lineage, app)):
                found.append("backups")
            if self.store.q("SELECT 1 FROM archives WHERE lineage=? AND app=?", (lineage, app)):
                found.append("deploy archives")
            if any(l["lab_id"] != lab_id and l["lifetime"] == "kept" and l["lineage"] == lineage and l["app_json"]
                   and (loads(l["app_json"]) or {}).get("name") == app for l in self.store.labs()):
                found.append("another kept version")
        if not self._claimed("svc", app):
            if any((r["lease_key"] or "").split("/", 1)[-1] == app
                   for r in self.store.q("SELECT lease_key FROM ports WHERE lease_key != ''")):
                found.append(f"the Service name svc:{app}")
            elif any(l["lab_id"] != lab_id and l["lifetime"] == "kept" and l["app_json"]
                     and (loads(l["app_json"]) or {}).get("name") == app for l in self.store.labs()):
                found.append(f"the Service name svc:{app}")
        return found

    def _guard_probe_namespace(self, lineage: str, app: str, lab_id: str) -> None:
        """A probe keeps only into a namespace that is empty or that probes made (review 11-05 CRIT 2: a probe
        took over the owner's destroyed-but-kept shop/web — its /data, secrets, port and Service name — and its
        retirement 24 h later deleted the owner's data)."""
        found = self._left_behind(lineage, app, lab_id)
        if found:
            raise LabError("owner_only", f"{lineage}/{app} holds {', '.join(found)} that a probe did not make",
                           hint="pick another app name (and lineage) for the probe, or ask the owner", field="app.name")

    def _validate_agent_api(self, url: str) -> None:
        import ipaddress
        m = re.match(r"^http://(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})$", url)
        if not m or ipaddress.ip_address(m.group(1)) not in ipaddress.ip_network("100.64.0.0/10"):
            raise LabError("bad_request", "agent_api_url must be http://<tailnet IP literal>:<port>", field="agent_api_url")

    def _volume_exists(self, name: str) -> bool:
        try:
            self.docker.volumes.get(name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _other_live_version(self, lineage: str, app: str, lab_id: str) -> Optional[str]:
        for l in self.store.labs():
            if l["lab_id"] != lab_id and l["lifetime"] == "kept" and l["lineage"] == lineage and l["app_json"]:
                o = loads(l["app_json"])
                if o.get("name") == app and o.get("mounted") and o.get("data") == "shared":
                    return l["lab_id"]
        return None

    def _live_holder(self, lineage: str, app: str, lab_id: str) -> Optional[str]:
        """Another kept lab whose version of (lineage, app) is mounted (live), whatever its data mode."""
        for l in self.store.labs():
            if l["lab_id"] != lab_id and l["lifetime"] == "kept" and l["lineage"] == lineage and l["app_json"]:
                o = loads(l["app_json"])
                if o.get("name") == app and o.get("mounted"):
                    return l["lab_id"]
        return None

    def _volumes_for(self, lab_id: str, member: str, app: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        row = self.store.lab(lab_id)
        name = data_volume_name(row["lineage"], app["name"], member, lab_id if app["data"] == "fresh" else "")
        if not self._volume_exists(name):
            self.docker.volumes.create(name, labels={"ghost.labs": "1", "ghost.data.lineage": row["lineage"],
                                                     "ghost.data.app": app["name"], "ghost.data.member": member})
        return {name: {"bind": "/data", "mode": "rw"}}

    def _data_env(self, image: str) -> Dict[str, str]:
        if image.startswith(("ghost-lab/pg:", "ghost-lab/patroni:")):
            return {"PGDATA": "/data/pg"}
        if image.startswith("ghost-lab/mysql:"):
            return {"MYSQL_DATADIR": "/data/mysql"}
        return {}

    def _recreate(self, lab_id: str, member: str, mount: bool) -> None:
        """Recreate one member from the spec, same static IP, kept labels,
        restart unless-stopped, with (mount) or without its /data volume."""
        row = self.store.lab(lab_id)
        app = loads(row["app_json"])
        spec = loads(row["spec_json"], {})
        sm = next(m for m in spec["members"] if m["name"] == member)
        mem = self.mgr.member(lab_id, member)
        try:
            # with its anonymous volumes (review 11-05 #3: mysql:8's VOLUME leaked one per recreate); kept data
            # is on the named /data volume (PGDATA=/data/pg, datadir=/data/mysql), which this never removes
            self.mgr.remove_container(self.mgr.container(lab_id, member))
        except Exception:  # noqa: BLE001
            pass
        gw = net.gateway_ip(row["subnet"]) if row["egress"] == "tor" else None
        vols = self._volumes_for(lab_id, member, app) if mount else None
        extra = {**(self._data_env(sm["image"]) if mount else {}),
                 **({"GHOST_API": f"http://agent-api:{SIDECAR_PORT}"} if row["agent_api_url"] else {})}
        self.mgr._create_member(lab_id, sm, mem["ip"], gw, spec, row["owner"], row["origin"], lifetime="kept",
                                lineage=row["lineage"], volumes=vols, extra_env=extra)

    def _lease(self, lab_id: str, app: Dict[str, Any], move: bool) -> None:
        row = self.store.lab(lab_id)
        if not move:
            return
        for h in self.store.q("SELECT lab_id FROM ports WHERE lease_key=?", (app["lease_key"],)):
            if h["lab_id"] and h["lab_id"] != lab_id:
                hr = self.store.lab(h["lab_id"])
                if hr and hr["app_json"] and loads(hr["app_json"]).get("mounted"):
                    raise LabError("busy", f"the app's port is held by the live version {h['lab_id']}",
                                   hint="switch to this version instead", live_lab=h["lab_id"])
        # the lab's own per-lab lease for this port is replaced by the stable (lineage, app) lease
        for p in self.mgr.ports.leases(lab_id):
            if p["member"] == app["member"] and p["cport"] == app["port"] and not p["lease_key"]:
                self.mgr.ports.unpublish(p["port"])
                self.store.x("DELETE FROM ports WHERE port=?", (p["port"],))
        port = self.mgr.ports.lease(lab_id, app["member"], app["port"], lease_key=app["lease_key"], lan=app["lan"])
        ip = self.mgr.member(lab_id, app["member"])["ip"]
        self.mgr.ports.publish(port, ip, app["port"], lan=app["lan"])
        if self.cfg.ts_services:
            self.mgr.ports.serve_service(app["name"], port)
        app["host_port"] = port
        app["url"] = self.mgr.ports.url(port)
        app["lan_url"] = self.mgr.ports.lan_url(port) if app["lan"] else None
        name_url = self._name_url(app["name"]) if self.cfg.ts_services else None
        if name_url and app.get("name_url") != name_url:
            # reported only once it answers (v2 §12: propagation takes up to ~1 min; the
            # name must also be defined in the Tailscale console by the owner)
            app["name_url"], app["name_state"] = None, "pending"
            threading.Thread(target=self._verify_name, args=(lab_id, app["name"], name_url), daemon=True).start()
        del row

    def _verify_name(self, lab_id: str, name: str, url: str, tries: int = 24, every: float = 5.0) -> bool:
        import ssl
        import urllib.error
        import urllib.request
        ctx = ssl.create_default_context()
        for _ in range(tries):
            try:
                urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ghost-labs-name-check"}),
                                       timeout=8, context=ctx).close()
                ok = True
            except urllib.error.HTTPError:
                ok = True          # any HTTP answer with a valid certificate means the name is live
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                with self.mgr.locked(lab_id, timeout=120):
                    row = self.store.lab(lab_id)
                    if row and row["app_json"]:
                        a = loads(row["app_json"])
                        if a.get("name") == name:
                            a["name_url"], a["name_state"] = url, "live"
                            self._save(lab_id, a)
                return True
            time.sleep(every)
        with self.mgr.locked(lab_id, timeout=120):
            row = self.store.lab(lab_id)
            if row and row["app_json"]:
                a = loads(row["app_json"])
                if a.get("name") == name and not a.get("name_url"):
                    a["name_state"] = "unreachable"
                    a["name_hint"] = (f"define the Service '{name}' (tcp:443, tag:labs-app) in the Tailscale admin console; "
                                      f"the port link works meanwhile")
                    self._save(lab_id, a)
        return False

    def _name_url(self, name: str) -> Optional[str]:
        dns = self.cfg.resolved_dns_name
        if not dns or "." not in dns:
            return None
        tailnet = dns.split(".", 1)[1]
        return f"https://{name}.{tailnet}/"

    # ------------------------------------------------------------ app process
    def _has_code(self, lab_id: str, app: Dict[str, Any]) -> bool:
        r = self._sh(lab_id, app["member"], f"test -d {shlex.quote(app['workdir'])} && ls -A {shlex.quote(app['workdir'])} | head -1", 20)
        return r["exit_code"] == 0 and bool(r["stdout"].strip())

    def _write_secrets_file(self, lab_id: str, app: Dict[str, Any]) -> None:
        env = self._env(lab_id, app)
        body = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items()))
        c = self.mgr.container(lab_id, app["member"])
        c.exec_run(["mkdir", "-p", RUN])
        self.mgr._put_bytes(c, f"{RUN}/secrets.env", body.encode(), 0o600)

    def _start_app(self, lab_id: str, app: Dict[str, Any], count: bool) -> None:
        if not app["cmd"]:
            app["state"], app["last_error"] = "stopped", "no app.cmd"
            return
        self._stop_proc(lab_id, app)
        self._write_secrets_file(lab_id, app)
        env = {**self._env(lab_id, app), "GHOST_APP_CMD": app["cmd"]}
        wd = shlex.quote(app["workdir"])
        script = (f"mkdir -p {RUN} /var/log/lab; cd {wd} || exit 3; "
                  f"setsid bash -c 'c=\"$GHOST_APP_CMD\"; unset GHOST_APP_CMD; exec bash -lc \"$c\"' >> /var/log/lab/app.log 2>&1 < /dev/null & "
                  f"echo $! > {RUN}/app.pid")
        c = self.mgr.container(lab_id, app["member"])
        r = c.exec_run(["bash", "-c", script], environment=env)
        if r.exit_code != 0:
            app["state"], app["last_error"] = "stopped", f"start failed: {r.output.decode('utf-8', 'replace')[-300:]}"
            return
        if count:
            app["restarts"] = int(app.get("restarts") or 0) + 1
        app.update(state="starting", started_at=time.time(), fails=0, next_restart_at=None)

    def _stop_proc(self, lab_id: str, app: Dict[str, Any]) -> None:
        try:
            c = self.mgr.container(lab_id, app["member"])
            c.exec_run(["bash", "-c", f"p=$(cat {RUN}/app.pid 2>/dev/null) || exit 0; kill -TERM -- -$p 2>/dev/null; "
                                      f"for i in $(seq 1 20); do kill -0 $p 2>/dev/null || break; sleep 0.5; done; "
                                      f"kill -KILL -- -$p 2>/dev/null; rm -f {RUN}/app.pid; true"])
        except Exception:  # noqa: BLE001
            pass

    def _alive(self, lab_id: str, app: Dict[str, Any]) -> bool:
        try:
            c = self.mgr.container(lab_id, app["member"])
            r = c.exec_run(["bash", "-c", f"kill -0 $(cat {RUN}/app.pid 2>/dev/null) 2>/dev/null && echo yes"])
            return r.output.decode().strip() == "yes"
        except Exception:  # noqa: BLE001
            return False

    def probe(self, lab_id: str, app: Dict[str, Any]) -> Dict[str, Any]:
        import http.client
        import socket
        ip = self.mgr.member(lab_id, app["member"])["ip"]
        out: Dict[str, Any] = {"alive": self._alive(lab_id, app), "ok": False}
        h = app.get("health") or {}
        try:
            if h.get("http"):
                conn = http.client.HTTPConnection(ip, app["port"], timeout=5)
                conn.request("GET", h["http"], headers={"User-Agent": "ghost-labs-health"})
                st = conn.getresponse().status
                conn.close()
                out["http_status"] = st
                out["ok"] = 200 <= st < 400
            elif h.get("tcp", True):
                with socket.create_connection((ip, app["port"]), timeout=5):
                    out["ok"] = True
            else:
                out["ok"] = out["alive"]
        except OSError as e:
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        out["ok"] = bool(out["ok"] and out["alive"])
        if not out["ok"]:
            out["why"] = self._why(app, out)
        return out

    @staticmethod
    def _why(app: Dict[str, Any], res: Dict[str, Any]) -> str:
        """One line naming the cause of a failed probe (req a3514318: "alive False, http_status 200" sent the
        agent guessing for 8 turns — the app had no cmd, so there was no process to be alive)."""
        if not res.get("alive"):
            if not str(app.get("cmd") or "").strip():
                return ("the app has no cmd: nothing runs, so it is never up — a kept app is the process labs "
                        "starts in its workdir; another server answering the port does not count")
            return "the app process is not running (it exited or never started) — see the app log (lab logs)"
        if res.get("error"):
            return f"the app runs but its port does not answer: {res['error']}"
        if res.get("http_status") is not None:
            return f"the app runs but the health path answered HTTP {res['http_status']}"
        return "the app runs but the health check failed"

    def wait_up(self, lab_id: str, app: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        res: Dict[str, Any] = {"ok": False}
        while time.time() < deadline:
            res = self.probe(lab_id, app)
            if res["ok"]:
                app.update(state="up", last_probe_at=time.time(), fails=0, last_error=None)
                return res
            if not res["alive"]:
                break
            time.sleep(2)
        app.update(last_probe_at=time.time(), last_error=res.get("why") or res.get("error") or f"probe failed: {res}")
        return res

    def _record_crash(self, lab_id: str, app: Dict[str, Any], why: str) -> None:
        now = time.time()
        win = int(self.cfg.crashloop["window_s"])
        app["crash_times"] = [t for t in app.get("crash_times") or [] if now - t < win] + [now]
        app["last_error"] = why
        if len(app["crash_times"]) >= int(self.cfg.crashloop["max"]):
            self._stop_proc(lab_id, app)
            app.update(state="crashloop", next_restart_at=None)
            log.error("app %s in %s is in crashloop (%s)", app["name"], lab_id, why)
            self.mgr.actions.write(lab_id, "app_crashloop", self._log_text(why))
        else:
            app.update(state="restarting", next_restart_at=now + _backoff(len(app["crash_times"])))
            self.mgr.actions.write(lab_id, "app_restart_scheduled", self._log_text(why))

    # ------------------------------------------------------------ supervisor
    def tick(self) -> None:
        now = time.time()
        for row in self.store.labs():
            if row["lifetime"] != "kept" or not row["app_json"]:
                continue
            lab_id = row["lab_id"]
            lk = self.mgr.lock(lab_id)
            if not lk.acquire(blocking=False):
                continue
            try:
                app = loads(self.store.lab(lab_id)["app_json"])
                st = app.get("state")
                running = self._member_running(lab_id, app) if app.get("mounted") and st not in ("staged", "crashloop") else True
                if running is None:
                    continue
                if not running:
                    # docker errors on every exec otherwise; the start event (or the branch below) brings it back.
                    # Only a running app is marked: another stopped app's last_error is a diagnosis to keep (r3 #7)
                    if st in ("up", "starting", "restarting"):
                        app.update(state="stopped", last_error=MEMBER_DOWN)
                        self._save(lab_id, app)
                        self.mgr.actions.write(lab_id, "app_member_down", MEMBER_DOWN)
                    continue
                if st == "stopped" and app.get("last_error") == MEMBER_DOWN and not app.get("stopped_by_owner"):
                    # the member is running again and the start event was missed
                    self.mgr.apply_route(lab_id, app["member"])
                    if self._has_code(lab_id, app):
                        self._start_app(lab_id, app, count=False)
                    app["last_error"] = None
                    self._save(lab_id, app)
                    self.mgr.actions.write(lab_id, "app_member_back", "member running again: app started")
                    continue
                if st == "restarting" and now >= float(app.get("next_restart_at") or 0):
                    self._start_app(lab_id, app, count=True)
                elif st == "starting":
                    res = self.probe(lab_id, app)
                    app["last_probe_at"] = now
                    if res["ok"]:
                        app.update(state="up", fails=0, last_error=None)
                    elif not res["alive"]:
                        self._exited(lab_id, app, "app exited during start")
                    elif now - float(app.get("started_at") or now) > START_GRACE_S:
                        app["fails"] = int(app.get("fails") or 0) + 1
                        if app["fails"] >= FAILS_TO_RESTART:
                            self._stop_proc(lab_id, app)
                            self._record_crash(lab_id, app, res.get("why") or res.get("error") or "health probe failing after start")
                elif st == "up" and not self._alive(lab_id, app):
                    app["last_probe_at"] = now
                    self._exited(lab_id, app, "app exited")
                elif st == "up" and now - float(app.get("last_probe_at") or 0) >= int(self.cfg.health_interval_s):
                    res = self.probe(lab_id, app)
                    app["last_probe_at"] = now
                    if res["ok"]:
                        app["fails"] = 0
                    elif not res["alive"]:
                        self._exited(lab_id, app, "app exited")
                    else:
                        app["fails"] = int(app.get("fails") or 0) + 1
                        if app["fails"] >= FAILS_TO_RESTART:
                            self._stop_proc(lab_id, app)
                            self._record_crash(lab_id, app, res.get("why") or res.get("error") or f"health probe failed {FAILS_TO_RESTART}×")
                if app.get("mounted") and now - float(app.get("last_backup_at") or 0) > 86400 \
                        and now - float(app.get("last_backup_attempt_at") or 0) > BACKUP_RETRY_S:
                    app["last_backup_attempt_at"] = now
                    # its own thread (review 11-05 #16: a hook may run 1800 s, and run here it stalled crash
                    # detection for every app); it waits for this lock, which the tick releases below
                    self._spawn_daily_backup(lab_id)
                self._save(lab_id, app)
            except LabError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("supervisor tick failed for %s", lab_id)
            finally:
                lk.release()
        try:
            self.ensure_leases()
        except Exception:  # noqa: BLE001
            log.exception("ensure_leases")
        if now - self._last_daily > 86400:
            self._last_daily = now
            self.backup_sqlite()

    def _spawn_daily_backup(self, lab_id: str) -> bool:
        with self._backups_guard:
            if lab_id in self._backups_running:
                return False
            self._backups_running.add(lab_id)
        t = threading.Thread(target=self._daily_backup, args=(lab_id,), daemon=True,
                             name=f"ghost-labs-backup-{lab_id}")
        self._backup_threads[lab_id] = t
        t.start()
        return True

    def join_backups(self, timeout: float = 30.0) -> None:
        """Wait for the running daily backups (tests; a clean shutdown)."""
        for t in list(self._backup_threads.values()):
            t.join(timeout)

    def _daily_backup(self, lab_id: str) -> None:
        """The daily backup, off the supervisor thread. It holds THIS lab's lock for the whole backup (the hook,
        then the copy): while it runs the supervisor skips this one app (its lock is taken non-blocking) and goes
        on supervising every other app. The lock is what keeps a deploy/switch/restore of this lab from running
        into the middle of the copy."""
        try:
            with self.mgr.locked(lab_id, timeout=600):
                row = self.store.lab(lab_id)
                if not row or row["lifetime"] != "kept" or not row["app_json"] or not loads(row["app_json"]).get("mounted"):
                    return
                self._backup_locked(lab_id, "daily")
        except Exception:  # noqa: BLE001
            log.exception("daily backup failed for %s", lab_id)
        finally:
            with self._backups_guard:
                self._backups_running.discard(lab_id)

    def _exited(self, lab_id: str, app: Dict[str, Any], why: str) -> None:
        """The app process is gone. A container restart (reboot, Docker restart, OOM of the
        member) is not an app crash: start it again without counting (backstop for a missed event)."""
        if self._restarted_since(lab_id, app):
            self.mgr.apply_route(lab_id, app["member"])
            self._start_app(lab_id, app, count=False)
            self.mgr.actions.write(lab_id, "app_start_after_member_restart", self._log_text(why))
        else:
            self._record_crash(lab_id, app, why)

    def _member_running(self, lab_id: str, app: Dict[str, Any]) -> Optional[bool]:
        """None: the container doesn't exist (reconcile flags that as containers_missing)."""
        try:
            return self.mgr.container(lab_id, app["member"]).status == "running"
        except Exception:  # noqa: BLE001
            return None

    def _restarted_since(self, lab_id: str, app: Dict[str, Any]) -> bool:
        try:
            st = str(self.mgr.container(lab_id, app["member"]).attrs["State"]["StartedAt"])
            return _docker_ts(st) > float(app.get("started_at") or 0)
        except Exception:  # noqa: BLE001
            return False

    def start_all(self, reason: str) -> None:
        try:
            for line in self.recover_moves():
                log.warning(line)
        except Exception:  # noqa: BLE001
            log.exception("recover_moves")
        try:
            self.ensure_leases()
        except Exception:  # noqa: BLE001
            log.exception("ensure_leases")
        for row in self.store.labs():
            if row["lifetime"] != "kept" or not row["app_json"]:
                continue
            try:
                app = loads(row["app_json"])
                if row["agent_api_url"]:
                    try:
                        self._ensure_sidecar(row["lab_id"])
                    except Exception:  # noqa: BLE001
                        log.exception("sidecar for %s", row["lab_id"])
                self.on_member_start(row["lab_id"], app["member"])
            except Exception:  # noqa: BLE001
                log.exception("start of kept app in %s failed (%s)", row["lab_id"], reason)

    @staticmethod
    def _startable(app: Dict[str, Any], member: str) -> bool:
        return app.get("member") == member and bool(app.get("mounted")) \
            and app.get("state") not in ("staged", "crashloop") and not app.get("stopped_by_owner") \
            and app.get("last_error") != "secrets_lost"

    def on_member_start(self, lab_id: str, member: str) -> None:
        """A kept member started (reboot, Docker restart): start its app without counting a crash."""
        row = self.store.lab(lab_id)
        if not row or row["lifetime"] != "kept" or not row["app_json"]:
            return
        if not self._startable(loads(row["app_json"]), member):
            return
        with self.mgr.locked(lab_id, timeout=60):
            row = self.store.lab(lab_id)
            if not row or not row["app_json"]:
                return
            app = loads(row["app_json"])
            if not self._startable(app, member) or self._alive(lab_id, app):
                return
            self.mgr.apply_route(lab_id, member)
            if self._has_code(lab_id, app):
                self._start_app(lab_id, app, count=False)
                self._save(lab_id, app)

    def apps_health(self) -> List[Dict[str, Any]]:
        out = []
        for row in self.store.labs():
            if row["lifetime"] == "kept" and row["app_json"]:
                a = loads(row["app_json"])
                out.append({"lab_id": row["lab_id"], "app": a.get("name"), "state": a.get("state"),
                            "url": a.get("url"), "name_url": a.get("name_url"), "lan_url": a.get("lan_url"),
                            "restarts": a.get("restarts", 0), "last_error": a.get("last_error")})
        return out

    def app_action(self, lab_id: str, action: str) -> Dict[str, Any]:
        if action not in ("start", "stop", "restart"):
            raise LabError("bad_request", "action must be start|stop|restart", field="action")
        with self.mgr.locked(lab_id, timeout=60):
            app = self._app(lab_id)
            if app.get("state") == "staged":
                raise LabError("not_ready", "this version is staged (not switched in)", hint="switch to it first",
                               state="staged")
            if action == "stop":
                self._stop_proc(lab_id, app)
                app.update(state="stopped", stopped_by_owner=True)
                res = {"ok": False}
            else:
                app.update(crash_times=[], stopped_by_owner=False)
                self._start_app(lab_id, app, count=False)
                res = self.wait_up(lab_id, app, 60)
            self._save(lab_id, app)
        return {"lab_id": lab_id, "app": self._view(app), "probe": res}

    def _view(self, app: Dict[str, Any]) -> Dict[str, Any]:
        return {k: app.get(k) for k in ("name", "member", "state", "port", "host_port", "url", "name_url", "name_state",
                                        "name_hint", "lan_url",
                                        "restarts", "last_probe_at", "last_error", "data", "secrets")}

    def rotate_secrets(self, lab_id: str, values: Dict[str, str]) -> Dict[str, Any]:
        with self.mgr.locked(lab_id, timeout=60):
            app = self._app(lab_id)
            row = self.store.lab(lab_id)
            for k, v in values.items():
                if not SECRET_NAME_RE.match(str(k)):
                    raise LabError("bad_request", f"bad secret name {k!r}", field="secret_values")
                self.store.set_secret(row["lineage"], app["name"], str(k), str(v))
            app["secrets"] = sorted(set(app.get("secrets") or []) | {str(k) for k in values})
            if app.get("state") in ("up", "starting", "restarting"):
                self._start_app(lab_id, app, count=False)
                self.wait_up(lab_id, app, 60)
            self._save(lab_id, app)
        return {"lab_id": lab_id, "secrets": app["secrets"], "app": self._view(app)}

    def _install_cron(self, lab_id: str, app: Dict[str, Any]) -> None:
        if not app.get("cron"):
            return
        body = "".join(f"{line}\n" for line in app["cron"])
        c = self.mgr.container(lab_id, app["member"])
        self.mgr._put_bytes(c, "/etc/cron.d/ghost-app", body.encode(), 0o644)
        c.exec_run(["bash", "-c", "pgrep -x cron >/dev/null || cron"])

    # ------------------------------------------------------------ deploy
    def deploy(self, lab_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        with self.mgr.locked(lab_id, timeout=60):
            app = self._app(lab_id)
            row = self.store.lab(lab_id)
            if body.get("archive_b64"):
                try:
                    data = base64.b64decode(str(body["archive_b64"]), validate=True)
                except ValueError:
                    raise LabError("bad_request", "archive_b64 is not valid base64", field="archive_b64")
                if len(data) > self.cfg.file_max_bytes:
                    raise LabError("bad_request", "archive over file_max_bytes; put it in the member and use archive_member_path",
                                   field="archive_b64")
            elif body.get("archive_member_path"):
                data = self.mgr.read_raw(lab_id, app["member"], str(body["archive_member_path"]),
                                         ARCHIVE_MEMBER_MAX, "200 MB (archive_member_path)")
            else:
                raise LabError("bad_request", "archive_b64 or archive_member_path is required", field="archive_b64")
            req = dict(body.get("requirements") or {})
            seed = check_seeds([str(p) for p in (body.get("data_seed") or [])])
            if app.get("mounted") and app.get("last_backup_at"):
                self._backup_locked(lab_id, "pre_deploy")
                app = self._app(lab_id)
            res = self._deploy_into(lab_id, app, data, req, seed)
            app["archive_ts"] = self._keep_archive(row["lineage"], app["name"], data, req, app.get("data_seed") or seed)
            self._save(lab_id, app)
        return {"lab_id": lab_id, "app": self._view(app), "probe": res}

    def _deploy_into(self, lab_id: str, app: Dict[str, Any], data: bytes, req: Dict[str, Any],
                     seed: List[str], start: bool = True) -> Dict[str, Any]:
        """Validate → unpack to <wd>.new and install deps while the running version keeps
        serving → stop → swap → link data → start. A failure before the swap leaves the
        running app untouched (the staging dir is removed)."""
        m = app["member"]
        q = shlex.quote
        c = self.mgr.container(lab_id, m)
        wd = q(app["workdir"])
        seeded = False
        marked: List[str] = []
        paths: List[str] = []
        if app.get("mounted"):
            mk = self._sh(lab_id, m, "cat /data/.ghost-seeded", 10)
            seeded = mk["exit_code"] == 0
            marked = [ln.strip() for ln in mk["stdout"].splitlines() if SEED_RE.match(ln.strip())] if seeded else []
            known = sorted(marked or app.get("data_seed") or [])     # the marker on /data is authoritative
            if seed and known and sorted(seed) != known:
                raise LabError("data_exists", "data_seed is set on the first deploy only",
                               hint="omit data_seed; the existing seeded paths are re-linked", seeded=known)
            if seed and seeded and not known:
                raise LabError("data_exists", "/data was already seeded", hint="omit data_seed")
            paths = check_seeds(known or seed)
        elif seed:
            # a staged version: its list must be the live version's (seeded paths are fixed for an app)
            live = self._live_holder(self.store.lab(lab_id)["lineage"], app["name"], lab_id) if self.store.lab(lab_id) else None
            fixed = (loads(self.store.lab(live)["app_json"]).get("data_seed") if live else None) or app.get("data_seed")
            if fixed and sorted(seed) != sorted(fixed):
                raise LabError("data_exists", f"data_seed is fixed for this app: {sorted(fixed)}",
                               hint="omit data_seed", seeded=sorted(fixed))
            app["data_seed"] = check_seeds(seed)
        try:
            self.mgr._put_bytes(c, "/tmp/ghost-deploy.tgz", data, 0o600)
            self._must(lab_id, m, f"rm -rf {wd}.new && mkdir -p {wd}.new && tar -xzf /tmp/ghost-deploy.tgz -C {wd}.new",
                       "unpack")
            if req.get("apt"):
                pk = " ".join(q(str(p)) for p in req["apt"])
                self._must(lab_id, m, f"DEBIAN_FRONTEND=noninteractive apt-get update -q && "
                                      f"DEBIAN_FRONTEND=noninteractive apt-get install -y -q {pk}", "apt install", 1800)
            if req.get("pip"):
                pk = " ".join(q(str(p)) for p in req["pip"])
                self._must(lab_id, m, f"[ -x /opt/venv/bin/pip ] || python3 -m venv /opt/venv; /opt/venv/bin/pip install -q {pk}",
                           "pip install", 1800)
            for p in paths:
                # a seed path must exist in /data or in the new archive — checked BEFORE the running app stops (r5 #6)
                r = self._sh(lab_id, m, f"[ -e {q('/data/' + p)} ] || [ -e {q(app['workdir'] + '.new/' + p)} ]", 20)
                if r["exit_code"] != 0:
                    raise LabError("bad_request", f"data_seed path {p!r} is in neither /data nor the archive",
                                   field="data_seed", hint="ship it in the archive (it seeds /data once)")
        except LabError:
            self._sh(lab_id, m, f"rm -rf {wd}.new /tmp/ghost-deploy.tgz", 60)
            raise
        self._stop_proc(lab_id, app)
        try:
            self._must(lab_id, m, f"rm -rf {wd} && mv {wd}.new {wd} && rm -f /tmp/ghost-deploy.tgz", "swap")
            for p in paths:
                self._must(lab_id, m, seed_link_cmd(app["workdir"], p), f"seed and link {p}")
            if paths and sorted(marked) != sorted(paths):
                listing = "".join(f"{p}\n" for p in paths)
                self.mgr._put_bytes(c, "/data/.ghost-seeded", listing.encode(), 0o644)
            if paths:
                app["data_seed"] = paths
            if app.get("mounted"):
                self._install_cron(lab_id, app)       # a staged version runs no cron (r3 #9)
        except Exception as e:
            # the old version is already stopped and replaced: say so instead of leaving "up" (review r2 #10)
            app.update(state="stopped" if app.get("mounted") else "staged",
                       last_error=f"deploy failed after the swap: {_msg(e)[:300]}")
            if self.store.lab(lab_id):
                self._save(lab_id, app)
            raise
        if not start or not app.get("mounted") or app.get("state") == "staged":
            if app.get("state") != "staged":
                app["state"] = "stopped"
            return {"ok": None, "note": "staged: installed, not started (switch to start it)"}
        app.update(crash_times=[], stopped_by_owner=False)
        self._start_app(lab_id, app, count=False)
        return self.wait_up(lab_id, app, 90)

    def _keep_archive(self, lineage: str, app: str, data: bytes, req: Dict[str, Any], seed: List[str]) -> float:
        d = self.archive_root / lineage / app
        d.mkdir(parents=True, exist_ok=True)
        ts = time.time()
        p = d / f"{int(ts * 1000)}-{uuid.uuid4().hex[:8]}.tgz"     # unique: pruning one never hits another
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.store.x("INSERT INTO archives VALUES (?,?,?,?,?,?)", (lineage, app, ts, str(p), dumps(req), dumps(seed)))
        rows = self.store.q("SELECT ts, path FROM archives WHERE lineage=? AND app=? ORDER BY ts DESC", (lineage, app))
        pinned = {ts}
        for l in self.store.labs():
            if l["lineage"] == lineage and l["app_json"]:
                o = loads(l["app_json"])
                if o.get("name") == app and o.get("archive_ts"):
                    pinned.add(float(o["archive_ts"]))
        for r in rows[5:]:
            if float(r["ts"]) in pinned:
                continue
            Path(r["path"]).unlink(missing_ok=True)
            self.store.x("DELETE FROM archives WHERE lineage=? AND app=? AND ts=?", (lineage, app, r["ts"]))
        return ts

    # ------------------------------------------------------------ backups
    def _backup_dir(self, lineage: str, app: str) -> Path:
        d = self.backup_root / lineage / app
        d.mkdir(parents=True, exist_ok=True)
        return d

    def backup(self, lab_id: str, kind: str = "manual") -> Dict[str, Any]:
        with self.mgr.locked(lab_id, timeout=60):
            return self._backup_locked(lab_id, kind)

    def _backup_locked(self, lab_id: str, kind: str, all_members: bool = False, only_member: str = "",
                       restart: bool = True) -> Dict[str, Any]:
        """Tar /data of the app member (all_members: of every member; only_member: of that one).
        A failed copy leaves no partial file and no row. restart=False leaves the app stopped
        (the final backup before a delete: nothing may write after it)."""
        app = self._app(lab_id)
        row = self.store.lab(lab_id)
        if not app.get("mounted"):
            raise LabError("not_ready", "this version has no /data mounted (staged)", state="staged")
        m = app["member"]
        hook = app.get("backup") or None
        was_running = app.get("state") in ("up", "starting")
        if hook:
            self._must(lab_id, m, hook["cmd"], "backup hook", 1800, env=self._env(lab_id, app))
            if not restart:
                self._stop_proc(lab_id, app)          # the final backup: nothing may write after it (r3 #6)
        elif was_running:
            self._stop_proc(lab_id, app)
        members = [only_member] if only_member else \
            [m] + ([x["name"] for x in self.store.members(lab_id) if x["name"] != m] if all_members else [])
        done: List[Dict[str, Any]] = []
        try:
            for mem in members:
                done.append(self._copy_data(lab_id, row["lineage"], app["name"], mem, kind))
        except Exception:
            for d in done:
                Path(d["path"]).unlink(missing_ok=True)
            raise
        finally:
            if not hook and was_running and restart:
                self._start_app(lab_id, app, count=False)
        ts = time.time()
        for d in done:
            self.store.x("INSERT INTO backups VALUES (?,?,?,?,?,?,?)",
                         (d["backup_id"], row["lineage"], app["name"], d["member"], ts, d["path"], kind))
            rows = self.store.q("SELECT backup_id, path FROM backups WHERE lineage=? AND app=? AND kind=? AND member=? "
                                "ORDER BY ts DESC", (row["lineage"], app["name"], kind, d["member"]))
            for r in rows[int(self.cfg.backup_keep):]:
                Path(r["path"]).unlink(missing_ok=True)
                self.store.x("DELETE FROM backups WHERE backup_id=?", (r["backup_id"],))
        app["last_backup_at"] = ts
        self._save(lab_id, app)
        main = done[0]
        return {"lab_id": lab_id, "backup_id": main["backup_id"], "kind": kind, "bytes": main["bytes"], "ts": ts,
                "hook": bool(hook), "members": [{"member": d["member"], "backup_id": d["backup_id"],
                                                 "bytes": d["bytes"]} for d in done]}

    def _copy_data(self, lab_id: str, lineage: str, app_name: str, member: str, kind: str) -> Dict[str, Any]:
        bid = "bk_" + uuid.uuid4().hex[:10]
        path = self._backup_dir(lineage, app_name) / f"{bid}.tar"
        try:
            stream, _ = self.mgr.container(lab_id, member).get_archive("/data")
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                for chunk in stream:
                    f.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {"backup_id": bid, "member": member, "path": str(path), "bytes": path.stat().st_size}

    def backups(self, lab_id: str) -> Dict[str, Any]:
        app = self._app(lab_id)
        row = self.store.lab(lab_id)
        rows = self.store.q("SELECT backup_id, member, ts, kind, path FROM backups WHERE lineage=? AND app=? ORDER BY ts DESC",
                            (row["lineage"], app["name"]))
        return {"lab_id": lab_id, "backups": [{"backup_id": r["backup_id"], "member": r["member"], "ts": r["ts"],
                                               "kind": r["kind"], "bytes": Path(r["path"]).stat().st_size
                                               if Path(r["path"]).exists() else None} for r in rows]}

    def restore(self, lab_id: str, backup_id: str) -> Dict[str, Any]:
        with self.mgr.locked(lab_id, timeout=60):
            return self._restore_locked(lab_id, backup_id)

    def _restore_locked(self, lab_id: str, backup_id: str, snapshot: bool = True, start: bool = True) -> Dict[str, Any]:
        app = self._app(lab_id)
        row = self.store.lab(lab_id)
        if not app.get("mounted"):
            raise LabError("not_ready", "this version has no /data mounted (staged)", state=app.get("state"),
                           hint="restore into the live version")
        rows = self.store.q("SELECT * FROM backups WHERE backup_id=? AND lineage=? AND app=?",
                            (backup_id, row["lineage"], app["name"]))
        if not rows or not Path(rows[0]["path"]).exists():
            raise LabError("unknown_backup", f"no backup {backup_id} for {app['name']}", hint="list backups")
        m = rows[0]["member"]
        if m not in {x["name"] for x in self.store.members(lab_id)}:
            raise LabError("bad_request", f"backup {backup_id} is of member {m!r}, not in this lab", field="backup_id")
        pre = self._backup_locked(lab_id, "pre_restore", only_member=m)["backup_id"] if snapshot else None
        app = self._app(lab_id)
        self._stop_proc(lab_id, app)
        # held from before /data is touched: a crash mid-restore must not restart the app on wiped data (r5 #2);
        # success below releases it
        app.update(state="stopped", stopped_by_owner=True,
                   last_error=f"restore of {backup_id} in progress (pre-restore snapshot {pre})")
        self._save(lab_id, app)
        try:
            self._must(lab_id, m, "find /data -mindepth 1 -maxdepth 1 -exec rm -rf {} +", "clear /data")
            with open(rows[0]["path"], "rb") as f:
                if not self.mgr.container(lab_id, m).put_archive("/", f):
                    raise LabError("internal", "docker refused the backup archive", hint=f"pre-restore snapshot {pre}")
            hook = app.get("backup") or {}
            if hook.get("restore_cmd") and m == app["member"]:
                self._must(lab_id, m, hook["restore_cmd"], "restore hook", 1800, env=self._env(lab_id, app))
        except Exception as e:
            msg = e.message if isinstance(e, LabError) else f"{type(e).__name__}: {e}"
            # held (stopped_by_owner): nothing may restart it on wiped data, not even a reboot (r4 #3)
            app.update(state="stopped", stopped_by_owner=True,
                       last_error=f"restore of {backup_id} failed: {msg[:300]}"
                                  + (f" (pre-restore snapshot {pre})" if pre else "") + "; held: start it after checking /data")
            self._save(lab_id, app)
            if isinstance(e, LabError):
                e.extra["pre_restore"] = pre
                raise
            raise LabError("internal", f"restore of {backup_id} failed: {msg[:300]}", pre_restore=pre) from e
        res: Dict[str, Any] = {"ok": None}
        app["last_error"] = None
        if not start:
            # the process is stopped: a later backup must not restart it (r3 #5); the hold stays until the
            # caller starts the app (rollback: the last restore)
            app["state"] = "stopped"
        if start:
            app.update(crash_times=[], stopped_by_owner=False)
            self._start_app(lab_id, app, count=False)
            res = self.wait_up(lab_id, app, 90)
        self._save(lab_id, app)
        return {"lab_id": lab_id, "restored": backup_id, "pre_restore": pre, "probe": res, "app": self._view(app)}

    def backup_sqlite(self) -> Optional[str]:
        try:
            d = self.backup_root / "_manager"
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"labs-{time.strftime('%Y%m%d')}.sqlite"
            src = sqlite3.connect(str(C.STATE_DIR / "labs.sqlite"))
            out = sqlite3.connect(str(dst))
            with out:
                src.backup(out)
            src.close()
            out.close()
            dst.chmod(0o600)
            for old in sorted(d.glob("labs-*.sqlite"))[:-int(self.cfg.backup_keep)]:
                old.unlink(missing_ok=True)
            return str(dst)
        except Exception:  # noqa: BLE001
            log.exception("manager sqlite backup failed")
            return None

    # ------------------------------------------------------------ rehearse
    def rehearse(self, lab_id: str) -> Dict[str, Any]:
        app = self._app(lab_id)
        row = self.store.lab(lab_id)
        last = self._archive_for(lab_id)
        if not last:
            raise LabError("not_ready", "nothing deployed to this lab yet", hint="deploy first", state=app.get("state"))
        data, req, seed = last
        spec = loads(row["spec_json"], {})
        with self.mgr.locked(lab_id, timeout=60):
            app = self._app(lab_id)
            data, req, seed = self._archive_for(lab_id) or last
            self.store.update_lab(lab_id, state="rehearsing")
            try:
                snap = self._backup_locked(lab_id, "pre_rehearse", all_members=True) if app.get("mounted") else None
                app = self._app(lab_id)
                self._stop_proc(lab_id, app)
                try:
                    for m in self.store.members(lab_id):
                        self._recreate(lab_id, m["name"], mount=bool(app.get("mounted")))
                    self._lease(lab_id, app, move=bool(app.get("mounted")))
                    main = self._deploy_into(lab_id, app, data, req, [], start=True)
                except Exception as e:
                    app.update(state="stopped" if app.get("mounted") else "staged",
                               last_error=f"rehearse failed: {_msg(e)}"
                               + (f" (snapshot {snap['backup_id']})" if snap else ""))
                    self._save(lab_id, app)
                    raise
                self._save(lab_id, app)
            finally:
                self.store.update_lab(lab_id, state="ready")
        twin = self._twin(row, spec, app, data, req)
        return {"lab_id": lab_id, "snapshot": snap, "recreated": main, "twin": twin, "app": self._view(app)}

    def _twin(self, row: Dict[str, Any], spec: Dict[str, Any], app: Dict[str, Any], data: bytes,
              req: Dict[str, Any]) -> Dict[str, Any]:
        body = {"members": [{k: m[k] for k in ("name", "image", "mem_mb", "cpus", "cap_add")} for m in spec["members"]],
                "egress": spec["egress"], "readiness": spec["readiness"], "owner": row["owner"],
                "purpose": f"rehearsal twin of {row['lab_id']}", "label": "twin", "wait": True}
        twin = self.mgr.create(body, "owner", pool="kept")
        tid = twin["lab_id"]
        # the twin never sees the real secrets and runs no cron (it is a throwaway copy)
        tlin = f"twin-{tid}"
        try:
            for n in app.get("secrets") or []:
                self.store.set_secret(tlin, app["name"], n, base64.b64encode(_secrets.token_bytes(32)).decode())
            tapp = {**app, "lineage": tlin, "mounted": False, "state": "stopped", "data_seed": [], "cron": [],
                    "backup": None}
            self._must(tid, tapp["member"], "mkdir -p /data", "twin /data")
            res = self._deploy_into(tid, tapp, data, req, [], start=False)
            tapp["mounted"] = True
            tapp["state"] = "stopped"
            self.store.update_lab(tid, app_json=dumps(tapp))
            self._start_app(tid, tapp, count=False)
            res = self.wait_up(tid, tapp, 90)
            return {"lab_id": tid, "probe": res, "last_error": tapp.get("last_error")}
        except LabError as e:
            return {"lab_id": tid, "probe": {"ok": False}, "error": e.code, "message": e.message}
        finally:
            self.store.x("DELETE FROM secrets WHERE lineage=?", (tlin,))
            self.mgr.destroy(tid, reason="deleted", by="system", origin="owner", _force=True)

    # ------------------------------------------------------------ switch / rollback
    def switch(self, lab_id: str, from_lab: str) -> Dict[str, Any]:
        if not from_lab or from_lab == lab_id:
            raise LabError("bad_request", "from_lab must be the live version's lab id", field="from_lab")
        new, old = self._app(lab_id), self._app(from_lab)
        rn, ro = self.store.lab(lab_id), self.store.lab(from_lab)
        if (rn["lineage"], new["name"]) != (ro["lineage"], old["name"]):
            raise LabError("bad_request", "from_lab runs a different app", field="from_lab")
        if new.get("data") != "shared":
            raise LabError("bad_request", "switch needs data: shared on the new version", field="data")
        if new.get("member") != old.get("member"):
            raise LabError("bad_request", "the new version must run in the same app.member", field="app.member")
        if old.get("data") != "shared":
            raise LabError("bad_request", "the live version uses data: fresh; its data can't move to another version",
                           field="from_lab")
        if not old.get("mounted") and self._live_holder(rn["lineage"], new["name"], lab_id):
            raise LabError("bad_request", "from_lab is not the live version", field="from_lab")
        if new.get("data_seed") and old.get("data_seed") and sorted(new["data_seed"]) != sorted(old["data_seed"]):
            raise LabError("bad_request", f"data_seed {new['data_seed']} differs from the live version's "
                           f"{old['data_seed']}; seeded paths are fixed for an app", field="data_seed")
        last = self._archive_for(lab_id)
        if not last:
            raise LabError("not_ready", "deploy the new version first (it is staged until switch)", state=new.get("state"))
        if not self._archive_for(from_lab):
            raise LabError("not_ready", "the live version has no deploy archive, so a failed switch could not be reversed",
                           hint="deploy the live version once (POST /deploy), then switch")
        first, second = sorted([lab_id, from_lab])
        with self._app_locked(new["name"]), self.mgr.locked(first, timeout=60), self.mgr.locked(second, timeout=60):
            if self._app(lab_id).get("mounted") or (not self._app(from_lab).get("mounted")
                                                    and self._live_holder(rn["lineage"], new["name"], lab_id)):
                raise LabError("busy", "the versions changed state meanwhile", hint="read both labs and retry")
            last = self._archive_for(lab_id) or last          # a deploy may have landed while we waited (r3 #11)
            return self._switch_locked(lab_id, from_lab, last)

    def _move_data(self, to_lab: str, from_lab: str, archive) -> Any:
        """Unmount every member of from_lab, mount every member of to_lab, deploy `archive`
        into to_lab's app member and start it. Returns (probe, to_app). Raises on failure.
        from_lab carries `moving_to` until the move is done: a restart in between is undone by
        recover_moves() (review r2 #6)."""
        fr_app = self._app(from_lab)
        self._stop_proc(from_lab, fr_app)
        fr_app.update(state="staged", mounted=False, moving_to=to_lab)
        self._save(from_lab, fr_app)
        for m in self.store.members(from_lab):
            self._recreate(from_lab, m["name"], mount=False)     # release the shared volumes
        res, to_app = self._mount(to_lab, archive, fr_app.get("data_seed") or [])
        return res, to_app      # from_lab keeps moving_to until the caller has moved the lease (_clear_marks)

    def ensure_leases(self) -> List[str]:
        """The keyed port/name must point at the live (mounted) version. Run at start and every supervisor
        tick: it retries a lease move that failed or was cut by a crash (r5 #3). Cheap: one query when all is well."""
        out = []
        for row in self.store.labs():
            if row["lifetime"] != "kept" or not row["app_json"]:
                continue
            app = loads(row["app_json"])
            if not app.get("mounted") or not app.get("lease_key"):
                continue
            lab_id = row["lab_id"]
            if time.time() < float(app.get("lease_retry_at") or 0):
                continue
            r = self.store.q("SELECT lab_id, member FROM ports WHERE lease_key=?", (app["lease_key"],))
            # lease_pending: a move that failed after the row moved (publish/serve timed out) — the row alone
            # would look fine (r6 #1)
            if r and r[0]["lab_id"] == lab_id and r[0]["member"] == app["member"] and not app.get("lease_pending"):
                continue
            lk = self.mgr.lock(lab_id)
            if not lk.acquire(blocking=False):
                continue
            try:
                a = loads(self.store.lab(lab_id)["app_json"])
                if a.get("mounted"):
                    try:
                        self._lease(lab_id, a, move=True)
                    except Exception as e:  # noqa: BLE001
                        # once a minute, not every tick; the reason is on the app (r6 #4)
                        a.update(lease_pending=True, lease_retry_at=time.time() + 60,
                                 last_error=f"the port/name did not move yet: {_msg(e)[:300]}; retried every minute")
                        self._save(lab_id, a)
                        log.warning("lease for %s not moved yet: %s", lab_id, _msg(e))
                        continue
                    a.pop("lease_pending", None)
                    a.pop("lease_retry_at", None)
                    if "port/name did not move" in (a.get("last_error") or ""):
                        a["last_error"] = None
                    self._save(lab_id, a)
                    out.append(f"lease of {a['name']} moved to {lab_id}")
                    self.mgr.actions.write(lab_id, "lease_moved", out[-1])
            except Exception as e:  # noqa: BLE001
                log.warning("lease check for %s: %s", lab_id, _msg(e))
            finally:
                lk.release()
        return out

    def _lease_or_note(self, lab_id: str, app: Dict[str, Any], what: str) -> Optional[str]:
        """Move the port/name to the live version. A failure here (tailscale or systemctl hanging) does not
        undo a finished switch: it is recorded and the reaper/next deploy retries."""
        try:
            self._lease(lab_id, app, move=True)
            app.pop("lease_pending", None)
            self._save(lab_id, app)
            return None
        except Exception as e:  # noqa: BLE001
            log.exception("lease move to %s", lab_id)
            app["lease_pending"] = True          # ensure_leases retries it even when the ports row already moved
            app["last_error"] = f"{what}, but the port/name did not move: {_msg(e)[:300]}; the supervisor retries it"
            self._save(lab_id, app)
            return _msg(e)

    def _clear_marks(self, *labs: str) -> None:
        for lab in labs:
            row = self.store.lab(lab)
            if row and row["app_json"]:
                a = loads(row["app_json"])
                if a.pop("moving_to", None) is not None:
                    self._save(lab, a)

    def _mount(self, to_lab: str, archive, seeds: List[str]) -> Any:
        to_app = self._app(to_lab)
        to_app.pop("moving_to", None)
        to_app.update(mounted=True, state="stopped", data_seed=to_app.get("data_seed") or list(seeds or []))
        self._save(to_lab, to_app)
        for m in self.store.members(to_lab):
            self._recreate(to_lab, m["name"], mount=True)
        res = self._deploy_into(to_lab, to_app, archive[0], archive[1], [], start=True)
        self._save(to_lab, to_app)
        return res, to_app

    def recover_moves(self) -> List[str]:
        """At start: report a move the process died in. Nothing is moved automatically — every
        automatic choice could land on the version that failed (review r4 #2). The owner picks:
        `switch` brings a version back even when no version is live."""
        out = []
        for row in self.store.labs():
            if row["lifetime"] != "kept" or not row["app_json"]:
                continue
            app = loads(row["app_json"])
            to = app.get("moving_to")
            if not to:
                continue
            lab_id = row["lab_id"]
            msg = (f"a switch/rollback between {lab_id} and {to} was interrupted by a restart; if no version is "
                   f"live, POST /v1/labs/<the version to run>/switch with from_lab=<the other one>")
            for lid in (lab_id, to):
                r = self.store.lab(lid)
                if r and r["app_json"]:
                    a = loads(r["app_json"])
                    # neither version is trusted with the data: both stopped and unmounted (the volumes stay),
                    # neither retired, so `switch` can bring either back (r5 #4/#5)
                    try:
                        self._stop_proc(lid, a)
                    except Exception:  # noqa: BLE001
                        pass
                    a.pop("moving_to", None)
                    prev = a.get("last_error") or ""
                    # a restore cut by the crash: its snapshot id must stay in front of the owner (r6 #2)
                    keep = f" | before the restart: {prev[:300]}" if "restore" in prev else ""
                    a.update(mounted=False, state="staged", last_error=msg + keep)
                    self._save(lid, a)
                    self.store.update_lab(lid, retire_at=self._live_retire(lid))
            out.append(msg)
            self.mgr.actions.write(lab_id, "interrupted_move", self._log_text(msg))
        return out

    def _reverse(self, live_lab: str, other: str, archive, why: str) -> Dict[str, Any]:
        """Put live_lab back on the data after a failed switch/rollback (contract v2 §11.8)."""
        try:
            if not archive:
                raise LabError("internal", "the archive to reverse to is gone")
            res, app = self._move_data(live_lab, other, archive)
            self._lease(live_lab, app, move=True)
            self._save(live_lab, app)
            self._clear_marks(live_lab, other)
            return res
        except Exception as e:  # noqa: BLE001
            # whatever happened, nobody is moving any more: a stale marker would make the next start
            # "recover" onto the version that failed (r3 #1)
            self._clear_marks(live_lab, other)
            log.exception("reverse to %s failed", live_lab)
            row = self.store.lab(live_lab)
            if row and row["app_json"]:
                a = loads(row["app_json"])
                a["last_error"] = f"{why}; reverse failed: {_msg(e)}"
                self._save(live_lab, a)
            self.mgr.actions.write(live_lab, "reverse_failed", self._log_text(_msg(e)))
            return {"ok": False, "error": _msg(e)}

    def _switch_locked(self, lab_id: str, from_lab: str, last) -> Dict[str, Any]:
        self.store.update_lab(lab_id, state="switching")
        try:
            # no version live (an interrupted move): there is nothing mounted to snapshot (r5 #1)
            snap = self._backup_locked(from_lab, "pre_switch", all_members=True) \
                if self._app(from_lab).get("mounted") else {"backup_id": None, "members": []}
            old_last = self._archive_for(from_lab)
            err = None
            try:
                res, new = self._move_data(lab_id, from_lab, last)
            except Exception as e:  # noqa: BLE001
                res, err = {"ok": False, "error": _msg(e)}, _msg(e)
            if res.get("ok"):
                # the switch is done: record it (rollback needs it) before anything else can fail (r4 #4)
                new.pop("moving_to", None)
                new.update(switched_from=from_lab, switch_snapshot=snap["backup_id"],
                           switch_snapshots=[x["backup_id"] for x in snap["members"]])
                self._save(lab_id, new)
                self.store.update_lab(from_lab, retire_at=time.time() + 86400, state="stopped")
                self.store.update_lab(lab_id, retire_at=self._live_retire(lab_id))
                self._clear_marks(from_lab)
                lease_err = self._lease_or_note(lab_id, new, "switched")
                out = {"lab_id": lab_id, "switched": True, "from_lab": from_lab, "probe": res,
                       "snapshot": snap["backup_id"], "app": self._view(new), "rollback_until": time.time() + 86400}
                if lease_err:
                    out["lease_error"] = lease_err
                return out
            if snap["backup_id"] is None:
                # a recovery switch (no version was live): don't pick the other version automatically (r6 #3)
                n = self._app(lab_id)
                n["last_error"] = f"recovery switch failed: {res.get('error') or 'probe failed'}; no version is live"
                self._save(lab_id, n)
                out = {"lab_id": lab_id, "switched": False, "reversed": False, "probe": res, "snapshot": None,
                       "hint": "fix and retry the switch, or switch to the other version"}
                if err:
                    out["error"] = err
                return out
            back = self._reverse(from_lab, lab_id, old_last, f"switch to {lab_id} failed")
            out = {"lab_id": lab_id, "switched": False, "reversed": bool(back.get("ok")), "probe": res,
                   "old_probe": back, "snapshot": snap["backup_id"]}
            if err:
                out["error"] = err
            return out
        finally:
            self._clear_marks(lab_id, from_lab)
            self.store.update_lab(lab_id, state="ready")

    def _archive_for(self, lab_id: str):
        row = self.store.lab(lab_id)
        app = loads(row["app_json"])
        if app.get("archive_ts"):
            r = self.store.q("SELECT * FROM archives WHERE lineage=? AND app=? AND ts=?",
                             (row["lineage"], app["name"], app["archive_ts"]))
            if r and Path(r[0]["path"]).exists():
                return Path(r[0]["path"]).read_bytes(), loads(r[0]["requirements_json"], {}), \
                    loads(r[0]["data_seed_json"], []) or []
        return None

    def _snapshot_ids(self, app: Dict[str, Any]) -> List[str]:
        """The pre-switch snapshot, app member LAST (it is the one the app is started on)."""
        ids = list(app.get("switch_snapshots") or ([app["switch_snapshot"]] if app.get("switch_snapshot") else []))
        main = app.get("switch_snapshot")
        return [i for i in ids if i != main] + ([main] if main in ids else [])

    def rollback(self, lab_id: str, restore_snapshot: bool) -> Dict[str, Any]:
        new = self._app(lab_id)
        from_lab = new.get("switched_from")
        if not from_lab or not self.store.lab(from_lab):
            raise LabError("no_rollback", "no previous version to roll back to", hint="the rollback window is 24 h")
        if not new.get("mounted"):
            raise LabError("no_rollback", "this version is not live", hint="roll back from the live version")
        if not self._archive_for(from_lab):
            raise LabError("no_rollback", "the previous version's archive is gone", hint="deploy it again instead")
        snaps = self._snapshot_ids(new) if restore_snapshot else []
        if restore_snapshot:
            for sid in snaps or [""]:
                r = self.store.q("SELECT path FROM backups WHERE backup_id=?", (sid,))
                if not r or not Path(r[0]["path"]).exists():
                    raise LabError("unknown_backup", f"the switch snapshot {sid or '(none)'} is gone",
                                   hint="roll back with restore_snapshot: false")
        first, second = sorted([lab_id, from_lab])
        with self._app_locked(new["name"]), self.mgr.locked(first, timeout=60), self.mgr.locked(second, timeout=60):
            if not self._app(lab_id).get("mounted"):
                raise LabError("busy", "this version changed state meanwhile", hint="read the lab and retry")
            self.store.update_lab(lab_id, state="switching")
            try:
                old_last, new_last = self._archive_for(from_lab), self._archive_for(lab_id)
                err, pres, note = None, [], None
                try:
                    res, old = self._move_data(from_lab, lab_id, old_last)
                    for i, sid in enumerate(snaps):
                        r = self._restore_locked(from_lab, sid, start=(i == len(snaps) - 1))
                        pres.append(r["pre_restore"])
                        res = r["probe"]
                    old = self._app(from_lab)
                except Exception as e:  # noqa: BLE001
                    res, err = {"ok": False, "error": _msg(e)}, _msg(e)
                    if isinstance(e, LabError) and e.extra.get("pre_restore"):
                        pres.append(e.extra["pre_restore"])
                if res.get("ok"):
                    old.pop("moving_to", None)
                    self._save(from_lab, old)
                    new = self._app(lab_id)
                    new.update(switched_from=None)
                    self._save(lab_id, new)
                    self.store.update_lab(from_lab, retire_at=self._live_retire(from_lab), state="ready")
                    self.store.update_lab(lab_id, retire_at=time.time() + 86400)
                    self._clear_marks(lab_id)
                    old = self._app(from_lab)
                    lease_err = self._lease_or_note(from_lab, old, "rolled back")
                    out = {"lab_id": from_lab, "rolled_back_from": lab_id, "probe": res, "app": self._view(old)}
                    if lease_err:
                        out["lease_error"] = lease_err
                    return out
                # put the data back as it was before any snapshot restore, then the new version back on it
                for pre in reversed([p for p in pres if p]):
                    try:
                        self._restore_locked(from_lab, pre, snapshot=False, start=False)
                    except Exception as e:  # noqa: BLE001
                        note = f"data could not be put back from {pre}: {_msg(e)[:200]}"
                        break
                if note:
                    # /data is wiped or half-restored: starting either version on it could serve (and initialise)
                    # empty data. Both stay stopped for the owner (r3 #2)
                    o = self._app(from_lab)
                    self._stop_proc(from_lab, o)
                    o.update(state="stopped", stopped_by_owner=True, last_error=f"rollback failed and {note}; held; pre-restore snapshots: "
                                                         f"{', '.join(p for p in pres if p)}")
                    self._save(from_lab, o)
                    return {"lab_id": lab_id, "rolled_back": False, "reversed": False, "probe": res, "error": err,
                            "data_note": note, "pre_restore": [p for p in pres if p],
                            "hint": f"restore the pre-restore snapshots into {from_lab} (POST restore), then retry"}
                if pres:
                    note = "the snapshot restore was undone (data is as it was before the rollback)"
                back = self._reverse(lab_id, from_lab, new_last, f"rollback to {from_lab} failed")
                if note:
                    a = self._app(lab_id)
                    a["last_error"] = f"rollback to {from_lab} failed; {note}"
                    self._save(lab_id, a)
                out = {"lab_id": lab_id, "rolled_back": False, "reversed": bool(back.get("ok")), "probe": res,
                       "new_probe": back}
                if err:
                    out["error"] = err
                if note:
                    out["data_note"] = note
                return out
            finally:
                self._clear_marks(lab_id, from_lab)
                self.store.update_lab(lab_id, state="ready")

    # ------------------------------------------------------------ delete with data
    def _live_retire(self, lab_id: str):
        """The retire_at of a version that serves (again): none — except a probe's lab, a test fixture that always
        retires within 24 h (agent §4SX D3 code r1 #2: a switch made it permanent)."""
        r = self.store.lab(lab_id)
        return time.time() + 86400 if r and str(r["origin"] or "") == "probe" else None

    def retire_probe_lab(self, lab_id: str) -> Dict[str, Any]:
        """The reaper's end of a probe-made kept lab (agent §4SX D3 code r1 #6): a test fixture — no final backup
        (agent D4 r2 #2: a backup that failed, or a missing container, wedged the reaper every tick). Its own
        (``fresh``) data always goes; with the last version of the app EVERY data volume of the app (by label: any
        member, any version, shared or fresh — agent D4 r3 #5 / r4 #2), the lease, Service and (the last kept lab of
        the lineage) secrets go too. Under the app lock, briefly (r4: a long deploy holding it makes this tick
        skip, never wait 120 s), and the row is re-read inside it: a switch that made it serve again wins."""
        row = self.mgr.get(lab_id)
        app = loads(row["app_json"]) if row["app_json"] else {}
        if not app.get("name"):
            return self._retire_probe_locked(lab_id)
        with self._app_locked(app["name"], timeout=5.0):
            return self._retire_probe_locked(lab_id)

    def _retire_probe_locked(self, lab_id: str) -> Dict[str, Any]:
        row = self.mgr.get(lab_id)
        ra = row["retire_at"]
        if ra is None or float(ra) > time.time():
            return {"lab_id": lab_id, "skipped": "no longer retiring (it serves again)"}
        app = loads(row["app_json"]) if row["app_json"] else {}
        lineage = row["lineage"]
        members = [m["name"] for m in self.store.members(lab_id)]
        fresh = app.get("data") == "fresh"
        # its own fresh volumes carry its lab id: no other lab can ever have mounted them
        vols = [data_volume_name(lineage, app["name"], m, lab_id) for m in members] if app.get("name") and fresh else []
        out = self.mgr.destroy(lab_id, reason="retired", by="reaper", origin="owner", include_kept=True, _force=True)
        others = [l for l in self.store.labs() if l["lifetime"] == "kept" and l["lineage"] == lineage]
        kept_back: List[str] = []
        if app.get("name") and not [l for l in others if l["app_json"]
                                    and (loads(l["app_json"]) or {}).get("name") == app["name"]]:
            # the last version: the namespace goes with it ONLY where a probe made it (review 11-05 CRIT 2)
            key = app.get("lease_key") or f"{lineage}/{app['name']}"
            if self._claimed("app", f"{lineage}/{app['name']}"):
                for p in self.store.q("SELECT port FROM ports WHERE lease_key=?", (key,)):
                    self.mgr.ports.unpublish(p["port"])
                self.store.x("DELETE FROM ports WHERE lease_key=?", (key,))
                try:
                    vols += [v.name for v in self.docker.volumes.list(filters={"label": [
                        f"ghost.data.lineage={lineage}", f"ghost.data.app={app['name']}"]}) if v.name not in vols]
                except Exception:  # noqa: BLE001
                    log.exception("data volumes of %s/%s not listed", lineage, app["name"])
                self.store.x("DELETE FROM secrets WHERE lineage=? AND app=?", (lineage, app["name"]))
            else:
                kept_back.append(f"{lineage}/{app['name']}: data, secrets and port lease (not made by a probe)")
            if self._claimed("svc", app["name"]):
                if self.cfg.ts_services:
                    try:
                        self.mgr.ports.unserve_service(app["name"])
                    except Exception:  # noqa: BLE001
                        log.exception("unserve %s", app["name"])
            else:
                kept_back.append(f"svc:{app['name']} (not made by a probe)")
            self._drop_claims(lineage, app["name"])
        removed = []
        for v in vols:
            try:
                self.docker.volumes.get(v).remove(force=True)
                removed.append(v)
            except Exception:  # noqa: BLE001
                pass
        if lineage and not others:
            # the lineage's last kept lab: the secrets of every app a probe made in it (and only those)
            for c in self.store.q("SELECT key FROM probe_claims WHERE kind='app'"):
                lin, _, a = c["key"].partition("/")
                if lin == lineage:
                    self.store.x("DELETE FROM secrets WHERE lineage=? AND app=?", (lineage, a))
        out["data_volumes_removed"] = removed
        if kept_back:
            out["kept_back"] = kept_back
        return out

    def destroy_with_data(self, lab_id: str, origin: str) -> Dict[str, Any]:
        row = self.mgr.get(lab_id)
        if not self.mgr.may_touch_kept(row, origin):
            raise LabError("owner_only", "only the owner may drop a kept lab's data", hint="ask the owner")
        if row["lifetime"] != "kept" or not row["app_json"]:
            return self.mgr.destroy(lab_id, reason="deleted", by=origin, origin=origin, include_kept=True)
        a = loads(row["app_json"]) or {}
        if not a.get("name"):
            # an adopted row (secrets_lost) has no app: destroy the lab, keep every volume
            out = self.mgr.destroy(lab_id, reason="deleted", by=origin, origin=origin, include_kept=True)
            out.update(data_volumes_removed=[], final_backup=None, final_backups=[],
                       note="no app record (adopted lab); data volumes kept")
            return out
        with self._app_locked(a["name"]):
            return self._destroy_with_data(lab_id, origin)

    def _destroy_with_data(self, lab_id: str, origin: str) -> Dict[str, Any]:
        final = None
        vols: List[str] = []
        note = None
        shared_with: List[str] = []
        with self.mgr.locked(lab_id, timeout=60):
            row = self.mgr.get(lab_id)
            app = loads(row["app_json"])
            # review §4UD F13: decided HERE, under the app lock (a /keep of this app waits on it) — the agent used to
            # decide from a list it read before its DELETE, so a keep landing in between lost the shared volume
            sharing = self._other_versions(lab_id, row["lineage"], app["name"])
            if app.get("mounted") and sharing and app["data"] != "fresh":
                # another version of this app is kept and uses the same data volume: the lab goes, the data stays
                note = (f"data volumes kept: {', '.join(sharing)} of the same app share(s) them — the data goes "
                        "only with the app's last version")
                shared_with = sharing
            elif app.get("mounted"):
                # the app stays stopped after this backup and the lab goes in the same lock: nothing writes after it
                final = self._backup_locked(lab_id, "final", all_members=True, restart=False)
                for m in self.store.members(lab_id):
                    vols.append(data_volume_name(row["lineage"], app["name"], m["name"],
                                                 lab_id if app["data"] == "fresh" else ""))
            else:
                # a staged / switched-out version owns no data: never remove volumes it did not back up (review r2 #2)
                note = "this version had no data mounted; data volumes kept"
            out = self.mgr.destroy(lab_id, reason="deleted", by=origin, origin=origin, include_kept=True, _have_lock=True)
        others = self._other_versions(lab_id, row["lineage"], app["name"])
        if not others:
            # the last version of this app: its stable port and Service name go too
            key = app.get("lease_key") or f"{row['lineage']}/{app['name']}"
            for p in self.store.q("SELECT port FROM ports WHERE lease_key=?", (key,)):
                self.mgr.ports.unpublish(p["port"])
            self.store.x("DELETE FROM ports WHERE lease_key=?", (key,))
            if self.cfg.ts_services:
                try:
                    self.mgr.ports.unserve_service(app["name"])
                except Exception:  # noqa: BLE001
                    log.exception("unserve %s", app["name"])
        removed = []
        for v in vols:
            try:
                self.docker.volumes.get(v).remove(force=True)
                removed.append(v)
            except Exception:  # noqa: BLE001
                pass
        out["data_volumes_removed"] = removed
        out["final_backup"] = final["backup_id"] if final else None
        out["final_backups"] = final["members"] if final else []
        if note:
            out["note"] = note
        if shared_with:
            out["data_shared_with"] = shared_with
        return out

    def _other_versions(self, lab_id: str, lineage: str, app_name: str) -> List[str]:
        """The OTHER kept labs that hold a version of this (lineage, app) — read under the app lock."""
        return [l["lab_id"] for l in self.store.labs() if l["lab_id"] != lab_id and l["lifetime"] == "kept"
                and l["lineage"] == lineage and l["app_json"] and (loads(l["app_json"]) or {}).get("name") == app_name]

    # ------------------------------------------------------------ agent-api sidecar (v2 §6, §11.11)
    def _ensure_sidecar(self, lab_id: str) -> None:
        row = self.store.lab(lab_id)
        url = row["agent_api_url"]
        name = f"ghost-labs-agentapi-{lab_id}"
        m = re.match(r"^http://([\d.]+):(\d+)$", url)
        ip, port = m.group(1), m.group(2)
        rule = ["-i", net.API_BRIDGE, "-d", ip, "-p", "tcp", "--dport", port, "-j", "ACCEPT"]
        if self.mgr.run(["iptables", "-t", "raw", "-C", "PREROUTING", *rule], capture_output=True, text=True).returncode != 0:
            self.mgr.run(["iptables", "-t", "raw", "-I", "PREROUTING", "1", *rule], capture_output=True, text=True)
        try:
            c = self.docker.containers.get(name)
            if c.status != "running":
                c.start()
            return
        except Exception:  # noqa: BLE001
            pass
        api = self.docker.api
        hc = api.create_host_config(init=True, mem_limit="96m", memswap_limit="96m", pids_limit=64,
                                    cap_drop=["NET_RAW"],     # no forged packets from the proxy (review 11-05 #5)
                                    sysctls={"net.ipv4.ip_forward": "0", "net.ipv6.conf.all.disable_ipv6": "1"},
                                    restart_policy={"Name": "unless-stopped"})
        sip = net.host_in(row["subnet"], SIDECAR_OCTET)
        ncfg = api.create_networking_config({net.API_NETWORK: api.create_endpoint_config()})
        cid = api.create_container("ghost-lab/agentapi:1", name=name, environment={"UPSTREAM": url},
                                   host_config=hc, networking_config=ncfg,
                                   labels={"ghost.labs": "1", "ghost.labs.role": "sidecar", "ghost.lab.sidecar_of": lab_id})["Id"]
        self.docker.networks.get(net.lab_network_name(lab_id)).connect(cid, ipv4_address=sip, aliases=["agent-api"])
        api.start(cid)

    def remove_sidecar(self, lab_id: str) -> None:
        try:
            self.mgr.remove_container(self.docker.containers.get(f"ghost-labs-agentapi-{lab_id}"))
        except Exception:  # noqa: BLE001
            pass
        self.drop_sidecar_rule(lab_id)

    def drop_sidecar_rule(self, lab_id: str) -> None:
        """Remove the raw ACCEPT toward the agent API unless another lab still uses that upstream."""
        row = self.store.lab(lab_id)
        url = row["agent_api_url"] if row else None
        m = re.match(r"^http://([\d.]+):(\d+)$", url or "")
        if not m:
            return
        if any(l["lab_id"] != lab_id and l["agent_api_url"] == url for l in self.store.labs()):
            return
        rule = ["-i", net.API_BRIDGE, "-d", m.group(1), "-p", "tcp", "--dport", m.group(2), "-j", "ACCEPT"]
        for _ in range(10):
            if self.mgr.run(["iptables", "-t", "raw", "-C", "PREROUTING", *rule], capture_output=True, text=True).returncode != 0:
                break
            self.mgr.run(["iptables", "-t", "raw", "-D", "PREROUTING", *rule], capture_output=True, text=True)
