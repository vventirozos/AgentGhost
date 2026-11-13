"""Template parsing (contract v1 §8 + v2 §11.9/§11.21)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .errors import LabError

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
EGRESS = ("tor", "none")
ALLOWED_CAPS = ("NET_ADMIN", "NET_RAW")


def _member(m: Dict[str, Any], where: str) -> Dict[str, Any]:
    if not isinstance(m, dict):
        raise LabError("bad_request", f"{where}: member must be a mapping", field="members")
    unknown = set(m) - {"name", "image", "mem_mb", "cpus", "ports", "cap_add", "env", "ready"}
    if unknown:
        raise LabError("bad_request", f"{where}: unknown member keys {sorted(unknown)}", field="members")
    name = str(m.get("name", ""))
    if not NAME_RE.match(name):
        raise LabError("bad_request", f"{where}: bad member name {name!r}", field="members")
    cap_add = list(m.get("cap_add") or [])
    # NET_RAW is dropped from every member unless listed here (review 11-05 #5); NET_ADMIN as before
    if any(c not in ALLOWED_CAPS for c in cap_add):
        raise LabError("bad_request", f"{where}: only {', '.join(ALLOWED_CAPS)} may be added", field="cap_add")
    ports = [int(p) for p in (m.get("ports") or [])]
    if any(not (1 <= p <= 65535) for p in ports):
        raise LabError("bad_request", f"{where}: bad port", field="ports")
    mem = int(m.get("mem_mb") or 0)
    if mem < 64:
        raise LabError("bad_request", f"{where}: mem_mb must be >= 64", field="mem_mb")
    # `ready`: a command that exits 0 once THIS member's service accepts work (agent §4UD F-A). Optional.
    ready = str(m.get("ready") or "").strip()
    if "\n" in ready or len(ready) > 500:
        raise LabError("bad_request", f"{where}: ready must be one command line (500 characters at most)", field="ready")
    return {"name": name, "image": str(m.get("image") or ""), "mem_mb": mem,
            "cpus": float(m.get("cpus") or 1.0), "ports": ports, "cap_add": cap_add,
            "env": {str(k): str(v) for k, v in (m.get("env") or {}).items()},
            **({"ready": ready} if ready else {})}


def parse(doc: Dict[str, Any], where: str = "template") -> Dict[str, Any]:
    name = str(doc.get("name", ""))
    if not NAME_RE.match(name):
        raise LabError("bad_request", f"{where}: bad template name {name!r}", field="name")
    egress = str(doc.get("egress", "tor"))
    if egress not in EGRESS:
        raise LabError("bad_request", f"{where}: egress must be tor|none", field="egress")
    members = [_member(m, where) for m in (doc.get("members") or [])]
    if not members:
        raise LabError("bad_request", f"{where}: no members", field="members")
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise LabError("bad_request", f"{where}: duplicate member names", field="members")
    readiness = doc.get("readiness") or {"member": names[0], "cmd": "true", "timeout_s": 60}
    readiness = {"member": str(readiness.get("member") or names[0]), "cmd": str(readiness.get("cmd") or "true"),
                 "check": str(readiness.get("check") or "exit_code"), "timeout_s": int(readiness.get("timeout_s") or 60)}
    if readiness["member"] not in names:
        raise LabError("bad_request", f"{where}: readiness member unknown", field="readiness")
    params = {str(k): str(v) for k, v in (doc.get("params") or {}).items()}
    app = _app_defaults(doc.get("app"), names, where)
    return {"name": name, "description": str(doc.get("description") or ""), "egress": egress,
            "members": members, "env": {str(k): str(v) for k, v in (doc.get("env") or {}).items()},
            "readiness": readiness, "params": params, **({"app": app} if app else {})}


#: §4UL: a template may name how its kept app runs — `keep` fills what the caller left out (a static site, a
#: database: nothing app-specific to choose)
APP_DEFAULT_KEYS = ("member", "port", "cmd", "health", "build", "service", "wait_for", "watch")
#: app.wait_for: the app's start waits (up to 90 s) until this host:port answers — python-pg's app waits for its db
WAIT_FOR_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,62}:[0-9]{1,5}$")


def _app_defaults(a: Any, names: List[str], where: str) -> Dict[str, Any]:
    if a is None:
        return {}
    if not isinstance(a, dict):
        raise LabError("bad_request", f"{where}: app must be a mapping", field="app")
    unknown = set(a) - set(APP_DEFAULT_KEYS)
    if unknown:
        raise LabError("bad_request", f"{where}: unknown app keys {sorted(unknown)}", field="app")
    out: Dict[str, Any] = {}
    if a.get("member") is not None:
        if str(a["member"]) not in names:
            raise LabError("bad_request", f"{where}: app.member is not a member", field="app")
        out["member"] = str(a["member"])
    if a.get("port") is not None:
        p = int(a["port"])
        if not 1 <= p <= 65535:
            raise LabError("bad_request", f"{where}: bad app.port", field="app")
        out["port"] = p
    for k in ("cmd", "build"):
        if a.get(k) is not None:
            v = str(a[k])
            if "\n" in v or len(v) > 1000:
                raise LabError("bad_request", f"{where}: app.{k} must be one line", field="app")
            out[k] = v
    if a.get("service") is not None:
        out["service"] = bool(a["service"])     # runs with nothing deployed (a template's own server, e.g. postgres)
    if a.get("health") is not None:
        if not isinstance(a["health"], dict):
            raise LabError("bad_request", f"{where}: app.health must be a mapping", field="app")
        out["health"] = dict(a["health"])
    if a.get("wait_for") is not None:
        if not WAIT_FOR_RE.match(str(a["wait_for"])):
            raise LabError("bad_request", f"{where}: app.wait_for must be host:port", field="app")
        out["wait_for"] = str(a["wait_for"])
    if a.get("watch") is not None:
        out["watch"] = watch_list(a["watch"], names, where)
    return out


def watch_list(v: Any, names: List[str], where: str = "app") -> List[Dict[str, str]]:
    """app.watch: [{member, cmd}] — a keeper process per OTHER member (python-pg's db: restart postgres if it died),
    started with the app; its own process group, so stopping the app never stops it."""
    if not isinstance(v, list) or len(v) > 4:
        raise LabError("bad_request", f"{where}: app.watch must be a list of {{member, cmd}} (4 at most)", field="app")
    out = []
    for w in v:
        if not isinstance(w, dict) or set(w) - {"member", "cmd"} or str(w.get("member") or "") not in names:
            raise LabError("bad_request", f"{where}: app.watch entries are {{member: <a member>, cmd}}", field="app")
        cmd = str(w.get("cmd") or "")
        if not cmd or "\n" in cmd or len(cmd) > 1000:
            raise LabError("bad_request", f"{where}: app.watch cmd must be one line", field="app")
        out.append({"member": str(w["member"]), "cmd": cmd})
    return out


def load_dir(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(path).glob("*.yaml")):
        t = parse(yaml.safe_load(f.read_text()) or {}, where=f.name)
        out[t["name"]] = t
    return out


# The templates that ship with the code: read from the code's own tree, so a deploy of the code IS a deploy of the
# templates (§4UD-c: the service read a copy in /etc that only install.sh refreshed; a template fix synced to
# /opt was dead for hours).
SHIPPED_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_all(shipped: Path, extra: Optional[Path] = None) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """The shipped templates, plus the operator's own from ``extra`` (/etc/ghost-labs/templates). A shipped name
    always wins: an ``extra`` file with that name is NOT loaded. Only new names come from ``extra``; two ``extra``
    files with one name: the first by file name. The second value lists, for the caller to log, each ``extra``
    file that was not loaded and why. A file there that does not load, for ANY reason, is one of those, never a
    reason the service cannot start (r3 m3, r4: a stale copy, `.inf`, deep nesting each raised out of the
    constructor). A shipped template that does not parse still raises."""
    out = load_dir(shipped)
    shipped_names = set(out)
    seen: Dict[str, str] = {}                      # an /etc name -> the file that gave it
    notes: List[str] = []
    if extra is None or not Path(extra).is_dir():
        return out, notes
    for f in sorted(Path(extra).glob("*.yaml")):
        try:
            t = parse(yaml.safe_load(f.read_text()) or {}, where=f.name)
        except Exception as e:  # noqa: BLE001 — any failure of ONE operator file is a note
            notes.append(f"{f.name}: does not load ({type(e).__name__}: {str(e)[:120]})")
            continue
        if t["name"] in shipped_names:
            notes.append(f"{f.name}: a shipped template is named {t['name']} (rename it to add one)")
        elif t["name"] in seen:
            notes.append(f"{f.name}: {seen[t['name']]} already gives a template named {t['name']}")
        else:
            seen[t["name"]] = f.name
            out[t["name"]] = t
    return out, notes


#: §4UL: what an owner's create may change on a template (the composition the template does not offer)
OVERRIDE_KEYS = ("mem_mb", "ports", "cap_add", "cpus", "egress")


def apply_overrides(spec: Dict[str, Any], ov: Any) -> Dict[str, Any]:
    """A template's spec with the owner's overrides: egress for the lab, and mem_mb / ports / cap_add / cpus for
    every member (a plain value) or per member ({member: value}). Every member is validated again."""
    if not ov:
        return spec
    if not isinstance(ov, dict):
        raise LabError("bad_request", "overrides must be an object", field="overrides")
    unknown = set(ov) - set(OVERRIDE_KEYS)
    if unknown:
        raise LabError("bad_request", f"unknown overrides {sorted(unknown)}", field="overrides",
                       hint=f"one of {', '.join(OVERRIDE_KEYS)}")
    out = dict(spec)
    if "egress" in ov:
        if str(ov["egress"]) not in EGRESS:
            raise LabError("bad_request", "egress must be tor|none", field="overrides")
        out["egress"] = str(ov["egress"])
    names = [m["name"] for m in spec["members"]]
    members = []
    for m in spec["members"]:
        raw = dict(m)
        for k in ("mem_mb", "ports", "cap_add", "cpus"):
            if k not in ov:
                continue
            v = ov[k]
            if isinstance(v, dict):
                bad = set(v) - set(names)
                if bad:
                    raise LabError("bad_request", f"overrides.{k}: unknown members {sorted(bad)}", field="overrides",
                                   members=names)
                if m["name"] not in v:
                    continue
                v = v[m["name"]]
            raw[k] = v
        try:
            members.append(_member(raw, "overrides"))
        except (TypeError, ValueError) as e:
            raise LabError("bad_request", f"overrides: {e}", field="overrides")
    out["members"] = members
    return out


_PARAM_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def substitute(tpl: Dict[str, Any], params: Dict[str, str]) -> Dict[str, Any]:
    """Apply ${param} substitution to env values. Unknown params → bad_request."""
    unknown = set(params) - set(tpl["params"])
    if unknown:
        raise LabError("bad_request", f"unknown params {sorted(unknown)}", field="params",
                       params=sorted(tpl["params"]))
    values = {**tpl["params"], **{k: str(v) for k, v in params.items()}}

    def sub(s: str) -> str:
        return _PARAM_RE.sub(lambda m: values.get(m.group(1), m.group(0)), s)

    out = dict(tpl)
    out["env"] = {k: sub(v) for k, v in tpl["env"].items()}
    out["members"] = [{**m, "env": {k: sub(v) for k, v in m["env"].items()}} for m in tpl["members"]]
    return out
