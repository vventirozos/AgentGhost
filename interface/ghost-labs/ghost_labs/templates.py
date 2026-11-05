"""Template parsing (contract v1 §8 + v2 §11.9/§11.21)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import yaml

from .errors import LabError

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
EGRESS = ("tor", "none")
ALLOWED_CAPS = ("NET_ADMIN", "NET_RAW")


def _member(m: Dict[str, Any], where: str) -> Dict[str, Any]:
    if not isinstance(m, dict):
        raise LabError("bad_request", f"{where}: member must be a mapping", field="members")
    unknown = set(m) - {"name", "image", "mem_mb", "cpus", "ports", "cap_add", "env"}
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
    return {"name": name, "image": str(m.get("image") or ""), "mem_mb": mem,
            "cpus": float(m.get("cpus") or 1.0), "ports": ports, "cap_add": cap_add,
            "env": {str(k): str(v) for k, v in (m.get("env") or {}).items()}}


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
    return {"name": name, "description": str(doc.get("description") or ""), "egress": egress,
            "members": members, "env": {str(k): str(v) for k, v in (doc.get("env") or {}).items()},
            "readiness": readiness, "params": params}


def load_dir(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(path).glob("*.yaml")):
        t = parse(yaml.safe_load(f.read_text()) or {}, where=f.name)
        out[t["name"]] = t
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
