"""`ghost-labs` CLI (contract v1 §1 + v2): talks to the local service over its API
(so locks and state stay in one process), except build-images."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

from . import config as C

IMAGES = [  # build order: parents first
    ("base", "ghost-lab/base:1"), ("tor-gw", "ghost-lab/tor-gw:1"), ("pg", "ghost-lab/pg:18"),
    ("patroni", "ghost-lab/patroni:18"), ("web", "ghost-lab/web:1"), ("python-app", "ghost-lab/python-app:1"),
    ("mysql", "ghost-lab/mysql:8"), ("browser", "ghost-lab/browser:1"), ("agentapi", "ghost-lab/agentapi:1"),
]


def _call(method: str, path: str, body=None):
    cfg = C.load()
    C.resolve_bind(cfg)
    url = f"http://{cfg.resolved_bind_host}:{cfg.bind_port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "X-Labs-Key": C.api_key(), "X-Labs-Origin": "owner", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


def build_images(name: str = "") -> int:
    root = C.OPT_DIR
    todo = [i for i in IMAGES if not name or i[0] == name]
    if not todo:
        print(f"unknown image {name}; one of {[i[0] for i in IMAGES]}", file=sys.stderr)
        return 2
    for d, tag in todo:
        if not (root / "images" / d / "Dockerfile").exists():
            print(f"skip {tag} (no Dockerfile)")
            continue
        print(f"== building {tag}", flush=True)
        r = subprocess.run(["docker", "build", "-t", tag, "-f", f"images/{d}/Dockerfile", "."], cwd=str(root))
        if r.returncode != 0:
            print(f"FAILED {tag}", file=sys.stderr)
            return r.returncode
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ghost-labs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("list")
    d = sub.add_parser("destroy")
    d.add_argument("lab_id", nargs="?")
    d.add_argument("--all", action="store_true")
    d.add_argument("--all-disposable", action="store_true")
    d.add_argument("--include-kept", action="store_true")
    sub.add_parser("gc")
    b = sub.add_parser("build-images")
    b.add_argument("name", nargs="?", default="")
    sub.add_parser("templates")
    sub.add_parser("url").add_argument("--key", action="store_true")
    a = ap.parse_args(argv)
    out = None
    if a.cmd == "status":
        out = _call("GET", "/v1/health")
    elif a.cmd == "list":
        out = _call("GET", "/v1/labs")
    elif a.cmd == "templates":
        out = _call("GET", "/v1/templates")
        digests = {}
        for _d, tag in IMAGES:
            r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture_output=True, text=True)
            digests[tag] = r.stdout.strip() or None
        out["image_ids"] = digests
    elif a.cmd == "gc":
        out = _call("POST", "/v1/gc")
    elif a.cmd == "build-images":
        return build_images(a.name)
    elif a.cmd == "url":
        cfg = C.load()
        C.resolve_bind(cfg)
        print(f"GHOST_LABS_HOST=http://{cfg.resolved_bind_host}:{cfg.bind_port}")
        print(f"key file: {C.CONFIG_DIR / 'api_key'}")
        return 0
    elif a.cmd == "destroy":
        if a.all:
            print("--all is not allowed: use --all-disposable (kept labs are removed one at a time with --include-kept)",
                  file=sys.stderr)
            return 2
        if a.all_disposable:
            labs = _call("GET", "/v1/labs").get("labs", [])
            out = {"destroyed": []}
            for lab in labs:
                if lab["lifetime"] != "kept":
                    r = _call("DELETE", f"/v1/labs/{lab['lab_id']}")
                    out["destroyed"].append(r)
        elif a.lab_id:
            q = "?include_kept=1" if a.include_kept else ""
            out = _call("DELETE", f"/v1/labs/{a.lab_id}{q}")
        else:
            print("give a lab id or --all-disposable", file=sys.stderr)
            return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 1 if isinstance(out, dict) and "error" in out else 0


if __name__ == "__main__":
    sys.exit(main())
