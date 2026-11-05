"""Integration helpers: the real service on this host, the real Docker."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest

KEY_FILE = "/etc/ghost-labs/api_key"


def _base():
    ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()[0]
    return os.environ.get("GHOST_LABS_HOST", f"http://{ip}:8700")


class Labs:
    def __init__(self, origin="owner"):
        self.base = _base()
        self.key = open(KEY_FILE).read().strip()
        self.origin = origin

    def call(self, method, path, body=None, origin=None, timeout=900):
        hdr = {"X-Labs-Key": self.key, "Content-Type": "application/json"}
        o = self.origin if origin is None else origin
        if o:
            hdr["X-Labs-Origin"] = o
        req = urllib.request.Request(self.base + path, method=method, headers=hdr,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def ok(self, method, path, body=None, **kw):
        st, out = self.call(method, path, body, **kw)
        assert st < 300, (st, out)
        return out

    def create(self, **body):
        return self.ok("POST", "/v1/labs", {"owner": "conversation:itest", **body})

    def sh(self, lab, member, cmd, timeout_s=120, **kw):
        return self.ok("POST", f"/v1/labs/{lab}/exec", {"member": member, "cmd": cmd, "timeout_s": timeout_s, **kw})

    def destroy(self, lab, **q):
        qs = "&".join(f"{k}={v}" for k, v in q.items())
        return self.call("DELETE", f"/v1/labs/{lab}" + (f"?{qs}" if qs else ""))

    def put(self, lab, member, path, data: bytes, mode="0644"):
        return self.ok("PUT", f"/v1/labs/{lab}/files", {"member": member, "path": path, "mode": mode,
                                                         "content_b64": base64.b64encode(data).decode()})


def leftovers(lab_id):
    out = []
    for kind in ("ps -a", "network ls", "volume ls"):
        r = subprocess.run(f"docker {kind} -q --filter label=ghost.lab.id={lab_id}", shell=True, capture_output=True, text=True)
        out += r.stdout.split()
    return out


def wait_for(fn, timeout, every=3, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


@pytest.fixture(scope="module")
def labs():
    return Labs()
