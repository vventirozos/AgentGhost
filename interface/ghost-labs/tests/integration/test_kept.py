"""Kept machines (contract v2 §5, §10, §11.12-18, §12) against the real service."""
import io
import json
import subprocess
import tarfile
import time
import uuid

import pytest

from .labs_client import Labs, labs, leftovers, wait_for  # noqa: F401

pytestmark = pytest.mark.integration
APP = "labs-test"   # the Service name defined in the Tailscale console for tests (UNIFIED §15.5)

APP_PY = r'''
import os, sys, json
from flask import Flask
app = Flask(__name__)
VERSION = "%s"
@app.get("/")
def index():
    return "ok " + VERSION
@app.get("/count")
def count():
    p = "data/count"
    n = int(open(p).read()) + 1 if os.path.exists(p) else 1
    open(p, "w").write(str(n))
    return str(n)
@app.get("/crash")
def crash():
    os._exit(3)
@app.get("/token")
def token():
    return os.environ.get("APP_TOKEN", "missing")
app.run(host="0.0.0.0", port=8080)
'''


def archive(version):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in (("app.py", APP_PY % version), ("data/README", "seed\n")):
            b = body.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def get(labs, lab, path):  # noqa: F811
    r = labs.sh(lab, "app", f"curl -sS -m 5 http://127.0.0.1:8080{path}")
    return r["stdout"]


@pytest.fixture(scope="module")
def project():
    return "itest" + uuid.uuid4().hex[:6]


@pytest.fixture(scope="module")
def kept(labs, project):  # noqa: F811
    lab = labs.create(template="python-app", owner=f"project:{project}", label="itest-kept")
    lid = lab["lab_id"]
    yield lid
    labs.destroy(lid, include_kept=1, drop_data=1)


def app_state(labs, lid):  # noqa: F811
    return labs.ok("GET", f"/v1/labs/{lid}")["app"]


def test_keep_owner_only(labs, kept):  # noqa: F811
    for origin in ("autopilot", ""):
        st, out = labs.call("POST", f"/v1/labs/{kept}/keep", {"app": {"name": APP}}, origin=origin)
        assert st == 403 and out["error"] == "owner_only"


def test_keep_and_deploy(labs, kept):  # noqa: F811
    out = labs.ok("POST", f"/v1/labs/{kept}/keep", {
        "app": {"name": APP, "member": "app", "cmd": "python3 app.py", "port": 8080, "health": {"http": "/"}},
        "secrets": ["APP_TOKEN"]})
    assert out["lifetime"] == "kept" and out["expires_at"] is None
    assert "APP_TOKEN" in out["app"]["secrets"] and "secret_values" not in json.dumps(out)
    d = labs.ok("POST", f"/v1/labs/{kept}/deploy", {"archive_b64": archive("v1"), "data_seed": ["data"]})
    assert d["probe"]["ok"] is True, d
    assert get(labs, kept, "/") == "ok v1"
    assert get(labs, kept, "/count") == "1"
    r = labs.sh(kept, "app", "readlink /app/data; ls /data")
    assert r["stdout"].split()[0] == "/data/data"


def test_secret_redacted_and_injected(labs, kept):  # noqa: F811
    assert get(labs, kept, "/token") == "<secret:APP_TOKEN>"
    r = labs.sh(kept, "app", "cat /run/ghost/secrets.env; stat -c %a /run/ghost/secrets.env")
    assert "<secret:APP_TOKEN>" in r["stdout"] and r["stdout"].strip().endswith("600")
    # the hex form of the value (one of the redacted forms, v2 §11.20)
    r = labs.sh(kept, "app", "grep -o 'APP_TOKEN=.*' /run/ghost/secrets.env | cut -d= -f2- | tr -d \"'\\n\" | od -An -tx1 | tr -d ' \\n'")
    assert r["stdout"] == "<secret:APP_TOKEN>", r


def test_redeploy_keeps_data_and_seed_refused(labs, kept):  # noqa: F811
    labs.ok("POST", f"/v1/labs/{kept}/deploy", {"archive_b64": archive("v1b")})
    assert get(labs, kept, "/") == "ok v1b"
    assert get(labs, kept, "/count") == "2"
    st, out = labs.call("POST", f"/v1/labs/{kept}/deploy", {"archive_b64": archive("v1c"), "data_seed": ["other"]})
    assert st == 409 and out["error"] == "data_exists"
    assert get(labs, kept, "/") == "ok v1b"          # a refused deploy leaves the running version alone
    assert app_state(labs, kept)["state"] == "up"


def test_kept_survives_gc_and_bulk_destroy(labs, kept):  # noqa: F811
    labs.ok("POST", "/v1/gc")
    r = subprocess.run(["ghost-labs", "destroy", "--all-disposable"], capture_output=True, text=True)
    assert r.returncode == 0
    assert labs.ok("GET", f"/v1/labs/{kept}")["lifetime"] == "kept"
    st, out = labs.destroy(kept)
    assert st == 400 and out["error"] == "bad_request"
    st, out = labs.call("DELETE", f"/v1/labs/{kept}?include_kept=1", origin="autopilot")
    assert st == 403
    r = subprocess.run(["ghost-labs", "destroy", "--all"], capture_output=True, text=True)
    assert r.returncode == 2


def test_https_port_url_and_not_plain(labs, kept):  # noqa: F811
    a = app_state(labs, kept)
    url = a["url"]
    assert url.startswith("https://") and url.endswith(f":{a['host_port']}/")
    r = subprocess.run(["curl", "-sS", "-m", "20", url], capture_output=True, text=True)
    assert r.stdout.startswith("ok "), r
    ts_ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()[0]
    r = subprocess.run(["curl", "-sS", "-m", "5", f"http://{ts_ip}:{a['host_port']}/"], capture_output=True, text=True)
    assert not r.stdout.startswith("ok ")
    lan = subprocess.run(["ip", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True).stdout.split()
    lan_ip = lan[lan.index("src") + 1]
    r = subprocess.run(["curl", "-sS", "-m", "5", f"http://{lan_ip}:{a['host_port']}/"], capture_output=True, text=True)
    assert not r.stdout.startswith("ok ")


def test_service_name_becomes_live(labs, kept):  # noqa: F811
    a = wait_for(lambda: (lambda x: x if x.get("name_state") in ("live", "unreachable") else None)(app_state(labs, kept)),
                 180, 5, "name check")
    assert a["name_state"] == "live", a
    r = subprocess.run(["curl", "-sS", "-m", "20", a["name_url"]], capture_output=True, text=True)
    assert r.stdout.startswith("ok "), r


def test_crash_restarts_then_crashloop(labs, kept):  # noqa: F811
    r0 = app_state(labs, kept)["restarts"]
    labs.sh(kept, "app", "curl -sS -m 3 http://127.0.0.1:8080/crash", timeout_s=10)
    a = wait_for(lambda: (lambda x: x if x["state"] == "up" and x["restarts"] > r0 else None)(app_state(labs, kept)),
                 90, 3, "restart after crash")
    assert get(labs, kept, "/count") == "3"
    for i in range(6):
        cur = app_state(labs, kept)
        if cur["state"] == "crashloop":
            break
        wait_for(lambda: app_state(labs, kept)["state"] == "up", 200, 3, f"up before crash {i}")
        labs.sh(kept, "app", "curl -sS -m 3 http://127.0.0.1:8080/crash", timeout_s=10)
        time.sleep(4)
    a = wait_for(lambda: (lambda x: x if x["state"] == "crashloop" else None)(app_state(labs, kept)), 300, 3, "crashloop")
    assert a["state"] == "crashloop"
    h = labs.ok("GET", "/v1/health")
    assert any(x["lab_id"] == kept and x["state"] == "crashloop" for x in h["apps"])
    out = labs.ok("POST", f"/v1/labs/{kept}/app/start")
    assert out["probe"]["ok"] is True and out["app"]["state"] == "up"


def test_backup_restore(labs, kept):  # noqa: F811
    before = get(labs, kept, "/count")
    b = labs.ok("POST", f"/v1/labs/{kept}/backups")
    get(labs, kept, "/count")
    get(labs, kept, "/count")
    out = labs.ok("POST", f"/v1/labs/{kept}/restore", {"backup_id": b["backup_id"]})
    assert out["probe"]["ok"] is True
    assert get(labs, kept, "/count") == str(int(before) + 1)
    lst = labs.ok("GET", f"/v1/labs/{kept}/backups")["backups"]
    assert any(x["backup_id"] == b["backup_id"] for x in lst)
    st, out = labs.call("POST", f"/v1/labs/{kept}/restore", {"backup_id": "bk_nope"})
    assert st == 404 and out["error"] == "unknown_backup"


def test_rehearse_keeps_data_and_port(labs, kept):  # noqa: F811
    port = app_state(labs, kept)["host_port"]
    n = int(get(labs, kept, "/count"))
    out = labs.ok("POST", f"/v1/labs/{kept}/rehearse")
    assert out["recreated"]["ok"] is True and out["twin"]["probe"]["ok"] is True, out
    assert leftovers(out["twin"]["lab_id"]) == []
    assert app_state(labs, kept)["host_port"] == port
    assert get(labs, kept, "/count") == str(n + 1)


def test_switch_and_rollback(labs, kept, project):  # noqa: F811
    port = app_state(labs, kept)["host_port"]
    n = int(get(labs, kept, "/count"))
    v2 = labs.create(template="python-app", owner=f"project:{project}", label="itest-v2")["lab_id"]
    try:
        out = labs.ok("POST", f"/v1/labs/{v2}/keep", {
            "app": {"name": APP, "member": "app", "cmd": "python3 app.py", "port": 8080, "health": {"http": "/"}},
            "secrets": ["APP_TOKEN"]})
        assert out["app"]["state"] == "staged"
        d = labs.ok("POST", f"/v1/labs/{v2}/deploy", {"archive_b64": archive("v2")})
        assert d["app"]["state"] == "staged"
        assert get(labs, kept, "/") == "ok v1b"                  # v1 still serving
        sw = labs.ok("POST", f"/v1/labs/{v2}/switch", {"from_lab": kept})
        assert sw["switched"] is True, sw
        assert get(labs, v2, "/") == "ok v2"
        assert get(labs, v2, "/count") == str(n + 1)              # shared data moved over
        assert app_state(labs, v2)["host_port"] == port
        r = subprocess.run(["curl", "-sS", "-m", "20", app_state(labs, v2)["url"]], capture_output=True, text=True)
        assert r.stdout == "ok v2"
        rb = labs.ok("POST", f"/v1/labs/{v2}/rollback", {"restore_snapshot": False})
        assert rb["probe"]["ok"] is True, rb
        assert get(labs, kept, "/") == "ok v1b"
        assert get(labs, kept, "/count") == str(n + 2)
        assert app_state(labs, kept)["host_port"] == port
    finally:
        labs.destroy(v2, include_kept=1)
