"""Host reboot (contract v2 §11.12): `make itest-reboot-pre`, reboot the host, `make itest-reboot-post`."""
import json
import subprocess
import uuid
from pathlib import Path

import pytest

from .labs_client import Labs, labs, wait_for  # noqa: F401
from .test_kept import archive

STATE = Path("/var/lib/ghost-labs/itest-reboot.json")


def _count(labs, lab):  # noqa: F811
    return labs.sh(lab, "app", "curl -sS -m 5 http://127.0.0.1:8080/count")["stdout"]


@pytest.mark.reboot_pre
def test_pre(labs):  # noqa: F811
    proj = "itestrb" + uuid.uuid4().hex[:4]
    kept = labs.create(template="python-app", owner=f"project:{proj}")["lab_id"]
    labs.ok("POST", f"/v1/labs/{kept}/keep", {"app": {"name": "rb-" + proj[-4:], "member": "app", "cmd": "python3 app.py",
                                                      "port": 8080, "health": {"http": "/"}}})
    assert labs.ok("POST", f"/v1/labs/{kept}/deploy", {"archive_b64": archive("rb"), "data_seed": ["data"]})["probe"]["ok"]
    n = _count(labs, kept)
    tor = labs.create(template="blank-dev", label="itest-reboot")["lab_id"]
    url = labs.ok("GET", f"/v1/labs/{kept}")["app"]["url"]
    STATE.write_text(json.dumps({"kept": kept, "tor": tor, "count": n, "url": url}))


@pytest.mark.reboot_post
def test_post(labs):  # noqa: F811
    st = json.loads(STATE.read_text())
    def healthy():
        try:
            x = labs.ok("GET", "/v1/health")
        except OSError:          # not listening yet
            return None
        return x if x.get("tor") == "ok" else None
    h = wait_for(healthy, 300, 5, "healthy")
    assert h["firewall"] == "ok" and h["docker"] == "ok"
    try:
        r = labs.sh(st["tor"], "dev", "ip -4 route show default; curl -sS -m 60 https://check.torproject.org/api/ip")
        assert "via" in r["stdout"] and '"IsTor":true' in r["stdout"].replace(" ", ""), r
        wait_for(lambda: labs.ok("GET", f"/v1/labs/{st['kept']}")["app"]["state"] == "up", 120, 3, "app up")
        assert _count(labs, st["kept"]) == str(int(st["count"]) + 1)     # /data survived
        out = subprocess.run(["curl", "-sS", "-m", "20", st["url"]], capture_output=True, text=True).stdout
        assert out == "ok rb", out
    finally:
        labs.destroy(st["tor"])
        labs.destroy(st["kept"], include_kept=1, drop_data=1)
        STATE.unlink(missing_ok=True)
