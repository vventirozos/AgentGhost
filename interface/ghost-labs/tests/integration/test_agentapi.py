"""agent-api sidecar (contract v2 §6, §11.11, §10): needs a real tailnet upstream.
Run with AGENT_API_UPSTREAM=http://<tailnet ip>:<port> serving an echo of the request
(method, path, headers) as JSON; OTHER_TAILNET_HOST=<ip> is a second tailnet node."""
import json
import os
import subprocess
import uuid

import pytest

from .labs_client import Labs, labs  # noqa: F401

UP = os.environ.get("AGENT_API_UPSTREAM", "")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not UP, reason="AGENT_API_UPSTREAM not set")]


@pytest.fixture(scope="module")
def lab(labs):  # noqa: F811
    proj = "itestapi" + uuid.uuid4().hex[:4]
    lid = labs.create(template="python-app", owner=f"project:{proj}", label="itest-agentapi")["lab_id"]
    labs.ok("POST", f"/v1/labs/{lid}/keep", {"app": {"name": "itest-api-" + proj[-4:], "member": "app", "cmd": "sleep infinity",
                                                     "port": 8080, "health": {"http": None, "tcp": False}},
                                             "agent_api_url": UP, "secret_values": {"GHOST_SERVICE_TOKEN": "tok-" + proj}})
    yield lid
    labs.destroy(lid, include_kept=1, drop_data=1)


def test_agent_api_url_validated(labs):  # noqa: F811
    lid = labs.create(template="python-app", owner="project:itestbad")["lab_id"]
    try:
        for bad in ("http://192.168.1.1:80", "http://example.com:80", "https://100.90.0.1:443", "http://100.90.0.1"):
            st, out = labs.call("POST", f"/v1/labs/{lid}/keep", {"app": {"name": "x", "member": "app", "port": 8080},
                                                                 "agent_api_url": bad})
            assert st == 400 and out["field"] == "agent_api_url", bad
    finally:
        labs.destroy(lid)


def test_game_path_proxied_headers_stripped(labs, lab):  # noqa: F811
    r = labs.sh(lab, "app", 'curl -sS -m 20 -H "X-Ghost-Key: K" -H "Cookie: c=1" -H "Authorization: Bearer x" '
                            '-H "X-Ghost-Service-Token: $(cat /run/ghost/secrets.env >/dev/null; echo tok)" '
                            '-H "X-Ghost-Origin: o" -H "X-Ghost-Requester: r" "$GHOST_API/api/game/games?x=1"')
    out = json.loads(r["stdout"])
    assert out["path"] == "/api/game/games?x=1" and out["method"] == "GET"
    hdrs = {k.lower(): v for k, v in out["headers"].items()}
    assert hdrs.get("x-ghost-service-token") == "tok"
    for h in ("x-ghost-key", "cookie", "authorization", "x-ghost-origin", "x-ghost-requester", "x-forwarded-for"):
        assert h not in hdrs, h
    assert hdrs["host"] == UP.split("//", 1)[1]
    r = labs.sh(lab, "app", 'curl -sS -m 20 -X POST -d "{\\"a\\":1}" -H "Content-Type: application/json" "$GHOST_API/api/game/move"')
    assert json.loads(r["stdout"])["body"] == '{"a":1}'


@pytest.mark.parametrize("path", ["/api/chat", "/api/game/../chat", "/api/game/%2e%2e/chat", "/api/game//x",
                                  "/api/gamex", "/v1/health", "/api/game/a%2fb"])
def test_other_paths_refused(labs, lab, path):  # noqa: F811
    r = labs.sh(lab, "app", f'curl -sS -m 10 -o /dev/null -w "%{{http_code}}" --path-as-is "$GHOST_API{path}"')
    assert r["stdout"] == "403", (path, r)


def test_methods_and_size(labs, lab):  # noqa: F811
    r = labs.sh(lab, "app", 'curl -sS -m 10 -o /dev/null -w "%{http_code}" -X DELETE "$GHOST_API/api/game/x"')
    assert r["stdout"] == "405"
    r = labs.sh(lab, "app", 'head -c 70000 /dev/zero | curl -sS -m 10 -o /dev/null -w "%{http_code}" --data-binary @- '
                            '"$GHOST_API/api/game/x"')
    assert r["stdout"] == "413"


def test_bad_framing_refused(labs, lab):  # noqa: F811
    # raw requests: a negative / non-numeric Content-Length and a chunked body are refused, a stalled client is dropped
    for hdr, want in (("Content-Length: -5", "400"), ("Content-Length: abc", "400"),
                      ("Transfer-Encoding: chunked", "411")):
        r = labs.sh(lab, "app", "python3 - <<'PY'\n"
                    "import socket, os, urllib.parse\n"
                    "u = urllib.parse.urlsplit(os.environ['GHOST_API'])\n"
                    "s = socket.create_connection((u.hostname, u.port), timeout=15)\n"
                    f"s.sendall(b'POST /api/game/x HTTP/1.1\\r\\nHost: a\\r\\n{hdr}\\r\\n\\r\\n')\n"
                    "print(s.recv(200).split(b' ')[1].decode())\n"
                    "PY")
        assert r["stdout"].strip() == want, (hdr, r)
    r = labs.sh(lab, "app", "python3 - <<'PY'\n"
                "import socket, os, time, urllib.parse\n"
                "u = urllib.parse.urlsplit(os.environ['GHOST_API'])\n"
                "s = socket.create_connection((u.hostname, u.port), timeout=30)\n"
                "s.sendall(b'POST /api/game/x HTTP/1.1\\r\\nHost: a\\r\\nContent-Length: 50\\r\\n\\r\\nabc')\n"
                "t = time.time(); d = s.recv(200)\n"
                "print('closed' if time.time() - t < 25 else 'hung')\n"
                "PY", timeout_s=60)
    assert r["stdout"].strip() == "closed", r


def test_member_cannot_reach_upstream_directly(labs, lab):  # noqa: F811
    r = labs.sh(lab, "app", f"curl -sS -m 10 {UP}/api/game/games")
    assert "headers" not in r["stdout"]


def test_sidecar_netns_reaches_only_upstream(lab):
    pid = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", f"ghost-labs-agentapi-{lab}"],
                         capture_output=True, text=True).stdout.strip()
    host, port = UP.split("//", 1)[1].split(":")

    def can(h, p):
        return subprocess.run(["nsenter", "-t", pid, "-n", "timeout", "6", "bash", "-c", f"exec 3<>/dev/tcp/{h}/{p}"],
                              capture_output=True).returncode == 0
    assert can(host, port)
    ts_ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()[0]
    gw = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True).stdout.split()[2]
    targets = [(host, 22), (ts_ip, 8700), (ts_ip, 22), (gw, 80), ("10.88.1.1", 22)]
    other = os.environ.get("OTHER_TAILNET_HOST")
    if other:
        targets.append((other, 22))
    for h, p in targets:
        assert not can(h, p), (h, p)
