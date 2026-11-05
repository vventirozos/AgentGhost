"""agent-api sidecar (contract v2 §6, §11.11): members call GHOST_API
(http://agent-api:8000); only GET|POST /api/game/* reach UPSTREAM."""
import http.client
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = urllib.parse.urlsplit(os.environ["UPSTREAM"])
PATH_RE = re.compile(r"^/api/game/[A-Za-z0-9/_-]*$")
BAD = re.compile(r"\.\.|//|%2e|%2f|%5c", re.I)
PASS = ("content-type", "content-length", "accept", "x-ghost-service-token")
REQ_MAX, RESP_MAX = 64 * 1024, 1024 * 1024


class H(BaseHTTPRequestHandler):
    server_version = "agent-api"
    sys_version = ""
    timeout = 10          # a client that stalls mid-request is dropped

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.command, self.path[:200]))

    def _deny(self, code, msg):
        body = ('{"error":"%s"}' % msg).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _go(self):
        raw = self.path
        path, _, query = raw.partition("?")
        if BAD.search(raw) or not PATH_RE.match(path):
            return self._deny(403, "path not allowed")
        if self.headers.get("Transfer-Encoding"):
            return self._deny(411, "chunked bodies not accepted; send Content-Length")
        cl = (self.headers.get("Content-Length") or "0").strip()
        if not cl.isdigit():
            return self._deny(400, "bad content-length")
        n = int(cl)
        if n > REQ_MAX:
            return self._deny(413, "request too large")
        body = self.rfile.read(n) if n else b""
        if len(body) != n:
            return self._deny(400, "short body")
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in PASS and k.lower() != "content-length"}
        hdrs["Content-Length"] = str(len(body))
        hdrs["Host"] = UP.netloc
        try:
            c = http.client.HTTPConnection(UP.hostname, UP.port or 80, timeout=30)
            c.request(self.command, raw, body=body, headers=hdrs)
            r = c.getresponse()
            data = r.read(RESP_MAX + 1)
        except (OSError, http.client.HTTPException):
            return self._deny(502, "upstream unreachable")
        if len(data) > RESP_MAX:
            return self._deny(502, "upstream response too large")
        if 300 <= r.status < 400:
            return self._deny(502, "upstream redirect refused")
        self.send_response(r.status)
        ct = r.getheader("Content-Type")
        if ct:
            self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _guarded(self):
        try:
            self._go()
        except (OSError, http.client.HTTPException):
            pass             # the client went away; nothing to answer

    do_GET = _guarded
    do_POST = _guarded

    def _no(self):
        self._deny(405, "method not allowed")

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _no


ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
