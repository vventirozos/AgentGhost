"""§4DW (2026-08-29) — HTTP-surface hardening.

Five findings from the HTTP-surface review, pinned by BEHAVIOUR rather than
by source text. Each test names the world in which it fails:

1. `redact_text` never touched a credential-bearing QUERY PARAMETER, which
   is exactly how the master key reached four world-readable files
   (44 + 40 + 35 + 1 hits). Fails if `url_query_secret` is removed — or if
   it is widened until it eats ordinary prose.
2. The interface page kept `?key=` in the URL bar and sent it onward as a
   Referer. Fails if the `history.replaceState` scrub or `Referrer-Policy`
   is dropped.
3. `add_subscription` accepted any `https://` URL, making every later
   `broadcast()` a stored SSRF with the daemon's network position. Fails if
   the allowlist is removed from EITHER the ingest or the egress side.
4. `/docs`, `/redoc` and `/openapi.json` published the full route map to
   anyone who could open the port. Fails if the built-ins come back.
5. The agent app had no request-body cap at all (a 150 MB POST moved live
   RSS 509 -> 960 MB). Fails if the middleware is unwired or the counting
   receive stops counting.
"""
import io
import json
import re
import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

from fastapi.testclient import TestClient  # noqa: E402

from ghost_agent.distill.redact import redact_text  # noqa: E402


# ── 1. the redactor covers credential-bearing query parameters ────────────

_FAKE_KEY = "K" * 64


@pytest.mark.parametrize("line", [
    f'127.0.0.1 - "GET /?key={_FAKE_KEY} HTTP/1.1" 200 -',
    f'INFO:     10.0.0.1:1 - "WebSocket /ws?key={_FAKE_KEY}" [accepted]',
    f'GET /manifest.webmanifest?key={_FAKE_KEY}',
    f'https://host/p?a=1&token={_FAKE_KEY}&b=2',
    f'https://host/p?api_key={_FAKE_KEY}',
    f'https://host/p?access_token={_FAKE_KEY}',
    f'https://host/p?SECRET={_FAKE_KEY}',
])
def test_query_credentials_are_redacted(line):
    """Every shape actually observed in the leaked logs, plus its siblings.

    Fails if `url_query_secret` is deleted: before it existed, all seven of
    these round-tripped the 64-char key verbatim.
    """
    out = redact_text(line)
    # ⚠ NOT just `_FAKE_KEY not in out`. That, plus `"<REDACTED>" in out`,
    # BOTH hold under a PARTIAL redaction: making the value quantifier lazy
    # leaves `?key=<REDACTED>KKK…KKK` — 63 of the 64 characters still in the
    # log — and the pin passed. What has to be asserted is that no run of
    # key material survives anywhere in the line.
    #
    # A whole-string equality would be wrong here in the other direction:
    # other legitimate rules fire on these lines too (the `ipv4` rule
    # rewrites the client address in the access-log shape), so the expected
    # output is not simply `line.replace(key, "<REDACTED>")`.
    runs = [r for r in re.findall(r"K+", out) if len(r) > 8]
    assert not runs, f"a run of {len(runs[0])} key characters survived: {out}"
    assert "=<REDACTED>" in out, f"parameter not redacted: {out}"


@pytest.mark.parametrize("line", [
    "the key: value pair is in the dict",
    "monkey=banana",
    "https://ahmia.fi/search/?q=onion+markets",
    "passkey holders are unaffected",
    "turnkey=false",
])
def test_ordinary_text_is_not_over_redacted(line):
    """The counterweight. `form_secret_assignment` deliberately refuses a
    bare `key=` because prose says `key: value` constantly; the query rule
    is allowed to be greedier ONLY because `[?&]` anchors it. Fails if the
    anchor is dropped, or if `key` is added to the prose rule."""
    assert redact_text(line) == line


def test_redaction_is_anchored_to_a_query_string():
    """`?key=` is redacted; the SAME token without the anchor is not. If
    these two ever agree, the rule stopped depending on its anchor."""
    anchored = redact_text(f"/page?key={_FAKE_KEY}")
    bare = redact_text(f"key={_FAKE_KEY}")
    assert _FAKE_KEY not in anchored
    assert bare == f"key={_FAKE_KEY}"


def test_importing_main_installs_the_filter_by_itself():
    """The guarantee must not depend on which entry point ran.

    A FRESH interpreter imports the module and nothing else — no call to
    `install_access_log_redaction`, no `main()`. Fails if the install moves
    back inside `main()`: the first version of this pin called the
    installer itself and therefore could not tell whether production ever
    did (mutation M03 survived it).
    """
    import subprocess
    import sys as _sys
    src = (
        "import logging\n"
        "import ghost_agent.main as m\n"
        "n = sum(1 for f in logging.getLogger('uvicorn.access').filters\n"
        "        if isinstance(f, m._RedactAccessLog))\n"
        "rec = logging.LogRecord('uvicorn.access', 20, '', 0, '%s',\n"
        "                        ('/?key=" + _FAKE_KEY + "',), None)\n"
        "for f in logging.getLogger('uvicorn.access').filters: f.filter(rec)\n"
        "print(n, '" + _FAKE_KEY + "' in rec.getMessage())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_ROOT / "src"),
           "GHOST_API_KEY": "test-ghost-key"}
    out = subprocess.run([_sys.executable, "-c", src], capture_output=True,
                         text=True, env=env, cwd=str(_ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    last = out.stdout.strip().splitlines()[-1]
    count, leaked = last.split()
    assert count == "1", f"import installed {count} filters"
    assert leaked == "False", "key survived a uvicorn record after import"


def test_agent_access_log_filter_scrubs_a_uvicorn_record():
    """uvicorn's access logger never calls pretty_log, so the operator-stream
    redaction did not reach it — and `log_config=None` gives it root's
    handlers. Fails if the filter stops touching `record.args`."""
    import logging
    import ghost_agent.main  # noqa: F401 — import is what installs it

    lg = logging.getLogger("uvicorn.access")
    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", "GET", f"/?key={_FAKE_KEY}", "1.1", 200), None)
    for f in lg.filters:
        f.filter(rec)
    assert _FAKE_KEY not in rec.getMessage()
    assert "<REDACTED>" in rec.getMessage()


def test_access_log_filter_is_idempotent():
    """Installed at import AND callable directly; a second install must not
    stack a second filter (which would double-scan every log line)."""
    import logging
    from ghost_agent.main import install_access_log_redaction, _RedactAccessLog

    lg = logging.getLogger("uvicorn.access")
    install_access_log_redaction()
    install_access_log_redaction()
    n = sum(1 for f in lg.filters if isinstance(f, _RedactAccessLog))
    assert n == 1, f"filter stacked {n} times"


def test_access_log_filter_never_raises():
    """Logging must not be able to take down the server. Fails if the bare
    `except` around the redaction is narrowed or removed."""
    import logging
    from ghost_agent.main import _RedactAccessLog

    class Exploding(str):
        def __str__(self):  # pragma: no cover - defensive
            raise RuntimeError("boom")

    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                            "%s", (Exploding("x"),), None)
    assert _RedactAccessLog().filter(rec) is True


# ── 2. the page does not keep the key in the URL bar ──────────────────────

@pytest.fixture(scope="module")
def iface():
    import interface.server as server
    return server


def test_index_scrubs_the_key_from_the_url(iface):
    """The key must ARRIVE as a query parameter (a browser cannot set a
    header on a top-level navigation) but must not STAY there. Fails if the
    replaceState snippet is removed: the URL then travels onward as a
    Referer, which is how it reached a sandbox service's 0644 access log.

    Compares the WHOLE constant, not the tokens inside it. The first
    version of this test asserted `"history.replaceState" in body` and
    survived a mutant that commented the snippet out — the tokens were
    still rendered, inside a comment, doing nothing.
    """
    c = TestClient(iface.app)
    r = c.get(f"/?key={iface.GHOST_API_KEY}")
    assert r.status_code == 200
    assert iface.URL_KEY_SCRUB_SCRIPT.strip() in r.text
    # ⚠ The assertion above is a TAUTOLOGY on its own: the page is BUILT by
    # concatenating that same constant, so both sides move together and it
    # can only detect "not injected", never "injected and inert". Two
    # mutants proved it — replacing the new URL with `location.href` (the
    # deletion applied to a discarded object) and wrapping the body in
    # `if(false&&...)` both kept every token and left the key in the address
    # bar. What the property actually needs is EXECUTION; see
    # `test_url_scrub_actually_removes_the_key_in_a_js_engine`.


def test_url_scrub_script_is_executable_javascript(iface):
    """The constant itself must be a live script, not a commented-out or
    truncated one. Balanced braces and no comment opener: a mutant that
    neutralises the snippet by wrapping it in `/* */` fails here."""
    js = iface.URL_KEY_SCRUB_SCRIPT
    assert js.startswith("<script>") and js.rstrip().endswith("</script>")
    body = js[len("<script>"):js.rstrip().rindex("</script>")]
    assert "/*" not in body and "//" not in body
    assert body.count("{") == body.count("}")
    assert body.count("(") == body.count(")")
    # ⚠ These are the tokens of the CURRENT implementation. The guard used
    # to be `location.search.includes("key=")`, which is false for `?%6bey=`
    # — a spelling Starlette decodes and authenticates exactly like `?key=`,
    # so the scrub silently did nothing for it. `searchParams` decodes
    # names, so asking it is encoding-independent.
    for token in ('searchParams.has("key")', 'searchParams.delete("key")',
                  "history.replaceState", "catch"):
        assert token in body, token
    assert "location.search.includes" not in body, (
        "the guard is string-matching the raw query again; `?%6bey=` "
        "authenticates and would not be scrubbed")


def test_index_sends_no_referrer(iface):
    """The other half of the same leak. Fails if the header is dropped."""
    c = TestClient(iface.app)
    r = c.get(f"/?key={iface.GHOST_API_KEY}")
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"


def test_index_still_delivers_the_key_to_the_page(iface):
    """The counterweight: scrubbing the URL must not break authentication.
    Fails if the scrub is applied BEFORE `window.GHOST_API_KEY` is set."""
    c = TestClient(iface.app)
    body = c.get(f"/?key={iface.GHOST_API_KEY}").text
    i_key = body.index("window.GHOST_API_KEY")
    i_scrub = body.index(iface.URL_KEY_SCRUB_SCRIPT.strip())
    assert i_key < i_scrub, "URL is scrubbed before the key is captured"


# ── 3. push endpoints are confined to real push services ──────────────────

_PUSH_OK = [
    "https://web.push.apple.com/QAbc123",
    "https://updates.push.services.mozilla.com/wpush/v2/xyz",
    "https://fcm.googleapis.com/fcm/send/xyz",
    "https://sub.notify.windows.com/w/?token=xyz",
]
_PUSH_BAD = [
    "http://web.push.apple.com/x",              # not https
    "https://100.93.181.31:8000/x",             # the tailnet
    "https://127.0.0.1/x",                      # loopback
    "https://[::1]/x",                          # loopback v6
    "https://web.push.apple.com.evil.tld/x",    # suffix confusion
    "https://web.push.apple.com@evil.tld/x",    # userinfo confusion
    "https://web.push.apple.com:8000/x",        # explicit port
    "https://evil.tld/?u=https://web.push.apple.com",
    "ftp://web.push.apple.com/x",
    "",
]


@pytest.mark.parametrize("url", _PUSH_OK)
def test_push_allowlist_admits_real_services(url):
    from interface import webpush_notify
    assert webpush_notify._push_endpoint_allowed(url)


@pytest.mark.parametrize("url", _PUSH_BAD)
def test_push_allowlist_rejects_everything_else(url):
    """Fails if the check goes back to `startswith("https://")`: five of
    these ten pass that test."""
    from interface import webpush_notify
    assert not webpush_notify._push_endpoint_allowed(url)


def test_add_subscription_refuses_a_non_push_host(tmp_path, monkeypatch):
    from interface import webpush_notify
    monkeypatch.setattr(webpush_notify, "_SUBS_FILE", tmp_path / "subs.json")
    keys = {"p256dh": "p", "auth": "a"}
    assert webpush_notify.add_subscription(
        {"endpoint": "https://web.push.apple.com/ok", "keys": keys}) is True
    assert webpush_notify.add_subscription(
        {"endpoint": "https://100.64.0.1:8000/pwn", "keys": keys}) is False


def test_broadcast_refuses_a_stored_non_push_endpoint(tmp_path, monkeypatch):
    """The egress is the SSRF sink, so the allowlist is enforced there too.

    The subscriptions file is plain JSON on disk and predates the ingest
    check, so an ingest-only guard leaves every already-stored endpoint
    unchecked. Fails if the `continue` in `broadcast()` is removed: the
    hostile endpoint is then passed to `webpush()`.
    """
    from interface import webpush_notify

    subs_file = tmp_path / "subs.json"
    keys = {"p256dh": "p", "auth": "a"}
    hostile = "https://100.64.0.1:8000/pwn"
    good = "https://web.push.apple.com/ok"
    subs_file.write_text(json.dumps({
        hostile: {"endpoint": hostile, "keys": keys},
        good: {"endpoint": good, "keys": keys},
    }))
    monkeypatch.setattr(webpush_notify, "_SUBS_FILE", subs_file)
    monkeypatch.setattr(webpush_notify, "vapid_config",
                        lambda: {"private_key_pem": "x",
                                 "public_key_b64url": "y",
                                 "sub": "mailto:t@t"})
    monkeypatch.setattr(webpush_notify, "_vapid_key_object", lambda: object())

    attempted = []

    class _FakeWebPushException(Exception):
        pass

    fake = types.ModuleType("pywebpush")
    fake.WebPushException = _FakeWebPushException

    def _webpush(*, subscription_info, **kw):
        attempted.append(subscription_info["endpoint"])

    fake.webpush = _webpush
    monkeypatch.setitem(sys.modules, "pywebpush", fake)

    webpush_notify.broadcast("t", "b")
    assert attempted == [good], f"broadcast reached {attempted}"


# ── 4. the schema is not published ────────────────────────────────────────

def _agent_client():
    from ghost_agent.api.app import create_app
    app = create_app()
    args = types.SimpleNamespace(api_key="test-ghost-key")
    app.state.agent = types.SimpleNamespace(
        context=types.SimpleNamespace(args=args))
    app.state.args = args
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_agent_schema_surface_requires_the_key(path):
    """Fails if `docs_url`/`redoc_url`/`openapi_url` go back to their
    defaults: FastAPI then serves all three to anyone who opens the port,
    and this app binds 0.0.0.0 whenever a key is configured."""
    r = _agent_client().get(path)
    assert r.status_code in (401, 403), f"{path} -> {r.status_code}"


def test_agent_openapi_is_reachable_with_the_key():
    """The counterweight: gating must not mean deleting. Also proves the
    route is registered BEFORE `router`'s `/{path:path}` catch-all — the
    first attempt at this landed after it and every schema request went to
    the upstream proxy instead (502)."""
    r = _agent_client().get("/openapi.json",
                            headers={"X-Ghost-Key": "test-ghost-key"})
    assert r.status_code == 200, r.text[:200]
    assert len(r.json().get("paths", {})) > 10


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_interface_docs_ui_is_gone(path, iface):
    r = TestClient(iface.app).get(path)
    assert r.status_code == 404


def test_interface_openapi_requires_the_key(iface):
    c = TestClient(iface.app)
    assert c.get("/openapi.json").status_code in (401, 403)
    r = c.get("/openapi.json", headers={"X-Ghost-Key": iface.GHOST_API_KEY})
    assert r.status_code == 200
    assert len(r.json().get("paths", {})) > 10


# ── 5. the agent caps request bodies ──────────────────────────────────────

def test_agent_rejects_an_oversized_declared_body():
    """413 on Content-Length alone — the body is never read. Fails if the
    middleware is unwired from `create_app`."""
    c = _agent_client()
    r = c.post("/api/feedback",
               content=b"x" * (11 * 1024 * 1024),
               headers={"X-Ghost-Key": "test-ghost-key",
                        "Content-Type": "application/json"})
    assert r.status_code == 413
    assert "too large" in r.json()["error"]


def test_agent_rejects_an_oversized_chunked_body():
    """No Content-Length to trust, so the bytes are counted as they arrive.
    Fails if `counting_receive` stops accumulating — a Content-Length-only
    check is trivially bypassed by chunked transfer-encoding."""
    def chunks():
        for _ in range(12):
            yield b"y" * (1024 * 1024)

    r = _agent_client().post(
        "/api/feedback", content=chunks(),
        headers={"X-Ghost-Key": "test-ghost-key",
                 "Content-Type": "application/json"})
    assert r.status_code == 413


def test_agent_upload_path_gets_the_larger_cap():
    """The cap is path-aware: a 20 MB upload is over the JSON ceiling but
    well under the file ceiling, so it must NOT be a 413. Fails if
    `cap_for_path` collapses to a single constant."""
    r = _agent_client().post(
        "/api/upload",
        files={"file": ("a.bin", io.BytesIO(b"z" * (20 * 1024 * 1024)))},
        headers={"X-Ghost-Key": "test-ghost-key"})
    # `!= 413` is the negation of ONE value, which a completely broken route
    # also satisfies (this stub agent 500s). Assert the cap specifically let
    # it through, by comparing against the cap the path is supposed to get.
    from ghost_agent.api import body_limit as _bl
    assert 20 * 1024 * 1024 < _bl.cap_for_path("/api/upload")
    assert 20 * 1024 * 1024 > _bl.max_json_bytes()
    assert r.status_code != 413


def test_agent_rejects_a_body_over_the_upload_cap():
    r = _agent_client().post(
        "/api/upload", content=b"z" * (102 * 1024 * 1024),
        headers={"X-Ghost-Key": "test-ghost-key",
                 "Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_body_cap_leaves_bodyless_methods_alone():
    """A GET must not be routed through the counting receive at all."""
    from ghost_agent.api.body_limit import _CAPPED_METHODS
    assert "GET" not in _CAPPED_METHODS
    r = _agent_client().get("/health", headers={"X-Ghost-Key": "test-ghost-key"})
    assert r.status_code != 413


def test_cap_for_path_is_path_aware():
    from ghost_agent.api import body_limit as bl
    assert bl.cap_for_path("/api/upload") > bl.cap_for_path("/api/feedback")
    assert (bl.cap_for_path("/api/workspace/save")
            > bl.cap_for_path("/api/feedback"))
    assert bl.cap_for_path("/anything/else") == bl.max_json_bytes()


@pytest.mark.parametrize("raw", ["", "   ", "not-a-number", "-5", "0"])
def test_body_cap_env_override_cannot_crash_the_process(raw, monkeypatch):
    """These run at import inside a KeepAlive LaunchDaemon: a raising
    override is an infinite crash-relaunch loop with no endpoint left to
    report it. Fails if `_env_num` goes back to a bare `int()`."""
    from ghost_agent.api import body_limit as bl
    monkeypatch.setenv("GHOST_AGENT_MAX_JSON_BYTES", raw)
    assert bl.max_json_bytes() == 10 * 1024 * 1024


def test_body_cap_env_override_is_honoured(monkeypatch):
    """The counterweight to the test above: a VALID override must actually
    take effect, or the guard is a constant wearing a knob."""
    from ghost_agent.api import body_limit as bl
    monkeypatch.setenv("GHOST_AGENT_MAX_JSON_BYTES", "12345")
    assert bl.max_json_bytes() == 12345
    assert bl.cap_for_path("/api/feedback") == 12345


# ── 5b. middleware-level pins the HTTP client cannot express ──────────────
#
# Three properties of the body cap are invisible from a TestClient, because
# the request still ends in 413 whether or not they hold. Mutation showed
# exactly that: disabling the Content-Length check, dropping DELETE from the
# capped methods, and demoting `BodyTooLarge` to a plain Exception ALL left
# the HTTP-level tests green. These drive the ASGI callable directly, where
# the difference is observable.

import asyncio  # noqa: E402


def _drive(app_callable, *, method="POST", path="/api/feedback",
           headers=None, chunks=(), inner=None):
    """Run one ASGI request through the middleware.

    Returns (status, body_was_read, sent_messages). `inner` is the app the
    middleware wraps; the default reads the body to exhaustion.
    """
    from ghost_agent.api.body_limit import BodySizeLimitMiddleware

    read_calls = {"n": 0}
    queue = list(chunks)

    async def receive():
        read_calls["n"] += 1
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body,
                    "more_body": bool(queue)}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    async def default_inner(scope, rcv, snd):
        while True:
            msg = await rcv()
            if not msg.get("more_body"):
                break
        await snd({"type": "http.response.start", "status": 200,
                   "headers": []})
        await snd({"type": "http.response.body", "body": b"ok"})

    scope = {"type": "http", "method": method, "path": path,
             "headers": headers or []}
    app = BodySizeLimitMiddleware(inner or default_inner)
    asyncio.run(app(scope, receive, send))
    status = next((m["status"] for m in sent
                   if m["type"] == "http.response.start"), None)
    return status, read_calls["n"] > 0, sent


def test_declared_oversize_is_rejected_without_reading_the_body():
    """The WHOLE point of the Content-Length branch: refuse before a single
    byte is received. The counting receive would also produce a 413, so an
    HTTP-level test cannot tell the two apart — a mutant that disabled this
    branch passed every one of them. Here the difference is the thing
    asserted."""
    over = 11 * 1024 * 1024
    status, body_read, _ = _drive(
        None, headers=[(b"content-length", str(over).encode())],
        chunks=[b"x" * over])
    assert status == 413
    assert not body_read, "oversized body was read despite Content-Length"


def test_undeclared_oversize_is_still_counted():
    """The counterweight: with no Content-Length there is nothing to trust,
    so the bytes must be counted as they arrive."""
    status, body_read, _ = _drive(
        None, headers=[], chunks=[b"y" * (1024 * 1024)] * 12)
    assert status == 413
    assert body_read


def test_delete_bodies_are_capped():
    """`/api/delete` reads a DELETE body. Dropping DELETE from the capped
    methods leaves that one route unbounded, and no HTTP-level test in this
    file exercises it — mutation M19 survived until this pin existed."""
    over = 11 * 1024 * 1024
    status, _, _ = _drive(
        None, method="DELETE", path="/api/delete",
        headers=[(b"content-length", str(over).encode())])
    assert status == 413


def test_a_body_under_the_cap_passes_through_untouched():
    """The counterweight to every 413 above: the middleware must be
    invisible to a normal request, on both a capped and an uncapped
    method."""
    for method in ("POST", "DELETE", "GET"):
        status, _, _ = _drive(None, method=method, chunks=[b"small"])
        assert status == 200, method


def test_overflow_survives_a_broad_except_in_the_app():
    """`BodyTooLarge` is a BaseException on purpose.

    FastAPI's body parsing wraps the read in `except Exception`, and the
    handlers add their own broad excepts; as a plain Exception the overflow
    would be swallowed and remapped to a 400 or a 502 — or silently
    ignored, leaving the oversized body accepted. This inner app catches
    `Exception` and answers 200, so a demoted BodyTooLarge yields 200 and a
    correct one still yields 413.
    """
    async def swallowing_inner(scope, rcv, snd):
        try:
            while True:
                msg = await rcv()
                if not msg.get("more_body"):
                    break
        except Exception:                      # noqa: BLE001 — the point
            pass
        await snd({"type": "http.response.start", "status": 200,
                   "headers": []})
        await snd({"type": "http.response.body", "body": b"swallowed"})

    status, _, _ = _drive(None, headers=[],
                          chunks=[b"y" * (1024 * 1024)] * 12,
                          inner=swallowing_inner)
    assert status == 413, (
        "a broad `except Exception` in the app swallowed the overflow")
