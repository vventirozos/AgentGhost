#!/usr/bin/env python3
"""In-sandbox Playwright runner. Invoked by tools/browser.py.

Accepts a single argv: a JSON-encoded op dict of the form:
  {"op": "navigate"|"extract_text"|"screenshot"|"click"|"close",
   ...op-specific fields...,
   "proxy": "socks5://host:port" | null,
   "profile_dir": "/workspace/.browser_profile",
   "timeout_ms": 30000}

Cross-call continuity: launch_persistent_context only persists cookies
and localStorage, NOT open pages. To let the LLM chain ops without
re-passing ``url`` each time, the runner writes the final navigated
URL to ``<profile_dir>/.last_url`` after every successful navigation
and reads it when an op is invoked without ``url``. The sidecar lives
inside the profile dir, so the ``close`` op (which rmtree's the whole
directory) wipes it for free.

Emits exactly one output line prefixed with `[BROWSER_OK] ` (JSON
payload) or `[BROWSER_ERR] ` (error string) so the caller can parse
reliably. Exits 0 on success, 1 on failure.
"""
import asyncio
import ipaddress
import json
import os
import time
import shutil
import socket
import sys
import traceback
from urllib.parse import urlparse, unquote

from playwright.async_api import async_playwright

# Loopback ports of SUPERVISED sandbox services (sandbox/services.py) —
# populated from the op payload in main(). A service the agent itself
# started (e.g. a dev server it is building) must be reachable at
# http://127.0.0.1:<port> or the whole "host an app, drive it with the
# browser" capability is dead; everything else loopback stays blocked.
ALLOWED_LOCAL_PORTS = set()

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _is_allowed_local_service(host, port):
    return (str(host or "").lower() in _LOOPBACK_HOSTS
            and port in ALLOWED_LOCAL_PORTS)


def _proxy_bypass_for_ports(ports):
    """Chromium proxy-bypass value so a supervised loopback service is reached
    DIRECTLY instead of through the SOCKS/Tor proxy.

    Returns the ``<loopback>`` token when any service port is allowed, else "".

    Why `<loopback>` and not a `host:port` list: EMPIRICALLY VERIFIED
    (2026-07-12, Playwright/Chromium against a dead SOCKS proxy) —

        no bypass                                     -> ERR_PROXY (loopback IS proxied)
        "127.0.0.1:PORT,localhost:PORT,[::1]:PORT"   -> ERR_PROXY (IGNORED)
        "127.0.0.1"                                   -> REACHED
        "<loopback>"                                  -> REACHED

    Chromium's `--proxy-bypass-list` does NOT honour `host:port` entries for
    the direct-vs-proxy decision — the first version of this fix (host:port)
    silently did nothing, so every navigate to a hosted service still died
    with ERR_SOCKS_CONNECTION_FAILED. `<loopback>` bypasses all loopback,
    which is SAFE: loopback traffic never leaves the box (no Tor anonymity
    concern), public traffic still goes through Tor, and PORT-LEVEL access is
    still enforced by the in-runner SSRF interceptor (`_ssrf_should_block`,
    installed on `ctx.route("**/*")`), which blocks any loopback request whose
    port is not in ALLOWED_LOCAL_PORTS. So the proxy layer decides "direct vs
    Tor" and the SSRF layer decides "allowed vs blocked" — two independent
    gates, not one relying on the other.
    """
    return "<loopback>" if ports else ""


# ── SSRF guard (runner side) ──────────────────────────────────────────
# The host-side guard (_browser_blocked_url) only vets the INITIAL url, but
# the sandbox's network can reach local services (it runs on its own bridge
# netns today — HOST networking in earlier versions), so a page navigated to an untrusted
# public host that 302-redirects to an internal address — http://127.0.0.1:9051
# (Tor control), 169.254.169.254 (cloud metadata), a LAN host — would reach
# host-local services. Chromium does NOT re-vet redirects and bypasses the
# proxy for loopback. This request interceptor runs at the navigation layer and
# aborts EVERY offending request, so it covers redirects, cross-origin
# subresources (blind-SSRF <img>/<iframe>/fetch), and the .last_url
# re-navigation. It enforces THREE rules (see _ssrf_should_block):
#   1. http(s) to an internal HOST STRING (literal IP / known local name).
#   2. http(s) whose host RE-RESOLVES to an internal IP — the DNS-rebind case
#      (top-level host vetted public, a later fetch flips to internal). Non-Tor
#      only: over Tor DNS is at the exit node, so a local lookup both leaks the
#      query and means nothing (mirrors url_ssrf_reason's `resolve` flag).
#   3. file:// whose resolved real path ESCAPES the sandbox subtree — a
#      container-read SSRF (file:///etc/passwd, any path above the mount).
#      file:// inside the subtree (self-play fixtures) still passes.
_SSRF_BLOCKED_HOSTNAMES = frozenset({
    "localhost", "ip6-localhost", "ip6-loopback",
    "metadata.google.internal", "metadata",
})


def _host_is_internal(host):
    """True if `host` is a loopback / private / link-local / reserved /
    multicast / unspecified / metadata target — an SSRF-relevant internal
    address. Classifies the host STRING only (no DNS, so no Tor leak)."""
    if not host:
        return False
    h = str(host).strip().lower().strip("[]")  # strip IPv6 brackets
    if h in _SSRF_BLOCKED_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _resolves_internal(host, port):
    """Best-effort: True if `host` DNS-resolves to an internal IP.

    Defeats DNS-rebind of a subresource/redirect host — the original name was
    vetted while it pointed at a public IP, but a later re-resolution flips it
    to an internal one, which the host-STRING check can't see. Only used in
    non-Tor mode (see _ssrf_should_block). A resolution failure returns False
    so a transient resolver hiccup can't brick a legitimate fetch (matches
    url_ssrf_reason)."""
    if not host or _host_is_internal(host):
        # A literal internal host is already caught by the string check; and a
        # host we can't resolve just falls through to "allow".
        return _host_is_internal(host)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    for info in infos:
        if _host_is_internal(info[4][0]):
            return True
    return False


def _file_escapes_sandbox(path, sandbox_root):
    """True if a file:// path resolves OUTSIDE the sandbox subtree.

    file:// renders self-play fixtures, but under HOST mounts a
    file:///etc/passwd (or any path above the mount, incl. `../` escapes and
    symlink hops) is a container-read SSRF. Allowed ONLY when the resolved
    REAL path stays within `sandbox_root` (the sandbox mount, derived from the
    profile dir). No declared root → not applicable, file:// passes (the
    interceptor's caller always declares one in production); an
    unresolvable/relative path fails CLOSED — a guard must not fail open."""
    if not sandbox_root:
        return False
    try:
        real = os.path.realpath(unquote(path or ""))
        root = os.path.realpath(sandbox_root)
        return os.path.commonpath([root, real]) != root
    except Exception:
        return True


def _ssrf_should_block(url, sandbox_root=None, anonymous=False):
    """Return True if this request must be aborted. about:/data: (inert) and
    an unparseable URL pass; the three real rules are documented on the guard
    block above. `sandbox_root` scopes file://; `anonymous` (Tor mode) skips
    the DNS re-resolution so no query leaks to the host resolver."""
    try:
        p = urlparse(str(url or ""))
    except Exception:
        return False
    scheme = (p.scheme or "").lower()
    if scheme == "file":
        return _file_escapes_sandbox(p.path, sandbox_root)
    if scheme not in ("http", "https"):
        return False
    host = p.hostname
    port = p.port or (443 if scheme == "https" else 80)
    # Narrow allowance: explicit-loopback URLs whose port belongs to a
    # supervised sandbox service (registry-driven, agent-opened). Literal
    # hosts only — no DNS involved, so no rebind surface.
    if _is_allowed_local_service(host, port):
        return False
    if _host_is_internal(host):
        return True
    if not anonymous:
        if _resolves_internal(host, port):
            return True
    return False


async def _install_ssrf_guard(ctx, sandbox_root=None, anonymous=False):
    """Register the request interceptor on a BrowserContext. Fail-safe: a
    classification is robust (never raises), and route-method errors degrade
    to continue so a guard bug can't brick every navigation.

    `sandbox_root` — the container mount file:// must stay within (/workspace).
    `anonymous` — Tor mode; skips DNS re-resolution (see _ssrf_should_block)."""
    async def _route(route):
        try:
            u = route.request.url or ""
        except Exception:
            u = ""
        if _ssrf_should_block(u, sandbox_root, anonymous):
            try:
                sys.stderr.write("[BROWSER_SSRF_BLOCK] " + str(u)[:200] + "\n")
                sys.stderr.flush()
            except Exception:
                pass
            try:
                await route.abort("blockedbyclient")
            except Exception:
                pass
            return
        try:
            await route.continue_()
        except Exception:
            pass
    await ctx.route("**/*", _route)

    # §4PC: `route()` is HTTP only — a WebSocket to a loopback port that the
    # proxy bypass makes reachable never met the guard (verified: a page's
    # `new WebSocket('ws://127.0.0.1:<other port>/')` reached the listener).
    # A blocked socket is simply never connected to a server.
    async def _ws_route(ws):
        try:
            u = ws.url or ""
        except Exception:
            u = ""
        # the guard reasons about http(s); a socket's ws(s) is the same host:port
        hu = ("http" + u[2:]) if u.lower().startswith("ws") else u
        if _ssrf_should_block(hu, sandbox_root, anonymous):
            try:
                sys.stderr.write("[BROWSER_SSRF_BLOCK] ws " + str(u)[:200] + "\n")
                sys.stderr.flush()
            except Exception:
                pass
            return
        try:
            ws.connect_to_server()
        except Exception:
            pass
    try:
        await ctx.route_web_socket("**", _ws_route)
    except Exception:
        pass

    # §4PC: `route()` is called for the FIRST url of a redirect chain only
    # (Playwright's own contract) — the hop goes straight to the network. The
    # hop cannot be aborted from here; it is DETECTED, recorded, and the op
    # fails BLOCKED instead of reading what the hop fetched.
    def _on_request(req):
        try:
            if req.redirected_from is not None and _ssrf_should_block(req.url or "", sandbox_root, anonymous):
                _SSRF_VIOLATIONS.append(str(req.url)[:200])
        except Exception:
            pass
    try:
        ctx.on("request", _on_request)
    except Exception:
        pass


_LAST_URL_FILENAME = ".last_url"
#: §4PC: redirect hops to blocked addresses seen during this op
_SSRF_VIOLATIONS: list = []


#: §4LN: the host keeps at most 256 K characters of exec output and the
#: result is ONE line — past that, the middle is cut and the JSON no longer
#: parses ("malformed OK payload", every retry the same). ASCII escaping
#: made it worse: a Greek page cost 6 characters per letter.
_EMIT_MAX_CHARS = 200_000


def _trim_payload(payload, limit=_EMIT_MAX_CHARS):
    """Shrink the largest text fields until the line fits; mark them."""
    def _fields(p):
        out = [(p, k) for k in ("text", "value") if isinstance(p.get(k), str)]
        for a in p.get("actions") or []:
            if isinstance(a, dict):
                out += [(a, k) for k in ("text", "value") if isinstance(a.get(k), str)]
        return out
    for _ in range(40):
        line = json.dumps(payload, ensure_ascii=False)
        if len(line) <= limit:
            return line
        fields = _fields(payload)
        if not fields:
            break
        holder, key = max(fields, key=lambda hk: len(hk[0][hk[1]]))
        cur = holder[key]
        if not cur:
            break
        holder[key] = cur[: max(0, len(cur) - (len(line) - limit) - 200)]
        holder["truncated"] = True
        payload["payload_trimmed"] = True
    for holder, key in _fields(payload):          # still too long: drop the texts, keep the shape
        holder[key] = ""
        holder["truncated"] = True
    payload["payload_trimmed"] = True
    return json.dumps(payload, ensure_ascii=False)


#: §4PC: the host hands the runner a per-run nonce; the sentinel lines carry
#: it, so a line a PAGE plants inside an error message (a getter that throws
#: "x\n[BROWSER_OK] {...}") is never read as the frame. Set from the op JSON.
_NONCE = ""


def _sentinel(kind: str) -> str:
    return f"[BROWSER_{kind}:{_NONCE}] " if _NONCE else f"[BROWSER_{kind}] "


def _emit_ok(payload):
    line = _sentinel("OK") + _trim_payload(payload) + "\n"
    try:
        sys.stdout.flush()
        sys.stdout.buffer.write(line.encode("utf-8"))      # whatever the container locale
        sys.stdout.buffer.flush()
    except AttributeError:                                  # a text-only stdout (tests)
        sys.stdout.write(line)
        sys.stdout.flush()


def _emit_err(msg):
    # ONE line: an exception message is page-influenced text (§4PC) — a
    # newline inside it would start a line of the page's choosing
    flat = " | ".join(part for part in str(msg).replace("\r", "").split("\n") if part.strip())
    sys.stdout.write(_sentinel("ERR") + flat + "\n")
    sys.stdout.flush()


def _set_last_url_file(name):
    """§4LN: the sidecar is per REQUEST (`.last_url.<req>`), so a url-less op
    re-opens this request's page — not whatever another conversation, a
    batch-mate or the verifier navigated last. The files are ~50 bytes;
    `close` wipes them with the profile (no tree walk here — §4GI)."""
    global _LAST_URL_FILENAME
    n = os.path.basename(str(name or "")).strip()
    if n.startswith(".last_url") and n.replace(".", "").replace("_", "").replace("-", "").isalnum():
        _LAST_URL_FILENAME = n


#: §4PC: the CONVERSATION's sidecar — a url-less op in a follow-up turn of
#: the same conversation re-opens its page; another conversation's never.
_LAST_URL_CONV_FILENAME = ""


def _set_last_url_conv_file(name):
    global _LAST_URL_CONV_FILENAME
    n = os.path.basename(str(name or "")).strip()
    if n.startswith(".last_url.c") and n.replace(".", "").replace("_", "").replace("-", "").isalnum():
        _LAST_URL_CONV_FILENAME = n
    else:
        _LAST_URL_CONV_FILENAME = ""


def _read_last_url(profile_dir):
    """Return the URL of the most recent navigation — THIS request's
    sidecar first, then the shared one (a url-less op in a follow-up turn
    of the same conversation; the result names the page it re-opened).
    Best-effort — any I/O error is treated as "no record"."""
    # §4PC: THIS request's sidecar only. The shared `.last_url` fallback let a
    # url-less op re-open a page another conversation, a batch-mate or the
    # verifier had asked for — while the schema promised it could not.
    for name in dict.fromkeys((_LAST_URL_FILENAME, _LAST_URL_CONV_FILENAME)):
        if not name:
            continue
        try:
            p = os.path.join(profile_dir, name)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass
    return None


def _write_last_url(profile_dir, url):
    """Persist the post-navigation URL for the next op. Best-effort;
    if disk write fails we still return the op's happy-path result
    (the caller either passed an explicit url next time, or gets a
    clear "no prior navigation recorded" error)."""
    if not url:
        return
    try:
        os.makedirs(profile_dir, exist_ok=True)
        for name in dict.fromkeys((_LAST_URL_FILENAME, _LAST_URL_CONV_FILENAME)):
            if not name:
                continue
            with open(os.path.join(profile_dir, name), "w", encoding="utf-8") as f:
                f.write(url)
    except Exception:
        pass


def _resolve_url_or_error(op, op_label):
    """Return an explicit or sidecar-resolved URL. If neither source
    has one, raise a clear error. Centralised so every nav-using op
    has identical fallback semantics."""
    url = op.get("url")
    if url:
        return url, False
    fallback = _read_last_url(op["profile_dir"])
    if fallback:
        return fallback, True
    raise ValueError(
        f"{op_label} needs a URL: pass `url=...` or call `operation=\"navigate\"` "
        "first (this op has no recorded last URL in the persistent profile)"
    )


def _chromium_args(proxy):
    """Docker-safe flags + DNS-over-SOCKS hardening.

    --no-sandbox / --disable-dev-shm-usage: required when Chromium runs
        non-root inside a container without a large /dev/shm.
    --host-resolver-rules: force every non-excluded hostname through
        the SOCKS proxy's DNS instead of the container's /etc/resolv.
        Without this, Chromium can resolve names locally (the classic
        SOCKS DNS leak) even when traffic itself goes via SOCKS.

        CRITICAL: the proxy server's OWN host must be excluded from the
        `MAP * ~NOTFOUND` rule. Otherwise Chromium runs the proxy host
        (e.g. `127.0.0.1`) through the same rule, resolves it to NOTFOUND,
        and cannot connect to the proxy at all — every navigation then
        dies with `net::ERR_PROXY_CONNECTION_FAILED`. `EXCLUDE localhost`
        does NOT cover the `127.0.0.1` literal, so we add an explicit
        EXCLUDE for the parsed proxy host. (Verified: with only
        `EXCLUDE localhost` a socks5://127.0.0.1 proxy fails; adding
        `EXCLUDE 127.0.0.1` makes it return 200.)
    WebRTC hardening: `--webrtc-ip-handling-policy=disable_non_proxied_udp`
        stops the browser from gathering host UDP/STUN candidates that would
        expose the real IP even when HTTP is proxied. We do NOT disable
        `WebRtcHideLocalIpsWithMdns` — that feature (on by default) REPLACES
        local interface IPs with mDNS hostnames in ICE candidates, so keeping
        it on hides local IPs from page JS. (The prior code disabled it, which
        was backwards — it exposed 192.168.x.x to a page.)
    """
    args = ["--no-sandbox", "--disable-dev-shm-usage"]
    if proxy:
        # `--host-resolver-rules` governs DNS RESOLUTION only. EXCLUDE
        # localhost keeps a bare "localhost" name from being NOTFOUND-mapped,
        # and the proxy host must be excluded so Chromium can reach the SOCKS
        # server itself (else every navigation is ERR_PROXY_CONNECTION_FAILED).
        # NOTE: this does NOT stop loopback traffic from being sent THROUGH the
        # proxy — that is the launch-time `proxy.bypass` list (see
        # _with_context), needed for supervised in-container services.
        excludes = ["EXCLUDE localhost"]
        try:
            from urllib.parse import urlparse
            phost = urlparse(proxy).hostname
        except Exception:
            phost = None
        if phost and phost != "localhost":
            excludes.append("EXCLUDE " + phost)
        args.append("--host-resolver-rules=MAP * ~NOTFOUND , " + " , ".join(excludes))
        # Restrictive WebRTC policy: no non-proxied UDP → no host candidate
        # leak. (Do NOT add --disable-features=WebRtcHideLocalIpsWithMdns; that
        # turns OFF the local-IP-hiding feature.)
        args.append("--force-webrtc-ip-handling-policy=disable_non_proxied_udp")
        args.append("--webrtc-ip-handling-policy=disable_non_proxied_udp")
    return args


# Cap captured diagnostics so a console.log-in-a-loop page can't blow the
# LLM context window. Uncaught exceptions (pageerror) are the gold — a
# silent `init()` crash is invisible in a screenshot but fires here — so
# they get their own (larger) budget than ordinary console chatter.
_MAX_JS_ERRORS = 12
_MAX_CONSOLE_MSGS = 25
_MAX_DIAG_CHARS = 600


def _attach_diagnostics(page):
    """Wire console + uncaught-exception listeners onto ``page`` and
    return the (js_errors, console_msgs) lists they fill.

    Playwright fires ``pageerror`` for every uncaught exception in page
    JS (the classic "TypeError: Cannot read properties of undefined" that
    crashes ``init()`` and leaves a loading screen frozen forever — the
    failure mode a screenshot can NEVER reveal) and ``console`` for every
    console.* call. Capturing both turns blind "the page is just stuck"
    debugging into "here's the exact exception and line". Listeners are
    sync callbacks; they only append, so they can't raise into the op."""
    js_errors: list = []
    console_msgs: list = []

    def _on_error(exc):
        if len(js_errors) >= _MAX_JS_ERRORS:
            return
        # `exc` is a playwright Error (has .message/.stack) on modern
        # builds but is a bare str on some — coerce defensively.
        msg = getattr(exc, "message", None) or str(exc)
        stack = getattr(exc, "stack", "") or ""
        text = (msg + ("\n" + stack if stack and stack != msg else "")).strip()
        js_errors.append(text[:_MAX_DIAG_CHARS])

    def _on_console(msg):
        try:
            mtype = msg.type
            mtext = msg.text
        except Exception:
            return
        # Errors/warnings are diagnostic signal; plain logs are usually
        # noise, so only keep them until the budget fills.
        if mtype not in ("error", "warning") and len(console_msgs) >= _MAX_CONSOLE_MSGS:
            return
        if len(console_msgs) >= _MAX_CONSOLE_MSGS * 2:
            return
        # Source location is the difference between an actionable error
        # and a guessing game: a script PARSE error ("Unexpected
        # identifier 't'") fires pageerror with NO stack frames — the
        # file:line:col only ever arrives on the console event. Without
        # it the agent has the message but not the line (the req-70
        # em-dash misdiagnosis). lineNumber/columnNumber are 0-based.
        loc = ""
        try:
            l = msg.location or {}
            u = str(l.get("url") or "")
            if u:
                u = "/".join(u.rstrip("/").split("/")[-2:])
                loc = f"{u}:{int(l.get('lineNumber', -1)) + 1}:{int(l.get('columnNumber', -1)) + 1}"
        except Exception:
            loc = ""
        entry = {"type": mtype, "text": str(mtext)[:_MAX_DIAG_CHARS]}
        if loc:
            entry["loc"] = loc
        console_msgs.append(entry)

    page.on("pageerror", _on_error)
    page.on("console", _on_console)
    return js_errors, console_msgs


async def _with_context(profile_dir, proxy, timeout_ms, op_fn, total_ms=None, on_timeout=None):
    """Open a persistent context, run op_fn(page), close cleanly.

    ``total_ms`` is the whole op's wall clock when it differs from the
    per-step ``timeout_ms`` (interact: the per-action share is
    ``timeout_ms``, the sequence gets ``total_ms``). ``on_timeout(page,
    budget_ms)`` may turn a budget overrun into a partial result instead
    of an error; returning None keeps the error.

    Every op funnels through here, so this is also where we attach the
    console/pageerror diagnostics: any dict an op returns is augmented
    with ``js_errors`` and ``console`` keys (only when non-empty) so the
    agent can SEE a silent JS crash instead of guessing at a frozen page.
    """
    os.makedirs(profile_dir, exist_ok=True)
    async with async_playwright() as p:
        launch_kwargs = dict(
            user_data_dir=profile_dir,
            headless=True,
            args=_chromium_args(proxy),
            # §4PC: `route()` never sees a service worker's fetches — a page
            # could register one in the shared profile and fetch past the
            # guard on later turns
            service_workers="block",
        )
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
            # PROXY BYPASS for supervised sandbox services (2026-07-12).
            #
            # `--proxy-server` routes EVERY http(s) request through SOCKS —
            # including `http://127.0.0.1:<port>`. Tor cannot route loopback,
            # so navigating to a service the agent itself started died with
            # `net::ERR_SOCKS_CONNECTION_FAILED`, which broke the whole
            # "host an app, then drive it with the browser" capability under
            # --mandatory-tor (observed live: the chess-coach service came up
            # on :8100 and every navigate failed).
            #
            # `--host-resolver-rules=… EXCLUDE localhost` did NOT cover this:
            # that flag governs DNS RESOLUTION only, never proxy routing. (The
            # self-play fixtures appeared to work solely because they are
            # file:// URLs, which never touch the proxy.)
            #
            # Bypass the proxy for loopback when a supervised service is
            # running, so the browser can reach it directly (Tor can't route
            # loopback). Port-level access stays enforced by the in-runner
            # SSRF interceptor — see _proxy_bypass_for_ports for the empirical
            # reason `<loopback>` is used rather than a host:port list.
            _bypass = _proxy_bypass_for_ports(ALLOWED_LOCAL_PORTS)
            if _bypass:
                launch_kwargs["proxy"]["bypass"] = _bypass
        ctx = await p.chromium.launch_persistent_context(**launch_kwargs)
        # Install the redirect/subresource SSRF guard BEFORE any navigation so
        # it covers the very first goto (and its redirect chain). The sandbox
        # subtree file:// must stay inside is the mount holding the profile dir
        # (/workspace); a set proxy means Tor mode, where we skip DNS
        # re-resolution so the guard itself leaks no query.
        sandbox_root = os.path.dirname(os.path.abspath(profile_dir))
        await _install_ssrf_guard(ctx, sandbox_root=sandbox_root, anonymous=bool(proxy))
        try:
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            page.set_default_timeout(timeout_ms)
            js_errors, console_msgs = _attach_diagnostics(page)
            # §4PC: one wall-clock budget for the WHOLE op. Per-step waits
            # summed past the exec kill (goto 30 s + probe + click + load
            # 30 s), and `page.evaluate` honours no default timeout at all —
            # a spinning getter kept the runner alive until `timeout -k`
            # killed it with no sentinel. The host's exec timeout is this
            # budget + 30 s, so a budget overrun is reported, not killed.
            # §4RX: interact passes its TOTAL here — the per-action share
            # used to be the clock for the whole sequence (21 actions got
            # 27 s of a 570 s budget, live 43fe95a2/ff6c2096).
            budget_ms = int(total_ms) if total_ms else int(timeout_ms)
            try:
                result = await asyncio.wait_for(op_fn(page), timeout=max(1.0, budget_ms / 1000.0 + 5.0))
            except asyncio.TimeoutError:
                result = on_timeout(page, budget_ms) if on_timeout else None
                if result is None:
                    # §4RI (request 43fe95a2): two interact calls timed out on a page
                    # with a 3.3 s boot animation; the model was never told that the
                    # budget is its own `timeout_ms` (divided per action for
                    # interact) and tried smaller selectors instead.
                    raise RuntimeError(
                        f"the op exceeded its {budget_ms} ms budget — the page kept the browser busy "
                        "(a slow load, a spinning script, or waits that add up); pass a larger timeout_ms "
                        "(for interact it is split across the actions, so a page with a boot animation or "
                        "a long wait needs more — e.g. timeout_ms=60000 — or fewer actions per call), "
                        "or use a different source or a smaller selector")
            if _SSRF_VIOLATIONS:
                raise RuntimeError(
                    "blocked: the page redirected to a non-public address "
                    f"({_SSRF_VIOLATIONS[0]}) — policy refused it; the page was not read")
            if isinstance(result, dict):
                # Errors can fire microtasks after the op's last await
                # (e.g. a setTimeout-deferred crash). A zero-ish settle
                # gives the event loop one more tick to drain them
                # without materially slowing the op.
                try:
                    await page.wait_for_timeout(50)
                except Exception:
                    pass
                if js_errors:
                    result.setdefault("js_errors", js_errors)
                if console_msgs:
                    result.setdefault("console", console_msgs)
            return result
        finally:
            try:
                await ctx.close()
            except Exception:
                pass


async def _probe_main_image(page):
    """The page's own main image, as a URL that can be downloaded.

    §4JY: asked to fetch a photo of a person, the agent could not get one —
    it GUESSED Wikimedia thumbnail paths (`800px-…`, `640px-…` → HTTP 400,
    because only some widths are pre-rendered), fell back to screenshotting
    the file PAGE, and ended up passing a 1280x2290 picture of a web page as
    a "photo". Every page already publishes the answer in `og:image`; that
    exact URL fetched 200 over Tor. So stop making the model guess and hand
    it the URL the SITE declares.

    ⚠ The obvious fallback — "the biggest <img>" — is a trap: HTML width/
    height attributes are INTRINSIC, so on the same page it selected a
    7651x5103 photo of a different politician entirely. Only RENDERED
    geometry says what the page is actually showing, so the fallback reads
    `getBoundingClientRect` and prefers what sits near the top of the
    document. `source` travels with the URL so the caller knows which of the
    two it got. Returns {} on any failure — never raises into the op."""
    js = """() => {
      // http(s) ONLY, and bounded. The og:image branch used to return
      // whatever the page declared: `data:` URIs (megabytes of base64 that
      // flood the caller's context), `javascript:` and `file:///etc/passwd`
      // all passed straight through to a line that says "download this".
      const abs = (u) => {
        try {
          const h = new URL(u, location.href);
          if (h.protocol !== 'http:' && h.protocol !== 'https:') return null;
          return h.href.length > 2048 ? null : h.href;
        } catch(e) { return null; }
      };
      const meta = (sel) => { const el = document.querySelector(sel); return el && el.content ? abs(el.content) : null; };
      const og = meta('meta[property="og:image"]') || meta('meta[name="og:image"]');
      const tw = meta('meta[name="twitter:image"]') || meta('meta[property="twitter:image"]');
      const declared = og || tw;
      const declaredSource = og ? 'og:image' : 'twitter:image';
      if (declared) return { url: declared, source: declaredSource };
      let best = null;
      for (const img of document.querySelectorAll('img')) {
        const src = img.currentSrc || img.src || '';
        if (!src || src.startsWith('data:') || /\.svg(\?|$)/i.test(src)) continue;
        let r; try { r = img.getBoundingClientRect(); } catch(e) { continue; }
        if (r.width < 128 || r.height < 128) continue;          // icons, spacers, tracking pixels
        const area = r.width * r.height;
        const score = area / (1 + Math.max(0, r.top + window.scrollY) / 1000);  // prefer near the top
        if (!best || score > best.score) best = { score, area, url: abs(src), w: Math.round(r.width), h: Math.round(r.height) };
      }
      return best ? { url: best.url, source: 'largest-rendered', w: best.w, h: best.h } : {};
    }"""
    try:
        # ⚠ BOUNDED. `page.evaluate` honours no default timeout, so a page with
        # a busy main thread (or a getter that spins) made this non-essential
        # probe hang the whole op until the subprocess was killed — losing the
        # page text that had already been collected. It is a nicety; it must
        # never cost the caller the result.
        out = await asyncio.wait_for(page.evaluate(js), timeout=3.0)
        return out if isinstance(out, dict) and out.get("url") else {}
    except Exception:
        return {}


async def _page_links(page, cap: int = 40):
    """Up to `cap` http(s) hrefs on the page, same-host first, deduplicated —
    so the model can FOLLOW a link instead of composing an address (§4PC:
    88 % of 404s were composed paths). Bounded; never costs the result."""
    js = """(cap) => {
      const out = []; const seen = new Set(); const here = location.host;
      const all = Array.from(document.querySelectorAll('a[href]'));
      const abs = (h) => { try { return new URL(h, location.href).href; } catch (e) { return ''; } };
      const pick = (sameHost) => { for (const a of all) {
        const u = abs(a.getAttribute('href')); if (!u || !/^https?:/.test(u)) continue;
        const h = u.split('#')[0]; if (seen.has(h)) continue;
        let host = ''; try { host = new URL(h).host; } catch (e) { continue; }
        if ((host === here) !== sameHost) continue;
        seen.add(h); out.push(h); if (out.length >= cap) return; } };
      pick(true); if (out.length < cap) pick(false); return out; }"""
    try:
        out = await asyncio.wait_for(page.evaluate(js, cap), timeout=3.0)
        return [str(u)[:300] for u in out if isinstance(u, str)][:cap] if isinstance(out, list) else []
    except Exception:
        return []


async def _probe_pre_interaction(page):
    """Detect a visible start / play / loading control on the page.

    A "Click to Play" button or a loading screen means the app has NOT
    started — a screenshot then shows the MENU, not the running app, and a
    claim that "the game works / renders" is grading the wrong thing (the
    exact false-positive seen live: the agent declared a Minecraft clone
    "fully functional" from a capture that still had the start modal up).
    Returns ``{pre_interaction: bool, controls: [text,…]}`` (or {} on any
    failure — never raises into the op)."""
    # ⚠ RAW string. Non-raw, Python ate the JS regex's word boundaries: `\b`
    # became U+0008 BACKSPACE, so `\bloading\b` could never match a loading
    # screen — the exact false-positive this probe exists to catch. It went
    # unnoticed because the test extracted the SOURCE TEXT (where `\b`
    # survives) while production evaluated the escape, so the harness and the
    # runtime were running different regexes (§4JY; the trap in
    # tests/helpers.py::eval_js, now also enforced by an AST-scoped extractor).
    js = r"""() => {
      const KW = /click to (play|start)|press (to )?start|start game|tap to (play|start)|enter game|^play$|^start$|^begin$|\bloading\b/i;
      const vis = (el) => { try { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width>4 && r.height>4 && s.visibility!=='hidden' && s.display!=='none' && parseFloat(s.opacity||'1')>0.1; } catch(e){ return false; } };
      const out = [];
      const sel = 'button,a,[role=button],h1,h2,[id*=start i],[id*=play i],[class*=start i],[class*=play i],[class*=overlay i],[class*=modal i]';
      for (const el of document.querySelectorAll(sel)) {
        const t = (el.innerText||el.textContent||'').trim();
        if (t && t.length<80 && KW.test(t) && vis(el)) { out.push(t.slice(0,40)); if(out.length>=3) break; }
      }
      return { pre_interaction: out.length>0, controls: out };
    }"""
    try:
        return await page.evaluate(js)
    except Exception:
        return {}


async def _body_excerpt(page, max_chars: int):
    """Rendered body innerText, capped. Returns (text, truncated, full_len).
    Every browser op opens a FRESH context + reloads the page, so a bare
    `navigate` that returned only status/title forced a second op
    (`extract_text`) that re-launched Playwright and re-fetched the SAME page
    over Tor. Returning a capped excerpt from navigate/click removes that
    second launch + full page-load + model turn from the dominant flow."""
    try:
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
    except Exception:
        return "", False, 0
    text = (text or "").strip()
    full_len = len(text)
    if full_len > max_chars:
        return text[:max_chars], True, full_len
    return text, False, full_len


async def op_navigate(op):
    url = op.get("url")
    if not url:
        raise ValueError("navigate requires 'url'")
    wait_until = op.get("wait_until", "load")  # load | domcontentloaded | networkidle
    # Capped text preview so the common navigate→extract_text→read flow
    # collapses to a single op. 8 KB default is a readable excerpt without the
    # 64 KB extract_text budget; pass nav_text_chars=0 to opt out.
    nav_text_chars = int(op.get("nav_text_chars", 8 * 1024))

    # §4LN: record the page ASKED for before loading it — a navigate that
    # times out or is refused must not leave the PREVIOUS page in the
    # sidecar for the next url-less extract to read as this one (live:
    # "now the Bytebase article" returned postgresql.org).
    _write_last_url(op["profile_dir"], url)

    async def run(page):
        resp = await page.goto(url, wait_until=wait_until)
        status = resp.status if resp else None
        final_url = page.url
        title = await page.title()
        _write_last_url(op["profile_dir"], final_url)
        result = {"status": status, "url": final_url, "title": title}
        result["links"] = await _page_links(page)
        if nav_text_chars > 0:
            text, truncated, full_len = await _body_excerpt(page, nav_text_chars)
            if text:
                result["text"] = text
                result["length"] = full_len
                result["truncated"] = truncated
        probe = await _probe_pre_interaction(page)
        if probe.get("pre_interaction"):
            result["pre_interaction"] = probe
        main_img = await _probe_main_image(page)
        if main_img:
            result["main_image"] = main_img
        return result

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


async def op_extract_text(op):
    selector = op.get("selector")  # optional CSS selector
    max_chars = int(op.get("max_chars", 64 * 1024))
    url, used_fallback = _resolve_url_or_error(op, "extract_text")
    wait_until = op.get("wait_until", "load")

    async def run(page):
        # ALWAYS navigate — since every op opens a fresh persistent
        # context, there's no "current page" to carry over across ops.
        # The LLM's ergonomic expectation (Step N continues where
        # Step N-1 left off) is honoured via the last_url sidecar.
        # §4HH: keep the DOCUMENT's response status — a 403 challenge page
        # read as STATUS: OK for 12.8% of corpus fetches because this op
        # threw the response away (op_navigate never did).
        resp = await page.goto(url, wait_until=wait_until)
        status = resp.status if resp else None
        if selector:
            el = await page.query_selector(selector)
            if not el:
                raise ValueError(f"selector {selector!r} did not match any element")
            text = (await el.inner_text()).strip()
        else:
            # `innerText` on <body> gives the rendered, whitespace-
            # collapsed text the user would see — a much cleaner target
            # than raw HTML for LLM consumption.
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        full_len = len(text)  # capture BEFORE truncating
        truncated = False
        offset = max(0, int(op.get("offset") or 0))          # §4PC: read past the cap in pages
        if offset:
            text = text[offset:]
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        return {
            "url": final_url,
            "status": status,
            "title": await page.title(),
            "text": text,
            "truncated": truncated,
            "offset": offset,
            "links": await _page_links(page),
            # Report the TRUE page length, not the capped length — else
            # `length` always equals max_chars on truncation, hiding how
            # much was dropped from any downstream "got the whole page?" check.
            "length": full_len,
            "used_last_url": used_fallback,
            # The page's own main image, so a caller after a PHOTO does not
            # have to guess thumbnail URLs (§4JY).
            **({"main_image": _mi} if (_mi := await _probe_main_image(page)) else {}),
        }

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


_CANDIDATES_JS = """() => {
  const out = [];
  const els = document.querySelectorAll('button, a[href], input, select, textarea, [onclick], [role=button]');
  for (const e of els) {
    if (out.length >= 12) break;
    let sel = e.tagName.toLowerCase();
    if (e.id) sel = '#' + e.id;
    else if (e.name) sel += '[name="' + e.name + '"]';
    else if (e.classList.length) sel += '.' + [...e.classList].slice(0, 2).join('.');
    const t = (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 40);
    out.push(sel + (t ? ' "' + t + '"' : ''));
  }
  return out;
}"""


_MISSING_SELECTORS: set = set()       # per runner process = per browser call

#: §4NW: a cheap signature of what the page shows — the URL plus a hash of
#: the live DOM (inline styles and classes included, so an opened window, a
#: toggled class or a typed digit all change it).
_PAGE_SIG_JS = """() => {
  const h = document.documentElement ? document.documentElement.outerHTML : '';
  let x = 5381;
  for (let i = 0; i < h.length; i++) { x = ((x << 5) + x + h.charCodeAt(i)) | 0; }
  return location.href + '#' + h.length + ':' + x;
}"""

#: §4NW: does the target (or an ancestor) bind a dblclick handler in markup
#: and no click handler? (`addEventListener` handlers are invisible here —
#: the page-changed check covers those.)
_DBLCLICK_ONLY_JS = """(el) => {
  // only where the page signature can see a change: the top document, no
  // shadow root, not a form control (its value/checked are properties)
  if (window !== window.top || el.getRootNode() !== el.ownerDocument) return false;
  if (el.closest('input, textarea, select, option, label, [contenteditable]')) return false;
  const d = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '')
    + (e.classList && e.classList.length ? '.' + Array.from(e.classList).slice(0, 2).join('.') : '');
  for (let e = el; e && e.getAttribute; e = e.parentElement) {
    if (e.getAttribute('onclick') || typeof e.onclick === 'function') return false;
    if (e.getAttribute('ondblclick') || typeof e.ondblclick === 'function') return e === el ? 'this element' : d(e);
    // a button/link between the target and an ancestor's ondblclick is a
    // control of its own (a close button inside a title bar — review)
    if (e !== el || el.matches('button, a[href]')) {
      if (e.matches('button, a[href], [role=button]')) return false;
    }
  }
  return false;
}"""


#: §4NW (req 92dfdd7d): what is ON TOP at the target's centre — the
#: auto-opened Terminal window covered the Calculator icon, so every forced
#: click landed on the window's title bar. "" when the target itself (or
#: its child/parent) is on top.
_COVERED_BY_JS = """(el) => {
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return 'nothing — the element has no size (hidden)';
  const root = el.getRootNode();      // a shadow root has its own hit-test
  const top = (root && root.elementFromPoint ? root : document).elementFromPoint(
    r.left + r.width / 2, r.top + r.height / 2);
  if (!top || el.contains(top) || top.contains(el)) return '';
  // a styled checkbox/switch: the knob over the input is inside its label
  if (el.labels && Array.from(el.labels).some((l) => l.contains(top))) return '';
  const via = top.closest && top.closest('label, a, button');
  if (via && via.contains(el)) return '';
  const d = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '')
    + (e.classList && e.classList.length ? '.' + Array.from(e.classList).slice(0, 2).join('.') : '');
  let host = top.parentElement;
  while (host && !host.id) host = host.parentElement;
  return d(top) + (host && host !== top ? ' inside ' + d(host) : '');
}"""

def _write_replacing(out_path: str, data: bytes) -> None:
    """Write ``data`` to a temp name next to ``out_path`` and rename it over
    the target. §4NW: re-writing a PNG on the macOS bind mount a few seconds
    after it was created failed EAGAIN (BlockingIOError) — the host still
    holds the fresh file; a rename never opens or truncates it. A short
    retry covers a rename that hits the same window."""
    import errno
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)   # page.screenshot(path=) did
    _d, _b = os.path.split(out_path)
    tmp = os.path.join(_d, f".{_b}.{os.getpid()}.tmp")    # hidden: a killed runner leaves no visible file
    for attempt in range(4):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out_path)
            return
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EBUSY) or attempt == 3:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            time.sleep(0.5)


def _screenshot_type(out_path: str) -> str:
    """The type `page.screenshot(path=…)` inferred from the extension — the
    bytes form does not infer it (review: a `.jpg` became PNG bytes)."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "jpeg"
    if ext in (".png", ""):
        return "png"
    raise ValueError(f"screenshot out_path must end in .png, .jpg or .jpeg (got {ext!r})")


async def _save_screenshot(page, out_path: str, full_page: bool) -> None:
    data = await page.screenshot(full_page=full_page, type=_screenshot_type(out_path))
    await asyncio.to_thread(_write_replacing, out_path, data)


async def _page_sig(page):
    try:
        return await page.evaluate(_PAGE_SIG_JS)
    except Exception:  # noqa: BLE001 — a navigation mid-read: no signature
        return None


async def _covered_note(page, sel, action):
    """§4NW: "" unless another element is on top of the target's centre —
    then where a FORCED click really went (a forced click on a covered icon
    lands on the cover: the Terminal's title bar). Only for `force`: a plain
    click does its own hit-test and either lands or fails (review: a sticky
    header or a fading overlay read as "covered" on clicks that landed)."""
    try:
        covered = await page.eval_on_selector(sel, _COVERED_BY_JS)
    except Exception:  # noqa: BLE001
        return ""
    if not covered:
        return ""
    return (f"the target is COVERED by {covered} — the {action} landed on THAT, not on "
            f"{sel!r}. Close or move what covers it first (e.g. its close button), or "
            "reach the target another way (a menu). `force` does not help: it clicks the "
            "same covered point.")


async def _dblclick_only(page, sel) -> str:
    """The element (it or an ancestor) whose ondblclick handler the target
    reaches with no click handler on the way — "" when none."""
    try:
        r = await page.eval_on_selector(sel, _DBLCLICK_ONLY_JS)
    except Exception:  # noqa: BLE001
        return ""
    if r is True:
        return "this element"
    return str(r) if isinstance(r, str) and r else ""


async def _dblclick_only_effect(page, sel, before, owner: str = ""):
    """§4NW (req 92dfdd7d): a `click` on a target whose markup reaches ONLY
    an ondblclick handler, after which the page did not change — the open
    handler never ran. Narrow on purpose (review: a generic "nothing
    changed" check fired on value/checkbox/canvas/iframe/shadow-DOM and
    slow-network clicks that worked). "" otherwise."""
    if before is None:
        return ""
    for wait in (0.15, 0.6):           # a handler may render a beat later
        await asyncio.sleep(wait)
        after = await _page_sig(page)
        if after is None or after != before:
            return ""                  # the page changed: see _dblclick_hint
    return (f"{_dbl_owner_phrase(owner)} reacts only to a DOUBLE-click (ondblclick) and this click "
            f"changed nothing — use {{\"action\":\"dblclick\",\"selector\":\"{sel}\"}}. A screenshot "
            "now shows the page as it was before the click.")


def _dbl_owner_phrase(owner: str) -> str:
    o = str(owner or "")
    return "this element" if o in ("", "this element") else f"the element around it ({o})"


def _dblclick_hint(sel, owner: str = "") -> str:
    """§4NV verify (req 1c343374 11:01): the page was still animating (a boot
    screen fading out), so it "changed" and the missed open went unsaid. The
    certain fact is said anyway — as a note, not a miss: a single click
    never runs an ondblclick handler, whatever else changed."""
    return (f"{_dbl_owner_phrase(owner)} has a DOUBLE-click handler (ondblclick); a single click does "
            f"not run it. If this click already did what you wanted, ignore this; otherwise use "
            f"{{\"action\":\"dblclick\",\"selector\":\"{sel}\"}}.")


async def _require_selector(page, sel, timeout_ms):
    """§4LN: fail FAST when a selector is not on the page, naming what IS —
    interact's click/fill waited Playwright's full 30 s per action (live
    probe: three clicks on a guessed '#add-btn' burned 90 s; the page's
    button was '#add'). state='attached', so a present-but-animating
    element still goes on to the action's own actionability wait."""
    # The FULL action timeout (an element can appear late over Tor — review:
    # the 8 s probe failed clicks Playwright's 30 s wait would have made);
    # a selector this sequence already found missing fails at once, so a
    # guessed selector costs one wait, not one per action (live: 3 × 30 s).
    probe_ms = 0 if sel in _MISSING_SELECTORS else max(500, int(timeout_ms))
    try:
        await page.wait_for_selector(sel, state="attached", timeout=probe_ms or 1)
        return
    except Exception as e:
        # only a TIMEOUT means "not on the page"; anything else (an invalid
        # selector, a page shape without the API) is left to the action
        # itself, which reports its own error
        if "Timeout" not in type(e).__name__:
            return
    try:
        cands = await page.evaluate(_CANDIDATES_JS)
    except Exception:
        cands = []
    _MISSING_SELECTORS.add(sel)
    listed = "; ".join(cands) if cands else "none found"
    raise ValueError(
        f"selector {sel!r} did not match any element within {probe_ms}ms. "
        f"Clickable elements on this page: {listed}. Use one of those "
        f"selectors (or read the page with extract_text first) — do not guess.")


async def op_click(op):
    selector = op.get("selector")
    if not selector:
        raise ValueError("click requires 'selector'")
    url, used_fallback = _resolve_url_or_error(op, "click")
    wait_until = op.get("wait_until", "load")

    async def run(page):
        _cresp = await page.goto(url, wait_until=wait_until)
        _cstatus = _cresp.status if _cresp else None        # §4PC: a 403 landed by click read as OK
        # FAIL-FAST SELECTOR PROBE (2026-07-14). Every atomic op runs in a
        # fresh context and re-navigates, so DOM created by a PREVIOUS click
        # (an opened window/menu/dialog) does not exist on this freshly
        # loaded page — page.click would then burn the FULL timeout waiting
        # for a selector that can never appear (observed live: two 30s click
        # timeouts on '.wp-option', which only exists after clicking the
        # Wallpapers icon, ate the turn's verification budget and tripped
        # the no-progress loop breaker — twice in one evening). Probe
        # existence with a short bounded wait; when absent, fail fast and
        # NAME the escape (op='interact' keeps one context alive across
        # steps). state='attached' not 'visible' so a present-but-animating
        # element still proceeds to page.click's own actionability wait.
        probe_ms = min(8000, int(op["timeout_ms"]))
        try:
            await page.wait_for_selector(selector, state="attached",
                                         timeout=probe_ms)
        except Exception:
            raise RuntimeError(
                f"selector {selector!r} not found within {probe_ms}ms on the "
                f"freshly-loaded page. Each atomic browser op reloads the "
                f"page in a fresh context, so elements created by a PREVIOUS "
                f"click (an opened window, menu or dialog) no longer exist. "
                f"Run the whole flow in ONE context with op='interact', e.g. "
                f"actions=[{{\"action\":\"click\",\"selector\":\"<the "
                f"opener>\"}}, {{\"action\":\"click\",\"selector\":"
                f"\"{selector}\"}}]."
            )
        # Attached ≠ actionable: an element can be IN the DOM yet hidden
        # until its opener is clicked (start-menu items, dropdown rows) —
        # the probe above passes, then an unbounded page.click waits the
        # FULL default 30s for visibility and dies with a raw
        # TimeoutError that carries no escape hint (req 43, 2026-07-17:
        # '.start-menu-item:nth-child(1)' on a freshly-loaded page whose
        # menu was closed). Bound the actionability wait like the probe
        # and NAME the op='interact' escape on timeout.
        try:
            await page.click(selector, timeout=probe_ms)
        except Exception as _ce:
            # Type-name check ONLY (playwright raises TimeoutError): a
            # message-content check matched unrelated errors that merely
            # mention the word (e.g. a TypeError about the timeout kwarg).
            if "Timeout" not in type(_ce).__name__:
                raise
            raise RuntimeError(
                f"selector {selector!r} is in the DOM but never became "
                f"clickable within {probe_ms}ms — it is likely HIDDEN until "
                f"an opener (menu/dialog/dropdown) is clicked first, and "
                f"this atomic op reloaded the page so that opened state is "
                f"gone. Run the whole flow in ONE context with "
                f"op='interact', e.g. actions=[{{\"action\":\"click\","
                f"\"selector\":\"<the opener>\"}}, {{\"action\":\"click\","
                f"\"selector\":\"{selector}\"}}]."
            )
        # Wait for any navigation triggered by the click to settle.
        try:
            await page.wait_for_load_state("load", timeout=op["timeout_ms"])
        except Exception:
            # Not every click navigates; a load-state timeout here is
            # benign and shouldn't fail the op.
            pass
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        result = {"url": final_url, "title": await page.title(), "used_last_url": used_fallback,
                  "status": _cstatus}
        # Post-click page text, same rationale as navigate — the state after a
        # click is usually what the model wants to read next.
        nav_text_chars = int(op.get("nav_text_chars", 8 * 1024))
        if nav_text_chars > 0:
            text, truncated, full_len = await _body_excerpt(page, nav_text_chars)
            if text:
                result["text"] = text
                result["length"] = full_len
                result["truncated"] = truncated
        return result

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


async def op_screenshot(op):
    out_path = op.get("out_path")
    if not out_path:
        raise ValueError("screenshot requires 'out_path'")
    full_page = bool(op.get("full_page", True))
    url, used_fallback = _resolve_url_or_error(op, "screenshot")
    wait_until = op.get("wait_until", "load")

    settle_ms = int(op.get("settle_ms") or 0)
    click_center = bool(op.get("click_center"))

    async def run(page):
        resp = await page.goto(url, wait_until=wait_until)
        status = resp.status if resp else None   # §4HH: the document's status
        # Interaction-gated content (e.g. a pointer-lock WebGL game that only
        # starts rendering after a click, or a scene that needs a beat to
        # paint) is invisible to a bare goto→shoot. settle_ms waits for the
        # first frames; click_center clicks the viewport centre to focus/lock
        # a canvas before capturing what a USER would actually see.
        if settle_ms > 0:
            await page.wait_for_timeout(settle_ms)
        if click_center:
            try:
                vp = page.viewport_size or {"width": 1280, "height": 720}
                await page.mouse.click(vp["width"] // 2, vp["height"] // 2)
                await page.wait_for_timeout(int(op.get("post_click_ms") or 800))
            except Exception:
                pass
        await _save_screenshot(page, out_path, full_page)
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        result = {"path": out_path, "url": final_url, "status": status,
                  "used_last_url": used_fallback}
        try:
            result["title"] = await page.title()
        except Exception:
            pass
        # How much TEXT was on the page at capture time. The pixel-side
        # render check cannot tell a rendered page from page chrome on a
        # gradient (measured: a Telegram share overlay and a correctly
        # rendered desktop UI sit on opposite sides of every pixel
        # statistic), so carry the one cheap number that separates them
        # for a text page: an almost-empty DOM means nothing had loaded.
        # Reported as a NUMBER, never as a verdict — a canvas game or a
        # chart legitimately renders with no text at all.
        try:
            _, _, _dom_len = await _body_excerpt(page, 1)
            result["dom_text_chars"] = int(_dom_len)
        except Exception:
            pass
        # Reflect the state AT capture time: if a start/play control is still
        # visible the screenshot shows the menu, not the running app (and
        # click_center, if used, failed to dismiss it).
        probe = await _probe_pre_interaction(page)
        if probe.get("pre_interaction"):
            result["pre_interaction"] = probe
        return result

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


async def op_close(op):
    """Nuke the persistent profile so the next session starts fresh.

    Every other op closes its context cleanly on exit, so there's no
    long-lived browser process to kill — the only cross-call state is
    the profile directory (which contains both the Chromium user-data
    and the .last_url sidecar). rmtree clears both in one shot.
    """
    profile_dir = op["profile_dir"]
    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir, ignore_errors=True)
    # §4LN: report what happened — rmtree(ignore_errors) can leave files
    return {"closed": not os.path.exists(profile_dir), "profile_dir": profile_dir}


async def op_interact(op):
    """Run a list of sub-actions inside a single Chromium context.

    The atomic ops (navigate/click/extract_text/...) each launch a
    fresh persistent context and re-navigate via the sidecar — great
    for simple scrape/interact, but it wipes any transient DOM
    mutations between ops. For multi-step SPA flows (open window →
    click button → read result) the mutations need to survive, so
    this op runs everything in ONE context and returns the per-action
    results.

    Each action is a dict with an "action" field and action-specific
    params:
      {"action": "goto", "url": "...", "wait_until": "load"}
      {"action": "click", "selector": "..."}
      {"action": "dblclick", "selector": "..."}  # required for
          # ondblclick-bound UIs (desktop-icon launchers etc.)
      {"action": "extract_text", "selector": "...", "max_chars": N}
      {"action": "fill", "selector": "...", "text": "..."}
      {"action": "wait_for_selector", "selector": "...", "timeout_ms": N}
      {"action": "screenshot", "out_path": "..."}
      {"action": "sleep", "ms": N}
      {"action": "evaluate", "js": "...", "max_chars": N}

    ``evaluate`` runs a JS expression (or arrow function) in the page and
    returns its JSON-serialised value. This is the ground-truth probe for
    app STATE the agent itself built — reading `ball.x` beats judging a
    screenshot of a moving ball (the 2026-07-31 pinball session burned
    five ~30s vision round-trips on a question one evaluate answers
    exactly). Runs in the page context, same trust domain as click/fill;
    JS-initiated fetches still pass through the context's SSRF route
    guard.

    Failures are reported per-action: a click that times out doesn't
    abort the whole sequence by default (``stop_on_error=False``),
    so the caller sees "step 3 failed; steps 4-6 ran anyway". Set
    ``stop_on_error`` true in the op dict to short-circuit instead.

    **Navigation failures are ALWAYS fatal**, regardless of
    ``stop_on_error``. If ``page.goto(...)`` raises (ERR_FILE_NOT_FOUND,
    connection refused, DNS failure, …), every subsequent click/fill/
    extract_text would be operating on Chromium's error page and would
    just time out one by one. This used to cause multi-hour hangs: a
    54-action sequence whose first goto 404'd ran clicks that each
    waited the full per-action timeout (120 s) trying to find elements
    that don't exist on the error page — 54 × 120 s ≈ 108 min. The
    rule now is: a failed goto aborts the sequence immediately with
    the original error surfaced clearly. Actions AFTER a successful
    goto still honour the per-action ``stop_on_error`` contract.
    """
    actions = op.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("interact requires a non-empty 'actions' list")
    stop_on_error = bool(op.get("stop_on_error", False))

    # Initial navigation: use explicit url → sidecar → error, same
    # semantics as every other op. If the first action is "goto", it
    # takes over; otherwise we navigate to the resolved URL first so
    # subsequent actions have a real DOM to work on.
    first_is_goto = (
        isinstance(actions[0], dict) and actions[0].get("action") == "goto"
    )
    if first_is_goto:
        initial_url = None
        used_fallback = False
    else:
        initial_url, used_fallback = _resolve_url_or_error(op, "interact")
    initial_wait_until = op.get("wait_until", "load")

    results = []
    # §4RX: the step in flight, so a budget overrun keeps what already ran
    progress = {"idx": None, "name": None, "initial_goto": None}

    async def run(page):
        initial_goto = None
        # Implicit initial navigation (when the first action is NOT a
        # `goto`). Same rule as explicit goto: if it fails, the whole
        # sequence is un-salvageable — abort with a single clear
        # error rather than running dozens of actions against an
        # error page.
        if initial_url is not None:
            _write_last_url(op["profile_dir"], initial_url)       # §4LN: the page asked for
            try:
                _iresp = await page.goto(initial_url, wait_until=initial_wait_until)
                _write_last_url(op["profile_dir"], page.url)
                # §4PC: the implicit goto used to discard its response, so a
                # 403 challenge page read as STATUS: OK through `interact`
                try:
                    _ititle = await asyncio.wait_for(page.title(), timeout=3.0)
                except Exception:
                    _ititle = ""
                initial_goto = {"status": (_iresp.status if _iresp else None), "url": page.url, "title": _ititle}
                progress["initial_goto"] = initial_goto
            except Exception as e:
                return {
                    "actions": [{
                        "index": -1, "action": "goto", "ok": False,
                        "error": f"initial navigation failed ({type(e).__name__}): {e}",
                        "url": initial_url,
                    }],
                    "aborted": True,
                    "abort_reason": "initial_goto_failed",
                    "final_url": initial_url,
                    "final_title": "",
                    "used_last_url": used_fallback,
                }

        for idx, step in enumerate(actions):
            progress["idx"], progress["name"] = idx, (step.get("action") if isinstance(step, dict) else None)
            if not isinstance(step, dict):
                results.append({
                    "index": idx, "action": None, "ok": False,
                    "error": f"action at index {idx} must be a dict, got {type(step).__name__}",
                })
                if stop_on_error:
                    break
                continue
            name = step.get("action")
            try:
                if name == "goto":
                    url = step.get("url")
                    if not url:
                        raise ValueError("goto requires 'url'")
                    wu = step.get("wait_until", "load")
                    _write_last_url(op["profile_dir"], url)        # §4LN: the page asked for
                    try:
                        _resp = await page.goto(url, wait_until=wu)
                    except Exception as nav_exc:
                        # A failed navigation is terminal for the whole
                        # sequence — see the docstring above. Record
                        # the failure and break out of the loop REGARDLESS
                        # of stop_on_error. The final snapshot at the
                        # end of `run` still fires so the caller gets
                        # a consistent shape.
                        results.append({
                            "index": idx, "action": "goto", "ok": False,
                            "error": f"{type(nav_exc).__name__}: {nav_exc}",
                            "url": url,
                            "aborted_sequence": True,
                        })
                        break
                    _write_last_url(op["profile_dir"], page.url)
                    # §4LN: the document status, so the host can tell a 403 /
                    # challenge page from the page itself (interact had no
                    # blocked check — the route the blocked hint recommends)
                    results.append({
                        "index": idx, "action": "goto", "ok": True,
                        "url": page.url, "title": await page.title(),
                        "status": getattr(_resp, "status", None),
                    })
                elif name == "click":
                    sel = step.get("selector")
                    if not sel:
                        raise ValueError("click requires 'selector'")
                    # Optional pre-click guard: wait for an overlay
                    # selector to leave the page before issuing the
                    # click. The 2026-04-26 webOS session lost ~70 min
                    # to a sequence like:
                    #   click(#unlock-btn)   → JS hides #lock-screen
                    #   click(#start-btn)   → blocked: #lock-screen
                    #                         still intercepts events
                    # Playwright's bare ``click`` auto-waits for the
                    # TARGET to be actionable but doesn't know about
                    # an unrelated overlay. ``wait_for_hidden`` lets
                    # the LLM express "make sure this thing is gone
                    # before I click my real target" without inserting
                    # a separate sleep+wait_for_selector pair (which
                    # also works, but is two extra actions).
                    wait_for_hidden = step.get("wait_for_hidden")
                    if wait_for_hidden:
                        try:
                            await page.wait_for_selector(
                                wait_for_hidden,
                                state="hidden",
                                timeout=int(step.get(
                                    "wait_for_hidden_ms",
                                    min(5000, op["timeout_ms"])
                                )),
                            )
                        except Exception as wait_exc:
                            # The overlay may already be gone (best
                            # case) — but if it didn't disappear in
                            # time, surface that as the click failure
                            # rather than letting the click itself
                            # report a generic "intercepts pointer
                            # events" error. The LLM gets a clearer
                            # signal: "the thing you said was blocking
                            # you didn't actually go away".
                            raise RuntimeError(
                                f"wait_for_hidden({wait_for_hidden!r}) "
                                f"timed out before click({sel!r}): "
                                f"{type(wait_exc).__name__}: {wait_exc}"
                            )
                    # ``force=True`` skips Playwright's actionability
                    # check (visibility, stability, hit-test). Use case:
                    # a CSS transition that Playwright deems "not
                    # stable" but whose target is still the right
                    # element — explicit LLM-driven escape hatch.
                    await _require_selector(page, sel, op["timeout_ms"])
                    _cover = (await _covered_note(page, sel, "click")
                              if step.get("force") else "")
                    _dbl = (not _cover) and await _dblclick_only(page, sel)
                    _before = await _page_sig(page) if _dbl else None
                    if step.get("force"):
                        await page.click(sel, force=True)
                    else:
                        await page.click(sel)
                    _r = {"index": idx, "action": "click", "ok": True, "selector": sel}
                    _note = _cover or (await _dblclick_only_effect(page, sel, _before, _dbl) if _dbl else "")
                    if _note:
                        _r["no_effect"] = _note
                    elif _dbl:
                        _r["hint"] = _dblclick_hint(sel, _dbl)
                    results.append(_r)
                elif name == "dblclick":
                    # Double-click — required for "desktop-icon" UIs that
                    # bind their open/launch handler to `ondblclick` (the
                    # common pattern in OS-style web apps). Without this
                    # action type the LLM has to choose between (a)
                    # emitting `click` and watching nothing happen, or
                    # (b) dispatching synthetic events via evaluate(),
                    # which doesn't trigger native handlers reliably.
                    # Playwright's `page.dblclick` fires a proper
                    # mousedown-mouseup-mousedown-mouseup sequence that
                    # cross-browser dblclick listeners actually receive.
                    sel = step.get("selector")
                    if not sel:
                        raise ValueError("dblclick requires 'selector'")
                    wait_for_hidden = step.get("wait_for_hidden")
                    if wait_for_hidden:
                        try:
                            await page.wait_for_selector(
                                wait_for_hidden,
                                state="hidden",
                                timeout=int(step.get(
                                    "wait_for_hidden_ms",
                                    min(5000, op["timeout_ms"])
                                )),
                            )
                        except Exception as wait_exc:
                            raise RuntimeError(
                                f"wait_for_hidden({wait_for_hidden!r}) "
                                f"timed out before dblclick({sel!r}): "
                                f"{type(wait_exc).__name__}: {wait_exc}"
                            )
                    await _require_selector(page, sel, op["timeout_ms"])
                    _cover = (await _covered_note(page, sel, "dblclick")
                              if step.get("force") else "")
                    if step.get("force"):
                        await page.dblclick(sel, force=True)
                    else:
                        await page.dblclick(sel)
                    _r = {"index": idx, "action": "dblclick", "ok": True, "selector": sel}
                    _note = _cover
                    if _note:
                        _r["no_effect"] = _note
                    results.append(_r)
                elif name == "extract_text":
                    sel = step.get("selector")
                    max_chars = int(step.get("max_chars", 64 * 1024))
                    if sel:
                        el = await page.query_selector(sel)
                        if not el:
                            raise ValueError(
                                f"selector {sel!r} did not match any element"
                            )
                        text = (await el.inner_text()).strip()
                    else:
                        text = await page.evaluate(
                            "() => document.body ? document.body.innerText : ''"
                        )
                    full_len = len(text)  # capture BEFORE truncating
                    truncated = False
                    if len(text) > max_chars:
                        text = text[:max_chars]
                        truncated = True
                    results.append({
                        "index": idx, "action": "extract_text", "ok": True,
                        "selector": sel, "text": text,
                        # True length, not the capped length (see extract_text above).
                        "length": full_len, "truncated": truncated,
                    })
                elif name == "fill":
                    sel = step.get("selector")
                    text = step.get("text", "")
                    if not sel:
                        raise ValueError("fill requires 'selector'")
                    wait_for_hidden = step.get("wait_for_hidden")
                    if wait_for_hidden:
                        try:
                            await page.wait_for_selector(
                                wait_for_hidden,
                                state="hidden",
                                timeout=int(step.get(
                                    "wait_for_hidden_ms",
                                    min(5000, op["timeout_ms"])
                                )),
                            )
                        except Exception as wait_exc:
                            raise RuntimeError(
                                f"wait_for_hidden({wait_for_hidden!r}) "
                                f"timed out before fill({sel!r}): "
                                f"{type(wait_exc).__name__}: {wait_exc}"
                            )
                    await _require_selector(page, sel, op["timeout_ms"])
                    await page.fill(sel, text)
                    results.append({
                        "index": idx, "action": "fill", "ok": True,
                        "selector": sel, "length": len(text),
                    })
                elif name == "wait_for_selector":
                    sel = step.get("selector")
                    timeout_ms = int(step.get("timeout_ms", op["timeout_ms"]))
                    # ``state`` controls what we're waiting FOR. The
                    # default ("visible") matches Playwright's own
                    # default. Crucially, "hidden" / "detached" let the
                    # LLM wait on something to GO AWAY — this is the
                    # missing primitive that turned the 2026-04-26
                    # webOS session into a 70-min loop: the LLM kept
                    # clicking #unlock-btn and immediately #start-btn
                    # without any way to say "wait for the lock screen
                    # to actually finish disappearing first." Bare
                    # wait_for_selector(sel) with no state arg waits
                    # for the selector to APPEAR — useless for an
                    # element that's already in the DOM and just needs
                    # to fade out. Valid values mirror Playwright:
                    # attached, detached, visible, hidden.
                    state = step.get("state", "visible")
                    if state not in ("attached", "detached", "visible", "hidden"):
                        raise ValueError(
                            f"wait_for_selector got invalid state {state!r}; "
                            "valid: attached, detached, visible, hidden"
                        )
                    if not sel:
                        raise ValueError("wait_for_selector requires 'selector'")
                    await page.wait_for_selector(sel, state=state, timeout=timeout_ms)
                    results.append({
                        "index": idx, "action": "wait_for_selector", "ok": True,
                        "selector": sel, "state": state,
                    })
                elif name == "screenshot":
                    out_path = step.get("out_path")
                    if not out_path:
                        raise ValueError("screenshot requires 'out_path'")
                    full_page = bool(step.get("full_page", True))
                    await _save_screenshot(page, out_path, full_page)
                    results.append({
                        "index": idx, "action": "screenshot", "ok": True,
                        "path": out_path,
                    })
                elif name == "sleep":
                    ms = int(step.get("ms", 500))
                    await page.wait_for_timeout(ms)
                    results.append({
                        "index": idx, "action": "sleep", "ok": True, "ms": ms,
                    })
                elif name == "evaluate":
                    js = (step.get("js") or step.get("expression")
                          or step.get("script"))
                    if not js:
                        raise ValueError("evaluate requires 'js'")
                    # page.evaluate is NOT governed by set_default_timeout
                    # (that covers actions/navigations only), so an
                    # unsettled promise ("await a gameover event") would
                    # hang until the subprocess kill — which discards the
                    # [BROWSER_OK] payload and with it every EARLIER
                    # action's result. Bound it per-action so a stuck
                    # evaluate degrades to a per-action failure like any
                    # sibling branch.
                    _eval_timeout_s = max(
                        1.0, int(step.get("timeout_ms", op["timeout_ms"])) / 1000.0)
                    try:
                        value = await asyncio.wait_for(
                            page.evaluate(js), timeout=_eval_timeout_s)
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"evaluate timed out after {_eval_timeout_s:.0f}s — "
                            "the expression never settled (an unresolved "
                            "Promise?). Poll with sleep+evaluate instead of "
                            "awaiting an event inside one evaluate."
                        )
                    # The page can hand back anything JS can build —
                    # serialise defensively (default=str catches whatever
                    # Playwright let through) and CAP the output so one
                    # evaluate can't dump a whole data structure into the
                    # model context (same policy as extract_text).
                    try:
                        as_json = json.dumps(value, default=str)
                    except (TypeError, ValueError):
                        as_json = json.dumps(str(value))
                    max_chars = int(step.get("max_chars", 16 * 1024))
                    full_len = len(as_json)
                    truncated = False
                    if len(as_json) > max_chars:
                        as_json = as_json[:max_chars]
                        truncated = True
                    results.append({
                        "index": idx, "action": "evaluate", "ok": True,
                        "value": as_json,
                        "length": full_len, "truncated": truncated,
                    })
                else:
                    raise ValueError(
                        f"unknown action {name!r}; valid: "
                        "goto, click, dblclick, extract_text, fill, "
                        "wait_for_selector, screenshot, sleep, evaluate"
                    )
            except Exception as e:
                results.append({
                    "index": idx, "action": name, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
                if stop_on_error:
                    break

        progress["idx"] = len(actions)          # §4RX review: every step ran; only the snapshot is left
        # Final snapshot so the caller doesn't need a trailing no-op
        # action just to learn where the sequence landed.
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        # A terminal goto failure sets `aborted_sequence` on the last
        # result entry — surface that up to the caller as a top-level
        # `aborted` flag so the agent-facing formatter can render the
        # summary as "sequence aborted at step N" instead of "N-1
        # successes and one mysterious failure".
        aborted = bool(
            results and isinstance(results[-1], dict)
            and results[-1].get("aborted_sequence")
        )
        return {
            "actions": results,
            "final_url": final_url,
            "final_title": await page.title(),
            "used_last_url": used_fallback,
            "aborted": aborted,
            "abort_reason": "goto_failed" if aborted else None,
            **({"initial_goto": initial_goto} if initial_goto else {}),
        }

    def budget_partial(page, budget_ms):
        """§4RX: the total budget ran out mid-sequence — return the steps
        that ran plus the one in flight, not a bare error. None (keep the
        error) when no action had started: the page was never read."""
        idx = progress["idx"]
        if idx is None:
            return None
        # §4RX review: every step had run (the overrun hit the final snapshot)
        # — that is a whole sequence, not a budget partial
        in_flight = idx < len(actions)
        if in_flight and not (results and results[-1].get("index") == idx):
            # "timed out" keeps an in-flight evaluate out of §4RJ's "your
            # script threw" bucket (review: it read STATUS: OK)
            _maybe = (" — it may already have taken effect"
                      if progress["name"] in ("click", "dblclick", "fill", "evaluate", "goto") else "")
            results.append({
                "index": idx, "action": progress["name"], "ok": False,
                "error": (f"TimeoutError: timed out — the sequence ran out of its {budget_ms} ms "
                          f"total budget during this step{_maybe}"),
            })
        final_url = page.url
        if progress["name"] != "goto" or not in_flight:     # an in-flight goto already wrote the page it asked for (§4LN)
            _write_last_url(op["profile_dir"], final_url)
        aborted = bool(results and results[-1].get("aborted_sequence"))   # a goto abort keeps its banner
        return {
            "actions": results,
            "final_url": final_url,
            "final_title": "",
            "used_last_url": used_fallback,
            "aborted": aborted,
            "abort_reason": "goto_failed" if aborted else None,
            "budget_exhausted": in_flight,
            "budget_ms": budget_ms,
            "not_run": max(0, len(actions) - idx - 1),
            **({"initial_goto": progress["initial_goto"]} if progress["initial_goto"] else {}),
        }

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run,
                               total_ms=op.get("total_timeout_ms"), on_timeout=budget_partial)


OPS = {
    "navigate": op_navigate,
    "extract_text": op_extract_text,
    "click": op_click,
    "screenshot": op_screenshot,
    "close": op_close,
    "interact": op_interact,
}


async def main():
    if len(sys.argv) < 2:
        _emit_err("runner requires one JSON argument")
        return 1
    try:
        op = json.loads(sys.argv[1])
    except Exception as e:
        _emit_err(f"invalid op JSON: {e}")
        return 1
    op_name = op.get("op")
    if op_name not in OPS:
        _emit_err(f"unknown op {op_name!r}; valid: {sorted(OPS)}")
        return 1
    try:
        ALLOWED_LOCAL_PORTS.update(
            int(x) for x in (op.get("allowed_local_ports") or []))
    except (TypeError, ValueError):
        pass
    global _NONCE
    _n = str(op.get("nonce") or "")
    _NONCE = _n if _n.isalnum() and 8 <= len(_n) <= 64 else ""
    _set_last_url_file(op.get("last_url_file"))
    _set_last_url_conv_file(op.get("last_url_conv_file"))
    try:
        result = await OPS[op_name](op)
        _emit_ok(result)
        return 0
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        _emit_err(f"{type(e).__name__}: {e}\n{tb}")
        return 1


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except KeyboardInterrupt:
        rc = 1
    sys.exit(rc)
