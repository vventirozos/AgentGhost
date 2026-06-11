"""Native headless-browser tool.

Wraps Playwright inside the sandbox so the LLM can navigate, extract
text, click, and screenshot without having to hand-write async
Playwright code. Compared to the stateful-Jupyter escape hatch documented
in prompts.py, this tool:

  1. Is the canonical path for the 80%-case scrape/interact flow; the
     LLM doesn't have to remember `await async_playwright().start()` vs
     `async with`, top-level-await rules, or cleanup order.
  2. Forces DNS-over-SOCKS via Chromium args (`--proxy-server` +
     `--host-resolver-rules`) so the browser path can't leak DNS to the
     host resolver even if the LLM forgets — the biggest footgun on the
     stateful-kernel path.
  3. Persists session state (cookies, localStorage) AND the last-
     navigated URL across tool calls, via Chromium's
     `launch_persistent_context(user_data_dir=...)` + a `.last_url`
     sidecar inside the profile dir. So the LLM can chain
     navigate → extract_text → click → screenshot without re-passing
     `url` each time: a subsequent op without an explicit `url` falls
     back to the sidecar and re-navigates under the hood.
  4. Fails loudly: each op exits non-zero on Playwright error so the
     planner sees the failure, rather than silently continuing with a
     half-dead page. When NO URL is available (no arg, no sidecar), the
     runner emits a clear "pass url=... or call navigate first" error
     rather than querying against an empty about:blank page.

All execution stays inside the Docker sandbox — same as the stateful
path. This tool is just a thin JSON-over-subprocess wrapper around a
tiny Playwright runner script.
"""

import asyncio
import json
import logging
import shlex
from pathlib import Path
from typing import Optional

from ..utils.logging import Icons, pretty_log
from .file_system import _get_safe_path, _to_container_path

logger = logging.getLogger("GhostAgent")

# Persistent per-sandbox browser profile — lives inside the sandbox
# workspace so it survives across turns / tool calls but doesn't leak
# onto the host filesystem outside GHOST_SANDBOX_DIR.
_BROWSER_PROFILE_DIR = ".browser_profile"
_BROWSER_RUNNER_FILENAME = ".browser_runner.py"

# Serializes Chromium launches against the shared persistent profile dir.
# `launch_persistent_context` takes an exclusive SingletonLock on its
# user-data-dir; when the agent fires several browser tool-calls in one
# turn (e.g. navigate three URLs at once), the concurrent Chromium
# processes contend for that lock and the losers SIGSEGV / TargetClosed —
# 2 of 3 succeed, the rest crash (verified). We can't just hand each op
# its own profile: the persistent profile is what gives cross-op cookie/
# session continuity and the `.last_url` sidecar. So instead we serialize
# launches — browser ops queue and run one at a time. They're seconds-long
# and inherently single-writer, so this is the correct semantics, not a
# perf regression. Module-global so it spans all tool_browser calls in the
# (single-event-loop) agent process.
_BROWSER_PROFILE_LOCK = asyncio.Lock()

# Keep outputs reasonable — a single page's HTML can be 5+ MB and would
# blow the LLM context window, so we cap before returning. These caps
# match `helper_fetch_url_content`'s 5 MB ceiling.
_MAX_TEXT_CHARS = 64 * 1024  # ~16k tokens — more than enough for LLM reasoning


def _runner_script() -> str:
    """Return the Playwright runner source.

    Kept as a single string so we can write it into the sandbox
    workspace and invoke with `python3 .browser_runner.py <op_json>`.
    The runner is intentionally defensive: all I/O goes through stdout
    with a `[BROWSER_OK]` / `[BROWSER_ERR]` sentinel so the outer tool
    can parse structured results even if Chromium writes stderr noise.
    """
    return r'''#!/usr/bin/env python3
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
import json
import os
import shutil
import sys
import traceback

from playwright.async_api import async_playwright


_LAST_URL_FILENAME = ".last_url"


def _emit_ok(payload):
    sys.stdout.write("[BROWSER_OK] " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_err(msg):
    sys.stdout.write("[BROWSER_ERR] " + str(msg) + "\n")
    sys.stdout.flush()


def _last_url_path(profile_dir):
    return os.path.join(profile_dir, _LAST_URL_FILENAME)


def _read_last_url(profile_dir):
    """Return the URL of the most recent successful navigation, or
    None if no prior navigation is recorded. Best-effort — any I/O
    error is treated as "no record" so a corrupt sidecar never blocks
    an op."""
    try:
        p = _last_url_path(profile_dir)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip() or None
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
        with open(_last_url_path(profile_dir), "w", encoding="utf-8") as f:
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
    --disable-webrtc: WebRTC can expose the real host IP via STUN
        even when HTTP traffic is proxied. Flat-disable for Tor paths.
    """
    args = ["--no-sandbox", "--disable-dev-shm-usage"]
    if proxy:
        # EXCLUDE localhost so the self-play fixture server (and any
        # in-container service) is reachable without routing through Tor,
        # AND exclude the proxy's own host so Chromium can actually reach
        # the SOCKS server. Everything else is forced through SOCKS DNS.
        excludes = ["EXCLUDE localhost"]
        try:
            from urllib.parse import urlparse
            phost = urlparse(proxy).hostname
        except Exception:
            phost = None
        if phost and phost != "localhost":
            excludes.append("EXCLUDE " + phost)
        args.append("--host-resolver-rules=MAP * ~NOTFOUND , " + " , ".join(excludes))
        args.append("--disable-features=WebRtcHideLocalIpsWithMdns")
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


async def _with_context(profile_dir, proxy, timeout_ms, op_fn):
    """Open a persistent context, run op_fn(page), close cleanly.

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
        )
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
        ctx = await p.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            page.set_default_timeout(timeout_ms)
            js_errors, console_msgs = _attach_diagnostics(page)
            result = await op_fn(page)
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


async def _probe_pre_interaction(page):
    """Detect a visible start / play / loading control on the page.

    A "Click to Play" button or a loading screen means the app has NOT
    started — a screenshot then shows the MENU, not the running app, and a
    claim that "the game works / renders" is grading the wrong thing (the
    exact false-positive seen live: the agent declared a Minecraft clone
    "fully functional" from a capture that still had the start modal up).
    Returns ``{pre_interaction: bool, controls: [text,…]}`` (or {} on any
    failure — never raises into the op)."""
    js = """() => {
      const KW = /click to (play|start)|press (to )?start|start game|tap to (play|start)|enter game|^play$|^start$|^begin$|loading/i;
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


async def op_navigate(op):
    url = op.get("url")
    if not url:
        raise ValueError("navigate requires 'url'")
    wait_until = op.get("wait_until", "load")  # load | domcontentloaded | networkidle

    async def run(page):
        resp = await page.goto(url, wait_until=wait_until)
        status = resp.status if resp else None
        final_url = page.url
        title = await page.title()
        _write_last_url(op["profile_dir"], final_url)
        result = {"status": status, "url": final_url, "title": title}
        probe = await _probe_pre_interaction(page)
        if probe.get("pre_interaction"):
            result["pre_interaction"] = probe
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
        await page.goto(url, wait_until=wait_until)
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
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        return {
            "url": final_url,
            "title": await page.title(),
            "text": text,
            "truncated": truncated,
            # Report the TRUE page length, not the capped length — else
            # `length` always equals max_chars on truncation, hiding how
            # much was dropped from any downstream "got the whole page?" check.
            "length": full_len,
            "used_last_url": used_fallback,
        }

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


async def op_click(op):
    selector = op.get("selector")
    if not selector:
        raise ValueError("click requires 'selector'")
    url, used_fallback = _resolve_url_or_error(op, "click")
    wait_until = op.get("wait_until", "load")

    async def run(page):
        await page.goto(url, wait_until=wait_until)
        await page.click(selector)
        # Wait for any navigation triggered by the click to settle.
        try:
            await page.wait_for_load_state("load", timeout=op["timeout_ms"])
        except Exception:
            # Not every click navigates; a load-state timeout here is
            # benign and shouldn't fail the op.
            pass
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        return {"url": final_url, "title": await page.title(), "used_last_url": used_fallback}

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
        await page.goto(url, wait_until=wait_until)
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
        await page.screenshot(path=out_path, full_page=full_page)
        final_url = page.url
        _write_last_url(op["profile_dir"], final_url)
        result = {"path": out_path, "url": final_url, "used_last_url": used_fallback}
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
    return {"closed": True, "profile_dir": profile_dir}


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

    async def run(page):
        # Implicit initial navigation (when the first action is NOT a
        # `goto`). Same rule as explicit goto: if it fails, the whole
        # sequence is un-salvageable — abort with a single clear
        # error rather than running dozens of actions against an
        # error page.
        if initial_url is not None:
            try:
                await page.goto(initial_url, wait_until=initial_wait_until)
                _write_last_url(op["profile_dir"], page.url)
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
                    try:
                        await page.goto(url, wait_until=wu)
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
                    results.append({
                        "index": idx, "action": "goto", "ok": True,
                        "url": page.url, "title": await page.title(),
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
                    if step.get("force"):
                        await page.click(sel, force=True)
                    else:
                        await page.click(sel)
                    results.append({
                        "index": idx, "action": "click", "ok": True,
                        "selector": sel,
                    })
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
                    if step.get("force"):
                        await page.dblclick(sel, force=True)
                    else:
                        await page.dblclick(sel)
                    results.append({
                        "index": idx, "action": "dblclick", "ok": True,
                        "selector": sel,
                    })
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
                    await page.screenshot(path=out_path, full_page=full_page)
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
                else:
                    raise ValueError(
                        f"unknown action {name!r}; valid: "
                        "goto, click, dblclick, extract_text, fill, "
                        "wait_for_selector, screenshot, sleep"
                    )
            except Exception as e:
                results.append({
                    "index": idx, "action": name, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
                if stop_on_error:
                    break

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
        }

    return await _with_context(op["profile_dir"], op.get("proxy"), op["timeout_ms"], run)


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
'''


def _build_op_payload(
    op: str,
    url: Optional[str],
    selector: Optional[str],
    out_path: Optional[str],
    wait_until: Optional[str],
    full_page: Optional[bool],
    max_chars: Optional[int],
    timeout_ms: int,
    tor_proxy: Optional[str],
    actions: Optional[list] = None,
    stop_on_error: Optional[bool] = None,
    click_center: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    """Assemble the op dict the runner expects.

    Centralised so the same validation & proxy-rewrite rules apply to
    every op (instead of being duplicated at each call site).
    """
    # Chromium only accepts `socks5://` — unlike httpx, there's no
    # `socks5h://` scheme; DNS-over-proxy is controlled via
    # --host-resolver-rules in the runner instead.
    proxy = tor_proxy
    if proxy and proxy.startswith("socks5h://"):
        proxy = "socks5://" + proxy[len("socks5h://"):]

    # Tor-robust navigation default: the full `load` event frequently
    # NEVER fires within the timeout on JS-heavy pages over Tor —
    # analytics beacons, long-poll sockets, and slow third-party
    # subresources stall on a slow exit, so a navigation that had already
    # delivered the real content times out and the runner exits 1 (the
    # exact "navigate: runner exit 1" seen in production). When the caller
    # hasn't pinned `wait_until` AND we're going through a proxy, default
    # to `domcontentloaded` (the HTML-parsed milestone) — far more
    # reliable over Tor and sufficient for navigate/extract_text. An
    # explicit `wait_until` from the caller always wins.
    if wait_until is None and proxy:
        wait_until = "domcontentloaded"

    payload: dict = {
        "op": op,
        "profile_dir": f"/workspace/{_BROWSER_PROFILE_DIR}",
        "timeout_ms": int(timeout_ms),
        "proxy": proxy,
    }
    if url is not None:
        payload["url"] = url
    if selector is not None:
        payload["selector"] = selector
    if out_path is not None:
        payload["out_path"] = out_path
    if wait_until is not None:
        payload["wait_until"] = wait_until
    if full_page is not None:
        payload["full_page"] = bool(full_page)
    if max_chars is not None:
        payload["max_chars"] = int(max_chars)
    if actions is not None:
        payload["actions"] = actions
    if stop_on_error is not None:
        payload["stop_on_error"] = bool(stop_on_error)
    if click_center is not None:
        payload["click_center"] = bool(click_center)
    if settle_ms is not None:
        payload["settle_ms"] = int(settle_ms)
    return payload


def _pre_interaction_line(parsed: dict) -> str:
    """Render the PRE_INTERACTION warning for a navigate/screenshot result.

    The verifier reads tool output as EVIDENCE, so surfacing "a start/play
    control is visible" here is what lets a 'the app works' claim made over a
    menu/loading screen be REFUTED instead of confabulated-and-confirmed."""
    pre = parsed.get("pre_interaction") if isinstance(parsed, dict) else None
    if not (pre and pre.get("pre_interaction")):
        return ""
    ctrls = ", ".join(repr(c) for c in (pre.get("controls") or [])[:3])
    return (
        f"\nPRE_INTERACTION: a start/play/loading control is visible "
        f"({ctrls}). The app has NOT started — this is the MENU/LOADING "
        f"screen, not the running app. Do NOT claim it works/renders from "
        f"this capture. Interact FIRST (screenshot with click_center=true, or "
        f"operation='click' the control, e.g. selector='#startBtn'), THEN "
        f"re-capture and judge that."
    )


def analyze_screenshot_render(host_path, sample_max: int = 200):
    """Objective "did anything actually render?" check on a screenshot PNG.

    The live failure was the agent claiming a Minecraft world rendered when
    the screenshot was empty blue sky (the world only loads after a pointer-
    lock click). A screenshot + an LLM/vision self-assessment couldn't catch
    it — the model described what it EXPECTED. This is the un-gameable
    ground-truth signal: a near-UNIFORM frame (one colour dominating, very
    few distinct colours) is the signature of a blank / loading / sky-only
    capture, regardless of what the model claims.

    Returns ``{dominant_pct, distinct_colors, verdict, note}`` or ``None``
    when PIL or the file is unavailable (caller treats None as "no signal").
    """
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(host_path).convert("RGB")
    except Exception:
        return None
    try:
        img.thumbnail((sample_max, sample_max))
        px = list(img.getdata())
    except Exception:
        return None
    total = len(px)
    if total == 0:
        return None
    from collections import Counter
    # Quantise to 5-bit/channel buckets so anti-aliasing/gradient noise
    # doesn't inflate the distinct-colour count (a sky gradient is still
    # "basically one colour" for our purposes).
    buckets = Counter((r >> 3, g >> 3, b >> 3) for r, g, b in px)
    dominant = buckets.most_common(1)[0][1]
    dominant_pct = dominant / total
    distinct = len(buckets)
    if dominant_pct >= 0.80 or distinct <= 6:
        verdict = "uniform"
        note = (
            f"{dominant_pct:.0%} of the frame is a single colour "
            f"({distinct} distinct colours) — this capture looks BLANK / "
            f"sky-only / a loading screen, NOT a populated scene. Any claim "
            f"that content (terrain, a chart, a UI) renders is UNSUPPORTED "
            f"by this screenshot. Interact first (e.g. screenshot with "
            f"click_center=true to focus/lock a canvas game, or raise "
            f"settle_ms), re-capture, or honestly report it is still broken."
        )
    else:
        verdict = "has_content"
        note = (
            f"{dominant_pct:.0%} dominant colour across {distinct} distinct "
            f"colours — the frame contains visual content."
        )
    return {
        "dominant_pct": round(dominant_pct, 3),
        "distinct_colors": distinct,
        "verdict": verdict,
        "note": note,
    }


def _parse_runner_output(stdout: str) -> tuple[bool, object]:
    """Pick the [BROWSER_OK]/[BROWSER_ERR] sentinel line out of stdout.

    Chromium / Playwright print warnings ("libpci.so.3 not found", etc.)
    that interleave with the runner's output. We scan for the last
    sentinel line so stray noise doesn't corrupt the result.
    """
    ok_line = None
    err_line = None
    for line in stdout.splitlines():
        if line.startswith("[BROWSER_OK] "):
            ok_line = line[len("[BROWSER_OK] "):]
        elif line.startswith("[BROWSER_ERR] "):
            err_line = line[len("[BROWSER_ERR] "):]
    if ok_line is not None:
        try:
            return True, json.loads(ok_line)
        except Exception as e:
            return False, f"malformed OK payload from runner: {e}: {ok_line[:200]}"
    if err_line is not None:
        return False, err_line
    # No sentinel at all — surface the raw tail so the agent can debug.
    tail = stdout[-2000:] if stdout else "(no output)"
    return False, f"runner emitted no sentinel. Raw tail:\n{tail}"


_VALID_OPS = {"navigate", "extract_text", "click", "screenshot", "close", "interact"}


def _browser_blocked_url(u: Optional[str]) -> Optional[str]:
    """SSRF guard for the browser: block http(s) navigation to internal /
    loopback / link-local / metadata hosts (which the host-network sandbox
    can otherwise reach), while ALLOWING file:// (self-play fixtures render
    as file:// pages) and about:/data:. Returns a refusal reason or None."""
    if not u:
        return None
    from urllib.parse import urlparse
    try:
        scheme = (urlparse(str(u)).scheme or "").lower()
    except Exception:
        return None
    if scheme in ("http", "https"):
        from ..utils.helpers import url_ssrf_reason
        return url_ssrf_reason(u)
    return None


def _resolve_file_url(sandbox_dir, url):
    """Rewrite a ``file://`` URL so it points at where the file actually
    lives when a project scopes the sandbox to ``<root>/projects/<id>/``.

    The model emits absolute container paths it can "see" — e.g.
    ``file:///workspace/browser_os/index.html`` — but the project-scoped
    ``file_system`` wrote that file to ``<root>/projects/<id>/browser_os/
    index.html`` (``/workspace/projects/<id>/browser_os/index.html`` in the
    container), so the bare ``/workspace/...`` URL is a 404. Resolve the
    path the SAME way file_system does (``_get_safe_path`` heals the
    ``/workspace/`` and redundant ``projects/<id>/`` prefixes) and translate
    back with ``_to_container_path`` (which un-scopes to the root mount).

    Only active when scoped (sandbox_dir's parent is literally ``projects``);
    otherwise the URL passes through byte-for-byte so non-project behaviour
    is untouched. Falls back to the sandbox-root location if the scoped file
    is absent but the root one exists (mirrors the vision_analysis fallback).
    Non-file URLs and unparseable inputs pass through unchanged.
    """
    if not isinstance(url, str) or not url.startswith("file://"):
        return url
    sb = Path(sandbox_dir) if sandbox_dir is not None else None
    if sb is None or sb.parent.name != "projects":
        return url
    try:
        path_part = url[len("file://"):]
        host = _get_safe_path(sb, path_part)
        if not host.exists():
            root = sb.parent.parent
            root_host = _get_safe_path(root, path_part)
            if root_host.exists():
                return "file://" + _to_container_path(root, root_host)
        return "file://" + _to_container_path(sb, host)
    except Exception:
        return url


def _format_js_diagnostics(parsed: dict) -> str:
    """Render captured uncaught exceptions + console errors as a block the
    LLM can act on. Returns "" when the page was clean.

    This is the antidote to the "loading screen frozen forever, agent
    burns 40 minutes guessing it's a perf problem" failure: an uncaught
    TypeError in init() shows up here verbatim, so the very next turn can
    fix the actual line instead of poll-screenshotting a dead page."""
    if not isinstance(parsed, dict):
        return ""
    out = []
    js_errors = parsed.get("js_errors") or []
    if js_errors:
        out.append(
            f"⚠ UNCAUGHT JS EXCEPTIONS ({len(js_errors)}) — these crash the "
            "page silently and are almost always the real bug; fix these "
            "BEFORE assuming a performance/timeout problem:"
        )
        for e in js_errors:
            first = str(e).splitlines()[0] if str(e).splitlines() else str(e)
            out.append(f"  • {first}")
            rest = str(e).splitlines()[1:]
            for ln in rest[:4]:
                out.append(f"      {ln}")
    console = parsed.get("console") or []
    errs = [c for c in console if isinstance(c, dict) and c.get("type") in ("error", "warning")]
    if errs:
        out.append(f"CONSOLE ({len(errs)} error/warning):")
        for c in errs[:10]:
            # `loc` (source file:line:col, captured from msg.location in
            # the runner) turns "Unexpected identifier 't'" into
            # "data.js:35:47 — Unexpected identifier 't'" — the agent can
            # open the exact line instead of hypothesizing causes.
            loc = c.get("loc") or ""
            prefix = f"{loc} — " if loc else ""
            out.append(f"  • [{c.get('type')}] {prefix}{c.get('text')}")
    return ("\n" + "\n".join(out)) if out else ""


async def tool_browser(
    operation: str = None,
    url: Optional[str] = None,
    selector: Optional[str] = None,
    out_path: Optional[str] = None,
    wait_until: Optional[str] = None,
    full_page: Optional[bool] = None,
    max_chars: Optional[int] = None,
    timeout_ms: int = 30000,
    actions: Optional[list] = None,
    stop_on_error: Optional[bool] = None,
    sandbox_dir: Path = None,
    sandbox_manager=None,
    tor_proxy: Optional[str] = None,
    workspace_model=None,
    container_workdir: Optional[str] = None,
    **kwargs,
):
    """Run a single browser operation inside the sandbox.

    Operations:
      navigate: go to a URL, return {status, url, title}.
      extract_text: go to URL (optional), return innerText — body or
                    a CSS selector. Truncates at `max_chars`.
      click: click a selector, wait for load, return {url, title}.
      screenshot: save a PNG to `out_path` inside /workspace.
      close: delete the persistent profile so the next session is fresh.
      interact: run a list of sub-actions in ONE Chromium context
                (click + extract + screenshot etc. share transient DOM
                state). Required for multi-step SPA flows where the
                atomic per-op re-navigation would wipe intermediate
                state.

    `tor_proxy` is forwarded as Chromium's `--proxy-server` with
    `--host-resolver-rules` forcing DNS through the proxy.
    """
    # --- PARAMETER HALLUCINATION HEALING (matches execute.py style) ---
    operation = operation or kwargs.get("op") or kwargs.get("action")
    url = url or kwargs.get("link") or kwargs.get("href")
    selector = selector or kwargs.get("css") or kwargs.get("query_selector")
    out_path = out_path or kwargs.get("path") or kwargs.get("filename")
    actions = actions or kwargs.get("steps") or kwargs.get("sequence")

    def _err(msg: str, hint: str = None) -> str:
        out = f"--- BROWSER RESULT ---\nSTATUS: ERROR\n{msg}"
        if hint:
            out += f"\n\n--- HINT ---\n{hint}\n-----------"
        return out

    valid_list = ", ".join(sorted(_VALID_OPS))
    if not operation:
        return _err(f"Missing 'operation'. Valid: {valid_list}.")
    if operation not in _VALID_OPS:
        return _err(f"Unknown operation {operation!r}. Valid: {valid_list}.")
    if not sandbox_dir or not sandbox_manager:
        return _err("Sandbox is not initialised — cannot run browser.")

    # SSRF guard: refuse http(s) navigation to internal/metadata hosts.
    # (file:// fixtures and about:/data: are allowed.)
    _b = _browser_blocked_url(url)
    if _b:
        return _err(f"Refused navigation: {_b}")
    for _a in (actions or []):
        if isinstance(_a, dict) and _a.get("action") == "goto":
            _b = _browser_blocked_url(_a.get("url"))
            if _b:
                return _err(f"Refused goto: {_b}")

    # Write the runner once per call. Cheap (~10 KB) and avoids stale-
    # runner bugs if the file is edited mid-session.
    try:
        runner_host_path = _get_safe_path(sandbox_dir, _BROWSER_RUNNER_FILENAME)
    except ValueError as ve:
        return _err(str(ve))
    try:
        await asyncio.to_thread(runner_host_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(runner_host_path.write_text, _runner_script())
    except Exception as e:
        return _err(f"Could not write browser runner: {e}")

    # Rewrite out_path for screenshot: must be inside /workspace and
    # reachable via the same relative path inside the container.
    container_out_path = None
    if operation == "screenshot":
        target = out_path or "screenshot.png"
        try:
            # `_get_safe_path` already rejects path escapes.
            host_out = _get_safe_path(sandbox_dir, target)
        except ValueError as ve:
            return _err(str(ve))
        await asyncio.to_thread(host_out.parent.mkdir, parents=True, exist_ok=True)
        # Translate host → container path. _to_container_path un-scopes a
        # project-scoped sandbox_dir to the root mount, so a scoped file at
        # <root>/projects/<id>/x.png maps to /workspace/projects/<id>/x.png
        # (not /workspace/x.png — which is where it would NOT exist).
        container_out_path = _to_container_path(sandbox_dir, host_out)
        screenshot_host_path = host_out  # host-side PNG, for the render check

    # Same translation for any screenshot sub-action inside interact.
    # Without this, the runner would try to write to paths that were
    # never safety-checked against the sandbox root, which either
    # silently escapes or fails in a confusing way.
    sanitised_actions = None
    if operation == "interact":
        if not isinstance(actions, list) or not actions:
            return _err(
                "interact requires a non-empty 'actions' list, e.g. "
                "[{\"action\":\"click\",\"selector\":\"...\"}, "
                "{\"action\":\"extract_text\",\"selector\":\"...\"}]."
            )
        sanitised_actions = []
        for idx, step in enumerate(actions):
            if not isinstance(step, dict):
                return _err(
                    f"actions[{idx}] must be a dict, got {type(step).__name__}"
                )
            new_step = dict(step)
            if new_step.get("action") == "screenshot":
                sub_target = new_step.get("out_path") or f"screenshot_{idx}.png"
                try:
                    host_sub = _get_safe_path(sandbox_dir, sub_target)
                except ValueError as ve:
                    return _err(f"actions[{idx}]: {ve}")
                await asyncio.to_thread(
                    host_sub.parent.mkdir, parents=True, exist_ok=True
                )
                new_step["out_path"] = _to_container_path(sandbox_dir, host_sub)
            # Heal a goto/navigate sub-action's file:// URL the same way the
            # top-level url is healed (so scoped files resolve mid-sequence).
            if new_step.get("action") in ("goto", "navigate") and new_step.get("url"):
                new_step["url"] = _resolve_file_url(sandbox_dir, new_step["url"])
            sanitised_actions.append(new_step)

    # Heal file:// URLs to the project-scoped location (no-op when not in a
    # project). Done after the SSRF guard (which already permits file://).
    url = _resolve_file_url(sandbox_dir, url)

    payload = _build_op_payload(
        op=operation,
        url=url,
        selector=selector,
        out_path=container_out_path,
        wait_until=wait_until,
        full_page=full_page,
        max_chars=max_chars,
        timeout_ms=timeout_ms,
        tor_proxy=tor_proxy,
        actions=sanitised_actions,
        stop_on_error=stop_on_error,
        click_center=kwargs.get("click_center"),
        settle_ms=kwargs.get("settle_ms"),
    )

    # For interact, the timeout budget grows with the number of actions —
    # a 30-action sequence with default per-action 30s needs substantially
    # more than the one-shot default. Cap generously (per-action * count)
    # but never drop below the single-op budget.
    effective_timeout_ms = int(timeout_ms)
    if operation == "interact":
        effective_timeout_ms = max(
            effective_timeout_ms,
            # Rough budget: each action gets the base timeout. Bound the
            # overall subprocess wait accordingly so a 10-action flow
            # doesn't get guillotined mid-sequence.
            int(timeout_ms) * max(1, len(sanitised_actions or [])),
        )

    pretty_log("Browser", f"{operation} {url or selector or ''}".strip(), icon=Icons.TOOL_BROWSER)

    cmd = (
        f"python3 -u {_BROWSER_RUNNER_FILENAME} "
        f"{shlex.quote(json.dumps(payload))}"
    )
    # Give the subprocess some slack over the in-runner timeout so an
    # actually-hung browser produces a runner-level error, not a
    # sandbox-level kill that swallows diagnostics.
    subprocess_timeout = max(60, (effective_timeout_ms // 1000) + 30)
    # When project-scoped, run from /workspace/projects/<id> so the runner
    # (written into the scoped dir as `.browser_runner.py`) is found by its
    # relative name. Passed ONLY when set, so managers without a `workdir`
    # param are unaffected (matches execute.py).
    _wd_kw = {"workdir": container_workdir} if container_workdir else {}
    try:
        # Serialize on the shared profile dir — concurrent Chromium launches
        # on one user-data-dir crash each other (see _BROWSER_PROFILE_LOCK).
        async with _BROWSER_PROFILE_LOCK:
            output, exit_code = await asyncio.to_thread(
                sandbox_manager.execute, cmd, timeout=subprocess_timeout, **_wd_kw
            )
    except Exception as e:
        pretty_log("Browser Failed", f"{operation}: {type(e).__name__}: {e}",
                   icon=Icons.TOOL_BROWSER, level="ERROR")
        return _err(f"sandbox execute failed: {e}")

    ok, parsed = _parse_runner_output(output or "")
    if not ok:
        # Surface the ACTUAL failure cause in the operator's live stream,
        # not just "runner exit 1". `parsed` holds the runner's
        # [BROWSER_ERR] line (or, when the runner died before emitting a
        # sentinel, the raw stdout+stderr tail — stderr is merged in via
        # docker exec demux=False). Truncated so a long Playwright
        # traceback doesn't flood the stream; the full text still goes to
        # the agent in the returned error below.
        _cause = str(parsed).replace("\n", " ⏎ ")
        if len(_cause) > 300:
            _cause = _cause[:300] + "…"
        pretty_log("Browser Failed", f"{operation}: runner exit {exit_code} — {_cause}",
                   icon=Icons.TOOL_BROWSER, level="WARNING")
        return _err(
            f"Runner failed (exit {exit_code}): {parsed}",
            hint=(
                "If this is a navigation timeout, try wait_until='domcontentloaded' "
                "or raise timeout_ms. If the error mentions 'headless_shell not "
                "found' or 'Executable doesn't exist', the sandbox was provisioned "
                "before the Chromium pre-install was added — delete "
                "`/root/.supercharged` inside the container and retry. If the "
                "error says the op needs a URL, call `operation=\"navigate\"` "
                "once first, or pass `url=...` on this call."
            ),
        )

    # Workspace research dedup: record the URL we actually loaded so
    # a later research turn can ask "did I already pull this?" via the
    # workspace tool. Operations that meaningfully fetch a page
    # (navigate / extract_text / click / interact / screenshot) carry
    # a `parsed['url']` (or `final_url`). Non-fatal — must never break
    # a successful browser turn.
    if workspace_model is not None and getattr(workspace_model, "enabled", False):
        try:
            _hit_url = parsed.get("url") or parsed.get("final_url")
            if _hit_url:
                workspace_model.record_research_artifact(
                    url=_hit_url, source="browser",
                    title=parsed.get("title") or parsed.get("final_title") or "",
                    note=operation,
                )
        except Exception:  # noqa: BLE001
            pass

    # Pretty-print the success result for the LLM. Keep each op's
    # return shape deterministic so downstream prompts can rely on it.
    header = f"--- BROWSER RESULT ---\nSTATUS: OK\nOP: {operation}"
    js_diag = _format_js_diagnostics(parsed)
    if operation == "navigate":
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"HTTP_STATUS: {parsed.get('status')}\n"
            f"TITLE: {parsed.get('title')}{js_diag}{_pre_interaction_line(parsed)}"
        )
    if operation == "extract_text":
        body = parsed.get("text", "")
        trunc = " (truncated)" if parsed.get("truncated") else ""
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"TITLE: {parsed.get('title')}\n"
            f"LENGTH: {parsed.get('length')}{trunc}{js_diag}\n"
            f"--- TEXT ---\n{body}"
        )
    if operation == "click":
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"TITLE: {parsed.get('title')}"
        )
    if operation == "screenshot":
        # Echo the host-relative path so the user can reference it via
        # /api/download/<name>.
        host_rel = str(Path(parsed.get("path", "")).relative_to("/workspace")) if parsed.get("path", "").startswith("/workspace/") else parsed.get("path", "")
        # Objective render check: read the PNG the runner just wrote and
        # report whether the frame actually contains visual content. This is
        # the un-gameable counter to the model confabulating "it renders"
        # over a blank/sky-only capture (verifier reads this as EVIDENCE).
        render_line = ""
        try:
            probe = analyze_screenshot_render(screenshot_host_path)
            if probe:
                render_line = (
                    f"\nRENDER_CHECK: {probe['verdict'].upper()} "
                    f"(dominant_colour={probe['dominant_pct']:.0%}, "
                    f"distinct_colours={probe['distinct_colors']}) — {probe['note']}"
                )
        except Exception:
            pass
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"SAVED: {host_rel}\n"
            f"DOWNLOAD: /api/download/{host_rel}{js_diag}{render_line}"
            f"{_pre_interaction_line(parsed)}"
        )
    if operation == "close":
        return f"{header}\nPROFILE_DIR: {parsed.get('profile_dir')}\nCLEARED: {parsed.get('closed')}"
    if operation == "interact":
        action_results = parsed.get("actions") or []
        ok_count = sum(1 for r in action_results if r.get("ok"))
        err_count = len(action_results) - ok_count
        lines = [
            header,
            f"FINAL_URL: {parsed.get('final_url')}",
            f"FINAL_TITLE: {parsed.get('final_title')}",
            f"ACTIONS: {ok_count} OK, {err_count} error{'s' if err_count != 1 else ''} "
            f"(of {len(action_results)} total)",
        ]
        # Aborted sequences get a loud banner so the agent's next-turn
        # planner can't miss the abort. Without this the "5 OK / 48 err"
        # summary could be mistaken for a partial success that needs
        # retry of individual actions, when the right fix is to retry
        # the whole sequence with a corrected goto URL.
        if parsed.get("aborted"):
            lines.append(
                f"⚠ SEQUENCE ABORTED: {parsed.get('abort_reason') or 'goto_failed'}. "
                "Remaining actions were NOT executed because the initial "
                "navigation failed — page.click/fill/extract on an error "
                "page would have just timed out one-by-one. Fix the URL "
                "and retry the whole interact call."
            )
        if js_diag:
            # Surface uncaught exceptions / console errors collected across
            # the whole interact sequence — strip the leading newline the
            # helper adds (we're already line-buffered here).
            lines.append(js_diag.lstrip("\n"))
        lines.append("--- PER-ACTION RESULTS ---")
        for r in action_results:
            status = "OK" if r.get("ok") else "ERR"
            idx = r.get("index")
            act = r.get("action")
            if r.get("ok"):
                if act == "extract_text":
                    text = r.get("text", "")
                    trunc = " (truncated)" if r.get("truncated") else ""
                    # Keep per-line output readable — newlines inside the
                    # extracted text get escaped so each action is ONE
                    # log line, but a TEXT block follows for full fidelity.
                    summary = f"len={r.get('length')}{trunc} sel={r.get('selector')!r}"
                    lines.append(f"  [{idx}] {status} {act}: {summary}")
                    lines.append(f"      TEXT: {text[:500]}" + (" ..." if len(text) > 500 else ""))
                elif act == "click":
                    lines.append(f"  [{idx}] {status} click {r.get('selector')!r}")
                elif act == "goto":
                    lines.append(
                        f"  [{idx}] {status} goto → {r.get('url')} "
                        f"(title={r.get('title')!r})"
                    )
                elif act == "screenshot":
                    host_rel = r.get("path", "")
                    if isinstance(host_rel, str) and host_rel.startswith("/workspace/"):
                        host_rel = host_rel[len("/workspace/"):]
                    lines.append(
                        f"  [{idx}] {status} screenshot → {host_rel} "
                        f"(download: /api/download/{host_rel})"
                    )
                elif act == "fill":
                    lines.append(
                        f"  [{idx}] {status} fill {r.get('selector')!r} "
                        f"len={r.get('length')}"
                    )
                elif act == "wait_for_selector":
                    lines.append(
                        f"  [{idx}] {status} wait_for_selector {r.get('selector')!r}"
                    )
                elif act == "sleep":
                    lines.append(f"  [{idx}] {status} sleep {r.get('ms')}ms")
                else:
                    lines.append(f"  [{idx}] {status} {act}")
            else:
                lines.append(
                    f"  [{idx}] {status} {act}: {r.get('error')}"
                )
        return "\n".join(lines)
    # Defensive default — never hit because we validated above.
    return f"{header}\nRAW: {json.dumps(parsed)}"
