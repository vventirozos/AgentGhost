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
import os
import time
from typing import Dict, Optional
from urllib.parse import urlparse as _urlparse

from ..utils.logging import Icons, pretty_log
from .file_system import _get_safe_path, _to_container_path
from .outcome import ToolOutcome

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


def _safe_int(v, default: int) -> int:
    """int() an LLM-supplied value without letting a non-numeric string
    (e.g. timeout_ms="30s") raise out of the tool."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


_RUNNER_SRC_CACHE = None


def _runner_script() -> str:
    """Return the Playwright runner source, read from `browser_runner.py`.

    ⚠ IT USED TO LIVE HERE AS A 1,205-LINE STRING — 48% of this file —
    and that was a trap generator. Four separate defects in one feature
    came from it, every one invisible to the usual checks:

      * helpers anchored on text that also appears INSIDE the string
        landed in the runner instead of the module. They compiled, and
        `grep -n "^def …"` showed them at column 0.
      * `urlparse` and `os` were imported only inside the string, so
        module code referencing them raised NameError — which a bare
        `except Exception` then turned into a plausible return value,
        silently disabling a whole feature.
      * worst: a SYNTAX ERROR anywhere in those 1,205 lines was
        undetectable. Measured — breaking `async def op_navigate(op):`
        left `py_compile` green, `import` green, and the browser tests
        green; it would have failed only inside the container, on the
        operator's next browser call.

    As a real file it is compiled by every ordinary tool, linted,
    navigable, and its names cannot be confused with this module's. The
    accessor is kept so all seven callers are unchanged.

    Never imported host-side (it imports playwright, which lives in the
    sandbox image); it is read as text and written into the workspace.
    """
    global _RUNNER_SRC_CACHE
    if _RUNNER_SRC_CACHE is None:
        _RUNNER_SRC_CACHE = (Path(__file__).parent
                             / "browser_runner.py").read_text(encoding="utf-8")
    return _RUNNER_SRC_CACHE


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
    post_click_ms: Optional[int] = None,
    nav_text_chars: Optional[int] = None,
    allowed_local_ports=None,
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
        # _safe_int, not int(): these are LLM-supplied (settle_ms="2s" was
        # possible) and a ValueError here raised OUT of the tool as a raw
        # traceback instead of an op error.
        payload["settle_ms"] = _safe_int(settle_ms, 0)
    if post_click_ms is not None:
        payload["post_click_ms"] = _safe_int(post_click_ms, 800)
    if nav_text_chars is not None:
        # Clamp like max_chars so a huge value can't flood the context.
        payload["nav_text_chars"] = max(0, min(_safe_int(nav_text_chars, 8 * 1024), _MAX_TEXT_CHARS))
    if allowed_local_ports:
        payload["allowed_local_ports"] = sorted(
            int(p) for p in allowed_local_ports)
    return payload


# ── Unreachable hidden services (2026-08-15) ──────────────────────────
# LIVE FAILURE. Asked for dark-web news, the agent got 7 onion results,
# picked #1, navigated to it — `net::ERR_SOCKS_CONNECTION_FAILED` — then
# navigated to THE SAME URL again, failed identically, and the no-progress
# loop-breaker forced a grounded conclusion. The operator got a list of
# links instead of content, with six untried candidates sitting there.
#
# Measured afterwards through the same Tor the browser uses: a live onion
# loads in ~3s, while the one it chose (a Keybase profile — that hidden
# service is gone) is indistinguishable from a fabricated address. So the
# error means "this service is dead", not "Tor is broken", and the only
# useful next action is a DIFFERENT result.
#
# Per-process and TTL'd: hidden services do come back, and a restart
# re-probes everything.
#
# ⚠ Defined HERE, after `_runner_script`, on purpose. That function
# returns the Playwright runner as one long string which contains its own
# `async def _install_ssrf_guard(...)` line — anchoring an insert on text
# that also appears inside it puts the code in the SCRIPT instead of the
# module. It still compiles, and the functions still look top-level to
# grep. Caught only by importing the module and finding them absent.
#: 600s, deliberately NOT longer. Tor's default MaxCircuitDirtiness is
#: 600s, so a longer memo outlives the circuit whose failure created it —
#: i.e. it would keep punishing a service after the cause was gone.
_DEAD_ONION_TTL = 600.0
_DEAD_ONIONS: Dict[str, float] = {}
#: Both dicts are swept only for the host being queried, so a long-lived
#: process that meets thousands of one-off dead onions would grow without
#: bound (measured: 5000 entries ≈ 1 MB). Cap them like this codebase's
#: other in-process caches, evicting oldest-first.
_ONION_MEMO_MAX = 512


def _cap(d: Dict[str, object], key_time) -> None:
    if len(d) <= _ONION_MEMO_MAX:
        return
    for host in sorted(d, key=key_time)[:len(d) - _ONION_MEMO_MAX]:
        d.pop(host, None)

#: The ONE Chromium shape that means "the SOCKS proxy could not reach the
#: target". Deliberately narrow (R1 review): the first version also listed
#: ERR_PROXY_CONNECTION_FAILED, ERR_TUNNEL_CONNECTION_FAILED and
#: ERR_NAME_NOT_RESOLVED — but those mean the proxy itself was
#: unreachable, or Chromium got no proxy and tried real DNS on a .onion.
#: Those are OUR failures, not the service's, and treating them as
#: "this hidden service is dead" blacklists LIVE sites (proven: the known-
#: good DuckDuckGo onion was memoised from a synthetic proxy error) while
#: handing the model a confidently false diagnosis that forecloses the
#: correct one. With Tor down, every candidate would be burned at zero
#: cost and the whole result set declared dead.
_ONION_UNREACHABLE_MARKERS = ("ERR_SOCKS_CONNECTION_FAILED",)

#: CONSECUTIVE strikes before a hidden service is treated as dead. TWO,
#: not one (R1): onion rendezvous failures are transient and circuit-
#: dependent, so a single failure is one bad circuit's opinion.
#:
#: CONSECUTIVE is load-bearing and was NOT what the first version did
#: (R2). With no success path, "2 strikes" meant "any 2 failures inside
#: the TTL" — measured: fail, load ten pages successfully, fail once more,
#: and the host is declared offline while the model is told "other onion
#: sites load fine… retrying will fail the same way" about a service it
#: read from a minute ago. The sibling engine breaker gets this right
#: (a win pops the record); the asymmetry was the tell.
_DEAD_ONION_STRIKES = 2
#: {onion_host: [strike_count, first_strike_monotonic]}
_ONION_STRIKES: Dict[str, list] = {}


def _runner_first_url(operation, url, actions, sandbox_dir):
    """The URL the runner will ACTUALLY dial first — the only URL the
    memo may refuse or blame.

    Mirrors `_runner_script`'s own rule, which R4 measured the host side
    disagreeing with in BOTH directions:

      * the runner tests `actions[0]["action"] == "goto"` — nothing else.
        The host also accepted `"navigate"` (not a valid interact action
        at all: the runner's dispatch is goto-only) and additionally
        required a url. Consequences measured: `actions[0]=navigate` made
        the host skip the check while the runner dialled the top-level
        url (banned host re-dialled, nothing learned); and
        `actions[0]={"action":"goto"}` with no url made the host blame
        the top-level url the runner never touched — a ban on an
        uncontacted host, the inversion R3 called the worst outcome.
      * when the first action is NOT a goto, the runner performs an
        IMPLICIT initial navigation to `url` — or, when there is no url,
        to the `.last_url` sidecar. The memo was blind to that entire
        path (R4): `navigate(A)` then `click`/`extract_text` with no url
        is the flow this tool's own docstring teaches, and on it the memo
        never armed and a banned host was re-dialled at full Tor cost.
    """
    # R5: ops that dial NOTHING. `close` only rmtree's the profile, and
    # `navigate` without a url is a parameter error the runner raises on —
    # neither consults the sidecar. Falling through to it meant `close`
    # was REFUSED whenever the sidecar happened to name a banned host,
    # i.e. the memo's own state blocked the one operation that clears the
    # profile, for the full TTL; and a plain missing-parameter mistake was
    # answered with "pick a DIFFERENT result from your search".
    if operation in ("close", "navigate"):
        return url or ""
    if operation == "interact" and actions:
        first = actions[0] if isinstance(actions[0], dict) else {}
        if first.get("action") == "goto":
            return first.get("url") or ""
    if url:
        return url
    # Sidecar fallback, read host-side from the file the runner writes.
    #
    # ⚠ UN-SCOPE FIRST (R5). The runner's profile is hardcoded to
    # `/workspace/.browser_profile`, and `/workspace` is the bind mount of
    # the sandbox ROOT — but `registry.py` hands this function the
    # PROJECT-scoped dir (`<root>/projects/<id>`) whenever a project is
    # active. Reading `<root>/projects/<id>/.browser_profile/.last_url`
    # finds nothing, so in a project session — the majority of real work —
    # the memo neither refused a banned host nor learned from the failure,
    # and the dead onion was re-dialled over Tor exactly as before the fix.
    # `_to_container_path` un-scopes for the same reason.
    try:
        root = Path(str(sandbox_dir or "."))
        if root.parent.name == "projects":
            root = root.parent.parent
        with open(root / _BROWSER_PROFILE_DIR / ".last_url",
                  "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:  # noqa: BLE001
        return ""


def _onion_host(url: str) -> str:
    """The .onion host of `url`, or "". Keyed on HOST: a dead hidden
    service is dead at every path, and the live failure retried a
    different path of the same host."""
    try:
        host = (_urlparse(str(url or "")).hostname or "").lower()
    except (ValueError, TypeError) as exc:
        # NARROW on purpose. The first version caught bare `Exception` and
        # returned "" — which silently turned a NameError (module-level
        # `urlparse` was never imported; the `from urllib.parse import`
        # near the top of this file lives INSIDE the runner-script string)
        # into "this is not an onion", disabling the whole feature while
        # every test and the compiler stayed happy.
        logger.debug("onion host parse failed for %r: %s", url, exc)
        return ""
    return host if host.endswith(".onion") else ""


def _mark_onion_alive(url: str) -> None:
    """A successful load clears the strike streak for this host.

    Without this the count is cumulative-within-a-window rather than
    consecutive, so an ordinarily-flaky-but-working onion accumulates its
    way to a ban. Does NOT lift an existing ban: that expires on its own
    TTL, and re-dialling a banned host is exactly what the memo prevents,
    so a success there is not observable anyway.
    """
    host = _onion_host(url)
    if host:
        _ONION_STRIKES.pop(host, None)


def _mark_onion_dead(url: str, cause: str) -> None:
    """Record a Tor-layer failure; declare the service dead on the SECOND.

    Clearnet hosts are never recorded — a SOCKS error there means
    something else entirely. `GHOST_DEAD_ONION_MEMO=0` disables the whole
    mechanism, matching the engine breaker's kill switch (R1 review: the
    memo had none, and a false positive was unrecoverable for its TTL)."""
    if os.environ.get("GHOST_DEAD_ONION_MEMO", "1") != "1":
        return
    host = _onion_host(url)
    if not host or not any(m in str(cause)
                           for m in _ONION_UNREACHABLE_MARKERS):
        return
    now = time.monotonic()
    rec = _ONION_STRIKES.get(host)
    # A strike older than the TTL is stale evidence — start over rather
    # than accumulating one failure an hour into a ban.
    if rec is None or (now - rec[1]) >= _DEAD_ONION_TTL:
        _ONION_STRIKES[host] = [1, now]
        _cap(_ONION_STRIKES, lambda h: _ONION_STRIKES[h][1])
        return
    rec[0] += 1
    if rec[0] >= _DEAD_ONION_STRIKES:
        _DEAD_ONIONS[host] = now
        _ONION_STRIKES.pop(host, None)
        _cap(_DEAD_ONIONS, lambda h: _DEAD_ONIONS[h])


def _dead_onion_notice(url: str):
    """A directive message when `url`'s host is a known-dead onion."""
    host = _onion_host(url)
    if not host:
        return None
    if os.environ.get("GHOST_DEAD_ONION_MEMO", "1") != "1":
        return None
    at = _DEAD_ONIONS.get(host)
    if at is None or (time.monotonic() - at) >= _DEAD_ONION_TTL:
        _DEAD_ONIONS.pop(host, None)
        return None
    return (
        f"The hidden service {host} is UNREACHABLE — an earlier navigation "
        f"failed at the Tor layer, which for an onion address means the "
        f"service is offline, not that Tor is broken (other onion sites "
        f"load fine). Retrying it will fail the same way. Pick a DIFFERENT "
        f"result from your search: onion indexes are full of dead links, so "
        f"working down the list is normal. If every candidate is dead, say "
        f"so and report the links you found rather than looping."
    )


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
    # "uniform" needs BOTH a dominant colour AND few distinct buckets. A
    # dominant-colour-only rule (the original `>= 0.80 or`) false-flagged
    # every white-background TEXT page as "BLANK/sky-only" — ordinary docs
    # pages are >80% white but their anti-aliased glyphs span dozens of
    # colour buckets, while a real blank/sky/loading frame has almost none.
    # Poisoned evidence cuts both ways: a false BLANK invites the verifier
    # to refute a true "the page renders" claim.
    if (dominant_pct >= 0.80 and distinct <= 24) or distinct <= 6:
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


# Schemes the browser is allowed to navigate. file:// renders self-play
# fixtures; about:/data: are inert. Everything else (chrome://, view-source:,
# ftp://, gopher://, …) is refused rather than allowed-by-default.
_BROWSER_ALLOWED_SCHEMES = frozenset({"http", "https", "file", "about", "data"})


def _browser_blocked_url(u: Optional[str], *, anonymous: bool = False,
                         allowed_local_ports=frozenset()) -> Optional[str]:
    """SSRF guard for the browser: block http(s) navigation to internal /
    loopback / link-local / metadata hosts (which the host-network sandbox
    can otherwise reach), while ALLOWING file:// (self-play fixtures render
    as file:// pages) and about:/data:. Returns a refusal reason or None.

    ``anonymous`` — in Tor mode we must NOT resolve the hostname on the host
    (getaddrinfo would leak the DNS query for the site we're about to visit,
    defeating the browser's DNS-over-SOCKS hardening). Tor can't route to an
    internal address anyway, so skipping resolution loses no protection there;
    literal-IP internal targets are still blocked without resolving.

    ``allowed_local_ports`` — loopback ports of SUPERVISED sandbox services
    (sandbox/services.py registry). An explicit-loopback URL on one of these
    ports is admitted (the agent hosting an app must be able to drive it);
    literal hosts only, so no DNS/rebind surface. Everything else loopback
    stays blocked."""
    if not u:
        return None
    from urllib.parse import urlparse
    try:
        parsed = urlparse(str(u))
        scheme = (parsed.scheme or "").lower()
    except Exception:
        # Fail CLOSED: an unparseable URL is refused, not allowed. (A security
        # guard should not fail open.)
        return f"refused unparseable URL: {u!r}"
    if scheme not in _BROWSER_ALLOWED_SCHEMES:
        return f"refused disallowed scheme {scheme!r} (only http/https/file/about/data)."
    if scheme in ("http", "https"):
        if allowed_local_ports:
            try:
                host = (parsed.hostname or "").lower()
                port = parsed.port or (443 if scheme == "https" else 80)
                if host in ("127.0.0.1", "localhost", "::1") \
                        and port in allowed_local_ports:
                    return None
            except ValueError:
                pass  # invalid port literal → let the SSRF guard refuse it
        from ..utils.helpers import url_ssrf_reason
        return url_ssrf_reason(u, resolve=not anonymous)
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
    allowed_local_ports=None,
    **kwargs,
):
    """Run a single browser operation inside the sandbox.

    Operations:
      navigate: go to a URL, return {status, url, title, text, length,
                truncated}. `text` is a capped (~8 KB) innerText preview —
                you usually do NOT need a follow-up extract_text; only call
                extract_text for the FULL page or a specific CSS selector.
      extract_text: go to URL (optional), return innerText — body or
                    a CSS selector. Truncates at `max_chars` (64 KB default).
      click: click a selector, wait for load, return {url, title, text} —
             the post-click page text (~8 KB preview), same as navigate.
             NOTE: reloads the page first (fresh context) — an element that
             only exists after a previous click (opened window/menu) will
             NOT be there; use `interact` for any multi-step flow.
      screenshot: save a PNG to `out_path` inside /workspace.
      close: delete the persistent profile so the next session is fresh.
      interact: run a list of sub-actions in ONE Chromium context
                (click + extract + screenshot etc. share transient DOM
                state). Required for multi-step SPA flows where the
                atomic per-op re-navigation would wipe intermediate
                state. Includes an `evaluate` sub-action that runs a JS
                expression in the page and returns its value — the
                exact-state probe for apps the agent built itself.

    `tor_proxy` is forwarded as Chromium's `--proxy-server` with
    `--host-resolver-rules` forcing DNS through the proxy.
    """
    # --- PARAMETER HALLUCINATION HEALING (matches execute.py style) ---
    operation = operation or kwargs.get("op") or kwargs.get("action")
    url = url or kwargs.get("link") or kwargs.get("href")
    selector = selector or kwargs.get("css") or kwargs.get("query_selector")
    out_path = out_path or kwargs.get("path") or kwargs.get("filename")
    actions = actions or kwargs.get("steps") or kwargs.get("sequence")

    # Coerce LLM-supplied numerics up front (JSON args aren't type-enforced):
    # a non-numeric timeout_ms/max_chars would otherwise raise ValueError out
    # of the tool. max_chars is also CLAMPED so a huge value can't flood the
    # model context with a whole page (the _MAX_TEXT_CHARS ceiling was dead).
    timeout_ms = _safe_int(timeout_ms, 30000)
    if max_chars is not None:
        max_chars = max(256, min(_safe_int(max_chars, _MAX_TEXT_CHARS), _MAX_TEXT_CHARS))

    def _err(msg: str, hint: str = None, ran: bool = False) -> "ToolOutcome":
        """A browser failure — and it says so.

        `--- BROWSER RESULT ---` heads the string, so the loop's anchored
        failure-prefix rule never matched and NONE of these were failures to
        the dispatch loop: measured, 42 live rows over 32 turns booked 0/42
        as failures — no strike, no pre-flight guard record, no competence
        signal, and `STATUS: ERROR` reported to the model as a SUCCEEDED
        operation. That is the single largest remaining loop-vs-corpus
        disagreement class, larger than the exit-code class §4DO closed.
        Nothing here ran, so nothing changed.
        """
        from .outcome import ToolOutcome
        out = f"--- BROWSER RESULT ---\nSTATUS: ERROR\n{msg}"
        if hint:
            out += f"\n\n--- HINT ---\n{hint}\n-----------"
        # `world_changed` is a PARAMETER now: `_err` is called both before
        # the runner starts and after it has already navigated, clicked and
        # filled. Declaring "nothing changed" for the post-execution case
        # lied to the pre-flight guard and to the loop-breaker's
        # world-changed reset on 12 live rows.
        return ToolOutcome.failed(out, world_changed=ran,
                                  reason_code="browser_error")

    def _reject(msg: str):
        """An ARGUMENT refusal: nothing ran, and the message names what to
        change. Booking these FAILED armed the pre-flight guard against the
        corrected re-issue — the pathology the guard's own docstring
        records."""
        from .outcome import ToolOutcome
        return ToolOutcome.rejected(
            f"--- BROWSER RESULT ---\nSTATUS: ERROR\n{msg}",
            reason_code="browser_bad_arguments")

    valid_list = ", ".join(sorted(_VALID_OPS))
    if not operation:
        return _reject(f"Missing 'operation'. Valid: {valid_list}.")
    operation = str(operation)
    if operation not in _VALID_OPS:
        return _reject(f"Unknown operation {operation!r}. Valid: {valid_list}.")
    if not sandbox_dir or not sandbox_manager:
        return _err("Sandbox is not initialised — cannot run browser.")

    # Fail-closed (§4P): under --mandatory-tor a falsy proxy on a PUBLIC target
    # is replaced with the loopback Tor default so Chromium (a subprocess the
    # socket guard can't see) never navigates a public URL cleartext. A
    # loopback/LAN target stays direct (Tor can't route it; resolve returns
    # unchanged), preserving the supervised-sandbox-service capability.
    from ..utils.egress_guard import resolve_egress_proxy as _resolve_egress_proxy
    # ⚠ DECIDE ON THE URL ACTUALLY DIALED FIRST, not the top-level `url`
    # (§5 lens A). For `interact` the runner dials `actions[0]`'s goto target
    # before the top-level url, so a loopback top-level url with a PUBLIC
    # first goto yielded proxy=None and would have launched Chromium cleartext
    # against the public host. Currently unreachable (the sole caller always
    # threads a truthy tor_proxy and --mandatory-tor aborts on a dead one),
    # but it becomes a live deanonymization leak the moment any in-process
    # caller runs with tor_proxy=None. `_runner_first_url` is the same
    # host/runner "which URL is dialed first" resolver the SSRF pre-flight
    # already uses below — reuse it here so both guards agree.
    _proxy_decision_url = _runner_first_url(operation, url, actions, sandbox_dir)
    tor_proxy = _resolve_egress_proxy(tor_proxy, _proxy_decision_url or url)

    # SSRF guard: refuse http(s) navigation to internal/metadata hosts.
    # (file:// fixtures and about:/data: are allowed; loopback ports of
    # supervised sandbox services are admitted — see sandbox/services.py.)
    # In Tor mode we skip host-side DNS resolution so the guard itself
    # can't leak the DNS query.
    _anon = bool(tor_proxy)
    try:
        _svc_ports = frozenset(int(p) for p in (allowed_local_ports or ()))
    except (TypeError, ValueError):
        _svc_ports = frozenset()
    _b = _browser_blocked_url(url, anonymous=_anon,
                              allowed_local_ports=_svc_ports)
    if _b:
        return _reject(f"Refused navigation: {_b}")
    # ("goto", "navigate"): the sanitiser below heals BOTH spellings'
    # file:// URLs, so the guard must inspect both — checking only "goto"
    # would let a future runner-side "navigate" alias skip the host-side
    # SSRF pre-flight silently.
    for _a in (actions or []):
        if isinstance(_a, dict) and _a.get("action") in ("goto", "navigate"):
            _b = _browser_blocked_url(_a.get("url"), anonymous=_anon,
                                      allowed_local_ports=_svc_ports)
            if _b:
                return _reject(f"Refused goto: {_b}")

    # Write the runner once per call. Cheap (~10 KB) and avoids stale-
    # runner bugs if the file is edited mid-session.
    try:
        runner_host_path = _get_safe_path(sandbox_dir, _BROWSER_RUNNER_FILENAME)
    except ValueError as ve:
        return _reject(str(ve))
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
            return _reject(str(ve))
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
    # container out_path → host path for every interact screenshot, so the
    # objective render check runs on those too (previously ONLY the atomic
    # screenshot op was checked — routing a capture through interact silently
    # bypassed the whole anti-"it renders" apparatus).
    _interact_shot_hosts: dict = {}
    if operation == "interact":
        if not isinstance(actions, list) or not actions:
            return _reject(
                "interact requires a non-empty 'actions' list, e.g. "
                "[{\"action\":\"click\",\"selector\":\"...\"}, "
                "{\"action\":\"extract_text\",\"selector\":\"...\"}]."
            )
        sanitised_actions = []
        # True when the runner will have ALREADY navigated (implicitly)
        # before the first explicit goto — see `_runner_first_url`. Its
        # target is checked by the `_nav_url` short-circuit instead.
        _saw_nav = not (isinstance(actions[0], dict)
                        and actions[0].get("action") == "goto") \
            if actions else False
        for idx, step in enumerate(actions):
            if not isinstance(step, dict):
                return _reject(
                    f"actions[{idx}] must be a dict, got {type(step).__name__}"
                )
            new_step = dict(step)
            # Clamp a per-step max_chars the same way as the top-level one so
            # one interact step can't dump a whole page into context.
            if "max_chars" in new_step:
                new_step["max_chars"] = max(256, min(_safe_int(new_step.get("max_chars"), _MAX_TEXT_CHARS), _MAX_TEXT_CHARS))
            if new_step.get("action") == "screenshot":
                sub_target = new_step.get("out_path") or f"screenshot_{idx}.png"
                try:
                    host_sub = _get_safe_path(sandbox_dir, sub_target)
                except ValueError as ve:
                    return _reject(f"actions[{idx}]: {ve}")
                await asyncio.to_thread(
                    host_sub.parent.mkdir, parents=True, exist_ok=True
                )
                new_step["out_path"] = _to_container_path(sandbox_dir, host_sub)
                _interact_shot_hosts[new_step["out_path"]] = host_sub
            # Heal a goto/navigate sub-action's file:// URL the same way the
            # top-level url is healed (so scoped files resolve mid-sequence).
            if new_step.get("action") in ("goto", "navigate") and new_step.get("url"):
                new_step["url"] = _resolve_file_url(sandbox_dir, new_step["url"])
                # R1 J2: the memo was navigate-only, so the identical-retry
                # loop it exists to break was still fully reachable through
                # `interact` — which is exactly what the OTHER failure hint
                # tells the model to switch to. Only the FIRST navigating
                # hop is refused (R3): a dead goto at index N>0 used to
                # return before ANYTHING ran, so a ten-step sequence whose
                # step 8 was dead lost steps 0-7 too. A later dead hop is
                # left to the runner, which aborts the sequence itself and
                # reports the failure — which the scan below then records.
                _dead_step = (_dead_onion_notice(new_step["url"])
                              if not _saw_nav else None)
                _saw_nav = True
                if _dead_step:
                    pretty_log("Browser Skipped",
                               f"{_onion_host(new_step['url'])} is a "
                               f"known-dead hidden service (actions[{idx}])",
                               icon=Icons.TOOL_BROWSER, level="WARNING")
                    return _err(_dead_step, hint=None)
            sanitised_actions.append(new_step)

    # Heal file:// URLs to the project-scoped location (no-op when not in a
    # project). Done after the SSRF guard (which already permits file://).
    url = _resolve_file_url(sandbox_dir, url)

    # Known-dead hidden service? Answer immediately with the next action
    # instead of paying another Tor round trip to relearn it. This is the
    # exact loop that cost a live turn: the same dead onion twice, a
    # loop-breaker, then links instead of content.
    # `_runner_first_url` replaced a hand-rolled `_first_is_goto` that
    # disagreed with the runner in both directions (R4).
    _nav_url = _runner_first_url(operation, url, sanitised_actions,
                                 sandbox_dir)
    _dead = _dead_onion_notice(_nav_url)
    if _dead:
        pretty_log("Browser Skipped",
                   f"{_onion_host(_nav_url)} is a known-dead hidden service — "
                   f"told the model to try another result",
                   icon=Icons.TOOL_BROWSER, level="WARNING")
        return _err(_dead, hint=None)

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
        post_click_ms=kwargs.get("post_click_ms"),
        nav_text_chars=kwargs.get("nav_text_chars"),
        allowed_local_ports=_svc_ports,
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
        return _err(f"sandbox execute failed: {e}", ran=True)

    ok, parsed = _parse_runner_output(output or "")

    # Chromium launch-race retry (2026-07-31, probe req 68033190). The first
    # atomic op after recent browser activity can hit `TargetClosedError:
    # BrowserType.launch_persistent_context: Target page, context or browser
    # has been closed` — the previous call's Chromium is still tearing down
    # on the SHARED profile dir when the new launch grabs it. The lock
    # serialises our runner subprocesses, not Chromium's own shutdown. One
    # bounded retry after a short settle absorbs it; without this the agent
    # self-heals but burns ~2 turns and an error strike doing so.
    if not ok and "TargetClosedError" in str(parsed):
        pretty_log("Browser Retry",
                   "Chromium launch race (TargetClosedError) — retrying once "
                   "after settle",
                   icon=Icons.RETRY, level="WARNING")
        await asyncio.sleep(1.5)
        try:
            async with _BROWSER_PROFILE_LOCK:
                output, exit_code = await asyncio.to_thread(
                    sandbox_manager.execute, cmd,
                    timeout=subprocess_timeout, **_wd_kw
                )
            ok, parsed = _parse_runner_output(output or "")
        except Exception as e:
            logger.debug("TargetClosedError retry failed: %s", e)

    # Degraded-milestone navigate retry (2026-07-18). Over Tor a slow exit
    # circuit stalls subresources and even `domcontentloaded` can miss the
    # window, producing two IDENTICAL timeouts that the no-progress breaker
    # then kills (observed: jordanmechner.com navigate failed 2x → turn
    # forced to conclude). Chromium cannot do SOCKS auth, so a fresh Tor
    # circuit per attempt isn't available — but `commit` (navigation
    # committed, HTML streaming) is a much weaker milestone that usually
    # lands on the same circuit. One retry, only for proxied navigates that
    # timed out, and only when the caller didn't already pin `commit`.
    if (not ok and operation == "navigate" and tor_proxy
            and payload.get("wait_until") != "commit"
            and "timeout" in str(parsed).lower()):
        pretty_log("Browser Retry",
                   "navigate timed out — retrying once with wait_until='commit' "
                   "(weakest milestone; slow Tor exit suspected)",
                   icon=Icons.RETRY, level="WARNING")
        retry_payload = dict(payload, wait_until="commit")
        retry_cmd = (
            f"python3 -u {_BROWSER_RUNNER_FILENAME} "
            f"{shlex.quote(json.dumps(retry_payload))}"
        )
        try:
            async with _BROWSER_PROFILE_LOCK:
                output, exit_code = await asyncio.to_thread(
                    sandbox_manager.execute, retry_cmd,
                    timeout=subprocess_timeout, **_wd_kw
                )
            ok, parsed = _parse_runner_output(output or "")
        except Exception as e:
            logger.debug("navigate commit-retry failed: %s", e)
    if ok and operation not in ("interact", "close"):
        # A successful load clears this host's strike streak, which is
        # what makes the count CONSECUTIVE rather than cumulative.
        #
        # NOT for `interact`: that returns a [BROWSER_OK] payload even
        # when the sequence ABORTED on a failed goto, so `ok` is True and
        # `final_url` is the host that just failed. Clearing there undid
        # the strike the per-action scan below was about to record — the
        # streak never reached two and the memo could never arm. The
        # per-action results carry their own ok flag and are handled in
        # that scan.
        _mark_onion_alive(
            (parsed or {}).get("url")
            or (parsed or {}).get("final_url") or url)
    if ok and operation == "interact":
        # R1 J2: an interact `goto` that fails at the Tor layer is
        # TERMINAL for the sequence but is reported as a per-action result
        # inside a [BROWSER_OK] payload — so `ok` is True and the failure
        # branch below never runs. The memo was therefore navigate-only,
        # leaving the identical-retry loop fully available through the very
        # operation the other hint recommends.
        try:
            # The key is "actions" — `op_interact` returns
            # `{"actions": results, ...}` (browser.py:1233) and the
            # initial-goto failure shape at :918 uses it too. "results" is
            # only the runner's LOCAL variable name; scanning for it would
            # have been a silent no-op, which is this feature's recurring
            # failure mode.
            for _step in (parsed or {}).get("actions", []) or []:
                if not isinstance(_step, dict):
                    continue
                _serr = str(_step.get("error") or "")
                # ⚠ NO `final_url` FALLBACK (R4, measured against real Tor
                # and the shipped runner). A connection that fails at the
                # SOCKS layer never commits a document, so `page.url` is
                # either `chrome-error://chromewebdata/` — which parses to
                # no host — or the PREVIOUS, WORKING page, which would
                # blame a live service. And the only action that can fail
                # a navigation, `goto`, always carries its own `url`. The
                # fallback could therefore never name the dead host and
                # could name the wrong one: a mechanism with no correct
                # outcome, so it is gone rather than "improved".
                _surl = _step.get("url") or ""
                # ALIVE only from a step that genuinely NAVIGATED, i.e.
                # one carrying its own `url` (the two goto shapes). R3
                # MAJOR: the previous version marked `final_url` alive for
                # ANY successful urlless step, so a trailing `sleep` /
                # `extract_text` / `screenshot` — the commonest shape a
                # model writes — erased the strike the failing step had
                # just recorded, and the memo could never arm. Measured:
                # 25 identical failing sequences, banned=True with no
                # trailing action, banned=False with one.
                if _step.get("ok"):
                    if _step.get("url"):
                        _mark_onion_alive(_step["url"])
                elif _serr and _surl:
                    _mark_onion_dead(_surl, _serr)
        except Exception as _ie:  # noqa: BLE001
            logger.debug("interact onion-failure scan skipped: %s", _ie)
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
        # R3 MAJOR: was `url`. With an interact whose actions[0] is a
        # goto, the runner never dialled it, so a failure banned the
        # HEALTHY top-level host for 600s while the dead one went
        # unrecorded — a perfect inversion, and the direction R1 called
        # the worst outcome. `_nav_url` is empty in exactly that case,
        # and `_mark_onion_dead("")` is a no-op; the interact scan is
        # what records the real host.
        _mark_onion_dead(_nav_url, str(parsed))
        _onion_note = _dead_onion_notice(_nav_url)
        if _onion_note:
            # An onion that failed at the Tor layer needs a DIFFERENT next
            # action, not the generic browser advice below — "raise the
            # timeout / use interact" is wrong here and is what invited the
            # identical retry that produced this fix.
            return _err(f"Runner failed (exit {exit_code}): {parsed}", ran=True,
                        hint=_onion_note)
        return _err(
            f"Runner failed (exit {exit_code}): {parsed}",
            ran=True,
            hint=(
                "If this is a navigation timeout, try wait_until='domcontentloaded' "
                "or raise timeout_ms. If a CLICK timed out or its selector was "
                "not found: each atomic op reloads the page in a fresh context, "
                "so elements created by a previous click (opened windows, menus, "
                "dialogs) are GONE — run the whole flow in one context with "
                "operation='interact' and an actions list "
                "([{\"action\":\"click\",...}, ...]). If the error mentions "
                "'headless_shell not found' or 'Executable doesn't exist', the "
                "sandbox was provisioned before the Chromium pre-install was "
                "added — delete `/root/.supercharged` inside the container and "
                "retry. If the error says the op needs a URL, call "
                "`operation=\"navigate\"` once first, or pass `url=...` on "
                "this call."
            ),
        )

    # Workspace research dedup: record the URL we actually loaded so
    # a later research turn can ask "did I already pull this?" via the
    # workspace tool. Operations that meaningfully fetch a page
    # (navigate / extract_text / click / interact / screenshot) carry
    # a `parsed['url']` (or `final_url`). Non-fatal — must never break
    # a successful browser turn.
    _nav_suggestion = ""
    if workspace_model is not None and getattr(workspace_model, "enabled", False):
        try:
            _hit_url = parsed.get("url") or parsed.get("final_url")
            if _hit_url:
                workspace_model.record_research_artifact(
                    url=_hit_url, source="browser",
                    title=parsed.get("title") or parsed.get("final_title") or "",
                    note=operation,
                )
                # Repeated-navigation nudge (feature 2C): if this is the 3rd
                # visit to the same page, suggest caching / a strategy switch.
                _nav_suggestion = workspace_model.record_navigation(_hit_url) or ""
        except Exception:  # noqa: BLE001
            pass

    # Pretty-print the success result for the LLM. Keep each op's
    # return shape deterministic so downstream prompts can rely on it.
    header = f"--- BROWSER RESULT ---\nSTATUS: OK\nOP: {operation}"
    if _nav_suggestion:
        header += f"\nNOTE: {_nav_suggestion}"
    js_diag = _format_js_diagnostics(parsed)

    def _text_block(p: dict) -> str:
        """Render the runner's capped innerText preview. The runner has
        computed and shipped this since the nav-preview feature landed, but
        the formatter silently DROPPED it — so every navigate/click forced
        the follow-up extract_text (a full Chromium relaunch + re-fetch over
        Tor) that the preview exists to eliminate."""
        text = p.get("text")
        if not text:
            return ""
        trunc = " (truncated)" if p.get("truncated") else ""
        return (f"\nLENGTH: {p.get('length')}{trunc}"
                f"\n--- PAGE TEXT (capped preview) ---\n{text}")

    if operation == "navigate":
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"HTTP_STATUS: {parsed.get('status')}\n"
            f"TITLE: {parsed.get('title')}{js_diag}{_pre_interaction_line(parsed)}"
            f"{_text_block(parsed)}"
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
        # js_diag + the post-click text preview were computed by the runner
        # but dropped here (same formatter gap as navigate) — a click that
        # crashed page JS looked identical to one that worked.
        return (
            f"{header}\nURL: {parsed.get('url')}\n"
            f"TITLE: {parsed.get('title')}{js_diag}"
            f"{_text_block(parsed)}"
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
        # The header says `STATUS: OK` for every interact, whatever happened
        # inside — 5 live rows where EVERY action failed were booked a clean
        # success, and 24 more where some did. The per-action results are
        # right here; the envelope just never used them.
        _interact_status = (
            "failed" if (err_count and not ok_count)
            else "partial" if err_count else "ok")
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
                    # Same objective render check as the atomic screenshot
                    # op — an interact-captured frame is evidence too.
                    try:
                        _hp = _interact_shot_hosts.get(r.get("path", ""))
                        probe = analyze_screenshot_render(_hp) if _hp else None
                        if probe:
                            lines.append(
                                f"      RENDER_CHECK: {probe['verdict'].upper()} "
                                f"(dominant_colour={probe['dominant_pct']:.0%}, "
                                f"distinct_colours={probe['distinct_colors']}) — "
                                f"{probe['note']}"
                            )
                    except Exception:
                        pass
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
                elif act == "evaluate":
                    val = r.get("value", "")
                    trunc = " (truncated)" if r.get("truncated") else ""
                    lines.append(f"  [{idx}] {status} evaluate: len={r.get('length')}{trunc}")
                    # The value IS the point of this action — surface it in
                    # full (it is already capped runner-side), not a 500-char
                    # teaser like extract_text's preview.
                    lines.append(f"      VALUE: {val}")
                else:
                    lines.append(f"  [{idx}] {status} {act}")
            else:
                lines.append(
                    f"  [{idx}] {status} {act}: {r.get('error')}"
                )
        _txt = "\n".join(lines)
        if _interact_status == "ok":
            return _txt
        # The runner already navigated, clicked and filled, so this DID
        # change the world even when every action errored.
        return (ToolOutcome.failed if _interact_status == "failed"
                else ToolOutcome.partial)(
            _txt, world_changed=True,
            reason_code=f"browser_interact_{_interact_status}")
    # Defensive default — never hit because we validated above.
    return f"{header}\nRAW: {json.dumps(parsed)}"
