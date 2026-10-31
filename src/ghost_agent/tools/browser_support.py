"""§4LN helpers for the browser tool — split out of `browser.py` so it stays
under its post-extraction size gate. Every name is re-exported there."""
import asyncio
import json
import re
import shlex
import time
from typing import Dict
from pathlib import Path

_BROWSER_PROFILE_DIR = ".browser_profile"     # the runner's profile dir (browser.py imports it)


#: screenshot paths this process wrote (overwriting those is the point)
_BROWSER_WRITTEN: set = set()
_PROTECTED_IMAGE_NAME_RE = re.compile(r"^(?:gen_[0-9a-f]{8}|vision_[0-9a-f]{12})\.[A-Za-z]{3,4}$")


#: §4SH: interact sub-actions that are not operations of their own — named as
#: the operation, the call runs as interact (click/extract_text/screenshot ARE ops)
_INTERACT_ONLY_ACTIONS = frozenset({"evaluate", "fill", "dblclick", "wait_for_selector", "sleep", "goto"})


def _as_interact_actions(operation, actions, kwargs):
    """The actions an interact-only action named as the operation runs with
    (the one rule `run_action_as_interact` and `browser_effective_call` share)."""
    _js = kwargs.get("js") or kwargs.get("expression") or kwargs.get("script")
    if not actions and operation == "evaluate" and _js:
        actions = [{"action": "evaluate", "js": _js}]
    return actions if isinstance(actions, list) and actions else None


def browser_effective_call(args) -> tuple:
    """(operation, actions) as the browser tool RUNS a recorded call. r1
    (§4SH review): `evaluate`/`fill`/`dblclick` sent as the operation run as
    interact, but every check reading the call saw the model's name — a page
    exercised that way stayed "never tested"."""
    a = args if isinstance(args, dict) else {}
    # r2: the tool's own aliases (browser.py's "hallucination healing")
    op = str(a.get("operation") or a.get("op") or a.get("action") or "").strip().lower()
    acts = a.get("actions") or a.get("steps") or a.get("sequence")
    if op in _INTERACT_ONLY_ACTIONS:
        ran = _as_interact_actions(op, acts, a)
        if ran:
            return "interact", [x for x in ran if isinstance(x, dict)]
    return op, [x for x in (acts if isinstance(acts, list) else []) if isinstance(x, dict)]


async def run_action_as_interact(tool_fn, call: dict, reject):
    """§4SH (req cb27058e): `operation='evaluate'` with an `actions` list was
    refused as unknown and booked a strike — the model meant interact. The
    same call (`call` = the tool's locals at the check) runs as interact and
    says the right form; with no actions and nothing to build one from, the
    refusal shows that form."""
    from .outcome import append_note
    operation, actions, kwargs = call["operation"], call.get("actions"), dict(call.get("kwargs") or {})
    actions = _as_interact_actions(operation, actions, kwargs)
    form = f"{{\"operation\": \"interact\", \"url\": \"…\", \"actions\": [{{\"action\": \"{operation}\", …}}]}}"
    if not actions:
        return reject(f"'{operation}' is an ACTION inside interact, not an operation of its own. Send {form}.")
    co = tool_fn.__code__
    named = {k: v for k, v in call.items() if k in co.co_varnames[:co.co_argcount]}
    rest = {k: v for k, v in kwargs.items()
            if k not in ("op", "action", "js", "expression", "script", "steps", "sequence")}
    res = await tool_fn(**{**named, "operation": "interact", "actions": actions}, **rest)
    return append_note(res, f"\nNOTE: '{operation}' is an ACTION inside interact, not an operation — this call "
                            f"ran as operation='interact' with your actions. Next time send {form}.")


def _overwrite_refusal(host_path) -> str:
    """Why a screenshot must NOT be written over this existing file, or "".
    A screenshot is a PNG; an existing file that is not one (a user's
    photo.jpg, a document) cannot be an earlier screenshot, and a generated
    image or a pasted attachment is never one — `out_path="photo.jpg"`
    silently replaced the user's photo (§4LN)."""
    try:
        p = Path(host_path)
        if not p.exists() or str(p) in _BROWSER_WRITTEN:
            return ""
        with open(p, "rb") as fh:
            head = fh.read(8)
        if head == b"\x89PNG\r\n\x1a\n" and not _PROTECTED_IMAGE_NAME_RE.match(p.name):
            return ""
    except OSError:
        return ""
    return (f"'{Path(host_path).name}' already exists and is not a browser screenshot — "
            "a screenshot will not overwrite it. Pick a new out_path (e.g. shot_1.png).")

async def _exec_holding_lock(fn, *args, **kwargs):
    """Run the (blocking) runner exec in a thread — and when the CALLER is
    cancelled, keep waiting for that thread before letting the lock go.
    A hard cancel freed `_BROWSER_PROFILE_LOCK` at once while Chromium ran
    on in the container for up to the exec timeout, and the next call
    launched a second Chromium on the same profile — the crash this lock
    exists to prevent (§4LN). Call it INSIDE `async with` the lock.
    §4RZ step 5: while it runs, an owner tool stop kills THIS runner — the
    lock is held, so the only runner in the container is ours."""
    box: dict = {}
    _unregister = _register_runner_kill(fn, box)
    from ..utils import tool_stop as _ts
    if _ts.requested():                          # §4RZ step 5: never start a runner after a stop
        _unregister()
        raise asyncio.CancelledError()
    fut = box["fut"] = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        while not fut.done():                    # a second cancel must not free it either
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                continue
            except BaseException:  # noqa: BLE001 — the cancel is what we re-raise
                break
        raise
    finally:
        _unregister()


#: the runner's exact argv head (`runner_cmd`), so nothing else matches (s5-r8:
#: a substring grep also TERMed `pytest tests/test_browser_runner.py`, and its
#: "k" booked a runner that had not started yet as stopped)
_RUNNER_ARGV_HEAD = "python3 -u .browser_runner.py "
_RUNNER_KILL_CMD = ("sh -c 'for p in /proc/[0-9]*; do a=$(tr \"\\000\" \" \" < $p/cmdline 2>/dev/null); "
                    "case \"$a\" in \"" + _RUNNER_ARGV_HEAD + "\"*) kill -TERM ${p#/proc/} && echo k;; esac; "
                    "done; true'")


def _register_runner_kill(fn, box: dict):
    """The kill for an owner tool stop, when one is bound and ``fn`` is a
    sandbox manager's bound `execute`; armed until the call is over.
    Returns its removal. ``box["fut"]``: the runner's thread, once started —
    a runner just launched may not exist yet, so the kill retries until it
    hits one or the runner returns."""
    try:
        from ..utils import tool_stop
        stop = tool_stop.current()
        mgr = getattr(fn, "__self__", None)
        if stop is None or stop.requested() or mgr is None or not hasattr(mgr, "execute"):
            return lambda: None         # stopped already: the caller never starts a runner
    except Exception:  # noqa: BLE001
        return lambda: None

    def _kill():
        import time as _time
        for _ in range(_KILL_TRIES):
            fut = box.get("fut")
            if fut is not None and fut.done():
                return
            out, _code = mgr.execute(_RUNNER_KILL_CMD, timeout=15, quiet=True)
            if "k" in str(out or "").split():
                stop.note_killed(key)
                return
            _time.sleep(_KILL_RETRY_S)
    key = tool_stop.call_key()
    stop.arm(key)
    remove = stop.on_stop(_kill)

    def _unregister():
        remove()
        stop.disarm(key)
    return _unregister


_KILL_TRIES, _KILL_RETRY_S = 5, 1.0

# Keep outputs reasonable — a single page's HTML can be 5+ MB and would
# blow the LLM context window, so we cap before returning. These caps
# match `helper_fetch_url_content`'s 5 MB ceiling.

#: §4RX (operator: "these are not mechanical errors, they shouldn't look like
#: they are in the agent log"): a browser failure the SITE or PAGE decided —
#: a slow page over Tor, a host that does not exist, a selector the page does
#: not have. Ordered: the first match names it. Anything unmatched (a runner
#: crash, a kill at the wall clock, a broken install, the sandbox down) stays a
#: warning, because something on OUR side broke.
_PAGE_OUTCOMES = (
    (r"ERR_SOCKS_CONNECTION_FAILED", "Tor could not reach the host"),
    (r"exceeded its \d+ ms budget|ran out of its \d+ ms total budget", "the page kept the browser past its budget"),
    (r"Page\.goto: Timeout|ERR_TIMED_OUT|navigation timeout", "the page did not load in time"),
    (r"did not match any element|not found within|never became clickable|waiting for locator",
     "the page has no such element"),
    (r"ERR_NAME_NOT_RESOLVED", "the host name does not resolve"),
    (r"ERR_CONNECTION_(?:REFUSED|CLOSED|RESET)|ERR_EMPTY_RESPONSE", "the site dropped the connection"),
    (r"ERR_HTTP2|ERR_SSL_PROTOCOL_ERROR|ERR_CERT", "the site's HTTPS failed"),
    (r"ERR_ABORTED", "the address is a download, not a page"),
    (r"ERR_BLOCKED_BY_CLIENT|redirected to a non-public address", "policy refused the address"),
    (r"ERR_FILE_NOT_FOUND", "no such file"),
    (r"Execution context was destroyed", "the page navigated away mid-read"),
    (r"STATUS: PARTIAL", "some actions failed on the page"),
    (r"STATUS: BLOCKED", "the site refused the page or has no such page"),
)
_MECHANICAL = re.compile(r"\(exit (?:124|137|143)\)|runner exit (?:124|137|143)|SANDBOX INFRA ERROR|headless_shell|"
                         r"Executable doesn't exist|Could not write browser runner|sandbox execute failed|"
                         r"TargetClosedError|Target (?:page, context or browser has been )?closed|Target crashed",
                         re.I)
_LOOPBACK = re.compile(r"\b(?:127\.0\.0\.1|localhost|\[::1\])\b", re.I)


def browser_outcome_note(text: str) -> str:
    """What the page or site did, when a browser failure is THAT and not a
    fault of ours — "" when something mechanical broke (§4RX: those log as
    INFO, these stay warnings)."""
    # page CONTENT must not pick the class (review r2): the text blocks of a
    # navigate/extract and an interact's TEXT:/VALUE: lines go; its ERR lines stay
    t = re.split(r"--- PAGE TEXT|--- TEXT ---", str(text or ""), maxsplit=1)[0]
    t = re.sub(r"(?m)^\s+(?:TEXT|VALUE): .*$", "", t)
    if not t or _MECHANICAL.search(t):
        return ""
    for pat, note in _PAGE_OUTCOMES:
        if re.search(pat, t, re.I):
            if "Tor could not" in note:
                # §4RX review: during a Tor outage EVERY host fails this way —
                # only a recent clean load makes it the host's doing
                from .host_memo import tor_recently_ok
                return note if tor_recently_ok() else ""
            if "dropped the connection" in note and _LOOPBACK.search(t):
                return ""                                    # the user's own app is down: act on it
            return note
    return ""


#: §4RX: what an exec must keep past the runner's clock — the runner's own
#: 5 s grace plus Chromium's launch before the clock and close after it.
#: Under the 30 s slack, so a normal call keeps its full total (review r2: a
#: 30 s + 10 s cut made a 1-action interact's clock shorter than before §4RX
#: and gave a call queued to 15–41 s a 1 s clock).
_EXEC_MARGIN_S = 15


def runner_cmd(filename, payload, operation, exec_s, runner_total_ms) -> str:
    """The runner command for ONE exec of `exec_s` seconds (§4RX). An
    interact's whole-sequence clock is sized to THIS exec's window, so a
    retry or a queued call with less time left never outlives its exec
    (review: the launch-race retry reused the first exec's total). The
    per-action share never exceeds that total."""
    if operation == "interact":
        total = max(1000, min(int(runner_total_ms), (int(exec_s) - _EXEC_MARGIN_S) * 1000))
        payload["total_timeout_ms"] = total
        payload["timeout_ms"] = min(int(payload["timeout_ms"]), total)
    return f"python3 -u {filename} {shlex.quote(json.dumps(payload))}"


def _log_runner_failure(pretty_log, operation, exit_code, parsed, cause, host="") -> None:
    """The operator's stream line for a failed runner (§4RX): what the page
    or site did (slow over Tor, no such host, no such element) is
    information, not a fault — only a mechanical failure keeps the warning.
    `pretty_log` is the caller's, so the browser module stays the one seam."""
    from ..utils.logging import Icons
    note = browser_outcome_note(f"(exit {exit_code}) {parsed}")
    if note:
        pretty_log("Browser Note", f"{operation}{' ' + host if host else ''}: {note} — {cause}",
                   icon=Icons.TOOL_BROWSER, level="INFO")
    else:
        pretty_log("Browser Failed", f"{operation}: runner exit {exit_code} — {cause}",
                   icon=Icons.TOOL_BROWSER, level="WARNING")


def budget_used_up_line(parsed: dict) -> str:
    """§4RX: an interact whose total budget ran out mid-sequence — what ran
    is real, what did not run is named."""
    nr = int(parsed.get("not_run") or 0)
    return (f"⚠ BUDGET USED UP: the sequence ran out of its {parsed.get('budget_ms')} ms total "
            "budget during the last step listed below"
            + (f"; the {nr} action{'s' if nr != 1 else ''} after it did NOT run" if nr else "")
            + ". The steps above it ran and their results are real. Re-run only the missing "
            "steps, or pass a larger timeout_ms (it is per action) if one step needs longer.")


def _runner_failure_hint(err: str) -> str:
    """The advice for THIS failure (§4LN): every runner error used to carry
    one paragraph naming every failure mode — Tor timeouts, click chains and
    a stale `/root/.supercharged` repair — so a missing `h1` was answered
    with Chromium-install instructions (live probe B5)."""
    e = str(err or "")
    parts = []
    if "did not match any element" in e:
        parts.append("The page has no element matching that selector. Read the page first "
                     "(extract_text with no selector) and pick a selector from what is there; "
                     "do not try more guessed selectors.")
    if re.search(r"Page\.goto: Timeout|ERR_TIMED_OUT|navigation timeout|Timeout \d+ms exceeded navigating", e, re.I):
        parts.append("The page did not load over Tor — another attempt usually times out the "
                     "same way; use a different source.")
    if re.search(r"not found within|never became clickable|waiting for locator", e, re.I):
        parts.append("Each atomic op reloads the page in a fresh context, so elements created by "
                     "a previous click (opened windows, menus, dialogs) are GONE — run the whole "
                     "flow in one context with operation='interact' and an actions list.")
    if "headless_shell" in e or "Executable doesn't exist" in e:
        parts.append("The sandbox's browser install is broken — a retry fails the same way. "
                     "Tell the user the browser is unavailable.")
    if "needs a URL" in e:
        parts.append("Call operation=\"navigate\" once first, or pass url=... on this call.")
    # §4PC: the common net:: errors had no route and got the generic line
    if "net::ERR_ABORTED" in e:
        parts.append("The address is a download (a PDF or other file), not a page — fetch it with "
                     "file_system(operation='download') and read it with knowledge_base, instead of the browser.")
    if "net::ERR_FILE_NOT_FOUND" in e:
        parts.append("That file path does not exist in the sandbox — list the directory and use a real path.")
    if "net::ERR_BLOCKED_BY_CLIENT" in e or "redirected to a non-public address" in e:
        parts.append("Policy refused that address (an internal, loopback or metadata host, possibly reached "
                     "by a redirect) — it cannot be read from here; use a different, public source.")
    if "net::ERR_SOCKS_CONNECTION_FAILED" in e:            # §4RX: no route; 37 of 80 failures
        parts.append("Tor could not connect to that host — over Tor this is how an address that does "
                     "not exist fails (a guessed or misspelled domain), or a site that refuses Tor. Take "
                     "the address from your search results; do not try spelling variants of it. If other "
                     "sites load, the address is the problem; if EVERY site fails this way, Tor is down — "
                     "say so instead of retrying.")
    if "net::ERR_NAME_NOT_RESOLVED" in e:
        parts.append("That host name does not resolve — check the address against your search results; "
                     "do not try spelling variants.")
    if "net::ERR_CONNECTION_REFUSED" in e:
        parts.append("Nothing is listening at that address — if it is your own app, it is down (check "
                     "manage_services); if it is a public site, use another source.")
    if "net::ERR_CERT" in e or "SSL" in e and "certificate" in e.lower():
        parts.append("The site's certificate failed — it cannot be read over this transport; use another source.")
    if "exceeded its" in e and "budget" in e:
        parts.append("The page kept the browser busy past its budget — a retry of the same page will too; "
                     "use a different source or a narrower selector.")
    if "SANDBOX INFRA ERROR" in e:
        parts.append("The sandbox that runs the browser is unavailable (infrastructure, not this call) — "
                     "do not call the browser again this turn; answer from search results or tell the user.")
    return " ".join(parts) or "Read the error above — an identical retry usually fails the same way."


def _last_url_filename() -> str:
    """The sidecar this REQUEST's url-less ops read (§4LN): `.last_url.<req>`,
    or the shared `.last_url` outside a request (background flows)."""
    try:
        from ..utils.logging import request_id_context
        rid = re.sub(r"[^A-Za-z0-9_-]", "", str(request_id_context.get() or ""))[:24]
    except Exception:  # noqa: BLE001
        rid = ""
    return f".last_url.{rid}" if rid and rid != "SYSTEM" else ".last_url"

#: §4LN: three live replies said "the screenshot shows …" with no vision call
#: in the turn; one was contradicted by the user minutes later.
_UNSEEN_SCREENSHOT_NOTE = (
    "\nNOTE: you have NOT seen this image. Before describing what it shows, call "
    "vision_analysis on it (or read the page text with extract_text).\n"
    "A NEW screenshot is a working file: at the end of this turn it is removed from the workspace "
    "unless your reply links it (e.g. ![page](/api/download/<path>)) or the user asked for it.")


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
        for name in (_last_url_filename(), _last_url_conv_filename()):      # §4PC: mirrors the runner
            if not name:
                continue
            try:
                with open(root / _BROWSER_PROFILE_DIR / name, "r", encoding="utf-8") as fh:
                    v = fh.read().strip()
                if v:
                    return v
            except Exception:  # noqa: BLE001
                continue
        return ""
    except Exception:  # noqa: BLE001
        return ""


def click_miss_summary(action_results) -> list:
    """§4NW (req 92dfdd7d): one up-front line when an interact's clicks did
    not reach their target — the per-action lines read "OK" (the click ran),
    so twelve such calls read as success. STATUS is left as it is: a PARTIAL
    there cost strikes and could end a working turn as failed (review)."""
    dead = [r for r in (action_results or []) if r.get("ok") and r.get("no_effect")]
    if not dead:
        return []
    return [f"⚠ {len(dead)} click(s) did not do what was intended "
            f"({', '.join(repr(r.get('selector')) for r in dead[:3])}) — what you meant to "
            "open is NOT open; a screenshot after them does not show it. Reason per click below."]


def no_effect_line(r) -> str:
    """The per-action warning under a click that missed (§4NW), or the note
    under a single click on a double-click-only element, or ""."""
    if r.get("no_effect"):
        return f"\n      ⚠ NO EFFECT: {r['no_effect']}"
    return f"\n      NOTE: {r['hint']}" if r.get("hint") else ""


def _note_shot(host_path) -> None:
    """A screenshot this call writes: overwritable by later screenshots, and
    — when it is NEW — noted so the end of the turn moves it to the host trash
    unless the reply or the user names it (§4NY)."""
    from .turn_scratch import note_created
    note_created(host_path, "screenshot")
    _BROWSER_WRITTEN.add(str(host_path))


def interact_notes(action_results) -> list:
    """The lines above an interact's per-action results: clicks that missed
    (§4NW) and, when it took screenshots, that they are working files (§4NY)."""
    out = click_miss_summary(action_results)
    if any(r.get("ok") and r.get("action") == "screenshot" for r in (action_results or [])):
        out.append("NOTE: new screenshots are working files — at the end of this turn they are removed from the "
                   "workspace unless your reply links them (![x](/api/download/<path>)) or the user asked for them.")
    return out


def ensure_runner_file(path, text: str) -> bool:
    """§4OE-5: write the browser runner only when the file's text differs
    (atomic replace). Returns True when it wrote. Called under the browser
    profile lock — written before it, a batch-mate's rewrite raced the exec
    through the bind mount (`can't open file .browser_runner.py`)."""
    import os as _os
    try:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return False
    except Exception:  # noqa: BLE001 — unreadable: rewrite it
        pass
    tmp = path.with_name(f"{path.name}.{_os.getpid()}.{id(path)}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _os.replace(tmp, path)
    return True


#: §4OG-5 (f): under a client deadline a 65,536-char preview per page is prefill
#: the clock cannot afford (a 240 s request spent 52 % of its wall prefilling
#: 46k tokens of previews, §4OF) — the preview is capped at a quarter of the max
DEADLINE_PREVIEW_CAP = 16 * 1024


def preview_cap_chars(max_chars: int = 64 * 1024) -> int:
    """The nav preview cap: `max_chars`, or DEADLINE_PREVIEW_CAP when the
    request carries a client deadline."""
    try:
        from ..utils.logging import client_deadline_context
        if float(client_deadline_context.get() or 0.0) > 0:
            return min(int(max_chars), DEADLINE_PREVIEW_CAP)
    except Exception:  # noqa: BLE001
        pass
    return int(max_chars)


#: §4PC: request id → monotonic time the sandbox daemon was seen down
_SANDBOX_DOWN: Dict[str, float] = {}
_SANDBOX_DOWN_TTL_S = 300.0


def _sandbox_down_key() -> str:
    try:
        from ..utils.logging import request_id_context
        return str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def _note_sandbox_down() -> None:
    k = _sandbox_down_key()
    if k:
        _SANDBOX_DOWN[k] = time.monotonic()
        for old in [r for r, t in _SANDBOX_DOWN.items() if time.monotonic() - t > _SANDBOX_DOWN_TTL_S]:
            _SANDBOX_DOWN.pop(old, None)


def _sandbox_down_recently() -> bool:
    k = _sandbox_down_key()
    t = _SANDBOX_DOWN.get(k)
    return bool(k and t is not None and (time.monotonic() - t) < _SANDBOX_DOWN_TTL_S)


def _sandbox_down_result():
    """A DESIGNED stop, not a strike: nothing ran, nothing changed, and a
    retry — of this page or any other — fails the same way."""
    return _ToolOutcome().rejected(
        "--- BROWSER RESULT ---\nSTATUS: ERROR\nThe sandbox that runs the browser is unavailable right now "
        "(infrastructure, not this call). Do NOT call the browser again this turn — any page fails the "
        "same way. Answer from your search results, or tell the user the browser is down.",
        world_changed=False, reason_code="sandbox_unavailable")


def _ToolOutcome():
    from .outcome import ToolOutcome
    return ToolOutcome


def _links_block(p: dict) -> str:
    """§4PC: the page's links, so the model FOLLOWS one instead of composing
    an address (88 % of 404s were composed paths)."""
    links = [u for u in (p.get("links") or []) if isinstance(u, str) and u]
    if not links:
        return ""
    return (f"\nLINKS ON THIS PAGE ({len(links)}; navigate to one of these, never to a composed path):\n"
            + "\n".join(links))


def _last_url_conv_filename() -> str:
    """§4PC: the CONVERSATION's sidecar (`.last_url.c<key>`): a url-less op
    in a follow-up turn of the same conversation re-opens its page; another
    conversation's page is never re-opened. "" outside a conversation."""
    try:
        from ..utils.logging import conversation_key_context
        key = re.sub(r"[^A-Za-z0-9_-]", "", str(conversation_key_context.get() or ""))[:24]
    except Exception:  # noqa: BLE001
        key = ""
    return f".last_url.c{key}" if key else ""


def _new_nonce() -> str:
    """A fresh per-call nonce for the runner's sentinel (§4PC). Live calls
    always get one; the test suite's legacy runner fakes run with "" (see
    tests/conftest.py), which keeps the un-nonced frame acceptable."""
    import secrets as _secrets
    return _secrets.token_hex(12)


def _parse_runner_output(stdout: str, nonce: str = "") -> tuple[bool, object]:
    """Pick the [BROWSER_OK]/[BROWSER_ERR] sentinel line out of stdout.

    Chromium / Playwright print warnings ("libpci.so.3 not found", etc.)
    that interleave with the runner's output. We scan for the last
    sentinel line so stray noise doesn't corrupt the result.

    §4PC: with a `nonce` (every live call has one) ONLY `[BROWSER_OK:<nonce>] `
    / `[BROWSER_ERR:<nonce>] ` lines count — a page planted
    `\n[BROWSER_OK] {...}` inside an exception message and the parser
    returned the forged result as a success (verified with the real runner).
    """
    ok_line = None
    err_line = None
    ok_tag = f"[BROWSER_OK:{nonce}] " if nonce else "[BROWSER_OK] "
    err_tag = f"[BROWSER_ERR:{nonce}] " if nonce else "[BROWSER_ERR] "
    # "\n" ONLY (§4LN): the payload is unescaped UTF-8, and splitlines()
    # also breaks on U+2028 / U+0085 / \x1c… that page text can contain.
    for line in stdout.split("\n"):
        line = line.rstrip("\r")
        if line.startswith(ok_tag):
            ok_line = line[len(ok_tag):]
        elif line.startswith(err_tag):
            err_line = line[len(err_tag):]
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


def text_block(p: dict) -> str:
    """Render the runner's capped innerText preview. The runner has
    computed and shipped this since the nav-preview feature landed, but
    the formatter silently DROPPED it — so every navigate/click forced
    the follow-up extract_text (a full Chromium relaunch + re-fetch over
    Tor) that the preview exists to eliminate."""
    text = p.get("text")
    if not text:
        return ""
    trunc = " (truncated)" if p.get("truncated") else ""
    from .browser import webgl_denial_note          # the tool module is loaded by the time this renders
    return (f"\nLENGTH: {p.get('length')}{trunc}"
            f"\n--- PAGE TEXT (capped preview) ---\n{text}"
            f"{webgl_denial_note(str(text))}")
