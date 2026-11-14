"""§4OE-4 — a request remembers what it fetched.

15 % of a research turn's page fetches repeated a URL already read in the
same request, and the only "already pulled" ledger lived behind a tool the
model had to call (§4OD). The memo answers a repeat fetch at dispatch time
with the head of the earlier result, so no Tor round trip, no strike, and
the model sees that it is repeating itself.

§4OG-3 (c): the key was (tool, url) and the body a 2,000-char head — so the
taught navigate → extract_text path got the truncated preview back with
"nothing has changed", and three 22–45 KB pages were never read past 2 KB
(§4OF). Now: the key carries the OPERATION and the extract's selector /
max_chars; a TRUNCATED result is never memoised (the next, fuller read runs);
a DECLARED FAILURE is remembered with its status, so a dead URL is not
re-fetched (4 of 4 probes re-fetched one).

Local pages (2026-11-07, req 298833ef): a ``file://`` or loopback page is the
agent's OWN work. The memo answered a re-render of ``index.html`` after an
edit of ``app.js`` with the render from BEFORE the edit ("nothing has
changed"). :func:`forget_local` drops every local entry; the dispatcher calls
it after any call that may have written the workspace or changed a page's
state (`core/request_effects.py`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

FETCH_TOOLS = frozenset({"browser", "web_search", "deep_research", "darkweb_search"})
_BROWSER_NAV_OPS = frozenset({"navigate", "goto"})
_BROWSER_EXTRACT_OPS = frozenset({"extract_text", "read", "fetch"})
MEMO_HEAD_CHARS = 2000
MEMO_MAX = 200
_TRUNCATED_RE = re.compile(r"LENGTH:\s*\d+\s*\(truncated\)|\[\.\.\. (?:range |output )?capped|\(truncated\)", re.I)
_FAILED = "FAILED"
_RETRY_ONCE = "RETRY_ONCE"
#: §4UN-web: a SEARCH that failed because the engines refused or could not be
#: reached says nothing about the query — and its own message tells the model
#: "Retry the SAME query once". The memo refused exactly that retry ("ALREADY
#: FAILED … do not fetch it again"). Such a failure is filed under a soft key
#: first; only a SECOND one is the declared failure the memo refuses.
_TRANSIENT_SEARCH_MARKERS = ("failed on TRANSPORT", "Tor proxy is not reachable")

Key = Tuple[str, str, str, str]


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def fetch_key(cname: str, args: Dict[str, Any]) -> Optional[Key]:
    """The memo key of a read-only fetch — (tool, url-or-query, op, extra) —
    or None for anything else. A navigate and an extract_text of one URL are
    DIFFERENT keys (the extract is the fuller read the schema teaches)."""
    if not isinstance(args, dict) or cname not in FETCH_TOOLS:
        return None
    if cname == "browser":
        op = _norm(args.get("operation") or args.get("action") or args.get("op") or "")
        url = _norm(args.get("url") or args.get("link") or args.get("href"))   # the aliases the tool heals
        if not url or args.get("actions"):
            return None
        if op in _BROWSER_NAV_OPS:
            return ("browser", url.rstrip("/"), "navigate", _norm(args.get("nav_text_chars") or ""))
        if op in _BROWSER_EXTRACT_OPS:
            extra = _norm(args.get("selector") or "") + "|" + _norm(args.get("max_chars") or "")
            return ("browser", url.rstrip("/"), "extract", extra)
        return None
    q = _norm(args.get("query") or args.get("q") or args.get("url"))
    return (cname, q, "", "") if q else None


def failed_key(key: Optional[Key]) -> Optional[Key]:
    """The key a declared failure is filed under: the URL alone — a 404 is a
    404 whatever operation asked for it."""
    if not key:
        return None
    return (key[0], key[1], _FAILED, "")


def is_truncated_result(text: Any) -> bool:
    """The LENGTH line sits in the result HEADER — after the title, the JS
    diagnostics and the image lines, before the page text (reader 2 M3)."""
    t = str(text or "")
    head = re.split(r"\n--- (?:PAGE )?TEXT", t, maxsplit=1)[0][:4000]
    return bool(_TRUNCATED_RE.search(head))


def retry_key(key: Optional[Key]) -> Optional[Key]:
    """§4UN-web: the key a first transient search failure is filed under."""
    if not key:
        return None
    return (key[0], key[1], _RETRY_ONCE, "")


def transient_search_failure(key: Optional[Key], text: Any) -> bool:
    """A search (not a page) that failed on transport or a refusal — the same
    query on fresh circuits can succeed."""
    if not key or key[0] == "browser":
        return False
    t = str(text or "")
    return any(m in t for m in _TRANSIENT_SEARCH_MARKERS)


def remember(memo: Dict[Key, str], key: Optional[Key], text: Any) -> bool:
    """Memoise a SUCCESSFUL, UNTRUNCATED result. Returns True when stored."""
    if not key or memo is None:
        return False
    if is_truncated_result(text):
        return False                            # the next, fuller read must run
    body = " ".join(str(text or "").split())
    if not body:
        return False
    _put(memo, key, body[:MEMO_HEAD_CHARS])
    memo.pop(failed_key(key), None)             # a later success clears an earlier failure
    memo.pop(retry_key(key), None)
    return True


#: §4PC: failures that are NOT about the page — a missed selector, the call's
#: own budget, a Chromium launch race, the sandbox daemon — must not file the
#: URL as "already failed, do not fetch again" (live: the tool's own hint said
#: "read the page first without a selector" and the memo then refused that read).
_NOT_PAGE_LEVEL = ("did not match any element", "retry as a fresh call", "TargetClosedError",
                   "SANDBOX INFRA ERROR", "sandbox that runs the browser is unavailable",
                   "exceeded its", "never became clickable", "not found within", "waiting for locator")


def page_level_failure(text: Any) -> bool:
    """True when the failure says something about the PAGE (404, challenge,
    timeout, dead host), not about this call's arguments or the runner."""
    t = str(text or "")
    return not any(m in t for m in _NOT_PAGE_LEVEL)


def remember_failure(memo: Dict[Key, str], key: Optional[Key], text: Any) -> bool:
    """Memoise a DECLARED failure (a 404, a bot challenge, a dead host) by
    its URL, with its status line. Returns True when stored."""
    fk = failed_key(key)
    if not fk or memo is None or not page_level_failure(text):
        return False
    flat = " ".join(str(text or "").split())
    if transient_search_failure(key, text) and retry_key(key) not in memo:
        _put(memo, retry_key(key), flat[:200])  # §4UN-web: the one retry the result asks for runs
        return True
    m = re.search(r"STATUS:\s*([A-Z]+[^-]{0,120})", flat)
    status = (m.group(1).strip() if m else flat[:160]) or "failed"
    _put(memo, fk, status[:200])
    return True


#: hosts whose pages are served from this machine or its sandbox
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "host.docker.internal"})
_URL_HOST_RE = re.compile(r"^[a-z][a-z0-9+.-]*://(?:[^@/?#]*@)?(\[[^\]]*\]|[^:/?#]*)")


def is_local_url(url: Any) -> bool:
    """A page the workspace itself serves: ``file:`` URLs, loopback hosts, a
    bare workspace path. Case and a trailing dot on the host do not matter.
    A host is local when it is named so (localhost, *.localhost) or is a
    loopback / unspecified IP in any spelling (review r1 m3: ``[0:…:1]``,
    ``0``; ``127.attacker.com`` is NOT local)."""
    u = _norm(url)
    if not u:
        return False
    if u.startswith("file:") or u.startswith("./") or (u.startswith("/") and not u.startswith("//")):
        return True
    m = _URL_HOST_RE.match(u)
    if not m:
        return False
    host = m.group(1).rstrip(".").strip("[]")
    if host in _LOCAL_HOSTS or host.endswith(".localhost"):
        return True
    import ipaddress
    try:
        ip = ipaddress.ip_address(int(host) if host.isdigit() else host)
    except ValueError:
        # review r2 m3: a browser also reads the short and hex IPv4 forms
        # (127.1, 0x7f000001, 0177.0.0.1) — inet_aton parses them the same way
        if not re.fullmatch(r"[0-9a-fx.]+", host):
            return False
        import socket
        try:
            ip = ipaddress.ip_address(socket.inet_aton(host))
        except (OSError, ValueError):
            return False
    return ip.is_loopback or ip.is_unspecified


def forget_local(memo: Optional[Dict[Key, str]]) -> int:
    """Drop every remembered result (and remembered failure) of a local page.
    Returns how many entries went."""
    if not isinstance(memo, dict):
        return 0
    gone = [k for k in list(memo) if isinstance(k, tuple) and len(k) >= 2
            and k[0] == "browser" and is_local_url(k[1])]
    for k in gone:
        memo.pop(k, None)
    return len(gone)


def _put(memo: Dict[Key, str], key: Key, value: str) -> None:
    if len(memo) >= MEMO_MAX and key not in memo:
        for old in list(memo)[: MEMO_MAX // 4]:
            memo.pop(old, None)
    memo[key] = value


def repeat_result(memo: Dict[Key, str], key: Optional[Key]) -> Optional[str]:
    """The synthetic result for a repeat, or None when the fetch is new."""
    if not key or memo is None:
        return None
    fk = failed_key(key)
    if fk in memo:
        what = "URL" if key[0] == "browser" else "search"
        head = ("--- BROWSER RESULT ---\nSTATUS: BLOCKED (already failed this request)\n" if key[0] == "browser"
                else "STATUS: ERROR (already failed this request)\n")        # reader 2 M4: a declared failure, so the URL vouches for nothing
        return (head + f"ALREADY FAILED this request: this exact {what} ({key[1][:120]}) was tried earlier and the "
                f"site answered: {memo[fk]}. It will answer the same way — do not fetch it again; use "
                f"another source (a different page or a search).")
    if key not in memo:
        return None
    what = "page" if key[0] == "browser" else "search"
    how = f" with the same {key[2]}" if key[2] else ""
    return (f"ALREADY READ this request: this exact {what} ({key[1][:120]}){how} was fetched earlier and "
            f"nothing has changed. Here is the head of that result again — work from it or from "
            f"the pages you have, do not fetch it a third time.\n\n{memo[key]}")
