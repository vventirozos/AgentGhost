"""§4VG: which outside hosts THIS request reached, and which it only failed to reach.

Req 9d5bbe77 (2026-11-22, "add DOOM as a pre-installed app"): the model wrote
`const DOOM_URL = "https://doom.30th.com/"` into a web app and called it "id
Software's official browser release". Every attempt the request made to reach
that host had failed (two downloads, TLS "unrecognized name"); none had
succeeded. The host is a parked domain. The page check passed (the page itself
loaded), the verifier passed (the files exist), and the owner got an app that
can only ever show a parking page.

The ledger here records, per request, each clearnet host a tool tried and
whether ANY attempt got an answer from it (an HTTP status of any kind counts:
the server exists). `note_for_written` is the advisory that rides on a code
write naming a host that this request tried and never reached. It never blocks
the write: a host can be fine for the user and unreachable from the sandbox
(Tor-only egress), and the note says so.

`GHOST_REACH_NOTE=0` turns the note off (the ledger still records).
"""
from __future__ import annotations

import os
import posixpath
import re
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

#: request id → {"t": first booking (monotonic), "hosts": {host: [reached: bool, failed attempts: int,
#: last failure cause: str, a url as written]}, "noted": {(host, path)}}
_LEDGER: "OrderedDict[str, dict]" = OrderedDict()
_MAX_REQUESTS = 64
_MAX_HOSTS = 256
#: review r1 m7: a scheduled task's request id (`sched-<job>`) is the SAME on every run — an entry older than
#: this is a previous run's, and is started over
_REQUEST_TTL_S = 3 * 3600.0

#: an http(s) URL inside code text, or a protocol-relative one right after a quote / bracket / `=`; quotes,
#: brackets and template braces end it
_URL_RE = re.compile(r"""(?:https?:|(?<=["'`(=]))//[^\s"'`<>()\[\]{}\\,;/][^\s"'`<>()\[\]{}\\,;]*""", re.I)

#: files whose text is code a user will run or load — a URL in a research note is not a dependency
CODE_EXTS = (".html", ".htm", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte", ".py", ".css")


def enabled() -> bool:
    return os.environ.get("GHOST_REACH_NOTE", "1").strip().lower() not in ("0", "off", "false", "no")


def _rid() -> str:
    try:
        from ..utils.logging import request_id_context
        return str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def _host(url) -> str:
    """The clearnet host, without a default port (review r1 n9: `:443` named in code is the same site) or a
    trailing root dot."""
    from .host_memo import _clearnet_host
    from urllib.parse import urlparse
    u = str(url or "")
    u = ("https:" + u) if u.startswith("//") else u
    h = _clearnet_host(u)
    try:
        port = urlparse(u).port
    except ValueError:
        port = None
    if h and port in (80, 443):
        h = h.rsplit(":", 1)[0]                  # review r2 n10: only an EXPLICIT port (IPv6 hosts hold colons)
    return re.sub(r"\.(?=:\d+$|$)", "", h)          # the root dot, before a port too (review r3 n1)


def forget_request(request_id: str) -> None:
    """Start `request_id` over (a scheduled task's run reuses its job's id)."""
    _LEDGER.pop(str(request_id or ""), None)


def _entry(rid: str, create: bool):
    e = _LEDGER.get(rid)
    if e is not None and time.monotonic() - e["t"] > _REQUEST_TTL_S:
        _LEDGER.pop(rid, None)
        e = None
    if e is None and create:
        e = _LEDGER[rid] = {"t": time.monotonic(), "hosts": {}, "noted": set()}
        while len(_LEDGER) > _MAX_REQUESTS:
            _LEDGER.popitem(last=False)
    elif e is not None:
        _LEDGER.move_to_end(rid)
        if create:
            e["t"] = time.monotonic()            # review r2 M3: the clock runs from the LAST booking (a long turn)
    return e


def note_reach(url, reached: bool, cause: str = "", *, request_id: str = None) -> None:
    """Record one attempt on `url`'s clearnet host for the current request. A reach is final for the
    request (a later failure does not undo it — the host exists); a failure counts and keeps its cause."""
    host = _host(url)
    rid = request_id if request_id is not None else _rid()
    if not (host and rid):
        return
    hosts = _entry(rid, True)["hosts"]
    rec = hosts.get(host)
    if rec is None:
        if len(hosts) >= _MAX_HOSTS:
            return
        rec = hosts[host] = [False, 0, "", str(url)[:200]]
    if reached:
        rec[0] = True
    else:
        rec[1] += 1
        if cause:
            rec[2] = re.sub(r"\s+", " ", str(cause)).strip()[:160]


def unreached(request_id: str = None) -> Dict[str, list]:
    """{host: record} — hosts this request tried and never reached."""
    rid = request_id if request_id is not None else _rid()
    e = _entry(rid, False)
    return {h: r for h, r in (e["hosts"] if e else {}).items() if not r[0] and r[1] > 0}


def hosts_named(text) -> List[Tuple[str, str]]:
    """[(host, url)] for every clearnet http(s) URL in `text`, first occurrence per host, in order."""
    out, seen = [], set()
    for m in _URL_RE.finditer(str(text or "")):
        url = m.group(0).rstrip(".:!?")
        url = ("https:" + url) if url.startswith("//") else url
        h = _host(url)
        if h and h not in seen:
            seen.add(h)
            out.append((h, url))
    return out


def is_code_path(path) -> bool:
    return str(path or "").lower().endswith(CODE_EXTS)


def note_for_written(files: Iterable[Tuple[str, str]], request_id: str = None) -> str:
    """The advisory for written code files [(path, text)] that name a host this request tried and never
    reached ("" when none does). Each (file, host) pair is noted once per request."""
    if not enabled():
        return ""
    rid = request_id if request_id is not None else _rid()
    bad = unreached(rid)
    if not bad:
        return ""
    noted = _entry(rid, True)["noted"]
    lines = []
    for path, text in files:
        if not is_code_path(path):
            continue
        where = posixpath.normpath(re.sub(r"^/workspace/", "", str(path)))     # `./a.js` and `a.js` are one file
        for host, url in hosts_named(text):
            rec = bad.get(host)
            if rec is None or (host, where) in noted:
                continue
            noted.add((host, where))
            n = rec[1]
            cause = f" (last: {rec[2]})" if rec[2] else ""
            lines.append(f"'{path}' uses {url} — the download and browser tools tried {host} {n} "
                         f"time{'s' if n != 1 else ''} in this request{cause} and never got an answer from it.")
    if not lines:
        return ""
    return ("\n\n[REACH CHECK] " + "\n[REACH CHECK] ".join(lines[:4])
            + "\nSo this code depends on an address nothing here has confirmed is a working site. Use an address a "
              "tool did reach, or tell the user plainly that this one is unverified (the sandbox reaches the web only "
              "over Tor, so it may still work for them) — do not describe it as working or official.")


# ── TLS "unrecognized name": the server has no site by that name ────────────
#: curl: `TLSV1_ALERT_UNRECOGNIZED_NAME` / LibreSSL `tlsv1 unrecognized name`; Chromium:
#: `net::ERR_SSL_UNRECOGNIZED_NAME_ALERT`
NO_SITE_RE = re.compile(r"unrecogni[sz]ed[_ ]name", re.I)

NO_SITE_HINT = (
    "The server at that address answered the TLS handshake with \"unrecognized name\": it has no HTTPS site "
    "for this host name. That is the SERVER's answer, not a Tor or network fault, and a retry gets the same "
    "answer — the name is wrong, parked or retired. Use another source; do not present this address as a "
    "working site.")


def is_no_site(text) -> bool:
    return bool(NO_SITE_RE.search(str(text or "")))


def _ours(cause) -> bool:
    """A failure of OUR side (the runner crashed, the page was closed, the sandbox is down) says nothing
    about the site (review r1 m8: "Page.goto: Target page … closed" read as the host's miss)."""
    from .browser_support import _MECHANICAL
    return bool(_MECHANICAL.search(str(cause or "")))


#: a browser failure that is the SITE not answering: a network error code. Review r5 M2: NOT any `Page.goto`
#: error — "Download is starting" is the server answering with a file, a timeout is a page that committed and
#: kept loading (the commonest browser failure over Tor), "interrupted by another navigation" a JS redirect,
#: ERR_ABORTED a cancelled or download navigation; a selector the page lacked reached its host too
#: (review r6 m5: not ERR_PROXY_CONNECTION_FAILED either — our Tor proxy down says nothing about the site)
_NAV_FAIL_RE = re.compile(r"net::ERR_(?!ABORTED\b|PROXY_CONNECTION_FAILED\b|HTTP_RESPONSE_CODE_FAILURE\b)[A-Z_]+")
#: the server answered, though the goto "failed" (ERR_HTTP_RESPONSE_CODE_FAILURE: an error status, empty body)
#: ("interrupted by another navigation" books nothing — review r7 m8: the page's own timer may have navigated
#: while a dead host was still pending)
_NAV_ANSWERED_RE = re.compile(r"Download is starting|ERR_HTTP_RESPONSE_CODE_FAILURE", re.I)


def note_browser_reach(url, reached: bool, cause: str = "") -> None:
    """The browser's attempts: a load (or an HTTP refusal) is a reach; only a network error is a miss."""
    try:
        if reached or _NAV_ANSWERED_RE.search(str(cause or "")):
            note_reach(url, True)
        elif not _ours(cause):
            m = _NAV_FAIL_RE.search(str(cause or ""))
            if m:
                note_reach(url, False, f"browser: {m.group(0)}")
    except Exception:  # noqa: BLE001 — bookkeeping never fails a browser call
        pass


def note_asked_reach(step, actions, op_url) -> None:
    """A successful interact goto `step` carries where a redirect LANDED; the host it ASKED for answered too
    (review r5 m3). `step["index"]` indexes the sanitised `actions` the runner got; -1 is the op's own url."""
    try:
        i = int(step.get("index"))
        asked = (actions or [])[i] if i >= 0 else {"url": op_url}
        if isinstance(asked, dict) and asked.get("url"):
            note_browser_reach(asked["url"], True)
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
