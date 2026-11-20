"""Browser failure routes (2026-09-15, §4HB).

Errors whose IDENTICAL retry cannot succeed, each with the action that can.
Split out of `browser.py` so the tool file stays under its post-extraction
size gate; the table has one consumer (`_deterministic_error_route`) and
one pin (`tests/test_browser_deterministic_routes.py`), which walks every
row and checks the route reaches the failure hint.

Keep entries to errors whose cause is the REQUEST itself, not the network:
a SOCKS/connection error is owned by the memos instead: the dead-onion memo
for onions, and since §4RX the clearnet host memo (two strikes, counted
only when a page loaded between them, so a Tor outage bans nothing).
"""

_DETERMINISTIC_ERROR_ROUTES: tuple = (
    ("ERR_HTTP2_PROTOCOL_ERROR",
     "This site rejected the headless client at the protocol layer — "
     "re-requesting the same URL fails the same way. Use a different "
     "source for this content (a web_search hit that quotes it, or an "
     "archive copy), do not retry this URL."),
    ("Execution context was destroyed",
     "The page navigated (redirect, consent wall, or a late script) while "
     "the text was being read — a bare extract_text re-issued unchanged "
     "will race it again. Re-run as operation='interact' with "
     "[{\"action\":\"goto\",\"url\":…,\"wait_until\":\"networkidle\"}, "
     "{\"action\":\"sleep\",\"ms\":2000}, {\"action\":\"extract_text\"}] so "
     "the read happens after the page settles."),
)


def _deterministic_error_route(error_text: str) -> str:
    """The specific next action for a known non-retryable error, or ""."""
    text = str(error_text or "")
    for marker, route in _DETERMINISTIC_ERROR_ROUTES:
        if marker in text:
            return route
    return ""


# ── §4HH (2026-09-16, req 7ee072f2) — the blocked page that read as OK ──
# ft.com answered a 403 "Security Verification" page (554 chars) and
# kelacyber.com a 403 "Just a moment…" challenge (0 chars); both came back
# `STATUS: OK` and the report listed them among its sources. The document's
# own response status is decisive; the wall itself is judged by THE page
# classifier (§4UY N5: `utils.content_check.classify_page` — the browser had
# its own title-only rule and booked the 51-char "Please wait while your
# request is being verified..." page OK while the search readers called it a
# bot check). The page's text, when the runner shipped it, is read too.
from ..utils.content_check import (  # noqa: E402 — one classifier (§4UY N5)
    CHALLENGE_TITLE_RE as _CHALLENGE_TITLE_RE, REFUSAL_STATUSES as _BLOCKED_STATUSES, WALL_REFUSAL,
    classify_page)


def blocked_page_reason(parsed: dict) -> str:
    """Why this fetch must not be read as the page — "" when it may.

    ``parsed`` is the runner's result for a url-loading op (navigate,
    extract_text, screenshot) or an interact goto: ``status`` (document HTTP
    status, when the runner captured it), ``title``, ``url``, ``text`` (the
    page text or its preview) and its full size under ``length`` /
    ``dom_text_chars`` / ``nav_text_chars``. Unknown size: the title alone
    decides nothing (§4LN).
    """
    if not isinstance(parsed, dict):
        return ""
    status = parsed.get("status")
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    title = str(parsed.get("title") or "")
    _raw_len = next((parsed.get(k) for k in ("length", "dom_text_chars", "nav_text_chars")
                     if parsed.get(k) is not None), None)
    try:
        length = int(_raw_len) if _raw_len is not None else None
    except (TypeError, ValueError):
        length = None
    text = parsed.get("text")
    text = text if isinstance(text, str) else None
    kind, detail = classify_page(status, title, text, str(parsed.get("url") or ""), length)
    if status in _BLOCKED_STATUSES:
        # a refusal status decides; a wall title/url under it names it a challenge even when the size is unknown
        _wall = (kind and kind != WALL_REFUSAL) or bool(_CHALLENGE_TITLE_RE.match(title))
        return f"HTTP {status} — {'bot challenge' if _wall else 'access refused'}"
    if status is not None and status >= 400:
        return f"HTTP {status}"
    if kind == "bot challenge":
        return f"bot challenge / interstitial ({title.strip()[:40] or detail})"
    if kind:
        return f"{kind} ({detail})"
    return ""


BLOCKED_PAGE_HINT = (
    "This site refused the fetch — the text below (if any) is the challenge or "
    "error page, NOT the article. Do not cite this page as a source you read. "
    "A retry from here will not pass the challenge: use a secondary source that "
    "quotes it and attribute the claim to that source. All you know of THIS "
    "page is its search-result snippet: if you use it, say it is from a search "
    "snippet of an unread page."
)



def _is_own_host(url: str) -> bool:
    """file://, loopback, a private/LAN address or a .local name — the
    agent's own app or service, never a third-party site."""
    from urllib.parse import urlparse
    import ipaddress
    try:
        u = urlparse(str(url or ""))
    except ValueError:
        return False
    if u.scheme == "file":
        return True
    host = (u.hostname or "").lower()
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


def blocked_page_hint(reason: str, url: str) -> str:
    """The next step for a page that must not be read, by CAUSE (§4LN): the
    agent's own server failing is not "a site refused you"; a 404 is a URL
    that does not exist (25 of 99 live BLOCKED results were 404s on URLs the
    model had guessed), not a challenge a secondary source works around."""
    r = str(reason or "")
    if _is_own_host(url):
        return ("This is YOUR OWN app/service, and it answered with an error — not a "
                "site blocking you. Read its log or the response, fix the route or the "
                "server, then load it again.")
    if r.startswith("HTTP 404") or r.startswith("HTTP 410"):
        return ("There is no page at this URL. If you built or guessed the URL, it is "
                "wrong — find the real one (web_search, or a link on a page you did "
                "load) instead of trying variants. Do not cite this URL.")
    return BLOCKED_PAGE_HINT
