"""Browser failure routes (2026-09-15, §4HB).

Errors whose IDENTICAL retry cannot succeed, each with the action that can.
Split out of `browser.py` so the tool file stays under its post-extraction
size gate; the table has one consumer (`_deterministic_error_route`) and
one pin (`tests/test_browser_deterministic_routes.py`), which walks every
row and checks the route reaches the failure hint.

Keep entries to errors whose cause is the REQUEST itself, not the network:
a SOCKS/connection error is a circuit's opinion and IS worth one retry
(and the dead-onion memo already owns those).
"""

import re

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
# `STATUS: OK`, the model's reasoning then read "FT (15 Sep 2026): … tied to
# pec.interno.it" as if it had read the FT, and the report listed FT and
# KELA among its sources. Everything it knew of the FT story came from a
# secondary site's summary. Corpus: 75 of 584 browser OK results were
# challenge or 4xx pages. The document's own response status is decisive;
# a challenge TITLE counts only with a near-empty body, so a real article
# that happens to be titled "Access Denied" is not misfiled.
_CHALLENGE_TITLE_RE = re.compile(
    r"^\s*(?:just a moment|security verification|attention required|"
    r"access (?:to this page has been )?denied|verify you are human|"
    r"are you a human|checking your browser|please enable cookies|"
    r"403 forbidden|401 authorization required|429 too many requests|"
    r"cloudflare|ddos-guard|one moment, please|bot verification|"
    # §4HV (2026-09-16, req d121dcc6): Hetzner's "Heray" gate — HTTP 200,
    # TITLE "Security Check", 211 chars, URL rewritten to /_ray/pow, body
    # "Checking that you are not a robot … Verifying…". Passed as a normal
    # page five times in one turn; the model re-navigated and the
    # no-progress breaker forced the conclusion. Word-bounded so a real
    # "Security Checklist …" article never matches.
    r"security check\b|checking that you are not a robot|robot check\b|"
    r"human verification|captcha\b)",
    re.IGNORECASE)
_CHALLENGE_URL_RE = re.compile(r"__cf_chl|/cdn-cgi/challenge|/_ray/pow\b", re.IGNORECASE)
_BLOCKED_STATUSES = frozenset({401, 403, 407, 429, 451, 503})
_CHALLENGE_MAX_CHARS = 2000


def blocked_page_reason(parsed: dict) -> str:
    """Why this fetch must not be read as the page — "" when it may.

    ``parsed`` is the runner's result for a url-loading op (navigate,
    extract_text, screenshot): ``status`` (document HTTP status, when the
    runner captured it), ``title``, ``url``, ``length``.
    """
    if not isinstance(parsed, dict):
        return ""
    status = parsed.get("status")
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    title = str(parsed.get("title") or "")
    url = str(parsed.get("url") or "")
    try:
        length = int(parsed.get("length") or 0)
    except (TypeError, ValueError):
        length = 0
    if status in _BLOCKED_STATUSES:
        kind = "bot challenge" if (_CHALLENGE_TITLE_RE.match(title) or _CHALLENGE_URL_RE.search(url)) else "access refused"
        return f"HTTP {status} — {kind}"
    if status is not None and status >= 400:
        return f"HTTP {status}"
    if (_CHALLENGE_TITLE_RE.match(title) or _CHALLENGE_URL_RE.search(url)) and length < _CHALLENGE_MAX_CHARS:
        return f"bot challenge / interstitial ({title.strip()[:40] or 'challenge url'})"
    return ""


BLOCKED_PAGE_HINT = (
    "This site refused the fetch — the text below (if any) is the challenge or "
    "error page, NOT the article. Do not cite this page as a source you read. "
    "A retry from here will not pass the challenge: use a secondary source that "
    "quotes it and attribute the claim to that source."
)

