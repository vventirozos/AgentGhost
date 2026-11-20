"""§4LH: links in a research answer must come from what the turn read.

Measured in the §4LG audit: 1 invented URL in 42 research replies
(``facebook.com/GKTeamBJJ/``, in no result), and the dark-web probes gave
a garbled DuckDuckGo onion and a retired v2 one. A link the user follows
should lead where the answer says; an invented deep link leads to a 404 or
to someone else's page, and a wrong onion can be a phishing clone.

``ground_links`` runs at finalize on a NON-STREAMED turn that called a
research tool (a streamed reply has already gone out word by word; there the
verifier's onion check is what remains). The in-loop judged text is not
grounded, so a removal costs one recomputed verdict. A clearnet URL with a path, or a v3 onion
address, that appears in nothing the conversation saw is removed: a markdown
link or image keeps its label, a bare URL keeps its domain, an onion address
is replaced by a marker. "Saw" = tool results, tool-call arguments, the
user's and system text, and the agent's EARLIER answers — never the reply
being checked. An onion of an impossible length (a garbled address) is
removed too. Left alone: a bare domain (``python.org``), any clearnet URL
inside code, and a v2 (16-character) onion — it cannot load anyway, and the
verifier says why. One italic line says how many were removed.
``GHOST_LINK_GROUNDING=0`` disables.

``onion_claim_issues`` is the verifier's deterministic onion check. The judge
confirmed a retired v2 address as "official v3" from memory (§4LG D2).
"""
from __future__ import annotations

import html
import os
import re
from typing import Iterable, List, Tuple
from urllib.parse import unquote

#: Tools whose results a research answer is built from.
RESEARCH_TOOLS = frozenset({
    "web_search", "darkweb_search", "deep_research", "darkweb_research",
    "browser", "fact_check", "news_headlines",
})

_UNI_PUNCT = "…。，、」「）（】【！？：；"
_MD_LINK_RE = re.compile(r"(!?)\[([^\]\n]{0,300})\]\((https?://(?:[^()\s]|\([^\s()]*\))+)\)")
# balanced (…) groups stay inside a URL: Wikipedia's Foo_(bar) lost its ")" (review)
_URL_RE = re.compile(r"https?://(?:[^\s<>()\[\]\"'`" + _UNI_PUNCT + r"]|\([^\s()<>]*\))+", re.I)
#: Any alphanumeric label: a garbled address carries characters base32 never
#: uses (the live D2 one had a "1"), and must still be caught.
_ONION_RE = re.compile(r"\b([a-z0-9]{10,}\.onion)\b", re.I)
_BASE32_RE = re.compile(r"[a-z2-7]+")
_CODE_RE = re.compile(r"```.*?```|`[^`\n]+`", re.S)
_TRAIL = ".,;:!?*_'\")]}>" + _UNI_PUNCT
_ONION_MARK = "[onion address removed — it is in no page or result I read]"
#: Start of the line ground_links appends; reply_smoothing strips it as ours.
REMOVED_NOTE_PREFIX = "_Removed "
#: Anchored on the literal, no leading ``\n*`` (a leading unbounded run was
#: quadratic on a reply of newlines — the §4FY lesson, repeated here once).
#: A whole LINE, wherever it sits — a source caveat appended after it (§4LH
#: final review) hid it from a tail-anchored match.
REMOVED_NOTE_RE = re.compile(r"(?m)^_Removed \d+ links? that appears? in no page or search result I read this turn\._[ \t]*$\n?")


def enabled() -> bool:
    return os.environ.get("GHOST_LINK_GROUNDING", "1") != "0"


def _norm(text: str) -> str:
    t = unquote(html.unescape(text or "")).lower()
    for p in ("https://", "http://", "www."):
        t = t.replace(p, "")
    return t


def _norm_url(url: str) -> str:
    u = _norm(url).split("#", 1)[0]
    return u.rstrip(_TRAIL).rstrip("/")


def _has_path(url: str) -> bool:
    host, _, path = _norm_url(url).partition("/")
    return bool(path.strip("/")) or "?" in host


def _domain(url: str) -> str:
    return _norm_url(url).split("/", 1)[0].split("?", 1)[0]


def _label(host: str) -> str:
    return host.lower()[: -len(".onion")].rsplit(".", 1)[-1]


def onion_v3_valid(host_or_url: str) -> bool:
    """§4UN-dark: a v3 onion address whose built-in checksum holds. The 56 characters are base32 of
    pubkey(32) + checksum(2) + version(1, = 3), checksum = SHA3-256(".onion checksum" + pubkey + 0x03)[:2]
    — so a one-character garble (a summarizer copying an address and changing a letter) fails here
    wherever it came from. Takes a host or a URL."""
    import base64
    import hashlib
    raw = str(host_or_url or "").strip().lower()
    if "://" in raw:
        from urllib.parse import urlparse
        raw = (urlparse(raw).hostname or "")
    raw = raw.split("/", 1)[0]
    if not raw.endswith(".onion"):
        return False
    label = raw[: -len(".onion")].rsplit(".", 1)[-1]
    if len(label) != 56 or not re.fullmatch(r"[a-z2-7]{56}", label):
        return False
    try:
        b = base64.b32decode(label.upper())
    except Exception:  # noqa: BLE001
        return False
    pub, chk, ver = b[:32], b[32:34], b[34:35]
    return ver == b"\x03" and hashlib.sha3_256(b".onion checksum" + pub + b"\x03").digest()[:2] == chk


def _onion_kind(host: str) -> str:
    """"v3", "v2", or "invalid" (wrong length, a non-base32 character, or — §4UN-dark — a 56-character
    address whose built-in v3 checksum fails: one garbled letter)."""
    lab = _label(host)
    if not _BASE32_RE.fullmatch(lab):
        return "invalid"
    kind = {56: "v3", 16: "v2"}.get(len(lab), "invalid")
    if kind == "v3":
        if not onion_v3_valid(lab + ".onion"):
            return "invalid"
    return kind


def _failed_browser_result(content) -> bool:
    """A browser result whose own header says it did not read the page."""
    from ..distill.outcome_heuristics import browser_result_status
    from ..tools.browser_routes import _is_own_host
    text = content if isinstance(content, str) else str(content or "")
    # the producer's DECLARED status first (a live row); the header for a
    # rehydrated one
    _declared = getattr(content, "status", None)
    _dv = str(getattr(_declared, "value", _declared or "")).lower()
    if _dv in ("ok", "partial", "unresolved"):
        return False
    st = browser_result_status(text)
    if st is None or st in ("ok", "partial"):
        return False
    # the agent's OWN app answering 4xx/5xx still exists (review)
    m = re.search(r"(?m)^(?:FINAL_)?URL:\s*(\S+)", text)
    return not (m and _is_own_host(m.group(1)))


def trim_result_url(url: str) -> str:
    """A URL as a tool result printed it, without the sentence punctuation after it: trailing `.`/`,` go, and
    a trailing `)` goes only when the URL has no `(` it closes — `…/Python_(programming_language)` keeps its
    own (2026-11-19: the Sources list showed that link cut)."""
    u = str(url or "")
    while u:
        if u[-1] in ".,":
            u = u[:-1]
        elif u[-1] == ")" and u.count(")") > u.count("("):
            u = u[:-1]
        else:
            break
    return u


def fetched_urls(tools_run) -> List[str]:
    """§4OE-4: the URLs this request actually READ — the URL/FINAL_URL line
    of every browser result that did not fail — in order, deduplicated."""
    out: List[str] = []
    for t in tools_run or []:
        if not isinstance(t, dict):
            continue
        if t.get("name") == "file_system":                     # §4OH reader M9: a downloaded URL was read
            _txt = str(t.get("content") or "")
            m = re.match(r"\s*SUCCESS: Downloaded '(https?://[^']+)'", _txt)
            if m and m.group(1) not in out:
                out.append(m.group(1))
            continue
        if t.get("name") == "web_search":
            # §4UN-read: the result pages web_search opened itself (its READ FROM PAGES section) were
            # read; the listed results were not
            try:
                from ..tools.search import web_read_urls
                out.extend(u for u in web_read_urls(t.get("content")) if u not in out)
            except Exception:  # noqa: BLE001
                pass
            continue
        if t.get("name") not in ("browser", "deep_research", "darkweb_research", "fact_check"):
            continue
        content = t.get("content")
        _dv = str(getattr(getattr(content, "status", None), "value", getattr(content, "status", "") or "")).lower()
        if _dv in ("failed", "rejected") or _failed_browser_result(content):
            continue                      # a declared failure read nothing (review 2)
        text = content if isinstance(content, str) else str(content or "")
        if t.get("name") == "fact_check":
            # §4TB (review): the sources a completed check LISTS are pages its
            # research fetched — a report citing one drew "NOT fetched". Only
            # the URL of each row counts (`factcheck_listed_urls`): a title or
            # an evidence line is page text and holds no link.
            try:
                from ..tools.search import factcheck_listed_urls
                listed = factcheck_listed_urls(text)
            except Exception:  # noqa: BLE001
                listed = []
            out.extend(u for u in listed if u not in out)
            continue
        if t.get("name") in ("deep_research", "darkweb_research"):
            # per-source blocks: a block the fetcher wrote as a failure is not a page read
            try:
                from ..tools.search import source_block_failed
            except Exception:  # noqa: BLE001
                source_block_failed = lambda _b: False  # noqa: E731
            for block in re.split(r"(?m)^(?=###\s*SOURCE:)", text):
                m = re.match(r"###\s*SOURCE:\s*(https?://\S+)", block)
                if m and not source_block_failed(block):
                    u = trim_result_url(m.group(1))
                    if u not in out:
                        out.append(u)
            try:            # §4UN-deep: a page that loaded and held nothing relevant was still READ
                from ..tools.search import nothing_relevant_urls
                out.extend(u for u in nothing_relevant_urls(text) if u not in out)
            except Exception:  # noqa: BLE001
                pass
            continue
        for m in re.finditer(r"(?m)^(?:FINAL_)?URL:\s*(https?://\S+)", text):
            u = trim_result_url(m.group(1))
            if u not in out:
                out.append(u)
    return out


#: tools whose results LIST links with snippets (the pages were not opened)
_SEARCH_LIST_TOOLS = ("web_search", "darkweb_search", "news_headlines")


def search_result_urls(tools_run) -> List[str]:
    """§4QA: the URLs this request's SEARCH results listed — the model saw a
    snippet of these pages, not the page. A cited one is unread, but it is
    not "maybe from an earlier request" either."""
    out: List[str] = []
    for t in tools_run or []:
        if not isinstance(t, dict) or t.get("name") not in _SEARCH_LIST_TOOLS:
            continue
        _txt = str(t.get("content") or "")
        if t.get("name") == "web_search":
            # §4UN-read r1: the pages web_search READ are not snippets (`fetched_urls` has them), and a
            # link inside their text was on a page, not in a result list
            try:
                from ..tools.search import without_read_section
                _txt = without_read_section(_txt)
            except Exception:  # noqa: BLE001
                pass
        for m in _URL_RE.finditer(_txt):
            u = m.group(0).rstrip(_TRAIL)
            if u not in out:
                out.append(u)
    return out


#: §4UY N7: at most this many links in the sources line
SOURCES_MAX = 5
#: §4UY N7: the head of the appended sources line (pages read) and of its snippet-only form
SOURCES_HEAD = "Sources:"
SOURCES_SNIPPET_HEAD = "Search results used (pages not opened):"
SOURCES_NOTE_RE = re.compile(r"(?m)^(?:Sources:|Search results used \(pages not opened\):)[ \t]*$")
#: §4UY N7 r2: the whole appended block (head + its "- " lines) — what a voice path drops
SOURCES_BLOCK_RE = re.compile(r"(?m)^(?:Sources:|Search results used \(pages not opened\):)[ \t]*\n"
                              r"(?:[ \t]*- [^\n]*(?:\n|$))*")
#: §4UY N7 r2: a reply this short is an answer, not a report — no list under it
SOURCES_MIN_REPLY_CHARS = 40
#: §4UY N7 r2: the request fixed the WHOLE reply's form — "reply only with …", "just the number", "as JSON",
#: "no links". Review r2: words that only shape the content ("just the top 3", "only the important parts",
#: "exactly which one") do not, and "no sources of lithium" is a topic, not a form.
OUTPUT_FORM_RE = re.compile(
    r"\b(?:reply|respond|answer)\b(?:\s+back)?\s+(?:only|just|exactly)\s+with\b"
    r"|\b(?:reply|respond|answer)\b(?:\s+with)?\s+(?:only|just)\s+(?:the|a|one)\s+(?:number|word|name|value|figure|digit|letter|line)\b"
    r"|\b(?:just|only)\s+the\s+(?:number|word|name|value|figure|digits?)\b(?!\s+of\b)"
    r"|\b(?:in|as)\s+(?:json|csv|yaml|xml)\b|\bjson\s+only\b|\bnothing else\b|\bone[- ]word answer\b"
    r"|\b(?:no|without)\s+(?:links?|urls?|citations?)\b"
    # r3: "no sources (please)", "without sources", "don't include (the) sources" — the word ends the clause
    # ("no sources of lithium" is a topic)
    r"|\b(?:no|without)\s+(?:the\s+)?sources\b(?!\s+(?:of|for|on|about|in|that|which|who)\b)"
    r"|\b(?:do\s*n[o'’]?t|do\s+not|never)\s+(?:include|add|list|give|show|append|cite)\s+(?:the\s+|any\s+)?"
    r"(?:sources|links|urls|citations)\b", re.I)
_PRIVATE_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home.arpa", ".ts.net", ".onion")


def is_public_source_url(url: str) -> bool:
    """§4UY N7 r2: a URL a reader could open on the internet — never loopback, a private or tailnet address, a
    name with no dot, an onion, or a local name (the agent's own app in the browser is not a source)."""
    try:
        import ipaddress
        from urllib.parse import urlsplit
        host = (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except Exception:  # noqa: BLE001
        return False
    if not host or host == "localhost" or host.endswith(_PRIVATE_SUFFIXES):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host


def reply_takes_sources(reply: str, request: str = "") -> bool:
    """§4UY N7 r2: a reply that may carry the list — at least SOURCES_MIN_REPLY_CHARS, not a bare JSON/code
    answer, and the request did not fix the reply's form."""
    text = str(reply or "").strip()
    if len(text) < SOURCES_MIN_REPLY_CHARS:
        return False
    if (text[0] in "{[" and text[-1] in "}]") or (text.startswith("```") and text.endswith("```")):
        return False
    return not OUTPUT_FORM_RE.search(str(request or ""))


def sources_note(reply: str, tools_run, request: str = "") -> str:
    """§4UY N7: the sources a research answer was built from, appended by the code — of 95 live replies
    built on the internet tools 0 carried a URL and 61 named no source at all; asking in the tool
    descriptions ("cite the SOURCE URL for each claim") did not do it.

    Only when a research tool (RESEARCH_TOOLS) ran this turn — a download or a file read alone is not
    research — and the reply may carry a list (`reply_takes_sources`). The public pages this request READ
    (`fetched_urls`: browser pages, deep research sources, the pages web_search opened, fact_check's listed
    sources, downloads) — or, when it read none, the results its searches LISTED (`search_result_urls`,
    said to be snippets); never a loopback/private/tailnet address (`is_public_source_url`). "" when the
    reply already links one of them (the model cited on purpose), or when nothing is left. One line per
    link, at most ``SOURCES_MAX``."""
    text = str(reply or "")
    if not reply_takes_sources(text, request):
        return ""
    if not any(isinstance(t, dict) and t.get("name") in RESEARCH_TOOLS for t in (tools_run or [])):
        return ""
    head, urls = SOURCES_HEAD, [u for u in fetched_urls(tools_run) if is_public_source_url(u)]
    if not urls:
        head, urls = SOURCES_SNIPPET_HEAD, [u for u in search_result_urls(tools_run) if is_public_source_url(u)]
    if not urls:
        return ""
    cited = {_cite_norm(m.group(0)) for m in _URL_RE.finditer(text)}
    if cited & {_cite_norm(u) for u in urls}:
        return ""
    seen: List[str] = []
    for u in urls:
        if _cite_norm(u) not in {_cite_norm(x) for x in seen}:
            seen.append(u)
    shown = seen[:SOURCES_MAX]
    more = len(seen) - len(shown)
    return ("\n\n" + head + "\n" + "\n".join(f"- {u}" for u in shown)
            + (f"\n- (+{more} more)" if more > 0 else ""))


def reply_is_contained(context) -> bool:
    """§4UY N7 r2: a contained turn — a coding leaf or an isolated sub-agent run (`_subagent_allowed_tools`
    set by `build_leaf_context` / `isolation`) — whose reply goes to code, not to a reader."""
    if context is None:
        return False
    allowed = getattr(context, "_subagent_allowed_tools", None)
    if isinstance(allowed, (set, frozenset, list, tuple)) and allowed:
        return True
    static = getattr(context, "trajectory_extra_static", None)
    return isinstance(static, dict) and bool(static.get("leaf_id"))


def sources_for_reply(reply: str, tools_run, request: str = "", context=None) -> str:
    """§4UY N7 r2: THE call both final-reply sinks make (finalize, the streamed final) — `sources_note`
    unless switched off (``GHOST_REPLY_SOURCES=0``) or the turn is contained (`reply_is_contained`)."""
    if os.environ.get("GHOST_REPLY_SOURCES", "1") == "0" or reply_is_contained(context):
        return ""
    return sources_note(reply, tools_run, request)


def _cite_norm(url: str) -> str:
    """One spelling for a citation and its fetch: scheme/host case, `www.`,
    a fragment, a trailing slash."""
    u = str(url or "").strip().rstrip(_TRAIL)
    u = re.sub(r"#.*$", "", u)
    m = re.match(r"(?i)(https?)://([^/]+)(.*)$", u)
    if not m:
        return u.lower()
    host = m.group(2).lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{m.group(3).rstrip('/')}"


#: §4SK (req a60aa2de): in an HTML page the URLs of its ASSETS — stylesheets,
#: fonts, scripts, images, frames, `url()` in CSS — are not citations; the
#: Google Fonts <link> of a web app was flagged twice as "a link written into
#: the file that was not opened". An <a href> is kept: that is how an HTML
#: report cites.
#: r2: a quoted attribute value may hold `>` (`alt="a>b"`, `onclick="if(a>b)…"`);
#: each value is bounded so an unclosed quote cannot scan the whole file per tag
_ASSET_TAG_RE = re.compile(
    r"<(?:link|script|img|image|input|source|iframe|meta|video|audio|track|embed|object|use|base)(?=[\s/>])"
    r"(?:\"[^\"]{0,1500}\"|'[^']{0,1500}'|[^'\">]){0,1500}>",
    re.IGNORECASE)
#: bounded runs (r2: an unclosed `url(https://…` scanned to the end of the
#: text from every match start — 20k repeats took 39 s)
_CSS_URL_RE = re.compile(r"url\(\s*['\"]?https?://[^)'\"\s]{0,2000}['\"]?\s*\)"
                         r"|@import\s+(?:url\()?['\"]?https?://[^;'\")\s]{1,2000}",
                         re.IGNORECASE)


#: r3: a real tag-name boundary (`<script-note>` is a custom element), case-insensitive
#: searches on the TEXT itself — offsets taken from `body.lower()` drift where lowering
#: changes the length ("İ" → 2 characters) and cut a citation instead of a script
_NOISE_OPEN_RE = re.compile(r"<!--|<script(?=[\s/>])", re.IGNORECASE)
_SCRIPT_END_RE = re.compile(r"</script\s*>", re.IGNORECASE)
_COMMENT_END_RE = re.compile(r"--!?>")


def _strip_script_bodies(body: str) -> str:
    """The code inside `<script>…</script>` — a `fetch()` of an API, an ES
    module import from a CDN, JSON-LD — is not a citation (r2), and neither
    is an HTML comment (r3: `<!-- TODO add <script> analytics -->` opened a
    "script" that swallowed every later citation). An UNCLOSED script or
    comment is left as it is: dropping the rest of the page would drop
    citations; flagging a script URL is the safer error. Linear."""
    out, i = [], 0
    while True:
        m = _NOISE_OPEN_RE.search(body, i)
        if not m:
            break
        if m.group(0) == "<!--":
            # r4: the HTML spec's early closes — `<!-->` and `<!--->` are empty
            # comments, `--!>` closes one too (each ran on to the next `-->`)
            if body.startswith(">", m.end()):
                close = m.end() + 1
            elif body.startswith("->", m.end()):
                close = m.end() + 2
            else:
                e = _COMMENT_END_RE.search(body, m.end())
                if not e:
                    break
                close = e.end()
            out.append(body[i:m.start()])
            i = close
            continue
        gt = body.find(">", m.end())
        if gt < 0:
            break
        e = _SCRIPT_END_RE.search(body, gt + 1)
        if not e:
            break
        out.append(body[i:gt + 1])
        i = e.start()
    out.append(body[i:])
    return "".join(out)


def is_markup_path(path) -> bool:
    return str(path or "").lower().rstrip().endswith((".html", ".htm", ".xhtml"))


def unfetched_links(text: str, fetched: Iterable[str], *, markup: bool = False) -> List[str]:
    """§4OG-3 (d): the http(s) URLs in `text` (prose or markdown links, code
    spans skipped) that no fetched-OK URL matches, in order, deduplicated.
    ``markup``: the text is an HTML page — its asset tags and CSS url()s are
    dropped first (§4SK)."""
    have = {_cite_norm(u) for u in (fetched or []) if u}
    body = str(text or "")
    if markup:
        # scripts first (r3): a backtick in one inline script paired with one
        # in a later script and the code-span strip took the text between
        body = _CSS_URL_RE.sub(" ", _ASSET_TAG_RE.sub(" ", _strip_script_bodies(body)))
    body = _CODE_RE.sub(" ", body)
    out: List[str] = []
    seen = set()
    for m in _URL_RE.finditer(body):
        raw = m.group(0).rstrip(_TRAIL)
        key = _cite_norm(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in have:
            out.append(raw)
    return out


def _fact_check_seen(content) -> str:
    """What a `fact_check` result adds to the haystack. A COMPLETED check: the
    URLs of its source rows (`factcheck_listed_urls`, the same reading
    `fetched_urls` uses) and nothing else — its verdict is the model's prose,
    its titles and evidence are page text, and none of them may vouch for a
    link (§4TB review r2). Any other result — a failed or partial check hands
    back the research report itself — is read like the research tool's."""
    text = str(content or "")
    if not text.startswith("FACT CHECK COMPLETE"):
        return text
    try:
        from ..tools.search import factcheck_listed_urls
        return "\n".join(factcheck_listed_urls(text))
    except Exception:  # noqa: BLE001
        return ""


def tool_row_text(row: dict) -> str:
    """One executed tool row as haystack text: what its result showed, and
    the arguments it was called with."""
    content = row.get("content")
    seen = _fact_check_seen(content) if row.get("name") == "fact_check" else str(content or "")
    return f"{seen}\n{getattr(content, 'call_args', '') or ''}"


def haystack_from(messages: Iterable, tool_texts: Iterable[str] = (), snippet_ids=()) -> str:
    """Everything the conversation SAW: every non-assistant message's text,
    every tool call's arguments, and the assistant's answers to EARLIER
    requests that sit before the turn's request row (a follow-up may repeat
    a link it gave before). The TEXT of every assistant-role row after the
    request — this turn's narration and reply, and the assistant-role
    summary/anchor rows a prune places there — is left out (its tool-call
    arguments stay), so an invented link cannot vouch for itself (§4RZ step
    3: an earlier answer carried only by such a summary is left out with
    it). Tool rows count wherever they sit, except a failed browser result
    and a search snippet named in ``snippet_ids``; a completed `fact_check`
    row counts for its listed source URLs only (`_fact_check_seen`)."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    # §4RZ step 3: the reply being checked starts after the REQUEST — this
    # turn's narration before a message the user sent mid-turn (or before a
    # loop steer) is the reply's own voice, not something the request saw
    from .queued_messages import request_row_index
    last_user = request_row_index(msgs)
    # §4LN: a browser call that FAILED (404, refused, blocked, runner error)
    # did not see its URL — its arguments and its result text are not
    # evidence that the page exists. They vouched for an invented URL that
    # had just answered 404.
    failed_ids = {str(m.get("tool_call_id")) for m in msgs
                  if m.get("role") == "tool" and m.get("tool_call_id") and _failed_browser_result(m.get("content"))}
    parts: List[str] = [str(t or "") for t in tool_texts if not _failed_browser_result(t)]
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and str(tc.get("id")) not in failed_ids:
                    fn = tc.get("function") or {}
                    parts.append(str(fn.get("arguments") or ""))
            if i > last_user:
                # (s3-r2: the summary/anchor rows after a lifted request stay
                # out too — they copy this turn's narration, and a text prefix
                # is one the model can write)
                continue
        if m.get("role") == "tool" and _failed_browser_result(m.get("content")):
            continue
        if m.get("role") == "tool" and str(m.get("tool_call_id")) in set(map(str, snippet_ids or ())):
            continue                      # §4OE-4: a search snippet is not a page the request read
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(it.get("text", "")) for it in c if isinstance(it, dict))
        if m.get("role") == "tool" and m.get("name") == "fact_check":
            c = _fact_check_seen(c)
        parts.append(str(c or ""))
    return _norm("\n".join(parts))


def _onion_ok(host: str, hay: str) -> bool:
    if _onion_kind(host) == "v2":
        return True                       # cannot load; the verifier explains it
    # v3 seen (as a whole host, not inside a longer token), or a garbled one the user typed
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(host.lower())}", hay))


def _seen_prefix(url: str, hay: str) -> str:
    """The longest leading part of ``url`` (whole path segments, at least one)
    that the conversation saw, or "" (§4LI live probe N1: the agent read
    github.com/nodejs/node/releases and linked …/releases/tag/v24.21.0)."""
    host, _, path = _norm_url(url).partition("/")
    segs = [x for x in path.split("/") if x]
    for k in range(len(segs) - 1, 0, -1):
        cand = host + "/" + "/".join(segs[:k])
        if re.search(rf"{re.escape(cand)}(?![\w-])", hay):
            return cand
    return ""


def _extends_a_seen_page(url: str, hay: str) -> bool:
    """``url`` adds path segments to a page the conversation saw, and every
    added segment's text appears in what was read — a link built from the
    page (its tag or anchor), not invented."""
    pre = _seen_prefix(url, hay)
    if not pre:
        return False
    rest = [x for x in _norm_url(url)[len(pre):].strip("/").split("/") if x]
    # structural path words ("tag", "blob") need not appear; the segments that
    # name the thing do: at least one added segment, and every one with a digit
    # (a version, an id)
    return (any(x in hay for x in rest)
            and all(x in hay for x in rest if any(c.isdigit() for c in x)))


def ground_links(reply: str, hay: str) -> Tuple[str, List[str]]:
    """Return ``(reply, removed)``. ``hay`` is ``haystack_from(...)``."""
    if not reply:
        return reply, []
    removed: List[str] = []

    def _url_ok(url: str) -> bool:
        m = _ONION_RE.search(url)
        if m:
            return _onion_ok(m.group(1), hay)
        return (not _has_path(url)) or _norm_url(url) in hay or _extends_a_seen_page(url, hay)

    def _md(m):
        label, url = m.group(2), m.group(3)
        if _url_ok(url):
            return m.group(0)
        removed.append(url)
        if _ONION_RE.search(label):
            return _ONION_MARK
        # a label that IS the url would be found again by the bare pass
        return _domain(url) if _URL_RE.fullmatch(label.strip()) else label

    def _bare(m):
        url = m.group(0)
        core = url.rstrip(_TRAIL)
        tail = url[len(core):]
        if _url_ok(core):
            return url
        removed.append(core)
        if _ONION_RE.search(core):
            return _ONION_MARK + tail
        # the nearest page that WAS read, else the domain
        _pre = _seen_prefix(core, hay)
        return ("https://" + _pre if _pre else _domain(core)) + tail

    def _bare_onion(m):
        host = m.group(1)
        if _onion_ok(host, hay):
            return m.group(0)
        removed.append(host)
        return _ONION_MARK

    def _prose(seg: str) -> str:
        seg = _MD_LINK_RE.sub(_md, seg)
        seg = _URL_RE.sub(_bare, seg)
        return re.sub(r"(?<![/\w.])" + _ONION_RE.pattern, _bare_onion, seg, flags=re.I)

    def _code(seg: str) -> str:
        # code is left as written, except an onion address: a wrong one is a
        # phishing risk wherever it is printed
        return re.sub(r"(?<![\w.])" + _ONION_RE.pattern, _bare_onion, seg, flags=re.I)

    out, last = [], 0
    for m in _CODE_RE.finditer(reply):
        out.append(_prose(reply[last:m.start()]))
        out.append(_code(m.group(0)))
        last = m.end()
    out.append(_prose(reply[last:]))
    text = "".join(out)
    if removed:
        n = len(removed)
        text = (text.rstrip() + "\n\n" + REMOVED_NOTE_PREFIX
                + (f"{n} link that appears" if n == 1 else f"{n} links that appear")
                + " in no page or search result I read this turn._")
    return text, removed


_V2_CONTEXT_RE = re.compile(r"\bv2\b|retired|deprecated|no longer|obsolete|legacy|old(?:er)? (?:v2 )?address",
                            re.I)


def _mostly_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.6


def onion_claim_issues(claim: str, haystack: str, *, user_text: str = "",
                       notes: "List[str] | None" = None) -> List[str]:
    """Deterministic onion issues in ``claim`` against what the turn saw:
    an address of an impossible length, a v3 address no tool output holds,
    and — in a Latin-script reply only (the context words are English) — a
    v2 address presented as current.

    §4UN-dark r2: a 56-character address whose checksum fails is NOT an issue when it is in ``user_text``
    (the user's own words: the reply quotes it). One that only tool output held gets a non-blocking NOTE
    (appended to ``notes`` and logged) — a page can carry a mistyped address, and the reply repeating it
    should say so, but it was not invented. Unseen anywhere: an issue."""
    issues: List[str] = []
    if not claim:
        return issues
    hay = _norm(haystack)
    user = _norm(user_text)
    latin = _mostly_latin(claim)
    seen = set()
    for m in _ONION_RE.finditer(claim):
        host = m.group(1).lower()
        if host in seen:
            continue
        seen.add(host)
        kind = _onion_kind(host)
        if kind == "v2":
            window = claim[max(0, m.start() - 160): m.end() + 160]
            if latin and not _V2_CONTEXT_RE.search(window):
                issues.append(f"{host} is a v2 onion address: Tor removed v2 onion services in October "
                              f"2021, so it cannot be a current address.")
        elif kind == "invalid":
            if len(_label(host)) == 56 and _BASE32_RE.fullmatch(_label(host)):
                _pat = rf"(?<![a-z0-9]){re.escape(host)}"
                if user and re.search(_pat, user):
                    continue                    # r2: the user typed it — the reply quotes it
                if re.search(_pat, hay):
                    _note = (f"{host} came from tool output but its built-in checksum fails: at least one "
                             f"character is wrong, so it cannot load as written.")
                    if notes is not None:
                        notes.append(_note)
                    import logging
                    logging.getLogger(__name__).info("onion guard note: %s", _note)
                    continue
                issues.append(f"{host} is not a valid onion address: its built-in checksum fails, so at "
                              f"least one character is wrong.")
            else:
                issues.append(f"{host} is not a valid onion address: a current (v3) address is 56 "
                              f"characters of a-z and 2-7.")
        elif not re.search(rf"(?<![a-z0-9]){re.escape(host)}", hay):
            issues.append(f"The onion address {host} appears in no tool output this turn — an onion "
                          f"address cannot be confirmed from memory.")
    return issues

