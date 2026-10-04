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


def _onion_kind(host: str) -> str:
    """"v3", "v2", or "invalid" (wrong length or a non-base32 character)."""
    lab = _label(host)
    if not _BASE32_RE.fullmatch(lab):
        return "invalid"
    return {56: "v3", 16: "v2"}.get(len(lab), "invalid")


def haystack_from(messages: Iterable, tool_texts: Iterable[str] = ()) -> str:
    """Everything the conversation SAW: every non-assistant message's text,
    every tool call's arguments, and the assistant's answers to EARLIER
    requests (a follow-up may repeat a link it gave before). The assistant
    text after the last user message — the reply being checked — is left
    out, so an invented link cannot vouch for itself."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    parts: List[str] = [str(t or "") for t in tool_texts]
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    parts.append(str(fn.get("arguments") or ""))
            if i > last_user:
                continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(it.get("text", "")) for it in c if isinstance(it, dict))
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


def onion_claim_issues(claim: str, haystack: str) -> List[str]:
    """Deterministic onion issues in ``claim`` against what the turn saw:
    an address of an impossible length, a v3 address no tool output holds,
    and — in a Latin-script reply only (the context words are English) — a
    v2 address presented as current."""
    issues: List[str] = []
    if not claim:
        return issues
    hay = _norm(haystack)
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
            issues.append(f"{host} is not a valid onion address: a current (v3) address is 56 "
                          f"characters of a-z and 2-7.")
        elif not re.search(rf"(?<![a-z0-9]){re.escape(host)}", hay):
            issues.append(f"The onion address {host} appears in no tool output this turn — an onion "
                          f"address cannot be confirmed from memory.")
    return issues
