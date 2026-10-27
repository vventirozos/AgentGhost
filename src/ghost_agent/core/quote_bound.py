"""§4OJ pilot: quote-bound answers (probe-only arm `quote_bound`).

Six levers failed to reduce invented specifics (§4NH, §4NI): prompt rules
that ASK the model to stay with the evidence, a judge-driven repair, lower
temperature, reading the page first. This arm BINDS instead: the model
writes a verbatim quote from a tool result after each specific, code checks
every quote by containment against the sources the model saw (no judge),
the markers are stripped, and a specific whose quote is not found — or whose
own figures are not in its quote — is marked "(unverified)" when
`GHOST_QUOTE_MARKS=1` (§4OU: hidden by default — 10 of 52 marks were right;
the check is logged either way).

Since §4OJ-b it runs on every NON-streamed reply (operator decision); the
probe arm `quote_bound` still forces it for a non-streamed probe, and the
route drops the arm on a streamed request (r1 M1: the rewrite runs on the
finished reply).
"""
from __future__ import annotations

import contextvars
import functools
import html
import json
import os
import re
import unicodedata
from typing import Iterable, List, Tuple

import logging
logger = logging.getLogger("GhostAgent")

ARM = "quote_bound"

#: §4OJ-b (operator 2026-10-13: "1yes"): ON for every NON-streamed reply (the
#: Slack bot, a member's turn, background callers) — the rewrite needs the
#: finished text, so streamed replies (web, CLI, ClockworkPi) never get it.
#: `GHOST_QUOTE_BOUND=0` switches it off. Set once per request by handle_chat.
quote_bound_context: "contextvars.ContextVar[bool]" = contextvars.ContextVar("quote_bound", default=False)
#: r2 m8/m9 + r3 M6: only a request that came in through the API's NON-streamed
#: branch is eligible. The mark rides on THAT request's body (routes.py sets it):
#: a contextvar was inherited by everything the request spawned — delegate
#: sub-agents (spawn_bg), coding leaves (wait_for copies context), and every later
#: scheduler fire after a Slack-created task.
ELIGIBLE_KEY = "_ghost_quote_bound"


def production_enabled() -> bool:
    return os.environ.get("GHOST_QUOTE_BOUND", "1").strip().lower() not in ("0", "false", "off", "no")


def marks_shown() -> bool:
    """§4OU (operator, 2026-10-14): the "(unverified)" marks and the streamed
    closing line are HIDDEN by default — 10 of 52 marks were right, 0 of 10 on
    real traffic. The model is still told to quote (that is what moved
    answer accuracy, §4OJ) and every check is still logged.
    ``GHOST_QUOTE_MARKS=1`` shows them again."""
    return os.environ.get("GHOST_QUOTE_MARKS", "0").strip().lower() in ("1", "true", "on", "yes")


def active() -> bool:
    """Whether THIS request's reply is quote-bound (rule shown + reply checked)."""
    return bool(quote_bound_context.get())


def mode() -> str:
    """"inline" — a non-streamed reply: a failed specific is marked where it
    stands; "append" — a streamed reply (§4OJ-c, operator "make quote checking
    the default"): text already on screen is never rewritten, so markers are
    hidden as they stream and failed checks are listed in one closing line."""
    v = quote_bound_context.get()
    return v if v in ("inline", "append") else ("inline" if v else "")

RULE = (
    "### ANSWER FORMAT\n"
    "In your final answer, after every specific fact you took from a TOOL RESULT in this conversation "
    "(a number, price, date, version, name, product model, statistic, or a claim about what a source "
    "says), add its source as [src: \"words copied exactly from that tool result\"] — 3 to 15 "
    "consecutive words on one line, copied verbatim, never paraphrased, and containing the figure you "
    "state. A specific you would present as found in a source but cannot quote from a tool result: "
    "leave it out, or say plainly that you could not verify it. Facts the user told you, facts from your "
    "earlier replies or your memory notes, general knowledge and your own reasoning or arithmetic need "
    "NO marker. Never put a marker inside code or a command. The system removes every [src: …] marker "
    "before the user sees the answer, and checks every quote against the "
    "tool results.\n\n"
)

# a quoted marker may contain "]" (r1 M3: `"version 18.0 [beta] shipped"`); never spans lines
_QUOTED_RE = re.compile(r"\[src:[ \t]*[\"“”]([^\"“”\n]{1,400})[\"“”][ \t]*\]", re.IGNORECASE)
_BARE_RE = re.compile(r"\[src:[ \t]*((?:(?!\[src:)[^\]\n]){1,400})\]", re.IGNORECASE)   # r4 M2: never swallows the next marker
# r4 M2: an unclosed marker ends after its closed quote, or at the next marker / line end
_UNCLOSED_RE = re.compile(r"\[src:[ \t]*(?:\"[^\"\n]*\"|“[^”\n]*”|[^\n\x00\x01\x02]*)", re.IGNORECASE)
_FENCE_RE = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]*`)", re.DOTALL)   # r2 m6: an unclosed fence is code to the end
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")
_NUM_RE = re.compile(r"(?<![^\W\d_])\d+(?:[.,]\d+)*")          # r4: "H1", "Q3", "M4" are names, not figures
_OK, _BAD = "\x00", "\x01"
UNVERIFIED = " (unverified)"
MIN_WORDS = 3
CLAIM_WORDS = 4


def _unescape_json(t: str) -> str:
    if "\\" not in t:
        return t
    t = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), t)
    return t.replace('\\"', '"').replace("\\/", "/").replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", _unescape_json(html.unescape(html.unescape(str(text or "")))))
    t = t.lower()
    t = re.sub(r"[‘’ʼ]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‐-―−]", "-", t)              # dashes and minus
    t = re.sub(r"\[([^\]\n]{0,300})\]\([^)\s]{0,500}\)", r"\1", t)   # markdown link → its text
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                 # 1,234,567 → 1234567
    # markdown marks; "_" only at a word's edge (§4OU r3: "sub_status" stays one key)
    t = re.sub(r"[*`#|]|(?<![^\W_])_|_(?![^\W_])", " ", t)
    # §4OU: "LLM=Active" / "LLM: Active" / "lse:eimi" are one label-value form —
    # after the markdown strip ("**LLM**=Active", r3) and before ">" goes, so
    # ">=" / "=>" / "==" never become a label (r2 M1)
    if "=" in t:
        t = re.sub(r"(?<=[^\W\d_])\s*=(?![=>])\s*(?=[^\W_])", ": ", t)
    if '"' in t:
        t = re.sub(r'"\s*:\s*', '": ', t)                    # r2/r3: compact JSON '"a":"b"' is '"a": "b"'
    t = t.replace(">", " ")
    if ":" in t:
        t = re.sub(r"(?<=[^\W\d_]):(?=[^\W_])", ": ", t)
    # §4OU: a source's "different,i believe" is the quote's "different, i believe"
    if "," in t or ";" in t:
        t = re.sub(r"(?<=[^\W\d_])([,;])(?=[^\W\d_])", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def _pieces(quote: str) -> List[str]:
    out = [_norm(x).strip(" \"'.,;:") for x in _ELLIPSIS_RE.split(str(quote or ""))]
    return [p for p in out if p]


def quote_found(quote: str, sources) -> bool:
    """The quote's ellipsis pieces stand IN ORDER inside ONE source (r1 M5),
    each of ≥ 2 words and ≥ MIN_WORDS words in all, after normalisation."""
    return _locate(quote, sources) is not None


# "llm: active", 'status": "released' — r1: a value of ≥ 4 characters ("q: a" is not one)
_LABEL_RE = re.compile(r"^[^\W\d_][\w .()/-]{0,40}\"?: \"?[^\W_][^\s\"]{3,}")
#: §4OU: a two-word quote this long names something ("connected (anonymous)"); "it is" does not
SHORT_QUOTE_MIN_CHARS = 16


def _identifier(piece: str) -> bool:
    """§4OU: one token that names something exactly — a model, a package, a
    ticker: long, with a digit ("qwen3.6-35b-a3b-uncensored-…"; r1: a
    hyphen alone made "state-of-the-art" a source)."""
    return " " not in piece and len(piece) >= 10 and bool(re.search(r"\d", piece))


_NAMED_PAIR_RE = re.compile(r"^[^\W\d_]{4,} \([^\W\d_]{4,}\)$")


def _named_pair(raw: str, piece: str) -> bool:
    """§4OU: a word and its bracketed qualifier, as a status line prints it
    ("Connected (Anonymous)"). r2: any capitalised pair was too much —
    "Executive Officer", "Customer Service" stand in any text; this form was
    the one real two-word miss in the labelled set."""
    return len(piece) >= SHORT_QUOTE_MIN_CHARS and bool(_NAMED_PAIR_RE.match(piece))


_ID_EDGE = "-._/:+@=~"
_ID_EDGE_LEFT = "-._+@=~"       # "owner/model" and "pkg:name" keep a whole name after the separator


def _find_identifier(src: str, piece: str, at: int = 0) -> int:
    """`_find_word` for one identifier token, which must not be part of a
    longer one (r1 M2: "qwen3-coder-30b" inside "qwen3-coder-30b-a3b-instruct")."""
    while True:
        k = _find_word(src, piece, at)
        if k < 0:
            return -1
        end = k + len(piece)
        before = k >= 2 and src[k - 1] in _ID_EDGE_LEFT and src[k - 2].isalnum()
        after = end + 1 < len(src) and src[end] in _ID_EDGE and src[end + 1].isalnum()
        if not before and not after:
            return k
        at = k + 1


def _label_rows(src: str, label: str) -> int:
    """How many rows of ``src`` give ``label`` a value with no figure (§4OU r3:
    "Memory: 45.1% used" is a different row from "Memory=Active"; a figure row
    is the figure rule's business). A label that is the tail of a longer one
    ("sub status:", "build-status:") is not the same label."""
    n = 0
    for m in re.finditer(re.escape(label) + r'\s*"?([^\s",;]*)', src):
        k = m.start() - 1
        if k >= 0 and (src[k].isalnum() or src[k] in "-_"):
            continue                                        # the tail of a longer key: "sub_status:"
        if not re.search(r"\d", m.group(1)):
            n += 1
    return n


def _locate(quote: str, sources):
    """``(source index, start, end)`` of the quote in the sources, or None.
    ``start``/``end`` span the whole source when only a near match found it."""
    if isinstance(sources, str):
        sources = [sources]
    # several quotes in one marker (`"Temp: 19.0°C", "Humidity: 83%"`, pilot): each must stand.
    # §4OU: NOT a JSON pair — `"status": "RELEASED"` is one quote
    qs = str(quote or "")
    found = list(re.finditer(r"[\"“”]([^\"“”]{2,400})[\"“”]", qs))
    seps = [qs[a.end():b.start()] for a, b in zip(found, found[1:])]
    if len(found) >= 2 and not any(":" in sp for sp in seps):
        locs = [_locate(m.group(1), sources) for m in found]
        if any(lc is None for lc in locs):
            return None
        return locs[-1]
    pieces = _pieces(quote)
    if not pieces:
        return None
    if any(len(p.split()) < 2 for p in pieces) and not (len(pieces) == 1 and _identifier(pieces[0])):
        return None                                         # "September" is not a source
    if (sum(len(p.split()) for p in pieces) < MIN_WORDS and not any(re.search(r"\d", p) for p in pieces)
            and not (len(pieces) == 1 and (_identifier(pieces[0]) or _named_pair(quote, pieces[0])
                                           or (_LABEL_RE.match(pieces[0]) and len(pieces[0]) >= 10)))):
        return None                                         # "it is" is not; "temp: 19.0°c", "llm: active" are
    single_id = len(pieces) == 1 and " " not in pieces[0]
    # r2 M2: a label pair with no figure must not back another row's value —
    # "status: active" proves nothing where several things have a status
    label = (pieces[0].split(":", 1)[0] + ":"
             if len(pieces) == 1 and _LABEL_RE.match(pieces[0]) and not re.search(r"\d", pieces[0]) else "")
    for idx, src in enumerate(sources):
        at, first = 0, -1
        for p in pieces:
            k = _find_identifier(src, p, at) if single_id else _find_word(src, p, at)
            if k < 0:
                break
            first = k if first < 0 else first
            at = k + len(p)
        else:
            if label and _label_rows(src, label) > 1:
                continue
            return (idx, first, at)
    budget = _NEAR_BUDGET.get()
    if budget is not None:
        if budget[0] <= 0:
            return None                                    # r3 M7: near-match work per reply is capped
        budget[0] -= 1
    q = _TOKEN_RE.findall(_norm(quote))
    if len(q) < NEAR_MIN_TOKENS:
        return None
    figs = [t for t in q if any(ch.isdigit() for ch in t)]
    longs = [t for t in q if len(t) >= 4 and t not in figs]
    for idx, src in enumerate(sources):
        # cheap prefilter in C before any tokenising: figures are exact, so they must stand;
        # at least half the longer words too (a typo costs one)
        if any(f not in src for f in figs) or (longs and sum(t in src for t in longs) * 2 < len(longs)):
            continue
        if budget is not None:
            if budget[1] < len(src):
                return None
            budget[1] -= len(src)
        if _near_found(quote, src):
            return (idx, 0, len(src))
    return None


def marker_ok(quote: str, claim: str, sources) -> Tuple[bool, bool]:
    """``(quote found, claim's figures in the quote)``. The figure rule stays
    strict: measured in §4OU, letting the figure stand near the quote cut the
    false marks (20 → 6 of 31) but accepted 26 more of 101 claims with a
    figure changed — an operator decision, not a default."""
    if _locate(quote, sources) is None:
        return False, False
    return True, not (_figures(claim) - _figures(quote))


def _find_word(src: str, piece: str, at: int = 0) -> int:
    """`src.find` at word edges (r3 M2: "79" stood inside "799", "Version 3.1"
    inside "3.12")."""
    while True:
        k = src.find(piece, at)
        if k < 0:
            return -1
        end = k + len(piece)
        if (k == 0 or not src[k - 1].isalnum() or not piece[0].isalnum()) and \
           (end >= len(src) or not src[end].isalnum() or not piece[-1].isalnum()):
            return k
        at = k + 1


# ── near-verbatim (§4OJ-b, r2) ─────────────────────────────────────────────── #
# 37 of 47 pilot flags were TRUE facts the model copied inexactly (a merged Greek
# word "χτυπήθηκαναπό", a typo). r2 M1/M2: a word-SHARE rule let a dropped "not",
# feet→meters, year→month, CEO→CFO and another row's figure pass. Now the quote
# must ALIGN word by word with one contiguous stretch of the source: each word
# equal, or a one-letter typo of a word of ≥ 6 letters (r3), or a plural, or two
# source words merged into one — never a word added, dropped or replaced, figures exact.
NEAR_MIN_TOKENS = 5
NEAR_MAX_PER_REPLY = 12            # r3 M7: near-match attempts per reply
NEAR_MAX_CHARS_PER_REPLY = 2_000_000   # …and source characters tokenised per reply
_NEAR_BUDGET: "contextvars.ContextVar" = contextvars.ContextVar("quote_near_budget", default=None)
NEAR_MAX_SOURCE = 400_000          # r2 m11: no index over megabyte outputs
_TOKEN_RE = re.compile(r"[^\W_]+(?:[.,:/][^\W_]+)*")


@functools.lru_cache(maxsize=16)
def _src_index(src: str):
    toks = _TOKEN_RE.findall(src)
    index: dict = {}
    for i, t in enumerate(toks):
        index.setdefault(t, []).append(i)
    return toks, index


def _near(a: str, b: str) -> bool:
    if a == b:
        return True
    if any(ch.isdigit() for ch in a + b):
        return False                                       # figures are never "near" each other
    if a + "s" == b or b + "s" == a:
        return True                                        # plural
    # r3 M1: a 0.85 similarity let million/billion, likely/unlikely, Austria/Australia
    # pass. Only ONE letter may differ, inside a word of ≥ 6 letters, after the same
    # first two letters (a prefix like un-/in- is never a typo).
    if min(len(a), len(b)) < 6 or a[:2] != b[:2] or abs(len(a) - len(b)) > 1:
        return False
    return _edit_distance_at_most_one(a, b)


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _aligned_at(q: list, toks: list, j: int) -> bool:
    i = 0
    n = len(toks)
    while i < len(q):
        if j >= n:
            return False
        if _near(q[i], toks[j]):
            i += 1; j += 1
        elif j + 1 < n and toks[j] + toks[j + 1] == q[i]:
            i += 1; j += 2                                 # "χτυπήθηκαναπό" = "χτυπήθηκαν" + "από"
        elif i + 1 < len(q) and q[i] + q[i + 1] == toks[j]:
            i += 2; j += 1
        else:
            return False
    return True


def _near_found(quote: str, src: str) -> bool:
    """The quote aligns word by word with one stretch of the source (see above)."""
    if len(src) > NEAR_MAX_SOURCE:
        return False
    q = _TOKEN_RE.findall(_norm(quote))
    if len(q) < NEAR_MIN_TOKENS:
        return False                                      # short quotes: exact only
    toks, index = _src_index(src)
    for k, t in enumerate(q[:4]):
        for pos in index.get(t, [])[:300]:
            for start in range(max(0, pos - k - 1), pos - k + 2):
                if _aligned_at(q, toks, start):
                    return True
    return False


_MONTHS = {m: i for i, m in enumerate(("january february march april may june july august september "
                                        "october november december").split(), 1)}
_MONTH_ALT = "|".join(sorted(list(_MONTHS) + ["sept"] + [m[:3] for m in _MONTHS], key=len, reverse=True))
# a month NAME counts only beside a day number ("you may see 4" is not May)
_MONTH_RE = re.compile(r"\b(" + _MONTH_ALT + r")\.?(?=\s+\d{1,2}\b)|(?<=\b\d)\s*(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\b"
                       r"|(?<=\b\d\d)\s*(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\b")


# whole month forms (r1: stems made "5 μάρτυρες" March and "2 μαιευτήρες" May)
_GREEK_MONTHS = ["ιανουαρ(?:ιου|ιος)", "φεβρουαρ(?:ιου|ιος)", "μαρτ(?:ιου|ιος|η)", "απριλ(?:ιου|ιος|η)",
                 "μα(?:ιου|ιος|η)", "ιουν(?:ιου|ιος|η)", "ιουλ(?:ιου|ιος|η)", "αυγουστ(?:ου|ος)",
                 "σεπτεμβρ(?:ιου|ιος|η)", "οκτωβρ(?:ιου|ιος|η)", "νοεμβρ(?:ιου|ιος|η)", "δεκεμβρ(?:ιου|ιος|η)"]
_GREEK_HINT = re.compile(r"[\u0370-\u03ff]")
# a month NAME counts only after a day number, as in English
_GREEK_MONTH_RE = re.compile(r"(?<![\d.])\d{1,2}(?:η|ης)?\s+(" + "|".join(f"({m})" for m in _GREEK_MONTHS)
                             + r")(?![^\W\d_])")


def _figures(text: str) -> set:
    """The figures of a text, leading zeros dropped, and a month NAME counted
    as its number — so "2026-09-24" and "September 24, 2026" carry the same
    figures (live probe 2026-10-13: a true date was flagged)."""
    t = _norm(text)
    out = {n.replace(",", "").lstrip("0") or "0" for n in _NUM_RE.findall(t)}
    for m in _MONTH_RE.finditer(t):
        name = next(g for g in m.groups() if g)
        out.add(str(_MONTHS.get(name) or next(i for k, i in _MONTHS.items() if k.startswith(name[:3]))))
    if _GREEK_HINT.search(t):                     # §4OU: "1η Νοεμβρίου 2025", "6 Οκτωβρίου"
        flat = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
        for m in _GREEK_MONTH_RE.finditer(flat):
            out.add(str(next(i for i in range(12) if m.group(i + 2)) + 1))
    return out


def _claim_before(text: str, end: int) -> str:
    """The clause a marker backs: back to the previous marker, sentence end or line."""
    seg = text[:end].rstrip(" \t\n.!?;:")          # r3 M3: "…5.1 billion. [src: …]" backs the sentence before
    cut = max(seg.rfind(_OK), seg.rfind(_BAD), seg.rfind("\n"), seg.rfind(". "), seg.rfind("! "), seg.rfind("? "))
    # the figures NEXT to the marker are what it backs ("PostgreSQL 18 came out on September 25,
    # 2025 [src]": the 18 names the subject, the date is the claim)
    words = [w for w in seg[cut + 1:].split() if re.search(r"[^\W_]", w)]   # r4: "/", "-" are not words
    return " ".join(words[-CLAIM_WORDS:])


def evidence_sources(contents: Iterable) -> List[str]:
    return [n for n in (_norm(c) for c in contents) if n]


def evidence_text(contents: Iterable) -> List[str]:          # kept name for the wiring
    return evidence_sources(contents)


def _finish(done: str) -> str:
    """Replace the sentinels, in one linear pass (r2 m10): whitespace before a
    marker goes, "( marker )" leaves no "()", a failed marker on its own line
    joins the line above, a run of markers with any failure is one
    "(unverified)"."""
    parts = re.split("([" + _OK + _BAD + "])", done)
    out: List[str] = []
    i = 0
    while i < len(parts):
        piece = parts[i]
        if piece not in (_OK, _BAD):
            out.append(piece)
            i += 1
            continue
        bad = False
        while i < len(parts) and (parts[i] in (_OK, _BAD) or (parts[i].strip(" \t") == "" and i + 1 < len(parts)
                                                              and parts[i + 1] in (_OK, _BAD))):
            bad = bad or parts[i] == _BAD
            i += 1
        bad = bad and marks_shown()          # §4OU: hidden, a failed marker goes like a found one
        prev = out.pop() if out else ""
        prev = prev.rstrip(" \t")
        nxt = parts[i] if i < len(parts) else ""
        if prev.endswith("(") and nxt.lstrip(" \t").startswith(")"):
            prev = prev[:-1].rstrip(" \t")
            nxt = nxt.lstrip(" \t")[1:]
        if bad and prev.endswith("\n"):
            prev = prev.rstrip("\n").rstrip(" \t")
        elif not bad and (prev.endswith("\n") or not prev) and nxt.startswith("\n"):
            nxt = nxt[1:]                                  # a verified marker on its own line leaves no blank line
        out.append(prev + (UNVERIFIED if bad else ""))
        if i < len(parts):
            parts[i] = nxt
        else:
            out.append(nxt)
    return "".join(out)


_PH_RE = re.compile("\x02(\\d+)\x03")


def _mask_markers(text: str):
    """(masked text, markers): every closed marker is replaced by a
    placeholder BEFORE code spans are found (r3 M5: a backtick inside a quote —
    `run \`make install\` and` — split the marker and leaked its tail)."""
    found = list(_QUOTED_RE.finditer(text))
    taken = [(m.start(), m.end()) for m in found]
    for m in _BARE_RE.finditer(text):
        if not any(m.start() < b and a < m.end() for a, b in taken):     # r4 M2: no overlap at all
            found.append(m)
            taken.append((m.start(), m.end()))
    found.sort(key=lambda m: m.start())
    out, pos, marks = [], 0, []
    for m in found:
        out.append(text[pos:m.start()])
        out.append(f"\x02{len(marks)}\x03")
        marks.append(m)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), marks


def apply_quotes(reply: str, sources) -> Tuple[str, dict]:
    """(reply with markers removed, stats). Code spans are left untouched.
    A failed quote — not in any source, or not carrying the figures of the
    clause it backs (r1 M4) — leaves "(unverified)" where the marker was, when
    the marks are shown (§4OU); hidden, it goes like a found one and is counted."""
    if isinstance(sources, str):
        sources = [sources]
    stats = {"quotes": 0, "found": 0, "missing": [], "figure_mismatch": 0}
    text = str(reply or "")
    if "[src:" not in text.lower() or "\x02" in text or _OK in text or _BAD in text:
        return (strip_leftover_markers(text) if "[src:" in text.lower() else text), stats
    masked, marks = _mask_markers(text)
    # r4 M1: the clause a marker backs is read on the WHOLE text — a figure in inline
    # code right before it (`$69.99` [src: …$59.99…]) belongs to the claim too
    claim_of = {}
    for ph in _PH_RE.finditer(masked):
        before = _PH_RE.sub(_OK, masked[:ph.start()])          # backticks: _figures normalises them away
        claim_of[int(ph.group(1))] = _claim_before(before, len(before))
    tok = _NEAR_BUDGET.set([NEAR_MAX_PER_REPLY, NEAR_MAX_CHARS_PER_REPLY])
    try:
        parts = _FENCE_RE.split(masked)
        out_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:                                    # code: markers inside it stay verbatim
                out_parts.append(_PH_RE.sub(lambda m: marks[int(m.group(1))].group(0), part))
                continue
            buf: List[str] = []
            pos = 0
            for ph in _PH_RE.finditer(part):
                buf.append(part[pos:ph.start()])
                stats["quotes"] += 1
                q = marks[int(ph.group(1))].group(1)
                found, figs_ok = marker_ok(q, claim_of.get(int(ph.group(1)), ""), sources)
                ok = found and figs_ok
                if found and not figs_ok:
                    stats["figure_mismatch"] += 1
                if ok:
                    stats["found"] += 1
                    buf.append(_OK)
                else:
                    stats["missing"].append(q[:120])
                    buf.append(_BAD)
                pos = ph.end()
            buf.append(part[pos:])
            done = "".join(buf)
            n_unclosed = len(_UNCLOSED_RE.findall(done))
            if n_unclosed:
                stats["quotes"] += n_unclosed
                stats["missing"].extend(["(unclosed marker)"] * n_unclosed)
                done = _UNCLOSED_RE.sub(_BAD, done)
            out_parts.append(_finish(done))
        return "".join(out_parts), stats
    finally:
        _NEAR_BUDGET.reset(tok)


def tool_contents(tools_run: Iterable, messages: Iterable = ()) -> List[str]:
    """What a quote may rest on: this turn's tool results and every tool
    message already in the conversation. Not the user's words (r2 M4: "Is
    it $59.99?" certified a false premise) — the rule tells the model that
    facts the user gave need no marker. Synthetic rows (parse errors,
    rejected calls) are the loop's words, never a source."""
    out: List[str] = []
    tools_run = list(tools_run or ())
    # a synthetic row is copied into `messages` WITHOUT its flag (r1 m8) — exclude it by content
    seen = {str(t.get("content") or "") for t in tools_run if isinstance(t, dict) and t.get("_synthetic")}

    def _add(c):
        c = str(c or "")
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    for t in tools_run:
        if isinstance(t, dict):
            if not t.get("_synthetic"):
                _add(t.get("content"))
        else:
            _add(getattr(t, "content", ""))
    for m in messages or ():
        if not isinstance(m, dict) or m.get("_synthetic"):
            continue
        if m.get("role") == "tool":                 # r2 M4: the user's question is not evidence
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(x.get("text") or "") for x in c if isinstance(x, dict))
            _add(c)
    return out


def strip_leftover_markers(text):
    """Safety net (§4OJ-b): a reply that left the turn without the check (the
    system's own reply, a cancelled turn's partial, a failed check) must never
    show `[src: …]` to the user. Removes the markers, marks nothing, never
    touches code spans."""
    if not isinstance(text, str) or "[src:" not in text.lower():
        return text
    clean = text.replace("\x02", "").replace("\x03", "").replace(_OK, "").replace(_BAD, "")
    masked, marks = _mask_markers(clean)
    parts = _FENCE_RE.split(masked)
    for i in range(len(parts)):
        if i % 2 == 1:
            parts[i] = _PH_RE.sub(lambda m: marks[int(m.group(1))].group(0), parts[i])
        else:
            parts[i] = _finish(_UNCLOSED_RE.sub(_OK, _PH_RE.sub(_OK, parts[i])))
    return "".join(parts)


def check_reply(reply: str, tools_run, messages) -> Tuple[str, dict]:
    """The whole check, for `asyncio.to_thread` (r4: normalising the sources
    ran on the event loop)."""
    return apply_quotes(reply, evidence_text(tool_contents(tools_run, messages)))


# ── streamed replies (§4OJ-c) ─────────────────────────────────────────────── #

_SRC_HEAD = "[src:"
_QUOTE_OPEN = {'"': '"', "“": "”", "”": "”", "'": "'"}
MARKER_MAX = 450


class MarkerFilter:
    """§4OJ-c r5/r6: the streamed view as a STATE MACHINE that only ever
    appends (r5 CRIT: a regex view recomputed over the whole text shrank and
    the frame filter dropped the rest of the reply). Text passes through;
    what could still become a marker — trailing spaces, a "(", a "[" that may
    grow into `[src:` — is held; code (inline `…`, fenced ```…```) passes
    untouched. A marker ends at `"…"]` — a quote mark closes the quote only
    when "]" follows it (r6 M1: `"the CEO said “we grew” last year"`) — or is
    abandoned at a newline, at MARKER_MAX, or when a new `[src:` starts; an
    abandoned marker whose quote had closed gives back the text after that
    quote. Every marker is RECORDED (quote, where it stood in the output,
    abandoned?) so the closing line checks exactly what was hidden (r6 M3).
    `strip_view(text, final=True)` is this machine run to the end."""

    def __init__(self) -> None:
        self.out: List[str] = []
        self.n_out = 0
        self.state = "text"            # text | maybe | marker
        self.buf = ""
        self.pending = ""              # held: [ \t]* "("? [ \t]*
        self.ticks = 0
        self.inline = False
        self.fence = False
        self.after_paren = None        # spaces held after a dropped "( marker" awaiting ")"
        self.quoted = None             # None: undecided; True: `"…"`; False: bare
        self.last_close = -1           # index in buf of the last quote mark after the opener
        self.paren_lead = ""
        self.closers = ('"', "”", "“")
        self.line_has_text = False     # r8: a fence opens only at the start of a line
        self.ticks_at_start = False
        self.markers: List[Tuple[str, int, bool]] = []

    # ── output ─────────────────────────────────────────────────────────
    def _emit(self, t: str) -> None:
        if t:
            if not self.n_out:
                t = t.lstrip(" \t\r\n")                           # r6/r8: a reply opening with a marker
                if not t:
                    return
            self.out.append(t)
            self.n_out += len(t)

    def _flush_pending(self) -> None:
        self._emit(self.pending)
        self.pending = ""

    def _ticks_done(self) -> None:
        if self.ticks >= 3 and (self.ticks_at_start or self.fence):
            self.fence = not self.fence
            self.inline = False
        elif self.ticks and not self.fence:
            self.inline = not self.inline
        self.ticks = 0

    # ── markers ────────────────────────────────────────────────────────
    def _quote_of(self, body: str) -> str:
        if self.quoted and self.last_close >= 0:               # between the opener and the last quote mark
            head = body[len(_SRC_HEAD):]
            start = len(_SRC_HEAD) + (len(head) - len(head.lstrip(" \t"))) + 1
            return body[start:self.last_close]
        q = body[len(_SRC_HEAD):].strip(" \t")
        if q.endswith("]"):
            q = q[:-1].rstrip(" \t")
        if q[:1] in _QUOTE_OPEN:
            q = q[1:]
        if q[-1:] in ('"', "”", "'", "“"):
            q = q[:-1]
        return q

    def _drop(self, body: str, abandoned: bool) -> None:
        self.markers.append((self._quote_of(body), self.n_out, abandoned))
        had_paren = "(" in self.pending
        self.paren_lead = self.pending.split("(")[0] if had_paren else ""   # spaces before a kept "("
        self.pending, self.buf, self.state = "", "", "text"
        self.quoted, self.last_close = None, -1
        self.after_paren = "" if had_paren else None

    def _abandon(self, then: str = "") -> None:
        """End an unfinished marker; replay what follows its closed quote."""
        buf = self.buf
        if self.quoted and self.last_close >= 0:
            body, rest = buf[:self.last_close + 1], buf[self.last_close + 1:]
            self._drop(body, abandoned=False)                      # a closed quote: check it
        else:
            body, rest = buf, ""
            self._drop(body, abandoned=True)
        for ch in rest + then:
            self._feed_char(ch)

    # ── characters ─────────────────────────────────────────────────────
    def _text(self, c: str) -> None:
        if c == "`":
            self._flush_pending()
            if self.after_paren is not None:
                self._emit(self.paren_lead + "(" + self.after_paren); self.after_paren = None
            if not self.ticks:
                self.ticks_at_start = not self.line_has_text
            self.ticks += 1
            self.line_has_text = True
            self._emit(c)
            return
        if self.ticks:
            self._ticks_done()
        if c == "\n":
            self.line_has_text = False
        elif c not in " \t":
            self.line_has_text = True
        if self.fence:                                         # r7: only a fence protects; a lone "`"
            self._emit(c)                                      # (Don`t) must never show a raw marker
            return
        if self.after_paren is not None:
            if c in " \t":
                self.after_paren += c
                return
            if c == ")":
                self.after_paren = None                            # "( marker )" leaves nothing
                return
            self._emit(self.paren_lead + "(" + self.after_paren); self.after_paren = None   # r6: the "(" stays
        if c in " \t":
            self.pending += c
            return
        if c == "(":
            if "(" in self.pending:
                self._flush_pending()
            self.pending += c
            return
        if c == "[":
            self.state, self.buf = "maybe", "["
            return
        self._flush_pending()
        self._emit(c)

    def _feed_char(self, c: str) -> None:
        if self.state == "text":
            self._text(c)
            return
        if self.state == "maybe":
            self.buf += c
            low = self.buf.lower()
            if low == _SRC_HEAD:
                self.state, self.quoted, self.last_close = "marker", None, -1
            elif not _SRC_HEAD.startswith(low):
                held, self.buf, self.state = self.buf, "", "text"
                self._flush_pending()
                self._emit("[")
                for ch in held[1:]:
                    self._feed_char(ch)
            return
        # marker
        if c == "\n":
            self._abandon("\n")
            return
        self.buf += c
        if len(self.buf) > MARKER_MAX:
            self._abandon()
            return
        if len(self.buf) > len(_SRC_HEAD) and self.buf.lower().endswith(_SRC_HEAD):
            self.buf = self.buf[:-len(_SRC_HEAD)]                  # a new marker starts: the old one ends
            pend = self.pending
            self._abandon()
            self.pending = pend if not self.pending else self.pending
            self.state, self.buf, self.quoted, self.last_close = "marker", _SRC_HEAD, None, -1
            return
        if self.quoted is None:
            body = self.buf[len(_SRC_HEAD):].lstrip(" \t")
            if not body:
                return
            self.quoted = body[0] in _QUOTE_OPEN
            self.closers = ("'",) if body[0] == "'" else ('"', "”", "“")
            return
        if not self.quoted:
            if c == "]":
                self._drop(self.buf, abandoned=False)
            return
        if c in self.closers:                                  # r8: only the opener's kind closes —
            self.last_close = len(self.buf) - 1                # an apostrophe never ends a "…" quote
            return
        if c == "]" and self.last_close >= 0:
            self._drop(self.buf, abandoned=False)                  # `"…"]` or `"Title" p. 3]` — complete

    def feed(self, chunk: str) -> str:
        n = len(self.out)
        for c in str(chunk or ""):
            self._feed_char(c)
        return "".join(self.out[n:])

    def finish(self) -> str:
        n = len(self.out)
        if self.state == "marker":
            self._abandon()
        if self.ticks:
            self._ticks_done()
        if self.state == "maybe":
            self._flush_pending(); self._emit(self.buf)
        elif self.state == "marker":
            self._drop(self.buf, abandoned=True)
        else:
            self._flush_pending()
        if self.after_paren is not None:
            self._emit(self.paren_lead + "(" + self.after_paren)
        self.pending, self.buf, self.state, self.after_paren = "", "", "text", None
        return "".join(self.out[n:])

    def fence_after_ticks(self) -> bool:
        """Whether a fence is open once the backticks still being counted
        count (r8: a reply ending exactly on ``` had not closed its fence yet)."""
        if self.ticks >= 3 and (self.ticks_at_start or self.fence):
            return not self.fence
        return self.fence

    def text(self) -> str:
        if len(self.out) > 64:
            self.out = ["".join(self.out)]                          # r6: keep text() cheap
        return "".join(self.out)

    def failed_claims(self, sources) -> List[str]:
        """The clauses whose marker failed: abandoned, quote not in the
        sources, or the figures beside it not in the quote."""
        text = self.text()
        cuts = [pos for _, pos, _ in self.markers]
        out: List[str] = []
        tok = _NEAR_BUDGET.set([NEAR_MAX_PER_REPLY, NEAR_MAX_CHARS_PER_REPLY])
        try:
            for k, (q, pos, abandoned) in enumerate(self.markers):
                prev = cuts[k - 1] if k and cuts[k - 1] <= pos else 0
                own = text[prev:pos]                               # r7 M1: from the previous marker on
                ok = not abandoned and all(marker_ok(q, _claim_before(own, len(own)), sources))
                if not ok:
                    clause = _plain_clause(own)
                    if clause and clause not in out:
                        out.append(clause)
        finally:
            _NEAR_BUDGET.reset(tok)
        return out


def _plain_clause(seg: str) -> str:
    seg = seg.rstrip(" \t\n.!?;:,(")
    cut = max(seg.rfind("\n"), seg.rfind(". "), seg.rfind("! "), seg.rfind("? "), seg.rfind("; "), seg.rfind("|"))
    clause = seg[cut + 1:].replace("**", "").replace("__", "").strip(" \t#>-•(,")
    clause = re.sub(r"^\[[ xX]\]\s*", "", clause)
    clause = clause.replace("`", "").replace("_", " ").strip()
    return clause if len(clause) <= 90 else "…" + clause[-88:].lstrip()


def run_filter(text: str) -> "MarkerFilter":
    f = MarkerFilter()
    f.feed(text)
    f.finish()
    return f


def strip_view(text: str, final: bool = False) -> str:
    """The reply as the user may see it (see `MarkerFilter`). ``final`` False:
    what may be shown while the text is still growing."""
    f = MarkerFilter()
    f.feed(text)
    if final:
        f.finish()
    return f.text()


def failed_claims(reply: str, sources) -> List[str]:
    """The clauses whose quote failed, in order — from the same machine that
    hid the markers (r6 M3: abandoned markers were hidden but never listed)."""
    return run_filter(str(reply or "")).failed_claims(sources)


def _clause_before(masked_prefix: str, marks) -> str:
    seg = _PH_RE.sub(_OK, masked_prefix).rstrip(" \t\n.!?;:,(")
    cut = max(seg.rfind(_OK), seg.rfind("\n"), seg.rfind(". "), seg.rfind("! "), seg.rfind("? "),
              seg.rfind("; "), seg.rfind("|"))                 # r5: an earlier marker or table cell ends it
    clause = seg[cut + 1:].replace("**", "").replace("__", "").strip(" \t#>-•(")
    clause = re.sub(r"^\[[ xX]\]\s*", "", clause)
    return clause if len(clause) <= 90 else "…" + clause[-88:].lstrip()


def closing_line(claims: List[str]) -> str:
    if not claims or not marks_shown():       # §4OU: hidden by default; the count is still logged
        return ""
    shown = "; ".join(f"“{c}”" for c in claims[:5]) + (f" and {len(claims) - 5} more" if len(claims) > 5 else "")
    return f"\n\n_Could not verify against the sources read: {shown}._"


def check_reply_append(reply: str, tools_run, messages) -> Tuple[str, dict]:
    """The streamed form (for finished-string replies of a streamed request):
    the view of the reply plus the closing line — it extends what the tap
    already showed, so nothing is retracted."""
    f = MarkerFilter()
    f.feed(reply)
    in_fence = f.fence_after_ticks()
    f.finish()
    claims = f.failed_claims(evidence_text(tool_contents(tools_run, messages)))
    line = closing_line(claims)
    return f.text() + ("\n```" if line and in_fence else "") + line, {"failed": len(claims), "markers": len(f.markers)}


async def filter_sse(gen, tools_run, messages, on_result=None):
    """Wrap a streamed final's frames (§4OJ-c r5/r6): every content delta goes
    through one `MarkerFilter`, so what is sent only ever grows. Finish frames
    are held and sent after the rest and the closing line, just before
    ``[DONE]`` (r6: content after a finish frame used to land after the
    closing line). An orphan ``</think>`` restarts the machine — what came
    before is reasoning the client hides (r6 M2). Stream-prefix frames pass
    through; a retract restarts the filter. A stream cut before ``[DONE]``
    gets the held rest but no closing line."""
    filt, closed = MarkerFilter(), False
    held_finish: List[str] = []

    def _content_event(template: dict, text: str) -> str:
        obj = {k: v for k, v in template.items() if k not in ("choices", "ghost", "usage")}
        obj["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        return "data: " + json.dumps(obj)

    def _close(template: dict, with_line: bool) -> List[str]:
        in_fence = filt.fence_after_ticks()
        rest = filt.finish()
        if with_line:
            try:
                claims = filt.failed_claims(evidence_text(tool_contents(tools_run, messages)))
                line = closing_line(claims)
                try:
                    from ..utils.logging import pretty_log, Icons
                    pretty_log("Quote Bound", f"streamed final · {len(filt.markers)} marker(s) · "
                                              f"{len(claims)} claim(s) not verified"
                                              + ("" if marks_shown() or not claims else " · marks hidden"),
                                              icon=Icons.BRAIN_CTX, durable_only=True)
                except Exception:  # noqa: BLE001
                    pass
                if on_result is not None:                     # §4PL: the census's quote row
                    try:
                        on_result({"markers": len(filt.markers), "failed": len(claims)})
                    except Exception:  # noqa: BLE001
                        pass
                rest += ("\n```" if line and in_fence else "") + line   # r7: never inside an open fence
            except Exception:  # noqa: BLE001 — never kill the stream over the closing line
                logger.warning("quote-bound closing line skipped", exc_info=True)
        return [_content_event(template, rest)] if rest else []

    def _feed(text: str) -> str:
        nonlocal filt
        k = text.rfind("</think>")
        if k < 0 or filt.fence:                                    # r7: a "</think>" inside code is code
            return filt.feed(text)
        out = filt.feed(text[:k + len("</think>")]) + filt.finish()
        filt = MarkerFilter()                                       # the answer starts here
        filt.n_out = 1                                              # not the reply's head: keep its spaces
        return out + filt.feed(text[k + len("</think>"):])

    template = {"object": "chat.completion.chunk"}
    async for chunk in gen:
        body = bytes(chunk).decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        out_events: List[str] = []
        for ev in body.split("\n\n"):
            if not ev.strip():
                continue
            if not ev.startswith("data:"):
                out_events.append(ev)
                continue
            payload = ev[5:].strip()
            if payload == "[DONE]":
                if not closed:
                    out_events.extend(_close(template, with_line=True))
                    closed = True
                out_events.extend(held_finish); held_finish = []
                out_events.append(ev)
                continue
            try:
                obj = json.loads(payload)
            except Exception:  # noqa: BLE001
                out_events.append(ev)
                continue
            if not isinstance(obj, dict):
                out_events.append(ev)
                continue
            for k in ("id", "created", "model"):
                if k in obj:
                    template[k] = obj[k]
            ghost = obj.get("ghost") or {}
            if ghost.get("retract"):
                filt = MarkerFilter()
                out_events.append(ev)
                continue
            chs = obj.get("choices")
            ch = chs[0] if isinstance(chs, list) and chs and isinstance(chs[0], dict) else {}
            delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else None
            content = delta.get("content") if delta is not None else None
            finishing = bool(ch.get("finish_reason"))
            if isinstance(content, str) and not ghost.get("stream_prefix") and not closed:
                shown = _feed(content)
                if finishing:
                    if shown:
                        out_events.append(_content_event(template, shown))
                    delta["content"] = ""
                    held_finish.append("data: " + json.dumps(obj))
                    continue
                delta["content"] = shown
                if shown or len(delta) > 1 or len(obj) > 5:
                    out_events.append("data: " + json.dumps(obj))
                continue
            if (finishing or ("usage" in obj and not chs)) and not closed:
                held_finish.append(ev)                             # r7: a usage frame waits with the finish
                continue
            out_events.append(ev)
        if out_events:
            yield ("\n\n".join(out_events) + "\n\n").encode("utf-8")
    if not closed:
        tail = _close(template, with_line=False) + held_finish
        if tail:
            yield ("\n\n".join(tail) + "\n\n").encode("utf-8")
