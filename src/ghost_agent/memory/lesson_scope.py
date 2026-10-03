"""§4KW — a lesson's SCOPE: does it apply to a class of situations, or to one
request?

Reflection, the post-mortem engine and the journal post-mortem keyed their
lessons on the failed request's own text. Retrieval embeds the WHOLE lesson
(situation + mistake + solution), so a case-specific plan whose fix reads like
generic advice matched almost any request: on the recorded traffic, 40 such
lessons were injected 927 times and the request was the lesson's own (or a
rewording of it) in a few dozen. "1. Directly answer 'Yes' to the user's
question regarding knowledge of Professor Spiros Denaxas" reached "what is my
name?" — 117 injections of that one lesson.

Two kinds of lesson, then:

* ``general`` — the trigger names a SITUATION CLASS and the fix is a rule that
  transfers ("When asked whether you know a named person, search before
  answering"). Retrieved by relevance, as before.
* ``request`` — a corrected plan for ONE request. Useful only when that
  request comes back (a retry, a rewording). Retrieved only when the new
  request IS that request (``same_request``).

``same_request`` is the same CONTENT WORDS (see the constants below). The
first rule (character similarity >= 0.85, or content-word containment) was
calibrated on the 927 injections and still matched different tasks across the
corpus ("start the chess service" ~ "stop …", "my name is vasilis" ~ "what is
my name", "Redo this" ~ every redo) — fresh review. On all 1,608 recorded
requests the strict rule matches 180 pairs; every typo-only match was read.
"""
from __future__ import annotations

import contextvars
import difflib
import re

#: the user's request for THIS turn (set by the turn loop); retrieval judges a
#: request-scoped lesson against it when no explicit request is passed
current_request = contextvars.ContextVar("lesson_scope_current_request", default="")

SCOPE_GENERAL = "general"
SCOPE_REQUEST = "request"

#: (fresh review, §4KW) A request-scoped lesson is admitted only when the new
#: request carries the SAME CONTENT WORDS — order, punctuation, accents,
#: function words and per-word typos aside. The first rule (character ratio
#: ≥ 0.85, or containment) matched different tasks on 531 corpus pairs:
#: "start the chess service" ~ "stop the chess service", "my name is vasilis"
#: ~ "what is my name", "Redo this" ~ every "redo …". Fail-closed by design:
#: a reworded retry that adds a word is not admitted.
#: per-word similarity for a typo ("projetcs" ~ "projects")
TYPO_RATIO = 0.86

# Function words only (English + Greek, accent-folded) — content words such as
# "old", "new", "show" carry the request's meaning and must stay.
_FUNCTION_WORDS = frozenset("""
a an the and or but if then than so of in on at by for with into onto about as is are was were be been
being am do does did done have has had having i me my mine you your yours we us our he him his she her it its
they them their these those there here which whom whose can could would should will shall may might must
please just also too very really now again let lets tell give say thanks thank ok okay hi hello
ο η το οι τα του της των τον την τη τις σε στο στη στην στον στα στους στις με για και θα
ειναι ηταν εγω εσυ μου σου μας σας τους ενα μια ενας ας τωρα παλι
""".split())
# Third review: negation, quantity, direction and question words carry the
# request's meaning ("do NOT restart" ≠ "restart", "stop ALL services" ≠ "stop
# this service", "celsius TO fahrenheit" ≠ the reverse, "WHY is it up" ≠ "is it
# up") — they are content now, and words are compared IN ORDER.
#: deictic words: a request that points at "this"/"it" means whatever is
#: current — never the same request twice ("delete this project")
_DEICTIC = frozenset("this that it these those αυτο αυτη αυτος αυτα εκεινο".split())

def normalize_request(text, redact: bool = False) -> str:
    """Lower-case, accents stripped, punctuation to spaces, whitespace
    collapsed."""
    import unicodedata
    # the RECORDED request went through the recorder's redaction (home paths,
    # IPs, e-mails, secrets — distill/redact.py): compare both sides in that
    # form, or a resent request never matches its own lesson (fourth/fifth review)
    t = _redacted(str(text or "")) if redact else str(text or "")
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # a dotted name stays ONE token ("a.txt" ≠ "b.txt", "10.0.0.1")
    t = re.sub(r"(?<=\w)\.(?=\w)", "\u2024", t)
    t = re.sub(r"[^\w\s\u2024]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_REDACTION_MARK = re.compile(r"<REDACTED|/(?:Users|home)/<user>")


def _redacted(text: str) -> str:
    try:
        from ..distill.redact import redact_text
        return redact_text(text)
    except Exception:  # noqa: BLE001
        return re.sub(r"/(Users|home)/[^/\s\"':]+", r"/\1/<user>", text)


def content_words(text) -> set:
    """The request's content words: normalised tokens of 2+ chars that are
    not function words."""
    # a number of ANY length is content ("round 1" ≠ "round 2", review)
    return {w for w in normalize_request(text).split()
            if (len(w) >= 2 or w.isdigit()) and w not in _FUNCTION_WORDS}


#: a prefix that turns a verb into its opposite or another task — never a
#: typo (fourth review: "install" ~ "uninstall", "compress" ~ "decompress",
#: "activate" ~ "deactivate", "archive" ~ "unarchive" all matched)
_TASK_PREFIXES = ("un", "de", "dis", "re", "in", "im", "en", "em", "non", "anti", "pre", "post", "over", "under",
                  "mis", "sub", "super", "inter", "co")


def _strip_prefix(w: str):
    for p in sorted(_TASK_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) - len(p) >= 4:
            return p, w[len(p):]
    return "", w


def _edit_distance(a: str, b: str) -> int:
    """Damerau–Levenshtein (adjacent transposition = 1)."""
    d = {(i, -1): i + 1 for i in range(-1, len(a))}
    d.update({(-1, j): j + 1 for j in range(-1, len(b))})
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ca != cb))
            if i and j and ca == b[j - 1] and a[i - 1] == cb:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1)
    return d[len(a) - 1, len(b) - 1]


def _word_match(a: str, b: str) -> bool:
    """The same word, or a TYPO of it: one edit (two for words of 8+ letters),
    no digits, and not the other word with a prefix added."""
    if a == b:
        return True
    if any(c.isdigit() for c in a + b):          # ids, ports, dates: exact only
        return False
    if min(len(a), len(b)) < 5:
        return False
    lo, hi = sorted((a, b), key=len)
    if hi.endswith(lo) and hi[: len(hi) - len(lo)] in _TASK_PREFIXES:
        return False
    if hi.startswith(lo) and hi[len(lo):] in ("s", "es"):
        return False                          # one vs many ("service" ≠ "services")
    # one prefix swapped for another: "encoding"/"decoding",
    # "reactivate"/"deactivate" (fifth review)
    (pa, sa_), (pb, sb_) = _strip_prefix(a), _strip_prefix(b)
    if pa != pb and sa_ == sb_:
        return False
    # a typo is not a word: two DIFFERENT dictionary words are two words
    # ("present"/"president", "inserted"/"inverted")
    if a.isascii() and b.isascii() and _is_english_word(a) and _is_english_word(b):
        return False
    if difflib.SequenceMatcher(None, a, b).ratio() < TYPO_RATIO:
        return False
    return _edit_distance(a, b) <= (2 if min(len(a), len(b)) >= 8 else 1)


def content_sequence(text, redact: bool = False) -> list:
    """The content words IN ORDER, repeats collapsed."""
    out = []
    for w in normalize_request(text, redact).split():
        if (len(w) >= 2 or w.isdigit()) and w not in _FUNCTION_WORDS and (not out or out[-1] != w):
            out.append(w)
    return out


#: a request whose content words are all of these means something different
#: each time ("redo", "proceed", "yes do it", "hello ghost")
_ANAPHORIC = frozenset("""redo retry again continue proceed go yes no yeah yep nope sure next more fix try repeat
done stop start ok okay thanks hello hi hey ghost same well good great fine right wait""".split())
#: the deictic rule applies to a SHORT request only: "delete this project" is
#: anaphoric; "an image that looks like …" uses "that" as a relative pronoun
#: (fourth review: 13 of 70 live scoped lessons could never match their own
#: request)
DEICTIC_MAX_CONTENT_WORDS = 3


#: words the content comparison drops but whose CHANGE changes the request
#: (fifth review: "what is my name" ~ "what is your name", "did the backup
#: run" ~ "will the backup run")
_PERSON = {**{w: 1 for w in "i me my mine myself we us our ours ourselves μου μας εγω".split()},
           **{w: 2 for w in "you your yours yourself yourselves σου σας εσυ".split()},
           **{w: 3 for w in "he him his she her hers they them their theirs it its".split()}}
_TENSE = {**{w: "past" for w in "did was were had".split()}, **{w: "future" for w in "will shall θα".split()}}


def _classes(text, table) -> set:
    return {table[w] for w in normalize_request(text).split() if w in table}


def _same_frame(a, b) -> bool:
    """The person and the tense, where both requests state one, agree."""
    for table in (_PERSON, _TENSE):
        ca, cb = _classes(a, table), _classes(b, table)
        if ca and cb and ca != cb:
            return False
    return True


def same_request(a, b) -> bool:
    """Is request ``b`` the same task as request ``a``? The same content
    words in the same order (each matched exactly, or as a typo — see
    ``_word_match``); not a request made only of anaphoric words; and not a
    short request pointing at "this"/"it"."""
    # a side that went through the recorder's redaction is compared with the
    # other side redacted the same way — and only then: two live requests
    # naming different IPs stay different requests
    red = bool(_REDACTION_MARK.search(f"{a}\n{b}"))
    sa, sb = content_sequence(a, red), content_sequence(b, red)
    if not sa or len(sa) != len(sb) or set(sa) <= _ANAPHORIC or not _same_frame(a, b):
        return False
    if len(sa) <= DEICTIC_MAX_CONTENT_WORDS and _DEICTIC & (
            set(normalize_request(a).split()) | set(normalize_request(b).split())):
        return False
    if len(sa) == 1:
        # one content word carries too little: the whole wording must agree,
        # courtesy words aside ("who am I?" = "Who am I"; "Hello, how are
        # you?" ≠ "how's the")
        _q = lambda t: [w for w in normalize_request(t, red).split() if w not in _COURTESY]
        return _q(a) == _q(b)
    return all(_word_match(x, y) for x, y in zip(sa, sb))


_COURTESY = frozenset("please thanks thank you ghost hello hi hey ok okay do a an the me give tell show get run now".split())


def lesson_scope(lesson) -> str:
    """``request`` or ``general``. A lesson without the field is general (the
    pre-§4KW default); the migration tags the request-keyed ones."""
    try:
        return SCOPE_REQUEST if str((lesson or {}).get("scope") or "") == SCOPE_REQUEST else SCOPE_GENERAL
    except Exception:  # noqa: BLE001
        return SCOPE_GENERAL


def lesson_request_text(lesson) -> str:
    """The request a request-scoped lesson belongs to (the stored
    ``source_request``, else its trigger)."""
    lesson = lesson or {}
    return str(lesson.get("source_request") or lesson.get("trigger") or lesson.get("task") or "")


def admits(lesson, query) -> bool:
    """May ``lesson`` enter the prompt for ``query``? A general lesson: yes
    (relevance is the retriever's job). A request-scoped one: only for the
    same request."""
    if lesson_scope(lesson) != SCOPE_REQUEST:
        return True
    src = lesson_request_text(lesson)
    # A stored request cut at its cap (legacy rows: 400 chars) is compared
    # with the same-length head of the new request (fresh review: a resent
    # long request never matched its own truncated copy).
    if len(src) in TRUNCATED_LENGTHS and len(str(query or "")) > len(src):
        return same_request(src, str(query)[:len(src)])
    return same_request(src, query)


#: the caps a stored request was cut at (legacy 400, current 4,000): only a
#: copy of EXACTLY that length is taken as cut (third review: any 395+ char
#: request admitted every longer request that started with it)
TRUNCATED_LENGTHS = frozenset({400, 4000})


# ── generality of a GENERAL lesson's trigger ─────────────────────────────────
#: Request-specific markers that must not appear in a general text: a URL, a
#: path, an @-mention, a quoted span. A NUMBER is specific only when it comes
#: from the request ("server errors such as 500 or 503" is general).
_SPECIFIC_RE = re.compile(r"https?://|(?:^|\s)~?/[\w.]|<@|\"[^\"]{3,}\"|«[^»]{3,}»")
_NUMBER_RE = re.compile(r"\b\d{2,}\b")


def is_general_trigger(trigger, request, *, max_shared_share: float = 0.5) -> bool:
    """Does ``trigger`` describe a situation CLASS rather than restate
    ``request``? False when it carries a specific marker or a number from the
    request, or shares more than ``max_shared_share`` of its content words
    with the request's (a restatement, its names, places and topics)."""
    t = str(trigger or "").strip()
    if len(t) < 12:
        return False
    # (a restatement of the request shares its content words — caught below)
    return is_general_text(t, request, max_shared_share=max_shared_share)


def is_general_text(text, request, *, max_shared_share: float = 0.5) -> bool:
    """No specific marker, no number taken from the request, and at most
    ``max_shared_share`` of its content words are the request's. An empty
    text is general (nothing specific in it)."""
    t = str(text or "").strip()
    if not t:
        return True
    if _SPECIFIC_RE.search(t):
        return False
    if set(_NUMBER_RE.findall(t)) & set(_NUMBER_RE.findall(str(request or ""))):
        return False
    tw, rw = content_words(t), content_words(request)
    if not tw:
        return False
    # a FILE or dotted name from the request ("report.txt", "kc_probe.py",
    # "10.0.0.1") is that request's, however general the rest (fourth review)
    if any("\u2024" in w and not re.fullmatch(r"[a-z]\u2024[a-z]", w) for w in tw & rw):   # not "e.g."
        return False
    # a NAME from the request, even misspelled ("Panerythraikos" for
    # "panerithraikos"), makes the text specific however many general words
    # dilute the ratio (third review)
    names = {normalize_request(m) for m in re.findall(r"(?<!^)(?<![.!?]\s)\b[A-ZΑ-Ω][\w'-]{3,}", t)}
    # a sentence-initial capital is a name too when it is no English word
    # (fourth review: "Panerithraikos questions: …" escaped the check)
    for m in re.findall(r"(?:^|[.!?]\s+)([A-ZΑ-Ω][\w'-]{3,})", t):
        if not _is_english_word(m):
            names.add(normalize_request(m))
    rwords = [w for w in rw if len(w) >= 4]
    # each part of a hyphenated name counts ("Leonidas-style", fifth review)
    parts = {p for n in names for p in n.split() if len(p) >= 4}
    if any(any(p == w or difflib.SequenceMatcher(None, p, w).ratio() >= 0.8 for w in rwords) for p in parts):
        return False
    return len(tw & rw) / len(tw) <= max_shared_share


_ENGLISH = None


def _is_english_word(word: str) -> bool:
    """In the system word list (an inflection stripped), or — without one —
    assumed to be (the pre-fourth-review behaviour)."""
    global _ENGLISH
    if _ENGLISH is None:
        try:
            with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
                _ENGLISH = frozenset(w.strip().lower() for w in fh if w.strip())
        except OSError:
            _ENGLISH = frozenset()
    if not _ENGLISH:
        return True
    w = word.lower().strip("'-")
    if not w.isascii():
        return False
    return any(x in _ENGLISH for x in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith(("es", "ed")) else w,
                                       w[:-3] if w.endswith("ing") else w, w[:-3] + "y" if w.endswith("ies") else w))
