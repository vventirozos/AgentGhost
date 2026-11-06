"""The current request's text and its content words.

Three small things other modules share, kept here since the lesson store was
removed (§4TD, 2026-11-06 — they lived in ``memory/lesson_scope.py`` and
``memory/skills.py``, which went with it):

* ``current_request`` — the user's request for THIS turn, set by the turn loop
  first thing. The fact ledger reads it for the owner's own words, and the
  forget tool for the request a preview was made in.
* ``normalize_request`` / ``content_words`` — a text folded to comparable
  words (lower-case, accents stripped, punctuation out, function words
  dropped). The forget tool matches a target against stored facts with them.
* ``mostly_non_latin`` — more letters outside the Latin script than in it;
  the English embedder cannot place such a query, so the semantic paths that
  would rank it against arbitrary rows skip it.
"""
from __future__ import annotations

import contextvars
import re

#: the user's request for THIS turn (set by the turn loop)
current_request = contextvars.ContextVar("request_text_current_request", default="")

# Function words only (English + Greek, accent-folded) — content words such as
# "old", "new", "show" carry the request's meaning and must stay. Negation,
# quantity, direction and question words are content too ("do NOT restart" ≠
# "restart", "stop ALL services" ≠ "stop this service").
_FUNCTION_WORDS = frozenset("""
a an the and or but if then than so of in on at by for with into onto about as is are was were be been
being am do does did done have has had having i me my mine you your yours we us our he him his she her it its
they them their these those there here which whom whose can could would should will shall may might must
please just also too very really now again let lets tell give say thanks thank ok okay hi hello
ο η το οι τα του της των τον την τη τις σε στο στη στην στον στα στους στις με για και θα
ειναι ηταν εγω εσυ μου σου μας σας τους ενα μια ενας ας τωρα παλι
""".split())


def normalize_request(text) -> str:
    """Lower-case, accents stripped, punctuation to spaces, whitespace
    collapsed."""
    import unicodedata
    t = unicodedata.normalize("NFD", str(text or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # a dotted name stays ONE token ("a.txt" ≠ "b.txt", "10.0.0.1")
    t = re.sub(r"(?<=\w)\.(?=\w)", "․", t)
    t = re.sub(r"[^\w\s․]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def content_words(text) -> set:
    """The text's content words: normalised tokens of 2+ chars that are not
    function words."""
    # a number of ANY length is content ("round 1" ≠ "round 2")
    return {w for w in normalize_request(text).split()
            if (len(w) >= 2 or w.isdigit()) and w not in _FUNCTION_WORDS}


def mostly_non_latin(text) -> bool:
    """More letters outside the Latin script than in it — the embedder
    (bge-small-en) cannot place such a query (§4LC: Greek queries sat closer
    to ARBITRARY rows, 0.26–0.33, than relevant English pairs)."""
    letters = [c for c in str(text or "") if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if not ("a" <= c.lower() <= "z")) > len(letters) / 2


__all__ = ["current_request", "normalize_request", "content_words", "mostly_non_latin"]
