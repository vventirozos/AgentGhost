"""Who said it (§4KY, 2026-10-03).

The background fact extractor (`run_smart_memory_task`) wrote facts ABOUT THE
OWNER from anything in the episode: the owner's QUESTION "i'm a doctor/medicine
denialist, is this male privilege?" became `user HAS_PROFESSION doctor`; a
role-play ("pretend you are my wife Maria and we live in London") replaced the
owner's real wife and home; a pasted colleague's bio replaced the employer; the
agent's own stats, tasks and files became "the user has"; a channel member's
"I have type 1 diabetes" in the owner's thread reached the owner's graph.

A fact about the owner is written only when the OWNER STATED it: in one of the
owner's own lines, in a first-person declarative sentence that is not a
question, not role-play / a hypothetical, and not inside a quote. A statement
that NEGATES it ("I'm not a doctor") is a correction: the matching owner fact
is removed instead.
"""
import re

from .profile import _fold

#: labels the Slack bot puts on other people's thread messages
from ..utils.logging import FOREIGN_MESSAGE_LABELS

_FIRST_PERSON = re.compile(
    r"\b(?:i|i'?m|i'?ve|i'?d|i'?ll|me|my|mine|myself|we|our|ours|us)\b"
    r"|(?<!\w)(?:μου|μας|ειμαι|εχω|εγω|μενω|δουλευω|λεγομαι|ζω)(?!\w)",
    re.IGNORECASE)
#: an instruction from the owner states the owner's preference ("Always write
#: tests", "Use metric units") — it has no "I" but it is theirs
_IMPERATIVE = re.compile(
    r"^\s*(?:please\s+)?(?:always|never|don'?t|do|use|prefer|avoid|keep|make|write|call|reply|answer|remember|"
    r"note|stop|start|give|show|tell|send|put|add|run|check|ask)\b"
    r"|^\s*(?:παντα|ποτε|μην|χρησιμοποιησε|θυμησου)(?!\w)",
    re.IGNORECASE)
_ROLE_PLAY = re.compile(
    r"\b(?:pretend|imagine|role[\s-]?play|act\s+as|let'?s\s+say|suppose|supposing|hypothetic\w*|what\s+if|"
    r"in\s+(?:my|a|the)\s+(?:story|novel|game|book|script)|character|fiction\w*|play\s+the\s+role|"
    r"play\s+a\s+game|you\s+are\s+(?:now\s+)?my|you'?re\s+(?:now\s+)?my|as\s+if)\b"
    r"|(?<!\w)(?:φαντασου|υποθεσε|ας\s+πουμε)(?!\w)",
    re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|no\s+longer|never|isn'?t|aren'?t|wasn'?t|don'?t|doesn'?t|didn'?t|nor|no)\b|(?<!\w)(?:δεν|οχι|ουτε)(?!\w)",
    re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»|```.*?```", re.DOTALL)
#: words that say nothing about WHICH fact it is
_STOP = frozenset("""user users the a an and or of to in on at for with is are was were has have had be been being
their his her its this that who which as by from also now currently am do does did very really just about
""".split())

#: predicates that name the AGENT's or a project's state, never the owner's
#: life (review: `user HAS_STAT total lessons`, `HAS_TASK create cli.py`,
#: `HAS_SANDBOX_FILES mars_distance.py`, `HAS_PROJECT_CODENAME zephyrine…`)
AGENT_STATE_TOKENS = frozenset({
    "PROJECT", "PROJECTS", "SKILL", "SKILLS", "TASK", "TASKS", "SANDBOX", "FILE", "FILES", "DOCUMENTATION",
    "STAT", "STATS", "LEARNING", "CODENAME", "WORKSPACE", "RESOURCE", "INTROSPECTION",
    "COMPETENCE", "TEST", "PROBE", "SESSION", "REQUEST", "REQUESTED", "ASKED", "QUERY",
})


def owner_lines(episode: str) -> list:
    """The OWNER's own lines of a smart-memory episode (``USER:`` lines that
    do not carry another person's label)."""
    text = str(episode or "")
    lines = text.splitlines()
    roles = [re.match(r"\s*(user|ai|assistant|system|tool)\s*:", ln, re.IGNORECASE) for ln in lines]
    if not any(roles):
        # no speaker marks at all (a legacy journal item): one speaker, the owner
        return [] if text.strip().startswith(FOREIGN_MESSAGE_LABELS) else [text.strip()]
    out = []
    for ln, m in zip(lines, roles):
        if m and m.group(1).lower() == "user":
            body = ln[m.end():].strip()
            if not body.startswith(FOREIGN_MESSAGE_LABELS):
                out.append(body)
    return out


def owner_statements(episode: str) -> list:
    """``(clause, negated)`` for each clause of a first-person or imperative
    declarative sentence the owner wrote: questions and quoted text excluded,
    and NOTHING from a turn that sets up role-play anywhere (§4KZ: "Let's play
    a game. You are my wife Maria. We live in London." passed sentence by
    sentence). Negation is per CLAUSE: "spelled Φωτεινή, not Fotini" states
    Φωτεινή and negates Fotini."""
    lines = owner_lines(episode)
    if any(_ROLE_PLAY.search(_fold(b)) for b in lines):
        return []
    out = []
    for body in lines:
        body = _QUOTED.sub(" ", body)
        for m in re.finditer(r"[^.!?;\n]+[.!?;]?", body):
            sent = m.group(0).strip()
            if not sent or sent.endswith("?") or sent.lstrip().startswith(">"):
                continue
            f = _fold(sent)
            if not (_FIRST_PERSON.search(f) or _IMPERATIVE.search(f)):
                continue
            for clause in re.split(r",|;|\bbut\b|\binstead\b|\bwhereas\b|(?<!\w)αλλα(?!\w)", f):
                clause = clause.strip(" .!")
                if clause:
                    out.append((clause, bool(_NEGATION.search(clause))))
    return out


#: what a statement must SAY for a single-valued owner fact to change — a
#: shared value word is not enough (§4KZ: "I'm in Kyllini this weekend" made
#: "lives in Kyllini" and replaced the owner's home)
PREDICATE_CUES = {
    "LIVES_IN": r"\b(?:live|lives|living|lived|moved?|moving|home|reside\w*|settled)\b|(?<!\w)(?:μενω|μενουμε|μετακομισ\w*|σπιτι)(?!\w)",
    "WORKS_AT": r"\b(?:work|works|working|worked|job|employ\w*|joined|hired|career)\b|(?<!\w)(?:δουλευω|δουλεια|εργαζομαι)(?!\w)",
    "MARRIED_TO": r"\b(?:wife|husband|married|spouse|partner)\b|(?<!\w)(?:γυναικα|συζυγ\w*|παντρε\w*)(?!\w)",
    "HAS_NAME": r"\b(?:name|named|call\s+me|called)\b|(?<!\w)(?:λεγομαι|ονομα\w*)(?!\w)",
    "HAS_BIRTHDATE": r"\b(?:born|birthday|birth|birthdate)\b|(?<!\w)(?:γεννη\w*|γενεθλια)(?!\w)",
    "HAS_PROFESSION": r"\b(?:i'?m\s+an?|i\s+am\s+an?|work\w*\s+as|profession|job|career)\b",
}


def _words(text) -> set:
    return {w for w in re.findall(r"\w+", _fold(text)) if len(w) >= 3 and w not in _STOP}


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k >= 4 and a[k:] in ("", "s", "es", "ed", "ing", "ies", "er", "ers") and \
        b[k:] in ("", "s", "es", "ed", "ing", "ies", "er", "ers")


def _coverage(words: set, sentence: str) -> float:
    sw = _words(sentence)
    return sum(1 for w in words if any(_same(w, x) for x in sw)) / len(words) if words else 0.0


def owner_said(value, statements, minimum: float = 0.5, cue: str = None):
    """``"stated"`` when an affirmative owner clause carries at least
    ``minimum`` of ``value``'s distinctive words (and, with ``cue``, says the
    predicate's kind of thing), ``"negated"`` when only a NEGATED one does,
    else None."""
    words = _words(value)
    if not words:
        return None
    rx = re.compile(cue, re.IGNORECASE) if cue else None
    ok = [(s, neg) for s, neg in statements if rx is None or rx.search(s)]
    best_pos = max((_coverage(words, s) for s, neg in ok if not neg), default=0.0)
    if best_pos >= minimum:
        return "stated"
    best_neg = max((_coverage(words, s) for s, neg in statements if neg), default=0.0)
    return "negated" if best_neg >= minimum else None


def is_owner_end(node) -> bool:
    return _fold(node).strip() in ("user", "the user", "owner", "me", "i")


def strip_negation(value) -> str:
    """"not a doctor" → "doctor" (the value a correction removes)."""
    return re.sub(r"^\s*(?:not|no\s+longer|no|never|isn'?t|is\s+not|am\s+not|δεν\s+ειμαι)\s+(?:an?\s+)?", "",
                  _fold(value)).strip()


def is_agent_state_predicate(predicate) -> bool:
    return bool(set(str(predicate or "").upper().split("_")) & AGENT_STATE_TOKENS)
