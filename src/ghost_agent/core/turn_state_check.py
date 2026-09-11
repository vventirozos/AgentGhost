"""State-aware, refute-only checks of one turn against its own record (§4FY).

WHY. The verifier judges the last reply against the last few tool outputs
— a trajectory-LOCAL judge. Six weeks of labelled turns (§4FD) put the
largest failure classes in what that judge structurally cannot see: the
request's own explicit constraints ("STRICT JSON and NOTHING else", "just
the number", "one word"), and a turn whose every retrieval came back empty
answered with confident prose. The literature names the same blind spot —
production judges miss ~1 in 5 errors and the misses are cross-turn STATE,
not local text (arXiv 2606.10315) — and the lever for a weak judge is a
state tracker beside it, not more model.

WHAT. Pure functions over the turn's record — the CURRENT request text,
the tool calls it ran (name, arguments, result) and the delivered reply —
producing NEGATIVE verdicts only:

  * a mechanically checkable constraint of the current request, violated
    by the reply's SHAPE (JSON-ness, an exact phrase, a word / line /
    sentence cap, a number-only answer);
  * every web retrieval this turn came back empty or errored and the reply
    asserts an answer anyway, without saying so.

Two rules were built, measured on the corpus and deleted — a mandated
opening phrase and a count-versus-own-list check; the notes beside where
they stood say what the measurement showed.

REFUTE-ONLY, BY CONSTRUCTION (§4EP/§4EQ). A satisfied constraint returns
NOTHING — not a CONFIRMED, not an UNCERTAIN — because confirm-carrying
coverage was measured to make the calibration comparison LESS resolvable
(2–16× across seeds). The module exports no way to express a pass, and an
empty result means "nothing to say", never "the reply is fine".

ONLY THE CURRENT REQUEST. Stored project constraints are deliberately NOT
read here: they were the §4FD constraint-bleed ("how's the weather?"
refuted for not opening with a project's mandated phrase) and they already
reach the LLM judge through `_active_constraint_note` under the relevance
gate. A request-level constraint has one author, one turn, no bleed.

A CONSTRAINT IS AN UNCONDITIONAL IMPERATIVE ABOUT THE REPLY. The first
version read "valid JSON" anywhere as a constraint and refuted the Flask
handler a user asked for ("the API should respond with valid JSON — write
the handler"), read "yes or no" as a literal phrase, and refuted the honest
branch of "if green reply GREEN, otherwise paste the failure" (§4FY review,
adversarial lens). So the parser works CLAUSE by clause and a clause yields
a constraint only when it is imperative about the reply (a reply verb or a
clause that is nothing but the cap), carries no condition, alternative,
per-item scope or "then …" continuation, and mentions no deliverable
(code, file, schema, API …) or question word. What that costs is recall on
odd phrasings; what it buys is that a label is never written from a clause
the user did not mean as a constraint.

PRECISION OVER RECALL, EVERYWHERE. A false refute here writes a `failed`
label that feeds calibration, playbook credit and post-mortem selection —
the one class the corpus cannot afford noise in. So every rule tolerates
cosmetic decoration (a bold number, a trailing period, a fenced block),
counts whitespace tokens rather than regex words ("3.12.4", "Open-source"
are one word), knows abbreviations are not sentence ends, and — for every
constraint rule — stands down on an HONEST INABILITY report ("I can't
access that file"): the 2026-07-31 honest-failure rule says a reply that
says it could not do the thing must not be taught that fabricating the
thing scores better. Measured before wiring: `scripts/turn_state_replay.py`
runs the checks over the whole trajectory corpus and prints every fire on
a passed or human-approved turn for reading — and every regex here is
linear in the input (the first version had three quadratic ones).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .evidence_gate import assess_turn_evidence

__all__ = ["refute_turn_state", "mechanical_constraints", "Constraint"]

#: Sub-check tags, carried on every issue so the override report can
#: count precision per rule rather than per module.
TAG = "turn-state"

_NUM_WORDS = {
    "one": 1, "a single": 1, "single": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50,
}
_NUM = r"(?:\d{1,3}|one|a single|single|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty)"


def _num(tok: str) -> Optional[int]:
    tok = (tok or "").strip().lower()
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok)


@dataclass(frozen=True)
class Constraint:
    kind: str          # strict_json | exact | number_only | word_cap | line_cap | sentence_cap
    value: Any         # cap (int) / allowed phrases (tuple of str) / None
    source: str        # the clause the rule was read from


# ── request-side parsing ─────────────────────────────────────────────────
#
# Each pattern names ONE reply shape the user asked for, in the words users
# actually used (surveyed on 1,863 real requests: "STRICT JSON on a single
# line and NOTHING else" ×189, "Reply with exactly: …" ×58, "one line" ×21,
# "just the number" ×11, "in one sentence" ×10, "one word" ×4).

#: A clause with any of these is conditional, scoped per item, or a
#: two-part instruction — never a whole-reply constraint (review C2/C10/C11).
_CLAUSE_SKIP_RE = re.compile(
    r"\b(?:if|unless|otherwise|else|in\s+case|when(?:ever)?|depending|"
    r"per|each|every|for\s+all|then)\b",
    re.IGNORECASE)
#: A clause about a DELIVERABLE, a question, or code — the JSON/line/word
#: phrase describes the thing being built or asked about, not the reply
#: (review #8, #12).
_DELIVERABLE_RE = re.compile(
    r"\b(?:write|script|code|handler|function|files?(?!:)|schema|fixture|"
    r"endpoint|api|server|should|convert|"
    r"diagnose|fix|bug|rewrite|return\s+fits|fits?|prints?|minif\w*|config|"
    r"editor|shows?|processor|"
    r"per\s+(?:line|row)|jsonl|newline)\b",
    re.IGNORECASE)
# (not "html|python|shell|bash|command": those name PATHS in real requests —
# "screenshot of …/pinball.html … tell me in one sentence" lost its cap —
# and the shapes they guarded are held by `_CAP_TAIL` ("in one line of
# shell") and the question / write words.)
#: A question / explanation word BEFORE the cap phrase makes the clause a
#: question about the phrase ("what does 'strictly JSON' mean", "how do I
#: put it in one line"); AFTER it, the phrase introduces the question ("in
#: one sentence, what is your favourite problem" — a real cap).
_QUESTION_BEFORE_RE = re.compile(r"\b(?:explain|why|what|how)\b", re.IGNORECASE)
_REPLY_VERB = r"(?:reply|respond|answer|output|return|say|confirm|tell\s+me|summari[sz]e|report|state)"
#: What may follow a cap phrase for it to be the whole clause's point:
#: end, punctuation, a comparative tail, or the question it introduces.
_CAP_TAIL = (r"(?=\s*(?:[.!?,;:)\]]|$)|\s+or\s+(?:fewer|less)\b|\s+(?:max|maximum|tops)\b|"
             r"\s+(?:whether|what|which|who|how|why|from|based|using|about)\b)")

_STRICT_JSON_RES = (
    # "Reply with STRICT JSON … and NOTHING else", "respond with only JSON"
    re.compile(r"\b" + _REPLY_VERB + r"\b[^\n]{0,40}?\b(?:strict(?:ly)?|only|pure|plain(?:\s+text)?|raw|valid)?\s*json\b", re.IGNORECASE),
    # "PLAIN TEXT JSON ONLY", "JSON only" — an exclusive form needs no verb
    re.compile(r"\bjson\s+only\b", re.IGNORECASE),
)
#: A clause that opens as a QUESTION ("Did they reply exactly: 'no'?") is
#: about someone else's reply, not an instruction for this one.
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(?:did|does|do|can|could|is|are|was|were|will|would|should|has|have|had)\s+"
    r"(?:they|he|she|it|you|we|i|the|that|this|anyone|someone)\b",
    re.IGNORECASE)
_EXCLUSIVE_MASK_RE = re.compile(r"\b(?:nothing|anything)\s+(?:else|more)\b", re.IGNORECASE)
_EXCLUSIVE_RE = re.compile(
    r"\b(?:nothing\s+(?:else|more)|only|no\s+(?:other|extra|additional)\s+(?:text|prose|output|words?))\b",
    re.IGNORECASE)
_EXACT_RE = re.compile(
    r"\b(?:reply|answer|respond|say)\s+(?:with\s+)?exactly(?:\s+(?:the\s+)?(?:words?|phrase|text|sentence))?\s*:\s*"
    r"(?P<phrase>[^\n]{1,160})",
    re.IGNORECASE)
#: "OK and nothing else", "DONE, nothing more", "ACK — no explanation",
#: "ACK (nothing else)": the trailer is an instruction, not the phrase.
_EXACT_TAIL_RE = re.compile(
    r"(?:\s*[.,;:—–-]\s*|\s+|\s*\(\s*)(?:and\s+)?(?:nothing\s+(?:else|more)|"
    r"no\s+(?:explanation|other|extra|more|further)\b|without\b|do\s*n[o']t\b|don't\b)[^\n]*$",
    re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"[<{\[][^>}\]]*[>}\]]")
_NUMBER_ONLY_RES = (
    re.compile(r"\b(?:reply|answer|respond)\s+with\s+(?:just|only)\s+(?:the\s+)?number\b(?!\s+of\b)", re.IGNORECASE),
    re.compile(r"\b(?:just|only)\s+the\s+number\b(?!\s+of\b)", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+)?number\s+only\s*[.!]?\s*$", re.IGNORECASE),
)
_WORD_CAP_RES = (
    re.compile(r"\b(?P<n>" + _NUM + r")\s+words?\s+or\s+(?:fewer|less)\b", re.IGNORECASE),
    re.compile(r"\b(?:at\s+most|no\s+more\s+than|max(?:imum)?(?:\s+of)?|up\s+to)\s+(?P<n>" + _NUM + r")\s+words?\b" + _CAP_TAIL, re.IGNORECASE),
    re.compile(r"\b(?P<n>" + _NUM + r")\s+words?\s+(?:max|maximum|tops)\b", re.IGNORECASE),
    re.compile(r"\b(?:in|with|using)\s+(?:exactly\s+)?(?P<n>" + _NUM + r")\s+words?\b" + _CAP_TAIL, re.IGNORECASE),
    re.compile(r"(?:^|[,.;:]\s*)(?:exactly\s+)?(?P<n>one|a single|1|two|three)\s+words?\s*[.!]?\s*$", re.IGNORECASE),
)
_LINE_CAP_RES = (
    re.compile(r"\b" + _REPLY_VERB + r"\b[^\n]{0,40}?\b(?:in|as|on)\s+(?:one|a\s+single|1)\s+line\b" + _CAP_TAIL, re.IGNORECASE),
    re.compile(r"\bone-line\s+(?:summary|confirmation|reply|answer|response|report|status)\b", re.IGNORECASE),
    re.compile(r"(?:^|[,.;:]\s*)one\s+line\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"\b(?:at\s+most|no\s+more\s+than|max(?:imum)?(?:\s+of)?|up\s+to)\s+(?P<n>" + _NUM + r")\s+lines\b" + _CAP_TAIL, re.IGNORECASE),
    re.compile(r"\b(?P<n>" + _NUM + r")\s+lines?\s+(?:max|maximum|tops)\b", re.IGNORECASE),
)
_SENTENCE_CAP_RE = re.compile(
    r"\b(?:in|with)\s+(?:one|a\s+single|1)\s+(?:short\s+|brief\s+)?sentence\b" + _CAP_TAIL, re.IGNORECASE)


def _clauses(text: str) -> List[str]:
    """Sentence-level clauses of a request: split on newlines and on
    `.!?` at a real boundary (not inside "3.12" or "x.py")."""
    out: List[str] = []
    for part in re.split(r"\n+|(?<!\d)(?<![A-Za-z]\.)\.+(?=\s|$)|[!?]+(?=\s|$)", text or ""):
        part = (part or "").strip()
        if part:
            out.append(part)
    return out


def _quoted(clause: str, start: int) -> bool:
    """The match is not an instruction for THIS reply: it sits inside quoted
    speech ("the user said 'reply exactly: foo'"), or a question /
    explanation word precedes it in the clause ("how do I put it in one
    line", "what does 'strictly JSON' mean")."""
    head = clause[:start].rstrip()
    if bool(head) and head[-1] in "\"'“‘`":
        return True
    return bool(_QUESTION_BEFORE_RE.search(head))


def _phrase_set(raw: str) -> Optional[Tuple[str, ...]]:
    """The allowed phrase(s) of an exact-reply constraint, or None when the
    capture is not a literal ("<the sha of HEAD>", a nine-word sentence,
    "the name, the price and the URL")."""
    phrase = _EXACT_TAIL_RE.sub("", raw.strip())
    phrase = phrase.strip().strip("\"'“”‘’`").strip()
    if phrase.endswith((".", "!")) and len(phrase) > 3:
        phrase = phrase[:-1].rstrip()
    if not phrase or _PLACEHOLDER_RE.search(phrase):
        return None
    # "yes or no", "yes / no": a SET of allowed answers, not one literal
    alts = [a.strip().strip("\"'“”‘’`") for a in re.split(r"\s+or\s+|\s*/\s*", phrase, flags=re.IGNORECASE)]
    alts = [a for a in alts if a]
    if not alts or any(len(a.split()) > 8 for a in alts) or len(alts) > 4:
        return None
    return tuple(alts)


def mechanical_constraints(request: str) -> List[Constraint]:
    """The reply-shape constraints the CURRENT request states, in order.

    Never raises. Reads only ``request`` — never a stored project
    constraint (see the module note). Returns an empty list for a request
    that asks nothing mechanically checkable, which is most of them.
    """
    out: List[Constraint] = []
    text = str(request or "")
    if not text.strip():
        return out
    try:
        seen = set()
        for clause in _clauses(text):
            # "NOTHING else" / "anything more" are exclusivity, not the
            # conditional "else" — masked before the skip test
            if _CLAUSE_SKIP_RE.search(_EXCLUSIVE_MASK_RE.sub(" ", clause)):
                continue
            if _QUESTION_LEAD_RE.match(clause):
                continue
            deliverable = bool(_DELIVERABLE_RE.search(clause))
            if "strict_json" not in seen and not deliverable:
                m = _STRICT_JSON_RES[0].search(clause)
                if m and not _quoted(clause, m.start()) and (
                        _EXCLUSIVE_RE.search(clause)
                        or re.search(r"\b(?:reply|respond|answer|output|return)\s+(?:with\s+|in\s+|as\s+)?"
                                     r"(?:strict(?:ly)?|only|pure|plain(?:\s+text)?|raw|valid)\s+json\b",
                                     clause, re.IGNORECASE)):
                    seen.add("strict_json")
                    out.append(Constraint("strict_json", None, clause[:120]))
                else:
                    m2 = _STRICT_JSON_RES[1].search(clause)
                    if m2 and not _quoted(clause, m2.start()):
                        seen.add("strict_json")
                        out.append(Constraint("strict_json", None, clause[:120]))
            if "exact" not in seen:
                m = _EXACT_RE.search(clause)
                if m and not _quoted(clause, m.start()):
                    phrases = _phrase_set(m.group("phrase"))
                    if phrases:
                        seen.add("exact")
                        out.append(Constraint("exact", phrases, clause[:120]))
            if "number_only" not in seen and not deliverable:
                for rx in _NUMBER_ONLY_RES:
                    m = rx.search(clause)
                    if m and not _quoted(clause, m.start()):
                        seen.add("number_only")
                        out.append(Constraint("number_only", None, clause[:120]))
                        break
            if "word_cap" not in seen and not deliverable:
                for rx in _WORD_CAP_RES:
                    m = rx.search(clause)
                    if m and not _quoted(clause, m.start()):
                        n = _num(m.group("n"))
                        if n:
                            seen.add("word_cap")
                            out.append(Constraint("word_cap", n, clause[:120]))
                        break
            if "line_cap" not in seen and not deliverable:
                for rx in _LINE_CAP_RES:
                    m = rx.search(clause)
                    if m and not _quoted(clause, m.start()):
                        n = _num(m.groupdict().get("n") or "one") or 1
                        seen.add("line_cap")
                        out.append(Constraint("line_cap", n, clause[:120]))
                        break
            if "sentence_cap" not in seen and not deliverable:
                m = _SENTENCE_CAP_RE.search(clause)
                if m and not _quoted(clause, m.start()):
                    seen.add("sentence_cap")
                    out.append(Constraint("sentence_cap", 1, clause[:120]))
    except Exception:  # noqa: BLE001 — a parser must never break a turn
        return out
    # "STRICT JSON on a single line": the JSON rule is the certain one and
    # every shape cap its noisier shadow (a fenced object is three lines
    # and still the answer; a JSON reply has no word or sentence count).
    if any(c.kind == "strict_json" for c in out):
        out = [c for c in out if c.kind not in ("line_cap", "word_cap", "sentence_cap")]
    return out


# NOT a rule here, deliberately: the opening-phrase mandate ("Start with:
# What it means to BE ghost"). `parse_start_with_phrase` reads that form as
# a reply-format mandate so finalize can HOIST the phrase-led segment; read
# as a refute it fired only on the originating request of the §4FD project
# (2 passed, 0 failed on the corpus), where the phrase names the topic to
# begin with, not the words to begin with. Ordering and format share one
# surface; a repair can afford the ambiguity, a label cannot.


# ── the reply as the model wrote it ──────────────────────────────────────
#
# The judged text is the MODEL-AUTHORED reply: finalize prepends the
# away-digest banners and appends system notes, and judging those produced
# self-refutes before (56221fad). Same separator contract as
# `agent._strip_leading_banners` / `autonomous_activity.summarize_turn_content`.
_BANNER_HEADS = ("**While you were away**",
                 "**Background activity while you were away:**")
_BANNER_SEP = "\n\n---\n\n"
_FENCE_LINE_RE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")
_MD_RE = re.compile(r"[*_`~>#]+")


def _reply_body(reply: str) -> str:
    from .reply_smoothing import strip_system_notes
    text = str(reply or "")
    for _ in range(4):
        if text.lstrip().startswith(_BANNER_HEADS) and _BANNER_SEP in text:
            text = text.split(_BANNER_SEP, 1)[1]
        else:
            break
    return strip_system_notes(text).strip()


def _unfenced_lines(text: str) -> List[str]:
    """The reply's lines with fence markers removed (fence bodies kept) —
    a line-by-line toggle, linear in the input (the first version's regex
    took 2.7 s on 33k backticks)."""
    out: List[str] = []
    for line in (text or "").splitlines():
        if _FENCE_LINE_RE.match(line):
            continue
        out.append(line)
    return out


def _plain(text: str) -> str:
    """Markdown emphasis / fences / headings removed, whitespace collapsed."""
    t = "\n".join(_unfenced_lines(text))
    t = _MD_RE.sub("", t)
    return " ".join(t.split())


def _words(text: str) -> List[str]:
    """Whitespace tokens carrying at least one letter or digit: "3.12.4",
    "Open-source", "2026-09-08", a URL are ONE word each (review #4)."""
    return [w for w in _plain(text).split() if re.search(r"[^\W_]", w)]


_ABBREV = frozenset({
    "i.e", "e.g", "etc", "vs", "approx", "dr", "mr", "mrs", "ms", "jr", "sr",
    "no", "st", "fig", "inc", "ltd", "p", "a", "m", "cf", "al", "ca", "esp",
})


def _sentences(text: str) -> int:
    """Sentence count that does not split on abbreviations, initials, list
    numbers or decimals (review #5): a boundary is a terminator followed by
    whitespace and an upper-case / quote / bracket opener, or the end."""
    t = _plain(text)
    if not t:
        return 0
    n = 0
    for m in re.finditer(r"[.!?…]+(?=\s+[A-ZΑ-Ω\"“(\[]|$)", t):
        before = t[:m.start()]
        tok = re.split(r"\s+", before.rstrip())[-1] if before.strip() else ""
        tok = tok.lstrip("(\"“'‘").lower()
        # an initial ("J."), a dotted pair ("i.e"), a short dotted token —
        # NOT a bare number: "About 26. Roughly" is two sentences, while
        # "Do 1. install" never reaches here (lowercase follows)
        if tok in _ABBREV or re.fullmatch(r"(?:[a-z]|[a-z]\.[a-z])", tok) or (tok.endswith(".") and len(tok) <= 4):
            continue
        n += 1
    return max(1, n)


# ── an honest inability report gets no verdict ───────────────────────────
#
# "The file is on your host machine, outside the sandbox — I can't access
# it" to "reply with just the number" violates the letter and is exactly
# the reply the 2026-07-31 honest-failure rule protects: refuting it writes
# "checked and WRONG" (0.0) and, on a tool turn, a repair directive that
# says "answer it now with NO new tool calls" — fabrication pressure. The
# inability has to LEAD (first sentence) or the reply has to be short and
# say so; a chess analysis that mentions "I can't recapture" in its second
# sentence is not an inability report.
#: An inability about the TASK: "I can't access / read / run / do …", or a
#: sentence that opens with the inability. "I can't directly capture it"
#: (a chess analysis, corpus 2d3fabeb) is neither and stays a violation.
_INABILITY_VERBS = (r"(?:access|read|open|reach|find|retrieve|run|do|complete|fulfil|comply|"
                    r"count|see|write|answer|help|provide|verify|check|process|fetch|download|"
                    r"execute|get|load|use|call|connect|locate|list|parse|transcribe|ingest)")
_INABILITY_RE = re.compile(
    r"(?:^\s*(?:sorry|unfortunately|i(?:'m| am) (?:sorry|afraid))?[,\s—-]*"
    r"(?:i\s+)?(?:can(?:no|')t|cannot|could\s*n[o']t|couldn't|unable|not\s+able|no\s+access|"
    r"δεν\s+μπορώ|δεν\s+έχω\s+πρόσβαση|αδυνατώ)\b)"
    r"|\b(?:can(?:no|')t|cannot|could\s*n[o']t|couldn't|unable\s+to|not\s+able\s+to|no\s+way\s+to)"
    r"\s+(?:\w+\s+){0,2}?" + _INABILITY_VERBS + r"\b"
    r"|\b(?:no\s+access|outside\s+(?:the|my|your)\s+sandbox|don't\s+have\s+access|not\s+possible\s+(?:from|for|to)|"
    r"δεν\s+έχω\s+πρόσβαση)\b",
    re.IGNORECASE)
_ACK_NO_MODAL_RE = re.compile(
    r"\b(?:(?:no|zero|0)\s+(?:results?|data|match(?:es)?|information|sources?|hits?)|"
    r"not\s+(?:found|available|retrieve)|nothing\s+(?:found|came\s+back|usable|useful)|"
    r"unavailable|failed|blocked|time[d\s-]*out|error|refused|empty|"
    r"δεν\s+(?:βρήκα|βρέθηκ|κατάφερα|υπάρχ|επέστρεψ)|καμία|κανένα|αποτυχ|σφάλμα|αδύνατ|αδυναμ)",
    re.IGNORECASE)
_ACK_RE = re.compile(
    r"\b(?:could\s*n[o']t|couldn't|cannot|can't|unable|(?:no|zero|0)\s+(?:results?|data|match(?:es)?|information|sources?|hits?)|"
    r"not\s+(?:found|available|able|retrieve)|nothing\s+(?:found|came\s+back|usable|useful)|"
    r"unavailable|failed|blocked|time[d\s-]*out|error|refused|empty|"
    r"δεν\s+(?:βρήκα|μπόρεσα|βρέθηκ|κατάφερα|υπάρχ|επέστρεψ)|καμία|κανένα|αποτυχ|σφάλμα|αδύνατ|αδυναμ)",
    re.IGNORECASE)


def _honest_inability(body: str) -> bool:
    """The inability leads: an inability phrase in the FIRST sentence, or a
    short reply (≤ 60 words) whose first sentence acknowledges the failure.
    A chess analysis with "I can't recapture" in its second sentence, or a
    long reply that mentions an error somewhere, is not an inability report."""
    p = _plain(body)
    first = re.split(r"[.!?…]+(?=\s|$)", p, maxsplit=1)[0] if p else ""
    if _INABILITY_RE.search(first[:200]):
        return True
    # the short branch reads FAILURE words, not the modals — "I can't
    # directly capture it" is not an inability and the modal alone would
    # exempt half the chess prose
    return len(_words(p)) <= 60 and bool(_ACK_NO_MODAL_RE.search(first[:200]))


# ── the checks ───────────────────────────────────────────────────────────

def _check_strict_json(body: str) -> Optional[str]:
    """The reply must BE a JSON document. One wrapping code fence is
    tolerated (a fenced object still parses for most consumers); prose
    around the object, or no object at all, is the violation."""
    t = "\n".join(_unfenced_lines(body)).strip()
    try:
        json.loads(t)
        return None
    except Exception:  # noqa: BLE001
        pass
    lo, hi = t.find("{"), t.rfind("}")
    if lo != -1 and hi > lo:
        try:
            json.loads(t[lo:hi + 1])
            return ("the request asked for strict JSON and nothing else, but "
                    "the reply carries prose around the JSON object")
        except Exception:  # noqa: BLE001
            pass
    return ("the request asked for strict JSON and nothing else, but the "
            "reply is not a JSON document")


def _norm_exact(s: str) -> str:
    return _plain(s).casefold().strip().rstrip(" \t.!?,;:")


def _check_exact(body: str, phrases: Tuple[str, ...]) -> Optional[str]:
    """The reply must be one of the phrases. Light decoration around it is
    tolerated (a period, bold, a short prefix); a reply that does not even
    contain any of them, or buries one in a paragraph, is the violation."""
    got = _norm_exact(body)
    wants = [w for w in (_norm_exact(p) for p in phrases) if w]
    if not wants or got in wants:
        return None
    if any(w in got and len(got) <= len(w) + 12 for w in wants):
        return None
    shown = " / ".join(phrases)
    return (f"the request asked for exactly '{shown}', but the reply "
            f"{'does not contain it' if not any(w in got for w in wants) else 'says more than that'}")


_NUMBER_WORD_RE = re.compile(
    r"\b(?:zero|none|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b", re.IGNORECASE)


def _check_number_only(body: str) -> Optional[str]:
    """A bare number, optionally with a unit or a short label, passes; a
    reply with no number at all, or a multi-sentence explanation, fails."""
    p = _plain(body)
    if re.fullmatch(r"[^\w]*[-+]?\d[\d,.]*(?:\s*%|\s+\w{1,12}){0,2}[^\w]*", p):
        return None
    words = _words(p)
    if not re.search(r"\d", p) and not _NUMBER_WORD_RE.search(p):
        return "the request asked for just the number, but the reply contains no number"
    if len(words) > 8 or _sentences(p) >= 2:
        return (f"the request asked for just the number, but the reply is "
                f"{len(words)} words of explanation")
    return None


def _check_word_cap(body: str, cap: int) -> Optional[str]:
    n = len(_words(body))
    if n > cap:
        return f"the request allowed at most {cap} word(s), but the reply has {n}"
    return None


def _check_line_cap(body: str, cap: int) -> Optional[str]:
    lines = [l for l in _unfenced_lines(body) if l.strip()]
    if len(lines) > cap:
        return f"the request asked for {cap} line(s), but the reply has {len(lines)}"
    return None


def _check_sentence_cap(body: str, cap: int) -> Optional[str]:
    n = _sentences(body)
    if n > cap + 1:          # one sentence of slack: sentence splitting is approximate
        return f"the request asked for {cap} sentence(s), but the reply has about {n}"
    return None


# NOT a rule here either: "the reply announces N items and lists fewer".
# Built for a live refute ("Claims 10 headlines but only lists 6"), measured
# on the corpus, and deleted: the reply behind that refute lists all ten —
# the LLM judge had read a packed (truncated) claim view — and of the nine
# fires the tightened rule still produced, six were a unit ("the last 48
# hours:"), a version ("PostgreSQL 20"), or a list split across sections
# ("all 19 tasks:" → a table of 14 DONE and a table of 5 open). A count
# against the reply's own list is a lexical proxy for "the list is
# complete", and every widening bought recall with precision.

# Every WEB retrieval this turn came back empty / unrelated / errored (the
# runtime gate's own assessment — one definition of "empty",
# `evidence_gate._empty_reason`, pinned against the tools' real strings)
# and the reply asserts an answer anyway, without saying that nothing was
# found. A reply that says so is an honest failure and gets no verdict.
#
# WEB RETRIEVALS ONLY. `execute`/`system_utility` were dropped first (three
# chess moves whose only call was a failed helper script, and "print what
# 0/0 does, I know it errors" answered correctly); then `recall` and
# `knowledge_base` (review #15): a memory miss says nothing about a correct
# general-knowledge answer, and absence never refutes ([[refute-only
# memory check]]). A successful `execute` COUNTS AS EVIDENCE (search failed
# → curl via execute → answer is a common live shape); only its failure is
# not an absent source. Browser rows count only for the page-reading ops.
_WEB_TOOLS = frozenset({
    "web_search", "darkweb_search", "deep_research", "darkweb_research",
    "fact_check", "news_headlines", "browser",
})
_BROWSER_READ_OPS = frozenset({"navigate", "extract_text", "get_text", "read", "open", ""})
_ASSERTIVE_MIN_WORDS = 30


def _tool_rows(tools_run: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalise the two row shapes this module is fed — the live loop's
    ``{"name", "content", "arguments"}`` and the trajectory store's
    ``{"name", "result", "arguments", "error"}`` — onto CONTENT ONLY. The
    stored ``error`` string is a copy of the result on 24 substantive rows
    in the corpus (a write with a syntax warning, a research result), and
    the live rows carry no such key; the gate's own content sniffer is the
    one definition of "empty" both shapes share (review #16)."""
    rows = []
    for t in tools_run or []:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if not (isinstance(content, str) and content.strip()):
            content = t.get("result")
        rows.append({
            "name": t.get("name"),
            "arguments": t.get("arguments") or t.get("args") or {},
            "content": content if isinstance(content, str) else ("" if content is None else str(content)),
            "_synthetic": bool(t.get("_synthetic")),
        })
    return rows


_BROWSER_HARD_EMPTY_RE = re.compile(r"^STATUS:\s*ERROR|^HTTP_STATUS:\s*[45]\d\d", re.M)


def _check_empty_evidence(body: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Per-row accounting, so a page that LOADED counts as evidence even
    when an earlier one 404'd (replay: a 404 then a 200 on the chess board,
    a 429 then a 200 on Google Maps — both refuted by the first cut, one of
    them human-approved)."""
    consulted, substantive, empties = 0, 0, []
    for r in rows:
        name = str(r.get("name") or "").strip().lower()
        if name not in _WEB_TOOLS or r.get("_synthetic"):
            continue
        content = str(r.get("content") or "")
        if name == "browser":
            args = r.get("arguments") if isinstance(r.get("arguments"), dict) else {}
            if str(args.get("operation") or "").lower() not in _BROWSER_READ_OPS:
                continue
            # The runtime gate's 40-character page floor is a STEER
            # threshold; for a label it is not evidence of nothing. Replay:
            # a 16-character page ("Start Soundscape") that the reply
            # described exactly was refuted. A page counts as empty here
            # only on a browser error or an HTTP 4xx/5xx — never on length.
            consulted += 1
            m = _BROWSER_HARD_EMPTY_RE.search(content)
            if m:
                empties.append("browser: " + m.group(0).replace("_", " ").lower())
            else:
                substantive += 1
            continue
        a = assess_turn_evidence([r])
        consulted += a.consulted
        substantive += a.substantive
        empties.extend(a.empty)
    if consulted == 0 or substantive > 0:
        return None

    class _A:                       # the shape the rest of the check reads
        empty = empties
    a = _A()
    # a successful command this turn is evidence (search failed → curl → answer)
    for r in rows:
        if str(r.get("name") or "").strip().lower() == "execute":
            c = str(r.get("content") or "")
            m = re.search(r"EXIT CODE:\s*(\d+)", c)
            # no output floor: `curl -o /dev/null -w "%{http_code}"` prints
            # three characters and IS the verification (replay 9fa6dc99)
            if m and m.group(1) == "0":
                return None
    if _ACK_RE.search(body):
        return None
    if len(_words(body)) < _ASSERTIVE_MIN_WORDS:
        return None
    reasons = "; ".join(a.empty[:3])
    return (f"every retrieval this turn came back empty or unrelated "
            f"({reasons}) and the reply asserts an answer without saying so")


def refute_turn_state(*, request: str, reply: str,
                      tools_run: Optional[Iterable[Dict[str, Any]]] = None
                      ) -> List[Tuple[str, str]]:
    """``[(rule, issue), …]`` for every state contradiction in this turn.

    ⚠ AN EMPTY LIST MEANS "NOTHING TO SAY", NEVER "THE REPLY IS FINE".
    Every caller must treat it as no-verdict.

    Total: never raises. A checker that can break a turn would be traded
    away the first time it did.
    """
    try:
        body = _reply_body(reply)
        if not body:
            return []
        issues: List[Tuple[str, str]] = []
        cons = mechanical_constraints(request)
        if cons and not _honest_inability(body):
            for c in cons:
                msg = None
                if c.kind == "strict_json":
                    msg = _check_strict_json(body)
                elif c.kind == "exact":
                    msg = _check_exact(body, tuple(c.value))
                elif c.kind == "number_only":
                    msg = _check_number_only(body)
                elif c.kind == "word_cap":
                    msg = _check_word_cap(body, int(c.value))
                elif c.kind == "line_cap":
                    msg = _check_line_cap(body, int(c.value))
                elif c.kind == "sentence_cap":
                    msg = _check_sentence_cap(body, int(c.value))
                if msg:
                    issues.append((c.kind, msg))
        msg = _check_empty_evidence(body, _tool_rows(tools_run))
        if msg:
            issues.append(("empty_evidence", msg))
        return issues
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return []
