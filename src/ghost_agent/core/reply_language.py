# src/ghost_agent/core/reply_language.py
"""Reply language versus request language — the one classifier (§4JR).

WHY. Req 84dc65c4 (2026-09-22, Slack): an English request ("i want to come
up with a new bjj submission … give me an action plan"), six tool turns all
reasoned in English, a 10.5 KB plan file written in English — and the final
reply came out in Greek. Nothing in the prompt asked for Greek: rule 5
LANGUAGE (§4JN) was in the system message, the recalled memories were
English, the Slack payload was a single message. The only Greek cue was the
profile's Athens address, which the model had been riffing on ("The
Thrakomakedon") when it wrote the long final generation. §4JN measured this
drift at 0.66% of English requests BEFORE the prompt rule; the rule did not
close it, and the finalize path had no language check at all — the judge
grades content, not script.

WHAT. Pure functions over the request text and the reply text:

  * `request_script`  — "latin" / "greek" / None (abstain): the script the
    user wrote in, decided only when it is unambiguous. Mixed-script
    requests, requests with too few letters, and Greeklish (Greek in Latin
    letters — "ti douleia kanei o sytistis sto strato", which the model
    rightly answers in Greek) all abstain.
  * `prose_lines`     — the reply minus fenced code, list items, table rows,
    headings and «quoted» lines: the part whose language is the model's own
    choice. A news digest quoting Greek headlines under an English lead is
    not a drift (§4JN's counting rule, moved here so the measurement and
    the guard share one definition).
  * `reply_language_mismatch` — `(expected, found)` such as
    ("English", "Greek") when the prose is in the other script, else None.

REFUTE-ONLY, PRECISION FIRST (§4EP/§4EQ, §4FY). None means "nothing to
say", never "the language is fine". Every abstention exists because a false
fire here costs a regeneration round AND a `failed` label. Measured on the
2,052 real user turns before wiring (`scripts/turn_state_replay.py --rule
reply_language`): the first cut fired 11 times, 4 of them wrong, each for a
different reason, and each reason is a rule here rather than a patch —
  * a bilingual answer to "Can you speak Greek?" sat at 0.54, so the
    Latin-request threshold is 0.6 (half-and-half is a choice, not a
    drift) and a request that names a language abstains outright;
  * a news digest's Greek summaries were INDENTED continuation lines under
    list items, so indented lines are the list's, not prose;
  * the agent's own canned "[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard
    limit…" note on a Greek ask is code-authored text, not the model's
    choice — every `[ATTEMPT_ABORTED_*]` note and the forced-final
    fallback abstain (the reply-shape check refutes those as no answer);
  * a restaurant list with Greek descriptions copied from Greek sources
    sat at 0.556 — under the 0.6 line.
The Greek-request threshold stays at §4JN's 0.2, and a reply must carry
enough letters to be judged at all. Consumers: `turn_state_check.refute_turn_state` (the
`reply_language` rule — both delivery paths, the verdict record and the
labels) and the finalize-time regeneration in `_run_internal_turn` (the
non-streaming path, where the draft has not been delivered yet).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

__all__ = [
    "GREEK", "LATIN", "script_share", "prose_lines", "request_script",
    "reply_language_mismatch", "LANGUAGE_OF", "EN_TO_EL_MIN_SHARE",
    "EL_TO_EN_MAX_SHARE",
]

GREEK = re.compile("[\\u0370-\\u03ff\\u1f00-\\u1fff]")
LATIN = re.compile(r"[A-Za-z]")

#: Lines whose language is dictated by the content, not chosen by the
#: model: list items, numbered items, table rows, headings, «quoted» lines,
#: and INDENTED lines (a list item's continuation — the news digest's
#: per-headline summary — or an indented code block).
_SKIP_LINE = re.compile(r"^(?:\s{2,}|\t|\s*(?:[-*•]|\d+[.)]|\||#{1,6}\s|\*\*[^*]*«|«))")
_FENCE_LINE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")
#: A request that names a language is asking about or for one; the reply's
#: language is then content, not a drift ("Can you speak Greek?" → a
#: bilingual answer; "translate this to Greek"; "answer in English").
_LANGUAGE_ASK = re.compile(
    r"greek|english|ελληνικ|αγγλικ|translat|μετ[άα]φρασ|\blanguage\b|γλ[ώω]σσα",
    re.IGNORECASE)

#: The reply's prose must carry this many Greek+Latin letters to be judged.
MIN_REPLY_LETTERS = 60
#: Prose Greek share ABOVE which a Latin-script request was answered in
#: Greek (0.6: a half-and-half bilingual reply is not a drift), and BELOW
#: which a Greek request was answered in English (§4JN's 0.2).
EN_TO_EL_MIN_SHARE = 0.6
EL_TO_EN_MAX_SHARE = 0.2
#: The request must carry this many letters to declare its script.
MIN_REQUEST_LETTERS = 12
#: A Latin-script request is Greek-in-Latin-letters when it carries this
#: many distinctive Greeklish tokens; the model answers those in Greek and
#: that is not a drift. Only forms that are not English words (no "den",
#: "min", "auto", "apo"); two hits, so one stray name ("Kai") decides nothing.
GREEKLISH_MIN_HITS = 2
_GREEKLISH = frozenset({
    "einai", "eimai", "eisai", "kanei", "kaneis", "kanw", "poios", "poia", "poies",
    "poioi", "giati", "pws", "mporeis", "mporw", "mporo", "thelw", "thelo", "theleis",
    "exei", "exeis", "exw", "exoume", "prepei", "sto", "stin", "ston", "stous", "stis",
    "kai", "oles", "oloi", "ayto", "ekei", "pou", "opws", "otan", "molis", "akoma",
    "akomh", "twra", "tora", "simera", "avrio", "xthes", "xtes", "arketa", "kalimera",
    "kalispera", "geia", "efharisto", "efxaristo", "parakalo", "parakalw", "ellada",
    "elliniki", "ellinika", "douleia", "spiti", "strato",
})
_TOKEN = re.compile(r"[A-Za-z]+")
#: The runtime's abort notes (`_with_abort_note` in core/agent.py) — the same
#: marker shape `distill.outcome_heuristics` and `agent.reply_carries_abort_marker`
#: read; kept local because agent.py imports this module through the
#: turn-state checker.
_ABORT_MARKER = re.compile(r"\[ATTEMPT_ABORTED_[A-Z_]+\]")

LANGUAGE_OF = {"latin": "English", "greek": "Greek"}


def script_share(text: str) -> float:
    """Greek letters over Greek+Latin letters; 0.0 when there are none."""
    g = len(GREEK.findall(text or ""))
    l = len(LATIN.findall(text or ""))
    return g / (g + l) if g + l else 0.0


def _letters(text: str) -> int:
    return len(GREEK.findall(text or "")) + len(LATIN.findall(text or ""))


def prose_lines(reply: str) -> str:
    """The reply minus fenced code, list items, table rows, headings and
    «quoted» lines — the parts whose language is the model's own choice.
    Fence bodies are dropped whole (a line-by-line toggle, linear)."""
    keep: List[str] = []
    in_fence = False
    for ln in (reply or "").splitlines():
        if _FENCE_LINE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence or not ln.strip() or _SKIP_LINE.match(ln):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _greeklish(text: str) -> bool:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    return sum(1 for t in toks if t in _GREEKLISH) >= GREEKLISH_MIN_HITS


def request_script(request: str) -> Optional[str]:
    """"latin" or "greek" when the request is unambiguously in one script;
    None (abstain) when it is mixed, too short, or Greeklish."""
    text = str(request or "")
    if _letters(text) < MIN_REQUEST_LETTERS or _LANGUAGE_ASK.search(text):
        return None
    share = script_share(text)
    if share >= 0.5:
        return "greek"
    if share < 0.05:
        return None if _greeklish(text) else "latin"
    return None  # mixed — the user chose both; nothing to enforce


def reply_language_mismatch(request: str, reply: str,
                            prior_user_messages: Iterable[str] = ()) -> Optional[Tuple[str, str]]:
    """``(expected, found)`` — e.g. ``("English", "Greek")`` — when the
    reply's prose is in the other script from the request; else None.

    ``prior_user_messages`` are the conversation's earlier user turns: a
    standing instruction there that names a language ("answer in Greek from
    now on", "μίλα μου αγγλικά") is the user's choice for the whole
    conversation, so the check abstains — the per-message rule must not
    overrule an instruction the user gave one message earlier. Callers
    without the conversation (the turn-state tier) get the per-message rule.

    None is "nothing to say", never a pass. Never raises.
    """
    try:
        want = request_script(request)
        if want is None:
            return None
        if any(_LANGUAGE_ASK.search(str(m or "")) for m in (prior_user_messages or ())):
            return None  # a standing language instruction earlier in the conversation
        from .reply_shape_check import refute_no_answer_fallback
        if _ABORT_MARKER.search(str(reply or "")) or refute_no_answer_fallback(reply):
            return None  # code-authored abort / fallback text — not the model's choice
        prose = prose_lines(reply)
        if _letters(prose) < MIN_REPLY_LETTERS:
            return None
        share = script_share(prose)
        if want == "latin" and share > EN_TO_EL_MIN_SHARE:
            return ("English", "Greek")
        if want == "greek" and share < EL_TO_EN_MAX_SHARE:
            return ("Greek", "English")
        return None
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return None
