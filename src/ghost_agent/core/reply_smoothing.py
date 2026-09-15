# src/ghost_agent/core/reply_smoothing.py
"""Finalize-time reply smoothing (2026-07-17).

The turn loop accumulates every iteration's visible text into the reply,
so a multi-tool fix arrives as stacked working narration ("Let me fix
both:", "Now add the resize logic in openWindow:") and often states the
summary twice — once before a verify/restart step, once after (operator
report: the WebOS minesweeper turn). The system prompt already forbids
narrating during tool use; the model doesn't reliably comply, so the
delivered reply is cleaned deterministically here.

Design constraints (option picked by the operator: finalize scrub, no
client changes):

* Pure text → text. The live stream still shows the narration as
  progress; only the delivered/persisted reply is smoothed.
* Callers gate on multi-tool turns — a single-tool or conversational
  reply is never touched.
* Fenced code blocks are atomic and never dropped.
* Only three shapes are ever removed, all non-final:
    1. connective working narration — a short paragraph opening with an
       agent-voice beat ("Let me…", "I'll…", "Good, …"), or with a
       temporal lead ("Now…", "Next,…") that a later paragraph RESTATES
       (§4FV, 2026-09-10: the bare lead also opens an instruction);
    2. a superseded summary group — a lead-in-plus-list block whose
       content is restated by a later near-duplicate group (the double
       summary), judged by content-word overlap;
    3. a stale announcement — a short paragraph that says work is IN
       PROGRESS ("… Now ingesting into the knowledge base.") when a later
       paragraph reports that same work as DONE and RESTATES the
       paragraph's content (2026-09-09).
* Fail-open: anything unmatched stays; if smoothing would empty the
  reply it returns the original.
"""

from __future__ import annotations

import re
from typing import List

# Progress connectives that open working narration. Anchored at the start
# of a paragraph; matched case-insensitively. Split in two by §4FV
# (2026-09-10) because the two halves carry different risk:
#
# * AGENT-VOICE beats — the model narrating its OWN next move ("Let me
#   fix both:", "I'll build the parser."). First person or an
#   acknowledgement; never an instruction addressed to the user. Dropped
#   on sight, as they have been since 2026-07-17.
# * TEMPORAL leads — "Now …", "Next, …", "Then …", "First, …". The same
#   words open a beat AND a step of a procedure the user is meant to
#   follow ("First, back up the database…", "Next, restarting the service
#   will pick it up — run `systemctl restart ghost`."). Pass 1 dropped
#   those too: §4FT's reviewers found four such deletions of user-facing
#   text, and the ≥2-tool gate was the only thing bounding the damage.
#   These now need the same evidence shape 3 needs — the reply must
#   RESTATE the paragraph (a stale beat is a beat the reply repeats).
#
# A temporal lead followed by an agent-voice beat ("Now let me fix both:",
# "Then I'll restart it") is a beat: the lead is stripped and the rest
# re-tested, so the 2026-07-17 coverage survives the narrowing.
_BEAT_RE = re.compile(
    r"^(?:let me\b|let's\b|good[,.! ]|okay\b|ok[,.! ]|alright\b|"
    r"great[,.! ]|perfect[,.! ]|time to\b|"
    r"i'll\b|i will\b|i need to\b)",
    re.IGNORECASE,
)
_TEMPORAL_LEAD_RE = re.compile(
    r"^(?:now[, ]|next[, ]|first[, ]|then[, ])",
    re.IGNORECASE,
)
# A narration paragraph is a beat, not content — long paragraphs opening
# with "Now …" are treated as content and kept.
_MAX_NARRATION_CHARS = 300

# First line of a block that "fulfils" a lead-in: list item or table row.
_LIST_START_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]\s|\|)")

_WORD_RE = re.compile(r"[0-9A-Za-zΑ-Ωά-ώα-ω_']+")

# Superseded-group thresholds: an earlier group is dropped when a later
# group shares ≥ this fraction of content words AND is at least this
# fraction of its size (a one-line echo must not delete a full summary).
_SUPERSEDE_JACCARD = 0.55
_SUPERSEDE_MIN_SIZE_RATIO = 0.6


def _split_blocks(text: str) -> List[str]:
    """Blank-line paragraph split that keeps ``` fences atomic (blank
    lines inside a fence do not split it; an unclosed fence swallows the
    rest of the text into one block)."""
    blocks: List[str] = []
    cur: List[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _has_fence(block: str) -> bool:
    return "```" in block


def _words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


#: Sentence boundary, good enough for "does this paragraph END on a beat?".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: ⚠ "Let me know if you want the tests too." is NOT a beat — it is an offer
#: addressed to the user, and it is the single most common way a reply ends.
#: The trailing-beat rule below would otherwise treat it as narration.
_OFFER_RE = re.compile(r"^let\s+(?:me|us)\s+know\b", re.IGNORECASE)


def _trailing_beat(block: str) -> bool:
    """Does this paragraph END on an agent-voice beat?

    §4GO (2026-09-14). Pass 1 anchored the connective at the START of the
    paragraph, and real beats rarely start there: the model reports what it
    just saw and THEN announces the next move —

        "The screenshot captured the embed widget but not the notification
         image clearly. Let me navigate to the embed view …"
        "The screenshot was captured. Now I'll READ THE IMAGE ITSELF …"

    Four such paragraphs opened a delivered reply (req 7b2da5be) and the
    smoother kept every one, because each opens with an observation. Shape 3
    had already learned this lesson for announcements ("Pass 1 missed it
    because the paragraph does not OPEN with a connective"); this is the
    same lesson for beats.
    """
    sents = [x for x in _SENTENCE_SPLIT_RE.split(block.strip()) if x.strip()]
    if len(sents) < 2:
        return False                      # one sentence: that is shape 1
    last = sents[-1].lstrip()
    if _OFFER_RE.match(last):
        return False
    m = _TEMPORAL_LEAD_RE.match(last)
    if m:
        last = last[m.end():].lstrip()
    return bool(_BEAT_RE.match(last))


def _is_narration(block: str, later_blocks: List[str] = ()) -> bool:
    """Is this paragraph a working-narration beat?

    `later_blocks` are the paragraphs that follow it; a TEMPORAL lead is a
    beat only when one of them restates it (§4FV), and so is a beat that
    does not open the paragraph (§4GO). Called with no later blocks both of
    those halves are inert — an ambiguous paragraph is content until the
    reply proves otherwise.
    """
    stripped = block.strip()
    if not stripped or _has_fence(stripped):
        return False
    if len(stripped) > _MAX_NARRATION_CHARS:
        return False
    if _LIST_START_RE.match(stripped):
        return False
    if _BEAT_RE.match(stripped):
        return True
    later_prose = [b for b in later_blocks if not _has_fence(b)]
    m = _TEMPORAL_LEAD_RE.match(stripped)
    if m:
        # "Now let me …" / "Then I'll …": the lead is decoration on a beat.
        if _BEAT_RE.match(stripped[m.end():].lstrip()):
            return True
        return bool(later_prose) and _restated_later(stripped, later_prose)
    # §4GO: the beat that arrives AFTER the observation it follows. Same
    # evidence shape the temporal half and shape 3 require — a beat that is
    # stale is a beat the reply repeats, and a paragraph the reply does not
    # repeat is content, whatever it opens with.
    if _trailing_beat(stripped):
        return bool(later_prose) and _restated_anywhere_later(stripped,
                                                              later_prose)
    return False


# ---------------------------------------------------------------------------
# Shape 3 — stale announcements (2026-09-09, request d50a34bd; tightened the
# same day after review)
# ---------------------------------------------------------------------------
# The delivered reply opened with
#     Download succeeded (16 MB, valid). Now ingesting into the knowledge base.
# and its second paragraph said the manual "has been successfully downloaded
# and ingested". The opener is the model's turn-3 progress beat; delivered as
# the first line, it told the operator a finished job was still running.
# Pass 1 missed it because the paragraph does not OPEN with a connective —
# "Now ingesting…" is its SECOND sentence.
#
# Three conditions, ALL required, on a short non-final paragraph:
#   1. it ANNOUNCES work in the progressive behind a lead adverb ("Now
#      ingesting…", "Next, indexing…", "Currently downloading…");
#   2. a LATER prose paragraph reports that verb as done (ingested /
#      ingestion / downloaded) — naive morphology, irregular verbs fail open;
#   3. the paragraph is RESTATED: at least half of its content words (light
#      stemming) appear in ONE later prose paragraph.
# Condition 3 is what a first version lacked, and without it the rule
# deleted user-facing text five times out of six in review: "Next, restarting
# the service will pick it up — run `systemctl restart …`" (an instruction)
# because "restarted" appeared later; "Next, deploying to production would
# need your sign-off" (a request) because staging had been "deployed"; "Now,
# acting on the two…" because "action" stems to "act". A beat that is stale
# is a beat the reply repeats; a paragraph the reply does not repeat is
# content, whatever tense it uses. Fenced blocks are neither candidates nor
# evidence. The lead adverb is required on purpose — a bare gerund is a
# subject ("Testing is complete.") — and a verb followed by an auxiliary is
# a statement, not an announcement. Fail-open, like every rule here.
_ANNOUNCE_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)"                                # a sentence start (not ':')
    r"(?:and\s+|so\s+|ok(?:ay)?,?\s+)?"                   # optional pre-lead
    r"(?:now|next|then|currently)"                        # the lead adverb
    r"[,\s]+([A-Za-z]{4,}ing)\b"                          # the progressive verb
    r"(?!\s+(?:is|are|was|were|has|have|had|will|would|should|can|could|"
    r"does|did|done|complete[ds]?|finished|passed|failed|succeeded)\b)",
    re.IGNORECASE,
)
_COMPLETION_SUFFIXES = ("ed", "d", "ion", "ions", "ation", "ations")
_RESTATED_MIN_STEMS = 3
_RESTATED_FRACTION = 0.5
_STOPWORDS = frozenset("""
the and into with for now next then this that has have been was were are will
from its of to a an in on at by is be as or it but not you your our we i my
""".split())


def _announced_verbs(block: str) -> List[str]:
    """The progressive verbs a paragraph announces ("ingesting")."""
    return [m.group(1).lower() for m in _ANNOUNCE_RE.finditer(block)]


def _completed_later(verb_ing: str, later_text: str) -> bool:
    """Does the text AFTER the announcement report this verb as done?
    "ingesting" → ingested / ingestion; "downloading" → downloaded."""
    stem = verb_ing[:-3]
    if len(stem) < 3:
        return False
    pat = r"\b" + re.escape(stem) + r"(?:" + "|".join(_COMPLETION_SUFFIXES) + r")\b"
    return re.search(pat, later_text, re.IGNORECASE) is not None


def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("ation", "ing", "ion", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)] + ("y" if suf == "ied" else "")
    return w


def _content_stems(text: str) -> set:
    return {_stem(w) for w in _WORD_RE.findall(text)
            if len(w) >= 3 and w.lower() not in _STOPWORDS}


def _restated_anywhere_later(block: str, later_prose: List[str]) -> bool:
    """Is at least half of this paragraph's content covered by the REST of
    the reply, taken together?

    §4GO. `_restated_later` asks one later paragraph to carry the whole
    restatement, which is right for a summary group (a summary is restated
    by a summary) and wrong for a work beat: the beat announces one step,
    and the answer reports its result spread across several paragraphs.
    Measured on req 7b2da5be, the four narration paragraphs scored 0.55 /
    0.44 / 0.36 / 0.26 against the best SINGLE paragraph and 0.82 / 0.69 /
    0.64 / 0.37 against the union — the rule was inert on three of four
    beats that the reply demonstrably does repeat.

    Used ONLY by the trailing-beat shape, whose last sentence is
    first-person agent voice and therefore never an instruction to the
    user — the case the per-paragraph threshold was tightened to protect.
    """
    mine = _content_stems(block)
    if len(mine) < _RESTATED_MIN_STEMS:
        return False
    union = set()
    for para in later_prose:
        union |= _content_stems(para)
    if not union:
        return False
    return len(mine & union) / len(mine) >= _RESTATED_FRACTION


def _restated_later(block: str, later_prose: List[str]) -> bool:
    """Is at least half of this paragraph's content restated by ONE later
    prose paragraph? The test that separates a repeated beat from content."""
    mine = _content_stems(block)
    if len(mine) < _RESTATED_MIN_STEMS:
        return False
    for para in later_prose:
        theirs = _content_stems(para)
        if theirs and len(mine & theirs) / len(mine) >= _RESTATED_FRACTION:
            return True
    return False


def _is_stale_announcement(block: str, later_blocks: List[str]) -> bool:
    stripped = block.strip()
    if not stripped or _has_fence(stripped):
        return False
    if len(stripped) > _MAX_NARRATION_CHARS:
        return False
    if _LIST_START_RE.match(stripped):
        return False
    verbs = _announced_verbs(stripped)
    if not verbs:
        return False
    later_prose = [b for b in later_blocks if not _has_fence(b)]
    if not later_prose:
        return False
    later = "\n".join(later_prose)
    if not any(_completed_later(v, later) for v in verbs):
        return False
    return _restated_later(stripped, later_prose)


def _group_blocks(blocks: List[str]) -> List[List[int]]:
    """Group a ':'-terminated lead-in with the list blocks that follow it
    (the "summary group" shape); everything else is its own group.
    Returns index lists into ``blocks``."""
    groups: List[List[int]] = []
    i = 0
    while i < len(blocks):
        cur = [i]
        stripped = blocks[i].strip()
        if stripped.endswith(":") and not _has_fence(stripped):
            j = i + 1
            while (j < len(blocks)
                   and _LIST_START_RE.match(blocks[j].strip())
                   and not _has_fence(blocks[j])):
                cur.append(j)
                j += 1
            i = j
        else:
            i += 1
        groups.append(cur)
    return groups


# ---------------------------------------------------------------------------
# System-note stripping (2026-08-01, req 56221fad post-mortem)
# ---------------------------------------------------------------------------
# Finalize appends operator-facing notes to the delivered reply (the
# ⚠ Unverified mutation note, the plan-postcondition note, the uncertainty
# risk summary) and prepends the async-verdict correction banner. Those are
# OUR text, not the model's — yet two consumers were reading them as if the
# model wrote them:
#   * the hedge auto-scan flagged the Unverified note's "I cannot confirm it
#     works" as a 40%-confidence assumption, and the risk summary then
#     re-rendered that self-echo into the reply — the garbled duplicated
#     footer the 56221fad late refute called "truncated";
#   * the verifier judged the claim WITH the notes, so "does not confirm the
#     update" was literally our own INCOMPLETE disclaimer contradicting the
#     model's ✅ confirmation lines.
# `strip_system_notes` removes exactly those appended shapes so scanners and
# judges see the model-authored reply. The markers are matched with their
# separator context (the `\n---\n` we append with / the banner's exact
# prefix), so ordinary model text mentioning "assumptions" is untouched.

# Trailing notes are stripped TERMINALLY and iteratively: each appended
# note is a single blank-line-free block riding the very end of the reply,
# so only an end-anchored block whose body contains no blank line is ours.
# A model-authored "**Assumptions I made:**" section followed by real
# blank-line-separated content fails the match and survives (fail-open —
# review catch 2026-08-01: the earlier earliest-marker cut deleted
# everything after a mid-reply lookalike, substance included).
_TRAILING_NOTE_TAIL_RE = re.compile(
    # `\n{0,4}`, not `\n*` (§4FY review): the appended seam is exactly
    # "\n\n---\n", and the unbounded run was quadratic on a reply of
    # newlines (2.7 s on 100k) — paid on EVERY reply by every consumer.
    r"\n{0,4}---\n(?:\*\*⚠ Unverified:\*\*|\*\*Plan check:\*\*|"
    # "Verifier note:" was MISSING from this list (§4L Lens-A MINOR-1)
    # while being appended BEFORE the hedge scan — a verifier note
    # quoting first-person text survived the strip and fired the hedge
    # regex, putting uncertainty_pressure on a refuted (label-0) turn:
    # the same label-echo channel as the λ leak, one banner over.
    r"\*\*Verifier note:\*\*|"
    r"\*\*Things I'm not certain about:\*\*|\*\*Assumptions I made:\*\*|"
    # §4FY: the project-promotion nudge ("💡 This looks like ongoing work
    # (…). Want me to promote it to a tracked project?") is appended by
    # finalize with the same `\n\n---\n` seam and was MISSING here — the
    # state-aware judge read it as prose around a strict-JSON answer and
    # refuted a compliant chess turn on the replay corpus (5ef33e14).
    r"💡 This looks like ongoing work|"
    # §4FY review R1: two more appenders shared the seam and were missing —
    # the principle gate's note and the §4ER label request. The class is now
    # pinned from the SOURCE: every `f"{final_ai_content}\n\n---\n<head>"`
    # in agent.py must be stripped here (tests/test_4fy_turn_state_check.py).
    r"\*\*Self-check \(principle\):\*\*|"
    r"\*This was one of the shakier answers)"
    r"(?:(?!\n\n).)*$",
    re.DOTALL,
)
# The correction banner is PREPENDED (see _consume_pending_corrections) with
# a fixed shape ending in a blank-line-separated rule.
_CORRECTION_BANNER_RE = re.compile(
    r"^⚠️ \*\*Correction to my previous answer:\*\* .*?\n\n---\n\n",
    re.DOTALL,
)


# ──────────────────────────────────────────────────────────────────────
# Risk-governor checkpoint answers (2026-09-15)
#
# When the governor fires it injects a numbered checkpoint: state what is
# CONFIRMED vs ASSUMED, name the SINGLE smallest check, or STOP and
# report. The model answers it in ordinary prose, and because the answer
# is emitted on an iteration that then calls more tools, it lands in the
# accumulated reply. Live (req 4b518a82) five such answers stacked up and
# shipped as the opening third of a forensic report — which then said all
# of it again in its own sections.
#
# This is deliberately NOT a prose heuristic standing alone. The caller
# only offers segments that are provably not the final answer (the
# iteration went on to call tools), and only while a steer is active; the
# match below is the third condition, not the only one. That ordering is
# what keeps a legitimate "**Confirmed:** … **Not confirmed:** …" section
# in a DELIVERABLE safe: it is written on the last iteration, so it is
# never a candidate.
_CHECKPOINT_MARKERS = (
    re.compile(r"\bconfirmed\b", re.IGNORECASE),
    re.compile(r"\bassum(?:ed|e|ption)\b", re.IGNORECASE),
    re.compile(r"\b(?:single|smallest|most valuable)\b[^.\n]{0,40}\bcheck\b",
               re.IGNORECASE),
    # Directive 3 splits in two: the model declares it has STOPPED, then
    # declares it is REPORTING anyway. The live stop-declaration carried
    # one of each and nothing else, so folding them into a single marker
    # left it at one hit and unrecognised.
    re.compile(r"\bno new (?:information|info)\b|\benough rounds\b|"
               r"\bstopped producing\b", re.IGNORECASE),
    re.compile(r"\breporting the (?:honest )?partial\b|"
               r"\bhonest partial answer\b|\bwhat is blocked\b", re.IGNORECASE),
)

# Two distinct directives must be answered before a segment is treated as
# a checkpoint answer. One alone ("confirmed") is ordinary English.
_CHECKPOINT_MIN_MARKERS = 2


def is_governor_checkpoint_answer(text: str) -> bool:
    """True when a segment reads as an answer to the risk-governor
    checkpoint (≥2 of its directives addressed).

    Shape test only — the caller supplies the structural evidence that
    the segment is interim. See ``core/risk.STEER_DIRECTIVE_TERMS``.
    """
    if not text or not text.strip():
        return False
    hits = sum(1 for pat in _CHECKPOINT_MARKERS if pat.search(text))
    return hits >= _CHECKPOINT_MIN_MARKERS


def drop_checkpoint_segments(text: str, segments) -> str:
    """Remove recorded checkpoint answers from an assembled reply.

    Exact-substring removal of segments the turn loop recorded, so
    nothing is matched by resemblance. Fail-open in both directions: a
    segment that is no longer present (a later stage rewrote it) is
    skipped, and if removal would leave nothing the original is returned.
    """
    if not text or not segments:
        return text
    out = text
    for seg in segments:
        # No "is it present?" pre-check: `str.replace` already no-ops on
        # an absent segment, and an explicit guard for it was an
        # unfalsifiable branch (no mutant of it could change behaviour —
        # found by the §R2 battery, removed rather than left standing).
        # The empty string is handled by the shape test, which rejects it.
        seg = (seg or "").strip()
        if not is_governor_checkpoint_answer(seg):
            continue
        out = out.replace(seg, "", 1)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out if out else text


def strip_system_notes(text: str) -> str:
    """Return *text* without finalize-appended system notes (trailing
    Unverified / Plan check / risk-summary blocks, leading correction
    banner). Model-authored content is preserved byte-for-byte."""
    if not text:
        return text
    out = _CORRECTION_BANNER_RE.sub("", text, count=1)
    while True:
        m = _TRAILING_NOTE_TAIL_RE.search(out)
        if not m:
            break
        out = out[:m.start()]
    return out.rstrip("\n") if out != text else out


# ---------------------------------------------------------------------------
# Unparsed tool-call markup (2026-09-09, §4FS; hardened the same day after review)
# ---------------------------------------------------------------------------
# 10 of 2,185 delivered replies since July contained a literal
#     <tool_call>
#     <function=execute>
#     <parameter=command>python3 -c "…
# — a call whose body failed to parse that the loop then delivered to the
# user as prose, verbatim, with the file it was meant to write never
# written. Those were STREAMED replies: the non-stream finalize already
# scrubs this markup (agent.py, the `<(tool_call|tool|function)` re.sub)
# and only lacked the NOTE; the stream path scrubbed the live text and
# persisted the raw copy. This module owns the one predicate both paths
# use, so the user-facing note and the durable record agree.
#
# What counts as an unparsed CALL: the call dialects (`<tool_call>`, the
# bare `<function=…>` / `<function name="…">`, and `<tool …>`), not
# `<tool_response>` (an echoed RESULT — the tool ran). Not preceded by a
# backtick (inline code: the user asked about the syntax). Not the
# cognitive watchdog's own synthetic `replan` call, which is appended to
# the durable text on purpose and replayed by the trajectory machinery.
# Fences: only BALANCED pairs are fences (an unclosed ``` must not shield
# everything after it); a match that starts inside one is documentation.
_CALL_MARKUP_RE = re.compile(
    r"(?<!`)<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|\Z)",
    re.DOTALL | re.IGNORECASE)
_PRESERVED_CALL_RE = re.compile(
    r"<function(?:=|\s+name=[\"']?)replan\b", re.IGNORECASE)
_BALANCED_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
UNPARSED_TOOL_CALL_NOTE = (
    "[A tool call in this reply could not be parsed and was NOT executed — "
    "the step it described did not happen.]")


def _fence_spans(text: str) -> List[tuple]:
    return [(m.start(), m.end()) for m in _BALANCED_FENCE_RE.finditer(text)]


def _call_markup_matches(text: str) -> List[re.Match]:
    """The unparsed-call blocks in `text`: not inline code, not fenced
    documentation, not the watchdog's replan marker."""
    if not text or "<" not in text:
        return []
    fences = _fence_spans(text)
    out = []
    for m in _CALL_MARKUP_RE.finditer(text):
        if any(a <= m.start() < b for a, b in fences):
            continue
        if _PRESERVED_CALL_RE.search(m.group(0)):
            continue
        out.append(m)
    return out


def unparsed_call_markup_present(text: str) -> bool:
    """The predicate behind the note, shared by both delivery paths."""
    return bool(_call_markup_matches(text))


def strip_unparsed_tool_calls(text: str) -> str:
    """Remove unparsed tool-call markup from a reply and say so once.

    Whole-text (a leaked call spans blank lines). Whitespace is collapsed
    only at the seams the removals leave, never elsewhere — the user's own
    spacing is content. A reply that would become empty keeps the note
    alone. Text without any such markup is returned untouched.
    """
    matches = _call_markup_matches(text)
    if not matches:
        return text
    pieces: List[str] = []
    last = 0
    for m in matches:
        if m.start() < last:
            continue
        pieces.append(text[last:m.start()])
        last = m.end()
        # swallow the whitespace the block leaves on either side; one
        # paragraph break replaces it
        while last < len(text) and text[last] in "\n\t ":
            last += 1
        if pieces[-1].rstrip() != pieces[-1]:
            pieces[-1] = pieces[-1].rstrip() + "\n\n"
    pieces.append(text[last:])
    out = "".join(pieces).rstrip()
    return (out + "\n\n" + UNPARSED_TOOL_CALL_NOTE) if out else UNPARSED_TOOL_CALL_NOTE


def _is_answer_block(block: str) -> bool:
    """Does this paragraph carry an ANSWER — something that is neither a
    work beat nor a note this module itself appended? Used to decide
    whether a run of beats has anything after it (§4GO)."""
    s = (block or "").strip()
    if not s or s == UNPARSED_TOOL_CALL_NOTE.strip():
        return False
    if _has_fence(s):
        return True
    return not (_BEAT_RE.match(s) or _trailing_beat(s))


def smooth_reply(text: str) -> str:
    """Remove working narration and superseded summary groups from an
    accumulated multi-turn reply. See module docstring for the rules."""
    if not text or "\n\n" not in text:
        return text
    blocks = _split_blocks(text)
    if len(blocks) < 2:
        return text

    drop = [False] * len(blocks)

    # Pass 1 — connective working narration and stale announcements
    # (never the final block).
    for i in range(len(blocks) - 1):
        if (_is_narration(blocks[i], blocks[i + 1:])
                or _is_stale_announcement(blocks[i], blocks[i + 1:])):
            drop[i] = True

    # Pass 1b — a RUN of work beats (§4GO). A trailing-beat paragraph that
    # fails its own restatement test is kept on its own: one observation
    # plus "Let me fix that" may be the only place a finding appears. But
    # sitting NEXT TO another beat it is part of a working log, and a
    # working log is not an answer — the reply moved on from it by writing
    # the next beat. Measured on req 7b2da5be: four consecutive beats
    # opened the delivered reply and the fourth ("The extract_text on
    # single=1 gave the same capped preview. Let me take a full-page
    # screenshot …") scored 0.37 against the rest of the reply, because it
    # was restated in different words ("I captured it via the embed widget
    # and ran vision OCR on the image directly"). Synonymy is exactly what
    # a stem test cannot see; adjacency is structural and needs no lexicon.
    # Never the final block, and never a LONE beat.
    # ⚠ AND ONLY WHEN AN ANSWER FOLLOWS THE RUN. A working log is noise
    # because the answer comes after it; with nothing but notes behind it,
    # the log IS the reply and dropping a member deletes the only substance
    # the user gets. Measured: the 5e9b9320 leak ("The initial diff
    # conflated tables and indexes. Let me refine the analysis…") is a beat
    # paragraph carrying the one finding that reply ever delivered, and the
    # first version of this rule ate it because the beat above it was
    # dropped.
    for i in range(len(blocks) - 1):
        if drop[i] or not _trailing_beat(blocks[i]):
            continue
        if len(blocks[i].strip()) > _MAX_NARRATION_CHARS:
            continue
        adjacent = (i and drop[i - 1]) or (i + 1 < len(blocks) - 1 and drop[i + 1])
        if not adjacent:
            continue
        if any(_is_answer_block(blocks[j])
               for j in range(i + 1, len(blocks)) if not drop[j]):
            drop[i] = True

    # Pass 2 — superseded summary groups. Compare each earlier group's
    # content words against every LATER group; a later near-duplicate of
    # comparable size supersedes the earlier statement (keep the last —
    # it is the one written after verification/restart, i.e. the truest).
    groups = _group_blocks(blocks)
    group_words = [
        _words("\n".join(blocks[k] for k in g)) for g in groups
    ]
    for a in range(len(groups)):
        if any(_has_fence(blocks[k]) for k in groups[a]):
            continue
        wa = group_words[a]
        if not wa:
            continue
        for b in range(a + 1, len(groups)):
            wb = group_words[b]
            if not wb or len(wb) < _SUPERSEDE_MIN_SIZE_RATIO * len(wa):
                continue
            union = len(wa | wb)
            if union and len(wa & wb) / union >= _SUPERSEDE_JACCARD:
                for k in groups[a]:
                    drop[k] = True
                break

    kept = [b for b, d in zip(blocks, drop) if not d]
    if not kept:
        return text
    return "\n\n".join(kept)


# ---------------------------------------------------------------------------
# The treated view — one reply, one treatment, every consumer (§4FV, 2026-09-10)
# ---------------------------------------------------------------------------
# The non-streaming finalize scrubs and smooths `final_ai_content` IN PLACE,
# so every consumer after it (post-mortem, episode, hydration judge,
# work_log, calibration, the hedge scan, the promise backstop) reads the
# treated text by construction. The streamed drain has no such variable: it
# hands each consumer the raw `full_content`, and §4FS treated only the
# trajectory copy. Seven readers on the common (web-UI) path were therefore
# learning from, and judging, text the user never received — narration the
# smoother removes and tool markup the live scrub had already taken out of
# the stream.
#
# `treat_reply` is that missing variable: the gate at the shared reader, so
# a new consumer in the drain cannot pick the wrong text by accident.
# Idempotent — the scrub leaves no markup behind and `smooth_reply` is
# pinned idempotent — so calling it per consumer is safe.

#: Real (non-synthetic) tool runs required before a reply is smoothed. The
#: 2026-07-17 decision, re-affirmed by §4FT after a one-day trial of 1.
#: `_finalize_and_return` spells the same threshold inline.
SMOOTHING_MIN_TOOLS = 2


# ---------------------------------------------------------------------------
# Narration-only replies (§4GH, 2026-09-13 — request e57ad0cf)
# ---------------------------------------------------------------------------
# The smoother removes beats that a LATER paragraph supersedes, and returns
# the original when removing them would empty the reply. A reply that is
# NOTHING BUT beats therefore ships untouched: request e57ad0cf's forced
# final turn emitted only (dropped) tool calls, and the user received five
# stacked "I have good coverage. Let me now dig into…" paragraphs after 339 s.
# `narration_only` names that shape so the loop can retry the final and the
# verifier can refute it mechanically. Deliberately NARROW — a paragraph
# counts only when it holds a forward-looking agent-voice beat ("Let me
# read…", "I'll fetch…"), every other sentence in it is short assessment glue
# ("I have good coverage."), and nothing in it carries content (a URL, a
# number, a quote, a list, code, emphasis). "Let me know if…" is addressed to
# the user and is not a beat. Measured on the live corpus (2026-09-13, 1,878
# user turns): 0 of 115 human-approved and 0 of 601 verifier-passed replies
# match; the matches are e57ad0cf itself and eight one-line beats.
_NARRATION_BEAT_SENT_RE = re.compile(
    r"^\s*(?:(?:now|next|then|first|ok(?:ay)?|good|great|perfect|alright)[,\s]+)*"
    r"(?:let me(?!\s+know)|let'?s(?!\s+say)|i'?ll|i will|i need to|i'?m going to|"
    r"i am going to|time to|i should|i want to)\b",
    re.IGNORECASE)
#: A beat announces WORK: the opener must be followed, in the same sentence,
#: by a work verb. "Let me be clear: that claim is false.", "I'll be direct:
#: the file does not exist.", "I will not do that.", "Let's go with option B."
#: open like beats and are answers (R3 review of §4GH) — none names work.
_NARRATION_WORK_RE = re.compile(
    r"\b(?:search|dig|read|fetch|re-?fetch|check|double-check|look|look up|take a look|"
    r"run|re-?run|try|start|begin|kick off|proceed|continue|investigate|extract|"
    r"navigate|open|load|gather|collect|pull|retrieve|verify|confirm|examine|explore|"
    r"scan|query|grab|review|analy[sz]e|summari[sz]e|nail down|figure out|work out|"
    r"sort out|go through|go ahead|write|rewrite|draft|compose|build|fix|apply|"
    r"implement|create|generate|render|update|edit|refactor|test|install|set up|"
    r"deploy|restart|launch|close|finish|complete|wrap up|mark|save|store|delete|"
    r"remove|add|move|copy|upload|download|send|post|call|compute|calculate|count|"
    r"list|find|locate|identify|compare|handle|process|parse|inspect|trace|debug|"
    r"resolve|clean|prepare|assemble|compile|make sure|ensure|parallel)\b",
    re.IGNORECASE)
#: A sentence that asks the user something, addresses them, or asks for
#: something ("I'm going to need the password…") is an answer.
_NARRATION_ADDRESSED_RE = re.compile(
    r"\?|\byou\b|\byour\b|\bneed (?:the|a|an|more|some)\b", re.IGNORECASE)


def _is_work_beat(sentence: str) -> bool:
    return (bool(_NARRATION_BEAT_SENT_RE.match(sentence))
            and bool(_NARRATION_WORK_RE.search(sentence))
            and not _NARRATION_ADDRESSED_RE.search(sentence))
_NARRATION_CONTENT_RE = re.compile(
    r"https?://|\d{2,}|`|^\s*[-*•]|^\s*\d+[.)]\s|\*\*|[\"“”]|\||!\[|\]\(",
    re.MULTILINE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: A non-beat sentence in a beat paragraph must be this short to count as
#: assessment glue rather than an answer.
_NARRATION_GLUE_MAX_CHARS = 140


def narration_only(text: str) -> bool:
    """True when EVERY paragraph of ``text`` is a working-narration beat and
    none carries content — the reply announces work and reports nothing."""
    blocks = [b.strip() for b in _split_blocks(text or "") if b.strip()]
    if not blocks:
        return False
    for b in blocks:
        if len(b) > _MAX_NARRATION_CHARS or _NARRATION_CONTENT_RE.search(b):
            return False
        sents = [s for s in _SENT_SPLIT_RE.split(b) if s.strip()]
        if not any(_is_work_beat(s) for s in sents):
            return False
        if any(len(s) > _NARRATION_GLUE_MAX_CHARS or _NARRATION_ADDRESSED_RE.search(s)
               for s in sents if not _is_work_beat(s)):
            return False
    return True


def forced_final_has_no_answer(this_turn_text: str, accumulated: str) -> bool:
    """The forced-final decision: would the reply that ships now — the
    accumulated narration plus this turn's own text, system notes aside —
    contain no answer at all (empty, or narration only)?"""
    parts = [p for p in ((accumulated or "").strip(), (this_turn_text or "").strip()) if p]
    body = strip_system_notes("\n\n".join(parts)).strip()
    return not body or narration_only(body)


def is_narration_only_trim(smoothed: str, original: str) -> bool:
    """True when smoothing reduced a reply to working narration — the
    inverted-trim failure (2026-07-25 live): the smoother kept "Let me
    search more specifically…" and dropped the findings. Lives here (not in
    agent.py, where it was written) so both delivery paths and
    `treat_reply` share one definition.

    ⚠ WIDENED (§4GO). The original test was "short (<90 chars) and opens
    with a connective" — a proxy for "what survived is a beat", from a time
    when only paragraph-INITIAL beats were dropped. Pass 1b trims deeper, so
    a trim can now leave a LONGER beat standing (measured: the e57ad0cf
    reply, every paragraph of it narration, smooths to one 131-char beat).
    `narration_only` measures the thing the length bound was proxying for,
    and it is the same predicate the forced-final guard uses, so the two
    cannot drift apart. The short-and-connective branch is kept as-is: it
    fires on fragments `narration_only` does not classify.
    """
    s = (smoothed or "").strip()
    if s == (original or "").strip():
        return False
    if len(s) < 90 and bool(re.match(
            r"(Let me|Now |Next,? |I'll |I will |First,? |Then )", s)):
        return True
    return narration_only(s)


def treat_reply(text: str, *, n_real_tools: int) -> str:
    """The delivered view of a reply: unparsed tool-call markup removed
    (and said so, once), then working narration trimmed when the turn ran
    at least `SMOOTHING_MIN_TOOLS` real tools and the trim did not leave
    narration only. Non-strings and empty text pass through untouched."""
    if not isinstance(text, str) or not text:
        return text
    out = strip_unparsed_tool_calls(text)
    if n_real_tools >= SMOOTHING_MIN_TOOLS:
        smoothed = smooth_reply(out)
        if not is_narration_only_trim(smoothed, out):
            out = smoothed
    return out


def count_real_tools(tools_run) -> int:
    """Tool records that actually ran — the synthetic ones the loop
    fabricates (plan markers, watchdog replans) do not open the smoothing
    gate. Shared so a caller cannot spell the gate differently."""
    return sum(1 for t in (tools_run or [])
               if t and not (t or {}).get("_synthetic"))
