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
* Only four shapes are ever removed, all non-final:
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
       paragraph's content (2026-09-09);
    4. a beat SENTENCE inside a surviving paragraph whose next delivered
       paragraph opens by delivering ("… Let me extract the quote." →
       "The investigation is complete.") — the sentence goes, the
       observation beside it stays (§4HD, 2026-09-15).
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
    # §4KJ added an "I have enough data to <verb>…" alternative; four review
    # rounds found it deleting delivered answers ("…to complete the booking.")
    # — a lexical proxy for "is this a beat?". Removed.
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
    # digits count (review §4IY: "ghost: disk 81% used, 12 GB free" and "eva: disk 43% used,
    # 61 GB free" shared every WORD and differed only in figures — the first was deleted)
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2 or w.isdigit()}


_LEAD_LABEL_RE = re.compile(r"^\s*(?:\*\*)?([^\n:*]{1,40}?)(?:\*\*)?\s*:\s+\S")


def _lead_label(block: str) -> str:
    """The label a per-item paragraph opens with ("ghost:", "**Staging:**",
    "Server A:"), lower-cased, or ""."""
    m = _LEAD_LABEL_RE.match(block or "")
    return m.group(1).strip().lower() if m else ""


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


def _strip_trailing_beat_sentences(block: str) -> str:
    """Remove the agent-voice beat sentence(s) a paragraph ENDS on, keeping
    everything before them and the paragraph's own whitespace (§4HJ).
    Sentences are peeled from the end while the last one is a beat — a
    temporal lead is decoration ("Now let me …") — and never an offer to
    the user. Returns ``block`` unchanged when nothing qualifies or when
    the whole paragraph is beats (that is pass 1's decision, not this one's).
    """
    text = block.strip()
    starts = [0] + [m.end() for m in _SENTENCE_SPLIT_RE.finditer(text)]
    if len(starts) < 2:
        return block
    keep = len(starts)
    while keep > 1:
        a = starts[keep - 1]
        b = starts[keep] if keep < len(starts) else len(text)
        last = text[a:b].strip()
        if _OFFER_RE.match(last):
            break
        m = _TEMPORAL_LEAD_RE.match(last)
        core = last[m.end():].lstrip() if m else last
        if not _BEAT_RE.match(core):
            break
        keep -= 1
    if keep == len(starts):
        return block
    out = text[:starts[keep]].rstrip()
    return out if out else block


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
# Two heads since §4IT: the correction ("⚠️ **Correction to my previous
# answer:**") and the caveat-only note ("ℹ️ **On my previous answer:**"); a
# banner may carry both. Fresh-eye review §4IX: the caveat head was NOT
# stripped, so the verifier read the banner's own years and citations as
# this reply's claims, re-queued the identical caveat every following turn,
# and the narration/no-answer checks saw "content" that was ours.
_CORRECTION_BANNER_RE = re.compile(
    r"^(?:⚠️ \*\*Correction to my previous answer:\*\*|ℹ️ \*\*On my previous answer:\*\*) .*?\n\n---\n\n",
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
    # §4HG (2026-09-16, req 095beab8): the second answer of a steered turn
    # — the STOP declaration after the check ran — echoed the steer's
    # OTHER words: "The distinguishing check (…) has now run twice with no
    # new agency name surfaced … I have enough to deliver." Zero of the
    # markers above matched it and it shipped as the reply's second
    # paragraph. Each pattern below is a phrase of the steer text itself
    # (`risk.STEER_DIRECTIVE_TERMS` pins "distinguish" and "no new
    # information" at the producer). Measured on 8,831 delivered
    # paragraphs: +5 recognised, every one a checkpoint answer.
    re.compile(r"\bdistinguish(?:ing)?\b[^.\n]{0,60}\b(?:check|assumption|alternative)\b",
               re.IGNORECASE),
    re.compile(r"\bno new\b[^.\n]{0,40}\b(?:surfaced|found|information|info|evidence|"
               r"facts?|leads?|results?)\b", re.IGNORECASE),
    re.compile(r"\b(?:run|ran|been run|searched|tried)\s+(?:once|twice|three times|"
               r"\d+ times|\w+ times)\b", re.IGNORECASE),
    re.compile(r"\bi have enough to (?:deliver|finali[sz]e|answer|report|conclude)\b",
               re.IGNORECASE),
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


def drop_checkpoint_segments(text: str, segments, *, keep_if_empty: bool = True) -> str:
    """Remove recorded checkpoint answers from an assembled reply.

    Exact-substring removal of segments the turn loop recorded, so
    nothing is matched by resemblance. Fail-open in both directions: a
    segment that is no longer present (a later stage rewrote it) is
    skipped, and if removal would leave nothing the original is returned
    — unless ``keep_if_empty`` is False (§4HG): on the STREAM path the
    text is the prefix of interim paragraphs and the answer follows it on
    the wire, so a prefix that was nothing but checkpoint answers should
    become empty rather than ship.
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
    if not out and keep_if_empty:
        return text
    return out


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
    # the unparsed-call note is ours too (review §4IY: the judge and the binder read it as the
    # model's words — "artifact: machine noise")
    if UNPARSED_TOOL_CALL_NOTE in out:
        out = out.replace("\n\n" + UNPARSED_TOOL_CALL_NOTE, "").replace(UNPARSED_TOOL_CALL_NOTE, "")
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
# The call dialects only (review §4IY: "Run `ghost <tool name> --help`" and a
# repr "<function tool_execute at 0x…>" were "unparsed calls" and the rest of
# the reply was deleted): `<tool_call>`, `<tool name=…>`, `<function=…>`,
# `<function name=…>`. A block is closed by the SAME tag that opened it — a
# `</function>` inside a `<tool_call>` block does not end the block (the
# §4IY first cut ended at any close tag or blank line and delivered the
# tail of the schema-compare leak: `</parameter></function></tool_call>`
# plus the code after its first blank line — full suite). An unclosed call
# runs to the end of the reply: what follows a truncated opener is the
# call's payload, not prose.
_CALL_OPEN_RE = re.compile(
    r"(?<!`)<(?:(tool_call)\b[^>]*>|(tool)(?:\s*>|\s+name\s*=[^>]*>)|(function)(?:(?:\s*=|\s+name\s*=)[^>]*>|\b[^>]*>(?=.*?</function\b)))",   # a bare `<tool>` is the old dialect; any `<function …>` that a `</function>` closes
    re.IGNORECASE | re.DOTALL)
_CALL_CLOSE_RES = {"tool_call": re.compile(r"</tool_call\b[^>]*>", re.IGNORECASE),
                   "tool": re.compile(r"</tool\b[^>]*>", re.IGNORECASE),
                   "function": re.compile(r"</function\b[^>]*>", re.IGNORECASE)}


class _CallSpan:
    """The `re.Match` surface `strip_unparsed_tool_calls` reads."""
    __slots__ = ("_s", "_e", "_t")

    def __init__(self, text: str, start: int, end: int):
        self._s, self._e, self._t = start, end, text[start:end]

    def start(self) -> int:
        return self._s

    def end(self) -> int:
        return self._e

    def group(self, _i: int = 0) -> str:
        return self._t


def _call_markup_spans(text: str) -> List[_CallSpan]:
    out: List[_CallSpan] = []
    pos = 0
    while True:
        m = _CALL_OPEN_RE.search(text, pos)
        if m is None:
            return out
        tag = "tool_call" if m.group(1) else ("tool" if m.group(2) else "function")
        c = _CALL_CLOSE_RES[tag].search(text, m.end())
        end = c.end() if c else len(text)
        out.append(_CallSpan(text, m.start(), end))
        pos = end
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
    for m in _call_markup_spans(text):
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


# §4HD (2026-09-15) — a beat INSIDE a paragraph that precedes the delivery.
# Live (req 9b6b8757) the delivered reply opened "The Telegram post 363
# snapshot returned the channel feed. Let me extract the notification
# quote…" and the next paragraph began "The investigation is complete."
# Three rules looked and stood down: the beat was not at paragraph start
# (`_BEAT_RE`); it was a LONE beat (§4GO's run rule); nothing later
# restated it (the temporal/trailing-beat rule). Nothing asked the one
# question that settles it — does the NEXT paragraph deliver? A beat that
# announces work immediately before the reply declares the work done is
# stale by construction; no restatement evidence is needed.
#
# Sentence-level, never the paragraph: §4GO's concern — the observation
# beside a lone beat may be the only place a finding appears — stays
# intact ("The extract_text on single=1 gave the same capped preview." is
# kept; "Let me take a full-page screenshot…" goes). Whitespace is
# preserved, so a line-broken paragraph is not re-flowed.
#
# Measured on 999 delivered replies (≥400 chars, multi-paragraph): 31
# paragraphs, every one audited. Two were NOT beats, and the exclusions
# below are theirs: a quoted passage the model was CITING ("*"I've been
# thinking about recursion… I'll create it, test it…"*") and a labelled
# section ("**Next session:** When you wake me up next… I'll evaluate…").
# Three more the first draft cut were addressed to the user ("Let me grab
# the latest headlines for you.", "I'll coach you in real-time — every
# move…") and are answers; the §4GH vocabulary (`_NARRATION_ADDRESSED_RE`,
# `_OFFER_RE`) already names that class. A colon-terminated beat is a
# lead-in ("Let me be clear: …") and is never cut, which is why this
# sentence splitter — unlike `_SENTENCE_SPLIT_RE` above — breaks at ':'.
# "Let's …" and "I need to …" are NOT openers here: on the corpus they
# were the model reasoning ("I need to recapture to maintain material
# equality"), not announcing tool work.
_MID_BEAT_RE = re.compile(
    r"^(?:now\s+|next,?\s+)?(?:let me|i'll|i will)\b", re.IGNORECASE)
_DELIVERY_OPENER_RE = re.compile(
    r"^\s*(?:#{1,3}\s|\*\*|"
    r"the (?:investigation|task|analysis|report|work|research)\b[^.\n]{0,40}"
    r"\b(?:is|are)\b[^.\n]{0,20}\b(?:complete|done|finished|ready)|"
    r"here(?:'s| is| are)\b|done\b|summary\b|results?\b|findings\b|bottom line\b)",
    re.IGNORECASE)
_BEAT_SENT_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+")
# §4HX (2026-09-16, req abb8fdb7): a delivery may open with ONE short status
# sentence before the opener — "Fixed. Here's what was wrong…", "Both files
# are read. Here's the comparison.", "The page loaded and read fine. Here's
# what's shown…". Corpus (Aug–Sep): 148 such paragraphs, 10 of them after a
# trailing-beat paragraph the passes then left standing.
_STATUS_SENTENCE_MAX = 80


def _opens_by_delivering(block: str) -> bool:
    """Does ``block`` open by delivering — its first sentence is an opener,
    or a short status sentence is followed by one? The single predicate
    every "next block delivers" test uses."""
    text = (block or "").strip()
    if not text:
        return False
    if _DELIVERY_OPENER_RE.match(text):
        return True
    sents = [x for x in _BEAT_SENT_SPLIT_RE.split(text) if x.strip()]
    return (len(sents) >= 2 and len(sents[0]) <= _STATUS_SENTENCE_MAX
            and bool(_DELIVERY_OPENER_RE.match(sents[1])))
_QUOTED_RE = re.compile(r'["“”]')
_MARKUP_START_RE = re.compile(r"^\s*[*_>#]")


_REASON_RE = re.compile(r"\b(?:because|since|so that|not|never|cannot|can't|won't)\b|\b(?:γιατί|επειδή|δεν|όχι)\b", re.IGNORECASE)


def _is_mid_beat(sentence: str) -> bool:
    s = sentence.strip()
    return (bool(_MID_BEAT_RE.match(s))
            and not s.endswith(":")
            and not _OFFER_RE.match(s)
            and not _NARRATION_ADDRESSED_RE.search(s)
            # §4IY: a recommendation ("I'll recommend Postgres because…") or a warning
            # ("I will not be able to recover rows…") opens like a beat and is the answer
            and bool(_NARRATION_WORK_RE.search(s))
            and not _REASON_RE.search(s)
            and not _LEAD_IN_RE.search(s)
            and not _NEGATED_OPENER_RE.match(s))


# §4HE (2026-09-15, req 5fa6aa97) — what pass 3 leaves behind. The delivered
# reply opened with three stacked one-liners: "I have enough to finalize." /
# "The investigation is complete." / "The investigation is complete. Here's
# the forensic synthesis." — two iterations' worth of hand-off, each a
# readiness declaration plus a beat, then the real opener. Pass 3 cut the
# beats (as designed) and left the declarations, which are not findings:
# one is a verbatim sentence of the very next paragraph, the other says
# only that the model is ready to deliver. A surviving paragraph before a
# delivery is dropped WHOLE when it carries no content (no URL, number,
# code, emphasis, quote, list) and is either (a) contained verbatim in the
# next delivered paragraph, or (b) a single readiness declaration. Measured
# on 1,000 delivered replies: 356 short paragraphs precede a delivery and
# nearly all are lead-ins or headings ("Here's what I changed:", "## ⚡
# Performance") — the rule must not touch those, so containment and the
# readiness vocabulary are the ONLY two triggers; the corpus hits are the
# live pair and one "I now have all three sources." A bare horizontal rule
# between a paragraph and the delivery is looked through.
_READINESS_RE = re.compile(
    r"^(?:i(?:'ve| have)(?: now)?(?: gathered| got| collected)? "
    r"(?:enough|sufficient|everything|all (?:the|of|three|four|five)|what i need)\b|"
    r"i now have\b|that(?:'s| is) enough\b|"
    # §4HJ: the residue pass 1c leaves — "I have strong consolidated
    # evidence." / "I have good coverage." — a readiness declaration in
    # the vocabulary of the research turns.
    r"i(?:'ve| have)(?: now)? (?:\w+ ){0,2}?(?:strong|solid|good|consolidated|confirmed|"
    r"enough|sufficient) (?:\w+ ){0,2}?(?:evidence|coverage|data|material|sources?)\b|"
    # §4HQ (2026-09-16, req 3d3e0681): the same declaration in the third
    # person — "The report is complete and verified against all
    # constraints." — a one-sentence, content-free hand-off that also
    # matches the delivery-opener shape, so pass 3 took it for the
    # delivery and left it standing. Two matches in 5,358 corpus
    # paragraphs (Aug–Sep), both hand-offs before a delivery.
    r"the (?:investigation|task|analysis|report|work|research)\b[^.\n]{0,40}"
    # §4IY: "…is ready in the Downloads folder" says WHERE — content; the
    # §4HQ "…is complete and verified against all constraints" conjoins a
    # second status and stays a hand-off (a conjoined tail without a
    # location/destination preposition)
    r"\b(?:is|are)\b[^.\n]{0,20}\b(?:complete|done|finished|ready)\b"
    r"(?:\s+and\b(?![^.\n]*\b(?:in|at|on|to|under|into|inside|from)\b)[^.\n]{0,60})?\s*[.!]?\s*$)",
    re.IGNORECASE)
_HRULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _is_empty_handoff(block: str, next_block: str) -> bool:
    """Is ``block`` a content-free hand-off that the next delivered
    paragraph makes redundant? See the §4HE note above."""
    if not _opens_by_delivering(next_block or ""):  # §4HX
        return False
    text = block.strip()
    if (not text or len(text) > 160 or _has_fence(text)
            or _MARKUP_START_RE.match(text) or _LIST_START_RE.match(text)
            or _NARRATION_CONTENT_RE.search(text)):
        return False
    if text in next_block:
        return True
    sents = [x for x in _BEAT_SENT_SPLIT_RE.split(text) if x.strip()]
    return (len(sents) == 1 and bool(_READINESS_RE.match(text))
            and not _NARRATION_ADDRESSED_RE.search(text))


def _strip_beats_before_delivery(block: str, next_block: str) -> str:
    """Drop the non-first agent-voice beat sentences of ``block`` when
    ``next_block`` opens by delivering. Returns ``block`` unchanged unless
    every condition holds. Removes sentence SPANS from the original text,
    so the paragraph's own line breaks survive."""
    if not _opens_by_delivering(next_block or ""):  # §4HX
        return block
    if (_has_fence(block) or _QUOTED_RE.search(block)
            or _MARKUP_START_RE.match(block) or _LIST_START_RE.match(block)):
        return block
    text = block.strip()
    starts = [0] + [m.end() for m in _BEAT_SENT_SPLIT_RE.finditer(text)]
    if len(starts) < 2:
        return block
    pieces = []
    for n, a in enumerate(starts):
        b = starts[n + 1] if n + 1 < len(starts) else len(text)
        if n and _is_mid_beat(text[a:b]):
            continue
        pieces.append(text[a:b])
    out = "".join(pieces).rstrip()
    return out if out and out != text else block


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
            continue
        # Pass 1c (§4HJ, 2026-09-16, req 6afaf940) — the working log over
        # the size bound. A trailing-beat paragraph the reply RESTATES is
        # kept whole above `_MAX_NARRATION_CHARS` (the bound protects long
        # content paragraphs), so "I have good coverage. Key candidate
        # emerging: Italy (…). Let me run targeted searches to confirm…"
        # (314 chars) shipped above "The investigation is complete." with
        # its restatement test already True. Corpus: 11 such paragraphs,
        # every trailing beat in them stale. Same evidence pass 1b needs
        # (restated by the rest of the reply), same cut pass 3 makes
        # (the beat sentence goes, the observation stays). No size test
        # here: a paragraph under the bound with this evidence was
        # already dropped whole by `_is_narration` above, so only the
        # over-bound ones reach this line (the battery found the explicit
        # bound to be a dead guard).
        stripped = blocks[i].strip()
        if (not _has_fence(stripped)
                and not _LIST_START_RE.match(stripped) and _trailing_beat(stripped)
                and _restated_anywhere_later(
                    stripped, [b for b in blocks[i + 1:] if not _has_fence(b)])):
            blocks[i] = _strip_trailing_beat_sentences(blocks[i])

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
            # two PARALLEL per-item paragraphs ("ghost: …" / "eva: …", "Staging:" /
            # "Production:") are two facts, not a restatement (review §4IY)
            la, lb = _lead_label(blocks[groups[a][0]]), _lead_label(blocks[groups[b][0]])
            if la and lb and la != lb:
                continue
            union = len(wa | wb)
            if union and len(wa & wb) / union >= _SUPERSEDE_JACCARD:
                for k in groups[a]:
                    drop[k] = True
                break

    kept = [b for b, d in zip(blocks, drop) if not d]
    if not kept:
        return text

    # Pass 3 — a beat inside a SURVIVING paragraph that precedes the
    # delivery (§4HD). Deliberately last, on the kept sequence: passes 1
    # and 1b judge the paragraph as written ("obs. Let me X." restated
    # later, or sitting in a run of beats, goes WHOLE — the observation is
    # part of the working log), and only what they keep is trimmed. Run
    # first, this pass would turn the last member of a beat run into a
    # one-sentence observation that 1b no longer recognises, and the
    # 7b2da5be fragment ("The extract_text on single=1 gave the same
    # capped preview.") would ship as the reply's opening line. "Next"
    # means the next DELIVERED paragraph — a dropped beat between the
    # observation and the delivery does not shield it, and neither does a
    # bare horizontal rule (§4HE). What the trim leaves is then judged
    # once more: a content-free hand-off the delivery makes redundant
    # goes whole (§4HE, `_is_empty_handoff`).
    out: List[str] = []
    for i, block in enumerate(kept[:-1]):
        j = i + 1
        while j < len(kept) - 1 and _HRULE_RE.match(kept[j]):
            j += 1
        trimmed = _strip_beats_before_delivery(block, kept[j])
        if _is_empty_handoff(trimmed, kept[j]):
            continue
        out.append(trimmed)
    out.append(kept[-1])          # the final block is never a candidate
    return "\n\n".join(out)


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
# §4IW (req a3ec5024): the same beat in Greek — "Ας κάνω έρευνα…", "Θα
# ψάξω…", "Πάμε να δούμε…" — shipped as the whole reply of a zero-tool turn
# and nothing here saw it; the detector was English-only.
_NARRATION_BEAT_SENT_RE = re.compile(
    r"^\s*(?:(?:now|next|then|first|ok(?:ay)?|good|great|perfect|alright|τώρα|πρώτα|λοιπόν|εντάξει|ωραία)[,\s]+)*"
    r"(?:let me(?!\s+know)|let'?s(?!\s+say)|i'?ll|i will|i need to|i'?m going to|"
    r"i am going to|time to|i should|i want to"
    # Greek openers. "θα" / "πρέπει να" / "χρειάζεται να" are person-agnostic
    # ("Θα ανοίξει το κατάστημα στις 9" states a fact) — they open a beat only
    # when the work verb below is FIRST person (fresh-eye review §4IX)
    r"|ας|θα|πάμε να|επιτρέψτε μου να|επίτρεψέ μου να|πρέπει να|χρειάζεται να)\b",
    re.IGNORECASE)
#: A beat announces WORK: the opener must be followed, in the same sentence,
#: by a work verb. "Let me be clear: that claim is false.", "I'll be direct:
#: the file does not exist.", "I will not do that.", "Let's go with option B."
#: open like beats and are answers (R3 review of §4GH) — none names work.
#: Whole words (review §4IX: "ready", "market", "address", "OpenAI" and
#: "Downloads" were work verbs by prefix); the Greek verbs carry their
#: first-person endings (ψάξω / ψάξουμε), never a bare stem.
_GREEK_1P = r"(?:ω|ουμε)"
# §4JI (req e69cab30): "searches" / "κάνω πιο στοχευμένες αναζητήσεις" — the
# work NOUN in the plural, or with modifiers between "κάνω" and the noun,
# named no work here; the announcement shipped after 36 searches.
_NARRATION_WORK_RE = re.compile(
    r"\b(?:search(?:es)?|dig|read|fetch|re-?fetch|check|double-check|look|look up|take a look|"
    r"run|re-?run|try|start|begin|kick off|proceed|continue|investigate|extract|"
    r"navigate|open|load|gather|collect|pull|retrieve|verify|confirm|examine|explore|"
    r"scan|query|grab|review|analy[sz]e|summari[sz]e|nail down|firm up|pin down|figure out|work out|"
    r"sort out|go through|go ahead|write|rewrite|draft|compose|build|fix|apply|"
    r"implement|create|generate|render|update|edit|refactor|test|install|set up|"
    r"deploy|restart|launch|close|finish|complete|wrap up|mark|save|store|delete|"
    r"remove|add|move|copy|upload|download|send|post|call|compute|calculate|count|"
    r"list|find|locate|identify|compare|handle|process|parse|inspect|trace|debug|"
    r"resolve|clean|prepare|assemble|compile|make sure|ensure|parallel|"
    # the synthesis verbs of a research turn's hand-off ("Let me synthesize the answer.", corpus 45675adf)
    r"synthesi[sz]e|consolidate|put together|write up|finali[sz]e|present|deliver|produce|report back"
    rf"|ψάξ{_GREEK_1P}|αναζητήσ{_GREEK_1P}|ερευνήσ{_GREEK_1P}|ελέγξ{_GREEK_1P}|διαβάσ{_GREEK_1P}|ανοίξ{_GREEK_1P}|"
    rf"τρέξ{_GREEK_1P}|δοκιμάσ{_GREEK_1P}|εξετάσ{_GREEK_1P}|βρω|βρούμε|δω|δούμε|κοιτάξ{_GREEK_1P}|ρίξ{_GREEK_1P} μια ματιά|"
    rf"συλλέξ{_GREEK_1P}|επαληθεύσ{_GREEK_1P}|εντοπίσ{_GREEK_1P}|αναλύσ{_GREEK_1P}|συγκρίν{_GREEK_1P}|φέρω|φέρουμε|"
    rf"κατεβάσ{_GREEK_1P}|γράψ{_GREEK_1P}|φτιάξ{_GREEK_1P}|ξεκινήσ{_GREEK_1P}|συνεχίσ{_GREEK_1P}|προχωρήσ{_GREEK_1P}|"
    # the work NOUNS in singular and plural — the accent moves (αναζήτηση /
    # αναζητήσεις, έλεγχο / ελέγχους), so both vowels are admitted
    rf"ψάχν{_GREEK_1P}|κάν{_GREEK_1P}(?:\s+\S+){{0,3}}?\s+(?:[εέ]ρευν(?:α|ες|ών)|[εέ]λ[εέ]γχ(?:ο|ος|ους|οι|ων)"
    r"|αναζ[ηή]τ[ηή]σ(?:η|ης|εις|εων)|επαλ[ηή]θε[υύ]σ(?:η|ης|εις|εων)))\b",
    re.IGNORECASE)
#: A sentence that asks the user something, addresses them, or asks for
#: something ("I'm going to need the password…") is an answer. The Greek
#: question mark is ";" — read as one only in a sentence written in Greek.
# §4JI: "the three items you mention" / "the file you sent" points BACK at
# the request — an echo, not an address; "you" followed by a reporting
# verb is exempt (the English twin of req e69cab30's reply read as addressed).
_NARRATION_ADDRESSED_RE = re.compile(
    r"\?|\byou\b(?!\s+(?:mention(?:ed)?|asked|said|described|noted|gave|provided|requested|wrote|"
    r"listed|cited|named|specified|quoted|pointed|shared|sent|uploaded|pasted|linked|attached)\b)"
    r"|\byour\b|\bneed (?:the|a|an|more|some)\b"
    r"|\b(?:σου|σας|σε|εσύ|εσείς|θέλεις|θες|θέλετε|θέτε|μπορείς|μπορείτε)\b", re.IGNORECASE)
_GREEK_QUESTION_RE = re.compile(r"[α-ωά-ώ][^;\n]*;(?:\s|$)")
#: A beat that introduces content with a colon ("Let me summarise: the agency
#: was never named…", "I'll be direct: the tests are failing") is a lead-in
#: to an answer, not an announcement (review §4IX).
_LEAD_IN_RE = re.compile(r"(?<!\d):\s*\S.{3,}")     # "I'll list them: a, b, c." is a lead-in too; a clock's "18:00" is not (review §4IY)
#: What may precede the first beat and still be glue: an assessment or a
#: readiness remark in the agent's voice ("I have good coverage.", "Good.",
#: "Αυτό είναι ενδιαφέρον ερώτημα — δεν το έχω συναντήσει."). A yes/no, a
#: verdict or any other statement before the beat is the answer (review §4IY:
#: "Yes. I'll check it tomorrow." / "Ναι. Θα το ελέγξω αύριο." were narration).
_ASSESSMENT_GLUE_RE = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|good|great|perfect|alright|right|interesting|noted|understood|fair|hmm|well)[,.!:]?\s*$"
    r"|(?:ok(?:ay)?|good|great|perfect|alright|now|so)[,\s]+"
    r"|i(?:'ve| have| now have| still have| don't have| do not have)\b|i(?:'m| am) (?:still |now )?(?:mid|in the middle|not (?:sure|certain)|missing)\b"
    r"|(?:this|that|it)(?:'s| is) (?:an? )?(?:interesting|good|tricky|hard|fair|useful|helpful|odd|strange|unusual|new)\b"
    r"|(?:αυτό|αυτή) είναι (?:ένα |μια )?(?:ενδιαφέρον|ενδιαφέρουσα|καλή|δύσκολ\w*|περίεργ\w*)\b|ενδιαφέρον\b|ωραία|καλά|εντάξει|λοιπόν|έχω\b|τώρα έχω\b|δεν (?:το )?έχω\b"
    # a remark about what a TOOL just returned ("The dark-web search returned mostly generic results.",
    # "The extract_text on single=1 gave the same capped preview.") is the working log, not a finding
    r"|(?:the|that|this|my|our|η|το|οι)\b[^.!?\n]{0,60}\b(?:search|query|fetch|scan|extract\w*|screenshot|lookup|results?|page|call|command|tool|attempt|αναζήτηση|σελίδα|εντολή)\b"
    r"[^.!?\n]{0,40}\b(?:returned|gave|came back|yielded|showed|shows|found|failed|timed out|capped|generic|empty|nothing|επέστρεψε|έδωσε|απέτυχε)\b)",
    re.IGNORECASE)
#: "I will not delete the production database." is a refusal — an answer.
_NEGATED_OPENER_RE = re.compile(
    r"^\s*(?:\w+[,\s]+)*(?:i will|i'?ll|i am going to|i'?m going to|i should|let me|let'?s)\s+(?:not|never)\b", re.IGNORECASE)


def _is_work_beat(sentence: str) -> bool:
    return (bool(_NARRATION_BEAT_SENT_RE.match(sentence))
            and bool(_NARRATION_WORK_RE.search(sentence))
            and not _NARRATION_ADDRESSED_RE.search(sentence)
            and not _GREEK_QUESTION_RE.search(sentence)
            and not _LEAD_IN_RE.search(sentence)
            and not _NEGATED_OPENER_RE.match(sentence))
# A quotation is content when it is long enough to be a finding; a short
# quoted term is the ask's own phrase echoed back ("… τη σχέση του με τον
# "συλλέκτη Φριζήρα"" — req a3ec5024 announced work around it and stopped).
_NARRATION_CONTENT_RE = re.compile(
    r"https?://|\d{2,}|`|^\s*[-*•]|^\s*\d+[.)]\s|\*\*|[\"“”«‘][^\"“”»’\n]{40,}[\"“”»’]|\||!\[|\]\(",   # « » and ‘ ’ are quotes too (review §4IY)
    re.MULTILINE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: A non-beat sentence in a beat paragraph must be this short to count as
#: assessment glue rather than an answer.
_NARRATION_GLUE_MAX_CHARS = 140


_ECHO_FIGURE_RE = re.compile(r"\d[\d.,:/-]*\d|\d{2,}")


def _mask_request_echoes(block: str, request: str) -> str:
    """§4JI: a figure the REQUEST already contains is an echo, not content.
    `_NARRATION_CONTENT_RE` counts any 2+-digit figure as content; req
    e69cab30 (2026-09-21) shipped "Ας κάνω πιο στοχευμένες αναζητήσεις για
    … 23.125.000 δρχ … 23.100.000 δρχ." after 36 searches — an announcement
    that restated the user's own three figures, which read as an answer to
    every guard. The English twin ("Let me do more targeted searches for …
    23,125,000 drachmas …") failed the same way: not a language gap, an echo
    gap. Figures present verbatim in the request are blanked before the
    content check; a figure the reply CONTRIBUTES still counts."""
    if not request:
        return block
    req_figs = {m.group(0) for m in _ECHO_FIGURE_RE.finditer(request)}
    if not req_figs:
        return block
    return _ECHO_FIGURE_RE.sub(lambda m: "" if m.group(0) in req_figs else m.group(0), block)


_YEAR_TOKEN_RE = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")


def _mask_beat_years(block: str) -> str:
    """§4JJ: a year inside a WORK BEAT names a target ("Let me do one final
    batch to firm up the 1995 video" — req 882f477c, 46 searches, 268 chars
    of narration shipped), not a finding. Years are blanked only in
    sentences that are beats once the year is gone; a year in any other
    sentence ("It was 2023.") stays content. Corpus 2,933: +1 = 882f477c."""
    sents = _SENT_SPLIT_RE.split(block)
    out = []
    for sent in sents:
        bare = _YEAR_TOKEN_RE.sub("", sent)
        out.append(bare if bare != sent and _is_work_beat(bare) else sent)
    return " ".join(out)


def narration_only(text: str, *, request: str = "") -> bool:
    """True when EVERY paragraph of ``text`` is a working-narration beat and
    none carries content — the reply announces work and reports nothing.
    ``request`` (the user's message) lets figures it already contains be
    read as echoes rather than content (§4JI)."""
    blocks = [b.strip() for b in _split_blocks(text or "") if b.strip()]
    if not blocks:
        return False
    for b in blocks:
        if len(b) > _MAX_NARRATION_CHARS or _NARRATION_CONTENT_RE.search(
                _mask_beat_years(_mask_request_echoes(b, request))):
            return False
        sents = [s for s in _SENT_SPLIT_RE.split(b) if s.strip()]
        beats = [_is_work_beat(s) for s in sents]
        if not any(beats):
            return False
        # a sentence BEFORE the first beat is an answer ("Yes. I'll check it tomorrow.",
        # "Ναι. Θα το ελέγξω αύριο." — review §4IY); glue after a beat must be short, not
        # addressed to the user (a Greek question included) and carry no quotation of its
        # own however short ('It says "closed until March".' is a finding — review §4IX)
        first_beat = beats.index(True)
        if any(not beats[i] and not _ASSESSMENT_GLUE_RE.match(sents[i]) for i in range(first_beat)):
            return False
        if any(len(s) > _NARRATION_GLUE_MAX_CHARS or _NARRATION_ADDRESSED_RE.search(s) or _GREEK_QUESTION_RE.search(s)
               or re.search(r"[\"“”«‘'][^\"“”»’'\n]{1,}[\"“”»’']", s)
               for i, s in enumerate(sents) if not beats[i]):
            return False
    return True


def forced_final_has_no_answer(this_turn_text: str, accumulated: str,
                               request: str = "") -> bool:
    """The forced-final decision: would the reply that ships now — the
    accumulated narration plus this turn's own text, system notes aside —
    contain no answer at all (empty, or narration only)? ``request`` lets
    echoed figures be read as echoes (§4JI)."""
    parts = [p for p in ((accumulated or "").strip(), (this_turn_text or "").strip()) if p]
    body = strip_system_notes("\n\n".join(parts)).strip()
    return not body or narration_only(body, request=request)


def is_narration_only_trim(smoothed: str, original: str, request: str = "") -> bool:
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
    return narration_only(s, request=request)


def treat_reply(text: str, *, n_real_tools: int) -> str:
    """The delivered view of a reply: unparsed tool-call markup removed
    (and said so, once), then working narration trimmed when the turn ran
    at least `SMOOTHING_MIN_TOOLS` real tools and the trim did not leave
    narration only. Non-strings and empty text pass through untouched."""
    if not isinstance(text, str) or not text:
        return text
    return smooth_gated(strip_unparsed_tool_calls(text), n_real_tools)


def smooth_gated(text: str, n_real_tools: int) -> str:
    """`smooth_reply` behind the two guards every delivered view applies:
    the ≥ `SMOOTHING_MIN_TOOLS` real-tool gate (2026-07-17: a single-tool
    turn's "First… Then… Finally…" are instructions, not beats) and the
    narration-only revert (2026-07-25: never reduce a reply to its one
    "Let me search…" line). The ONE implementation behind `treat_reply`
    (the streamed view), `delivery_view` (finalise and the in-loop
    verifier gate)."""
    if not isinstance(text, str) or not text:
        return text
    if n_real_tools >= SMOOTHING_MIN_TOOLS:
        smoothed = smooth_reply(text)
        if not is_narration_only_trim(smoothed, text):
            return smoothed
    return text


def delivery_view(text: str, tools_run) -> str:
    """The reply as it will be DELIVERED: `smooth_reply` behind the two
    guards finalisation applies — the ≥2-real-tool gate (2026-07-17: a
    single-tool turn's "First… Then… Finally…" are instructions, not beats)
    and the narration-only revert (2026-07-25: never reduce a reply to its
    one "Let me search…" line). ONE implementation, two readers: finalise
    (what ships) and the in-loop verifier gate (what is judged), so the
    judged text and the delivered text agree and a verdict is not thrown
    away for a fingerprint mismatch (live 2026-09-24: 26.7 s of in-loop
    verification recomputed after a 502→219-char trim). Never raises."""
    try:
        return smooth_gated(text, count_real_tools(tools_run or []))
    except Exception:  # noqa: BLE001
        return text


def count_real_tools(tools_run) -> int:
    """Tool records that actually ran — the synthetic ones the loop
    fabricates (plan markers, watchdog replans) do not open the smoothing
    gate. Shared so a caller cannot spell the gate differently."""
    return sum(1 for t in (tools_run or [])
               if t and not (t or {}).get("_synthetic"))
