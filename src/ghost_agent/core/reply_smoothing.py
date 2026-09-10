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


def _is_narration(block: str, later_blocks: List[str] = ()) -> bool:
    """Is this paragraph a working-narration beat?

    `later_blocks` are the paragraphs that follow it; a TEMPORAL lead is a
    beat only when one of them restates it (§4FV). Called with no later
    blocks the temporal half is inert — an ambiguous opener is content
    until the reply proves otherwise.
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
    m = _TEMPORAL_LEAD_RE.match(stripped)
    if not m:
        return False
    # "Now let me …" / "Then I'll …": the lead is decoration on a beat.
    if _BEAT_RE.match(stripped[m.end():].lstrip()):
        return True
    later_prose = [b for b in later_blocks if not _has_fence(b)]
    return bool(later_prose) and _restated_later(stripped, later_prose)


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
    r"\n*---\n(?:\*\*⚠ Unverified:\*\*|\*\*Plan check:\*\*|"
    # "Verifier note:" was MISSING from this list (§4L Lens-A MINOR-1)
    # while being appended BEFORE the hedge scan — a verifier note
    # quoting first-person text survived the strip and fired the hedge
    # regex, putting uncertainty_pressure on a refuted (label-0) turn:
    # the same label-echo channel as the λ leak, one banner over.
    r"\*\*Verifier note:\*\*|"
    r"\*\*Things I'm not certain about:\*\*|\*\*Assumptions I made:\*\*)"
    r"(?:(?!\n\n).)*$",
    re.DOTALL,
)
# The correction banner is PREPENDED (see _consume_pending_corrections) with
# a fixed shape ending in a blank-line-separated rule.
_CORRECTION_BANNER_RE = re.compile(
    r"^⚠️ \*\*Correction to my previous answer:\*\* .*?\n\n---\n\n",
    re.DOTALL,
)


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


def is_narration_only_trim(smoothed: str, original: str) -> bool:
    """True when smoothing reduced a reply to a short working-narration
    line — the inverted-trim failure (2026-07-25 live): the smoother kept
    "Let me search more specifically…" and dropped the findings. Lives here
    (not in agent.py, where it was written) so both delivery paths and
    `treat_reply` share one definition."""
    s = (smoothed or "").strip()
    return (s != (original or "").strip()
            and len(s) < 90
            and bool(re.match(
                r"(Let me|Now |Next,? |I'll |I will |First,? |Then )", s)))


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
