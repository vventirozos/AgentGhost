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
* Only two shapes are ever removed, both non-final:
    1. connective working narration — a short paragraph opening with a
       progress connective ("Let me…", "Now…", "Good, …", "I'll…");
    2. a superseded summary group — a lead-in-plus-list block whose
       content is restated by a later near-duplicate group (the double
       summary), judged by content-word overlap.
* Fail-open: anything unmatched stays; if smoothing would empty the
  reply it returns the original.
"""

from __future__ import annotations

import re
from typing import List

# Progress connectives that open working narration. Anchored at the start
# of a paragraph; matched case-insensitively. Deliberately verb-shaped —
# nouns/answers ("Now playing: …" is unlikely from this agent mid-fix)
# are accepted collateral, bounded by the length cap below.
_CONNECTIVE_RE = re.compile(
    r"^(?:let me\b|let's\b|now[, ]|good[,.! ]|okay\b|ok[,.! ]|alright\b|"
    r"great[,.! ]|perfect[,.! ]|next[, ]|first[, ]|then[, ]|time to\b|"
    r"i'll\b|i will\b|i need to\b)",
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


def _is_narration(block: str) -> bool:
    stripped = block.strip()
    if not stripped or _has_fence(stripped):
        return False
    if len(stripped) > _MAX_NARRATION_CHARS:
        return False
    if _LIST_START_RE.match(stripped):
        return False
    return bool(_CONNECTIVE_RE.match(stripped))


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


def smooth_reply(text: str) -> str:
    """Remove working narration and superseded summary groups from an
    accumulated multi-turn reply. See module docstring for the rules."""
    if not text or "\n\n" not in text:
        return text
    blocks = _split_blocks(text)
    if len(blocks) < 2:
        return text

    drop = [False] * len(blocks)

    # Pass 1 — connective working narration (never the final block).
    for i in range(len(blocks) - 1):
        if _is_narration(blocks[i]):
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
