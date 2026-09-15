"""Risk-governor checkpoint answers must not ship to the user (2026-09-15).

When the governor fires it injects a numbered checkpoint — state what is
CONFIRMED vs ASSUMED, name the SINGLE smallest check, or STOP and report.
The model answers it in prose on an iteration that then calls more tools,
so the answer lands in the accumulated reply. Live (req 4b518a82) five
such answers stacked up and shipped as the opening third of a forensic
report, which then restated all of it in its own sections; the narration
smoother trimmed 209 of 11,498 characters.

The removal is structural, not lexical: only segments the turn loop
RECORDED — emitted while a steer was live, on an iteration that went on
to call tools — are eligible. The prose match is the third condition.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.risk import (
    STEER_DIRECTIVE_TERMS,
    RiskReading,
    risk_steer_message,
)
from ghost_agent.core.reply_smoothing import (
    drop_checkpoint_segments,
    is_governor_checkpoint_answer,
)

# Reproduced from the delivered reply of req 4b518a82.
_LIVE_CHECKPOINT = (
    "**CONFIRMED (observed across sources):** - Reuters (12 Sep 2026) + IT Pro: "
    "Revolut disclosed customer data via fraudulent requests.\n"
    "**ASSUMED / NOT CONFIRMED:** - Specific country (US/France/Romania) — all "
    "speculation, none substantiated.\n"
    "Single most valuable remaining check: read the actual notification body "
    "from the t.me share page."
)

_LIVE_STOP_DECLARATION = (
    "I've done enough rounds (Reuters, TechCrunch, IT Pro, Register). The "
    "distinguishing check — the exact sender domain — remains unpublished. "
    "Reporting the honest partial answer."
)

# The deliverable's own summary. Uses the SAME vocabulary and must survive.
_REPORT_BOTTOM_LINE = (
    "## Bottom line\n**Confirmed:** mechanism (real government domain + valid "
    "SPF/DKIM/DMARC), data categories, timeline, ~680 'limited' scope.\n"
    "**Not confirmed:** the exact government country, agency, email domain, "
    "or sender address."
)


def test_the_live_checkpoint_answer_is_recognised():
    """FAILS IF: the matcher is absent or needs more than the governor asks."""
    assert is_governor_checkpoint_answer(_LIVE_CHECKPOINT) is True
    assert is_governor_checkpoint_answer(_LIVE_STOP_DECLARATION) is True


def test_a_deliverables_confirmed_section_is_not_a_checkpoint():
    """FAILS IF: one marker is enough.

    The report's own bottom line says "Confirmed:" and "Not confirmed:" —
    the SAME word the governor asks for. A one-marker rule deletes the
    conclusion of the very report the user asked for.
    """
    assert is_governor_checkpoint_answer(_REPORT_BOTTOM_LINE) is False


def test_ordinary_prose_using_one_marker_is_safe():
    """FAILS IF: 'confirmed' alone trips the matcher."""
    assert is_governor_checkpoint_answer(
        "I confirmed the file exists and wrote the parser.") is False
    assert is_governor_checkpoint_answer("") is False


def test_checkpoint_segments_are_removed_and_the_report_survives():
    """FAILS IF: removal is by resemblance rather than exact segment.

    The assembled shape from the live turn: checkpoint answers first,
    deliverable last.
    """
    assembled = f"{_LIVE_CHECKPOINT}\n\n{_LIVE_STOP_DECLARATION}\n\n{_REPORT_BOTTOM_LINE}"
    out = drop_checkpoint_segments(
        assembled, [_LIVE_CHECKPOINT, _LIVE_STOP_DECLARATION])
    assert "Single most valuable remaining check" not in out
    assert "I've done enough rounds" not in out
    assert "**Confirmed:** mechanism" in out
    assert "Not confirmed:" in out


def test_a_recorded_segment_that_is_not_a_checkpoint_is_kept():
    """FAILS IF: the shape test is skipped at removal time.

    Belt and braces: the loop records under structural conditions, but a
    recorded segment still has to LOOK like a checkpoint answer before it
    is deleted — the recorder must never be the only judge.
    """
    seg = "Here is the parser you asked for, with the fix applied."
    assembled = f"{seg}\n\n{_REPORT_BOTTOM_LINE}"
    assert drop_checkpoint_segments(assembled, [seg]) == assembled.strip()


def test_removal_that_would_empty_the_reply_is_refused():
    """FAILS IF: a turn that ONLY produced a checkpoint delivers nothing.

    The governor's third directive is "STOP and report" — a turn that
    complies has a checkpoint answer as its entire reply, and the user
    must still receive it.
    """
    out = drop_checkpoint_segments(_LIVE_CHECKPOINT, [_LIVE_CHECKPOINT])
    assert out == _LIVE_CHECKPOINT


def test_a_segment_no_longer_present_is_skipped():
    """FAILS IF: a stale recording corrupts the reply.

    Earlier finalize stages rewrite text (prompt-bleed truncation, status
    scrubs); a segment recorded before them may no longer appear.
    """
    assembled = _REPORT_BOTTOM_LINE
    assert drop_checkpoint_segments(assembled, [_LIVE_CHECKPOINT]) == assembled


def test_no_segments_is_a_no_op():
    """FAILS IF: the scrub rewrites replies on turns the governor never touched."""
    assert drop_checkpoint_segments(_REPORT_BOTTOM_LINE, []) == _REPORT_BOTTOM_LINE
    assert drop_checkpoint_segments("", [_LIVE_CHECKPOINT]) == ""


def test_steer_vocabulary_stays_in_sync_with_the_steer():
    """R5 cross-surface — FAILS IF: the steer is reworded without updating
    the terms the smoother keys on.

    The producer owns the vocabulary; this is the pin that makes the
    ownership real. Without it, re-wording risk.py silently strips the
    smoother of its ability to recognise the answers.
    """
    msg = risk_steer_message(RiskReading(
        score=0.51, depth_prior=0.52, effort_struggle=0.3,
        failure_pressure=0.2, step=8, band="high"))
    for term in STEER_DIRECTIVE_TERMS:
        assert term in msg, f"steer no longer asks for {term!r}"


def test_every_steer_term_is_matchable_by_the_smoother():
    """FAILS IF: a term is added to the vocabulary that the matcher cannot
    see — the two surfaces would disagree about one input.
    """
    joined = " ".join(STEER_DIRECTIVE_TERMS)
    # Two of the four directives in one blob must read as a checkpoint.
    assert is_governor_checkpoint_answer(joined) is True
