"""A beat inside a paragraph, right before the delivery (§4HD, 2026-09-15).

Request 9b6b8757 delivered a forensic report whose first paragraph was

    The Telegram post 363 snapshot returned the channel feed. Let me extract
    the notification quote from the authoritative sources and find the
    customer notification screenshot details.

followed by "The investigation is complete. Here's the forensic synthesis."
Three smoother rules looked and stood down: the beat did not OPEN the
paragraph (`_BEAT_RE`); it was a LONE beat (§4GO's run rule); nothing later
restated it (the trailing-beat rule). Pass 0 asks the question none of them
asked — does the NEXT paragraph deliver? — and cuts the beat SENTENCE only,
keeping the observation beside it. Each pin names the world it fails in.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import reply_smoothing as RS
from ghost_agent.core.reply_smoothing import smooth_reply, treat_reply

_OBS = "The Telegram post 363 snapshot returned the channel feed."
_BEAT = ("Let me extract the notification quote from the authoritative sources "
         "and find the customer notification screenshot details.")
_DELIVERY = "The investigation is complete. Here's the forensic synthesis."
_BODY = ("## Revolut September 2026 — OSINT Report\n\n"
         "**Best-supported attribution:** Italy, via the PEC channel; the sender "
         "address is not established.")
_LIVE = f"{_OBS} {_BEAT}\n\n{_DELIVERY}\n\n{_BODY}"


def test_the_live_opener_loses_its_beat_and_keeps_its_observation():
    """FAILS IF: pass 0 is absent — every earlier rule stands down on this
    shape (verified: the reply shipped verbatim)."""
    out = smooth_reply(_LIVE)
    assert _BEAT not in out
    assert out.startswith(_OBS)
    assert _DELIVERY in out and _BODY in out


def test_the_delivery_gate_is_the_condition():
    """FAILS IF: the next-paragraph check is dropped — the same beat with an
    ordinary paragraph after it is a finding the reply may still need."""
    out = smooth_reply(f"{_OBS} {_BEAT}\n\nThe feed lists 40 posts.\n\nNone name the sender.")
    assert _BEAT in out


@pytest.mark.parametrize("delivery", [
    "The investigation is complete.",
    "The report is ready.",
    "Here's what I verified and found.",
    "Here are the results.",
    "Done — the session is captured.",
    "## Findings",
    "**CONFIRMED:** the feed was live.",
    "Summary of the run:",
    "Results: two sources agree.",
    "Findings: the sender is unnamed.",
    "Bottom line: unknown sender.",
])
def test_every_delivery_opener_shape_arms_the_cut(delivery):
    """FAILS IF: one alternative of `_DELIVERY_OPENER_RE` is lost — each was
    the next paragraph of a real corpus hit."""
    out = smooth_reply(f"{_OBS} {_BEAT}\n\n{delivery}\n\n{_BODY}")
    assert _BEAT not in out


@pytest.mark.parametrize("beat", [
    "Let me extract the quote.",
    "I'll extract the quote.",
    "I will extract the quote.",
    "Now let me extract the quote.",
    "Now I'll extract the quote.",
    "Next, let me extract the quote.",
])
def test_every_opener_shape_is_a_beat(beat):
    """FAILS IF: one opener of `_MID_BEAT_RE` is lost."""
    out = smooth_reply(f"{_OBS} {beat}\n\n{_DELIVERY}\n\n{_BODY}")
    assert beat not in out


@pytest.mark.parametrize("kept", [
    # a colon-terminated beat is a lead-in to what follows (corpus: "Let me
    # be honest about what I mean by 'feels':")
    "Let me be clear about one thing:",
    # the colon can sit MID-sentence; the splitter must break there or the
    # whole "Let me be clear: the sender is unnamed." reads as one beat
    "Let me be clear: the sender is unnamed.",
    # the model reasoning, not announcing work (corpus: chess commentary)
    "I need to recapture to keep material equality.",
    "Let's not just talk about it.",
    # addressed to the user — an answer (corpus: "Let me grab the latest
    # headlines for you.", "I'll coach you in real-time — …")
    "Let me grab the latest headlines for you.",
    "I'll fix this so you can pick a session from a list.",
    "Let me know if you want the raw feed too.",
    # an offer with no "you" in it — only `_OFFER_RE` protects it
    "Let me know if the PDF is needed as well.",
])
def test_sentences_that_open_like_beats_but_are_answers_survive(kept):
    """FAILS IF: the colon / opener-set / addressed / offer exclusions are
    dropped — each string is a corpus false positive of the first draft."""
    out = smooth_reply(f"{_OBS} {kept}\n\n{_DELIVERY}\n\n{_BODY}")
    assert kept in out


def test_the_first_sentence_is_never_cut_by_pass_0():
    """FAILS IF: the `n and` guard goes — a paragraph that OPENS with a beat
    is pass 1's business, and pass 1 keeps a long one as content. This block
    is over `_MAX_NARRATION_CHARS`, so only pass 0 could touch it."""
    first = ("I'll summarise the evidence chain in order, source by source, "
             "quoting each notice as published and flagging where the "
             "secondary outlets copied one another instead of the primary.")
    rest = ("The first notice is Revolut's own. " * 6).strip()
    block = f"{first} {rest}"
    assert len(block) > RS._MAX_NARRATION_CHARS
    out = smooth_reply(f"{block}\n\n{_DELIVERY}\n\n{_BODY}")
    assert first in out


@pytest.mark.parametrize("block", [
    # a quoted passage the model is CITING (corpus 8a9e0d30 — there the
    # block also opened with '*'; here it opens with plain text, so only
    # the quote exclusion protects it)
    f'The earlier turn said: "The feed is live. {_BEAT}" That was all.',
    # a labelled section (corpus 87a13e41; '**' is also a list marker)
    f"**Next session:** the briefing surfaces this. {_BEAT}",
    # a blockquote — markup-led, and not a list marker
    f"> {_OBS} {_BEAT}",
    # a list item
    f"- {_OBS} {_BEAT}",
    # a fence
    f"```\n{_OBS} {_BEAT}\n```",
])
def test_quoted_labelled_listed_and_fenced_paragraphs_are_untouched(block):
    """FAILS IF: one of the block-level exclusions goes — the first two are
    the corpus's only non-beat hits of the sentence rule."""
    out = smooth_reply(f"{block}\n\n{_DELIVERY}\n\n{_BODY}")
    assert block in out


def test_the_paragraphs_own_line_breaks_survive_the_cut():
    """FAILS IF: the sentences are split and re-joined with spaces (the
    first draft) — a hard-wrapped paragraph would be re-flowed."""
    block = f"{_OBS}\nThe feed lists 40 posts.\n{_BEAT}"
    out = smooth_reply(f"{block}\n\n{_DELIVERY}\n\n{_BODY}")
    assert f"{_OBS}\nThe feed lists 40 posts." in out
    assert _BEAT not in out


def test_a_beat_deep_in_the_reply_is_also_cut():
    """FAILS IF: pass 0 runs on the first paragraph only — 16 of the 29
    corpus hits sit at paragraph 1 or later."""
    text = (f"Intro paragraph with the request restated.\n\nThe page renders "
            f"correctly. Let me take a screenshot to confirm the aesthetic.\n\n"
            f"Done — **GlassOS** is built and running.\n\n{_BODY}")
    out = smooth_reply(text)
    assert "Let me take a screenshot" not in out
    assert "The page renders correctly." in out


# The 7b2da5be shape (§4GO): three beat paragraphs the reply restates, then
# a fourth ("obs. Let me X.") that it does not — dropped by pass 1b because
# it sits in the run and an answer follows.
_RUN = (
    "The screenshot captured the embed widget but not the notification image. "
    "Let me navigate to the embed view.\n\n"
    "The embed view loaded the notification image. Now I'll read the image itself.\n\n"
    "The image text is the customer notice. Let me extract the text on single=1.\n\n"
    "The extract_text on single=1 gave the same capped preview. "
    "Let me take a full-page screenshot.\n\n"
    "Here's what I verified and found.\n\n"
    "I captured the embed widget, navigated to the embed view, loaded and read "
    "the notification image, and extracted the customer notice text on single=1 "
    "via vision OCR.\n\n" + _BODY)


def test_the_last_member_of_a_beat_run_still_goes_whole():
    """FAILS IF: pass 3 runs BEFORE passes 1/1b — it would trim the fourth
    paragraph to a one-sentence observation that 1b no longer recognises
    as a beat, and "The extract_text on single=1 gave the same capped
    preview." would open the delivered reply (the §4GO regression)."""
    out = smooth_reply(_RUN)
    assert "The extract_text on single=1 gave the same capped preview." not in out
    assert out.startswith("Here's what I verified and found.")


def test_next_means_the_next_delivered_paragraph():
    """FAILS IF: pass 3 looks at the paragraph's ORIGINAL neighbour — a
    dropped beat between the observation and the delivery would shield
    the beat. The observation is long, so 1b cannot take it whole."""
    long_obs = ("The page lists forty posts from the channel, each with a "
                "timestamp, a view count and the first line of the body; the "
                "notice is post 363, dated 12 September, and its attachment "
                "is a single image of the customer notification that the "
                "feed view truncates to the first four lines of text, "
                "cutting the sender field before it is shown.")
    assert len(long_obs) > RS._MAX_NARRATION_CHARS
    beat = "Let me open the single-post view to read the rest."
    text = (f"{long_obs} {beat}\n\nLet me check the attachment size first.\n\n"
            f"{_DELIVERY}\n\n{_BODY}")
    out = smooth_reply(text)
    assert "Let me check the attachment size first." not in out   # pass 1
    assert beat not in out                                        # pass 3
    assert long_obs in out


def test_the_rule_is_idempotent_and_reaches_the_delivered_view():
    """FAILS IF: `treat_reply`'s narration-only veto swallows the trim, or a
    second pass changes the text again."""
    once = treat_reply(_LIVE, n_real_tools=2)
    assert _BEAT not in once and _OBS in once
    assert treat_reply(once, n_real_tools=2) == once


def test_the_smoothing_gate_still_applies():
    """FAILS IF: pass 0 bypasses the ≥2-real-tools gate."""
    assert treat_reply(_LIVE, n_real_tools=1) == _LIVE
