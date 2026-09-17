"""The working log over the size bound (§4HJ, 2026-09-16, req 6afaf940).

A trailing-beat paragraph the reply restates is dropped by pass 1 — unless it
is over `_MAX_NARRATION_CHARS`, the bound that protects long content
paragraphs. Rerun 1 of the operator's loop shipped three such paragraphs
above "The investigation is complete.": "I have good coverage. Key candidate
emerging: Italy (…). Let me run targeted searches to confirm…" (314 chars,
restatement test already True). Corpus: 11 such paragraphs, every trailing
beat in them stale. Pass 1c cuts the beat sentence(s) and keeps the
observation. Each pin names the world it fails in.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import reply_smoothing as RS
from ghost_agent.core.reply_smoothing import smooth_reply

_OBS = ("I have good coverage. Key candidate emerging: **Italy** (`pec.interno.it` / "
        "`interno.it` — Italian Ministry of Interior PEC domain via InfoCert certified "
        "email) from tech-insider.org, with the sender mailbox still unnamed by Revolut.")
_BEAT = ("Let me run targeted searches to confirm the exact sender address, the "
         "customer-notification screenshot, and validate the Italian attribution.")
# ⚠ Every pin puts a plain observation paragraph BETWEEN the beat paragraph and
# the answer. Directly before a delivery opener, pass 3 already cuts the beat
# (§4HD) — the first draft of these pins sat there and passed with 1c raising
# (the battery's KNOWNBAD control survived). The live paragraphs sat three
# deep above the opener; that is the world 1c exists for.
_BETWEEN = "The second search round returned the same three secondary outlets and nothing new."
_ANSWER = ("The investigation is complete. Here's the forensic synthesis.\n\n"
           "# Revolut September 2026\n\n**Best-supported attribution: Italy — Ministry of "
           "Interior — via its PEC certified-email channel (`pec.interno.it` / `interno.it`), "
           "operated through InfoCert.** The exact sender mailbox was not publicly exposed; "
           "Revolut never named the agency. Key candidate from the coverage: the Italian "
           "government domain; the customer-notification screenshot confirms no sender "
           "address, and the Interior Ministry attribution was validated against the "
           "targeted searches. Sources: tech-insider.org, KELA.")


def test_the_live_paragraph_keeps_its_finding_and_loses_its_beat():
    """FAILS IF: pass 1c is absent — the paragraph is over the bound and
    ships whole (the live world)."""
    para = f"{_OBS} {_BEAT}"
    assert len(para) > RS._MAX_NARRATION_CHARS
    out = smooth_reply(f"{para}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert _OBS in out
    assert _BEAT not in out


def test_the_trim_needs_the_restatement_evidence():
    """FAILS IF: 1c cuts on shape alone — an over-bound observation + beat
    that the reply never repeats may be the only place the finding lives."""
    para = f"{_OBS} {_BEAT}"
    out = smooth_reply(f"{para}\n\nThe feed lists forty unrelated posts.\n\nNothing else was found today.")
    assert _BEAT in out


def test_a_paragraph_under_the_bound_is_still_pass_1s_whole_drop():
    """FAILS IF: 1c reaches below the bound — there pass 1 drops the whole
    paragraph (§4GO), and the two must not compete."""
    short_obs = ("Best-supported attribution: Italy, Ministry of Interior, PEC channel via "
                 "InfoCert; sender mailbox not exposed.")
    para = f"{short_obs} {_BEAT}"
    assert len(para) <= RS._MAX_NARRATION_CHARS
    out = smooth_reply(f"{para}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert short_obs not in out and _BEAT not in out


def test_two_trailing_beats_are_both_peeled():
    """FAILS IF: only the last sentence is examined — the corpus shape
    "…obs. Now I'll restructure it. Let me first read the code." ends on
    two beats."""
    second = "Now I'll confirm the Interior Ministry attribution."
    para = f"{_OBS} {second} {_BEAT}"
    out = smooth_reply(f"{para}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert _OBS in out and second not in out and _BEAT not in out


def test_an_offer_at_the_end_is_not_a_beat():
    """FAILS IF: the offer guard goes — "Let me know if…" ends more replies
    than any beat and is addressed to the user. The peel must stop AT the
    offer when a beat follows it (a paragraph ending on the offer alone
    never reaches 1c: `_trailing_beat` already excludes it)."""
    offer = "Let me know if you want the raw feed as well."
    para = f"{_OBS} {offer} {_BEAT}"
    out = smooth_reply(f"{para}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert offer in out and _BEAT not in out


def test_the_paragraphs_own_whitespace_survives():
    """FAILS IF: the trim re-flows the paragraph — a hard-wrapped one keeps
    its line breaks."""
    # the break sits at a sentence boundary — exactly what a split-and-join
    # would consume
    obs_wrapped = _OBS.replace("coverage. Key", "coverage.\nKey")
    out = smooth_reply(f"{obs_wrapped} {_BEAT}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert obs_wrapped in out and _BEAT not in out


def test_the_trimmed_paragraph_is_no_longer_a_run_member():
    """FAILS IF: 1c runs after 1b — the trimmed paragraph must be judged as
    content by the run rule (it carries the finding), not swept as a beat
    beside a dropped one."""
    beat_only = "Let me check the last source too."
    para = f"{_OBS} {_BEAT}"
    out = smooth_reply(f"{beat_only}\n\n{para}\n\n{_BETWEEN}\n\n{_ANSWER}")
    assert _OBS in out


@pytest.mark.parametrize("ready", [
    "I have strong consolidated evidence.",
    "I have good coverage.",
    "I have solid data on three of the four crashes.",
])
def test_the_residue_is_a_readiness_hand_off_before_a_delivery(ready):
    """FAILS IF: the readiness vocabulary does not know the research turns'
    phrasing — the residue 1c leaves ships above the answer."""
    out = smooth_reply(f"{ready}\n\n{_ANSWER}")
    assert ready not in out and out.startswith("The investigation is complete.")


def test_readiness_needs_a_content_noun():
    """FAILS IF: the widening matches "I have good news for you" — 'news'
    is not in the vocabulary and the sentence is addressed to the user."""
    keep = "I have good news for the team."
    out = smooth_reply(f"{keep}\n\n{_ANSWER}")
    assert keep in out
