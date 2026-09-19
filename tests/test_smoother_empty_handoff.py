"""What pass 3 leaves behind: the empty hand-off (§4HE, 2026-09-15).

Request 5fa6aa97 delivered

    I have enough to finalize.

    The investigation is complete.

    The investigation is complete. Here's the forensic synthesis.

    ## Revolut September 2026 — …

Two iterations' worth of hand-off: each was "readiness + beat", pass 3 cut
the beats as designed and left the declarations, which are not findings —
one is a verbatim sentence of the next paragraph, the other says only that
the model is ready. Measured: 356 short paragraphs precede a delivery on
the corpus and nearly all are lead-ins or headings, so the rule has two
triggers only — verbatim containment and the readiness vocabulary — and
needs the paragraph to carry no content. Each pin names the world it
fails in.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.reply_smoothing import smooth_reply, treat_reply

_OPENER = "The investigation is complete. Here's the forensic synthesis."
_BODY = ("## Revolut September 2026 — Fraudulent Government Request: Forensic Summary\n\n"
         "### The confirmed core (confidence ~90%)\n- **12 Sep 2026:** Revolut confirmed…")
_LIVE = ("I have enough to finalize. Let me update the report file with the KELA "
         "findings and then give the final answer.\n\n"
         "The investigation is complete. Let me verify the report file is in good "
         "shape and then deliver.\n\n" + _OPENER + "\n\n" + _BODY)


def test_the_live_reply_opens_with_its_real_opener():
    """FAILS IF: the hand-off rule is absent — pass 3 alone leaves
    "I have enough to finalize." and a duplicate "The investigation is
    complete." above the opener (verified: the reply shipped that way)."""
    out = treat_reply(_LIVE, n_real_tools=2)
    assert out.startswith(_OPENER), out[:120]
    assert "I have enough to finalize" not in out
    assert out.count("The investigation is complete.") == 1
    assert _BODY in out


def test_a_verbatim_sentence_of_the_next_paragraph_goes():
    """FAILS IF: the containment trigger is dropped. §4HQ: the sample is a
    lead-in that is NOT readiness vocabulary ("Here is the forensic
    synthesis."), so only containment can drop it — "The investigation is
    complete." now falls to the third-person readiness trigger as well."""
    lead = "The domain question is settled."
    nxt = f"Findings. {lead} The sender used pec.interno.it."
    out = smooth_reply(f"{lead}\n\n{nxt}\n\n{_BODY}")
    assert out.startswith(nxt)
    assert out.count(lead) == 1


@pytest.mark.parametrize("ready", [
    "I have enough to finalize.",
    "I now have all three sources.",
    "I have gathered sufficient evidence across the required targets.",
    "I have enough concrete data to deliver the forecast.",
    "I've got everything I need.",
    "That's enough.",
])
def test_a_readiness_declaration_before_the_delivery_goes(ready):
    """FAILS IF: one alternative of the readiness vocabulary is lost —
    each is a corpus hit (45675adf, 193a1054, 44723cc9, 0a435424)."""
    out = smooth_reply(f"{ready}\n\n{_OPENER}\n\n{_BODY}")
    assert ready not in out
    assert out.startswith(_OPENER)


@pytest.mark.parametrize("kept", [
    # lead-ins and headings are the 356-class: never touched
    "Here's what I changed:",
    "## ⚡ Performance",
    "System health check complete. Here's the summary:",
    # readiness with CONTENT (a number, code, emphasis) is a finding
    "I now have all 12 sources.",
    "I have enough: `legalmail.it` is the recipient.",
    "I have **enough**.",
    # addressed to the user
    "I have enough to answer your question.",
    # two sentences — the second is not a beat, so it is not a hand-off
    "I have enough to finalize. The sender is still unnamed.",
    # a long readiness sentence summarises the evidence — content, whatever
    # it opens with (the 160-char bound)
    "I have enough evidence now to write the forecast with confidence, covering "
    "the three harvests, the two export markets, the retail series and the "
    "ministry's own projection for the coming season.",
])
def test_paragraphs_that_are_not_empty_hand_offs_survive(kept):
    """FAILS IF: the content / addressed / single-sentence guards go, or
    the rule widens past its two triggers."""
    out = smooth_reply(f"{kept}\n\n{_OPENER}\n\n{_BODY}")
    assert kept in out


def test_a_heading_contained_in_the_next_heading_survives():
    """FAILS IF: the markup guard goes — "## Results" is a verbatim
    substring of "## Results by source", and a heading is never a
    hand-off."""
    text = "## Results\n\n## Results by source\n\n**Reuters:** confirmed the core."
    assert smooth_reply(text).startswith("## Results\n\n## Results by source")


def test_the_rule_needs_a_delivery_after_it():
    """FAILS IF: the delivery gate is dropped — "I have enough." followed
    by an ordinary paragraph may precede more work, not the answer."""
    text = "I have enough to finalize.\n\nThe feed lists forty posts.\n\nNone name the sender."
    assert "I have enough to finalize." in smooth_reply(text)


def test_a_horizontal_rule_is_looked_through():
    """FAILS IF: `---` between the hand-off and the delivery shields it —
    corpus 45675adf: "I now have all three sources. Let me synthesize the
    answer." / "---" / "## 1. Freedom in the 1776 US Declaration…"."""
    text = ("I now have all three sources. Let me synthesize the answer.\n\n---\n\n"
            "## 1. Freedom in the 1776 US Declaration of Independence\n\n"
            "**Source:** the Declaration itself.")
    out = smooth_reply(text)
    assert out.startswith("---"), out[:80]
    assert "three sources" not in out


def test_the_beat_trim_and_the_hand_off_compose():
    """FAILS IF: the hand-off test runs on the UNTRIMMED paragraph — the
    live shape is "readiness + beat", two sentences, and only after pass 3
    cuts the beat is it a single readiness declaration."""
    text = ("I have enough concrete data to deliver the forecast. Let me write it up "
            f"as a report file and present it.\n\n{_OPENER}\n\n{_BODY}")
    assert smooth_reply(text).startswith(_OPENER)


def test_the_final_paragraph_is_never_a_candidate():
    """FAILS IF: the final block is judged against nothing and dropped —
    the reply would lose its last (here its only) statement."""
    text = "I have enough to finalize.\n\nThe investigation is complete."
    assert smooth_reply(text).endswith("The investigation is complete.")


# ── §4HQ (2026-09-16, req 3d3e0681) — the hand-off in the third person ──
_Q_OBS = ("The ilpost.it article and multiple sources confirm the domain is "
        "`pec.interno.it` (Ministry of Interior), not `pec.intero.it` as the "
        "earlier report had stated.")
_Q_HANDOFF = "The report is complete and verified against all constraints."
_Q_OPENER = "The investigation is complete. Here's the forensic summary."
_Q_BODY = "---\n\n## Best-Supported Attribution: Italy\n\n**Domain:** `pec.interno.it`."


def test_a_third_person_readiness_declaration_before_the_delivery_is_dropped():
    """FAILS IF: the third-person shape is not readiness vocabulary — the
    live record (04dff0ad) kept it because it also matches the delivery
    opener, so pass 3 took it for the delivery."""
    text = "\n\n".join([_Q_OBS, _Q_HANDOFF, _Q_OPENER, _Q_BODY])
    out = smooth_reply(text)
    assert _Q_HANDOFF not in out
    assert _Q_OBS in out                      # the observation paragraph stays
    assert _Q_OPENER in out and "## Best-Supported" in out


def test_the_declaration_with_content_is_kept():
    withc = "The report is complete: 8 pages, 21 sources, `pec.interno.it` validated."
    out = smooth_reply("\n\n".join([_Q_OBS, withc, _Q_OPENER, _Q_BODY]))
    assert withc in out
    # §4IY / battery 70 W20: a conjoined tail that says WHERE is content too
    where = "The report is complete and saved to the Downloads folder."
    out = smooth_reply("\n\n".join([_Q_OBS, where, _Q_OPENER, _Q_BODY]))
    assert where in out
    where2 = "The report is complete and lives in the sandbox now."
    assert where2 in smooth_reply("\n\n".join([_Q_OBS, where2, _Q_OPENER, _Q_BODY]))


def test_the_declaration_not_followed_by_a_delivery_is_kept():
    out = smooth_reply("\n\n".join([_Q_OBS, _Q_HANDOFF, "It was reviewed twice by the team.", _Q_BODY]))
    assert _Q_HANDOFF in out


def test_a_two_sentence_declaration_is_kept():
    two = _Q_HANDOFF + " Nothing further is required."
    out = smooth_reply("\n\n".join([_Q_OBS, two, _Q_OPENER, _Q_BODY]))
    assert two in out


def test_treat_reply_on_the_live_record_shape():
    text = "\n\n".join([
        "The investigation is essentially complete. I have comprehensive data from "
        "ilpost.it (confirmed `pec.interno.it`), pasqualepillitteri.it and the full article.",
        _Q_HANDOFF, _Q_OPENER, _Q_BODY])
    out = treat_reply(text, n_real_tools=25)
    assert _Q_HANDOFF not in out
    assert "pasqualepillitteri.it" in out and "## Best-Supported" in out
