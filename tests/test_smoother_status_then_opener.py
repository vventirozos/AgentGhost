"""§4HX — a delivery may open with one short status sentence before the opener.

Live (req abb8fdb7): "…Let me screenshot and read the actual prices." stood
above "The page loaded and read fine. Here's what's shown on …" because
every "next block delivers" test looked only at the paragraph's FIRST
sentence. Corpus (Aug–Sep): 148 such paragraphs, 10 after a trailing beat.
"""
from __future__ import annotations

from ghost_agent.core.reply_smoothing import _opens_by_delivering, smooth_reply, treat_reply

_BEAT_PARA = ("The page loaded but prices show as `max/mo.` placeholders in the text preview "
              "(JS-rendered). Let me screenshot and read the actual prices.")
_DELIVERY = "The page loaded and read fine. Here's what's shown on https://www.hetzner.com/cloud/server/:"
_BODY = "**Cheapest shared-vCPU plan: \"Cost-Optimized\" — from €5.99/month** (incl. VAT)."


def test_status_then_opener_is_a_delivery():
    assert _opens_by_delivering(_DELIVERY)
    assert _opens_by_delivering("Fixed. Here's what was wrong and what changed:")
    assert _opens_by_delivering("Both files are read. Here's the comparison.")
    assert _opens_by_delivering("Here's the comparison.")          # the old shape still is


def test_a_long_first_sentence_is_not_a_status():
    long_first = ("The page loaded after three retries through two different Tor circuits and "
                  "one full profile reset, which took most of the turn. Here's what's shown:")
    assert len(long_first.split(". ")[0]) > 80
    assert not _opens_by_delivering(long_first)
    assert not _opens_by_delivering("The page loaded. It has three plan cards.")   # no opener at all
    # exactly ONE status sentence — an opener buried third is a paragraph
    # that talks first and delivers later, not one that opens by delivering
    assert not _opens_by_delivering("The page loaded. It read fine. Here's what's shown:")
    assert not _opens_by_delivering("")


def test_the_live_beat_above_a_status_then_opener_delivery_is_cut():
    """FAILS IF: pass 3 still keys on the first sentence only — the beat
    sentence survives; the observation before it must stay."""
    out = smooth_reply("\n\n".join([_BEAT_PARA, _DELIVERY, _BODY]))
    assert "Let me screenshot" not in out
    assert "placeholders in the text preview" in out
    assert _DELIVERY in out and _BODY in out


def test_treat_reply_on_the_live_record():
    out = treat_reply("\n\n".join([_BEAT_PARA, _DELIVERY, _BODY]), n_real_tools=3)
    assert "Let me screenshot" not in out and _BODY in out


def test_a_one_sentence_handoff_above_a_status_then_opener_is_dropped_too():
    """§4HE's hand-off rule uses the same predicate."""
    out = smooth_reply("\n\n".join(["I have enough to finalize.", _DELIVERY, _BODY]))
    assert "I have enough to finalize" not in out and _BODY in out
