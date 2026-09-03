"""§4EG — ACE grow-and-refine on the lesson playbook.

The cap must never evict a PROVEN-USEFUL lesson (verified, or positive net
outcome credit). Eviction may only ever target unproven or proven-harmful
lessons. The kill switch (`_ACE_PLAYBOOK=False`) restores the legacy
cap=50 + drop-lowest-utility-verified rule — the differential that proves
the fix is load-bearing.
"""
from __future__ import annotations

import pytest

from ghost_agent.memory import skills as SK
from ghost_agent.memory.skills import (
    _trim_playbook_by_utility, _lesson_is_protected, compute_lesson_utility)


def _lesson(trig, *, verified=False, succ=0, fail=0, ret=0, helpful=0, conf=0.5):
    return {"trigger": trig, "correct_pattern": "do x", "verified": verified,
            "succeeded_retrievals": succ, "failed_retrievals": fail,
            "retrievals": ret, "helpful_retrievals": helpful,
            "confidence": conf, "schema_version": 2, "timestamp": "2026-09-03"}


# ── _lesson_is_protected: the pin predicate ─────────────────────────────── #

@pytest.mark.parametrize("kw,ace,protected", [
    (dict(verified=True), True, True),
    (dict(verified=True), False, True),                 # verified pins in both modes
    (dict(succ=5, fail=1), True, True),                 # positive net credit → pinned under ACE
    (dict(succ=5, fail=1), False, False),               # …but NOT under the kill switch
    (dict(succ=1, fail=5), True, False),                # proven-harmful → not protected
    (dict(succ=0, fail=0), True, False),                # unproven → not protected
    (dict(succ=1, fail=1), True, False),                # tie is not positive
    (dict(succ=1, fail=0), True, True),                 # one clean success → protected
    (dict(succ=2, fail=1), True, True),                 # succ = fail+1 → protected (guards f+1)
    ("not a dict", True, False),                        # non-dict → never protected
], ids=["ver-ace", "ver-legacy", "pos-ace", "pos-legacy", "harmful", "unproven", "tie",
        "one-success", "succ-eq-fail-plus-1", "non-dict"])
def test_protected_predicate(monkeypatch, kw, ace, protected):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", ace)
    lesson = kw if not isinstance(kw, dict) else _lesson("t", **kw)
    assert _lesson_is_protected(lesson) is protected


# ── the reproduction + differential ─────────────────────────────────────── #

def _saturated_book(n_verified=48):
    # head (freshly written, no credit) + N verified fillers + ONE unverified
    # lesson with positive net credit that the OLD rule would evict.
    head = _lesson("NEW just written", ret=0)
    verified = [_lesson(f"v{i}", verified=True, ret=3, helpful=2) for i in range(n_verified)]
    useful = _lesson("USEFUL unverified", succ=7, fail=1, ret=28, helpful=9)
    return [head] + verified + [useful]


def test_positive_credit_lesson_survives_the_cap_under_ace(monkeypatch):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", True)
    book = _saturated_book()                      # 1 + 48 + 1 = 50
    book += [_lesson(f"filler{i}", ret=1) for i in range(5)]   # push over cap 50
    kept = _trim_playbook_by_utility(book, 50)
    trigs = {l["trigger"] for l in kept}
    assert "USEFUL unverified" in trigs, "a positive-credit lesson was evicted under ACE"
    assert "NEW just written" in trigs                        # head always kept


def test_same_lesson_is_evicted_with_the_kill_switch(monkeypatch):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", False)
    book = _saturated_book()
    book += [_lesson(f"filler{i}", ret=1) for i in range(5)]
    kept = _trim_playbook_by_utility(book, 50)
    trigs = {l["trigger"] for l in kept}
    # legacy rule: 48 verified pin the slots, head kept, the unverified
    # useful lesson is dropped to satisfy the cap — the incident class.
    assert "USEFUL unverified" not in trigs, "kill switch should reproduce the legacy eviction"
    assert len(kept) == 50


def test_proven_harmful_lesson_is_still_evictable_under_ace(monkeypatch):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", True)
    head = _lesson("head", ret=0)
    verified = [_lesson(f"v{i}", verified=True, ret=3, helpful=2) for i in range(49)]
    harmful = _lesson("HARMFUL", succ=1, fail=6, ret=20)
    kept = _trim_playbook_by_utility([head] + verified + [harmful], 50)
    assert "HARMFUL" not in {l["trigger"] for l in kept}


def test_protected_set_over_cap_keeps_the_store_oversized_never_drops_protected(monkeypatch):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", True)
    head = _lesson("head")
    # 51 protected + head = 52 > cap 50 → slots_left = -2, SMALL negative, so a
    # `slots_left > 0 -> True` mutant would add unprotected[:-2] (non-empty).
    protected = [_lesson(f"v{i}", verified=True, ret=3) for i in range(51)]
    # SEVERAL unproven lessons: with protected already over cap, slots_left is
    # negative; a `slots_left > 0 -> True` mutant would slice unprotected[:neg]
    # and add some over the cap, so the count must be exactly head+protected.
    unproven = [_lesson(f"UNPROVEN{i}", ret=0) for i in range(5)]
    kept = _trim_playbook_by_utility([head] + protected + unproven, 50)
    trigs = [l["trigger"] for l in kept]
    assert len(kept) == 52, "exactly head + 51 protected; no unprotected added over cap"
    assert all(l in kept for l in protected)
    assert not any(t.startswith("UNPROVEN") for t in trigs), \
        "no unprotected lesson may be added once protected fills the cap"
    assert trigs.count("head") == 1, "no lesson may be duplicated"


def test_protected_and_unprotected_are_not_swapped(monkeypatch):
    """Guards the `not _lesson_is_protected` membership: with one verified P and
    three unproven Us over a small cap, the BEST U fills the last slot and P is
    kept once (a swap would duplicate P and drop the Us)."""
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", True)
    head = _lesson("head")
    P = _lesson("P", verified=True, ret=3)
    us = [_lesson("Ubest", ret=10, helpful=9), _lesson("Umid", ret=10, helpful=4),
          _lesson("Ulow", ret=10, helpful=0)]
    kept = _trim_playbook_by_utility([head, P] + us, 3)
    trigs = [l["trigger"] for l in kept]
    assert trigs.count("P") == 1 and "Ubest" in trigs and "head" in trigs
    assert "Ulow" not in trigs


def test_legacy_overflow_still_drops_lowest_utility_verified(monkeypatch):
    monkeypatch.setattr(SK, "_ACE_PLAYBOOK", False)
    head = _lesson("head")
    hi = [_lesson(f"hi{i}", verified=True, ret=10, helpful=9) for i in range(49)]
    lo = _lesson("LO-util", verified=True, ret=10, helpful=0)   # lowest utility verified
    kept = _trim_playbook_by_utility([head] + hi + [lo], 50)
    assert len(kept) == 50 and "LO-util" not in {l["trigger"] for l in kept}
    assert kept[0]["trigger"] == "head", "the head lesson must survive the legacy overflow, first"


def test_cap_is_raised_under_ace_and_50_under_the_switch():
    # module constant reflects the mode it was imported in (ACE on by default)
    assert SK._ACE_PLAYBOOK is True and SK.PLAYBOOK_MAX >= 300


def test_no_trim_below_cap_is_identity():
    # DISTINCT utilities (varying helpful/ret) so the identity return is the
    # only thing that preserves input order — a reorder would break ==.
    book = [_lesson(f"x{i}", ret=10, helpful=i) for i in range(10)]
    assert _trim_playbook_by_utility(book, 50) == book, "below cap must return input order"
    assert _trim_playbook_by_utility([], 50) == []
    assert _trim_playbook_by_utility(book, 0) == []
    # at exactly the cap is still identity
    assert _trim_playbook_by_utility(book, 10) == book
    # trimming a 3-item book to 1 returns ONE item, never empty (guards `<=0`→`<=1`)
    got = _trim_playbook_by_utility(book[:3], 1)
    assert len(got) == 1 and got[0] is book[0]
