"""Regression tests for UncertaintyTracker's object-or-index API.

Before the fix, `flag_unknown` returned an `Unknown` object while
`resolve_unknown` only accepted an int index. A caller holding the
returned object and passing it back (the most natural pattern) hit:

    TypeError: '<=' not supported between instances of 'int' and 'Unknown'

Same asymmetry on `flag_assumption` → `verify_assumption`. These
tests pin the symmetric API so a future edit can't silently reintroduce
the mismatch.
"""

import pytest

from ghost_agent.core.uncertainty import (
    UncertaintyTracker, Unknown, Assumption,
)


# ------------------------------------------------------------------
# resolve_unknown: accepts both Unknown object AND int index
# ------------------------------------------------------------------

def test_resolve_unknown_accepts_returned_object():
    ut = UncertaintyTracker()
    u = ut.flag_unknown("timezone?", impact=4)
    assert ut.resolve_unknown(u, "UTC") is True
    assert u.resolved is True
    assert u.resolved_value == "UTC"


def test_resolve_unknown_accepts_int_index():
    """Backwards compat: the old int-index path still works."""
    ut = UncertaintyTracker()
    u = ut.flag_unknown("locale?", impact=3)
    assert ut.resolve_unknown(0, "en_US") is True
    assert u.resolved is True
    assert u.resolved_value == "en_US"


def test_resolve_unknown_rejects_orphan_object():
    """An Unknown not in the tracker's list doesn't accidentally
    resolve some other entry by field equality."""
    ut = UncertaintyTracker()
    ut.flag_unknown("real entry", impact=3)
    orphan = Unknown(what="not in tracker", impact=1, resolution="x")
    assert ut.resolve_unknown(orphan, "irrelevant") is False


def test_resolve_unknown_rejects_out_of_range_int():
    ut = UncertaintyTracker()
    ut.flag_unknown("one", impact=2)
    assert ut.resolve_unknown(99, "x") is False
    assert ut.resolve_unknown(1, "x") is False  # len=1, valid index is 0


def test_resolve_unknown_rejects_negative_int():
    """No Python-style negative indexing. `-1` is a bug, not a feature."""
    ut = UncertaintyTracker()
    ut.flag_unknown("x", impact=2)
    assert ut.resolve_unknown(-1, "last") is False


def test_resolve_unknown_rejects_unexpected_types():
    ut = UncertaintyTracker()
    ut.flag_unknown("x", impact=2)
    for bad in ("string", None, 3.14, [0], {"idx": 0}):
        assert ut.resolve_unknown(bad, "v") is False, (
            f"unexpected type {type(bad).__name__} shouldn't resolve anything"
        )


def test_resolve_unknown_identity_distinguishes_duplicate_text():
    """Two unknowns with the same `what` text are LEGITIMATELY
    distinct entries (the agent might flag the same question at
    different points in a turn). Identity comparison must distinguish
    them — equality on `what` would falsely resolve both."""
    ut = UncertaintyTracker()
    u1 = ut.flag_unknown("same text", impact=2)
    u2 = ut.flag_unknown("same text", impact=2)
    ut.resolve_unknown(u1, "resolved first")
    assert u1.resolved is True
    assert u2.resolved is False, "u2 must be untouched by u1's resolve"


# ------------------------------------------------------------------
# verify_assumption mirrors the same API
# ------------------------------------------------------------------

def test_verify_assumption_accepts_returned_object():
    ut = UncertaintyTracker()
    a = ut.flag_assumption("user wants python", confidence=0.4)
    assert ut.verify_assumption(a, was_correct=True) is True
    assert a.verified is True
    assert a.was_correct is True


def test_verify_assumption_accepts_int_index():
    ut = UncertaintyTracker()
    a = ut.flag_assumption("user is on mac", confidence=0.9)
    assert ut.verify_assumption(0, was_correct=False) is True
    assert a.verified is True
    assert a.was_correct is False


def test_verify_assumption_rejects_orphan_object():
    ut = UncertaintyTracker()
    ut.flag_assumption("real", confidence=0.5)
    orphan = Assumption(claim="not in tracker", confidence=0.5, basis="")
    assert ut.verify_assumption(orphan, True) is False


def test_verify_assumption_rejects_out_of_range_int():
    ut = UncertaintyTracker()
    ut.flag_assumption("x", confidence=0.5)
    assert ut.verify_assumption(99, True) is False


# ------------------------------------------------------------------
# Integration: mixed object + int usage in the same session
# ------------------------------------------------------------------

def test_mixed_object_and_index_calls_share_state():
    ut = UncertaintyTracker()
    u1 = ut.flag_unknown("u1", impact=2)
    u2 = ut.flag_unknown("u2", impact=3)
    u3 = ut.flag_unknown("u3", impact=4)
    # Resolve by object, by int, by object again
    assert ut.resolve_unknown(u1, "a") is True
    assert ut.resolve_unknown(1, "b") is True
    assert ut.resolve_unknown(u3, "c") is True
    assert [u.resolved_value for u in (u1, u2, u3)] == ["a", "b", "c"]


def test_downstream_filters_see_resolved_state_regardless_of_api():
    """`get_critical_unknowns` filters out resolved entries. Both
    object-form and int-form resolution must flip that flag equally."""
    ut = UncertaintyTracker()
    u_high_obj = ut.flag_unknown("critical via obj", impact=5)
    u_high_idx = ut.flag_unknown("critical via idx", impact=5)
    assert len(ut.get_critical_unknowns(min_impact=4)) == 2

    ut.resolve_unknown(u_high_obj, "x")
    ut.resolve_unknown(1, "y")

    assert ut.get_critical_unknowns(min_impact=4) == []
