"""Tests for the verifier gate gaining real authority over the outcome.

Regression target: the gate was advisory-only. A REFUTED verdict merely
stapled a note onto the reply; the finalize confidence was computed from
historical domain competence + a hardcoded neutral entropy, so it equalled
competence (0.92/0.96) regardless of whether the work was actually broken.
Result: req_44 finished "REFUTED (95%)" yet reported C=0.92 below=no, and
req_C0 finalised an untested file write ("gate skipped") at C=0.96. Two
fixes locked in here:
  1. confidence.score(outcome_penalty=...) — an objective negative pulls
     the reading below threshold no matter how strong the priors are.
  2. _is_unverified_mutation(...) — finalising on an untested write/replace
     is recorded as a failed outcome, not a silent success.
"""
import pytest

from ghost_agent.core.confidence import CompositeConfidence
from ghost_agent.core.agent import _is_unverified_mutation


# ── outcome_penalty on the confidence reading ────────────────────────
def test_outcome_penalty_default_is_noop():
    cc = CompositeConfidence()
    base = cc.score(normalised_entropy=0.5, competence_p_success=0.95, n_observations=600)
    pen0 = cc.score(normalised_entropy=0.5, competence_p_success=0.95,
                    n_observations=600, outcome_penalty=0.0)
    assert base.composite == pytest.approx(pen0.composite)


def test_refuted_penalty_flips_below_threshold():
    # Mimic the PRODUCTION weight-collapse the calibration refit produced:
    # entropy weight ~0, competence dominates → composite ≈ competence.
    # This is the regime where req_44/C0 reported C=0.92/0.96 below=no on a
    # REFUTED/untested result.
    cc = CompositeConfidence(w_entropy=0.02, w_competence=0.98, threshold=0.89)
    good = cc.score(normalised_entropy=0.5, competence_p_success=0.96, n_observations=600)
    assert not good.below_threshold  # the broken pre-fix behaviour
    assert good.composite > 0.89
    # ...but with the verifier's REFUTED ground-truth, it must drop below.
    refuted = cc.score(normalised_entropy=0.5, competence_p_success=0.96,
                       n_observations=600, outcome_penalty=0.8)
    assert refuted.below_threshold
    assert refuted.composite < good.composite


def test_outcome_penalty_is_monotonic():
    cc = CompositeConfidence()
    a = cc.score(normalised_entropy=0.5, competence_p_success=0.9,
                 n_observations=100, outcome_penalty=0.2)
    b = cc.score(normalised_entropy=0.5, competence_p_success=0.9,
                 n_observations=100, outcome_penalty=0.8)
    assert b.composite < a.composite


# ── _is_unverified_mutation classification ───────────────────────────
def test_write_success_is_unverified_mutation():
    tool = {"name": "file_system",
            "content": "SUCCESS: Wrote 21044 chars to 'index.html'. Script-side path: 'index.html'."}
    assert _is_unverified_mutation(tool) is True


def test_replace_success_is_unverified_mutation():
    tool = {"name": "file_system",
            "content": "SUCCESS: Exact match found and replaced in 'index.html'."}
    assert _is_unverified_mutation(tool) is True


def test_file_read_is_not_a_mutation():
    # a READ returns the file body, not a write/replace SUCCESS line
    tool = {"name": "file_system",
            "content": "<html><body><script>const player = {...}</script></body></html>"}
    assert _is_unverified_mutation(tool) is False


def test_non_filesystem_tool_is_not_a_mutation():
    # an execute that happens to print "wrote" is not a file mutation
    tool = {"name": "execute", "content": "SUCCESS: the program wrote output"}
    assert _is_unverified_mutation(tool) is False


def test_none_and_garbage_are_safe():
    assert _is_unverified_mutation(None) is False
    assert _is_unverified_mutation({}) is False
    assert _is_unverified_mutation({"name": "file_system"}) is False
