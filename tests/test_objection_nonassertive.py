"""objection.py rule 4 — non-assertive fragments (§4BD, 2026-08-12).

A stated NEXT STEP ("I'll open the parser next and re-run the test") or a
QUESTION back to the user ("Want me to look at the service log?") asserts
nothing about the world, so no evidence can contradict it. The cheap judge
refutes over exactly these: after retiring the GEPA adjudicate artifact, 4
of the 6 residual false-refutes on the honest-failure FP-trap set were
future-plan complaints, and prompt text does not fix it (measured p=1.0 —
the E4B judge reads the rule and does not follow it).

The dismissal is PROOF-BASED, in this module's tradition: the fragment must
appear VERBATIM in the claim, and the guards (no digits, no result verbs)
are what stop an assertion from riding inside an intention.
"""

import sys
import os

import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import objection as ob


CLAIM = ("The test run FAILED: `argparse.ArgumentError: unrecognized arguments: "
         "--url`, exit code 2. No output was produced. I'll open the parser "
         "next and re-run the test.")
EVIDENCE = ("[execute] Traceback (most recent call last):\n"
            "argparse.ArgumentError: unrecognized arguments: --url\nEXIT CODE: 2")

QUESTION_CLAIM = ("The restart FAILED — the service exited immediately with "
                  "status 78 and the port is not listening. Want me to look at "
                  "the service log?")
QUESTION_EV = "[manage_services] restart: pid exited after 0.4s, status=78."


def _d(issue, claim=CLAIM, evidence=EVIDENCE):
    return ob.resolve_issue(issue, claim, evidence)[0]


class TestDismissesNonAssertive:
    def test_bare_quote_of_a_next_step(self):
        assert _d("I'll open the parser next and re-run the test.") == ob.DISMISS

    def test_quoted_next_step_inside_a_complaint(self):
        assert _d("The statement 'I'll open the parser next and re-run the "
                  "test' is a future plan.") == ob.DISMISS

    def test_parenthetical_quote(self):
        assert _d("The agent stated a future plan ('I'll open the parser "
                  "next').") == ob.DISMISS

    def test_alignment_vocabulary_escalates(self):
        """v3 (review N4): `requested/required/asked/summary/outcome/result/
        alignment/constraint/task/test` were REMOVED from the framing
        allowlist — they are the vocabulary of a genuine alignment or
        constraint complaint ("the reply is 'Let me check the log', not what
        the user asked"), and 10 of 10 such complaints laundered while they
        were listed. The cost is that some innocent framings now escalate;
        that is the safe direction."""
        assert _d("The agent stated a future plan ('I'll open the parser "
                  "next') not requested by the user.") != ob.DISMISS
        assert _d("The statement 'I'll open the parser next' is a future "
                  "plan, not a summary of the test outcome.") != ob.DISMISS
        # the shape the removal exists for
        assert ob.resolve_issue(
            "The reply is 'Let me check the service log for you', not what "
            "the user asked.",
            "Let me check the service log for you.",
            "[web] Iceland population 396,960.")[0] != ob.DISMISS

    def test_question_back_to_the_user(self):
        assert _d("Want me to look at the service log?",
                  QUESTION_CLAIM, QUESTION_EV) == ob.DISMISS

    def test_double_quote_delimiters(self):
        assert _d("The reply ends with 'I'll open the parser next'."
                  ) == ob.DISMISS

    def test_backtick_delimiters(self):
        assert _d("The reply states `I'll open the parser next` as its next "
                  "step.") == ob.DISMISS

    def test_possessive_before_the_quote_still_matches(self):
        """Review m9: a possessive apostrophe used to eat the opening
        delimiter, so the rule silently missed the fragment."""
        assert _d("The reply's statement 'I'll open the parser next' is a "
                  "future plan.") == ob.DISMISS

    def test_content_complaint_wording_escalates_even_with_a_quoted_plan(self):
        """v2 contract (review C1/C3): the objection must reduce ENTIRELY to
        the fragment. Words like 'evidence'/'support' are deliberately NOT
        framing vocabulary, so an objection that also complains about
        support keeps escalating — the cost is one main-model call, and the
        alternative is laundering a real complaint."""
        assert _d("The reply ends with 'I'll open the parser next'."
                  ) == ob.DISMISS
        assert _d("'I'll open the parser next' is not corroborated by the "
                  "evidence.") != ob.DISMISS

    def test_contraction_does_not_split_the_span(self):
        """The apostrophe in "I'll" must not terminate a single-quoted span —
        a naive [^']+ captured "ll open the parser next", which no longer
        matches the first-person intent pattern, so the rule silently missed
        its own headline case (found by probe, not by test)."""
        spans = [next(g for g in m.groups() if g) for m in
                 ob._QUOTED_SPAN_RE.finditer(
                     "The statement 'I'll open the parser next' is a plan.")]
        assert spans == ["I'll open the parser next"]


class TestRefusesToDismiss:
    """The guards. Each of these must keep escalating — a dismissal here
    would launder a real defect past the main model with no call."""

    def test_plain_assertion_is_not_dismissed(self):
        assert _d("No output was produced.") != ob.DISMISS

    def test_result_claim_inside_an_intention_is_not_dismissed(self):
        claim = CLAIM + " I'll ship the fix that passed review."
        assert _d("I'll ship the fix that passed review.", claim) != ob.DISMISS

    def test_digits_in_the_fragment_veto_dismissal(self):
        claim = CLAIM + " I'll retry all 8 exits."
        assert _d("I'll retry all 8 exits.", claim) != ob.DISMISS

    def test_past_tense_first_person_is_not_a_next_step(self):
        claim = "I fixed the parser and re-ran the test."
        assert _d("I fixed the parser and re-ran the test.", claim) != ob.DISMISS

    def test_paraphrase_without_a_verbatim_fragment_is_not_dismissed(self):
        # The judge's own words are not evidence; only a fragment found in
        # the claim can be checked.
        assert _d("The claim promises a fix that the evidence does not verify."
                  ) != ob.DISMISS

    def test_fragment_absent_from_the_claim_is_not_dismissed(self):
        assert _d("The reply says 'I'll delete the database next' which is "
                  "alarming.") != ob.DISMISS

    def test_short_fragment_is_not_trusted(self):
        assert ob._nonassertive_fragment("I'll go", "I'll go") is None

    def test_empty_inputs_are_safe(self):
        assert ob._nonassertive_fragment("", CLAIM) is None
        assert ob._nonassertive_fragment("I'll open the parser next", "") is None


class TestWholeVerdictStillNeedsUnanimity:
    def test_a_real_defect_beside_a_next_step_still_escalates(self):
        """resolve_refute drops a refute only when EVERY objection is
        dismissed — the per-objection rule must not shortcut that."""
        decision, _reasons, unresolved = ob.resolve_refute(
            ["I'll open the parser next and re-run the test.",
             "The claim states the run produced output when it did not."],
            CLAIM, EVIDENCE)
        assert decision is None
        assert len(unresolved) == 1

    def test_all_next_steps_dismisses_the_whole_refute(self):
        decision, reasons, unresolved = ob.resolve_refute(
            ["I'll open the parser next and re-run the test.",
             "The statement 'I'll open the parser next' is a future plan."],
            CLAIM, EVIDENCE)
        assert decision == ob.DISMISS
        assert not unresolved and len(reasons) == 2


class TestUpholdOnlyExemption:
    """Rule 4 ships EXEMPT from the uphold-only default (the general DISMISS
    direction has been off since 2026-08-07: 3 rescues / 9 damage). Without
    the exemption the rule would be a silent inoperative subsystem — it
    would compute a dismissal that the caller then discards."""

    def test_general_dismiss_is_still_off_by_default(self):
        assert ob.dismiss_enabled() is False

    def test_ships_default_OFF(self):
        """v3 decision (adversarial review, 2026-08-12): default-OFF. The
        guard is a lexical proxy for a semantic property ("this fragment
        asserts nothing"), two rounds of tightening each fell to shapes the
        author had not modelled, and the failure mode is SILENT — a
        laundered fabrication looks exactly like a correct rescue in the
        ledger. A false negative costs one main-model call on a turn that
        was making one anyway; a false positive erases a real catch. That
        asymmetry does not justify default-ON for 2 measured rescues."""
        assert ob.nonassertive_enabled() is False

    def test_opt_in_switch(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_OBJECTION_NONASSERTIVE", "1")
        assert ob.nonassertive_enabled() is True

    def test_rule4_only_dismissal_is_flagged_exempt(self):
        _d, reasons, _u = ob.resolve_refute(
            ["I'll open the parser next and re-run the test."],
            CLAIM, EVIDENCE)
        assert ob.nonassertive_dismissal(reasons) is True

    def test_mixed_with_a_factual_dismissal_is_NOT_exempt(self):
        """Unanimity: one factual dismissal in the mix and the whole verdict
        goes back through the gate the 2026-08-07 measurement closed."""
        factual = ("The claim mentions 'unrecognized arguments' which is not "
                   "in the evidence.")
        ev = ("[execute] argparse error: unrecognized arguments --url. "
              "EXIT CODE: 2")
        decision, reasons, _u = ob.resolve_refute(
            ["I'll open the parser next and re-run the test.", factual],
            CLAIM, ev)
        assert decision == ob.DISMISS          # both objections dismissed…
        assert len(reasons) == 2
        assert ob.nonassertive_dismissal(reasons) is False   # …but not exempt

    def test_empty_reasons_are_not_exempt(self):
        assert ob.nonassertive_dismissal([]) is False
        assert ob.nonassertive_dismissal(None) is False


class TestWiredEndToEnd:
    """The rule must reach production, not just compute a decision the
    caller discards — the bench run that shipped alongside this recorded
    `objection dismissed: 0` because the judge happened not to refute a
    clean, so the end-to-end path needs its own deterministic pin."""

    class _CountingClient:
        """Counts main-model calls. NOT a raising stub: the verifier catches
        exceptions from the LLM call and fails open, so a raise is swallowed
        and proves nothing — the count is the honest instrument."""

        critic_clients = [object()]      # a cheap route exists
        worker_clients = None

        def __init__(self):
            self.calls = 0

        async def chat_completion(self, *a, **k):
            self.calls += 1
            return {"choices": [{"message": {"content": ""}}]}

    @staticmethod
    def _cheap_refute():
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        return VerifyResult(
            verdict=VerifyVerdict.REFUTED, confidence=0.9,
            reasoning="the reply states a future plan",
            issues=["I'll open the parser next and re-run the test."])

    async def test_next_step_refute_is_dismissed_without_a_main_call(
            self, monkeypatch):
        from ghost_agent.core.verifier import Verifier, VerifyVerdict
        monkeypatch.setenv("GHOST_VERIFY_OBJECTION_NONASSERTIVE", "1")
        client = self._CountingClient()
        out = await Verifier(llm_client=client)._escalate_refute(
            self._cheap_refute(), CLAIM, EVIDENCE, "please test the pipeline.")
        assert client.calls == 0, "the main model was called — no short-circuit"
        # CONFIRMED at the judge's OWN confidence, never lifted.
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.confidence == 0.9
        assert getattr(out, "objection_dismissed", False) is True

    async def test_default_off_means_the_escalation_still_runs(self, monkeypatch):
        from ghost_agent.core.verifier import Verifier
        monkeypatch.delenv("GHOST_VERIFY_OBJECTION_NONASSERTIVE", raising=False)
        client = self._CountingClient()
        await Verifier(llm_client=client)._escalate_refute(
            self._cheap_refute(), CLAIM, EVIDENCE, "please test the pipeline.")
        assert client.calls >= 1, ("shipping default-OFF: the dismissal must "
                                   "be discarded and the escalation must run")
