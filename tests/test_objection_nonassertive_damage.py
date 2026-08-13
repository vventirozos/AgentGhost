"""Rule 4 DAMAGE suite — replies that MUST keep escalating (§4BD-b, v2).

Why this file exists, and why it is the real measurement. The first
justification for shipping rule 4 default-ON was an LLM probe that scored
"0 damage over 31 corrupted trials". An adversarial review showed that
number was **structurally incapable of being nonzero**: the bench's
`fault_fabrication` strings carry digits and result verbs (vetoed before
rule 4 can act) and `fault_silent_failure` keeps the claim and swaps in a
tool error, so its objections are absence-shaped and settle at rule 2. The
shape that actually threatens this rule — a fabricated assertion sharing a
judge-quoted span, or riding beside a next step — is one no fault injector
produces.

So the damage surface is measured HERE instead, deterministically, on
hand-built adversarial fixtures. Every case below is a genuinely defective
reply plus the objection a judge would raise about it. **A DISMISS on any
of them is a laundering bug**: the refute would be erased with no
main-model call. Three of these are the CRITICALs the review reproduced
end-to-end against v1.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import objection as ob


# (label, claim, evidence, objection)
DAMAGE_CASES = [
    (
        "C1: real contradiction cited BESIDE a quoted next step",
        "The test run completed with exit code 0 and all checks green. "
        "I'll open that parser next and re-run the coverage job.",
        "[execute] argparse.ArgumentError  EXIT CODE: 2",
        "The claim asserts exit code 0 while the evidence shows exit code 2; "
        "it only offers 'I'll open that parser next and re-run the coverage "
        "job' as support.",
    ),
    (
        "C2a: fabricated predicates in the same span as the intention",
        "The endpoint is reachable and the config is valid. I'll add the "
        "retry logic next.",
        "[execute] curl: (7) Failed to connect to api.internal port 443",
        "The endpoint is reachable and the config is valid. I'll add the "
        "retry logic next.",
    ),
    (
        "C2b: false 'healthy' conclusion beside a stated plan",
        "Nothing in the logs mentions the outage, so the service is healthy. "
        "I'll keep watching it.",
        "[manage_services] status: pid exited, status=78, port 8080 dead.",
        "Nothing in the logs mentions the outage, so the service is healthy. "
        "I'll keep watching it.",
    ),
    (
        "C2c: unvetoed predicative verbs (exists / matches)",
        "The migration script exists and the schema matches production. "
        "Let me apply it to staging.",
        "[file_system] ls migrations/: No such file or directory",
        "The migration script exists and the schema matches production. "
        "Let me apply it to staging.",
    ),
    (
        "C2d: unvetoed 'found' / 'was' assertions",
        "I found the root cause: the token was never rotated and the key "
        "file is empty. I'll patch the rotation job next.",
        "[execute] grep: no matches for 'rotate' in auth.log",
        "I found the root cause: the token was never rotated and the key "
        "file is empty. I'll patch the rotation job next.",
    ),
    (
        "C3a: absence synonym missing from _ABSENCE_RE ('corroborated')",
        "The cache was cleared successfully. I'll re-run the import next.",
        "[execute] redis-cli FLUSHALL -> (error) NOAUTH Authentication required.",
        "'I'll re-run the import next' is not corroborated by the evidence.",
    ),
    (
        "C3b: absence synonym 'backed by'",
        "The cache was cleared successfully. I'll re-run the import next.",
        "[execute] redis-cli FLUSHALL -> (error) NOAUTH Authentication required.",
        "'I'll re-run the import next' is not backed by the evidence.",
    ),
    (
        "C3c: 'has no support in'",
        "The cache was cleared successfully. I'll re-run the import next.",
        "[execute] redis-cli FLUSHALL -> (error) NOAUTH Authentication required.",
        "'I'll re-run the import next' has no support in the evidence.",
    ),
    (
        "M6: a question ENDING a paragraph of assertions",
        "The old table is empty and unused, and no service reads from it. "
        "Should I drop it?",
        "[database] SELECT count(*) FROM orders_v1 -> 48211 rows; "
        "2 grants to reporting_svc.",
        "The old table is empty and unused, and no service reads from it. "
        "Should I drop it?",
    ),
    (
        "M5: a result claim wrapped inside the intention itself",
        "I'll ship the fix that passed review.",
        "[execute] pytest -> 3 failed, 0 passed. EXIT CODE: 1",
        "I'll ship the fix that passed review.",
    ),
    (
        "multi-sentence span: assertion AFTER the intention",
        "I'll re-run the job next. The database is already fully migrated.",
        "[database] migration 004 pending; 3 of 7 applied.",
        "I'll re-run the job next. The database is already fully migrated.",
    ),
    (
        "paraphrase with no verbatim fragment must never dismiss",
        "The deploy failed. I'll roll back next.",
        "[execute] deploy: FAILED (exit 1)",
        "The claim promises a rollback the evidence cannot verify, and "
        "misstates the failing stage.",
    ),
]


@pytest.mark.parametrize("label,claim,evidence,issue",
                         DAMAGE_CASES,
                         ids=[c[0].split(":")[0] for c in DAMAGE_CASES])
def test_defective_reply_is_never_dismissed(label, claim, evidence, issue):
    decision, _why = ob.resolve_issue(issue, claim, evidence)
    assert decision != ob.DISMISS, (
        f"LAUNDERING: rule 4 dismissed a defective reply — {label}")


@pytest.mark.parametrize("label,claim,evidence,issue",
                         DAMAGE_CASES,
                         ids=[c[0].split(":")[0] for c in DAMAGE_CASES])
def test_whole_verdict_is_never_dismissed(label, claim, evidence, issue):
    """The verdict-level path is what actually suppresses a refute."""
    decision, reasons, _u = ob.resolve_refute([issue], claim, evidence)
    exempt = decision == ob.DISMISS and ob.nonassertive_dismissal(reasons)
    assert not exempt, (
        f"LAUNDERING at verdict level — {label}")


# Mutations that PRESERVE the defect. A suite that only pins the literal
# strings that motivated it does not pin the mechanism: the reviewer
# reworded each fixture minimally and 8 of 9 laundered against v2. Every
# mutant of a defective reply must still escalate.
def _mutate(text):
    yield text.replace(". ", "; ", 1)          # clause break instead of stop
    yield text.replace(". ", ": ", 1)
    yield text.replace(". ", ", and ", 1)
    yield text.replace(". ", "\n", 1)
    yield text.replace("passed", "passes").replace("exists", "exist")
    yield text.replace(" and ", " which means ", 1)
    yield text.replace("I'll", "Let me", 1)
    yield text.replace("I'll", "Next I'll", 1)


@pytest.mark.parametrize("label,claim,evidence,issue", DAMAGE_CASES,
                         ids=[c[0].split(":")[0] for c in DAMAGE_CASES])
def test_mutations_of_defective_replies_never_dismiss(label, claim,
                                                      evidence, issue):
    for m_issue in _mutate(issue):
        if m_issue == issue:
            continue
        decision, _why = ob.resolve_issue(m_issue, claim, evidence)
        assert decision != ob.DISMISS, (
            f"LAUNDERING via mutation — {label}\n  mutant: {m_issue!r}")
        # and the same claim mutated alongside it (the judge quotes what
        # the reply actually said)
        for m_claim in _mutate(claim):
            d2, _ = ob.resolve_issue(m_issue, m_claim, evidence)
            assert d2 != ob.DISMISS, (
                f"LAUNDERING via claim+issue mutation — {label}\n"
                f"  mutant: {m_issue!r}")


def test_damage_suite_is_non_vacuous():
    """A suite of cases that can never dismiss ANYTHING proves nothing.
    The rule must still fire on the shapes it exists for, through the same
    entry point — otherwise every assertion above is green because the rule
    is dead, not because the guards bind. These three are the objection
    shapes observed live on 2026-08-12."""
    claim = ("The test run FAILED: exit code 2. No output was produced. "
             "I'll open the parser next and re-run the test.")
    ev = "[execute] argparse error. EXIT CODE: 2"
    assert ob.resolve_issue(
        "I'll open the parser next and re-run the test.", claim, ev
    )[0] == ob.DISMISS
    assert ob.resolve_issue(
        "The statement 'I'll open the parser next and re-run the test' is a "
        "future plan.", claim, ev)[0] == ob.DISMISS
    q_claim = ("The restart FAILED — the service exited with status 78. "
               "Want me to look at the service log?")
    assert ob.resolve_issue(
        "Want me to look at the service log?", q_claim,
        "[manage_services] status=78, port dead.")[0] == ob.DISMISS
