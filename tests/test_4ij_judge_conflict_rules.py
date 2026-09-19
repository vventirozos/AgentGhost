"""§4IJ — the judge's conflicting-evidence and internal-contradiction rules.

Two live failures, one blind spot. (1) Bench fault `omitted_contradiction`
(§4IA): the evidence states two different values for one quantity and the
reply reports one of them — the judge CONFIRMED 23/23 at mean 0.96, because
its "support" rule read "REAL only if the fact appears in NO tool output or
contradicts one", and a fact that IS in the output passes that test even
when the same output also says otherwise. (2) Probe ifs21382… (2026-09-17):
the reply stated "~10 km nominal resolution" and "the nearest grid point is
~86 km away", both numbers present in its own script's output (longitudes
in radians labelled degrees), and the judge CONFIRMED at 0.90.

Property: **a fact appearing in a tool output does not rescue a claim when
the same output also contradicts it, and a claim whose own statements
cannot both be true is refuted whatever the output shows.** Both rules are
in every judge stage; the deciding stage's copies are pinned so a tuned
template that sheds them is rejected; the live resolver serves them.

The catch rate itself is measured by the bench (`scripts/verify_bench.py`,
faults clean + omitted_contradiction, arm A/B recorded in the journal) —
a prompt's effect on a model is not a unit-testable property.
"""
import pytest

from ghost_agent.core import verifier as V

CONFLICT = "CONFLICTING EVIDENCE is a REAL problem"
INTERNAL = "INTERNAL CONTRADICTION is a REAL problem"


@pytest.mark.parametrize("name", ["_VERIFY_CLAIM_PROMPT", "_VERIFY_ADJUDICATE_PROMPT"])
@pytest.mark.parametrize("rule", [CONFLICT, INTERNAL])
def test_every_deciding_stage_carries_both_rules(name, rule):
    assert rule in getattr(V, name)


def test_the_naming_stage_asks_for_both_shapes_as_support_suspects():
    """Stage 1 only NAMES suspects, so it carries the shapes, not the verdict
    rules: the two-values wording under "support", and the cannot-both-be-true
    sentence."""
    t = V._VERIFY_ENUMERATE_PROMPT
    assert "states two different values for that same quantity" in t
    assert "cannot both be true" in t
    assert CONFLICT not in t and INTERNAL not in t


def test_the_rules_say_what_is_not_a_conflict():
    """The FPR guard: the same rule that refutes two values for ONE quantity
    must say that different quantities are not a conflict, or the clean
    arm's load averages / price lists / hourly readings become refutes."""
    for name in ("_VERIFY_CLAIM_PROMPT", "_VERIFY_ADJUDICATE_PROMPT"):
        t = getattr(V, name)
        assert "Different quantities are NOT a conflict" in t
        assert "load averages" in t and "different items" in t and "different times" in t


@pytest.mark.parametrize("rule", [CONFLICT, INTERNAL])
def test_both_rules_are_pinned_so_a_tuned_template_cannot_shed_them(rule):
    stage, baseline = "verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT
    assert rule in V._REQUIRED_RULE_MARKERS[stage]
    assert V._validate_stage_template(stage, baseline) is True
    shed = baseline.replace(rule, rule.replace("REAL", "possible"))
    assert V._template_reject_reason(stage, shed).startswith("pinned rule missing")
    assert rule in V._stage_template(stage, baseline)          # the live resolver serves it


def test_the_extra_problems_paragraph_names_both_shapes():
    """The stage-2 "starting point, not a boundary" paragraph is where the
    judge is told what to look for beyond the suspects; both shapes are
    listed there so a stage-1 miss does not end the search."""
    t = V._VERIFY_ADJUDICATE_PROMPT
    i = t.index("The SUSPECTS list is a starting point, not a boundary")
    para = t[i:t.index("\n", i)]
    assert "a quantity the evidence states two ways while the CLAIM states one" in para
    assert "two CLAIM statements that cannot both be true" in para


def test_templates_still_format_probe_cleanly():
    for name, t in (("verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT),
                    ("verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT)):
        assert V._template_reject_reason(name, t) == ""
    assert V._VERIFY_CLAIM_PROMPT.format(claim="c", evidence="e", context="x")
