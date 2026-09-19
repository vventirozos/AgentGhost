"""§4IJ (part 2) — the strong judge sees what the cheap judge found.

Bench arm A (2026-09-17, 237 trials): the cheap judge refuted fabrication
27/29, fact_swap 16/23, omitted_contradiction 10/23 — and the refute
escalation on the main model overturned 22, 15 and 9 of them (94
overturns: 15 rescues of clean cases, 79 damage). Arm B added the
conflicting-evidence rules: the cheap judge then named the conflict
precisely ("Temperature conflict between 34°C and 35°C"; "the evidence
shows two different sizes for it") and the escalation still overturned —
it was asked the classic claim prompt FROM SCRATCH and never saw the
finding.

Now the escalation's classic prompt carries a PRIOR AUDIT block with the
cheap judge's itemized objections, framed as claims to check (no quote
burden — that variant lost on FPR; no forced suspects — that variant
anchored the strong model into refuting clean replies). Flag
GHOST_VERIFY_ESCALATION_OBJECTIONS — DEFAULT OFF since the mined-pool
arm D' (2026-09-18): on live-derived replies it doubled clean refutes
(gloss and truncation objections upheld); the bench records it in
provenance.

World where each pin fails: the block is not inserted, lands after the
checks, loses the objections, breaks the claim prompt's own placeholders,
the escalation site stops using it (or uses it with the flag off), or the
flag goes invisible to the bench.
"""
import ast
import inspect

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.eval import verify_bench as B

ISSUES = ["The claim states the largest file is manage.py at 18 KB, but the evidence shows two different sizes for it.",
          "The count 9,592 is inconsistent with {braces} in the row."]


def test_block_is_inserted_before_the_checks_and_keeps_the_placeholders():
    t = V.claim_prompt_with_objections(ISSUES, "r")
    out = t.format(claim="THE-CLAIM", evidence="THE-EVIDENCE", context="THE-ASK")
    assert out.index("PRIOR AUDIT") < out.index("Check, in order:")
    assert out.index("USER REQUEST") < out.index("PRIOR AUDIT")
    assert "1. The claim states the largest file is manage.py" in out
    assert "2. The count 9,592 is inconsistent with {braces} in the row." in out
    assert "THE-CLAIM" in out and "THE-EVIDENCE" in out and "THE-ASK" in out
    # everything of the classic prompt survives around the block
    classic = V._VERIFY_CLAIM_PROMPT.format(claim="THE-CLAIM", evidence="THE-EVIDENCE", context="THE-ASK")
    head, tail = classic.split("Check, in order:", 1)
    assert out.startswith(head) and out.endswith("Check, in order:" + tail)


def test_block_frames_objections_as_claims_to_check_not_verdicts():
    t = V._ESCALATION_OBJECTIONS_BLOCK
    assert "every objection may be a FALSE ALARM" in t
    assert "do not take it on trust and do not dismiss it unread" in t
    assert "two different values for that quantity" in t
    assert "keep the objections that hold" in t


def test_reasoning_stands_in_when_the_refute_has_no_itemized_issues():
    t = V.claim_prompt_with_objections([], "the numbers disagree")
    out = t.format(claim="c", evidence="e", context="x")
    assert "1. (no itemized issues; the auditor's reasoning:) the numbers disagree" in out
    t2 = V.claim_prompt_with_objections(["  ", None], "why")
    assert "the auditor's reasoning:) why" in t2.format(claim="c", evidence="e", context="x")


def test_escalation_site_uses_the_block_under_the_flag():
    """AST of `_escalate_refute_impl`: the branch that formats the classic
    prompt picks `claim_prompt_with_objections(result.issues, …)` when
    `_escalation_objections_enabled()` and `_VERIFY_CLAIM_PROMPT` otherwise,
    and formats THAT template."""
    tree = ast.parse(inspect.getsource(V))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_escalate_refute_impl")
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "_escalation_objections_enabled()"]
    assert len(ifs) == 1
    g = ifs[0]
    body = "\n".join(ast.unparse(s) for s in g.body)
    orelse = "\n".join(ast.unparse(s) for s in g.orelse)
    assert "template = claim_prompt_with_objections(result.issues" in body
    assert "template = _VERIFY_CLAIM_PROMPT" in orelse
    # the formatted prompt is built from `template`, not from the constant
    fmt = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "format"
           and getattr(n.func.value, "id", "") == "template"]
    assert len(fmt) == 1
    stale = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "format"
             and getattr(n.func.value, "id", "") == "_VERIFY_CLAIM_PROMPT"]
    assert stale == [], "the escalation still formats the bare classic prompt somewhere"


@pytest.mark.parametrize("value,expected", [("1", True), ("", False), ("0", False), ("off", False), ("yes", True)])
def test_flag_parsing(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_VERIFY_ESCALATION_OBJECTIONS", raising=False)
    else:
        monkeypatch.setenv("GHOST_VERIFY_ESCALATION_OBJECTIONS", value)
    assert V._escalation_objections_enabled() is expected


def test_bench_provenance_records_the_flag():
    """The §4BF invisible-flag class: a flag that changes the strong judge's
    prompt must be in the bench's verify_flags list."""
    src = inspect.getsource(B.bench_provenance)
    tree = ast.parse(src)
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "GHOST_VERIFY_ESCALATION_OBJECTIONS" in consts
    assert "GHOST_VERIFY_OVERTURN_QUOTE" in consts       # the sibling it sits beside
