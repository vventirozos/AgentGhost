"""A mechanical refutation survives a judge — and an ESCALATION — that says
CONFIRMED (§4GJ, 2026-09-13).

Measured, not assumed. The judge readout owed since §4FN was run on
2026-09-13 over 6 weeks of live data:

    escalations            201 rows (2026-08-04 → 09-13)
    cheap REFUTE overturned by the main model   84
      …of which a human later labelled the turn 15
        human said PASSED (the overturn was RIGHT) 12
        human said FAILED (the overturn was WRONG)  3

So the high overturn rate is not, by itself, label corruption: where there is
ground truth the escalation is right four times in five, and the weak link is
the cheap judge (the `ppi-judge-too-weak` memory). Of the three WRONG
overturns, two are shapes a mechanical tier now catches — a finalize-fallback
dump (§4FN, req f41e4c6c) and a narration-only non-answer (§4GH, req
e57ad0cf) — and the third is a genuine quality disagreement over a long
synthesis, which no arithmetic rule should try to settle.

What protects those two is an ORDERING that nothing asserted: the judge (and
its escalation) runs first, and `_merge_mechanical_refute` is applied to the
result afterwards, REPLACING any verdict that is not already a refute. The
guarantee is therefore "a mechanical refute cannot be argued away", and it is
true by construction today — which is exactly the kind of invariant that a
later reordering silently breaks. These pins state it.

The world each pin fails in: any tree where the mechanical merge runs before
the judge, or where an escalated CONFIRMED is allowed to stand over a
mechanical REFUTED.
"""
import ast
import inspect
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import make_context

# The §4FN finalize-fallback dump: a reply that is not an answer by SHAPE.
DUMP_REPLY = ("Process finished successfully.\n\n### Final Output:\n```text\n"
              "### 1. Some search result\n[Source: https://example.org/a]\n```")
GOOD_REPLY = "The sender domain was a spoofed government address, confirmed by Reuters."

EVIDENCE_ROW = {
    "role": "tool", "tool_call_id": "c1", "name": "web_search",
    "content": ToolOutcome.ok(
        "### 1. Reuters — the request came from a spoofed government domain\n"
        "[Source: https://reuters.example/a]\n" + "detail " * 80,
        call_args={"query": "revolut breach"}),
}


def _agent_with_a_judge_that_confirms():
    """A verifier whose judge always returns CONFIRMED at high confidence —
    the state an escalation leaves behind when it overturns a cheap refute."""
    calls = []

    async def _confirm(*a, **k):
        calls.append(1)
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95,
                            reasoning="the main model says the answer is supported",
                            issues=[])
    ctx = make_context()
    ctx.verifier = SimpleNamespace(llm_client=object(), verify_claim=_confirm,
                                   verify_code_output=_confirm)
    return GhostAgent(ctx), calls


async def test_an_escalated_confirmed_cannot_stand_over_a_dump_shaped_reply():
    """req f41e4c6c (2026-08-14): the cheap judge refuted, the escalation
    overturned it to CONFIRMED, and the human later said FAILED. The reply
    was the finalize fallback pasted as the answer."""
    agent, judge_calls = _agent_with_a_judge_that_confirms()
    v, _lt = await agent._compute_verifier_verdict(
        tools_run_this_turn=[EVIDENCE_ROW], messages=[],
        final_ai_content=DUMP_REPLY, last_user_content="κάνε έρευνα σε παρακαλώ",
        lc="κάνε έρευνα σε παρακαλώ", req_id="r-dump", trajectory_id="t-dump")
    assert judge_calls, "the judge should still run — this is not the no-claim exit"
    assert v is not None and v.verdict == VerifyVerdict.REFUTED, v
    assert "reply-shape" in str(getattr(v, "override", "")), getattr(v, "override", "")


async def test_the_same_judge_verdict_stands_on_an_answer_shaped_reply():
    """Control: the two worlds must disagree only on the SHAPE. With a real
    answer the escalated CONFIRMED survives — the mechanical tier is silent,
    not a blanket veto."""
    agent, judge_calls = _agent_with_a_judge_that_confirms()
    v, _lt = await agent._compute_verifier_verdict(
        tools_run_this_turn=[EVIDENCE_ROW], messages=[],
        final_ai_content=GOOD_REPLY, last_user_content="who sent the request?",
        lc="who sent the request?", req_id="r-ok", trajectory_id="t-ok")
    assert judge_calls
    assert v is not None and v.verdict == VerifyVerdict.CONFIRMED, v


async def test_a_no_claim_reply_never_reaches_the_judge_at_all():
    """req e57ad0cf (2026-09-13), closed by §4GH: the escalation cannot
    overturn what it is never asked about."""
    agent, judge_calls = _agent_with_a_judge_that_confirms()
    narration = ("I have good coverage. Let me now dig into the specifics — the exact "
                 "email domain and the country.\n\n"
                 "Let me read the most detailed sources in parallel to extract the sender.")
    v, _lt = await agent._compute_verifier_verdict(
        tools_run_this_turn=[EVIDENCE_ROW], messages=[],
        final_ai_content=narration, last_user_content="identify the agency",
        lc="identify the agency", req_id="r-narr", trajectory_id="t-narr")
    assert judge_calls == []
    assert v is not None and v.verdict == VerifyVerdict.REFUTED


def test_the_merge_is_applied_after_the_judge_not_before():
    """The ORDERING the guarantee rests on, read from the AST rather than
    from a comment: inside `_compute_verifier_verdict`, every
    `_merge_mechanical_refute` call must come after every judge call
    (`verify_claim` / `verify_code_output`). A reordering that puts the
    merge first would let the judge's CONFIRMED overwrite it."""
    tree = ast.parse(inspect.getsource(agent_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_compute_verifier_verdict")
    judge_lines, merge_lines = [], []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        attr = getattr(node.func, "attr", "")
        if attr in ("verify_claim", "verify_code_output"):
            judge_lines.append(node.lineno)
        elif attr == "_merge_mechanical_refute":
            merge_lines.append(node.lineno)
    assert judge_lines, "no judge call found — re-point this pin"
    assert merge_lines, "no mechanical merge found — re-point this pin"
    # The no-claim early exit (§4GH) merges BEFORE the judge textually but
    # RETURNS, so it never reaches one; the tool-turn merge is the last word.
    assert max(merge_lines) > max(judge_lines), (merge_lines, judge_lines)


def test_the_merge_replaces_a_non_refute_verdict():
    """The unit the ordering depends on: a standing CONFIRMED is REPLACED,
    a standing refute keeps its own grounded issues first."""
    mech = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9,
                        reasoning="reply-shape check", issues=["not an answer"])
    confirmed = VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95,
                             reasoning="judge", issues=[])
    out = GhostAgent._merge_mechanical_refute(confirmed, mech, "reply-shape")
    assert out is mech and out.verdict == VerifyVerdict.REFUTED
    standing = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9,
                            reasoning="judge refute", issues=["grounded issue"])
    out2 = GhostAgent._merge_mechanical_refute(standing, mech, "reply-shape")
    assert out2.verdict == VerifyVerdict.REFUTED
    assert out2.issues[0] == "grounded issue"          # the standing refute leads
    assert GhostAgent._merge_mechanical_refute(confirmed, None, "reply-shape") is confirmed
