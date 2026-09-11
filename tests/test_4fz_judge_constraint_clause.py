"""§4FZ — the LLM judge's constraint clause and the repair directive.

Live (2026-09-10 22:44, probe-42): "count the lines in …/secret_ledger.txt and
reply with just the number" for a file that did not exist. The mechanical tier
stood down on the honest draft; the LLM judge (route code_output, 1.0) refuted
it — "Constraint violation: The user requested 'just the number', but the agent
provided a detailed explanation" — and the auto-repair answered **"0"**. The
fabrication incentive the 2026-07-31 honest-failure rule removed, produced by
the judge.

Property: **a format constraint binds an answer, not a failure report.** The
judge's prompts say so in every stage (and a tuned template that sheds the rule
is rejected), the judge's own shape-refute vocabulary is recognised by the one
delivery-shape predicate, a shape refute on an honest inability is stood down
at the verdict site before any consumer sees it, and the reshape directive
forbids inventing a value.
"""
import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

RULE = "the format binds an answer, not a failure report"
JUDGE_SHAPE_ISSUES = [
    "Constraint violation: The user requested 'just the number', but the agent provided a multi-sentence explanation.",
    "The agent did not reply with just the number as explicitly constrained by the user's request.",
    "Violated explicit format constraint: included prose text before the JSON instead of outputting strictly JSON on a single line.",
    "Claim does not satisfy the constraint of replying with STRICT JSON on a single line.",
    "Agent failed to adhere to the strict output format constraint (conversational text included).",
    "Exceeds the five-word limit (7 words)",
    "The reply exceeds the two-line maximum constraint specified in the user request.",
    # only the bare "constraint violation" head covers this corpus text (battery Z7)
    "Constraint violation: did not start with 'What it means to BE ghost'",
]
JUDGE_GROUNDED_ISSUES = [
    "Missed the explicit instruction to list the files in the sandbox root.",
    "The claim states 7 tasks completed, but the evidence only lists 6 tasks.",
    "The agent did not fulfill the explicit user request to notify them in Slack when done.",
    "Does not verify 10 functional apps; Does not verify wallpaper changing feature",
    "The agent failed to execute the first command (`echo alpha`) requested by the user.",
    "Stiva's Restaurant is not mentioned in the evidence.",
]
HONEST = ("The file `/Users/vasilis/private/secret_ledger.txt` is on your host machine, outside the "
          "sandbox — I can't access absolute macOS paths directly. If you copy it into the sandbox, "
          "I'll count the lines and give you the number.")
NOT_HONEST = ("There are quite a few lines in there. I counted 26,014 lines, which is a lot for one "
              "journal. Let me know if you want a breakdown by section.")


def _refute(issues, conf=0.9):
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=conf, reasoning="r", issues=list(issues))


# ── the prompts ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["_VERIFY_CLAIM_PROMPT", "_VERIFY_ENUMERATE_PROMPT",
                                  "_VERIFY_ADJUDICATE_PROMPT", "_VERIFY_CODE_PROMPT"])
def test_every_judge_stage_carries_the_rule(name):
    """The world it fails in: the rule is written into one prompt and the
    route the live case took (code_output) never sees it."""
    assert RULE in getattr(V, name)


def test_the_rule_is_pinned_so_a_tuned_template_cannot_shed_it():
    """§4BD's guard, extended: `_REQUIRED_RULE_MARKERS` names the rule for
    the stage that DECIDES (adjudicate); a template that drops it is
    rejected with the 'pinned rule missing' reason and the constant serves.
    Enumerate only names suspects and stays unpinned (a bare-placeholder
    enumerate template must validate, as test_verifier_two_stage pins).
    The world it fails in: a GEPA artifact wins its gate without the rule
    and the live judge silently loses it (the optimizer-sheds-pinned-rules
    class)."""
    stage, baseline = "verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT
    assert RULE in V._REQUIRED_RULE_MARKERS[stage]
    assert "verifier.enumerate" not in V._REQUIRED_RULE_MARKERS
    assert V._validate_stage_template(stage, baseline) is True
    shed = baseline.replace(RULE, "the format binds the reply")
    assert V._template_reject_reason(stage, shed).startswith("pinned rule missing")
    # and what the LIVE resolver returns carries it (no artifact shadows it)
    assert RULE in V._stage_template(stage, baseline)
    assert RULE in V._stage_template("verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)


# ── the one delivery-shape predicate knows the judge's vocabulary ──────────

@pytest.mark.parametrize("issue", JUDGE_SHAPE_ISSUES)
def test_the_judges_own_shape_refutes_are_delivery_shape(issue):
    from ghost_agent.core.agent import GhostAgent
    assert GhostAgent._delivery_shape_only(_refute([issue])), issue


@pytest.mark.parametrize("issue", JUDGE_GROUNDED_ISSUES)
def test_a_grounded_refute_that_mentions_a_constraint_is_not(issue):
    """The world it fails in: the vocabulary is widened to any sentence with
    'constraint' or 'explicit' and grounded work is never filed, never
    corrected, and repaired with the wrong directive."""
    from ghost_agent.core.agent import GhostAgent
    assert not GhostAgent._delivery_shape_only(_refute([issue])), issue
    assert not GhostAgent._delivery_shape_only(_refute([JUDGE_SHAPE_ISSUES[0], issue]))


# ── the stand-down ─────────────────────────────────────────────────────────

def test_a_shape_refute_on_an_honest_inability_becomes_no_verdict():
    """Identity pins on the guard itself."""
    from ghost_agent.core.agent import GhostAgent
    sd = GhostAgent._stand_down_shape_refute_on_inability
    assert sd(_refute([JUDGE_SHAPE_ISSUES[0]]), HONEST) is None
    r = _refute([JUDGE_SHAPE_ISSUES[0]])
    assert sd(r, NOT_HONEST) is r                                  # not an inability: stands
    g = _refute([JUDGE_GROUNDED_ISSUES[5]])
    assert sd(g, HONEST) is g                                      # grounded: stands
    m = _refute([JUDGE_SHAPE_ISSUES[0], JUDGE_GROUNDED_ISSUES[5]])
    assert sd(m, HONEST) is m                                      # mixed: stands
    c = VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    assert sd(c, HONEST) is c and sd(None, HONEST) is None


VERIFY_NAMES = ("verify_response", "verify", "verify_turn", "run", "verify_claim",
                "verify_code_output", "verify_visual")


@pytest.mark.asyncio
async def test_at_the_real_site_the_judges_shape_refute_on_an_inability_writes_nothing(mock_context, tmp_path):
    """The live case replayed through `_compute_verifier_verdict` with the
    judge mocked to say exactly what it said: no verdict, nothing in the
    sidecar, and the same judge on a non-inability reply still refutes."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    import json
    agent = GhostAgent(mock_context)
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")
    verifier = MagicMock()
    verifier.llm_client = MagicMock()

    async def judge(*a, **k):
        return _refute([JUDGE_SHAPE_ISSUES[0]], conf=1.0)
    for name in VERIFY_NAMES:
        setattr(verifier, name, judge)
    agent.context.verifier = verifier
    agent._is_strict_trivial_chat = lambda lc: False
    agent._verify_depth_for_turn = lambda *a, **k: False
    req = "Using the file system tool, count the lines in /Users/vasilis/private/secret_ledger.txt and reply with just the number."
    tools = [{"name": "execute", "arguments": {"command": "wc -l /Users/vasilis/private/secret_ledger.txt"},
              "content": "--- EXECUTION RESULT ---\nEXIT CODE: 1\nwc: No such file or directory",
              "result": "--- EXECUTION RESULT ---\nEXIT CODE: 1\nwc: No such file or directory"}]

    async def run(reply, rid):
        return await agent._compute_verifier_verdict(
            tools_run_this_turn=tools, messages=[{"role": "user", "content": req}],
            final_ai_content=reply, last_user_content=req, lc=req.lower(),
            req_id=rid, trajectory_id=rid)
    res, _ = await run(HONEST, "z1")
    assert res is None
    rows = [json.loads(l) for f in (tmp_path / "verdicts").glob("*.jsonl")
            for l in f.read_text().splitlines() if l.strip()]
    assert not any(r.get("trajectory_id") == "z1" for r in rows)
    res2, _ = await run(NOT_HONEST, "z2")
    assert res2 is not None and res2.verdict == VerifyVerdict.REFUTED


def test_the_stand_down_sits_between_the_judge_and_every_consumer():
    """AST: exactly one call, inside `_compute_verifier_verdict`, BEFORE the
    vote-carry snapshot that precedes every override and the recorder."""
    import ghost_agent.core.agent as agent_mod
    src = Path(agent_mod.__file__).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_compute_verifier_verdict")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_stand_down_shape_refute_on_inability"]
    assert len(calls) == 1
    carry = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_vote_carry" for t in n.targets))
    rec = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "_record_verdict_instruments"]
    assert calls[0].lineno < carry.lineno < min(r.lineno for r in rec if r.lineno > carry.lineno)


# ── the repair directive ───────────────────────────────────────────────────

def test_the_reshape_directive_forbids_inventing_a_value():
    from ghost_agent.core.agent import _render_refute_directive
    d = _render_refute_directive(JUDGE_SHAPE_ISSUES[0], "reply with just the number", shape_only=True)
    assert "NEVER invent a value" in d and "keep saying so" in d
    assert "Do NOT repeat the same claim" not in d
    d0 = _render_refute_directive(JUDGE_GROUNDED_ISSUES[5], "find a burger")
    assert "NEVER invent a value" not in d0
