"""A cheap late REFUTE must not erase a strong CONFIRMED (2026-09-15, §4HB).

Live chain (req 14af0b6b, trajectory 97b402e8): the turn gate CONFIRMED at
0.92 and the strong model upheld it; sixty seconds later a late cheap-tier
pass REFUTED at 0.90 with escalation outcome "unavailable" (the strong model
never answered). The refute won by recency: outcome flipped ok→failed, the
lessons were scrubbed, the reflector wrote a lesson asserting the agent had
NOT said the thing it had said, and two reruns re-hunted an answered
question.

The rule is precedence by TIER, not recency, and deliberately narrow: only
a REFUTE whose escalation came back "unavailable", only when a
strong-adjudicated CONFIRMED already stands for that trajectory.
"""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core import verifier as V
from ghost_agent.core.verifier import (
    ESCALATION_NOT_ADJUDICATED,
    ESCALATION_OUTCOMES,
    ESCALATION_STRONG_ADJUDICATED,
    VerifyResult,
    VerifyVerdict,
)
# The ContextVar and the ledger writer are resolved through the MODULE at
# call time, never bound at import: three sibling files `importlib.reload`
# the verifier, which replaces its globals in place — an import-time
# binding then holds a ContextVar nothing writes to any more
# (memory: reload-contaminates-the-session).

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "core" / "verifier.py"


@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


def _v(verdict, conf=0.9, escalation="", issues=None):
    r = VerifyResult(verdict=verdict, confidence=conf, reasoning="r", issues=issues or [])
    r.escalation = escalation
    return r


# ---------------------------------------------------------------- the rule

def test_the_live_chain_is_withheld(agent):
    """FAILS IF: recency still wins.

    Strong-upheld CONFIRMED first, unescalated cheap REFUTED second — the
    exact sequence from the log. The refute must reach neither the outcome
    backfill nor the lesson scrub.
    """
    agent._note_strong_verdict("traj-97b4", _v(VerifyVerdict.CONFIRMED, 0.92, "upheld"))
    agent._backfill_trajectory_outcome = MagicMock()
    agent._record_withheld_verdict = MagicMock()
    agent.context.skill_memory = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_bg") as spawn:
        agent._record_late_verdict(
            _v(VerifyVerdict.REFUTED, 0.9, "unavailable", ["Sep 12 vs 11-12 September"]),
            trajectory_id="traj-97b4")
    agent._backfill_trajectory_outcome.assert_not_called()
    spawn.assert_not_called()                      # no lesson scrub
    agent._record_withheld_verdict.assert_called_once()   # kept as a measurement
    assert agent._record_withheld_verdict.call_args[0][1] == "failed"


def test_a_strong_adjudicated_refute_still_applies(agent):
    """FAILS IF: the rule blocks refutes the strong model actually upheld.

    The world where the fix over-reaches: escalation ran and the strong
    model AGREED with the refute. That verdict outranks the earlier one.
    """
    agent._note_strong_verdict("t2", _v(VerifyVerdict.CONFIRMED, 0.92, "upheld"))
    agent._backfill_trajectory_outcome = MagicMock()
    agent.context.skill_memory = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_bg"):
        agent._record_late_verdict(
            _v(VerifyVerdict.REFUTED, 0.9, "upheld", ["real defect"]),
            trajectory_id="t2")
    agent._backfill_trajectory_outcome.assert_called_once()
    assert agent._backfill_trajectory_outcome.call_args[0][1] == "failed"


def test_no_prior_strong_verdict_leaves_todays_behaviour(agent):
    """FAILS IF: the rule widens to every unescalated refute.

    Scoped on purpose: with nothing stronger on record, the cheap refute
    is still the only verdict there is, and it applies as before.
    """
    agent._backfill_trajectory_outcome = MagicMock()
    agent.context.skill_memory = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_bg"):
        agent._record_late_verdict(
            _v(VerifyVerdict.REFUTED, 0.9, "unavailable", ["x"]),
            trajectory_id="t3")
    agent._backfill_trajectory_outcome.assert_called_once()


def test_a_prior_strong_REFUTED_does_not_shield_a_late_refute(agent):
    """FAILS IF: the memo check compares the wrong verdict.

    A strong REFUTED on record and a cheap late REFUTED agree; there is
    nothing to protect and the late one applies.
    """
    agent._note_strong_verdict("t4", _v(VerifyVerdict.REFUTED, 0.9, "upheld"))
    agent._backfill_trajectory_outcome = MagicMock()
    agent.context.skill_memory = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_bg"):
        agent._record_late_verdict(
            _v(VerifyVerdict.REFUTED, 0.9, "unavailable", ["x"]),
            trajectory_id="t4")
    agent._backfill_trajectory_outcome.assert_called_once()


def test_a_late_CONFIRMED_is_never_withheld_by_this_rule(agent):
    """FAILS IF: the guard keys on escalation alone and not on the verdict."""
    agent._note_strong_verdict("t5", _v(VerifyVerdict.CONFIRMED, 0.92, "upheld"))
    agent._backfill_trajectory_outcome = MagicMock()
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_late_verdict(
            _v(VerifyVerdict.CONFIRMED, 0.9, "unavailable"), trajectory_id="t5")
    agent._backfill_trajectory_outcome.assert_called_once()
    assert agent._backfill_trajectory_outcome.call_args[0][1] == "passed"


# ---------------------------------------------------------------- the memo

def test_memo_records_only_strong_adjudicated_verdicts(agent):
    """FAILS IF: a cheap verdict can enter the memo and later shield itself."""
    agent._note_strong_verdict("a", _v(VerifyVerdict.CONFIRMED, 0.9, "unavailable"))
    agent._note_strong_verdict("b", _v(VerifyVerdict.CONFIRMED, 0.9, ""))
    agent._note_strong_verdict("c", _v(VerifyVerdict.CONFIRMED, 0.9, "downgraded"))
    assert agent._strong_verdict_for("a") == ""
    assert agent._strong_verdict_for("b") == ""
    assert agent._strong_verdict_for("c") == ""
    for good in ESCALATION_STRONG_ADJUDICATED - {"claim_binding"}:
        agent._note_strong_verdict("k-" + good, _v(VerifyVerdict.CONFIRMED, 0.9, good))
        assert agent._strong_verdict_for("k-" + good) == "CONFIRMED", good
    # §4IP R7 i2: "claim_binding" is strong for the REFUTED it validates, not
    # for a CONFIRMED (the absence of a contradiction, figures possibly unbound)
    agent._note_strong_verdict("k-cb-r", _v(VerifyVerdict.REFUTED, 0.9, "claim_binding"))
    agent._note_strong_verdict("k-cb-c", _v(VerifyVerdict.CONFIRMED, 0.9, "claim_binding"))
    assert agent._strong_verdict_for("k-cb-r") == "REFUTED"
    assert agent._strong_verdict_for("k-cb-c") == ""


def test_memo_is_bounded_and_evicts_the_oldest(agent):
    """FAILS IF: the memo grows without bound over a long-lived process."""
    cap = agent._STRONG_VERDICT_MEMO_MAX
    for i in range(cap + 10):
        agent._note_strong_verdict(f"t{i}", _v(VerifyVerdict.CONFIRMED, 0.9, "upheld"))
    assert len(agent._strong_verdict_memo) == cap
    assert agent._strong_verdict_for("t0") == ""
    assert agent._strong_verdict_for(f"t{cap + 9}") == "CONFIRMED"


def test_the_recorder_feeds_the_memo(agent):
    """Pin the CALL SITE, not the guard — FAILS IF: `_record_verdict_instruments`
    no longer notes the strong verdict. The memo is only ever written from
    there in production; a direct-call test cannot see that wiring.
    """
    agent.context.trajectory_collector = None      # sidecar returns early
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_verdict_instruments(
            _v(VerifyVerdict.CONFIRMED, 0.92, "upheld"),
            req_id="r1", trajectory_id="traj-wired", verify_route="claim")
    assert agent._strong_verdict_for("traj-wired") == "CONFIRMED"
    # …and the same recorder, handed an UNescalated verdict, notes nothing.
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_verdict_instruments(
            _v(VerifyVerdict.CONFIRMED, 0.92, "unavailable"),
            req_id="r2", trajectory_id="traj-cheap", verify_route="claim")
    assert agent._strong_verdict_for("traj-cheap") == ""


def test_the_sidecar_row_carries_the_escalation(agent, tmp_path):
    """FAILS IF: the durable record cannot answer 'which judge produced this'.

    The question that took log archaeology to ask must be answerable from
    the sidecar alone.
    """
    import json
    coll = MagicMock(); coll.root = tmp_path / "trajectories"
    (tmp_path / "trajectories").mkdir()
    agent.context.trajectory_collector = coll
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_verdict_instruments(
            _v(VerifyVerdict.REFUTED, 0.9, "unavailable"),
            req_id="r3", trajectory_id="traj-side", verify_route="claim")
    rows = [json.loads(l) for p in (tmp_path / "verdicts").glob("*.jsonl")
            for l in p.read_text().splitlines() if l.strip()]
    assert rows and rows[-1]["escalation"] == "unavailable"


# ---------------------------------------------------------------- the stamp

@pytest.mark.asyncio
async def test_record_escalation_stamps_the_outcome_even_when_the_ledger_is_off(monkeypatch):
    """FAILS IF: the in-process stamp is gated behind the ledger's own gates.

    The ledger refuses rows without a req_id (bench gate) and when the
    log is disabled; the verdict's `escalation` must not depend on either.
    """
    monkeypatch.setattr(V, "_escalation_log_enabled", lambda: False)
    cv = V._LAST_ESCALATION_OUTCOME
    token = cv.set("")
    try:
        V.record_escalation(kind="refute", route="claim", outcome="unavailable",
                            cheap_verdict="REFUTED", trace={})   # no req_id
        assert cv.get() == "unavailable"
    finally:
        cv.reset(token)


@pytest.mark.asyncio
async def test_the_wrapper_stamps_the_returned_verdict():
    """FAILS IF: the wrapper is bypassed or stamps the wrong object.

    The impl records "unavailable" and returns the CHEAP result; the
    wrapper must stamp that same object.
    """
    v = V.Verifier.__new__(V.Verifier)
    cheap = _v(VerifyVerdict.REFUTED, 0.9)

    async def _impl(result, *a, **k):
        V.record_escalation(kind="refute", route="claim", outcome="unavailable",
                            cheap_verdict="REFUTED", trace={})
        return result

    v._escalate_refute_impl = _impl
    out = await v._escalate_refute(cheap, "claim", "evidence", "ctx")
    assert out is cheap
    assert out.escalation == "unavailable"


@pytest.mark.asyncio
async def test_no_escalation_leaves_the_stamp_empty():
    """FAILS IF: a stale contextvar value leaks into an unescalated verdict."""
    v = V.Verifier.__new__(V.Verifier)
    cheap = _v(VerifyVerdict.CONFIRMED, 0.9)

    async def _impl(result, *a, **k):
        return result

    v._escalate_confirm_impl = _impl
    cv = V._LAST_ESCALATION_OUTCOME
    token = cv.set("upheld")   # stale value from elsewhere
    try:
        out = await v._escalate_confirm(cheap, high_stakes=True, retry=None)
    finally:
        cv.reset(token)
    assert out.escalation == ""


# ---------------------------------------------------------------- R1 enumeration

def _outcome_literals_in_escalation_functions():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # §4IP R7 i7: every function that records an outcome, not a name
        # prefix — `_settle_claim_binding` (claim_binding) and
        # `_guard_truncated_absence` (truncation_guard) were unwalked
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "record_escalation":
                for kw in call.keywords:
                    if kw.arg == "outcome" and isinstance(kw.value, ast.Constant):
                        found.add(str(kw.value.value))
    return found


def test_every_outcome_literal_is_classified():
    """R1 — FAILS IF: an escalation site records an outcome the precedence
    table does not know. An unclassified outcome would be treated as
    'not strong' by the memo and 'not unavailable' by the withhold —
    silently on neither side of the rule.
    """
    literals = _outcome_literals_in_escalation_functions()
    assert literals, "enumeration found no outcome literals — it has stopped working"
    missing = literals - ESCALATION_OUTCOMES
    assert not missing, f"unclassified escalation outcome(s): {sorted(missing)}"
    assert not (ESCALATION_STRONG_ADJUDICATED & ESCALATION_NOT_ADJUDICATED)


def test_enumeration_fires_on_an_unknown_outcome():
    """R7.2 — the enumeration must be able to go red."""
    assert {"upheld", "made_up_outcome"} - ESCALATION_OUTCOMES == {"made_up_outcome"}


def test_enumeration_walks_every_recording_function():
    literals = _outcome_literals_in_escalation_functions()
    assert {"claim_binding", "truncation_guard"} <= literals
