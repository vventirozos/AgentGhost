"""CONFIRM escalation: a HIGH-STAKES cheap CONFIRMED is re-adjudicated on the
MAIN model before it is allowed to certify a turn (2026-08-04, §4J item 2).

Measured motivation, over the live stores (`$GHOST_HOME/system/...`):

* escalation was ONE-DIRECTIONAL by construction — 50 REFUTED escalations in
  the recorded window (84% overturned by the main model, matching the ~81%
  the journal records) and **0** CONFIRMED escalations;
* the >=0.7 consumption gate is a NO-OP on the cheap judge: 130 of 130
  recorded cheap verdicts came back at 0.9 or 1.0 — nothing was ever
  filtered by it, so every cheap CONFIRMED is consumed;
* since the 2026-07-31 honest-failure decision a CONFIRMED outranks a
  structural execution failure (`resolve_turn_outcome` rule 3 before rule 4),
  which makes those confirmations load-bearing: 61 of 1488 live trajectories
  (4.1%) are `outcome=passed` with a failed tool call in the turn;
* replaying 10 of those cheap CONFIRMEDs on the main model: 6 agreed, 1 was
  overturned, 3 came back unparseable (a no-op — the original stands).

The fix is deliberately NOT symmetric with `_escalate_refute`. A refute is
punitive (auditor note to the user, lesson retraction, FAILED corpus label,
auto-repair round); "the strong judge would not confirm this" is weaker
evidence than "the strong judge says this is wrong". So a withheld
confirmation keeps the CONFIRMED verdict and CAPS its confidence below every
>=0.7 consumption gate — the same idiom as agent.py's
`_WEB_EXEC_SKIP_CONF_CAP`. The turn is then recorded as unverified: no
fabricated PASSED, no manufactured failure.
"""
from __future__ import annotations

import pytest

from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict, _escalate_confirm_enabled,
    _CONFIRM_WITHHELD_CONF_CAP,
)


def _res(verdict, conf=0.95, issues=None):
    return VerifyResult(verdict=verdict, confidence=conf,
                        reasoning="r", issues=list(issues or []))


class _Client:
    """LLM client stub that advertises a cheap route."""

    def __init__(self, worker=True, critic=False):
        self.worker_clients = [object()] if worker else []
        self.critic_clients = [object()] if critic else []


@pytest.fixture
def v():
    return Verifier(llm_client=_Client())


async def _never(*a, **k):
    raise AssertionError("escalation must not have happened")


class TestConfirmEscalationBehaviour:
    @pytest.mark.asyncio
    async def test_withholds_when_main_will_not_confirm(self, v):
        async def _strong():
            return _res(VerifyVerdict.REFUTED, 0.95, ["format constraint"])

        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        out = await v._escalate_confirm(cheap, high_stakes=True, retry=_strong)
        # Verdict text is KEPT — the cap is what removes its authority.
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.confidence == _CONFIRM_WITHHELD_CONF_CAP
        assert out.confirm_withheld is True
        assert "CONFIRM escalation" in out.reasoning

    @pytest.mark.asyncio
    async def test_capped_confidence_is_below_every_consumption_gate(self, v):
        """The whole point: 0.7 is the threshold at which agent.py turns a
        verdict into `verifier_backfill=("passed", ...)`, which is what makes
        `resolve_turn_outcome` return PASSED over a structural failure."""
        assert _CONFIRM_WITHHELD_CONF_CAP < 0.7

    @pytest.mark.asyncio
    async def test_uncertain_from_main_also_withholds(self, v):
        async def _strong():
            return _res(VerifyVerdict.UNCERTAIN, 0.5)

        out = await v._escalate_confirm(
            _res(VerifyVerdict.CONFIRMED, 1.0), high_stakes=True,
            retry=_strong)
        assert out.confirm_withheld is True
        assert out.confidence == _CONFIRM_WITHHELD_CONF_CAP

    @pytest.mark.asyncio
    async def test_main_agreement_returns_the_strong_verdict(self, v):
        strong_res = _res(VerifyVerdict.CONFIRMED, 0.85)

        async def _strong():
            return strong_res

        out = await v._escalate_confirm(
            _res(VerifyVerdict.CONFIRMED, 1.0), high_stakes=True,
            retry=_strong)
        assert out is strong_res
        assert out.confirm_withheld is False

    @pytest.mark.asyncio
    async def test_low_stakes_confirmed_is_never_escalated(self, v):
        """A CONFIRMED on a turn where nothing failed is not load-bearing —
        escalating it would double the verifier cost on the common path."""
        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        out = await v._escalate_confirm(cheap, high_stakes=False,
                                        retry=_never)
        assert out is cheap
        assert out.confidence == 1.0

    @pytest.mark.asyncio
    async def test_non_confirmed_verdicts_are_untouched(self, v):
        for verdict in (VerifyVerdict.REFUTED, VerifyVerdict.UNCERTAIN):
            out = await v._escalate_confirm(_res(verdict), high_stakes=True,
                                            retry=_never)
            assert out.verdict == verdict
        assert await v._escalate_confirm(None, high_stakes=True,
                                         retry=_never) is None

    @pytest.mark.asyncio
    async def test_already_escalated_verdict_is_not_re_escalated(self, v):
        """A CONFIRMED produced by `_escalate_refute` already IS the main
        model's verdict — re-asking it is a wasted main-slot round-trip."""
        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        cheap.escalated_overturn = True
        out = await v._escalate_confirm(cheap, high_stakes=True, retry=_never)
        assert out is cheap

    @pytest.mark.asyncio
    async def test_no_cheap_route_means_nothing_to_escalate_to(self):
        v = Verifier(llm_client=_Client(worker=False, critic=False))
        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        assert (await v._escalate_confirm(cheap, high_stakes=True,
                                          retry=_never)) is cheap

    @pytest.mark.asyncio
    async def test_escalation_error_keeps_the_original_verdict(self, v):
        async def _explode():
            raise RuntimeError("main model down")

        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        out = await v._escalate_confirm(cheap, high_stakes=True,
                                        retry=_explode)
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.confidence == 1.0
        assert out.confirm_withheld is False

    @pytest.mark.asyncio
    async def test_unparseable_strong_verdict_keeps_the_original(self, v):
        """3 of 10 live replays came back unparseable — that must be a
        no-op, not a downgrade."""
        async def _none():
            return None

        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        out = await v._escalate_confirm(cheap, high_stakes=True, retry=_none)
        assert out is cheap
        assert out.confidence == 1.0

    @pytest.mark.asyncio
    async def test_kill_switch_restores_one_directional_behaviour(
            self, v, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "0")
        cheap = _res(VerifyVerdict.CONFIRMED, 1.0)
        assert (await v._escalate_confirm(cheap, high_stakes=True,
                                          retry=_never)) is cheap
        assert _escalate_confirm_enabled() is False

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("GHOST_VERIFY_ESCALATE_CONFIRM", raising=False)
        assert _escalate_confirm_enabled() is True

    def test_withheld_flag_is_persisted(self):
        r = _res(VerifyVerdict.CONFIRMED, 0.6)
        assert "confirm_withheld" not in r.to_dict()
        r.confirm_withheld = True
        assert r.to_dict()["confirm_withheld"] is True


class TestWiredThroughPublicEntryPoints:
    """A guard that only fires when a caller asks for it is worthless if no
    caller asks — these pin the plumbing, not the policy."""

    @pytest.mark.asyncio
    async def test_verify_claim_escalates_high_stakes_confirm(
            self, v, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            if force_main:
                return {"verdict": "REFUTED", "confidence": 0.9,
                        "issues": ["claimed success over a failed tool"]}
            return {"verdict": "CONFIRMED", "confidence": 1.0}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_claim("c", "e", "ctx", high_stakes=True)
        assert seen == [False, True]        # cheap first, then the main model
        assert out.confirm_withheld is True
        assert out.confidence == _CONFIRM_WITHHELD_CONF_CAP

    @pytest.mark.asyncio
    async def test_verify_claim_low_stakes_makes_one_call(self, v, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            return {"verdict": "CONFIRMED", "confidence": 1.0}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_claim("c", "e", "ctx")
        assert seen == [False]
        assert out.confidence == 1.0

    @pytest.mark.asyncio
    async def test_verify_code_output_escalates_high_stakes_confirm(
            self, v, monkeypatch):
        """The execute-shaped path matters MOST here: those are the turns
        that actually have a structural failure to override."""
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            if force_main:
                return {"verdict": "REFUTED", "confidence": 0.9}
            return {"verdict": "CONFIRMED", "confidence": 1.0}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_code_output("code", "EXIT CODE: 1", "intent",
                                         response="all good",
                                         high_stakes=True)
        assert seen == [False, True]
        assert out.confirm_withheld is True

    @pytest.mark.asyncio
    async def test_refute_escalation_still_runs_first(self, v, monkeypatch):
        """The two escalations must compose: a cheap REFUTED overturned to
        CONFIRMED by the main model is NOT then re-escalated."""
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            if force_main:
                # Escalation discipline (2026-08-06): the overturn must be
                # earned — a bare CONFIRMED is refused. fp_class, because
                # this micro-trial's one-char evidence can't host a quote.
                return {"verdict": "CONFIRMED", "confidence": 1.0,
                        "reasoning": "known false-positive pattern",
                        "rebuttals": [{"issue": 1, "kind": "fp_class",
                                       "fp_class": "subjective_gloss"}]}
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_claim("c", "e", "ctx", high_stakes=True)
        assert seen == [False, True]        # exactly ONE main-model call
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.escalated_overturn is True
        assert out.confirm_withheld is False


class TestHighStakesSignal:
    """`agent._turn_had_tool_failure` is the trigger. It must agree with the
    corpus's own failure sniffer, or the escalation fires on the wrong turns."""

    def test_detects_failed_tool_output(self):
        from ghost_agent.core.agent import _turn_had_tool_failure
        assert _turn_had_tool_failure([
            {"name": "file_system", "content": "ok"},
            {"name": "execute", "content": "boom\nEXIT CODE: 127"},
        ]) is True

    def test_synthetic_strikes_count(self):
        """Parse errors / blocked calls are real strikes on
        `execution_failure_count`, which is what rule 4 reads."""
        from ghost_agent.core.agent import _turn_had_tool_failure
        assert _turn_had_tool_failure([
            {"name": "system_parse_error",
             "content": "SYSTEM ERROR: Your previous output was CUT OFF",
             "_synthetic": True},
        ]) is True

    def test_clean_turn_is_not_high_stakes(self):
        from ghost_agent.core.agent import _turn_had_tool_failure
        assert _turn_had_tool_failure([
            {"name": "execute", "content": "hello\nEXIT CODE: 0"},
            {"name": "web_search", "content": "3 results"},
        ]) is False
        assert _turn_had_tool_failure([]) is False
        assert _turn_had_tool_failure(None) is False

    def test_uses_the_corpus_sniffer(self):
        """Same function, not a second copy — a drift here is how the corpus
        label and the escalation trigger would come to disagree."""
        from ghost_agent.distill.outcome_heuristics import (
            looks_like_tool_error, _looks_like_tool_error,
        )
        assert looks_like_tool_error is not _looks_like_tool_error
        for s in ("Error: nope", "EXIT CODE: 2", "Traceback (most recent",
                  "fine", "EXIT CODE: 0"):
            assert looks_like_tool_error(s) == _looks_like_tool_error(s)


class TestAgentGateWiring:
    """The trigger is computed inside `_compute_verifier_verdict` — the ONE
    place every verdict is produced (finalize gate, in-loop auto-repair, and
    the streamed late-verdict path all funnel through it and share its cached
    result). Pinning it here is what stops the escalation from being live on
    one delivery path and dark on the others."""

    def _agent(self, captured):
        from types import SimpleNamespace
        from ghost_agent.core.agent import GhostAgent

        class StubVerifier:
            llm_client = object()

            async def verify_claim(self, claim, evidence, context="",
                                   *, high_stakes=False, trace=None):
                captured["high_stakes"] = high_stakes
                captured["trace"] = trace
                return None

            async def verify_code_output(self, code, output, intent, *,
                                         response="", high_stakes=False,
                                         trace=None):
                captured["high_stakes"] = high_stakes
                captured["trace"] = trace
                return None

        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace(
            verifier=StubVerifier(), args=SimpleNamespace(no_verifier=False))
        agent._active_constraint_note = lambda limit=5: ""
        return agent

    @pytest.mark.asyncio
    async def test_failed_tool_marks_the_turn_high_stakes(self):
        captured = {}
        agent = self._agent(captured)
        v, _ = await agent._compute_verifier_verdict(
            tools_run_this_turn=[
                {"name": "browser", "content": "GOOD PAGE: results"},
                {"name": "file_system", "content": "Error: no such file"},
            ],
            messages=[], final_ai_content="Here are the results.",
            last_user_content="find the results", lc="find the results")
        assert captured["high_stakes"] is True

    @pytest.mark.asyncio
    async def test_clean_turn_is_not_high_stakes(self):
        captured = {}
        agent = self._agent(captured)
        await agent._compute_verifier_verdict(
            tools_run_this_turn=[
                {"name": "browser", "content": "GOOD PAGE: results"},
            ],
            messages=[], final_ai_content="Here are the results.",
            last_user_content="find the results", lc="find the results")
        assert captured["high_stakes"] is False


class TestCorpusEffect:
    """What the cap actually buys: the fabricated PASSED never lands."""

    def test_withheld_confirm_leaves_a_structural_failure_failed(self):
        from ghost_agent.distill.outcome_heuristics import resolve_turn_outcome
        # A capped CONFIRMED never becomes verifier_backfill=("passed", ...),
        # so the caller passes verifier=None and rule 4 stands.
        assert resolve_turn_outcome(
            current="unknown", verifier=None, execution_failed=True) == "failed"
        # ...while an UNESCALATED / confirmed-by-main pass still upgrades,
        # which is the 2026-07-31 honest-failure decision, kept intact.
        assert resolve_turn_outcome(
            current="unknown", verifier="passed",
            execution_failed=True) == "passed"
