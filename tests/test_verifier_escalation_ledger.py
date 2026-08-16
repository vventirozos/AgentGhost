"""The verifier's escalation ledger — the DURABLE home for the §4F
false-positive watch metrics (2026-08-04).

THE DEFECT THIS CLOSES. `escalated_overturn` (a cheap-judge REFUTE the main
model overturned) and `confirm_withheld` (a high-stakes cheap CONFIRMED the
main model would not confirm) were recorded into `VerifyResult.to_dict()` —
a serializer with **zero production callers**. Live proof at the time of the
fix: 160 "OVERTURNED a cheap-judge refute" lines in `ghost-agent.log`, and
ZERO occurrences of `escalated_overturn` anywhere under `$GHOST_HOME/system/`.

WHY NOT THE TRAJECTORY RECORD, the obvious candidate. On the STREAMED
delivery path `_record_turn_trajectory` runs in the SSE drain BEFORE the
stream verifier gate spawns the verdict at all, so a `turn_facts` stamp —
right for a fact known mid-turn — cannot carry a late verdict's flags there.
`test_streamed_trajectory_is_written_before_the_verdict_exists` pins that
ordering, because it is the reason this ledger exists rather than an
`extra` key. Writing where the escalation RESOLVES is path-independent by
construction, which these tests then prove on BOTH paths.

Both OUTCOMES are recorded (upheld as well as overturned/withheld): the
watch metric is a RATE, and a ledger of numerators cannot produce one.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from ghost_agent.core import verifier as vmod
from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict, record_escalation,
    _escalate_code_refute_enabled, _escalation_log_enabled,
    _escalation_log_path, _ESCALATION_LOG_MAX_BYTES,
)


@pytest.fixture(autouse=True)
def _legacy_escalation_flow(monkeypatch):
    """This suite's SUBJECT is the ledger writer and its wiring, and its
    stubs monkeypatch `_verify_claim_two_stage` — the leg the 2026-08-06
    rebuttal contract no longer uses. Pin the legacy flow so the stubs
    keep reaching it; the contract's own ledger rows (rebuttal field,
    outcome=downgraded) are covered in test_escalation_discipline.py."""
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """conftest deletes GHOST_HOME; the ledger resolves it per call."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    return tmp_path


def _ledger(home):
    p = home / "system" / "verifier" / "escalations.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


class _Client:
    """LLM client stub that advertises a cheap route (so escalation fires)."""
    def __init__(self, worker=True, critic=False):
        self.worker_clients = [object()] if worker else []
        self.critic_clients = [object()] if critic else []


@pytest.fixture
def v():
    return Verifier(llm_client=_Client())


def _res(verdict, conf=0.9, issues=None):
    return VerifyResult(verdict=verdict, confidence=conf, reasoning="r",
                        issues=list(issues or []))


# ── the writer itself ────────────────────────────────────────────────


class TestLedgerWriter:
    def test_writes_one_json_line_under_ghost_home(self, home):
        assert record_escalation(
            kind="refute", route="claim", outcome="overturned",
            cheap_verdict="REFUTED", cheap_confidence=0.9,
            strong_verdict="CONFIRMED", final_confidence=1.0,
            trace={"req_id": "r1", "trajectory_id": "t1"}) is True
        rows = _ledger(home)
        assert len(rows) == 1
        r = rows[0]
        assert r["kind"] == "refute" and r["route"] == "claim"
        assert r["outcome"] == "overturned"
        assert r["cheap_verdict"] == "REFUTED"
        assert r["strong_verdict"] == "CONFIRMED"
        assert r["cheap_confidence"] == 0.9
        assert r["final_confidence"] == 1.0
        assert r["req_id"] == "r1" and r["trajectory_id"] == "t1"
        assert r["ts"].endswith("Z")

    def test_appends_rather_than_overwrites(self, home):
        for i in range(3):
            record_escalation(kind="refute", route="claim", outcome="upheld",
                              trace={"req_id": f"r{i}"})
        assert [r["req_id"] for r in _ledger(home)] == ["r0", "r1", "r2"]

    def test_no_ghost_home_writes_nothing_and_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        assert _escalation_log_path() is None
        assert record_escalation(kind="refute", route="claim",
                                 outcome="upheld",
                                 trace={"req_id": "r"}) is False

    def test_kill_switch(self, home, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_ESCALATION_LOG", "0")
        assert _escalation_log_enabled() is False
        assert record_escalation(kind="refute", route="claim",
                                 outcome="upheld",
                                 trace={"req_id": "r"}) is False
        assert _ledger(home) == []
        monkeypatch.setenv("GHOST_VERIFY_ESCALATION_LOG", "1")
        assert _escalation_log_enabled() is True

    def test_absent_confidences_are_omitted_not_faked(self, home):
        assert record_escalation(kind="confirm", route="code",
                                 outcome="withheld",
                                 trace={"req_id": "r"}) is True
        r = _ledger(home)[0]
        assert r["trajectory_id"] == ""
        # A fabricated neutral is worse than a missing field for a metric.
        assert "cheap_confidence" not in r and "final_confidence" not in r

    def test_write_failure_never_raises(self, home, monkeypatch):
        def _boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(vmod.Path, "mkdir", _boom)
        assert record_escalation(kind="refute", route="claim",
                                 outcome="upheld",
                                 trace={"req_id": "r"}) is False

    def test_rotates_instead_of_growing_without_bound(self, home, monkeypatch):
        monkeypatch.setattr(vmod, "_ESCALATION_LOG_MAX_BYTES", 200)
        for i in range(8):
            record_escalation(kind="refute", route="claim", outcome="upheld",
                              trace={"req_id": f"r{i}"})
        base = home / "system" / "verifier" / "escalations.jsonl"
        rotated = home / "system" / "verifier" / "escalations.jsonl.1"
        # Rotation, not truncation: the previous generation is still on disk.
        assert rotated.exists()
        assert base.stat().st_size <= 200 + 300
        assert _ESCALATION_LOG_MAX_BYTES > 200  # module default untouched


# ── the escalation methods write it ──────────────────────────────────


class TestRefuteEscalationRecords:
    @pytest.mark.asyncio
    async def test_overturn_is_recorded(self, v, home, monkeypatch):
        # §4BK: these ledger tests exercise the two-stage retry path,
        # which is opt-in since classic became the designed MAIN
        # adjudicator — arm the switch (the legacy mix stays supported).
        monkeypatch.setenv("GHOST_VERIFY_MAIN_TWO_STAGE", "1")
        async def _strong(claim, evidence, context, force_main=False):
            return _res(VerifyVerdict.CONFIRMED, 1.0)
        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)
        out = await v._escalate_refute(
            _res(VerifyVerdict.REFUTED, 0.9, ["derived fact"]),
            "c", "e", "ctx", trace={"req_id": "rq", "trajectory_id": "tj"})
        assert out.escalated_overturn is True
        r = _ledger(home)[0]
        assert (r["kind"], r["route"], r["outcome"]) == (
            "refute", "claim", "overturned")
        assert r["strong_verdict"] == "CONFIRMED"
        assert r["req_id"] == "rq" and r["trajectory_id"] == "tj"

    @pytest.mark.asyncio
    async def test_upheld_is_recorded_too(self, v, home, monkeypatch):
        """The DENOMINATOR. Only logging overturns makes the rate
        uncomputable — which is how "84% of refutes are overturned" needed a
        hand-run log archaeology probe to establish in the first place."""
        # §4BK: arms the opt-in two-stage retry path (see the overturn test).
        monkeypatch.setenv("GHOST_VERIFY_MAIN_TWO_STAGE", "1")
        async def _strong(claim, evidence, context, force_main=False):
            return _res(VerifyVerdict.REFUTED, 0.95)
        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)
        await v._escalate_refute(_res(VerifyVerdict.REFUTED, 0.9),
                                 "c", "e", "ctx", trace={"req_id": "rq"})
        assert _ledger(home)[0]["outcome"] == "upheld"

    @pytest.mark.asyncio
    async def test_unparseable_strong_verdict_is_recorded_as_unavailable(
            self, v, home, monkeypatch):
        """A call was spent and produced nothing. 2 of 14 live main-model
        code-path replays came back empty at the 2048-token budget; an
        escalation that silently no-ops is indistinguishable from one that
        never ran."""
        async def _none(*a, **k):
            return None
        monkeypatch.setattr(v, "_verify_claim_two_stage", _none)
        monkeypatch.setattr(v, "_call_llm", lambda *a, **k: _none())
        out = await v._escalate_refute(_res(VerifyVerdict.REFUTED, 0.9),
                                       "c", "e", "ctx",
                                       trace={"req_id": "rq"})
        assert out.verdict == VerifyVerdict.REFUTED
        assert _ledger(home)[0]["outcome"] == "unavailable"

    @pytest.mark.asyncio
    async def test_escalation_error_is_recorded_as_unavailable(
            self, v, home, monkeypatch):
        async def _explode(*a, **k):
            raise RuntimeError("main model down")
        monkeypatch.setattr(v, "_verify_claim_two_stage", _explode)
        monkeypatch.setattr(v, "_call_llm", _explode)
        out = await v._escalate_refute(_res(VerifyVerdict.REFUTED, 0.9),
                                       "c", "e", "ctx",
                                       trace={"req_id": "rq"})
        assert out.verdict == VerifyVerdict.REFUTED   # unchanged behaviour
        assert _ledger(home)[0]["outcome"] == "unavailable"

    @pytest.mark.asyncio
    async def test_no_escalation_writes_no_row(self, v, home, monkeypatch):
        """A non-refute, a disabled switch and a missing cheap route are all
        NON-EVENTS — they must not pollute the rate."""
        async def _boom(*a, **k):
            raise AssertionError("must not escalate")
        tr = {"req_id": "rq"}
        monkeypatch.setattr(v, "_verify_claim_two_stage", _boom)
        await v._escalate_refute(_res(VerifyVerdict.CONFIRMED), "c", "e", "x",
                                 trace=tr)
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "0")
        await v._escalate_refute(_res(VerifyVerdict.REFUTED), "c", "e", "x",
                                 trace=tr)
        monkeypatch.delenv("GHOST_VERIFY_ESCALATE_REFUTE")
        v2 = Verifier(llm_client=_Client(worker=False, critic=False))
        await v2._escalate_refute(_res(VerifyVerdict.REFUTED), "c", "e", "x",
                                  trace=tr)
        assert _ledger(home) == []

    @pytest.mark.asyncio
    async def test_a_broken_ledger_never_breaks_a_verdict(
            self, v, home, monkeypatch):
        """`record_escalation` is called OUTSIDE the escalation's try/except
        (the try guards the LLM call, not the bookkeeping), so the whole of
        the writer — including path resolution and the enable check — has to
        be self-guarding. Break it at the deepest point and the verdict must
        still come back."""
        # §4BK: arms the opt-in two-stage retry path (see the overturn test).
        monkeypatch.setenv("GHOST_VERIFY_MAIN_TWO_STAGE", "1")
        async def _strong(claim, evidence, context, force_main=False):
            return _res(VerifyVerdict.CONFIRMED, 1.0)
        monkeypatch.setattr(v, "_verify_claim_two_stage", _strong)

        def _boom(*a, **k):
            raise OSError("read-only filesystem")
        monkeypatch.setattr(vmod.Path, "mkdir", _boom)
        out = await v._escalate_refute(_res(VerifyVerdict.REFUTED, 0.9),
                                       "c", "e", "ctx",
                                       trace={"req_id": "rq"})
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.escalated_overturn is True

    def test_writer_is_guarded_even_before_the_path_resolves(
            self, home, monkeypatch):
        """The enable check and path resolution used to sit outside the
        try/except; a hostile GHOST_HOME would then raise out of a verdict."""
        def _boom(*a, **k):
            raise RuntimeError("env exploded")
        monkeypatch.setattr(vmod, "_escalation_log_enabled", _boom)
        assert record_escalation(kind="refute", route="claim",
                                 outcome="upheld",
                                 trace={"req_id": "r"}) is False


class TestBenchAndSimulationAreExcluded:
    """The ledger measures the LIVE cheap judge's false-positive rate. Two
    callers fire the same escalation code for entirely different reasons and
    would corrupt that rate — this is the shape that let self-play write the
    production calibration corpus for weeks (§4J)."""

    def test_no_turn_identity_means_no_row(self, home):
        """`scripts/verify_bench.py` / `optimize_verifier.py` drive
        `verify_claim` with a two-leg client so `_escalate_refute` fires,
        dozens of times a run, in a shell where GHOST_HOME is exported. They
        pass no turn identity."""
        assert record_escalation(kind="refute", route="claim",
                                 outcome="overturned") is False
        assert record_escalation(kind="refute", route="claim",
                                 outcome="overturned", trace={}) is False
        assert record_escalation(kind="refute", route="claim",
                                 outcome="overturned",
                                 trace={"req_id": "   "}) is False
        assert record_escalation(kind="refute", route="claim",
                                 outcome="overturned",
                                 trace={"trajectory_id": "t"}) is False
        assert _ledger(home) == []

    @pytest.mark.asyncio
    async def test_bench_shaped_call_writes_nothing(self, v, home,
                                                    monkeypatch):
        """End to end through the public entry point the bench actually
        calls, with no `trace=` — exactly how `run_trials` invokes it."""
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            if force_main:
                return {"verdict": "CONFIRMED", "confidence": 1.0}
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_claim("c", "e", "ctx")
        assert out.escalated_overturn is True     # the escalation DID happen
        assert _ledger(home) == []                # it just is not live data

    @pytest.mark.asyncio
    async def test_self_play_turns_withhold_the_identity(self, home):
        """dream/self-play runs the full turn loop on a shallow-copied
        context; the simulation marker is the same one calibration uses."""
        from types import SimpleNamespace as NS
        from ghost_agent.core.agent import GhostAgent
        captured = {}

        class StubVerifier:
            llm_client = object()

            async def verify_claim(self, claim, evidence, context="",
                                   *, high_stakes=False, trace=None):
                captured["trace"] = trace
                return None

        agent = GhostAgent.__new__(GhostAgent)
        agent.context = NS(verifier=StubVerifier(),
                           args=NS(no_verifier=False),
                           skill_memory=NS(is_read_only=True))
        agent._active_constraint_note = lambda limit=5: ""
        await agent._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "web_search", "content": "x"}],
            messages=[], final_ai_content="answer",
            last_user_content="q", lc="q",
            req_id="SIM-1", trajectory_id="SIM-T")
        assert captured["trace"] is None
        assert _ledger(home) == []

    @pytest.mark.asyncio
    async def test_real_turns_keep_the_identity(self, home):
        from types import SimpleNamespace as NS
        from ghost_agent.core.agent import GhostAgent
        captured = {}

        class StubVerifier:
            llm_client = object()

            async def verify_claim(self, claim, evidence, context="",
                                   *, high_stakes=False, trace=None):
                captured["trace"] = trace
                return None

        agent = GhostAgent.__new__(GhostAgent)
        agent.context = NS(verifier=StubVerifier(),
                           args=NS(no_verifier=False),
                           skill_memory=NS(is_read_only=False))
        agent._active_constraint_note = lambda limit=5: ""
        await agent._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "web_search", "content": "x"}],
            messages=[], final_ai_content="answer",
            last_user_content="q", lc="q",
            req_id="REAL-1", trajectory_id="REAL-T")
        assert captured["trace"] == {"req_id": "REAL-1",
                                     "trajectory_id": "REAL-T"}


class TestConfirmEscalationRecords:
    @pytest.mark.asyncio
    async def test_withheld_records_the_precap_confidence(self, v, home):
        """The ledger must show what the CHEAP judge said (1.0), not the
        capped value the withhold writes back."""
        async def _strong():
            return _res(VerifyVerdict.REFUTED, 0.9)
        out = await v._escalate_confirm(
            _res(VerifyVerdict.CONFIRMED, 1.0), high_stakes=True,
            retry=_strong, route="code", trace={"req_id": "rc"})
        assert out.confirm_withheld is True
        r = _ledger(home)[0]
        assert (r["kind"], r["route"], r["outcome"]) == (
            "confirm", "code", "withheld")
        assert r["cheap_confidence"] == 1.0
        assert r["final_confidence"] == out.confidence < 0.7
        assert r["req_id"] == "rc"

    @pytest.mark.asyncio
    async def test_upheld_confirm_is_recorded(self, v, home):
        async def _strong():
            return _res(VerifyVerdict.CONFIRMED, 0.95)
        await v._escalate_confirm(_res(VerifyVerdict.CONFIRMED, 1.0),
                                  high_stakes=True, retry=_strong,
                                  trace={"req_id": "rc"})
        assert _ledger(home)[0]["outcome"] == "upheld"

    @pytest.mark.asyncio
    async def test_low_stakes_is_a_non_event(self, v, home):
        async def _strong():
            raise AssertionError("must not escalate")
        await v._escalate_confirm(_res(VerifyVerdict.CONFIRMED, 1.0),
                                  high_stakes=False, retry=_strong,
                                  trace={"req_id": "rc"})
        assert _ledger(home) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["returns_none", "raises"])
    async def test_both_failure_shapes_record_unavailable(self, v, home, mode):
        """Two distinct record sites in `_escalate_confirm` — an unparseable
        strong verdict and an exception — and neither may drop the call from
        the denominator."""
        async def _strong():
            if mode == "raises":
                raise RuntimeError("main model down")
            return None
        out = await v._escalate_confirm(
            _res(VerifyVerdict.CONFIRMED, 1.0), high_stakes=True,
            retry=_strong, trace={"req_id": "rc"})
        assert out.verdict == VerifyVerdict.CONFIRMED   # original stands
        assert out.confidence == 1.0                    # uncapped
        r = _ledger(home)[0]
        assert r["outcome"] == "unavailable" and r["kind"] == "confirm"
        assert r["cheap_confidence"] == 1.0


# ── wired through the public entry points ────────────────────────────


class TestWiredThroughEntryPoints:
    @pytest.mark.asyncio
    async def test_verify_claim_labels_the_route_and_carries_trace(
            self, v, home, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            if force_main:
                return {"verdict": "CONFIRMED", "confidence": 1.0}
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        await v.verify_claim("c", "e", "ctx",
                             trace={"req_id": "R", "trajectory_id": "T"})
        r = _ledger(home)[0]
        assert r["route"] == "claim" and r["outcome"] == "overturned"
        assert r["req_id"] == "R" and r["trajectory_id"] == "T"

    @pytest.mark.asyncio
    async def test_verify_code_output_labels_the_code_route(
            self, v, home, monkeypatch):
        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            if force_main:
                return {"verdict": "REFUTED", "confidence": 0.9}
            return {"verdict": "CONFIRMED", "confidence": 1.0}

        monkeypatch.setattr(v, "_call_llm", _call)
        await v.verify_code_output("code", "EXIT CODE: 1", "intent",
                                   response="all good", high_stakes=True,
                                   trace={"req_id": "RC"})
        r = _ledger(home)[0]
        assert r["route"] == "code" and r["kind"] == "confirm"
        assert r["req_id"] == "RC"


# ── code-path REFUTE escalation: wired, measured, default OFF ────────


class TestCodeRefuteEscalation:
    def test_default_is_off(self, monkeypatch):
        """Measured 2026-08-04 on the live recordings: 14/14 code-path cheap
        REFUTES upheld by the main model (7 cases replayed twice), against
        84% overturned on the claim path. The claim-path rate does not
        transfer, so this stays OFF until the ledger says otherwise."""
        monkeypatch.delenv("GHOST_VERIFY_ESCALATE_CODE_REFUTE", raising=False)
        assert _escalate_code_refute_enabled() is False

    @pytest.mark.asyncio
    async def test_off_means_one_call_and_the_refute_stands(
            self, v, home, monkeypatch):
        monkeypatch.delenv("GHOST_VERIFY_ESCALATE_CODE_REFUTE", raising=False)
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_code_output("code", "out", "intent",
                                         response="r", high_stakes=True,
                                         trace={"req_id": "X"})
        assert seen == [False]                    # no main-model call
        assert out.verdict == VerifyVerdict.REFUTED
        assert _ledger(home) == []

    @pytest.mark.asyncio
    async def test_on_escalates_with_the_CODE_prompt(
            self, v, home, monkeypatch):
        """Re-asking the CLAIM prompt would adjudicate a different question
        than the one the cheap judge answered."""
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CODE_REFUTE", "1")
        prompts = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            prompts.append((force_main, prompt))
            if force_main:
                return {"verdict": "CONFIRMED", "confidence": 1.0}
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_code_output("code", "out", "intent",
                                         response="r", trace={"req_id": "X"})
        assert [p[0] for p in prompts] == [False, True]
        assert prompts[0][1] == prompts[1][1]      # same CODE prompt
        assert "code output auditor" in prompts[1][1]
        assert out.verdict == VerifyVerdict.CONFIRMED
        assert out.escalated_overturn is True
        r = _ledger(home)[0]
        assert (r["kind"], r["route"], r["outcome"]) == (
            "refute", "code", "overturned")

    @pytest.mark.asyncio
    async def test_on_does_not_double_escalate_with_confirm(
            self, v, home, monkeypatch):
        """An overturned refute IS already the main model's verdict — the
        confirm escalation must not re-ask it (same guard the claim path
        has)."""
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CODE_REFUTE", "1")
        seen = []

        async def _call(prompt, temperature=0.1, max_tokens=2048,
                        json_only=False, force_main=False):
            seen.append(force_main)
            if force_main:
                return {"verdict": "CONFIRMED", "confidence": 1.0}
            return {"verdict": "REFUTED", "confidence": 0.9}

        monkeypatch.setattr(v, "_call_llm", _call)
        out = await v.verify_code_output("code", "out", "intent",
                                         response="r", high_stakes=True,
                                         trace={"req_id": "X"})
        assert seen == [False, True]               # exactly ONE main call
        assert out.confirm_withheld is False
        assert len(_ledger(home)) == 1


# ── BOTH delivery paths reach the ledger ─────────────────────────────


def _agent_source():
    import ghost_agent.core.agent as agent_mod
    return inspect.getsource(agent_mod)


class TestBothFinalizePaths:
    """This project has repeatedly shipped a fix live on one delivery path
    and dark on the other. The ledger is written where the escalation
    RESOLVES (inside the verifier), so both paths reach it by construction —
    but the JOIN key (req_id / trajectory_id) is threaded by the caller, and
    that is what can silently differ."""

    def _agent(self, captured):
        from ghost_agent.core.agent import GhostAgent

        class StubVerifier:
            llm_client = object()

            async def verify_claim(self, claim, evidence, context="",
                                   *, high_stakes=False, trace=None):
                captured["trace"] = trace
                return None

            async def verify_code_output(self, code, output, intent, *,
                                         response="", high_stakes=False,
                                         trace=None):
                captured["trace"] = trace
                return None

        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace(
            verifier=StubVerifier(), args=SimpleNamespace(no_verifier=False))
        agent._active_constraint_note = lambda limit=5: ""
        return agent

    @pytest.mark.asyncio
    async def test_verdict_funnel_forwards_identity_to_the_verifier(self):
        captured = {}
        agent = self._agent(captured)
        await agent._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "web_search", "content": "results"}],
            messages=[], final_ai_content="Here are the results.",
            last_user_content="find them", lc="find them",
            req_id="REQ-1", trajectory_id="TRAJ-1")
        assert captured["trace"] == {"req_id": "REQ-1",
                                     "trajectory_id": "TRAJ-1"}

    @pytest.mark.asyncio
    async def test_identity_is_optional_and_defaults_empty(self):
        """Legacy/isolated callers must not crash — they just record no
        identity."""
        captured = {}
        agent = self._agent(captured)
        await agent._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "web_search", "content": "results"}],
            messages=[], final_ai_content="answer",
            last_user_content="q", lc="q")
        assert captured["trace"] == {"req_id": "", "trajectory_id": ""}

    def test_every_verdict_call_site_forwards_identity(self):
        """AST, not string slicing: EVERY call to the verdict funnel (or its
        gated front door) anywhere in agent.py must pass req_id and a
        trajectory id. A new call site that forgets them is a new blind spot
        — the finalize gate, the in-loop auto-repair verdict and the streamed
        drain are three separate producers and this project has shipped a fix
        on one of them before."""
        import ast
        import ghost_agent.core.agent as agent_mod
        tree = ast.parse(_agent_source())
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute):
                continue
            if fn.attr not in ("_compute_verifier_verdict",
                               "_compute_verifier_verdict_gated"):
                continue
            kw = {k.arg for k in node.keywords}
            sites.append((fn.attr, node.lineno, kw))
        # finalize (gated) + two in-loop + the streamed drain + the two
        # forwards inside the gated front door.
        assert len(sites) >= 5, sites
        for name, lineno, kw in sites:
            assert "req_id" in kw, f"{name} at line {lineno} drops req_id"
            assert "trajectory_id" in kw, (
                f"{name} at line {lineno} drops trajectory_id")

    def test_streamed_gate_is_one_of_them(self):
        src = _agent_source()
        gate = src.split("VERIFIER GATE (STREAM)")[1][:6000]
        assert "_compute_verifier_verdict(" in gate
        assert "req_id=req_id" in gate
        assert "trajectory_id=current_trajectory_id" in gate

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", ["finalize", "stream_drain"])
    async def test_escalation_reaches_disk_from_either_path(
            self, home, monkeypatch, shape):
        """END TO END, no stubbed verifier: the REAL `Verifier`, driven by
        `_compute_verifier_verdict` with the kwargs each delivery path
        actually passes, must leave a line ON DISK under $GHOST_HOME.

        The two shapes differ only in identity and evidence source — which is
        the point: the ledger write lives at the escalation site, so neither
        path can be the dark one."""
        from ghost_agent.core.agent import GhostAgent

        calls = []

        class _LLM:
            worker_clients = [object()]
            critic_clients = []

            async def chat_completion(self, payload, **kw):
                calls.append(("main", kw))
                return {"choices": [{"message": {"content": json.dumps(
                    {"verdict": "CONFIRMED", "confidence": 1.0,
                     "reasoning": "supported", "issues": []})}}]}

            async def route(self, label, payload, **kw):
                calls.append(("worker", label))
                return json.dumps({"verdict": "REFUTED", "confidence": 0.9,
                                   "reasoning": "no", "issues": ["bad"]})

        monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = SimpleNamespace(
            verifier=Verifier(llm_client=_LLM()),
            args=SimpleNamespace(no_verifier=False))
        agent._active_constraint_note = lambda limit=5: ""

        ident = ({"req_id": "REQ-FIN", "trajectory_id": "TRJ-FIN"}
                 if shape == "finalize"
                 else {"req_id": "REQ-STR", "trajectory_id": "TRJ-STR"})
        v_result, _last = await agent._compute_verifier_verdict(
            tools_run_this_turn=[{"name": "web_search",
                                  "content": "the page says 42"}],
            messages=[], final_ai_content="The answer is 42.",
            last_user_content="what is the answer", lc="what is the answer",
            **ident)

        assert v_result.verdict == VerifyVerdict.CONFIRMED
        assert v_result.escalated_overturn is True
        rows = _ledger(home)
        assert len(rows) == 1, rows
        assert rows[0]["outcome"] == "overturned"
        assert rows[0]["route"] == "claim"
        assert rows[0]["req_id"] == ident["req_id"]
        assert rows[0]["trajectory_id"] == ident["trajectory_id"]

    def test_streamed_trajectory_is_written_before_the_verdict_exists(self):
        """THE reason the flags do not ride the trajectory record.

        In the SSE drain the trajectory is appended FIRST and the verifier
        gate spawns the verdict AFTER it, so no `turn_facts` stamp could
        ever carry a streamed turn's escalation into `Trajectory.extra`.
        If this ordering is ever reversed, revisit the ledger decision —
        which is exactly why it is pinned here rather than left in prose."""
        src = _agent_source()
        drain = src.split("Streamed-turn trajectory + work_log")[1]
        traj_at = drain.index("self._record_turn_trajectory(")
        verdict_at = drain.index("VERIFIER GATE (STREAM)")
        assert traj_at < verdict_at
