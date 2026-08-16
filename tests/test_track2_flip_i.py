"""§4BF Track 2 flip (i) — verify_bench instrumentation for the
GHOST_VERIFY_LOGIT_EXPECT probe A/B.

Three gaps the Track-2 scout found, each of which would have made the
pre-registered comparison silently unreadable:

  * the probe flags were INVISIBLE to provenance — a probe-on vs
    probe-off pair produced byte-identical verify_flags, so the status
    tool called a probe-flipped system "unchanged";
  * TrialResult had no probe_score — a null result could not
    distinguish "probe uninformative" from "informative but washed out
    by the blend" (the 2026-07-30 w=0.5 ambiguity);
  * there was no way to PIN the flag for a run — both arms inherited
    the shell, so one exported variable made the A/B measure nothing.
"""
import json
import os

import pytest

from ghost_agent.eval.verify_bench import (
    BenchCase,
    TrialResult,
    BenchTrial,
    run_bench,
    run_trials,
)
from ghost_agent.core.verifier import Verifier

pytestmark = pytest.mark.asyncio

CASE = BenchCase("c1", "The report has 3 sections.",
                 "[tool] wc: 3 sections found", "summarize the report")


class _RecordingStub:
    """Judge client that records the env the verifier saw per call."""

    def __init__(self, text):
        self.text = text
        self.seen_env = []

    async def chat_completion(self, payload, **_kw):
        self.seen_env.append(
            os.environ.get("GHOST_VERIFY_LOGIT_EXPECT", "<unset>"))
        return {"choices": [{"message": {"content": self.text}}]}


def _stub_verifier():
    stub = _RecordingStub(json.dumps(
        {"verdict": "CONFIRMED", "confidence": 0.9,
         "reasoning": "r", "issues": []}))
    return Verifier(llm_client=stub), stub


async def test_logit_expect_on_pins_env_and_provenance(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFY_LOGIT_EXPECT", raising=False)
    v, stub = _stub_verifier()
    report = await run_bench([CASE], v, arms=["two_stage_off"],
                             fault_names=["silent_failure"],
                             logit_expect="on")
    # The verifier saw the pin on every call…
    assert stub.seen_env and all(e == "1" for e in stub.seen_env)
    # …the report RECORDS the pinned value, not the pre-arm shell…
    assert report["provenance"]["verify_flags"][
        "GHOST_VERIFY_LOGIT_EXPECT"] == "1"
    # …and the shell is restored to unset afterwards.
    assert "GHOST_VERIFY_LOGIT_EXPECT" not in os.environ


async def test_logit_expect_off_pins_zero_and_restores_prior(monkeypatch):
    # An operator shell that exported the flag must not leak into an
    # "off" arm — and must get its value back afterwards.
    monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "1")
    v, stub = _stub_verifier()
    await run_bench([CASE], v, arms=["two_stage_off"],
                    fault_names=["silent_failure"], logit_expect="off")
    assert stub.seen_env and all(e == "0" for e in stub.seen_env)
    assert os.environ["GHOST_VERIFY_LOGIT_EXPECT"] == "1"


async def test_logit_expect_none_inherits_the_shell(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_LOGIT_EXPECT", "0")
    v, stub = _stub_verifier()
    report = await run_bench([CASE], v, arms=["two_stage_off"],
                             fault_names=["silent_failure"])
    assert stub.seen_env and all(e == "0" for e in stub.seen_env)
    # Inherited value is recorded as the shell had it.
    assert report["provenance"]["verify_flags"][
        "GHOST_VERIFY_LOGIT_EXPECT"] == "0"
    assert os.environ["GHOST_VERIFY_LOGIT_EXPECT"] == "0"


async def test_logit_expect_invalid_raises():
    v, _ = _stub_verifier()
    with pytest.raises(ValueError):
        await run_bench([CASE], v, arms=["two_stage_off"],
                        fault_names=["silent_failure"],
                        logit_expect="banana")


async def test_provenance_lists_both_probe_flags(monkeypatch):
    # Pipeline-selecting flags must ride provenance (and DEAD flags
    # must not — §4BL retired the weight with the blend).
    v, _ = _stub_verifier()
    report = await run_bench([CASE], v, arms=["two_stage_off"],
                             fault_names=["silent_failure"])
    flags = report["provenance"]["verify_flags"]
    assert "GHOST_VERIFY_LOGIT_EXPECT" in flags
    # §4BL: the WEIGHT flag died with the blend — recording a dead flag
    # as pipeline-selecting reads as phantom drift.
    assert "GHOST_VERIFY_LOGIT_EXPECT_WEIGHT" not in flags
    # §4BL cap selectors, recorded from birth.
    assert "GHOST_VERIFY_PROBE_CAP_T" in flags
    assert "GHOST_VERIFY_PROBE_CONF_CAP" in flags
    # §4BK R2: the MAIN-leg pipeline selectors were invisible to
    # provenance — the §4BK A/B's own config-equality check was
    # trivially blind to the flag under test.
    assert "GHOST_VERIFY_MAIN_TWO_STAGE" in flags
    assert "GHOST_VERIFY_MAIN_STAGE_STOP" in flags


async def test_trial_result_carries_probe_score():
    class _ProbeResult:
        class _V:
            value = "CONFIRMED"
        verdict = _V()
        confidence = 0.83
        reasoning = "r"
        issues = []
        suspects = None
        probe_score = 0.71

    class _StubVerifier:
        async def verify_claim(self, claim, evidence, context, **_kw):
            return _ProbeResult()

    trial = BenchTrial(case_id="c1", fault="clean", claim="c",
                       evidence="e", context="x", expected="CONFIRMED")
    results = await run_trials(_StubVerifier(), [trial])
    assert results[0].probe_score == 0.71
    assert results[0].to_dict()["probe_score"] == 0.71


async def test_trial_result_probe_score_defaults_none():
    # A probe that never fired must serialize as null, not vanish —
    # the comparator distinguishes "no probe" from "probe scored 0".
    trial = BenchTrial(case_id="c1", fault="clean", claim="c",
                       evidence="e", context="x", expected="CONFIRMED")
    tr = TrialResult(trial=trial, verdict="CONFIRMED")
    assert tr.probe_score is None
    assert "probe_score" in tr.to_dict()
    assert tr.to_dict()["probe_score"] is None


# ──────────────────────────────────────────────────────────────────
# Post-A/B instrumentation batch (R1 deferred fixes, applied after
# arm B run 1 — the ruler was frozen between arms)
# ──────────────────────────────────────────────────────────────────

def test_score_trials_probe_aggregation():
    # R1 C1: without an in-report aggregate, a silently-dead probe
    # minted a fake NULL. fire_rate/mean over eligible (C/R) trials.
    from ghost_agent.eval.verify_bench import score_trials
    def _tr(verdict, probe):
        t = BenchTrial(case_id="c", fault="clean", claim="c",
                       evidence="e", context="x", expected="CONFIRMED")
        return TrialResult(trial=t, verdict=verdict, confidence=0.9,
                           probe_score=probe)
    results = [_tr("CONFIRMED", 0.8), _tr("REFUTED", 0.2),
               _tr("CONFIRMED", None), _tr("UNCERTAIN", None)]
    pb = score_trials(results)["overall"]["probe"]
    assert pb["eligible"] == 3          # UNCERTAIN excluded
    assert pb["fired"] == 2
    assert pb["fire_rate"] == 0.667
    assert pb["mean_score"] == 0.5


def test_verifier_fingerprint_covers_entropy():
    # R1 M2: the probe's payload shape comes from entropy.request_logprobs;
    # a fingerprint blind to it certifies a stale probe-armed baseline.
    from ghost_agent.eval.verify_bench import verify_path_sources
    paths = verify_path_sources()["verifier"]
    assert any(p.endswith("core/entropy.py") for p in paths)


@pytest.fixture
def _esc_client():
    from ghost_agent.eval.verify_bench import EscalatingChatClient
    c = EscalatingChatClient.__new__(EscalatingChatClient)
    # Minimal attribute surface — no network objects.
    c.leg = "critic"
    c._model = ""
    c.main_model = ""
    c._timeout = 1.0
    c.base_url = "http://cheap"
    c.main_base_url = "http://main"
    c._client = object()
    c._main_client = object()
    c.route_calls = 0
    c.route_failures = 0
    c.route_timeouts = 0
    c.route_empty = 0
    c.probe_calls = 0
    c.probe_failures = 0
    return c


async def test_probe_calls_counted_apart_and_never_fall_to_main(
        _esc_client, monkeypatch):
    # R1 M4: probe failures used to increment route_failures → the
    # report branded a valid arm "judged by the MAIN model"; and the
    # fallback would have answered the probe on the strong model —
    # blending a different distribution than registered.
    calls = []

    async def boom(self, client, url, body, **kw):
        calls.append(url)
        raise RuntimeError("probe transport down")

    monkeypatch.setattr(type(_esc_client), "_post_cached", boom)
    probe_payload = {"messages": [], "logprobs": True, "top_logprobs": 10}
    with pytest.raises(RuntimeError):
        await _esc_client.chat_completion(probe_payload, use_critic=True)
    assert _esc_client.probe_calls == 1
    assert _esc_client.probe_failures == 1
    assert _esc_client.route_failures == 0
    assert calls == ["http://cheap"]      # never retried on main
    rh = _esc_client.route_health()
    assert rh["clean"] is True            # probe failure ≠ blend event
    assert rh["probe_failures"] == 1


async def test_judge_failures_still_fall_to_main(_esc_client, monkeypatch):
    seen = []

    async def flaky(self, client, url, body, **kw):
        seen.append(url)
        if url == "http://cheap":
            raise RuntimeError("cheap down")
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(type(_esc_client), "_post_cached", flaky)
    res = await _esc_client.chat_completion(
        {"messages": []}, use_critic=True)
    assert res["choices"]
    assert seen == ["http://cheap", "http://main"]
    assert _esc_client.route_failures == 1
    assert _esc_client.probe_failures == 0


async def test_use_worker_routes_to_cheap_on_worker_leg(
        _esc_client, monkeypatch):
    # R1 M1: use_worker was swallowed into **_kw — on --leg worker the
    # probe silently rode the MAIN endpoint.
    _esc_client.leg = "worker"
    seen = []

    async def capture(self, client, url, body, **kw):
        seen.append(url)
        return {"choices": [{"message": {"content": "5"}}]}

    monkeypatch.setattr(type(_esc_client), "_post_cached", capture)
    await _esc_client.chat_completion(
        {"messages": [], "logprobs": True}, use_worker=True)
    assert seen == ["http://cheap"]
    assert _esc_client.probe_calls == 1


# ──────────────────────────────────────────────────────────────────
# R4 pins — probe_score carry-through on reconstructed results
# ──────────────────────────────────────────────────────────────────

def _cheap(verdict, conf=0.9, issues=None, probe=0.87):
    from ghost_agent.core.verifier import VerifyResult
    r = VerifyResult(verdict=verdict, confidence=conf, reasoning="r",
                     issues=list(issues or []))
    r.probe_score = probe
    return r


class _CheapRouteClient:
    def __init__(self):
        self.worker_clients = [object()]
        self.critic_clients = []


async def test_r4_upheld_rebuttal_refute_keeps_probe_score(monkeypatch):
    # R4 MAJ-2: deleting any probe_score= carry-through line survived
    # the suite — the only detector was a future full bench run's V1.
    # This pins the REBUTTAL-CONTRACT upheld construction (the default
    # non-contract upheld path returns the STRONG re-verify's result,
    # whose own reading — or honest None — is the right attribution).
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "1")
    v = Verifier(llm_client=_CheapRouteClient())

    async def _rebuttal(prompt, **_kw):
        return {"verdict": "REFUTED", "confidence": 0.95,
                "reasoning": "objection holds",
                "rebuttals": [{"issue": 1, "kind": "concede"}]}

    monkeypatch.setattr(v, "_call_llm", _rebuttal)
    cheap = _cheap(VerifyVerdict.REFUTED, 0.9, ["fabricated version"])
    out = await v._escalate_refute(cheap, "c", "e", "ctx")
    assert out.verdict == VerifyVerdict.REFUTED
    assert out.probe_score == 0.87


async def test_r4_overturned_refute_keeps_probe_score(monkeypatch):
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "1")
    v = Verifier(llm_client=_CheapRouteClient())

    async def _rebuttal(prompt, **_kw):
        return {"verdict": "CONFIRMED", "confidence": 1.0,
                "reasoning": "evidence supports the claim",
                "rebuttals": [{"issue": 1, "kind": "quote",
                               "quote": "the derived fact is stated"}]}

    monkeypatch.setattr(v, "_call_llm", _rebuttal)
    cheap = _cheap(VerifyVerdict.REFUTED, 0.9,
                   ["derived fact 42 not restated"])
    out = await v._escalate_refute(
        cheap, "c", "evidence: the derived fact is stated here", "ctx")
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.escalated_overturn is True
    assert out.probe_score == 0.87


async def test_r4_tier_routed_downgrade_keeps_probe_score(monkeypatch):
    # The tier-routing downgrade constructs a fresh UNCERTAIN result —
    # the reading must ride along (reporting-only; the downgrade's own
    # confidence stands).
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "1")
    v = Verifier(llm_client=_CheapRouteClient())

    async def _no_answer(prompt, **_kw):
        return None

    monkeypatch.setattr(v, "_call_llm", _no_answer)
    cheap = _cheap(VerifyVerdict.REFUTED, 0.9,
                   ["the vibes feel off somehow"])
    out = await v._escalate_refute(
        cheap, "the report has three sections",
        "[tool] wc: 3 sections found", "summarize")
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert out.escalation_downgraded is True
    assert out.probe_score == 0.87
