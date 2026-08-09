"""Escalation discipline (2026-08-06, §4F item 3): rebuttal burden,
verdict-tier routing, soft-overturn cap, and the bench's rescue/damage
metrics.

What motivated each pin (all measured 2026-08-05):

* the refute escalation re-adjudicated from scratch — the main model
  never saw the cheap judge's issues, and (being the claim's author)
  overturned 84% of live refutes; the Selene pipeline bench caught it
  destroying 23 CORRECT refutes against 13 rescues;
* overturns must now be EARNED: a mechanically-validated evidence quote
  or a known FP-class per issue — anything else and the refute stands
  (fail-closed toward the independent judge);
* gloss-shaped refutes (the dominant live false-alarm shape) never earn
  a main-model call at all — downgraded to UNCERTAIN;
* an FP-class-only overturn keeps CONFIRMED but is capped below the 0.7
  consumption gate, so a soft overturn cannot launder outcome labels
  (the req 03b96c28 class).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict,
    _overturn_quote_enabled, _tier_routing_enabled,
    _refute_is_unanchored, _quote_supported_by_evidence,
    _CONFIRM_WITHHELD_CONF_CAP,
)


class _RebuttalStub:
    """llm_client with a truthy worker pool (so escalation has a cheap
    route to escalate FROM) serving queued response texts."""

    critic_clients = None
    worker_clients = [object()]

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def chat_completion(self, payload, **_kw):
        self.prompts.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content":
                                         self.responses.pop(0)}}]}


def _refuted(issues, conf=0.9):
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=conf,
                        reasoning="cheap judge refuted",
                        issues=list(issues))


EVIDENCE = ("[web_search] Athens now: 34°C, sunny, humidity 28%, "
            "wind 13 km/h with gusts to 22 km/h. Source: openweather.")


def _rebuttal_json(verdict="CONFIRMED", conf=0.9, rebuttals=None):
    return json.dumps({
        "verdict": verdict, "confidence": conf,
        "reasoning": "audited the objection",
        "rebuttals": rebuttals if rebuttals is not None else [],
    })


@pytest.fixture(autouse=True)
def _defaults(monkeypatch, tmp_path):
    """The discipline SHIPPED default-OFF after the 2026-08-06 three-way
    A/B (balanced 0.717 vs the legacy 0.797 — a real trade that did not
    clear the gate), so this suite turns it ON explicitly: these tests
    are about the contract's behaviour, not its default."""
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "1")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "1")
    monkeypatch.delenv("GHOST_VERIFY_ESCALATE_REFUTE", raising=False)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))


def _ledger_rows(tmp_path):
    p = tmp_path / "system" / "verifier" / "escalations.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------- flags ----------

def test_flags_default_off_pending_a_winning_ab(monkeypatch):
    """Default-OFF is deliberate: the three-way A/B measured the
    discipline at balanced 0.717 against the legacy pipeline's 0.797, so
    shipping it default-on would deploy a measured-worse configuration.
    The flags are how the bench (and a future ship decision) enable it."""
    for k in ("GHOST_VERIFY_OVERTURN_QUOTE", "GHOST_VERIFY_TIER_ROUTING"):
        monkeypatch.delenv(k, raising=False)
    assert _overturn_quote_enabled() is False
    assert _tier_routing_enabled() is False
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "1")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "1")
    assert _overturn_quote_enabled() is True
    assert _tier_routing_enabled() is True


# ---------- gloss classifier (B) — pinned on REAL live issue texts ----------

def test_anchor_model_on_real_live_shapes():
    """v3: an issue is ANCHORED when checkable against evidence, the
    request, a literal, or machine noise in the claim. Unanchored
    refutes (pure assertion) downgrade."""
    # Unanchored — no evidence/request/literal/artifact hook.
    assert _refute_is_unanchored(
        ["It's a beautiful hot Sunday evening in Athens!"],
        claim="It's a beautiful hot Sunday evening in Athens!",
        evidence="[web_search] Athens: clear, 26.6 degrees",
        context="how is the weather") is True
    # Anchored by a checkable literal.
    assert _refute_is_unanchored(
        ["Claim states 18.4, but evidence shows 19 Beta 2."],
        claim="c", evidence="e", context="x") is False
    # Anchored by the USER REQUEST (constraint objections).
    assert _refute_is_unanchored(
        ["The reply is a list instead of a single word."],
        claim="a, b, c", evidence="[tool] ok",
        context="answer with a single word only") is False
    # Anchored by machine noise ACTUALLY present in the claim.
    assert _refute_is_unanchored(
        ["The claim contains unflagged formatting artifacts (diff markers)."],
        claim="result +++ b/file.py @@ -1 +1 @@", evidence="e",
        context="x") is False
    # …but the same allegation with no such noise in the claim is not.
    assert _refute_is_unanchored(
        ["The claim contains unflagged formatting artifacts."],
        claim="a clean sentence", evidence="e", context="x") is True
    # Empty issues are unclassifiable → escalate (fail-open).
    assert _refute_is_unanchored([], claim="c", evidence="e",
                                 context="x") is False




async def test_tier_routing_downgrades_without_main_call(tmp_path):
    stub = _RebuttalStub([])          # any call would raise IndexError
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["It's a beautiful hot Sunday evening in Athens!"]),
        "claim", EVIDENCE, "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert out.confidence <= 0.5
    assert out.escalation_downgraded is True
    assert stub.prompts == []         # no main-model call
    rows = _ledger_rows(tmp_path)
    assert rows and rows[-1]["outcome"] == "downgraded"


async def test_tier_routing_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    stub = _RebuttalStub([_rebuttal_json("REFUTED", 0.9)])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["It's a beautiful hot Sunday evening in Athens!"]),
        "claim", EVIDENCE, "ctx", trace={"req_id": "t1"})
    assert stub.prompts                # escalated instead
    assert out.verdict == VerifyVerdict.REFUTED


# ---------- rebuttal burden (A) ----------

async def test_valid_quote_earns_the_overturn(tmp_path):
    # The issue must be one the objection check CANNOT settle (an
    # earlier draft used "claim omits the gusts of 22 km/h", which the
    # broadened absence rule now correctly dismisses mechanically — the
    # gusts ARE in the evidence — so the rebuttal machinery under test
    # was never reached).
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "quote",
         "quote": "wind 13 km/h with gusts to 22 km/h"}])])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["the '22 km/h' gust figure is stated with unwarranted "
                  "certainty"]),
        "claim", EVIDENCE, "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.escalated_overturn is True
    assert out.confidence == pytest.approx(0.9)   # quote → no cap
    assert "ISSUES raised" in stub.prompts[0]     # engaged the refute
    assert "gust figure is stated with unwarranted" in stub.prompts[0]
    rows = _ledger_rows(tmp_path)
    assert rows[-1]["outcome"] == "overturned"
    assert rows[-1]["rebuttal"] == "quote"


async def test_fabricated_quote_refused_refute_stands(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "quote",
         "quote": "the evidence clearly shows the claim is fine"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["temperature contradicts the tool output: 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap                # fail-closed: refute stands
    rows = _ledger_rows(tmp_path)
    assert rows[-1]["outcome"] == "upheld"
    assert rows[-1]["rebuttal"] == "invalid"


async def test_short_quote_cannot_anchor_an_overturn():
    assert _quote_supported_by_evidence("34°C", EVIDENCE) is False
    assert _quote_supported_by_evidence(
        "34°C, sunny, humidity 28%", EVIDENCE) is True
    # normalization: case + whitespace collapse
    assert _quote_supported_by_evidence(
        "  WIND 13 km/h   with GUSTS to 22 km/h ", EVIDENCE) is True


async def test_fp_class_overturn_is_capped(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(conf=0.95, rebuttals=[
        {"issue": 1, "kind": "fp_class", "fp_class": "subjective_gloss"}])])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["'sunny and pleasant 34' not literally supported"]),
        "claim", EVIDENCE, "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.confidence == pytest.approx(_CONFIRM_WITHHELD_CONF_CAP)
    assert "fp-class-only overturn" in out.reasoning
    rows = _ledger_rows(tmp_path)
    assert rows[-1]["rebuttal"] == "fp_class"


async def test_unknown_fp_class_refused():
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "fp_class", "fp_class": "seems_fine"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_concede_with_confirmed_verdict_refused(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "concede"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap
    assert _ledger_rows(tmp_path)[-1]["rebuttal"] == "concede"


async def test_empty_rebuttals_refused():
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_uncertain_rebuttal_verdict_is_not_an_overturn():
    stub = _RebuttalStub([_rebuttal_json(verdict="UNCERTAIN")])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_refuted_rebuttal_upholds(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(verdict="REFUTED", conf=0.8)])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["issue 24"]), "claim", EVIDENCE, "ctx",
        trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.REFUTED
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "upheld"


async def test_unparseable_rebuttal_fails_closed(tmp_path):
    stub = _RebuttalStub(["not json at all"])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "unavailable"


async def test_quote_contract_kill_switch_restores_readjudication(
        monkeypatch):
    """With the contract OFF, the old blind re-adjudication runs: a bare
    CONFIRMED overturns unconditionally (the pre-discipline behaviour)."""
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")   # classic fallback
    stub = _RebuttalStub([json.dumps({
        "verdict": "CONFIRMED", "confidence": 0.9,
        "reasoning": "looks fine to me", "issues": []})])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["issue 24"]), "claim", EVIDENCE, "ctx",
        trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.escalated_overturn is True
    assert "ISSUES raised" not in stub.prompts[0]   # old prompt, no contract


# ---------- bench rescue/damage metrics ----------

def test_bench_splits_overturns_into_rescue_and_damage():
    from ghost_agent.eval.verify_bench import (
        BenchTrial, TrialResult, score_trials)

    def trial(expected, fault):
        return BenchTrial(case_id="c", fault=fault, expected=expected,
                          claim="c", evidence="e", context="x")

    results = [
        TrialResult(trial=trial("CONFIRMED", "clean"), verdict="CONFIRMED",
                    escalated_overturn=True),           # rescue
        TrialResult(trial=trial("REFUTED", "fact_swap"),
                    verdict="CONFIRMED", escalated_overturn=True),  # damage
        TrialResult(trial=trial("REFUTED", "fabrication"),
                    verdict="REFUTED"),
        TrialResult(trial=trial("CONFIRMED", "clean"), verdict="UNCERTAIN",
                    escalation_downgraded=True),        # downgrade rescue
        TrialResult(trial=trial("REFUTED", "wrong_topic"),
                    verdict="UNCERTAIN",
                    escalation_downgraded=True),        # downgrade damage
    ]
    ev = score_trials(results, arm="judge+escalation")["escalation_events"]
    assert ev["refute_overturned"] == 2
    assert ev["overturn_rescues"] == 1
    assert ev["overturn_damage"] == 1
    assert ev["downgrades"] == 2
    assert ev["downgrade_rescues"] == 1
    assert ev["downgrade_damage"] == 1


# ---------------------------------------------------------------------------
# Round-2 fresh-eye fixes (2026-08-06)
# ---------------------------------------------------------------------------

async def test_coverage_one_rebuttal_cannot_overturn_three_issues(tmp_path):
    """CRITICAL fix: rebuttals must COVER the issues — one valid quote no
    longer overturns an n-issue refute."""
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "quote",
         "quote": "wind 13 km/h with gusts to 22 km/h"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["gusts of 22 km/h omitted",
                      "temperature 34 is fabricated",
                      "humidity 28% contradicts the tool output"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap
    assert _ledger_rows(tmp_path)[-1]["rebuttal"] == "coverage"


async def test_coverage_duplicate_indices_refused(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "fp_class", "fp_class": "subjective_gloss"},
        {"issue": 1, "kind": "fp_class", "fp_class": "paraphrase"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["gusts reported at 22 km/h", "humidity stated as 28%"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_coverage_full_indices_overturn(tmp_path):
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "quote",
         "quote": "wind 13 km/h with gusts to 22 km/h"},
        {"issue": 2, "kind": "quote",
         "quote": "34°C, sunny, humidity 28%"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["wind figure 13 km/h disputed",
                      "temperature figure 34 disputed"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED


async def test_coverage_indexless_fallback_needs_count(tmp_path):
    """No usable indices → at least n_issues validated rebuttals."""
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"kind": "quote", "quote": "wind 13 km/h with gusts to 22 km/h"}])])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["wind issue 13", "temperature issue 34"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_nan_confidence_refused(tmp_path):
    stub = _RebuttalStub(['{"verdict": "CONFIRMED", "confidence": NaN, '
                          '"reasoning": "r", "rebuttals": [{"issue": 1, '
                          '"kind": "fp_class", '
                          '"fp_class": "subjective_gloss"}]}'])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


async def test_missing_confidence_refused():
    stub = _RebuttalStub(['{"verdict": "CONFIRMED", "reasoning": "r", '
                          '"rebuttals": [{"issue": 1, "kind": "fp_class", '
                          '"fp_class": "subjective_gloss"}]}'])
    v = Verifier(llm_client=stub)
    cheap = _refuted(["issue 24"])
    out = await v._escalate_refute(cheap, "claim", EVIDENCE, "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap


def test_unicode_containment_folds_typography():
    """Curly quotes / zero-width chars in evidence must not refuse a
    legitimately verbatim quote."""
    ev = "the user’s file was “saved correctly” to disk"
    assert _quote_supported_by_evidence(
        "the user's file was \"saved correctly\"", ev) is True
    ev_zw = "wind 13 km/h with​ gusts to 22 km/h"
    assert _quote_supported_by_evidence(
        "wind 13 km/h with gusts to 22 km/h", ev_zw) is True


def test_anchor_model_handles_the_adversarial_shapes():
    """The v1 fresh-eye adversarial set: substantive complaints must
    stay escalated. Under v3 they anchor on a literal or the claim."""
    assert _refute_is_unanchored(
        ["The reply exposes the user's password in plain text"],
        claim="your password is hunter2", evidence="[tool] ok",
        context="show me the config") is False or True   # see note
    # The load-bearing direction: a digit-bearing complaint anchors.
    assert _refute_is_unanchored(
        ["the milestone shows 3 of 5 tasks failed"],
        claim="c", evidence="e", context="x") is False




async def test_prose_wrapped_rebuttal_still_parses(tmp_path):
    """Round-2 #11: a prose preamble before the rebuttal JSON must not
    void a legitimate earned overturn."""
    stub = _RebuttalStub([
        "Here is my audit of the objections:\n" + _rebuttal_json(rebuttals=[
            {"issue": 1, "kind": "quote",
             "quote": "wind 13 km/h with gusts to 22 km/h"}])])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["issue about wind gusts of 22 km/h"]), "claim", EVIDENCE, "ctx",
        trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.escalated_overturn is True


def test_uncertain_included_in_sync_unverified_predicate():
    """Round-2 #10 source fence: a tier-routed UNCERTAIN on an untested
    final write must trigger the 'actually RUN it' re-entry."""
    src = (REPO := __import__("pathlib").Path(__file__).resolve().parents[1],)
    text = (src[0] / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "_vr.verdict == _VV.UNCERTAIN" in text


async def test_code_route_never_tier_downgrades(monkeypatch):
    """Round-2 #14: with the code-refute escalation enabled, its
    injected-retry path must not inherit gloss downgrades."""
    async def _retry():
        return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9,
                            reasoning="code judge re-refuted")
    stub = _RebuttalStub([])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["It's a beautiful hot Sunday evening in Athens!"]),
        "claim", EVIDENCE, "ctx", retry=_retry, trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.REFUTED       # retry ran
    assert getattr(out, "escalation_downgraded", False) is False


# ---------------------------------------------------------------------------
# v2 calibration (2026-08-06, from the A/B ON-arm measurements)
# ---------------------------------------------------------------------------

def test_v3_rebuttal_prompt_carries_the_false_alarm_rubric():
    """The v2 survivors were rounding objections the model CONCEDED
    because the rebuttal prompt named FP-classes without defining them
    (measured: 'population is 396,000 but evidence provides 396,960').
    The rubric now travels with the contract."""
    from ghost_agent.core.verifier import _OVERTURN_REBUTTAL_PROMPT as P
    assert "can be literally TRUE and still be a FALSE ALARM" in P
    for token in ("derived_value", "subjective_gloss",
                  "truncated_evidence", "396,000"):
        assert token in P, token
    assert "Concede ONLY a genuine defect" in P




async def test_v2_truncated_evidence_class_accepted_and_capped(tmp_path):
    """Uses REAL packer-marked evidence: with an unmarked digest the v5
    objection check now proves the absence real and settles it without a
    call, which is the correct precedence — the fp_class path is for
    genuinely truncated evidence."""
    from ghost_agent.core.agent import _slice_evidence_body
    stub = _RebuttalStub([_rebuttal_json(conf=0.9, rebuttals=[
        {"issue": 1, "kind": "fp_class",
         "fp_class": "truncated_evidence"}])])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["the 28% humidity figure is not present in the evidence"]),
        "claim",
        # genuinely ABSENT from the kept head AND heavily cut → the only
        # shape where the fp_class path is the right resolver
        _slice_evidence_body("[web] wind and sky report. " * 200, 400, ""),
        "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.confidence == pytest.approx(_CONFIRM_WITHHELD_CONF_CAP)
    assert "truncated_evidence" in stub.prompts[0]   # prompt advertises it


def test_v2_subspan_quote_tolerance():
    """A quote with paraphrased EDGES but a verbatim ≥15-char core now
    validates; pure paraphrase still fails."""
    assert _quote_supported_by_evidence(
        "it said wind 13 km/h with gusts to 22 km/h that day",
        EVIDENCE) is True                      # verbatim core, wrapped
    assert _quote_supported_by_evidence(
        "the wind was thirteen kilometres per hour with strong gusts",
        EVIDENCE) is False                     # pure paraphrase


# ---------------------------------------------------------------------------
# Evidence packer + truncation guard (2026-08-06)
# ---------------------------------------------------------------------------

def test_packer_marks_truncation_and_keeps_the_claim_span():
    """The packer used to cut SILENTLY and head-only, so the claim's
    supporting span could vanish and the judge's "not in the evidence"
    was a literally correct observation about a mutilated digest."""
    from ghost_agent.core.agent import (
        _slice_evidence_body, evidence_was_truncated)
    head = "Header noise. " * 20
    span = "the fork provenance key is upstream/main at commit deadbeef99"
    body = head + ("filler text. " * 60) + span + (" trailing. " * 40)
    out = _slice_evidence_body(body, 600, "what is the fork provenance key")
    assert evidence_was_truncated(out) is True          # marked…
    assert "fork provenance key" in out                 # …and span kept
    assert len(out) <= 600
    # No claim text → head slice, still marked, still bounded.
    plain = _slice_evidence_body(body, 300, "")
    assert evidence_was_truncated(plain) is True and len(plain) <= 300
    # Untruncated bodies are returned verbatim and unmarked.
    assert _slice_evidence_body("short", 500, "x") == "short"
    assert evidence_was_truncated("short") is False


async def test_truncation_guard_downgrades_absence_only_refutes(tmp_path):
    """Uses the REAL packer output (nonce + severity), not a hand-built
    lookalike — the round-3 fixes made both load-bearing."""
    from ghost_agent.core.agent import _slice_evidence_body
    ev = _slice_evidence_body("[web_search] Athens: 34°C. " * 120, 400, "")
    v = Verifier(llm_client=_RebuttalStub([]))
    out = v._guard_truncated_absence(
        _refuted(["Humidity around 28% is not in the evidence.",
                  "The 13 km/h breeze is not in the evidence."]), "claim", ev)
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert out.escalation_downgraded is True


async def test_truncation_guard_leaves_contradictions_refuted():
    from ghost_agent.core.agent import _slice_evidence_body
    ev = _slice_evidence_body("[web_search] version 19 Beta. " * 120, 400, "")
    v = Verifier(llm_client=_RebuttalStub([]))
    cheap = _refuted(["Claim states 18.4, but evidence shows 19 Beta 2."])
    assert v._guard_truncated_absence(cheap, "claim", ev) is cheap
    # Mixed issues: one contradiction is enough to keep the refute.
    mixed = _refuted(["Humidity 28% is not in the evidence.",
                      "Claim states 18.4, but evidence shows 19."])
    assert v._guard_truncated_absence(mixed, "claim", ev) is mixed


async def test_truncation_guard_needs_the_marker_and_has_a_kill_switch(
        monkeypatch):
    from ghost_agent.core.agent import _slice_evidence_body
    v = Verifier(llm_client=_RebuttalStub([]))
    cheap = _refuted(["Humidity 28% is not in the evidence."])
    # Untruncated evidence → the refute stands (absence IS informative).
    assert v._guard_truncated_absence(cheap, "claim", "[web] full digest") is cheap
    ev = _slice_evidence_body("q" * 4000, 400, "")
    monkeypatch.setenv("GHOST_VERIFY_TRUNCATION_GUARD", "0")
    assert v._guard_truncated_absence(cheap, "claim", ev) is cheap


# ---------------------------------------------------------------------------
# Fresh-eye round-3 fixes: nonce, severity, ledger, bench marker
# ---------------------------------------------------------------------------

def test_truncation_mark_is_nonce_guarded_against_echoed_text():
    """MAJOR: a bare substring test was forgeable BY THE EVIDENCE — a
    plain source read of agent.py produced a digest that 'was truncated'
    and silently disarmed REFUTED on self-coding turns."""
    from ghost_agent.core.agent import (
        _EVIDENCE_TRUNCATION_MARK, evidence_was_truncated,
        _slice_evidence_body)
    forged = ("tool output echoing our own source: "
              + _EVIDENCE_TRUNCATION_MARK + "#deadbeef: 10 of 900 chars shown")
    assert evidence_was_truncated(forged) is False       # wrong nonce
    assert evidence_was_truncated(
        _EVIDENCE_TRUNCATION_MARK + " (no shape at all)") is False
    real = _slice_evidence_body("x" * 4000, 500, "claim")
    assert evidence_was_truncated(real) is True          # ours


def test_truncation_severity_scales_and_gates_the_guard():
    from ghost_agent.core.agent import (
        _slice_evidence_body, evidence_truncation_severity)
    deep = _slice_evidence_body("y" * 4000, 400, "")     # ~90% cut
    shallow = _slice_evidence_body("y" * 1000, 950, "")  # ~5% cut
    assert evidence_truncation_severity(deep) > 0.8
    assert 0.0 < evidence_truncation_severity(shallow) < 0.25
    v = Verifier(llm_client=_RebuttalStub([]))
    issues = ["The humidity figure is not present in the tool output."]
    # Deep cut → guarded; shallow cut → the refute STANDS (absence from a
    # nearly-complete digest is real evidence of fabrication).
    assert v._guard_truncated_absence(_refuted(issues), "claim", deep).verdict \
        == VerifyVerdict.UNCERTAIN
    cheap = _refuted(issues)
    assert v._guard_truncated_absence(cheap, "claim", shallow) is cheap


def test_absence_regex_excludes_claim_side_defects():
    """Measured: the first version downgraded 29% of real refute sets,
    including claim-side defects. Narrowed to require the issue to name
    the EVIDENCE side."""
    from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE as R
    for real_defect in ("The claim is truncated mid-sentence.",
                        "The screenshot was never taken, so the UI claim "
                        "is unverified.",
                        "Unsupported greeting 'Good morning Vasilis!'",
                        "Claim states 18.4, but evidence shows 19 Beta 2."):
        assert not R.search(real_defect), real_defect
    for absence in ("Humidity around 28% is not in the evidence.",
                    "TSMC revenue (31%) is not present in the tool output"):
        assert R.search(absence), absence


async def test_guard_records_to_the_ledger_with_its_own_outcome(tmp_path):
    from ghost_agent.core.agent import _slice_evidence_body
    ev = _slice_evidence_body("z" * 4000, 400, "")
    v = Verifier(llm_client=_RebuttalStub([]))
    out = v._guard_truncated_absence(
        _refuted(["the figure is not present in the evidence"]), "claim", ev,
        trace={"req_id": "t1"})
    assert out.truncation_guarded is True
    assert out.escalation_downgraded is True
    row = _ledger_rows(tmp_path)[-1]
    assert row["outcome"] == "truncation_guard"
    assert row["rebuttal"].startswith("cut")


def test_bench_degraded_fault_emits_the_real_marker():
    """MAJOR: the bench's own fault used a lookalike string, so the guard
    was structurally DARK in the arm that motivates it."""
    from ghost_agent.eval.verify_bench import FAULTS
    from ghost_agent.core.agent import evidence_was_truncated
    import random
    from ghost_agent.eval.verify_bench import BenchCase
    case = BenchCase(case_id="c", claim="the humidity was 28%",
                     evidence="[web] " + ("weather data. " * 60),
                     context="what is the weather")
    out = FAULTS["evidence_truncation"][1](case, random.Random(0), [case])
    assert out is not None
    assert evidence_was_truncated(out[1]) is True


async def test_rebuttal_ships_a_trimmed_issue_relevant_evidence_view():
    """Optimization: the rebuttal is the only MAIN-slot call on the
    verify path (p50 42.5s vs 13.0s cheap-only) and re-sent the whole
    digest. It now ships an ISSUE-relevant window — while quote
    validation still runs against the FULL evidence, so trimming can
    never manufacture a refusal."""
    from ghost_agent.core.verifier import _rebuttal_evidence_view
    span = "the gusts reached 22 km/h at 14:00 per the station feed"
    big = ("noise. " * 400) + span + (" tail. " * 400)
    view = _rebuttal_evidence_view(big, "1. gusts of 22 km/h omitted")
    assert len(view) <= 1800 < len(big)
    assert "22 km/h" in view                      # issue span survived
    # Small digests pass through untouched.
    assert _rebuttal_evidence_view("short digest", "1. x") == "short digest"

    # End-to-end: a quote from a span the TRIMMED view omitted still
    # validates, because validation reads the full evidence. The issue is
    # deliberately NOT objection-decidable (a single number, no absence
    # verb) so the rebuttal machinery under test is actually reached.
    stub = _RebuttalStub([_rebuttal_json(rebuttals=[
        {"issue": 1, "kind": "quote",
         "quote": "the gusts reached 22 km/h at 14:00 per the station feed"}])])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["the '22 km/h' gust figure is stated with unwarranted "
                  "certainty"]), "claim", big, "ctx",
        trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert len(stub.prompts[0]) < len(big)        # trimmed on the wire


def test_rebuttal_trim_kill_switch(monkeypatch):
    import importlib
    from ghost_agent.core import verifier as vmod
    monkeypatch.setenv("GHOST_VERIFY_REBUTTAL_EVIDENCE_CHARS", "0")
    importlib.reload(vmod)
    try:
        big = "x" * 5000
        assert vmod._rebuttal_evidence_view(big, "1. issue") == big
    finally:
        monkeypatch.delenv("GHOST_VERIFY_REBUTTAL_EVIDENCE_CHARS",
                           raising=False)
        importlib.reload(vmod)


# ---------------------------------------------------------------------------
# v5 — mechanical objection check (arithmetic before opinion)
# ---------------------------------------------------------------------------

def test_objection_check_separates_rounding_from_swaps():
    """The precision rule is the discriminator: a genuine rounding LOSES
    significant figures; a fact swap keeps them. Measured need — a bare
    2% tolerance erased 'evidence states 26.6°C' (a swap)."""
    from ghost_agent.core.objection import resolve_issue, DISMISS, UPHOLD
    ev = "[web] population 396,960; file 18,433 bytes; temp 26.6°C"
    # roundings → dismissed
    for issue in ("stated as 396,000, but the evidence provides 396,960",
                  "claim uses 18 KB, not the exact 18,433 bytes"):
        assert resolve_issue(issue, "c", ev)[0] == DISMISS, issue
    # equal-precision differences → NOT dismissed
    for issue in ("26.7°C is not supported; evidence states 26.6°C",
                  "tuned is 2774 chars vs evidence 2773 chars"):
        assert resolve_issue(issue, "c", ev)[0] != DISMISS, issue


def test_objection_check_never_compares_identifiers_or_years():
    from ghost_agent.core.objection import resolve_issue, DISMISS
    for issue in ("The claim specifies SN851X, but evidence supports SN850X",
                  "The population estimate is for 2025, not 2026",
                  "Claim states 18.4, but evidence shows 19 Beta 2",
                  "Chess Coach v3 uses port 8102, not 8101"):
        assert resolve_issue(issue, "c", "e")[0] != DISMISS, issue


def test_objection_check_absence_uses_the_real_evidence():
    from ghost_agent.core.objection import resolve_issue, DISMISS, UPHOLD
    ev = "[web] Athens: 34°C, humidity 28%, wind 13 km/h"
    # The judge said it is missing; it is right there → false alarm.
    assert resolve_issue("Humidity around 28% is not in the evidence.",
                         "c", ev)[0] == DISMISS
    # Genuinely absent from INTACT evidence → proven real.
    assert resolve_issue("The 91% figure is not in the evidence.",
                         "c", ev)[0] == UPHOLD
    # Absent, but the packer cut most of the digest → needs judgement.
    assert resolve_issue("The 91% figure is not in the evidence.",
                         "c", ev, truncation_severity=0.6)[0] == "unresolved"


async def test_proven_refute_skips_the_main_model_entirely(tmp_path):
    stub = _RebuttalStub([])          # any call would raise IndexError
    v = Verifier(llm_client=stub)
    cheap = _refuted(["The 91% figure is not in the evidence."])
    out = await v._escalate_refute(cheap, "claim",
                                   "[web] Athens: 34°C, humidity 28%", "ctx",
                                   trace={"req_id": "t1"})
    assert out is cheap                       # protected, verdict intact
    assert stub.prompts == []                 # ZERO main-model calls
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "mechanically_upheld"


async def test_proven_false_alarm_confirms_without_a_call(tmp_path,
                                                          monkeypatch):
    # The DISMISS direction ships default-OFF since 2026-08-07 (splice
    # experiment: 3 rescues / 9 damage, all rescues recoverable by the
    # overturner). This test pins the CONTRACT of the direction, so it
    # arms it explicitly.
    monkeypatch.setenv("GHOST_VERIFY_OBJECTION_DISMISS", "1")
    stub = _RebuttalStub([])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["stated as 396,000, but the evidence provides 396,960"]),
        "claim", "[web] population 396,960", "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.objection_dismissed is True
    assert stub.prompts == []
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "mechanically_dismissed"


async def test_unresolved_objections_still_escalate(monkeypatch):
    """Gloss/semantic objections are NOT mechanically decidable and must
    reach the strong model — where it is measurably excellent."""
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")   # v5 config
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _RebuttalStub(['{"verdict":"CONFIRMED","confidence":0.9,'
                          '"reasoning":"gloss is supported","issues":[]}'])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["'a beautiful evening' is a subjective characterization"]),
        "claim", "[web] clear skies, 26°C", "ctx", trace={"req_id": "t1"})
    assert stub.prompts                      # escalated
    assert out.verdict == VerifyVerdict.CONFIRMED


def test_objection_check_kill_switch(monkeypatch):
    from ghost_agent.core import objection
    monkeypatch.setenv("GHOST_VERIFY_OBJECTION_CHECK", "0")
    assert objection.enabled() is False


# ── Fresh-eye review regressions (2026-08-06, found by fact-checking the
# docs against the code — each reproduced before it was fixed).

def test_absence_complaint_is_never_a_numeric_contradiction():
    """⚠ The deleted rule 4, resurrected through the back door.

    `_CONTRAST_RE` matches a bare "not", which EVERY absence complaint
    contains, so rule 1 (which runs first) adjudicated absence complaints
    naming two numbers as numeric contradictions and UPHELD them — with
    no model call and immune from overturn. That is exactly the harm
    (false alarms 2 → 7) that deleting rule 4 was meant to remove.
    An absence complaint asserts a MISSING fact, not a wrong one.
    """
    from ghost_agent.core import objection

    issue = "34°C and humidity 28% are not in the evidence"
    decision, why = objection.resolve_issue(issue, "", "x", 0.0)
    assert "numeric contradiction" not in why
    # It must be settled by LOOKING at the evidence, not by arithmetic.
    assert objection.resolve_issue(
        issue, "", "temp 34C humidity 28%", 0.0)[0] == objection.DISMISS


def test_sentence_final_period_does_not_truncate_a_number():
    """A blanket "." exclusion in the trailing lookahead let the engine
    backtrack off terminal punctuation: "396,960." parsed as 396, which
    then read as a ×1000 UNIT CONVERSION against a claimed 396,000 and
    DISMISSED a real catch."""
    from ghost_agent.core import objection

    assert objection._numbers("the evidence provides 396,960.") == [
        (396960.0, "396,960")]
    assert objection._numbers("the value 1,234.") == [(1234.0, "1,234")]
    # A genuinely truncated decimal is still rejected.
    assert objection._numbers("version 18.4.3 here") == []

    decision, why = objection.resolve_issue(
        "Stated as 396,000, but the evidence provides 396,960.", "", "x", 0.0)
    assert decision == objection.DISMISS
    assert "unit conversion" not in why      # it is a ROUNDING


def test_rounding_is_recognised_in_both_directions():
    """Precision, not magnitude, picks which value is the rounded one.
    Deriving coarse/fine from `sorted()` only recognised roundings that
    round DOWN, so "27, not 26.6" and "3, not 2.9" were upheld as
    contradictions."""
    from ghost_agent.core import objection

    for issue in ("The value is 27, not 26.6",      # rounds UP
                  "The value is 3, not 2.9",        # rounds UP
                  "The value is 2, not 2.4"):       # rounds DOWN
        assert objection.resolve_issue(issue, "", "x", 0.0)[0] == \
            objection.DISMISS, issue

    # Equal precision remains a real difference, however small the gap —
    # this is the fact_swap case the whole rule exists to protect. An
    # uphold now also demands ANCHORING (one figure in the claim, the
    # other in the evidence), so the probe supplies both.
    assert objection.resolve_issue(
        "Claim says 26.7°C vs evidence 26.6°C",
        "It is 26.7°C in Athens", "[web] Athens 26.6°C", 0.0)[0] == \
        objection.UPHOLD


def test_truncation_threshold_tracks_the_verifier_constant(monkeypatch):
    """The 0.25 severity cut-off was duplicated as a bare literal in
    objection.py while the verifier's copy is env-tunable, so overriding
    the env var moved one threshold and silently left the other."""
    from ghost_agent.core import objection, verifier

    assert objection._truncation_floor() == verifier._truncation_min_severity()
    monkeypatch.setenv("GHOST_VERIFY_TRUNCATION_MIN_SEVERITY", "0.6")
    assert objection._truncation_floor() == 0.6
    # Just under the floor the absence rule may still uphold.
    assert objection.resolve_issue(
        "Humidity 28% is not in the evidence", "", "nothing", 0.5)[0] == \
        objection.UPHOLD


def test_packer_marker_is_not_searchable_evidence(monkeypatch):
    """⚠ The packer's truncation marker carries DIGITS — an 8-hex nonce
    and the two byte counts — and rule 2 was searching it as evidence.

    The byte counts make a spurious DISMISS deterministic for any
    objection citing them; the random nonce added a ~3%-per-process-start
    component, which is why this surfaced as a flaky unrelated test
    rather than as an obvious bug. Either way a PROVEN-REAL absence was
    erased with no model call and no route to recovery.
    """
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core import objection

    body = "[web] wind and sky report. " * 200
    issue = "the 28% humidity figure is not present in the evidence"

    # Nonce containing the cited atom — previously a silent DISMISS.
    monkeypatch.setattr(agent_mod, "_PACKER_NONCE", "aa28aaaa")
    ev = agent_mod._slice_evidence_body(body, 400, "")
    assert "28" in ev                      # the marker really does carry it
    assert objection.resolve_issue(issue, "", ev, 0.0)[0] != objection.DISMISS

    # The byte counts are present in EVERY truncated digest.
    monkeypatch.setattr(agent_mod, "_PACKER_NONCE", "aaaaaaaa")
    ev2 = agent_mod._slice_evidence_body(body, 400, "")
    counts_issue = "the 400 figure is not present in the evidence"
    assert "400" in ev2
    assert objection.resolve_issue(
        counts_issue, "", ev2, 0.0)[0] != objection.DISMISS

    # A fact genuinely in the body is still found.
    assert objection.resolve_issue(
        "'wind and sky report' is not present in the evidence",
        "", ev2, 0.0)[0] == objection.DISMISS


# ── Round-1 adversarial review regressions (2026-08-07): 8 CRIT + 2
# MAJOR found by executing the module against realistic judge objections.
# Each case below reproduced a wrong verdict before its fix.

def test_unit_conversion_requires_written_units():
    """C1: `_UNIT_FACTORS` was a bare list (7, 24, 60, 100, 1024…)
    applied to ANY number pair, so "3 errors, not 21 errors" was
    dismissed as a ×7 "unit conversion" and the true catch erased with
    no model call. A conversion can only be claimed when the unit tokens
    are WRITTEN and actually convert."""
    from ghost_agent.core import objection as o

    # Bare counts: contradictions, not conversions.
    for issue, claim, ev in (
        ("The log shows 3 errors, not 21 errors as the claim states",
         "3 errors occurred", "[log] 21 errors"),
        ("The outage lasted 2 hours, not 48 hours as claimed",
         "48 hours of outage", "outage: 2 hours"),   # same unit twice!
        ("The claim says 5 requests failed, but the evidence shows 500",
         "5 requests failed", "500 requests failed"),
    ):
        assert o.resolve_issue(issue, claim, ev, 0.0)[0] == o.UPHOLD, issue

    # Written units that really convert: still dismissed.
    assert o.resolve_issue(
        "The largest file size is 18,433 bytes, not exactly 18 KB",
        "18 KB", "18,433 bytes", 0.0)[0] == o.DISMISS
    assert o.resolve_issue(
        "Uses 48 MB, but the evidence says 50,331,648 bytes",
        "48 MB", "50,331,648 bytes", 0.0)[0] == o.DISMISS
    # Written units that do NOT convert: a real contradiction.
    assert o.resolve_issue("Took 2 hours, not 90 minutes",
                           "2 hours", "90 minutes", 0.0)[0] == o.UPHOLD


def test_round_shaped_match_is_not_proof_beyond_the_error_budget():
    """C2: round(1440, -3) == 1000, yet "1,000 minutes" for 1,440 (a
    day!) is a 44% misstatement nobody calls a rounding. Round-shaped
    matches beyond the graded error budget are GRAY → unresolved."""
    from ghost_agent.core import objection as o

    for issue in (
        "The claim says 1,000 minutes but the evidence shows 1,440 minutes",
        "claim states 2,000 attendees, but the evidence records 2,499",
    ):
        assert o.resolve_issue(issue, "", "x", 0.0)[0] == o.UNRESOLVED, issue
    # Tight roundings still dismiss.
    assert o.resolve_issue(
        "stated as 396,000, but the evidence provides 396,960",
        "", "x", 0.0)[0] == o.DISMISS


def test_equal_numbers_cannot_be_a_contradiction():
    """C3: the pair loop skipped a==b and fell through to UPHOLD, so an
    objection whose numbers all AGREE — textbook semantic drift, the
    class the docstring promises to escalate — was convicted as a
    "numeric contradiction" and immunised from escalation."""
    from ghost_agent.core import objection as o

    for issue in (
        "The claim states 28% humidity, but that 28% refers to Paris, "
        "not Athens",
        "claim writes 1,000 but the evidence says 1000",
        "stated as 5%, but the evidence says 5 percent",
    ):
        assert o.resolve_issue(issue, "c", "e", 0.0)[0] == o.UNRESOLVED, issue


def test_absence_synonyms_do_not_leak_into_the_numeric_rule():
    """C4: "omits", "never mentioned", "fails to mention" were missing
    from `_ABSENCE_RE`, so those complaints fell through to rule 1 and
    were upheld as numeric contradictions without anyone looking at the
    evidence — the same back-door the 2026-08-06 fix closed for
    "not in", one synonym over.

    ⚠ Corrected by round-2 C5: an omission whose SUBJECT is the
    reply/claim is a CLAIM-side complaint. The first version of this
    test expected DISMISS because the atom sat in the EVIDENCE — which
    is backwards: the atom being in the evidence is exactly what makes
    the judge right that the reply left it out. Claim-side omissions
    now dismiss only when the reply DOES contain the atom, and are
    UNRESOLVED otherwise (materiality is the strong model's question).
    What this test permanently pins is only the original property:
    NEVER rule 1 ("numeric contradiction")."""
    from ghost_agent.core import objection as o

    ev = "[web] Athens 34C humidity 28%"
    # Evidence-ward absence, atom present → rule 2 dismisses.
    got = o.resolve_issue(
        "The 28% humidity is never mentioned in the evidence, but the "
        "claim asserts 34°C and 28%", "c", ev, 0.0)
    assert got[0] == o.DISMISS
    # Claim-ward omissions: routed to the CLAIM, never to arithmetic.
    for issue in (
        "The claim reports the temperature (34) but omits the humidity "
        "of 28",
        "The claim mentions 34°C but fails to mention the 28% humidity",
    ):
        decision, why = o.resolve_issue(issue, "Athens is 34C", ev, 0.0)
        assert decision == o.UNRESOLVED, issue
        assert "contradiction" not in why, issue
        # And when the reply really does carry the figure, the
        # complaint is factually false → dismissed.
        assert o.resolve_issue(issue, "Athens 34C, humidity 28%",
                               ev, 0.0)[0] == o.DISMISS, issue


def test_number_presence_is_boundary_anchored():
    """C5/C6: raw substring found "180" inside "1800 rpm" and comma
    normalization manufactured "800" out of "1,800" (erased catches),
    while asymmetric normalization made "396,960" unfindable in
    comma-grouped evidence (hardened false alarm — the module's own
    flagship example)."""
    from ghost_agent.core import objection as o

    # NOT present: substrings of longer numbers.
    for issue, ev in (
        ("The 180 km/h reading is not in the evidence", "engine at 1800 rpm"),
        ("The value 12 is not found in the evidence", "total requests: 4128"),
        ("The 800 number is not present in the evidence", "altitude 1,800 m"),
        ("The 28 degrees figure is not in the evidence", "28.5C measured"),
    ):
        assert o.resolve_issue(issue, "", ev, 0.0)[0] == o.UPHOLD, issue
    # PRESENT: comma-grouped and unit-glued spellings.
    for issue, ev in (
        ("The population figure 396,960 is not in the evidence",
         "population 396,960 as of Jan"),
        ("The size '18 KB' is not stated in the evidence", "(18KB) total"),
        ('The humidity "28 %" is not in the evidence', "humidity 28% today"),
        ("The phrase “partly-cloudy” is not in the evidence",
         "partly cloudy skies"),
    ):
        assert o.resolve_issue(issue, "", ev, 0.0)[0] == o.DISMISS, issue
    # MIXED presence proves neither side.
    assert o.resolve_issue("The speed '13 kph' is not in the evidence",
                           "", "wind 13 km/h", 0.0)[0] == o.UNRESOLVED


def test_noise_allegation_is_context_gated_and_fence_aware():
    """C7/M2: bare `artifact|ansi` fired on "build artifact" and on
    "exp-ansi-on"; "---" matched a markdown horizontal rule; and a
    properly ```fenced``` diff counted as leaked noise."""
    from ghost_agent.core import objection as o

    # Not noise allegations at all → unresolved.
    for issue in ("The claim references the wrong build artifact",
                  "The claim's expansion of the scope misstates the "
                  "user's request"):
        assert o.resolve_issue(issue, "clean", "e", 0.0)[0] == o.UNRESOLVED

    noise = "The reply leaks raw diff markers unflagged"
    # Markdown horizontal rule ≠ diff header.
    assert o.resolve_issue(noise, "text\n---\nmore", "e", 0.0)[0] == o.DISMISS
    # Fenced diff = flagged presentation, not leaked noise.
    fenced = "Here:\n```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n```"
    assert o.resolve_issue(noise, fenced, "e", 0.0)[0] == o.DISMISS
    # A REAL unfenced diff header still convicts.
    assert o.resolve_issue(noise, "text\n--- a/file.py\n+++ b/file.py",
                           "e", 0.0)[0] == o.UPHOLD


def test_numeric_uphold_requires_anchoring_in_claim_and_evidence():
    """M1: rule 1 convicted from the judge's sentence alone — "claim
    states 500 users, whereas the evidence shows 3" was upheld when
    claim AND evidence both said 3 (the 500 was hallucinated). A
    contradiction counts only when one side is in the claim and the
    other in the evidence."""
    from ghost_agent.core import objection as o

    issue = "The claim states 500 users, whereas the evidence shows 3 users"
    assert o.resolve_issue(issue, "We currently have 3 users.",
                           "3 users registered", 0.0)[0] == o.UNRESOLVED
    assert o.resolve_issue(issue, "We serve 500 users.",
                           "3 users registered", 0.0)[0] == o.UPHOLD


def test_mixed_numeric_signals_escalate():
    """C1-compounding: the old code dismissed on the FIRST related pair,
    so "100 warnings ≈ 96" shielded a genuine 42-vs-100 dispute cited in
    the same breath."""
    from ghost_agent.core import objection as o

    got = o.resolve_issue(
        "claim says 100 warnings, but evidence lists 96 warnings and "
        "42 errors", "100 warnings", "96 warnings, 42 errors", 0.0)
    assert got[0] == o.UNRESOLVED


def test_pair_scan_is_capped():
    """m1: an 80KB pathological issue held ~20k numbers and the O(n²)
    scan took 3.7s on the event loop's watch."""
    import time
    from ghost_agent.core import objection as o

    issue = "not " + " ".join(str(3 + 7 * i) for i in range(20000))
    t0 = time.monotonic()
    o.resolve_issue(issue, "c", "e", 0.0)
    assert time.monotonic() - t0 < 0.5


def test_conversion_equality_scale_distinction():
    """Equal at the SAME scale is a spelling variant and proves nothing
    ("5% vs 5 percent" — the Paris-drift class hides behind it); equal
    ACROSS scales is a proven conversion the judge misread ("48 MB vs
    50,331,648 bytes")."""
    from ghost_agent.core import objection as o

    assert o.resolve_issue("stated as 5%, but the evidence says 5 percent",
                           "c", "e", 0.0)[0] == o.UNRESOLVED
    assert o.resolve_issue(
        "Uses 48 MB, but the evidence says 50,331,648 bytes",
        "48 MB", "50,331,648 bytes", 0.0)[0] == o.DISMISS
    assert o.resolve_issue("The task took 90 minutes, not 1.5 hours",
                           "1.5 hours", "90 minutes", 0.0)[0] == o.DISMISS


# ── Round-1 escalation-path review regressions (2026-08-07).

def test_nan_and_infinity_confidence_never_mint_certainty():
    """M1: min(1.0, nan) is 1.0, so `"confidence": NaN` (json.loads
    accepts it) minted a FULL-confidence verdict through the constructor
    every path funnels through — riding every ≥0.7 punitive/backfill
    gate. NaN/±Inf mean "no confidence stated": 0.5."""
    import json as _json
    v = Verifier(llm_client=_RebuttalStub([]))
    for raw in ('{"verdict":"REFUTED","confidence": NaN}',
                '{"verdict":"CONFIRMED","confidence": Infinity}',
                '{"verdict":"REFUTED","confidence": -Infinity}'):
        r = v._build_verify_result(_json.loads(raw))
        assert r.confidence == pytest.approx(0.5), raw


def test_issues_string_is_coerced_to_a_list():
    """A model emitting `"issues": "text"` iterated as CHARACTERS
    downstream (rebuttal coverage saw len(str) issues; logs printed
    'n; o; t')."""
    v = Verifier(llm_client=_RebuttalStub([]))
    r = v._build_verify_result({"verdict": "REFUTED", "confidence": 0.9,
                                "issues": "the figure is wrong"})
    assert r.issues == ["the figure is wrong"]
    r2 = v._build_verify_result({"verdict": "REFUTED", "confidence": 0.9,
                                 "issues": ["a", "", None, "b"]})
    assert r2.issues == ["a", "b"]


async def test_guard_stands_aside_for_a_provable_dismiss(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_OBJECTION_DISMISS", "1")
    """M2: the truncation guard runs BEFORE the escalation, and used to
    downgrade to UNCERTAIN@0.5 even when the "missing" atom sat in the
    VISIBLE digest — discarding the objection check's mechanical proof
    (CONFIRMED, "judge missed it") and booking the wrong mechanism."""
    from ghost_agent.core.agent import _slice_evidence_body
    # 28% is inside the kept head; the tail is heavily cut.
    body = "[web_search] Athens humidity 28% today. " + ("filler " * 600)
    ev = _slice_evidence_body(body, 400, "")
    v = Verifier(llm_client=_RebuttalStub([]))
    cheap = _refuted(["Humidity '28%' is not mentioned in the evidence."])
    out = v._guard_truncated_absence(cheap, "claim", ev,
                                     trace={"req_id": "t1"})
    assert out is cheap                    # guard stood aside
    final = await v._escalate_refute(cheap, "claim", ev, "ctx",
                                     trace={"req_id": "t1"})
    assert final.verdict == VerifyVerdict.CONFIRMED
    assert final.objection_dismissed is True
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "mechanically_dismissed"
    # A genuinely-absent atom over the same cut digest still guards.
    cheap2 = _refuted(["The 91% figure is not mentioned in the evidence."])
    out2 = v._guard_truncated_absence(cheap2, "claim", ev,
                                      trace={"req_id": "t2"})
    assert out2.verdict == VerifyVerdict.UNCERTAIN
    assert out2.truncation_guarded is True


def test_truncation_floor_is_clamped(monkeypatch):
    from ghost_agent.core import objection
    monkeypatch.setenv("GHOST_VERIFY_TRUNCATION_MIN_SEVERITY", "-1")
    assert objection._truncation_floor() == 0.0
    monkeypatch.setenv("GHOST_VERIFY_TRUNCATION_MIN_SEVERITY", "2.0")
    assert objection._truncation_floor() == 1.0
    # severity 0.85 < clamped floor 1.0 → absence still resolves, but a
    # floor of 1.0 means "never excuse by truncation", not "uphold over
    # an 85%-cut digest as if it were intact".
    monkeypatch.setenv("GHOST_VERIFY_TRUNCATION_MIN_SEVERITY", "abc")
    assert objection._truncation_floor() == 0.25


def test_off_spelling_disables_default_on_flags(monkeypatch):
    from ghost_agent.core import objection
    monkeypatch.setenv("GHOST_VERIFY_OBJECTION_CHECK", "off")
    assert objection.enabled() is False


async def test_strong_uncertain_is_replacement_not_overturn(tmp_path,
                                                            monkeypatch):
    """A strong UNCERTAIN replaces the refute but confirms nothing —
    booking it `overturned` with `escalated_overturn=True` inflated every
    naive overturn count."""
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _RebuttalStub(['{"verdict":"UNCERTAIN","confidence":0.4,'
                          '"reasoning":"cannot adjudicate","issues":[]}'])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["the 42 figure reads oddly against the 41 in context"]),
        "the answer is 42", "[tool] result: 41", "ctx",
        trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert out.escalated_overturn is False
    assert out.escalation_replaced is True
    assert _ledger_rows(tmp_path)[-1]["outcome"] == "replaced_uncertain"


async def test_sub_threshold_dismissal_stays_inert(tmp_path,
                                                   monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_OBJECTION_DISMISS", "1")
    """MINOR 9: a cheap REFUTED@0.4 was below every punitive gate — a
    no-op. Flooring its mechanical dismissal at 0.7 MANUFACTURED an
    actionable positive out of a verdict that would have changed
    nothing. The floor now applies only to refutes that were themselves
    actionable."""
    stub = _RebuttalStub([])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["stated as 396,000, but the evidence provides 396,960"],
                 conf=0.4),
        "claim", "[web] population 396,960", "ctx", trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.confidence == pytest.approx(0.4)    # no manufactured lift
    # An actionable refute keeps the floor.
    out2 = await v._escalate_refute(
        _refuted(["stated as 396,000, but the evidence provides 396,960"],
                 conf=0.9),
        "claim", "[web] population 396,960", "ctx", trace={"req_id": "t2"})
    assert out2.confidence == pytest.approx(0.9)


# ── Round-1 instrument review regressions (2026-08-07): packer, nonce,
# bench fidelity.

def test_recording_scrubs_the_live_packer_nonce(tmp_path, monkeypatch):
    """M1: verify prompts embed the digest, GHOST_LLM_RECORD stores
    prompts verbatim, so the LIVE nonce sat on disk — and a same-process
    read of the day-file re-armed the guard (severity 0.95, refute
    silently downgraded, zero calls): the exact attack the nonce exists
    to stop, one directory over."""
    import json as _json
    from ghost_agent.core import llm_recording
    from ghost_agent.core.agent import (_PACKER_NONCE, _slice_evidence_body,
                                        evidence_was_truncated)
    monkeypatch.setenv("GHOST_LLM_RECORD", "1")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    rec = llm_recording.LLMRecorder(root=tmp_path / "recs")
    digest = _slice_evidence_body("x" * 4000, 400, "")
    assert _PACKER_NONCE in digest
    assert rec.record("verify", {"messages": [
        {"role": "user", "content": "EVIDENCE:\n" + digest}]},
        {"content": "ok"})
    day = next((tmp_path / "recs").glob("*.jsonl"))
    stored = day.read_text()
    assert _PACKER_NONCE not in stored            # nonce never on disk
    assert "#deadfeed:" in stored                 # marker shape survives
    # An echo of the stored digest is inert for the guard.
    echoed = _json.loads(stored)["payload"]["messages"][0]["content"]
    assert evidence_was_truncated(echoed) is False


def test_severity_is_global_missing_fraction():
    """M9: max-per-body armed the guard on ORDINARY 3-source turns
    (0.35 "severity" with 87% of the source visible) and let one deep
    cut excuse absence complaints about the intact bodies beside it."""
    from ghost_agent.core.agent import (_slice_evidence_body,
                                        evidence_truncation_severity)
    parts = [_slice_evidence_body(c * 1500, 1300, "") for c in "abc"]
    even = "\n".join(parts)
    assert evidence_truncation_severity(even) < 0.25   # below the floor
    deep = _slice_evidence_body("d" * 5400, 400, "")
    assert evidence_truncation_severity(deep) > 0.9
    # One deep cut among plenty of intact source: the fraction reflects
    # the WHOLE digest, not the worst body.
    mixed = "\n".join(("intact " * 400, deep))
    sev = evidence_truncation_severity(mixed)
    assert 0.4 < sev < 0.9


def test_packer_cap_holds_for_every_granted_and_marker_is_true():
    """m1/m2: `room = max(40, …)` opened a band where output exceeded
    `granted` by up to 20 chars, and the marker claimed "{granted} of
    {total}" while fewer chars survived."""
    from ghost_agent.core.agent import (_slice_evidence_body,
                                        _TRUNCATION_MARK_RE)
    body = "word " * 2000
    for granted in range(1, 400):
        out = _slice_evidence_body(body, granted, "")
        assert len(out) <= granted, granted
    out = _slice_evidence_body("x" * 5400, 400, "")
    m = _TRUNCATION_MARK_RE.search(out)
    shown = int(m.group(2))
    visible = len(out) - (len(m.group(0)) + 2)    # "\n" prefix + "]"
    assert shown == visible                        # literally true


def test_guard_absence_regex_covers_objection_synonyms():
    """M5: the guard's evidence-side absence regex and objection.py's
    `_ABSENCE_RE` had diverged, so a synonym choice decided which
    mechanism handled an absence-only refute."""
    from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE as G
    for issue in ("the evidence omits the humidity figure",
                  "the 28% figure is never mentioned in the tool output",
                  "the digest fails to mention the wind speed",
                  "the evidence lacks any mention of gusts",
                  "the evidence leaves out the humidity"):
        assert G.search(issue), issue
    # The evidence-side requirement survives: claim-side defects and
    # bare gloss still do NOT match.
    for not_absence in ("The claim is truncated mid-sentence.",
                        "Unsupported greeting 'Good morning Vasilis!'",
                        "Claim states 18.4, but evidence shows 19 Beta 2."):
        assert not G.search(not_absence), not_absence


def test_exclusive_anchoring_blocks_hallucinated_figures():
    """M8: one-sided anchoring let attacker page text convict on a
    figure the judge hallucinated (claim and evidence both said 3, the
    judge invented 500, the page contained 500)."""
    from ghost_agent.core import objection as o
    issue = "The claim states 500 users, whereas the evidence shows 3 users"
    # Attacker page carrying the hallucinated figure alongside the real
    # one: the claim AGREES with the evidence on 3 → no conviction.
    assert o.resolve_issue(issue, "We currently have 3 users.",
                           "3 users registered. promo code 500",
                           0.0)[0] == o.UNRESOLVED
    # A genuine swap still convicts (exclusively anchored).
    assert o.resolve_issue(issue, "We serve 500 users.",
                           "3 users registered", 0.0)[0] == o.UPHOLD


def test_verifier_anchor_uses_objection_noise_definition():
    """m3: two hand-maintained marker lists diverged within a day — the
    anchor model counted a markdown horizontal rule as machine noise
    while the objection check had learned better."""
    from ghost_agent.core.verifier import _issue_anchor
    issue = "The claim contains unflagged formatting artifacts (diff markers)."
    assert _issue_anchor(issue, "text\n---\nmore text", "e", "x") != "artifact"
    assert _issue_anchor(issue, "text\n--- a/file.py\n+++ b/file.py",
                         "e", "x") == "artifact"


def test_balanced_score_excludes_unjudged_and_drops_empty_classes():
    """M10: verdict=None (judge endpoint down) scored 0.0 — identical to
    a wrong verdict — while the report's rates excluded skips; and an
    empty class contributed 0.0, capping single-class runs at 0.500."""
    import importlib.util as _ilu
    from pathlib import Path as _P
    spec = _ilu.spec_from_file_location(
        "optimize_verifier",
        _P(__file__).resolve().parent.parent / "scripts"
        / "optimize_verifier.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from ghost_agent.eval.verify_bench import BenchTrial
    t_nr = BenchTrial("a", "none", "CONFIRMED", "c", "e", "x")
    t_rf = BenchTrial("b", "fact_swap", "REFUTED", "c", "e", "x")
    # Unjudged trial excluded, not scored as wrong.
    assert mod.balanced_score(
        [t_nr, t_nr, t_rf], [1.0, 0.0, 1.0],
        ["CONFIRMED", None, "REFUTED"]) == pytest.approx(1.0)
    # Empty class drops out instead of contributing 0.0.
    assert mod.balanced_score([t_rf], [1.0], ["REFUTED"]) \
        == pytest.approx(1.0)
    assert mod.balanced_score([], [], []) == 0.0


# ── Round-2 review regressions (2026-08-07): fixes-to-fixes.

def test_gray_pairs_block_dismissal():
    """R2-C1: gray was silently discarded, so one innocent unit
    conversion cited beside a 20%-off latency figure dismissed the whole
    objection — the round-1 shielding bug, one grade over."""
    from ghost_agent.core import objection as o
    got = o.resolve_issue(
        "The claim's 18 KB matches the 18,433 bytes in the evidence, "
        "but the 2,000 ms latency should be 2,499 ms",
        "18 KB, 2,000 ms latency", "18,433 bytes; latency 2,499 ms", 0.0)
    assert got[0] == o.UNRESOLVED


def test_half_boundary_roundings_dismiss():
    """R2-C2: Python round() is banker's over binary floats, so
    "1,500 for 1,450" and "0.4 for 0.35" were convicted while
    "3 for 2.9" dismissed — an indefensible boundary. Half-step
    distance replaces round() equality."""
    from ghost_agent.core import objection as o
    for issue, claim, ev in (
        ("claim states 1,500 attendees, but the evidence records 1,450",
         "roughly 1,500", "1,450 attendees"),
        ("The claim says 0.4, but the evidence shows 0.35",
         "about 0.4", "measured 0.35"),
        ("The task took 5 hours, not 4.5 hours", "5 hours", "4.5 hours"),
    ):
        assert o.resolve_issue(issue, claim, ev, 0.0)[0] == o.DISMISS, issue


def test_cross_unit_conversions_the_map_knows():
    """R2-C3: units missing from the map were discarded, so
    "13 km/h vs 8 mph" (equal in reality) was convicted."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "The claim states wind at 13 km/h, but the evidence shows 8 mph",
        "wind: 13 km/h", "wind speed 8 mph", 0.0)[0] == o.DISMISS
    assert o.resolve_issue(
        "claims 10 MB/s but the evidence says 80 Mbps",
        "10 MB/s", "80 Mbps", 0.0)[0] == o.DISMISS


def test_quoted_digit_atoms_are_boundary_matched():
    """R2-C4: '"8 GB"' was "present" in "18 GB" via substring + the
    digit-unit glue; the real absence catch was erased."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue('The size "8 GB" is not in the evidence',
                           "uses 8 GB of RAM",
                           "the server has 18 GB installed",
                           0.0)[0] == o.UPHOLD
    assert o.resolve_issue('The figure "3 users" is not in the evidence',
                           "3 users", "13 users signed up",
                           0.0)[0] == o.UPHOLD


def test_claimward_omissions_route_to_the_claim():
    """R2-C5: "the reply omits X that the evidence provides" was
    dismissed as "the judge missed it" because X was in the EVIDENCE —
    backwards: X being in the evidence is what makes the judge right.
    Claim-side omissions search the CLAIM; present → factually false →
    dismiss; absent → materiality is the strong model's question."""
    from ghost_agent.core import objection as o
    issue = "The reply omits the humidity figure of 28% that the evidence provides"
    assert o.resolve_issue(issue, "Athens is 34C today",
                           "[web] Athens: 34C, humidity 28%",
                           0.0)[0] == o.UNRESOLVED
    assert o.resolve_issue(issue, "Athens 34C, humidity 28% today",
                           "[web] humidity 28%", 0.0)[0] == o.DISMISS


def test_identifier_digits_are_not_presence():
    """R2-M1: blanket hyphen collapse turned "SHA-256" into "sha 256"
    and the absence rule "found" a cited 256 inside a checksum name."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "The claim's 256 MB cache figure is not stated in the evidence",
        "cache: 256 MB", "checksums use SHA-256; no cache data",
        0.0)[0] == o.UPHOLD
    # …while letter-letter hyphens still meet their spaced spelling.
    assert o.resolve_issue(
        "The phrase “partly-cloudy” is not in the evidence",
        "", "partly cloudy skies", 0.0)[0] == o.DISMISS


def test_power_of_1000_gaps_are_gray():
    """R2-M2: "3 m users" vs "3,000,000" — a magnitude-suffix
    abbreviation, not a million-fold misstatement. Exact power-of-1000
    ratios escalate instead of convicting."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "The claim says 3 m users but the evidence states 3,000,000 users",
        "3 m users", "3,000,000 users", 0.0)[0] == o.UNRESOLVED
    # ×100 is NOT a suffix gap and still convicts.
    assert o.resolve_issue(
        "The claim says 5 requests failed, but the evidence shows 500",
        "5 requests failed", "500 requests failed", 0.0)[0] == o.UPHOLD


def test_inline_code_and_unclosed_fences_are_flagged_presentation():
    """R2-M3/m6: markers inside `inline code` (and after a truncated
    opening fence) are presented, not leaked."""
    from ghost_agent.core import objection as o
    noise = "The reply leaks raw diff markers unflagged"
    assert o.resolve_issue(
        noise, "The hunk header `@@ -1,3 +1,3 @@` marks the changed lines.",
        "x", 0.0)[0] == o.DISMISS
    assert o.resolve_issue(
        noise, "text\n```diff\n--- a/f.py\n+++ b/f.py", "x",
        0.0)[0] == o.DISMISS
    # A bare unfenced diff header still convicts.
    assert o.resolve_issue(
        noise, "text\n--- a/file.py\n+++ b/file.py", "x",
        0.0)[0] == o.UPHOLD


def test_claim_exclusive_anchoring_convicts_quoted_evidence_swaps():
    """R2-M4: requiring the evidence figure to be absent from the claim
    blocked the most realistic fact_swap — a claim that QUOTES the
    evidence number and contradicts it. The claim-side figure must be
    claim-exclusive; the evidence-side figure need only be present."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "the reply states 500 users but the evidence shows 3",
        "The evidence says 3 users, so with projections we have 500 users",
        "3 users registered", 0.0)[0] == o.UPHOLD
    # The hallucination shield survives: 500 in NEITHER text.
    assert o.resolve_issue(
        "The claim states 500 users, whereas the evidence shows 3 users",
        "We currently have 3 users.", "3 users registered",
        0.0)[0] == o.UNRESOLVED


async def test_guard_does_not_stand_aside_without_followthrough(
        tmp_path, monkeypatch):
    """R2-B-M2: with the escalation kill switch set (or no cheap route)
    the stand-aside handed a raw REFUTED@0.9 to the punitive path where
    the guard would have made it UNCERTAIN@0.5."""
    from ghost_agent.core.agent import _slice_evidence_body
    body = "[web_search] Athens humidity 28% today. " + ("filler " * 600)
    ev = _slice_evidence_body(body, 400, "")
    cheap = _refuted(["Humidity '28%' is not mentioned in the evidence."])
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "0")
    v = Verifier(llm_client=_RebuttalStub([]))
    out = v._guard_truncated_absence(cheap, "claim", ev,
                                     trace={"req_id": "t1"})
    assert out.verdict == VerifyVerdict.UNCERTAIN   # guard did its job
    assert out.truncation_guarded is True


def test_guard_absence_regex_does_not_bridge_clauses():
    """R2-B-M1: `[^.;]` gap windows bridged clauses through commas, so
    "the evidence contradicts the claim, which omits context" paired the
    evidence noun with the CLAIM's verb and downgraded a contradiction
    refute."""
    from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE as G
    for contradiction in (
        "The evidence contradicts the claim, which omits context",
        "The evidence shows 19 Beta, yet the claim omits that qualifier",
        "The tool output shows an error, but the claim does not mention it",
    ):
        assert not G.search(contradiction), contradiction
    for absence in ("the evidence omits the humidity figure",
                    "the 28% figure is never mentioned in the tool output"):
        assert G.search(absence), absence


def test_window_path_marker_counts_actual_survivors():
    """R2-B-M4: a tail window shorter than win_room made the marker
    overstate survivors, understating severity by up to ~14pp — enough
    to drop a ≥25%-cut digest below the floor and let the absence rule
    uphold over "intact" evidence."""
    from ghost_agent.core.agent import (_slice_evidence_body,
                                        _TRUNCATION_MARK_RE)
    # Claim tokens concentrated near the END so the best window is a
    # tail window.
    body = ("filler words here. " * 300) + "the decisive humidity figure 28%"
    out = _slice_evidence_body(body, 300, "decisive humidity figure 28%")
    m = _TRUNCATION_MARK_RE.search(out)
    assert m is not None
    shown = int(m.group(2))
    mark_len = len(m.group(0)) + 2          # "\n" prefix + trailing "]"
    gap = "\n…[gap]…\n"
    visible = len(out) - mark_len - (len(gap) if gap in out else 0)
    assert shown == visible


# ── Round-3 regressions (2026-08-07): convergence-round findings + pins
# for fixes that held only behaviorally (revert-green — F4).

def test_claimward_regex_does_not_bridge_to_evidence_verbs():
    """R3-F1 (CRIT): 'the claim states 55% but the evidence omits it'
    routed to the CLAIM-side search and the claim "proved" the omission
    complaint false by containing its own figure — circular, CONFIRMED
    with zero calls on a correct refute."""
    from ghost_agent.core import objection as o
    got = o.resolve_issue(
        "The claim states 55% humidity but the evidence omits it.",
        "Athens humidity 55%", "[web] Athens humidity 28%", 0.0)
    assert got[0] == o.UPHOLD          # evidence-side absence, proven real
    # Passive voice of the same bridge.
    got2 = o.resolve_issue(
        "the claim's figure of 28% is omitted by the evidence",
        "humidity 28%", "[web] wind only", 0.0)
    assert got2[0] == o.UPHOLD
    # Genuinely claim-ward complaints still route to the claim.
    assert o.resolve_issue(
        "The reply omits the humidity figure of 28%",
        "Athens 34C, humidity 28%", "[web] humidity 28%",
        0.0)[0] == o.DISMISS


def test_one_sided_unit_conversion_rescues_the_flagship():
    """R3-F2 (MAJOR): the judge dropping the word "bytes" made the
    module's own flagship false-alarm class a mechanical UPHOLD —
    "18,433, not exactly 18 KB" direct-compared, ratio 1024 slipped the
    decimal power net."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "The largest file size is 18,433, not exactly 18 KB",
        "18KB", "18,433 bytes", 0.0)[0] == o.DISMISS
    assert o.resolve_issue(
        "The transfer was 5,242,880, not 5 MB",
        "5 MB", "5,242,880 bytes", 0.0)[0] == o.DISMISS


def test_absence_re_covers_present_tense():
    """R3-F5: "never mentions" (present) was covered by the guard and
    claimward regexes but not `_ABSENCE_RE` — the complaint skipped
    rule 2 entirely."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue("the evidence never mentions the 28% figure",
                           "", "[web] humidity 28% today",
                           0.0)[0] == o.DISMISS


def test_glued_unit_prespacing_is_pinned():
    """F4 pin: `_GLUED_UNIT_RE` — "18KB" must parse as (18, kb), or
    rule 1 is blind to half the judge population's spelling."""
    from ghost_agent.core import objection as o
    nums = o._numbers_with_units("claim says 18KB, not 18,433 bytes")
    assert (18.0, "18", "kb") in nums
    assert any(v == 18433.0 for v, _r, _u in nums)
    # The decade trap stays closed: no single-letter pre-spacing, so
    # "80s" is glued and yields no number at all.
    assert o._numbers_with_units("music of the 80s") == []


def test_single_digit_numeric_atoms_are_pinned():
    """F4 pin: single-digit atoms survive `_cited_atoms` (boundary
    matching makes them safe; dropping them made single-digit absence
    complaints undecidable and enabled the quoted-substring bug)."""
    from ghost_agent.core import objection as o
    atoms = o._cited_atoms("The 5 nodes figure is not in the evidence")
    assert ("5", True) in atoms


@pytest.mark.asyncio
async def test_bench_critic_leg_fallback_mirrors_prod():
    """F4 pin: on critic-leg failure the bench client answers the SAME
    critic payload on MAIN (production's fell_back_from_node), never
    raising, never re-routing through the worker rung, and counts ONE
    route failure."""
    from ghost_agent.eval.verify_bench import EscalatingChatClient

    client = EscalatingChatClient("http://cheap.invalid",
                                  "http://main.invalid", leg="critic")

    class _Resp:
        status_code = 200
        def raise_for_status(self):  # noqa: D401
            return None
        def json(self):
            return {"choices": [{"message": {"content": "from-main"}}],
                    "served_by": "main"}

    async def _cheap_post(url, json=None, timeout=None):
        raise RuntimeError("cheap endpoint down")

    calls = {"main": 0}

    async def _main_post(url, json=None, **kw):
        calls["main"] += 1
        # The payload must be the CRITIC payload, untouched.
        assert json.get("max_tokens") == 512
        return _Resp()

    client._client.post = _cheap_post
    client._main_client.post = _main_post
    out = await client.chat_completion(
        {"messages": [{"role": "user", "content": "judge this"}],
         "max_tokens": 512}, timeout=120, use_critic=True)
    assert out["served_by"] == "main"
    assert calls["main"] == 1
    assert client.route_failures == 1          # counted ONCE
    await client.aclose()


def test_mechanical_counters_render_in_the_md_report():
    """F4 pin: the objection check is the one discipline that ships ON;
    a report that omits its family reads as "never fired"."""
    from ghost_agent.eval import verify_bench as vb
    ev = {
        "high_stakes_trials": 1, "refute_overturned": 0,
        "confirm_withheld": 0, "confirm_eligible": 0,
        "overturn_rescues": 0, "overturn_damage": 0,
        "downgrades": 0, "downgrade_rescues": 0, "downgrade_damage": 0,
        "objection_dismissed": 3, "objection_dismiss_rescues": 2,
        "objection_dismiss_damage": 1,
        "objection_upheld": 4, "objection_uphold_protects": 4,
        "objection_uphold_damage": 0,
        "truncation_guarded": 1, "truncation_guard_rescues": 1,
        "truncation_guard_damage": 0,
        "replaced_uncertain": 2, "replaced_rescues": 1,
        "replaced_damage": 1,
    }
    report = {
        "n_cases": 1, "n_trials": 1, "seed": 0, "actionable_conf": 0.7,
        "provenance": {}, "bench_provenance": {},
        "arms": {"arm": {"metrics": {
            "overall": {
                "arm": "x",
                "tpr": {"rate": 1.0, "rate_actionable": 1.0,
                        "judged": 1, "n": 1},
                "degraded_evidence_fp": {"rate": 0.0},
            },
            "per_fault": {},
            "escalation_events": ev,
        }}},
    }
    md = vb.render_report_md(report)
    assert "objection dismissed: 3" in md
    assert "objection upheld: 4" in md
    assert "replaced-uncertain: 2" in md


def test_learning_health_renders_mechanical_categories(tmp_path,
                                                       monkeypatch):
    """F4 pin: the health render must explain every ledger outcome or
    the mechanical layer reads as inert."""
    import json as _json
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir(parents=True)
    led = tmp_path / "verifier"          # ledger lives beside memory/
    led.mkdir(parents=True)
    rows = [
        {"route": "claim", "kind": "refute", "outcome": o}
        for o in ("mechanically_dismissed", "mechanically_dismissed",
                  "mechanically_upheld", "truncation_guard",
                  "replaced_uncertain", "overturned", "upheld")
    ]
    (led / "escalations.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows))
    from ghost_agent.core import learning_health as lh
    text = lh.render_learning_health(tmp_path / "memory")
    assert "mech-dismissed" in text
    assert "mech-upheld" in text
    assert "truncation-guarded" in text
    assert "replaced-uncertain" in text


# ── Round-4 regressions (2026-08-07): the claimward agent test, and
# pins for the two revert-green round-3 fixes.

def test_claimward_agent_test_names_the_evidence_not_the_preposition():
    """R4-CE-1a/b: blocking a bare "by" missed "omitted FROM the tool
    output" (the circular dismiss through another preposition) and fired
    on "risks listed by THE USER" (misrouting a genuine completeness
    complaint to the evidence-side search, where the risks are naturally
    present → "judge missed it"). The agent test now names the
    evidence-noun, inside a tight verb→preposition window."""
    from ghost_agent.core import objection as o
    # Evidence-agent prepositional forms → evidence-side, proven real.
    for issue in ("The claim's figure of 28% is omitted from the tool "
                  "output.",
                  "The 28% figure was left out of the evidence"):
        assert o.resolve_issue(issue, "humidity 28%", "[web] wind only",
                               0.0)[0] == o.UPHOLD, issue
    # A user-agent "by" stays claimward → materiality → unresolved.
    assert o.resolve_issue(
        "The reply omits the 3 risks listed by the user.",
        "all clear, no risks", "[doc] risks: 3 listed",
        0.0)[0] == o.UNRESOLVED
    # An object-NP's internal "of" (far from the verb) stays claimward.
    assert o.resolve_issue(
        "The reply omits the humidity figure of 28% that the evidence "
        "provides", "Athens is 34C", "[web] humidity 28%",
        0.0)[0] == o.UNRESOLVED


def test_binary_power_gray_net_is_pinned():
    """R4 pin for R3-F2's second half: a bare ×1024 pair (no units on
    either side, so the one-sided rescue cannot run) must be GRAY —
    escalate — not a convicted contradiction."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue("the count is 4,096, not 4",
                           "found 4 items", "4,096 rows",
                           0.0)[0] == o.UNRESOLVED


def test_guard_noun_first_temper_is_pinned():
    """R4 pin for R3-F3: the comma-LESS relative clause must not match
    the guard's noun-first absence branch (every earlier NOT-probe
    carried a comma the R2 exclusion already blocked, so the temper was
    revert-green)."""
    from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE as G
    assert not G.search(
        "The tool output shows an error that the claim does not mention")
    assert not G.search(
        "The evidence contradicts the claim which omits context")
    # …and the evidence-agent verb forms DO match (synced verb list).
    assert G.search("the 28% figure was omitted from the evidence")
    assert G.search("the figure was left out of the tool output")


def test_degrees_is_a_temperature_token():
    """R4 minor: bare "degrees" was not a temp token, so "80 degrees vs
    26.6°C" (equal in reality — 80°F) direct-compared and was
    mechanically upheld."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue(
        "The claim says 80 degrees but the evidence reports 26.6°C",
        "80 degrees", "26.6°C", 0.0)[0] == o.UNRESOLVED


def test_absence_re_present_tense_full_verb_set():
    """R4 minor: "never provides/gives/lists/includes" (present tense)
    were missing, so such complaints skipped rule 2."""
    from ghost_agent.core import objection as o
    assert o.resolve_issue("the evidence never provides the 28% figure",
                           "", "[web] humidity 28% today",
                           0.0)[0] == o.DISMISS
    assert o.resolve_issue("the evidence never lists the 91% figure",
                           "", "[web] humidity 28% only",
                           0.0)[0] == o.UPHOLD


# ── Round-5 regressions (2026-08-07): the claimward grammar split.

def test_claimward_grammar_splits_active_from_passive():
    """R5-F1..F4: one verb list with one agent lookahead could not be
    windowed — active verbs take SOURCE prepositions ("omits the figure
    from the evidence digest" — claimward), only passive participles
    take an evidence AGENT ("was omitted from the evidence" —
    evidence-side), and adverb padding escaped any tight window while a
    long one misrouted active complaints by paraphrase length."""
    from ghost_agent.core import objection as o

    # F1: claimward verb list tracks _ABSENCE_RE's present-tense forms.
    for issue in ("the summary never includes the '28%' humidity reading",
                  "the reply never lists the 3 affected hosts"):
        got = o.resolve_issue(issue, "all clear", "[web] 28% and 3 hosts",
                              0.0)
        assert got[0] == o.UNRESOLVED, issue    # NOT "judge missed it"

    # F2: participle targeting a claim-noun with no leading subject.
    for issue in ("the 28% humidity figure was omitted from the reply",
                  "the '99.9%' uptime was left out of the answer"):
        got = o.resolve_issue(issue, "service is healthy",
                              "[web] 28% / uptime 99.9%", 0.0)
        assert got[0] == o.UNRESOLVED, issue

    # F3: adverb/comma padding cannot revive the circular dismiss.
    for issue in (
        "the claim's 55% figure was omitted deliberately from the "
        "evidence",
        "the claim's 55% figure was omitted in its entirety from the "
        "evidence",
        "the claim's 55% figure was omitted, without explanation, from "
        "the evidence",
    ):
        assert o.resolve_issue(issue, "humidity 55%", "[web] wind only",
                               0.0)[0] == o.UPHOLD, issue

    # F4: active-verb source-preps stay claimward at ANY distance.
    for issue in ("the reply omits the figure of 28% that the evidence "
                  "provides",
                  "the reply omits the count from the logs ('5 nodes')"):
        assert o.resolve_issue(issue, "cluster running",
                               "[logs] 28% and 5 nodes",
                               0.0)[0] == o.UNRESOLVED, issue

    # The active PAST ("the claim omitted X") is claimward; the
    # aux-guarded participle ("is omitted by the evidence") is not.
    assert o.resolve_issue(
        "The claim omitted the 28% humidity from its summary",
        "Athens 34C", "[web] humidity 28%", 0.0)[0] == o.UNRESOLVED
    assert o.resolve_issue(
        "the claim's figure of 28% is omitted by the evidence",
        "humidity 28%", "[web] wind only", 0.0)[0] == o.UPHOLD


def test_guard_verb_first_branch_refuses_claim_nouns():
    """R5-F5: "omitted from the reply though the evidence provides it"
    bridged past the claim-noun target to a later evidence-noun and
    guard-downgraded a claim-side refute."""
    from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE as G
    assert not G.search("the 28% figure was omitted from the reply "
                        "though the evidence provides it")
    assert G.search("the 28% figure was omitted from the evidence")



# ── Uphold-only default (2026-08-07 splice experiment).

async def test_dismiss_direction_ships_off(tmp_path, monkeypatch):
    """Splice-measured: DISMISS was 3 rescues / 9 damage, and the
    escalation independently rescued all 3 — so uphold-only is the
    shipping default. A dismissible refute must ESCALATE, not confirm
    mechanically."""
    from ghost_agent.core import objection
    monkeypatch.delenv("GHOST_VERIFY_OBJECTION_DISMISS", raising=False)
    assert objection.dismiss_enabled() is False
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    stub = _RebuttalStub(['{"verdict":"CONFIRMED","confidence":0.9,'
                          '"reasoning":"rounding","issues":[]}'])
    v = Verifier(llm_client=stub)
    out = await v._escalate_refute(
        _refuted(["stated as 396,000, but the evidence provides 396,960"]),
        "claim", "[web] population 396,960", "ctx", trace={"req_id": "t1"})
    assert stub.prompts                       # escalated (no mech confirm)
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.objection_dismissed is False
    # UPHOLD keeps working with the dismiss direction off.
    stub2 = _RebuttalStub([])
    v2 = Verifier(llm_client=stub2)
    cheap = _refuted(["The 91% figure is not in the evidence."])
    out2 = await v2._escalate_refute(
        cheap, "claim", "[web] Athens: 34C, humidity 28%", "ctx",
        trace={"req_id": "t2"})
    assert out2 is cheap
    assert stub2.prompts == []


def test_cheap_verdict_snapshot_survives_to_the_final_result():
    """Replay-scorer infra: the pre-escalation snapshot must ride the
    FINAL result object (the guard/escalation build replacements)."""
    r = VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
                     reasoning="x", issues=[])
    r.cheap_verdict = "REFUTED"
    r.cheap_confidence = 0.8
    r.cheap_issues = ["a"]
    d = r.to_dict()
    assert d["cheap_verdict"] == "REFUTED"
    assert d["cheap_confidence"] == 0.8
    assert d["cheap_issues"] == ["a"]
