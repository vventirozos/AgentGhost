"""Tests for the verifier fault-injection calibration bench
(src/ghost_agent/eval/verify_bench.py, added 2026-07-18).

The bench injects KNOWN corruptions into known-good (claim, evidence,
context) triples and measures the verifier's catch rate at a controlled
false-positive rate. These tests pin the offline machinery — the fault
library's applicability rules, rendered-prompt extraction from
GHOST_LLM_RECORD day-files, scoring math (including the production 0.7
actionable-confidence gate), and the arm runner's env hygiene — with a
stubbed judge, no network.
"""

from __future__ import annotations

import json
import os

import pytest

from ghost_agent.core.verifier import Verifier, _VERIFY_CLAIM_PROMPT
from ghost_agent.eval.verify_bench import (
    ACTIONABLE_CONF,
    BenchCase,
    BenchTrial,
    FAULTS,
    TrialResult,
    build_trials,
    extract_cases_from_recordings,
    fault_artifact_leak,
    fault_constraint_violation,
    fault_evidence_truncation,
    fault_fabrication,
    fault_fact_swap,
    fault_silent_failure,
    fault_wrong_topic,
    load_cases_jsonl,
    parse_rendered_claim_prompt,
    render_report_md,
    run_bench,
    score_trials,
)

import random

RNG = random.Random(0)

WEATHER = BenchCase(
    case_id="w1",
    claim="It's 34°C and sunny in Athens with humidity around 28%.",
    evidence="[web_search] Athens current: 34°C, sunny, humidity 28%, "
             "wind N 13 km/h. Tonight's low 26°C. Source meteo.gr, "
             "station Thiseio, updated 14:20 EEST. Air quality 42 good. "
             "No precipitation expected through midnight tonight.",
    context="what's the weather in Athens?")

PRIMES = BenchCase(
    case_id="p1",
    claim="There are 9592 primes below 100000.",
    evidence="[execute] count = 9592\nexit code 0",
    context="count primes below 100000")


# ---------- seed loading ----------


def test_load_cases_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(
        json.dumps({"id": "a", "claim": "c", "evidence": "e",
                    "context": "x"}) + "\n"
        + "not json\n"
        + json.dumps({"id": "b", "claim": "", "evidence": "e"}) + "\n",
        encoding="utf-8")
    cases = load_cases_jsonl(p)
    assert [c.case_id for c in cases] == ["a"]
    assert cases[0].source == "seed"


def test_shipped_seed_set_loads():
    from pathlib import Path
    seed = (Path(__file__).resolve().parents[1] / "scripts"
            / "verify_bench_cases.jsonl")
    cases = load_cases_jsonl(seed)
    assert len(cases) >= 10
    assert all(c.claim and c.evidence and c.context for c in cases)


# ---------- rendered-prompt extraction ----------


def test_parse_rendered_claim_prompt_round_trip():
    rendered = _VERIFY_CLAIM_PROMPT.format(
        claim="THE CLAIM", evidence="[t] THE EVIDENCE", context="THE ASK")
    parsed = parse_rendered_claim_prompt(rendered)
    assert parsed == {"claim": "THE CLAIM",
                      "evidence": "[t] THE EVIDENCE",
                      "context": "THE ASK"}


def test_parse_rendered_claim_prompt_rejects_other_text():
    assert parse_rendered_claim_prompt("") is None
    assert parse_rendered_claim_prompt("You are a code output auditor.") \
        is None


def test_extract_cases_from_recordings(tmp_path):
    rendered = _VERIFY_CLAIM_PROMPT.format(
        claim="claim A", evidence="[x] evidence A", context="ask A")
    verify_rec = {"ts": "2026-07-18T00:00:00Z", "ordinal": 1,
                  "kind": "route",
                  "payload": {"messages": [{"role": "user",
                                            "content": rendered}]},
                  "response": "CONFIRMED", "meta": {"task": "VERIFY"}}
    other_rec = {"ts": "2026-07-18T00:00:01Z", "ordinal": 2,
                 "kind": "chat_completion",
                 "payload": {"messages": [{"role": "user",
                                           "content": "hello"}]},
                 "response": {}, "meta": {}}
    p = tmp_path / "2026-07-18.jsonl"
    lines = [json.dumps(verify_rec), json.dumps(other_rec),
             json.dumps(verify_rec)]  # duplicate must dedup
    p.write_text("\n".join(lines), encoding="utf-8")
    cases = extract_cases_from_recordings([p, tmp_path / "missing.jsonl"])
    assert len(cases) == 1
    assert cases[0].claim == "claim A"
    assert cases[0].evidence == "[x] evidence A"
    assert cases[0].context == "ask A"
    assert cases[0].source == "recording"
    assert cases[0].case_id.startswith("rec-")


# ---------- fault library ----------


def test_fact_swap_contradicts_evidence():
    out = fault_fact_swap(WEATHER, random.Random(0), [WEATHER])
    assert out is not None
    claim, evidence, context, note = out
    assert claim != WEATHER.claim
    assert evidence == WEATHER.evidence
    # The mutated token is no longer the one the evidence backs.
    assert "swapped" in note


def test_fact_swap_none_without_shared_numbers():
    case = BenchCase("n1", "The sky is blue today.",
                     "[t] sky: blue, clear", "how's the sky?")
    assert fault_fact_swap(case, random.Random(0), [case]) is None


def test_fabrication_appends_absent_fact():
    out = fault_fabrication(WEATHER, random.Random(0), [WEATHER])
    assert out is not None
    claim, evidence, _, _ = out
    added = claim[len(WEATHER.claim):]
    assert added.strip()
    assert added.strip()[:20] not in evidence


def test_wrong_topic_swaps_donor_claim():
    out = fault_wrong_topic(WEATHER, random.Random(0), [WEATHER, PRIMES])
    assert out is not None
    assert out[0] == PRIMES.claim
    assert out[1] == WEATHER.evidence


def test_wrong_topic_none_without_context_or_donor():
    blank = BenchCase("b1", "c", "e", "")
    assert fault_wrong_topic(blank, random.Random(0), [blank, PRIMES]) \
        is None
    assert fault_wrong_topic(WEATHER, random.Random(0), [WEATHER]) is None


def test_silent_failure_keeps_claim_replaces_evidence():
    out = fault_silent_failure(WEATHER, random.Random(0), [WEATHER])
    claim, evidence, _, _ = out
    assert claim == WEATHER.claim
    assert evidence != WEATHER.evidence
    assert evidence.startswith("[web_search]")


def test_artifact_leak_injects_markers():
    out = fault_artifact_leak(WEATHER, random.Random(0), [WEATHER])
    assert "<<<<<<< SEARCH" in out[0]
    assert out[1] == WEATHER.evidence


def test_constraint_violation_adds_explicit_constraint():
    out = fault_constraint_violation(WEATHER, random.Random(0), [WEATHER])
    assert out is not None
    assert out[2].endswith("Answer with a single word only.")
    assert out[0] == WEATHER.claim
    short = BenchCase("s1", "Yes.", "[t] yes", "is it on?")
    assert fault_constraint_violation(short, random.Random(0), [short]) \
        is None


def test_evidence_truncation_is_not_refuted_class():
    assert FAULTS["evidence_truncation"][0] == "NOT_REFUTED"
    out = fault_evidence_truncation(WEATHER, random.Random(0), [WEATHER])
    assert out is not None
    assert out[0] == WEATHER.claim
    assert out[1].endswith("…[truncated]")
    assert len(out[1]) < len(WEATHER.evidence)
    assert fault_evidence_truncation(PRIMES, random.Random(0), [PRIMES]) \
        is None  # evidence too short to degrade meaningfully


# ---------- trial building ----------


def test_build_trials_deterministic_and_typed():
    cases = [WEATHER, PRIMES]
    t1 = build_trials(cases, seed=7)
    t2 = build_trials(cases, seed=7)
    assert [(t.case_id, t.fault, t.claim) for t in t1] == \
        [(t.case_id, t.fault, t.claim) for t in t2]
    clean = [t for t in t1 if t.fault == "clean"]
    assert len(clean) == 2
    assert all(t.expected == "CONFIRMED" for t in clean)
    assert any(t.fault == "fact_swap" for t in t1)


def test_build_trials_unknown_fault_raises():
    with pytest.raises(ValueError):
        build_trials([WEATHER], fault_names=["nope"])


def test_build_trials_fault_subset():
    trials = build_trials([WEATHER], fault_names=["silent_failure"],
                          include_clean=False)
    assert [t.fault for t in trials] == ["silent_failure"]


# ---------- scoring ----------


def _tr(fault, expected, verdict, conf=0.9, error=""):
    return TrialResult(
        trial=BenchTrial("c", fault, expected, "cl", "ev", "cx"),
        verdict=verdict, confidence=conf, error=error)


def test_score_trials_actionable_gate_mirrors_production():
    results = [
        _tr("fact_swap", "REFUTED", "REFUTED", conf=0.9),
        _tr("fact_swap", "REFUTED", "REFUTED",
            conf=ACTIONABLE_CONF - 0.1),
        _tr("fact_swap", "REFUTED", "CONFIRMED", conf=0.8),
        _tr("fact_swap", "REFUTED", None, conf=0.0),  # skipped
    ]
    m = score_trials(results)
    e = m["per_fault"]["fact_swap"]
    assert e["n"] == 4 and e["judged"] == 3 and e["skipped"] == 1
    assert e["catch_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert e["catch_rate_actionable"] == pytest.approx(1 / 3, abs=1e-3)


def test_score_trials_overall_tpr_fpr():
    results = [
        _tr("fact_swap", "REFUTED", "REFUTED", conf=0.9),
        _tr("fabrication", "REFUTED", "CONFIRMED", conf=0.9),
        _tr("clean", "CONFIRMED", "CONFIRMED", conf=0.9),
        _tr("clean", "CONFIRMED", "REFUTED", conf=0.9),
        _tr("evidence_truncation", "NOT_REFUTED", "UNCERTAIN", conf=0.4),
        _tr("evidence_truncation", "NOT_REFUTED", "REFUTED", conf=0.9),
    ]
    o = score_trials(results)["overall"]
    assert o["tpr"]["rate"] == pytest.approx(0.5)
    assert o["fpr"]["rate"] == pytest.approx(0.5)
    assert o["degraded_evidence_fp"]["rate"] == pytest.approx(0.5)


def test_score_trials_empty_denominators_are_none():
    m = score_trials([_tr("clean", "CONFIRMED", None)])
    assert m["per_fault"]["clean"]["false_alarm_rate"] is None
    assert m["overall"]["tpr"]["rate"] is None


# ---------- runner ----------


class _FixedVerdictStub:
    """Judge stub: same JSON verdict for every call."""
    critic_clients = None

    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def chat_completion(self, payload, **_kw):
        self.calls += 1
        return {"choices": [{"message": {"content": self.text}}]}


async def test_run_bench_report_shape_and_env_hygiene(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "1")
    stub = _FixedVerdictStub(json.dumps(
        {"verdict": "CONFIRMED", "confidence": 0.9,
         "reasoning": "r", "issues": []}))
    v = Verifier(llm_client=stub)
    report = await run_bench(
        [WEATHER, PRIMES], v, arms=["two_stage_off"],
        fault_names=["silent_failure"], seed=1)
    assert report["n_cases"] == 2
    arm = report["arms"]["two_stage_off"]
    assert arm["metrics"]["overall"]["fpr"]["n"] == 2  # two clean trials
    assert len(arm["trials"]) == report["n_trials"]
    # env restored to what the caller had set
    assert os.environ["GHOST_VERIFY_TWO_STAGE"] == "1"
    # single-stage arm => exactly one LLM call per trial
    assert stub.calls == report["n_trials"]


async def test_run_bench_unknown_arm_raises():
    with pytest.raises(ValueError):
        await run_bench([WEATHER], Verifier(llm_client=None),
                        arms=["sideways"])


async def test_run_bench_captures_judge_exceptions():
    class _Boom:
        critic_clients = None

        async def chat_completion(self, payload, **_kw):
            raise RuntimeError("node down")

    v = Verifier(llm_client=_Boom())
    report = await run_bench([PRIMES], v, arms=["two_stage_off"],
                             fault_names=["silent_failure"])
    trials = report["arms"]["two_stage_off"]["trials"]
    # Verifier swallows the direct-call failure -> verdict None (skip),
    # never an exception out of the bench.
    assert all(t["verdict"] is None for t in trials)


# ---------- report rendering ----------


def test_render_report_md_has_headline_and_table():
    results = [
        _tr("fact_swap", "REFUTED", "REFUTED", conf=0.9),
        _tr("clean", "CONFIRMED", "CONFIRMED", conf=0.9),
    ]
    report = {"n_cases": 1, "n_trials": 2, "seed": 0,
              "actionable_conf": ACTIONABLE_CONF,
              "arms": {"two_stage_on": {
                  "metrics": score_trials(results),
                  "trials": [r.to_dict() for r in results]}}}
    md = render_report_md(report)
    assert "TPR (catch rate)" in md
    assert "| fact_swap | REFUTED |" in md
    assert "## two_stage_on" in md
