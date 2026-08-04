"""verify_bench case pool: coverage, poisoning, determinism, provenance.

Before these fixes the bench was, in the reviewers' words, "measuring a
materially different system than production": it harvested only the 9% of
verify traffic that took the two-stage FALLBACK path, admitted turns
production had REFUTED as clean CONFIRMED cases, rewrote existing trials
whenever the pool grew, and recorded no provenance — so a T0/T+7d comparison
could silently be comparing two different benchmarks.
"""
from __future__ import annotations

import json

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.eval import verify_bench as vb


def _record(prompt: str, request_id: str, response_text: str = "") -> str:
    return json.dumps({
        "request_id": request_id,
        "payload": {"messages": [{"role": "user", "content": prompt}]},
        "response": {"choices": [{"message": {"content": response_text}}]},
        "ts": "2026-08-04T00:00:00Z",
    })


def _render(template: str, claim: str, evidence: str, context: str) -> str:
    return template.format(claim=claim, evidence=evidence, context=context)


def _adjudicate(claim: str) -> str:
    """A REAL rendered adjudicate prompt. The verdict join is keyed on the
    CLAIM the prompt carries, not on request_id — a bare "adjudicate" string
    carries no claim and so joins to nothing."""
    return V._VERIFY_ADJUDICATE_PROMPT.format(
        claim=claim, evidence="[tool] ev", context="ctx", suspects="[]")


def test_the_enumerate_prompt_is_invertible_not_just_the_fallback():
    """281 of 618 recorded verify calls are enumerate; 55 are classic — and
    classic is the FALLBACK, so harvesting only it selected cases on
    two-stage failure."""
    rendered = _render(V._VERIFY_ENUMERATE_PROMPT, "The total was 42.",
                       "[tool] total=42", "user asked for the total")
    got = vb.parse_rendered_claim_prompt(rendered)
    assert got is not None, "enumerate prompts must invert"
    assert got["claim"].strip() == "The total was 42."
    assert got["evidence"].strip() == "[tool] total=42"


def test_the_classic_prompt_still_inverts():
    rendered = _render(V._VERIFY_CLAIM_PROMPT, "The total was 42.",
                       "[tool] total=42", "user asked for the total")
    got = vb.parse_rendered_claim_prompt(rendered)
    assert got is not None and got["claim"].strip() == "The total was 42."


def test_escaped_braces_do_not_break_inversion():
    """The templates carry `{{`/`}}` for their JSON output spec, which
    `.format()` collapses. Matching raw segments against a rendered prompt
    failed ~800 chars in — 0 of 580 live records inverted, while the opening
    sentence matched perfectly."""
    for tpl in (V._VERIFY_CLAIM_PROMPT, V._VERIFY_ENUMERATE_PROMPT):
        parts = vb._split_template(tpl)
        assert parts is not None
        assert "{{" not in parts[3], "template tail must be unescaped"


def test_turns_production_refuted_are_not_admitted_as_clean_cases(tmp_path):
    """The poisoning case: a REFUTED turn entering as expected=CONFIRMED
    inflates measured FPR and steers the optimizer toward a MORE PERMISSIVE
    judge — the bench rewarding the failure it exists to detect. Measured on
    the live corpus: 62 of 106 candidate cases (58.5%)."""
    good = _render(V._VERIFY_ENUMERATE_PROMPT, "Clean claim here.",
                   "[tool] supports it", "ctx")
    bad = _render(V._VERIFY_ENUMERATE_PROMPT, "Refuted claim here.",
                  "[tool] contradicts it", "ctx")
    day = tmp_path / "2026-08-04.jsonl"
    day.write_text("\n".join([
        _record(good, "req-good"),
        _record(_adjudicate("Clean claim here."), "req-good",
                '{"verdict": "CONFIRMED"}'),
        _record(bad, "req-bad"),
        _record(_adjudicate("Refuted claim here."), "req-bad",
                '{"verdict": "REFUTED"}'),
    ]) + "\n")

    kept = vb.extract_cases_from_recordings([day])
    claims = {c.claim.strip() for c in kept}
    assert "Clean claim here." in claims
    assert "Refuted claim here." not in claims, (
        "a turn production REFUTED is not a clean case")

    unfiltered = vb.extract_cases_from_recordings([day], exclude_refuted=False)
    assert len(unfiltered) == 2, "the override must still admit both"


def test_the_production_verdict_is_recorded_on_the_case(tmp_path):
    good = _render(V._VERIFY_ENUMERATE_PROMPT, "A claim.", "[tool] ev", "ctx")
    day = tmp_path / "d.jsonl"
    day.write_text("\n".join([
        _record(good, "r1"),
        _record(_adjudicate("A claim."), "r1",
                '{"verdict": "CONFIRMED"}')]) + "\n")
    case = vb.extract_cases_from_recordings([day])[0]
    assert "prod_verdict=CONFIRMED" in case.notes


def test_mined_cases_are_redacted(tmp_path):
    """Recording day-files are unredacted by design; a bench case file is a
    durable artifact that gets copied into ablation bundles."""
    secret = "reach me at 555 867 5309 or 4111 1111 1111 1111"
    rendered = _render(V._VERIFY_ENUMERATE_PROMPT, secret, "[tool] ev", "ctx")
    day = tmp_path / "d.jsonl"
    day.write_text(_record(rendered, "r1") + "\n")
    case = vb.extract_cases_from_recordings([day], exclude_refuted=False)[0]
    assert "4111 1111 1111 1111" not in case.claim


# ── determinism ───────────────────────────────────────────────────────

def _case(i: int) -> vb.BenchCase:
    return vb.BenchCase(
        case_id=f"c{i}", claim=f"The total was {i * 7} units and it finished.",
        evidence=f"[tool] total={i * 7} units; status=ok",
        context=f"user asked about job {i}")


def test_growing_the_pool_does_not_rewrite_existing_trials():
    """A single shared rng made every trial depend on how many cases preceded
    it: re-running the identical command after the pool grew changed 20% of
    the original trials while still looking like the same benchmark."""
    small = [_case(i) for i in range(1, 8)]
    big = small + [_case(i) for i in range(8, 40)]
    a = {(t.case_id, t.fault): (t.claim, t.evidence)
         for t in vb.build_trials(small, seed=0)}
    b = {(t.case_id, t.fault): (t.claim, t.evidence)
         for t in vb.build_trials(big, seed=0)}
    shared = set(a) & set(b)
    changed = {k[1] for k in shared if a[k] != b[k]}
    assert shared, "the two runs must share trials"
    # wrong_topic draws a donor FROM the pool, so it alone may move.
    assert changed <= {"wrong_topic"}, f"unexpected drift in {changed}"


def test_trials_are_reproducible_for_identical_input():
    cases = [_case(i) for i in range(1, 10)]
    a = [(t.case_id, t.fault, t.claim) for t in vb.build_trials(cases, seed=0)]
    b = [(t.case_id, t.fault, t.claim) for t in vb.build_trials(cases, seed=0)]
    assert a == b


def test_a_different_seed_changes_the_trials():
    cases = [_case(i) for i in range(1, 10)]
    a = [(t.case_id, t.fault, t.claim) for t in vb.build_trials(cases, seed=0)]
    b = [(t.case_id, t.fault, t.claim) for t in vb.build_trials(cases, seed=99)]
    assert a != b


# ── provenance ────────────────────────────────────────────────────────

def test_provenance_pins_what_the_run_measured():
    cases = [_case(i) for i in range(1, 5)]
    prov = vb.bench_provenance(cases)
    assert prov["n_cases"] == 4
    assert len(prov["cases_sha256"]) == 16
    assert "ghost_home" in prov
    assert "verifier.enumerate" in prov["templates"]
    assert "sha256" in prov["templates"]["verifier.enumerate"]


def test_provenance_changes_when_the_pool_changes():
    """Two reports must be comparable only when built from the same pool."""
    a = vb.bench_provenance([_case(i) for i in range(1, 5)])
    b = vb.bench_provenance([_case(i) for i in range(1, 6)])
    assert a["cases_sha256"] != b["cases_sha256"]


def test_provenance_is_order_independent():
    cases = [_case(i) for i in range(1, 6)]
    assert (vb.bench_provenance(cases)["cases_sha256"]
            == vb.bench_provenance(list(reversed(cases)))["cases_sha256"])


def test_an_escalated_refutation_does_not_disqualify_the_turn(tmp_path):
    """`['REFUTED', 'CONFIRMED']` inside one request is the `_escalate_refute`
    SIGNATURE: the cheap judge false-refutes and the main model overturns it.
    Production did NOT refute that claim, so it is a clean case — and it is
    among the BEST available, because it survived two judges.

    A request-level join called it refuted and dropped it. Measured on the
    live corpus: 177 of 240 exclusions were this, 44 of 59 multi-verdict
    requests show the pattern, and the survivors were biased toward turns
    that never triggered a refutation anywhere.
    """
    good = _render(V._VERIFY_ENUMERATE_PROMPT, "Escalated claim.",
                   "[tool] supports it", "ctx")
    day = tmp_path / "d.jsonl"
    day.write_text("\n".join([
        _record(good, "req-1"),
        _record(_adjudicate("Escalated claim."), "req-1",
                '{"verdict": "REFUTED"}'),      # cheap judge
        _record(_adjudicate("Escalated claim."), "req-1",
                '{"verdict": "CONFIRMED"}'),    # main model overturns
    ]) + "\n")

    kept = vb.extract_cases_from_recordings([day])
    assert [c.claim.strip() for c in kept] == ["Escalated claim."]
    assert "prod_verdict=CONFIRMED" in kept[0].notes


def test_a_verdict_about_a_DIFFERENT_claim_does_not_disqualify_this_one(tmp_path):
    """`request_id` is per-TURN — 346 distinct ids over 11,929 live records,
    one of them (`"SYSTEM"`) holding 9,677. So a request-level join let the
    code auditor's verdict about an unrelated claim, or one REFUTED anywhere
    in the shared bucket, poison every case in it."""
    mine = _render(V._VERIFY_ENUMERATE_PROMPT, "My claim.", "[tool] ev", "ctx")
    day = tmp_path / "d.jsonl"
    day.write_text("\n".join([
        _record(mine, "SYSTEM"),
        _record(_adjudicate("My claim."), "SYSTEM",
                '{"verdict": "CONFIRMED"}'),
        _record(_adjudicate("A totally unrelated claim."), "SYSTEM",
                '{"verdict": "REFUTED"}'),
    ]) + "\n")

    kept = vb.extract_cases_from_recordings([day])
    assert [c.claim.strip() for c in kept] == ["My claim."]


def test_the_seed_varies_the_wrong_topic_donor(tmp_path):
    """`--seed` is recorded in the report as if it parameterised the whole
    trial set, but the wrong_topic donor was a pure hash of the case-id pair,
    so seed 0 and seed 999 produced identical trials for that fault."""
    cases = [vb.BenchCase(case_id=f"c{i}", claim=f"claim {i}",
                          evidence=f"[tool] ev {i}", context="a user request")
             for i in range(8)]
    def donors(seed):
        return {t.case_id: t.claim
                for t in vb.build_trials(cases, fault_names=["wrong_topic"],
                                         seed=seed, include_clean=False)}
    assert donors(0) != donors(999), "the seed must vary this fault too"
    assert donors(0) == donors(0), "and must stay deterministic within a seed"
