"""§4JE (2026-09-20): two mechanisms the night re-bench exposed.

A. The adjudicate template had lost its artifact-scan clause ("Additionally,
   scan the raw CLAIM for unflagged formatting artifacts …"); the cheap
   judge's artifact_leak catch fell 48/58 → 26/58 before any escalation.
B. A strong UNCERTAIN born of a §4IJ concession that names the SAME figures
   the cheap judge refuted on replaced the refute ("minor rounding") — 27
   lost catches on refute-expected faults, every fact_swap one a last-digit
   swap (3.842 → 3.9, 12.41 → 12.5, 31 → 32). The binder's arithmetic now
   arbitrates: uphold when the claim's figure does not agree with the
   evidence's at the claim's precision; rounding, units and hedges agree.

World where these fail: the tree of 2026-09-20 18:39 (the night run).
"""
import json

import pytest

from ghost_agent.core.verifier import (
    Verifier, VerifyResult, VerifyVerdict, conceded_figures_disagree,
    _stage_template, _VERIFY_ADJUDICATE_PROMPT,
)


# ── A. the artifact-scan clause is part of what the judge is TOLD ────────

def test_adjudicate_prompt_tells_the_judge_to_scan_for_artifacts():
    """Rendered through the real template path (not a source read): the
    instruction the September edits dropped, measured at −22 artifact
    catches on the bench."""
    prompt = _stage_template("verifier.adjudicate", _VERIFY_ADJUDICATE_PROMPT).format(
        claim="c", evidence="e", context="x", suspects="1. [artifact] noise")
    assert "scan the raw CLAIM for unflagged formatting artifacts" in prompt
    assert "<<<<<<<" in prompt and ">>>>>>>" in prompt


# ── B. the corroboration helper ──────────────────────────────────────────

@pytest.mark.parametrize("claim, issue, conceded, evidence", [
    ("Temperature is 32°C.", "32°C stated, evidence 31°C", "32°C vs 31°C, minor", "temp: 31°C"),
    ("The suite ran in 12.5s.", "claims 12.5 seconds, tool recorded 12.41s", "12.41s vs 12.5s is rounding", "12.41s elapsed"),
    ("Average RTT 3.9 ms over 5 packets.", "3.9 ms stated; evidence 3.842 ms", "rounds from 3.842 ms to 3.9 ms, valid", "5 packets, avg 3.842 ms"),
])
def test_a_last_digit_swap_the_strong_model_conceded_is_upheld(claim, issue, conceded, evidence):
    assert conceded_figures_disagree(claim, [issue], [conceded], evidence=evidence)


@pytest.mark.parametrize("claim, issue, conceded, evidence, why", [
    ("Iceland has about 396,000 people.", "says 396,000, evidence 396,960", "about 396,000 vs 396,960 is rounding", "396,960", "hedged claim — agrees"),
    ("The file is 48 KB.", "48 KB vs 49152 bytes", "48 KB vs 49152 bytes, unit conversion", "49152 bytes", "unit conversion — agrees"),
    ("Restarted at 14:03.", "restarted 14:02:11 not 14:03", "14:03 vs 14:02:11", "14:02:11 restart", "clock times are not quantities"),
    ("Newest stable is 7.1.6.", "stable is 7.1.5, not 7.1.6", "evidence says 7.1.5, claim says 7.1.6", "stable 7.1.5", "dotted versions are not quantities"),
    ("The report lists 3 warnings.", "the file was deleted; evidence shows it exists", "timestamp 14:03 vs 14:02:11 minor", "3 WARN", "a DIFFERENT discrepancy"),
    ("Temperature is 32°C.", "32°C stated, evidence 31°C", "", "31°C", "no concession at all"),
    ("Temperature is 32°C.", "", "32°C vs 31°C", "31°C", "no cheap issue to corroborate"),
])
def test_the_helper_stays_silent_where_the_binder_agrees_or_cannot_see(claim, issue, conceded, evidence, why):
    assert conceded_figures_disagree(claim, [issue] if issue else [], [conceded] if conceded else [], evidence=evidence) is None, why


def test_the_closest_pair_in_the_concession_decides():
    """A context figure ("5 packets", "20/20 checks") shared by both texts
    must not manufacture a disagreement: the pair the concession names
    closest together is the disputed one. Presence of the swapped figure
    elsewhere in the evidence is NOT a disqualifier — that is exactly the
    shape that escapes the objection layer (long-weather-1)."""
    # hedged near-miss + a same-family context figure: the closest pair (12 vs 12.41) agrees → silent
    assert conceded_figures_disagree(
        "5 packets sent, about 12 s total.", ["about 12 s vs 12.41 s; 5 packets"],
        ["5 packets were sent; the total of about 12 s vs 12.41 s is rounding"], evidence="5 packets, 12.41 s") is None
    # a same-family context figure (3 warnings) precedes a hedged near-miss (about 20 vs 19, inside
    # HEDGE_REL_TOL): first-pair semantics would pair 3 vs 19 and manufacture a disagreement;
    # closest-pair pairs 20 vs 19, which the binder accepts
    assert conceded_figures_disagree(
        "3 warnings and about 20 checks passed.", ["3 warnings; claims about 20 checks, evidence 19 checks"],
        ["3 warnings match; about 20 checks vs 19 checks is a rounding"], evidence="3 WARN; 19 checks") is None
    # the swapped 32 also appears elsewhere in the evidence — still upheld
    assert conceded_figures_disagree(
        "Athens is 32°C now.", ["stated 32°C, evidence shows 31°C"],
        ["The claim states 32°C but the evidence states 31°C — minor"],
        evidence="current: 31°C (feels like 33°C); yesterday's high 32°C; humidity 44%") == "32 °c vs 31 °c"


# ── B. executed through the real escalation ─────────────────────────────

class _Stub:
    critic_clients = None
    worker_clients = [object()]

    def __init__(self, responses):
        self.responses = list(responses)

    async def chat_completion(self, payload, **_kw):
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


def _refuted(issues):
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95,
                        reasoning="cheap judge refuted", issues=list(issues))


def _ledger(tmp_path):
    p = tmp_path / "system" / "verifier" / "escalations.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_TIER_ROUTING", "0")
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.delenv("GHOST_VERIFY_CONCESSION_DOWNGRADE", raising=False)
    monkeypatch.delenv("GHOST_VERIFY_ESCALATE_REFUTE", raising=False)


async def test_a_corroborating_concession_upholds_the_cheap_refute(tmp_path):
    """The night's shape: strong CONFIRMED 0.95 that concedes "32°C vs 31°C"
    → §4IJ downgrade to UNCERTAIN → used to REPLACE the refute. Now the
    cheap REFUTED stands, the ledger says upheld with the corroboration."""
    strong = json.dumps({"verdict": "CONFIRMED", "confidence": 0.95,
                         "reasoning": "accurate summary", "issues": [],
                         "conceded": ["The claim states 32°C but the evidence states 31°C — minor"]})
    v = Verifier(llm_client=_Stub([strong]))
    out = await v._escalate_refute(
        _refuted(["The current temperature is stated as 32°C but the evidence shows 31°C."]),
        "Athens is 32°C and sunny.",
        "[web_search] Athens now: 31°C (feels like 33°C), sunny; yesterday's high 32°C; humidity 44%",
        "ctx", trace={"req_id": "t-4je"})
    assert out.verdict == VerifyVerdict.REFUTED
    assert out.escalation_replaced is False and out.escalated_overturn is False
    row = _ledger(tmp_path)[-1]
    assert row["outcome"] == "upheld"
    assert row.get("rebuttal", "").startswith("conceded_corroborates:")


async def test_a_concession_that_the_binder_accepts_still_replaces(tmp_path):
    """The counterweight: "about 396,000" for 396,960 IS a fair rounding —
    the binder agrees, the strong UNCERTAIN replaces the refute as before."""
    strong = json.dumps({"verdict": "CONFIRMED", "confidence": 0.95,
                         "reasoning": "fair summary", "issues": [],
                         "conceded": ["about 396,000 vs the evidence's 396,960 — rounding"]})
    v = Verifier(llm_client=_Stub([strong]))
    out = await v._escalate_refute(
        _refuted(["The claim says 396,000 but the evidence gives 396,960."]),
        "Iceland has about 396,000 people.", "[web_search] population 396,960 (2025)", "ctx",
        trace={"req_id": "t-4je-2"})
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert out.escalation_replaced is True
    assert _ledger(tmp_path)[-1]["outcome"] == "replaced_uncertain"


async def test_an_uncertain_without_a_concession_replaces_as_before(tmp_path):
    """A strong UNCERTAIN that concedes nothing has nothing to corroborate."""
    strong = json.dumps({"verdict": "UNCERTAIN", "confidence": 0.4,
                         "reasoning": "cannot adjudicate", "issues": []})
    v = Verifier(llm_client=_Stub([strong]))
    out = await v._escalate_refute(
        _refuted(["32°C stated, evidence 31°C"]), "Athens is 32°C.", "[tool] 31°C", "ctx",
        trace={"req_id": "t-4je-3"})
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert _ledger(tmp_path)[-1]["outcome"] == "replaced_uncertain"


# ── §4JH (2026-09-21): opaque tokens — versions, clock times, hex identifiers ──
#
# Rule B arbitrates with the binder's arithmetic and so cannot see the night
# bench's remaining replacements: `7.1.5` vs `7.1.6` ("no evidence supports
# 7.1.6" — still UNCERTAIN), `14:02:11` vs `14:03` ("minor rounding"),
# `4cdc5f063a2e` vs `4cdc5f064a2e` ("likely a transcription error"). An exact
# identifier that both judges name as differing IS a disagreement; a clock
# time rounds only at the claim's own precision.

from ghost_agent.core.verifier import conceded_tokens_disagree


@pytest.mark.parametrize("claim, issue, conceded, pair", [
    ("The newest stable kernel is 7.1.6.", "The stable kernel version is 7.1.5, not 7.1.6.",
     "The evidence states the stable version is 7.1.5, but the claim states 7.1.6.", "7.1.6 vs 7.1.5"),
    ("The service restarted at 14:03.", "The service restarted at 14:02:11, not 14:03.",
     "The claim states 14:03 while the log shows 14:02:11; minor rounding.", "14:03 vs 14:02:11"),
    ("The WebOS project id is e4e240b631f6.", "The project id is e4e240b630f6 in the ledger, not e4e240b631f6.",
     "The project ID in the claim e4e240b631f6 differs from the evidence e4e240b630f6, a single-digit difference.", "e4e240b631f6 vs e4e240b630f6"),
    ("Task 4cdc5f064a2e is pending.", "the task id is 4cdc5f063a2e, not 4cdc5f064a2e",
     "The task ID in the claim (4cdc5f064a2e) differs slightly from the evidence (4cdc5f063a2e), likely a typo.", "4cdc5f064a2e vs 4cdc5f063a2e"),
])
def test_an_opaque_token_the_strong_model_conceded_is_upheld(claim, issue, conceded, pair):
    assert conceded_tokens_disagree(claim, [issue], [conceded]) == pair


@pytest.mark.parametrize("claim, issue, conceded, why", [
    ("The service restarted at 14:02.", "restarted at 14:02:11, claim says 14:02", "14:02 vs 14:02:11 is the same minute", "a clock time agrees at the claim's precision"),
    ("The report is dated August 2, 2026 for project e4e240b630f6.", "The date August 2, 2026 is not supported by any tool output.",
     "The project ID e4e240b631f6 vs e4e240b630f6 differs by one digit.", "the concession is about a DIFFERENT discrepancy"),
    ("Version 7.1.5 is stable and 7.1.6 is rc.", "7.1.5 vs 7.1.6 confusion", "7.1.5 and 7.1.6 both appear", "both tokens are stated by the claim"),
    ("Temperature is 32°C.", "32 vs 31", "32°C vs 31°C", "no opaque tokens — the binder's territory"),
    ("RTT 3.9 ms", "3.9 vs 3.842", "3.842 ms rounds to 3.9 ms", "a decimal is not a version"),
    ("Build 12345678 passed.", "12345678 vs 12345679", "12345678 vs 12345679", "digits-only is not a hex id (binder's territory)"),
    ("The newest stable kernel is 7.1.6.", "", "7.1.5 vs 7.1.6", "no cheap issue to corroborate"),
])
def test_the_token_rule_stays_silent_outside_its_families(claim, issue, conceded, why):
    assert conceded_tokens_disagree(claim, [issue] if issue else [], [conceded]) is None, why


async def test_a_conceded_version_swap_upholds_the_cheap_refute(tmp_path):
    """Executed through the real escalation, in the escaping shape (both
    versions present in a long evidence, so the objection layer cannot
    prove absence): strong CONFIRMED conceding "7.1.5 vs 7.1.6" → §4IJ
    UNCERTAIN → used to REPLACE the refute; now upheld."""
    strong = json.dumps({"verdict": "CONFIRMED", "confidence": 0.9,
                         "reasoning": "the agent answered the question", "issues": [],
                         "conceded": ["The evidence states the stable version is 7.1.5, but the claim states 7.1.6 — a minor version slip"]})
    v = Verifier(llm_client=_Stub([strong]))
    out = await v._escalate_refute(
        _refuted(["The stable kernel version is 7.1.5, not 7.1.6."]),
        "The newest stable kernel is 7.1.6.",
        "[web_search] kernel.org: stable 7.1.5 (2026-09-12) · mainline 7.2-rc1 · longterm 6.12.44 · the 7.1.6 stable is queued for review",
        "ctx", trace={"req_id": "t-4jh"})
    assert out.verdict == VerifyVerdict.REFUTED
    row = _ledger(tmp_path)[-1]
    assert row["outcome"] == "upheld" and row.get("rebuttal", "").startswith("conceded_corroborates:")


async def test_a_conceded_same_minute_time_still_replaces(tmp_path):
    """The counterweight: 14:02 for 14:02:11 is the claim's own precision —
    no disagreement, the strong UNCERTAIN replaces as before."""
    strong = json.dumps({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "fine", "issues": [],
                         "conceded": ["The claim says 14:02 while the log shows 14:02:11"]})
    v = Verifier(llm_client=_Stub([strong]))
    out = await v._escalate_refute(
        _refuted(["restarted at 14:02:11, claim says 14:02"]),
        "The service restarted at 14:02.", "[execute] INFO service restarted 14:02:11", "ctx",
        trace={"req_id": "t-4jh-2"})
    assert out.verdict == VerifyVerdict.UNCERTAIN
    assert _ledger(tmp_path)[-1]["outcome"] == "replaced_uncertain"
