# src/ghost_agent/eval/verify_bench.py
"""Verifier fault-injection calibration bench.

Measures the verifier's REAL catch rate instead of assuming it: take
known-good (claim, evidence, context) triples, inject a known corruption
from a typed fault library, run each variant through
``Verifier.verify_claim`` against a live judge model, and score verdicts
against the expectation the fault class defines. Clean cases measure the
false-positive rate; corrupted cases measure the true-positive (catch)
rate; evidence-degradation cases measure robustness against punishing a
correct claim for pipeline noise (the 2026-07-17 evidence-packer failure
shape).

Methodology borrowed from "Mechanisms of Introspective Awareness"
(arXiv:2603.21396): inject a KNOWN anomaly, then report detection at a
controlled false-positive rate — a detector is only meaningful with both
numbers. "Actionable" rates use the same confidence gate (>= 0.7) as the
production repair/consumption sites in agent.py, so the bench reports
what the agent would actually have acted on.

Case sources:
  - a hand-authored seed set (scripts/verify_bench_cases.jsonl), so the
    bench runs before any recordings exist;
  - GHOST_LLM_RECORD day-files: every recorded VERIFY call embeds the
    rendered claim prompt, and ``extract_cases_from_recordings`` parses
    the (claim, evidence, context) triple back out of it using the live
    template's own literal segments — real production turns become bench
    cases for free.

Offline by design on the data side; the ONE thing it talks to is the
judge model endpoint (the system under test). See scripts/verify_bench.py
for the CLI runner.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..core.verifier import _VERIFY_CLAIM_PROMPT, Verifier

logger = logging.getLogger("GhostAgent")

# Mirrors the production consumption gates (agent.py in-loop AUTO-REPAIR
# and the post-loop verdict gate both act only on confidence >= 0.7). A
# REFUTED below this is visible in logs but changes nothing the user sees.
ACTIONABLE_CONF = 0.7

# Literal segments of the live claim template, split around its
# placeholders. Parsing rendered prompts with these (instead of
# hand-copied header strings) keeps extraction correct automatically when
# the template text changes. The {{ }} format-escapes (the JSON schema at
# the template's tail) must be unescaped to match RENDERED text.
def _unescape(seg: str) -> str:
    return seg.replace("{{", "{").replace("}}", "}")


_PRE_CLAIM, _rest = _VERIFY_CLAIM_PROMPT.split("{claim}")
_PRE_EVIDENCE, _rest2 = _rest.split("{evidence}")
_PRE_CONTEXT, _POST_CONTEXT = _rest2.split("{context}")
_PRE_CLAIM = _unescape(_PRE_CLAIM)
_PRE_EVIDENCE = _unescape(_PRE_EVIDENCE)
_PRE_CONTEXT = _unescape(_PRE_CONTEXT)
_POST_CONTEXT = _unescape(_POST_CONTEXT)


# ── Cases ────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    """One known-good (claim, evidence, context) triple."""
    case_id: str
    claim: str
    evidence: str
    context: str
    source: str = "seed"       # "seed" | "recording"
    notes: str = ""


@dataclass
class BenchTrial:
    """One verify call to make: a case, possibly corrupted by a fault."""
    case_id: str
    fault: str                 # "clean" or a fault-library name
    expected: str              # "CONFIRMED" | "REFUTED" | "NOT_REFUTED"
    claim: str
    evidence: str
    context: str
    note: str = ""


@dataclass
class TrialResult:
    trial: BenchTrial
    verdict: Optional[str]     # None => verifier skipped / unparseable
    confidence: float = 0.0
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    suspects: Optional[List[Dict[str, str]]] = None
    elapsed_s: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.trial.case_id,
            "fault": self.trial.fault,
            "expected": self.trial.expected,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "suspects": self.suspects,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "note": self.trial.note,
        }


def load_cases_jsonl(path: str | Path) -> List[BenchCase]:
    """Load seed cases. Keys: id/case_id, claim, evidence, context,
    notes (optional). Lines that fail to parse or lack a claim+evidence
    are skipped with a warning — a bad line must not kill the bench."""
    cases: List[BenchCase] = []
    for lineno, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("verify_bench: %s:%d unparseable (%s)",
                           path, lineno, exc)
            continue
        claim = str(obj.get("claim") or "")
        evidence = str(obj.get("evidence") or "")
        if not claim.strip() or not evidence.strip():
            logger.warning("verify_bench: %s:%d missing claim/evidence",
                           path, lineno)
            continue
        cases.append(BenchCase(
            case_id=str(obj.get("case_id") or obj.get("id")
                        or f"seed-{lineno}"),
            claim=claim,
            evidence=evidence,
            context=str(obj.get("context") or ""),
            source="seed",
            notes=str(obj.get("notes") or ""),
        ))
    return cases


def parse_rendered_claim_prompt(text: str) -> Optional[Dict[str, str]]:
    """Invert ``_VERIFY_CLAIM_PROMPT.format(...)``: given the rendered
    prompt of a recorded VERIFY call, recover claim/evidence/context.
    Returns None when the text is not a rendered claim prompt."""
    if not text or not text.startswith(_PRE_CLAIM):
        return None
    body = text[len(_PRE_CLAIM):]
    i = body.find(_PRE_EVIDENCE)
    if i < 0:
        return None
    claim = body[:i]
    body = body[i + len(_PRE_EVIDENCE):]
    j = body.find(_PRE_CONTEXT)
    if j < 0:
        return None
    evidence = body[:j]
    body = body[j + len(_PRE_CONTEXT):]
    k = body.rfind(_POST_CONTEXT)
    if k < 0:
        return None
    return {"claim": claim, "evidence": evidence, "context": body[:k]}


def extract_cases_from_recordings(
        paths: Iterable[str | Path]) -> List[BenchCase]:
    """Mint bench cases out of GHOST_LLM_RECORD day-files: any recorded
    exchange whose first message is a rendered claim prompt yields one
    (claim, evidence, context) triple. Deduplicates on content hash.

    NOTE: these come from real turns, so the claim is what the agent
    actually said — treat them as *presumed-good* cases. A turn the
    verifier refuted in production will look like a clean-case false
    positive here; prune those from the seed set by hand if they recur.
    """
    seen: set = set()
    cases: List[BenchCase] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("verify_bench: cannot read %s: %s", p, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = (rec.get("payload") or {}).get("messages") or []
            if not msgs:
                continue
            parsed = parse_rendered_claim_prompt(
                str(msgs[0].get("content") or ""))
            if not parsed or not parsed["claim"].strip() \
                    or not parsed["evidence"].strip():
                continue
            digest = hashlib.sha256(
                (parsed["claim"] + "\x00" + parsed["evidence"] + "\x00"
                 + parsed["context"]).encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            cases.append(BenchCase(
                case_id=f"rec-{digest[:10]}",
                claim=parsed["claim"],
                evidence=parsed["evidence"],
                context=parsed["context"],
                source="recording",
                notes=str(rec.get("ts") or ""),
            ))
    return cases


# ── Fault library ────────────────────────────────────────────────────
#
# A fault takes (case, rng, pool) and returns a corrupted
# (claim, evidence, context, note) tuple, or None when it does not apply
# to this case (no shared number to swap, no donor claim, ...). Each
# fault carries the verdict it SHOULD produce:
#   "REFUTED"     — the corruption must be caught (TPR side)
#   "NOT_REFUTED" — the claim is still true; refuting it is a false
#                   alarm (robustness side). CONFIRMED and UNCERTAIN
#                   both count as correct here.

_NUM_RE = re.compile(r"\d[\d,.]*\d|\d")


def _mutate_number(tok: str) -> str:
    """Deterministically perturb a numeric token so it contradicts the
    evidence but keeps its shape (digit count, separators)."""
    digits = [c for c in tok if c.isdigit()]
    # Bump the last digit; if that wraps to the same token (single "9"),
    # bump the first too.
    out = list(tok)
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].isdigit():
            out[idx] = str((int(out[idx]) + 1) % 10)
            break
    mutated = "".join(out)
    if mutated == tok and digits:
        for idx in range(len(out)):
            if out[idx].isdigit():
                out[idx] = str((int(out[idx]) + 3) % 10)
                break
        mutated = "".join(out)
    return mutated


def fault_fact_swap(case: BenchCase, rng: random.Random,
                    pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Change a number that the claim shares with the evidence, so the
    claim now contradicts its own source."""
    claim_nums = _NUM_RE.findall(case.claim)
    shared = sorted({t for t in claim_nums
                     if len(t) >= 2 and t in case.evidence})
    if not shared:
        return None
    target = rng.choice(shared)
    mutated = _mutate_number(target)
    if mutated == target:
        return None
    new_claim = case.claim.replace(target, mutated, 1)
    return (new_claim, case.evidence, case.context,
            f"swapped {target!r} -> {mutated!r}")


def fault_fabrication(case: BenchCase, rng: random.Random,
                      pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Append a concrete, plausible-sounding fact that appears in no
    tool output — the rubric's canonical fabrication."""
    fabrications = [
        "The figure was independently confirmed by the Karlsen "
        "Institute's annual audit on 12 March 2019.",
        "It also won the Meridian Prize for this category in 2021.",
        "The lead maintainer, Dr. Elin Vasquez, verified the result "
        "earlier today.",
    ]
    usable = [f for f in fabrications
              if "Karlsen" not in case.evidence
              and f.split()[2] not in case.evidence]
    if not usable:
        return None
    fab = rng.choice(usable)
    return (case.claim.rstrip() + " " + fab, case.evidence, case.context,
            f"appended fabrication: {fab[:60]}")


def fault_wrong_topic(case: BenchCase, rng: random.Random,
                      pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Answer a different question: swap in another case's claim wholesale.
    Needs a non-empty user request (the alignment check is skipped when
    the request is blank) and a donor case."""
    if not case.context.strip():
        return None
    donors = [c for c in pool
              if c.case_id != case.case_id and c.claim != case.claim]
    if not donors:
        return None
    donor = rng.choice(donors)
    return (donor.claim, case.evidence, case.context,
            f"claim replaced with {donor.case_id}'s")


def fault_silent_failure(case: BenchCase, rng: random.Random,
                         pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """The tool actually failed but the claim still reports success:
    replace the evidence with an error body, keep the claim."""
    m = re.match(r"\[([^\]\n]{1,40})\]", case.evidence)
    tool = m.group(1) if m else "web_search"
    failures = [
        f"[{tool}] ERROR: HTTP 403 Forbidden — request blocked",
        f"[{tool}] (empty output)",
        f"[{tool}] Traceback (most recent call last):\n  ...\n"
        f"TimeoutError: request timed out after 30s",
    ]
    return (case.claim, rng.choice(failures), case.context,
            "evidence replaced with tool failure")


def fault_artifact_leak(case: BenchCase, rng: random.Random,
                        pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Inject machine noise into the reply — the historical
    SEARCH/REPLACE marker-leak shape (2026-07-14)."""
    noise = "\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
    mid = len(case.claim) // 2
    cut = case.claim.rfind(" ", 0, mid)
    cut = cut if cut > 0 else mid
    new_claim = case.claim[:cut] + noise + case.claim[cut:]
    return (new_claim, case.evidence, case.context,
            "diff markers injected into reply")


def fault_constraint_violation(case: BenchCase, rng: random.Random,
                               pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Add an explicit one-word format constraint to the request that the
    (unchanged, long) reply clearly violates."""
    if len(case.claim.split()) <= 8 or not case.context.strip():
        return None
    new_context = case.context.rstrip() + " Answer with a single word only."
    return (case.claim, case.evidence, new_context,
            "request now demands a single word")


def fault_evidence_truncation(case: BenchCase, rng: random.Random,
                              pool: List[BenchCase]) -> Optional[Tuple[str, str, str, str]]:
    """Degrade the EVIDENCE, not the claim: cut it hard mid-stream. The
    claim is still true — refuting it for pipeline noise is the
    evidence-packer failure shape (spurious REFUTE on truncated output,
    fixed 2026-07-17). Expected NOT_REFUTED."""
    if len(case.evidence) < 200:
        return None
    keep = max(80, int(len(case.evidence) * 0.35))
    return (case.claim, case.evidence[:keep] + " …[truncated]",
            case.context, "evidence hard-truncated, claim still true")


# name -> (expected verdict, fault fn)
FAULTS: Dict[str, Tuple[str, Callable]] = {
    "fact_swap": ("REFUTED", fault_fact_swap),
    "fabrication": ("REFUTED", fault_fabrication),
    "wrong_topic": ("REFUTED", fault_wrong_topic),
    "silent_failure": ("REFUTED", fault_silent_failure),
    "artifact_leak": ("REFUTED", fault_artifact_leak),
    "constraint_violation": ("REFUTED", fault_constraint_violation),
    "evidence_truncation": ("NOT_REFUTED", fault_evidence_truncation),
}


def build_trials(cases: List[BenchCase],
                 fault_names: Optional[List[str]] = None,
                 seed: int = 0,
                 include_clean: bool = True) -> List[BenchTrial]:
    """Expand cases into the trial list: one clean trial per case plus
    one trial per applicable fault. Deterministic for a given seed."""
    rng = random.Random(seed)
    names = list(fault_names) if fault_names else list(FAULTS)
    unknown = [n for n in names if n not in FAULTS]
    if unknown:
        raise ValueError(f"unknown fault(s): {unknown}")
    trials: List[BenchTrial] = []
    for case in cases:
        if include_clean:
            trials.append(BenchTrial(
                case_id=case.case_id, fault="clean", expected="CONFIRMED",
                claim=case.claim, evidence=case.evidence,
                context=case.context))
        for name in names:
            expected, fn = FAULTS[name]
            out = fn(case, rng, cases)
            if out is None:
                continue
            claim, evidence, context, note = out
            trials.append(BenchTrial(
                case_id=case.case_id, fault=name, expected=expected,
                claim=claim, evidence=evidence, context=context,
                note=note))
    return trials


# ── Scoring ──────────────────────────────────────────────────────────

def _rate(num: int, denom: int) -> Optional[float]:
    return round(num / denom, 3) if denom else None


def score_trials(results: List[TrialResult]) -> Dict[str, Any]:
    """Aggregate trial results into per-fault and overall metrics.

    Rates are computed over JUDGED trials (verdict produced); skips and
    errors are reported alongside, never silently folded into a rate.
    """
    by_fault: Dict[str, List[TrialResult]] = {}
    for r in results:
        by_fault.setdefault(r.trial.fault, []).append(r)

    per_fault: Dict[str, Any] = {}
    for fault, rs in sorted(by_fault.items()):
        expected = rs[0].trial.expected
        skipped = [r for r in rs if r.verdict is None]
        judged = [r for r in rs if r.verdict is not None]
        refuted = [r for r in judged if r.verdict == "REFUTED"]
        refuted_act = [r for r in refuted if r.confidence >= ACTIONABLE_CONF]
        confirmed = [r for r in judged if r.verdict == "CONFIRMED"]
        uncertain = [r for r in judged if r.verdict == "UNCERTAIN"]
        entry: Dict[str, Any] = {
            "expected": expected,
            "n": len(rs),
            "judged": len(judged),
            "skipped": len(skipped),
            "confirmed": len(confirmed),
            "refuted": len(refuted),
            "uncertain": len(uncertain),
            "mean_confidence": round(
                sum(r.confidence for r in judged) / len(judged), 3)
                if judged else None,
        }
        if expected == "REFUTED":
            entry["catch_rate"] = _rate(len(refuted), len(judged))
            entry["catch_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
        elif expected == "CONFIRMED":
            entry["false_alarm_rate"] = _rate(len(refuted), len(judged))
            entry["false_alarm_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
            entry["confirm_rate"] = _rate(len(confirmed), len(judged))
        else:  # NOT_REFUTED
            entry["false_alarm_rate"] = _rate(len(refuted), len(judged))
            entry["false_alarm_rate_actionable"] = _rate(
                len(refuted_act), len(judged))
        per_fault[fault] = entry

    def _overall(expected_values: Tuple[str, ...],
                 positive: str) -> Dict[str, Any]:
        rs = [r for r in results if r.trial.expected in expected_values]
        judged = [r for r in rs if r.verdict is not None]
        hit = [r for r in judged if r.verdict == positive]
        hit_act = [r for r in hit if r.confidence >= ACTIONABLE_CONF]
        return {
            "n": len(rs), "judged": len(judged),
            "rate": _rate(len(hit), len(judged)),
            "rate_actionable": _rate(len(hit_act), len(judged)),
        }

    return {
        "per_fault": per_fault,
        "overall": {
            # TPR: corrupted trials refuted.
            "tpr": _overall(("REFUTED",), "REFUTED"),
            # FPR: clean trials refuted.
            "fpr": _overall(("CONFIRMED",), "REFUTED"),
            # Robustness false alarms: degraded-evidence trials refuted.
            "degraded_evidence_fp": _overall(("NOT_REFUTED",), "REFUTED"),
        },
    }


# ── Runner ───────────────────────────────────────────────────────────

class HttpChatClient:
    """Minimal OpenAI-compatible chat client duck-typing the single
    surface Verifier needs when no critic pool / router is present:
    ``chat_completion(payload)``. Deliberately defines neither
    ``critic_clients`` truthy nor ``route``, so ``Verifier._call_llm``
    goes straight to the direct path against the given endpoint."""

    critic_clients: Any = None

    def __init__(self, base_url: str, timeout: float = 90.0,
                 api_key: str = "", model: str = ""):
        import httpx
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout,
            headers=headers)

    async def chat_completion(self, payload: dict, **_kw) -> dict:
        body = dict(payload)
        if self._model:
            body.setdefault("model", self._model)
        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()


ARM_ENV = {"two_stage_on": "1", "two_stage_off": "0"}


async def run_trials(verifier: Verifier, trials: List[BenchTrial],
                     concurrency: int = 1,
                     on_result: Optional[Callable] = None
                     ) -> List[TrialResult]:
    """Run every trial through ``verifier.verify_claim``. Exceptions are
    captured per-trial (error + verdict None), never raised out."""
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[TrialResult]] = [None] * len(trials)

    async def _one(i: int, t: BenchTrial) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                vr = await verifier.verify_claim(
                    t.claim, t.evidence, t.context)
                res = TrialResult(
                    trial=t,
                    verdict=vr.verdict.value if vr else None,
                    confidence=vr.confidence if vr else 0.0,
                    reasoning=vr.reasoning if vr else "",
                    issues=list(vr.issues) if vr else [],
                    suspects=vr.suspects if vr else None,
                    elapsed_s=time.monotonic() - t0,
                )
            except Exception as exc:
                res = TrialResult(
                    trial=t, verdict=None,
                    elapsed_s=time.monotonic() - t0, error=str(exc))
            results[i] = res
            if on_result:
                on_result(res)

    await asyncio.gather(*(_one(i, t) for i, t in enumerate(trials)))
    return [r for r in results if r is not None]


async def run_bench(cases: List[BenchCase],
                    verifier: Verifier,
                    arms: Optional[List[str]] = None,
                    fault_names: Optional[List[str]] = None,
                    seed: int = 0,
                    concurrency: int = 1,
                    on_result: Optional[Callable] = None) -> Dict[str, Any]:
    """Full bench: build trials once, run them under each arm
    (two-stage on / off, toggled via GHOST_VERIFY_TWO_STAGE, which the
    verifier reads per call), score, and return the report dict."""
    arms = arms or ["two_stage_on", "two_stage_off"]
    unknown = [a for a in arms if a not in ARM_ENV]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")
    trials = build_trials(cases, fault_names=fault_names, seed=seed)
    report: Dict[str, Any] = {
        "n_cases": len(cases),
        "n_trials": len(trials),
        "seed": seed,
        "actionable_conf": ACTIONABLE_CONF,
        "arms": {},
    }
    prev = os.environ.get("GHOST_VERIFY_TWO_STAGE")
    try:
        for arm in arms:
            os.environ["GHOST_VERIFY_TWO_STAGE"] = ARM_ENV[arm]
            results = await run_trials(
                verifier, trials, concurrency=concurrency,
                on_result=on_result)
            report["arms"][arm] = {
                "metrics": score_trials(results),
                "trials": [r.to_dict() for r in results],
            }
    finally:
        if prev is None:
            os.environ.pop("GHOST_VERIFY_TWO_STAGE", None)
        else:
            os.environ["GHOST_VERIFY_TWO_STAGE"] = prev
    return report


def render_report_md(report: Dict[str, Any]) -> str:
    """Human-readable summary: one table per arm plus the headline
    TPR/FPR line — paste-able into PROJECT_JOURNAL.md."""
    lines = [
        "# Verifier fault-injection bench",
        "",
        f"cases: {report['n_cases']} · trials/arm: {report['n_trials']}"
        f" · seed: {report['seed']}"
        f" · actionable conf ≥ {report['actionable_conf']}",
        "",
    ]
    for arm, data in report["arms"].items():
        m = data["metrics"]
        o = m["overall"]
        lines.append(f"## {arm}")
        lines.append("")
        lines.append(
            f"**TPR (catch rate)**: {o['tpr']['rate']} raw / "
            f"{o['tpr']['rate_actionable']} actionable "
            f"({o['tpr']['judged']}/{o['tpr']['n']} judged) — "
            f"**FPR (clean refuted)**: {o['fpr']['rate']} raw / "
            f"{o['fpr']['rate_actionable']} actionable — "
            f"**degraded-evidence FP**: "
            f"{o['degraded_evidence_fp']['rate']}")
        lines.append("")
        lines.append("| fault | expected | n | judged | skipped | "
                     "confirmed | refuted | uncertain | rate | "
                     "actionable | mean conf |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for fault, e in m["per_fault"].items():
            rate = e.get("catch_rate", e.get("false_alarm_rate"))
            rate_act = e.get("catch_rate_actionable",
                             e.get("false_alarm_rate_actionable"))
            lines.append(
                f"| {fault} | {e['expected']} | {e['n']} | {e['judged']} "
                f"| {e['skipped']} | {e['confirmed']} | {e['refuted']} "
                f"| {e['uncertain']} | {rate} | {rate_act} "
                f"| {e['mean_confidence']} |")
        lines.append("")
    return "\n".join(lines)
