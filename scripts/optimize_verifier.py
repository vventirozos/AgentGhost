#!/usr/bin/env python3
"""GEPA optimization of the verifier's two-stage prompts (§4F Phase 2).

Optimizes `verifier.enumerate` + `verifier.adjudicate` as a two-component
candidate using the standalone `gepa` library with a CUSTOM adapter: each
evaluation runs the REAL two-stage pipeline (core.verifier.Verifier) over
fault-injected bench trials (eval.verify_bench) against the judge endpoint
that serves VERIFY in production (the worker node). The metric is verdict
correctness on those trials — the same instrument as scripts/verify_bench.py,
never a toy re-implementation.

Eval hygiene (§4F Phase 0) applies:
  * bench CASES are hash-split PUBLIC/PRIVATE (stable per case_id);
    the optimizer sees only public trials, the ship-gate judges only
    private trials;
  * iterations are clamped to optim.run_gepa.MAX_OPT_ITERATIONS;
  * candidates that lose a format placeholder score 0 with explicit
    feedback (and the ship-gate re-validates before promoting).

Usage:
    PYTHONPATH=src python scripts/optimize_verifier.py \
        --base-url http://nova:8088 \
        --reflection-url http://127.0.0.1:8088 \
        --max-iterations 8

Ships (on private win > --min-delta) to:
    $GHOST_HOME/system/optim/verifier.enumerate.json
    $GHOST_HOME/system/optim/verifier.adjudicate.json
Rejected candidates are kept as *.candidate.rejected for post-mortem.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# The tuned templates only exist on the two-stage path — force it for the
# whole optimization run regardless of the caller's environment.
os.environ["GHOST_VERIFY_TWO_STAGE"] = "1"

from ghost_agent.core import verifier as verifier_mod  # noqa: E402
from ghost_agent.core.verifier import Verifier  # noqa: E402
from ghost_agent.eval.verify_bench import (  # noqa: E402
    BenchTrial,
    HttpChatClient,
    build_trials,
    extract_cases_from_recordings,
    load_cases_jsonl,
    run_trials,
)
from ghost_agent.optim.run_gepa import MAX_OPT_ITERATIONS  # noqa: E402
from ghost_agent.optim.trainset import holdout_tier  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "scripts" / "verify_bench_cases.jsonl"

COMPONENTS = {
    "verifier.enumerate": verifier_mod._VERIFY_ENUMERATE_PROMPT,
    "verifier.adjudicate": verifier_mod._VERIFY_ADJUDICATE_PROMPT,
}


def _trial_score(trial: BenchTrial, verdict: Optional[str]) -> float:
    """Graded verdict correctness. UNCERTAIN gets partial credit against
    a hard expectation (better than the opposite verdict, worse than
    right); NOT_REFUTED accepts either non-REFUTED verdict."""
    if verdict is None:
        return 0.0
    if trial.expected == "NOT_REFUTED":
        return 1.0 if verdict in ("CONFIRMED", "UNCERTAIN") else 0.0
    if verdict == trial.expected:
        return 1.0
    if verdict == "UNCERTAIN":
        return 0.3
    return 0.0


def _is_nonrefute(trial: BenchTrial) -> bool:
    return trial.expected in ("CONFIRMED", "NOT_REFUTED")


def balanced_score(trials: List[BenchTrial], raw_scores: List[float]) -> float:
    """Macro-average over the two expectation classes: catch-rate mass and
    false-alarm mass count EQUALLY, however lopsided the trial mix is. This
    is the ship-gate metric — the 2026-07-30 ship optimized a mean dominated
    ~5:1 by REFUTED-expecting trials and traded FPR for TPR (live cost: ~18
    escalation overturns/day)."""
    nr = [s for t, s in zip(trials, raw_scores) if _is_nonrefute(t)]
    rf = [s for t, s in zip(trials, raw_scores) if not _is_nonrefute(t)]
    nr_m = sum(nr) / len(nr) if nr else 0.0
    rf_m = sum(rf) / len(rf) if rf else 0.0
    return 0.5 * nr_m + 0.5 * rf_m


class VerifierBenchAdapter:
    """gepa.GEPAAdapter over the real two-stage verifier pipeline.

    A fresh HttpChatClient + Verifier is built INSIDE each evaluate()
    event loop: httpx.AsyncClient pools become loop-affine after first
    use, and this adapter runs one asyncio.run() per candidate
    evaluation — a shared client would die on the second loop.

    `refute_weight` down-scales REFUTED-expecting trial scores so the two
    expectation classes carry equal mass in gepa's mean/Pareto view
    (weight = n_nonrefute / n_refute over the public trials). Raw scores
    are preserved on trajectories for reflection feedback and gate math."""

    # gepa's reflective proposer probes this OPTIONAL hook with a direct
    # attribute access (`if self.adapter.propose_new_texts is not None`),
    # so a duck-typed adapter MUST define it even when unused — None means
    # "use gepa's default instruction-proposal prompt".
    propose_new_texts = None

    def __init__(self, base_url: str, *, timeout: float = 90.0,
                 model: str = "", concurrency: int = 2,
                 refute_weight: float = 1.0):
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self.concurrency = concurrency
        self.refute_weight = refute_weight

    def evaluate(self, batch: List[BenchTrial], candidate: Dict[str, str],
                 capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        # A candidate that lost a placeholder can never run — fail the
        # whole batch with feedback GEPA's reflector can act on.
        broken = [name for name, tmpl in candidate.items()
                  if not verifier_mod._validate_stage_template(name, tmpl)]
        if broken:
            fb = (f"INVALID TEMPLATE: {', '.join(broken)} lost a required "
                  f"placeholder or broke {{{{ }}}} JSON-brace escaping. "
                  f"Required placeholders: "
                  + "; ".join(
                      f"{n}: {verifier_mod._TEMPLATE_PLACEHOLDERS[n]}"
                      for n in broken))
            traj = [{"trial": t, "verdict": None, "suspects": None,
                     "reasoning": fb, "score": 0.0} for t in batch]
            return EvaluationBatch(
                outputs=[None] * len(batch), scores=[0.0] * len(batch),
                trajectories=traj if capture_traces else None)

        async def _run() -> list:
            client = HttpChatClient(self.base_url, timeout=self.timeout,
                                    model=self.model)
            verifier = Verifier(llm_client=client)
            try:
                return await run_trials(
                    verifier, list(batch), concurrency=self.concurrency)
            finally:
                await client.aclose()

        prev = dict(verifier_mod._TEMPLATE_OVERRIDES)
        verifier_mod._TEMPLATE_OVERRIDES.clear()
        verifier_mod._TEMPLATE_OVERRIDES.update(candidate)
        try:
            results = asyncio.run(_run())
        finally:
            verifier_mod._TEMPLATE_OVERRIDES.clear()
            verifier_mod._TEMPLATE_OVERRIDES.update(prev)

        scores, outputs, trajectories = [], [], []
        for r in results:
            s = _trial_score(r.trial, r.verdict)
            w = 1.0 if _is_nonrefute(r.trial) else self.refute_weight
            scores.append(s * w)
            outputs.append(r.verdict)
            trajectories.append({
                "trial": r.trial, "verdict": r.verdict,
                "suspects": r.suspects, "reasoning": r.reasoning,
                "issues": r.issues, "error": r.error, "score": s,
            })
        return EvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate: Dict[str, str],
                                eval_batch, components_to_update):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for comp in components_to_update:
            records = []
            for traj in (eval_batch.trajectories or []):
                t: BenchTrial = traj["trial"]
                verdict = traj["verdict"]
                expected = ("CONFIRMED or UNCERTAIN"
                            if t.expected == "NOT_REFUTED" else t.expected)
                if traj["score"] >= 1.0:
                    feedback = f"CORRECT: verdict {verdict} as expected."
                elif verdict is None:
                    feedback = ("FAILED TO PARSE / no verdict — the stage "
                                "output was unusable. "
                                + str(traj.get("reasoning") or ""))
                else:
                    feedback = (
                        f"WRONG: expected {expected}, got {verdict}. "
                        f"Injected fault: {t.fault}"
                        + (f" ({t.note})" if t.note else "")
                        + ". A 'clean' trial must be CONFIRMED; a faulted "
                          "trial must be REFUTED (the corruption named in "
                          "the fault note is really present).")
                if comp == "verifier.enumerate":
                    gen = json.dumps({"suspects": traj["suspects"]},
                                     default=str)
                else:
                    gen = json.dumps({
                        "verdict": verdict,
                        "issues": traj.get("issues"),
                        "reasoning": traj.get("reasoning"),
                    }, default=str)
                records.append({
                    "Inputs": {
                        "claim": t.claim[:600],
                        "evidence": t.evidence[:900],
                        "user_request": t.context[:300],
                    },
                    "Generated Outputs": gen[:900],
                    "Feedback": feedback,
                })
            out[comp] = records
        return out


def _make_reflection_lm(url: str, model: str):
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"

    def _call(prompt: str) -> str:
        payload = json.dumps({
            "model": model or "local",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return ((data.get("choices") or [{}])[0]
                .get("message", {}).get("content", "") or "")

    return _call


def main() -> int:
    # Synchronous by design: adapter.evaluate() drives the async verifier
    # via asyncio.run(), which must never be called from a running loop —
    # gepa.optimize() is synchronous and calls evaluate() many times.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True,
                    help="judge endpoint (the model serving VERIFY in prod)")
    ap.add_argument("--reflection-url", default="http://127.0.0.1:8088",
                    help="endpoint for GEPA's reflection LM (the main box)")
    ap.add_argument("--model", default="", help="judge model name, if needed")
    ap.add_argument("--reflection-model", default="",
                    help="reflection model name, if needed")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--recordings", default="",
                    help="GHOST_LLM_RECORD day-files to mint extra cases")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--private-pct", type=int, default=30)
    ap.add_argument("--max-iterations", type=int, default=8,
                    help="budget in full evals of the public trial set "
                         f"(clamped to {MAX_OPT_ITERATIONS})")
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="",
                    help="gepa state directory — enables checkpointing so a "
                         "killed run resumes instead of losing its candidates")
    args = ap.parse_args()

    cases = load_cases_jsonl(args.cases)
    if args.recordings:
        rec = Path(args.recordings)
        paths = sorted(rec.glob("*.jsonl")) if rec.is_dir() else [rec]
        cases.extend(extract_cases_from_recordings(paths))
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if len(cases) < 4:
        print(f"only {len(cases)} case(s) — too few to split; add cases "
              f"or --recordings", file=sys.stderr)
        return 2

    pub_cases = [c for c in cases
                 if holdout_tier(f"vbcase:{c.case_id}",
                                 private_pct=args.private_pct) == "public"]
    priv_cases = [c for c in cases if c not in pub_cases]
    if not pub_cases or not priv_cases:
        print("degenerate public/private case split — adjust --private-pct",
              file=sys.stderr)
        return 2

    pub_trials = build_trials(pub_cases, seed=args.seed)
    priv_trials = build_trials(priv_cases, seed=args.seed)
    print(f"{len(pub_cases)} public cases -> {len(pub_trials)} trials | "
          f"{len(priv_cases)} PRIVATE cases -> {len(priv_trials)} trials")

    n_rf = sum(1 for t in pub_trials if not _is_nonrefute(t))
    n_nr = len(pub_trials) - n_rf
    refute_weight = (n_nr / n_rf) if n_rf else 1.0
    print(f"class mix (public): {n_rf} refute-expecting / {n_nr} non-refute "
          f"-> refute_weight {refute_weight:.3f}")
    adapter = VerifierBenchAdapter(
        args.base_url, timeout=args.timeout, model=args.model,
        concurrency=args.concurrency, refute_weight=refute_weight)

    # Seed from the LIVE templates when valid artifacts exist — GEPA refines
    # the incumbent instead of re-deriving from the hand-written baseline,
    # and the ship-gate compares against what production actually runs.
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    optim_dir = base / "system" / "optim"
    seed_candidate: Dict[str, str] = {}
    for name, baseline in COMPONENTS.items():
        text, src = baseline, "baseline"
        try:
            live = json.loads((optim_dir / f"{name}.json").read_text())[
                "optimized_instruction"]
            if verifier_mod._validate_stage_template(name, live):
                text, src = live, "LIVE artifact"
        except Exception:
            pass
        seed_candidate[name] = text
        print(f"seed {name}: {src} ({len(text)} chars)")

    baseline_eval = adapter.evaluate(priv_trials, seed_candidate,
                                     capture_traces=True)
    baseline_raw = [t["score"] for t in baseline_eval.trajectories]
    baseline_bal = balanced_score(priv_trials, baseline_raw)
    print(f"INCUMBENT on PRIVATE trials: balanced={baseline_bal:.3f} "
          f"raw-mean={sum(baseline_raw) / len(baseline_raw):.3f}")

    iterations = min(args.max_iterations, MAX_OPT_ITERATIONS)
    max_metric_calls = iterations * len(pub_trials)

    import gepa
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=pub_trials,
        adapter=adapter,
        reflection_lm=_make_reflection_lm(args.reflection_url,
                                          args.reflection_model),
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
        seed=args.seed,
        run_dir=args.run_dir or None,
    )
    best = dict(result.best_candidate)

    cand_eval = adapter.evaluate(priv_trials, best, capture_traces=True)
    cand_raw = [t["score"] for t in cand_eval.trajectories]
    cand_bal = balanced_score(priv_trials, cand_raw)
    delta = cand_bal - baseline_bal
    valid = all(verifier_mod._validate_stage_template(n, t)
                for n, t in best.items())
    ships = valid and delta > args.min_delta
    print(f"A/B (PRIVATE trials, n={len(priv_trials)}, BALANCED metric): "
          f"incumbent={baseline_bal:.3f} candidate={cand_bal:.3f} "
          f"delta={delta:+.3f} valid={valid} ships={ships}")

    optim_dir.mkdir(parents=True, exist_ok=True)
    for name, template in best.items():
        payload = json.dumps({
            "signature_name": name,
            "baseline_instruction": seed_candidate[name],
            "optimized_instruction": template,
            "optimizer": "GEPA-verifier-bench-balanced",
            "iterations": iterations,
            "private_incumbent_balanced": round(baseline_bal, 4),
            "private_candidate_balanced": round(cand_bal, 4),
            "refute_weight": round(refute_weight, 4),
        }, indent=2)
        live = optim_dir / f"{name}.json"
        if ships:
            live.write_text(payload)
            print(f"PROMOTED {live}")
        else:
            rej = optim_dir / f"{name}.json.candidate.rejected"
            rej.write_text(payload)
            print(f"rejected -> {rej}")
    if not ships:
        print("A/B gate REJECTED the candidate — baselines stand.")

    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
