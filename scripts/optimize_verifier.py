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

WHICH SYSTEM THE GATE MEASURES (--escalate, §4J item 1). Production runs
verify as TWO models: a cheap judge on the worker route, then a MAIN-model
re-adjudication of every REFUTED (`Verifier._escalate_refute`). A
single-endpoint judge client cannot escalate, so `--escalate off` gates on
the CHEAP JUDGE STANDALONE — and on the live recorded corpus the main model
overturned 42 of 50 (84%) of that judge's refutes, i.e. most of the
false-alarm mass the gate would be crediting a candidate for is mass
production already removes. `--escalate gate` (needs --main-base-url)
makes the SHIP DECISION production-equivalent at the cost of one extra
main-model call per REFUTED private trial; `--escalate all` also runs the
GEPA loop that way (much slower). The arm is recorded in every shipped
artifact as `gate_arm`, so a promoted template carries the identity of the
pipeline that promoted it.

Usage (live endpoints, measured 2026-08-04: worker/judge = nova:8088 =
100.83.184.117, main = 127.0.0.1:8088):
    PYTHONPATH=src python scripts/optimize_verifier.py \
        --base-url http://nova:8088 \
        --main-base-url http://127.0.0.1:8088 --escalate gate \
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
import re
import sys
import time
import urllib.request
from contextlib import contextmanager
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
    ARM_ESCALATED,
    BenchTrial,
    EscalatingChatClient,
    HttpChatClient,
    build_trials,
    escalation_arm,
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


# Scoring-semantics version. 2 = 2026-08-07: unjudged trials excluded
# from the gate + imputed at judged-mean in training batches; empty
# classes drop from the macro-average. Recorded baselines carry it;
# comparisons across versions are invalid.
SCORER_VERSION = 2


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


def _batch_scores(results) -> "tuple[list[float], int]":
    """Per-trial raw scores with UNJUDGED trials imputed at the batch's
    judged mean.

    ⚠ §4L Lens-B MAJOR-1: unjudged (verdict None — judge endpoint down,
    timeout) used to score 0.0 = "wrong answer", so an infra flake
    during one candidate's batch steered GEPA's accept/reject exactly
    the way the ship gate was distorted before its 2026-08-07 fix.
    Mean-imputation makes the flaky trial NEUTRAL for candidate
    comparison; an all-unjudged batch scores 0.0 loudly rather than
    inventing a mean from nothing.
    """
    judged = [_trial_score(r.trial, r.verdict)
              for r in results if r.verdict is not None]
    judged_mean = sum(judged) / len(judged) if judged else 0.0
    out = [(_trial_score(r.trial, r.verdict)
            if r.verdict is not None else judged_mean) for r in results]
    return out, sum(1 for r in results if r.verdict is None)


class StatusBoard:
    """Glanceable run status. Written ATOMICALLY to <run-dir>/status.txt
    (human) and status.json (machine) on every event — monitoring is
    `cat status.txt` / `watch -n 30 cat status.txt`, no log spelunking,
    no \\r-unwrapping, no grep patterns. Fed by BOTH the adapter (every
    evaluation batch) and gepa's own log stream (iteration decisions)."""

    def __init__(self, run_dir: Optional[str]):
        self.dir = Path(run_dir) if run_dir else None
        self.t0 = time.time()
        self.d: Dict[str, Any] = {
            "run": "verifier-optimization", "phase": "starting",
            "incumbent_private_balanced": None, "best_public_valset": None,
            "proposals": 0, "subsample_accepted": 0,
            "subsample_rejected": 0, "cap_zeroed_batches": 0,
            "new_best_count": 0, "batches": 0, "trials_evaluated": 0,
            "last_event": "", "verdict": "(pending)",
        }
        self.write()

    def event(self, **kw) -> None:
        for k, v in kw.items():
            if k.startswith("inc_"):
                key = k[4:]
                self.d[key] = int(self.d.get(key) or 0) + v
            else:
                self.d[k] = v
        self.write()

    def gepa_log(self, message: str) -> None:
        m = str(message)
        if "Proposed new text" in m:
            self.d["proposals"] += 1
        elif "is not better than old score" in m:
            self.d["subsample_rejected"] += 1
        elif "is better than old score" in m:
            self.d["subsample_accepted"] += 1
        elif "Found a better program" in m:
            self.d["new_best_count"] += 1
        bm = re.search(r"Best valset aggregate score so far: ([0-9.]+)", m)
        if bm:
            self.d["best_public_valset"] = round(float(bm.group(1)), 4)
        self.d["last_event"] = (time.strftime("%H:%M:%S") + " "
                                + m.strip()[:180])
        self.write()

    def write(self) -> None:
        if self.dir is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            el = int(time.time() - self.t0)
            hum = (
                f"verifier optimization — {self.d['phase']} — "
                f"elapsed {el // 3600}h{(el % 3600) // 60:02d}m\n"
                f"incumbent (private, balanced): "
                f"{self.d['incumbent_private_balanced']}   "
                f"best public valset: {self.d['best_public_valset']}\n"
                f"proposals {self.d['proposals']} | subsample "
                f"{self.d['subsample_accepted']} accepted / "
                f"{self.d['subsample_rejected']} rejected | cap-zeroed "
                f"{self.d['cap_zeroed_batches']} | new-best "
                f"{self.d['new_best_count']}\n"
                f"trials evaluated {self.d['trials_evaluated']} "
                f"(batches {self.d['batches']})\n"
                f"last: {self.d['last_event']}\n"
                f"VERDICT: {self.d['verdict']}\n"
            )
            tmp = self.dir / ".status.tmp"
            tmp.write_text(hum)
            tmp.rename(self.dir / "status.txt")
            (self.dir / "status.json").write_text(
                json.dumps({**self.d, "elapsed_s": el}))
        except Exception:
            pass


class _GepaLogger:
    """gepa LoggerProtocol shim: passthrough to stdout + StatusBoard."""

    def __init__(self, board: StatusBoard):
        self.board = board

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.board.gepa_log(message)


def _is_nonrefute(trial: BenchTrial) -> bool:
    return trial.expected in ("CONFIRMED", "NOT_REFUTED")


@contextmanager
def gate_arm_active(adapter: "VerifierBenchAdapter", escalate_mode: str):
    """Run the enclosed evaluation in the SHIP-GATE arm.

    `--escalate gate` trains cheap and gates production-equivalent, so the
    adapter's arm has to change for exactly the two private evaluations and
    change BACK — including when one of them raises. A bare flip/restore
    pair around three call sites does not guarantee that, and an adapter
    left in the wrong arm would run the rest of the GEPA loop through a
    pipeline the report does not name.
    """
    prev = adapter.escalate
    adapter.escalate = escalate_mode in ("gate", "all")
    try:
        yield adapter
    finally:
        adapter.escalate = prev


def balanced_score(trials: List[BenchTrial], raw_scores: List[float],
                   verdicts: Optional[List[Optional[str]]] = None) -> float:
    """Macro-average over the two expectation classes: catch-rate mass and
    false-alarm mass count EQUALLY, however lopsided the trial mix is. This
    is the ship-gate metric — the 2026-07-30 ship optimized a mean dominated
    by REFUTED-expecting trials and traded FPR for TPR.

    Both numbers this docstring used to quote were wrong and are now
    measured. The imbalance was **3.41:1**, not "~5:1": the T0 bundle
    (`ablation_out/watch-4f/t0/vb_preship_baseline_results.json`, 13 cases
    / 97 trials) holds 75 refute-expecting against 22 non-refute. On the
    current 107-case pool it is 3.06:1 public (428/140) and 3.07:1 private
    (166/54). And the live cost was **~8 escalation overturns/day**, not
    "~18": the recorded corpus 2026-07-30..08-04 holds 42 overturns over 5
    days (5/17/12/4/4) — 18 was the peak day quoted as an average.

    ⚠ Two silent distortions fixed by measurement (2026-08-07 review):
    (a) an UNJUDGED trial (verdict None — judge endpoint down, timeout)
    scored 0.0, identical to a WRONG verdict, while `score_trials`
    excludes skips from every rate it reports — five unjudged non-refute
    trials cost ≈4.6pp, four times the 0.011 ship margin, invisibly.
    Unjudged trials are now EXCLUDED here too, and `verdicts` (when
    supplied) is what identifies them. (b) an EMPTY class contributed
    0.0 to the macro-average, so a single-class run capped at 0.500 and
    read as a catastrophic regression; an empty class now DROPS out of
    the mean. Callers must surface `unjudged` — a gate that quietly
    excluded a third of its trials would be the same defect inverted.
    """
    if verdicts is not None:
        keep = [(t, sc) for t, sc, v in zip(trials, raw_scores, verdicts)
                if v is not None]
    else:
        keep = list(zip(trials, raw_scores))
    nr = [sc for t, sc in keep if _is_nonrefute(t)]
    rf = [sc for t, sc in keep if not _is_nonrefute(t)]
    parts = [sum(c) / len(c) for c in (nr, rf) if c]
    return sum(parts) / len(parts) if parts else 0.0


class VerifierBenchAdapter:
    """gepa.GEPAAdapter over the real two-stage verifier pipeline.

    A fresh client + Verifier is built INSIDE each evaluate() event loop:
    httpx.AsyncClient pools become loop-affine after first use, and this
    adapter runs one asyncio.run() per candidate evaluation — a shared
    client would die on the second loop.

    `refute_weight` down-scales REFUTED-expecting trial scores so the two
    expectation classes carry equal mass in gepa's mean/Pareto view
    (weight = n_nonrefute / n_refute over the public trials). Raw scores
    are preserved on trajectories for reflection feedback and gate math.

    `escalate` selects the VERDICT PIPELINE each batch runs through: False
    builds a single-endpoint `HttpChatClient` (cheap judge standalone —
    `_escalate_refute` is structurally a no-op), True builds an
    `EscalatingChatClient` against `main_base_url` so a REFUTED verdict is
    re-adjudicated on the main model exactly as in production. `main()`
    flips it per phase for `--escalate gate`, so the field is read at
    evaluate() time, not captured at construction."""

    # gepa's reflective proposer probes this OPTIONAL hook with a direct
    # attribute access (`if self.adapter.propose_new_texts is not None`),
    # so a duck-typed adapter MUST define it even when unused — None means
    # "use gepa's default instruction-proposal prompt".
    propose_new_texts = None

    def __init__(self, base_url: str, *, timeout: float = 90.0,
                 model: str = "", concurrency: int = 2,
                 refute_weight: float = 1.0,
                 length_caps: Optional[Dict[str, int]] = None,
                 seed_texts: Optional[Dict[str, str]] = None,
                 main_base_url: str = "", main_model: str = "",
                 escalate: bool = False):
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self.concurrency = concurrency
        self.refute_weight = refute_weight
        self.main_base_url = main_base_url
        self.main_model = main_model
        self.escalate = escalate
        # Compression constraint (2026-08-03): the E4B judge FAILS TO FOLLOW
        # rules it carries in a ~5.4KB prompt (capacity-bound rule-following).
        # The "100% of its refutes overturned by the 35B on the SAME prompt"
        # figure this comment used to carry was ONE DAY at n=4 (2026-08-03:
        # 4/4). Re-measured 2026-08-04 across the whole recorded corpus
        # (2026-07-30..08-04): 42 of 50 = 84%, per-day 5/8, 17/18, 12/15,
        # 4/5, 4/4. Still the dominant failure mode, still worth compressing
        # for — but it is 84%, not 100%.
        # MUTATED components longer than their cap score zero with explicit
        # feedback, so GEPA must compress to compete. The seed (live
        # incumbent) is exempt — it's the reference to beat, not a candidate.
        self.length_caps = length_caps or {}
        self.seed_texts = seed_texts or {}
        self.board: Optional[StatusBoard] = None

    def build_client(self):
        """The client for ONE evaluation batch, in the arm currently set.

        Kept public and separate from `evaluate` so `arm_label()` and the
        tests can ask "which pipeline would this adapter run?" without
        making an LLM call — the mislabelling risk this whole change
        exists to remove is exactly a report that names an arm nobody
        verified.
        """
        if self.escalate and self.main_base_url:
            return EscalatingChatClient(
                self.base_url, self.main_base_url, timeout=self.timeout,
                model=self.model, main_model=self.main_model,
                # Production rides the CRITIC leg since 2026-08-06 —
                # the gate must measure the same rung (task #25).
                leg="critic")
        return HttpChatClient(self.base_url, timeout=self.timeout,
                              model=self.model)

    def arm_label(self) -> Dict[str, Any]:
        """`escalation_arm()` for the pipeline this adapter would run now.

        Builds a real client and asks the shared detector rather than
        re-deriving the predicate here: a second copy of "does escalation
        fire?" is precisely how a report ends up naming an arm that never
        ran. The client opens no socket (httpx connects lazily) and is
        dropped unused — there is no loop here to aclose() it on.
        """
        return escalation_arm(Verifier(llm_client=self.build_client()))

    def evaluate(self, batch: List[BenchTrial], candidate: Dict[str, str],
                 capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        # A mutated component over its length cap defeats the whole point
        # (the small judge can't follow long rule lists) — fail the batch
        # with feedback the reflector can act on.
        too_long = [
            (name, len(tmpl)) for name, tmpl in candidate.items()
            if self.length_caps.get(name)
            and tmpl != self.seed_texts.get(name)
            and len(tmpl) > self.length_caps[name]
        ]
        if too_long:
            from gepa.core.adapter import EvaluationBatch
            fb = ("TEMPLATE TOO LONG: " + "; ".join(
                f"{n} is {ln} chars, cap {self.length_caps[n]}"
                for n, ln in too_long)
                + ". The judge model cannot reliably follow long rule "
                  "lists — COMPRESS: keep the decision procedure and the "
                  "false-alarm protections, cut redundancy and examples.")
            # Loud on stdout: run 3a burned 37 iterations on silently
            # zeroed over-cap candidates before anyone saw the pattern.
            print(f"[cap-guard] zeroing candidate: {fb[:160]}", flush=True)
            if self.board:
                self.board.event(inc_cap_zeroed_batches=1)
            traj = [{"trial": t, "verdict": None, "suspects": None,
                     "reasoning": fb, "score": 0.0} for t in batch]
            return EvaluationBatch(
                outputs=[None] * len(batch), scores=[0.0] * len(batch),
                trajectories=traj if capture_traces else None)

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

        # Cheap-leg health, accumulated across every batch of this run. A
        # failed route call falls through to the MAIN model, so the trial
        # stops measuring the judge under test — on a GATE evaluation that
        # silently shifts the number the ship decision rests on.
        self.route_health = getattr(self, "route_health", None) or {
            "route_calls": 0, "route_failures": 0, "route_timeouts": 0,
            "route_empty_replies": 0, "fell_through_to_main": 0}

        async def _run() -> list:
            client = self.build_client()
            verifier = Verifier(llm_client=client)
            # ⚠ Per-trial board tick (2026-08-08): gepa's initial seed
            # evaluation runs the FULL valset as ONE silent batch —
            # raw-judge trials log nothing and the board only ticked at
            # batch END, so an hour-plus of healthy grinding read as a
            # wedge. Three runs were killed for being quietly busy; the
            # third's SIGABRT stack proved the loop healthy in select().
            # Silence is not a stall — but only an instrument can tell
            # you that at 2am.
            _tick = ((lambda _res: self.board.event(inc_trials_evaluated=1))
                     if self.board else None)
            try:
                return await run_trials(
                    verifier, list(batch), concurrency=self.concurrency,
                    on_result=_tick)
            finally:
                _h = getattr(client, "route_health", None)
                if callable(_h):
                    for k, v in _h().items():
                        if isinstance(v, int):
                            self.route_health[k] = (
                                self.route_health.get(k, 0) + v)
                await client.aclose()

        prev = dict(verifier_mod._TEMPLATE_OVERRIDES)
        verifier_mod._TEMPLATE_OVERRIDES.clear()
        verifier_mod._TEMPLATE_OVERRIDES.update(candidate)
        try:
            results = asyncio.run(_run())
        finally:
            verifier_mod._TEMPLATE_OVERRIDES.clear()
            verifier_mod._TEMPLATE_OVERRIDES.update(prev)
        if self.board:
            # trials already ticked per-result; count only the batch.
            self.board.event(inc_batches=1)

        scores, outputs, trajectories = [], [], []
        raw_scores, n_unjudged = _batch_scores(results)
        if n_unjudged:
            print(f"⚠ batch has {n_unjudged}/{len(results)} UNJUDGED "
                  f"trials — imputed at judged-mean, not 0.0", flush=True)
        for r, s in zip(results, raw_scores):
            w = 1.0 if _is_nonrefute(r.trial) else self.refute_weight
            scores.append(s * w)
            outputs.append(r.verdict)
            trajectories.append({
                "trial": r.trial, "verdict": r.verdict,
                "suspects": r.suspects, "reasoning": r.reasoning,
                "issues": r.issues, "error": r.error, "score": s,
                # Direct evidence that each escalation direction fired,
                # carried through so `--incumbent-only` can audit a
                # recorded baseline instead of asserting its arm.
                "confidence": r.confidence,
                "escalated_overturn": r.escalated_overturn,
                "confirm_withheld": r.confirm_withheld,
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
    ap.add_argument("--cap-enumerate", type=int, default=0,
                    help="length cap (chars) for MUTATED enumerate "
                         "candidates; 0 = off")
    ap.add_argument("--cap-adjudicate", type=int, default=0,
                    help="length cap (chars) for MUTATED adjudicate "
                         "candidates; 0 = off")
    ap.add_argument("--main-base-url", default="",
                    help="MAIN model the refute-escalation adjudicates on "
                         "(live: http://127.0.0.1:8088). Required by "
                         "--escalate gate/all.")
    ap.add_argument("--main-model", default="",
                    help="model name for --main-base-url, if needed")
    ap.add_argument("--incumbent-only", default="",
                    help="path to write the INCUMBENT's private-tier "
                         "baseline (JSON) and exit, without running GEPA. "
                         "This is the number the next round's ship gate "
                         "compares against, so it is recorded with full "
                         "provenance — pool hash, template hashes, "
                         "escalation arm and both endpoints. Re-record it "
                         "whenever any of those change.")
    ap.add_argument("--escalate", choices=("off", "gate", "all"),
                    default="off",
                    help="which pipeline is measured: 'off' = cheap judge "
                         "standalone (production overturns 84%% of its "
                         "refutes, so the gate credits false alarms "
                         "production already removes); 'gate' = the "
                         "SHIP-GATE private evals run judge+escalation "
                         "(production-equivalent decision, GEPA still "
                         "trains cheap); 'all' = escalate everywhere.")
    args = ap.parse_args()

    if args.escalate != "off" and not args.main_base_url:
        print(f"--escalate {args.escalate} needs --main-base-url (the model "
              f"the refute-escalation adjudicates on; live "
              f"http://127.0.0.1:8088). Refusing to run an arm it cannot "
              f"actually take.", file=sys.stderr)
        return 2

    cases = load_cases_jsonl(args.cases)
    # The MINED pool, by default — same artifact `scripts/verify_bench.py`
    # defaults to. Without it this gate runs on the 21 seed cases, whose
    # private tier resolves to 0.0833 and therefore REFUSES TO RUN against
    # the 0.02 --min-delta below: the optimizer was unrunnable at its own
    # defaults. Measured: seed only -> 4 private cases / 30 trials / 0.0833;
    # seed + mined -> 29 / 220 / 0.0093.
    _mined = (Path(os.getenv("GHOST_HOME", "")) / "system" / "eval"
              / "verify_bench_cases_mined.jsonl") if os.getenv("GHOST_HOME") else None
    if _mined and _mined.exists():
        _have = {(c.claim.strip(), c.evidence.strip()) for c in cases}
        _extra = [c for c in load_cases_jsonl(_mined)
                  if (c.claim.strip(), c.evidence.strip()) not in _have]
        print(f"loaded {len(_extra)} mined case(s) from {_mined}")
        cases.extend(_extra)
    if args.recordings:
        rec = Path(args.recordings)
        paths = sorted(rec.glob("*.jsonl")) if rec.is_dir() else [rec]
        _have = {(c.claim.strip(), c.evidence.strip()) for c in cases}
        cases.extend([c for c in extract_cases_from_recordings(paths)
                      if (c.claim.strip(), c.evidence.strip()) not in _have])
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

    # ── RESOLUTION GATE ───────────────────────────────────────────────
    # The balanced metric is 0.5*mean(non-refute) + 0.5*mean(refute), so its
    # smallest possible non-zero step is 0.5/len(smaller arm). Shipping on a
    # --min-delta FINER than that means ONE flipped trial decides the run.
    # Measured before the case-pool fix: 6 non-refute private trials -> a
    # 0.0833 step against a 0.02 threshold, 4x too coarse — which is the
    # arithmetic behind the journal's "+-0.08 private-gate noise" and makes
    # the +0.087 ship one trial's worth.
    _p_nr = sum(1 for t in priv_trials if _is_nonrefute(t))
    _p_rf = len(priv_trials) - _p_nr
    _step = 0.5 / min(_p_nr or 1, _p_rf or 1)
    print(f"private class mix: {_p_rf} refute-expecting / {_p_nr} non-refute "
          f"-> smallest resolvable delta {_step:.4f}")
    if _step > args.min_delta:
        print(f"REFUSING TO RUN: the private tier cannot resolve its own "
              f"--min-delta ({_step:.4f} > {args.min_delta}). One flipped "
              f"trial would decide the ship. Grow the case pool (mine "
              f"recording-derived cases; see eval/verify_bench."
              f"extract_cases_from_recordings) or raise --min-delta.",
              file=sys.stderr)
        return 2

    n_rf = sum(1 for t in pub_trials if not _is_nonrefute(t))
    n_nr = len(pub_trials) - n_rf
    refute_weight = (n_nr / n_rf) if n_rf else 1.0
    print(f"class mix (public): {n_rf} refute-expecting / {n_nr} non-refute "
          f"-> refute_weight {refute_weight:.3f}")

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

    caps = {k: v for k, v in {
        "verifier.enumerate": args.cap_enumerate,
        "verifier.adjudicate": args.cap_adjudicate,
    }.items() if v > 0}
    if caps:
        print(f"length caps on MUTATED candidates: {caps}")
    adapter = VerifierBenchAdapter(
        args.base_url, timeout=args.timeout, model=args.model,
        concurrency=args.concurrency, refute_weight=refute_weight,
        length_caps=caps, seed_texts=dict(seed_candidate),
        main_base_url=args.main_base_url, main_model=args.main_model,
        escalate=(args.escalate == "all"))
    board = StatusBoard(args.run_dir or None)
    adapter.board = board
    if args.run_dir:
        print(f"monitor: cat {args.run_dir}/status.txt")

    # ── ARM ───────────────────────────────────────────────────────────
    # The gate arm is resolved by ASKING the adapter what it would build
    # and running the shared `escalation_arm()` detector over it, not by
    # echoing the flag: `--escalate gate` with an unreachable main URL
    # still has to report the arm it really takes.
    with gate_arm_active(adapter, args.escalate):
        gate_arm = adapter.arm_label()
    train_arm = adapter.arm_label()
    print(f"VERDICT PIPELINE — ship gate: {gate_arm['arm']} | "
          f"gepa training: {train_arm['arm']}")
    if gate_arm["arm"] != ARM_ESCALATED:
        print(f"  ⚠ the SHIP GATE measures {gate_arm['measures']}. "
              f"A balanced-score win here is not necessarily a win for "
              f"the pipeline production runs. Pass --escalate gate "
              f"--main-base-url http://127.0.0.1:8088 for the "
              f"production-equivalent decision.")

    board.event(phase="incumbent private eval "
                      f"({gate_arm['arm']})")

    with gate_arm_active(adapter, args.escalate):
        baseline_eval = adapter.evaluate(priv_trials, seed_candidate,
                                         capture_traces=True)
    baseline_raw = [t["score"] for t in baseline_eval.trajectories]
    baseline_verdicts = [t["verdict"] for t in baseline_eval.trajectories]
    baseline_bal = balanced_score(priv_trials, baseline_raw,
                                  baseline_verdicts)
    _unjudged = sum(1 for v in baseline_verdicts if v is None)
    print(f"INCUMBENT on PRIVATE trials: balanced={baseline_bal:.3f} "
          f"raw-mean={sum(baseline_raw) / len(baseline_raw):.3f} "
          f"unjudged={_unjudged}"
          + (" ⚠ unjudged trials are EXCLUDED from the gate — infra "
             "failures are not wrong answers, but a high count voids "
             "the comparison" if _unjudged else ""))
    board.event(phase="optimizing (gepa loop)",
                incumbent_private_balanced=round(baseline_bal, 4))

    if args.incumbent_only:
        # A recorded baseline is only usable if a later run can prove it
        # measured the SAME system. Everything that would invalidate the
        # comparison goes in the file: pool hash, fault set, template
        # hashes, the escalation arm and both endpoints — the same
        # provenance contract verify_bench reports under.
        from ghost_agent.eval.verify_bench import bench_provenance
        _traj = baseline_eval.trajectories or []
        # Class split taken from each trajectory's OWN trial, not by
        # zipping against `priv_trials`: zip truncates silently if the
        # runner ever drops a result, and a silently shortened denominator
        # is exactly the kind of wrong-but-plausible number this file
        # exists to stop shipping.
        # Per-class means EXCLUDE unjudged trials, matching the gate's
        # balanced_score semantics (§4L Lens-B MINOR-1: the same payload
        # used to mix an unjudged-as-0.0 mean with an unjudged-excluded
        # balanced — internally inconsistent).
        _nr = [t["score"] for t in _traj
               if _is_nonrefute(t["trial"]) and t["verdict"] is not None]
        _rf = [t["score"] for t in _traj
               if not _is_nonrefute(t["trial"]) and t["verdict"] is not None]
        _n_unjudged = sum(1 for t in _traj if t["verdict"] is None)
        if len(_traj) != len(priv_trials):
            print(f"WARNING: {len(_traj)} trajectories for "
                  f"{len(priv_trials)} private trials — the baseline is "
                  f"not over the full tier", file=sys.stderr)
        payload = {
            "kind": "verifier-incumbent-private-baseline",
            # ⚠ Bump whenever _trial_score/balanced_score semantics
            # change (§4L Lens-B MINOR-2): a recorded baseline is only
            # comparable to a run whose scorer_version matches — the
            # 2026-08-07 unjudged/empty-class fixes silently invalidated
            # every earlier recorded number, and nothing marked them.
            "scorer_version": SCORER_VERSION,
            "n_unjudged_excluded": _n_unjudged,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
            "metric": "balanced = 0.5*mean(non-refute) + 0.5*mean(refute)",
            "private_incumbent_balanced": round(baseline_bal, 4),
            "private_raw_mean": round(
                sum(baseline_raw) / len(baseline_raw), 4),
            "nonrefute_mean": round(sum(_nr) / len(_nr), 4) if _nr else None,
            "refute_mean": round(sum(_rf) / len(_rf), 4) if _rf else None,
            "n_private_cases": len(priv_cases),
            "n_private_trials": len(priv_trials),
            "class_mix": {"refute_expecting": _p_rf, "non_refute": _p_nr},
            "smallest_resolvable_delta": round(_step, 4),
            "private_pct": args.private_pct,
            "seed": args.seed,
            "high_stakes_trials": sum(1 for t in priv_trials
                                      if t.high_stakes),
            "escalation_events": {
                "refute_overturned": sum(
                    1 for t in _traj if t.get("escalated_overturn")),
                # `confirm_withheld == 0` is ambiguous on its own —
                # "the main model agreed every time" and "no high-stakes
                # trial ever reached a CONFIRMED, so the direction was
                # never invoked" look identical. Eligible separates them.
                "confirm_eligible": sum(
                    1 for t in _traj
                    if t["trial"].high_stakes
                    and t.get("verdict") == "CONFIRMED"
                    and not t.get("escalated_overturn")),
                "confirm_withheld": sum(
                    1 for t in _traj if t.get("confirm_withheld")),
            },
            # Per-trial rows, so a recorded baseline can be re-scored
            # rather than re-run: every number above is recomputable from
            # these. Compact by design — no reasoning text.
            "trials": [
                {"case_id": t["trial"].case_id, "fault": t["trial"].fault,
                 "expected": t["trial"].expected,
                 "high_stakes": t["trial"].high_stakes,
                 "verdict": t.get("verdict"),
                 "confidence": t.get("confidence"),
                 "score": t.get("score"),
                 "escalated_overturn": t.get("escalated_overturn"),
                 "confirm_withheld": t.get("confirm_withheld")}
                for t in _traj
            ],
            "verdicts": {
                v: sum(1 for t in _traj if t.get("verdict") == v)
                for v in ("CONFIRMED", "REFUTED", "UNCERTAIN", None)
            },
            # Whether the cheap judge actually served every trial. A route
            # failure falls through to the MAIN model, so a baseline with
            # `fell_through_to_main > 0` is partly a main-model baseline —
            # recorded rather than argued away.
            "route_health": dict(getattr(adapter, "route_health", {}) or {}),
            "seed_templates": {n: len(t)
                               for n, t in seed_candidate.items()},
            "provenance": bench_provenance(
                priv_cases, judge=gate_arm.get("judge"),
                escalation=gate_arm),
        }
        # `None` is not a JSON key — the skipped-verdict count needs a name.
        payload["verdicts"]["skipped"] = payload["verdicts"].pop(None)
        out = Path(args.incumbent_only)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in payload.items()
                          if k != "provenance"}, indent=2))
        _fell = payload["route_health"].get("fell_through_to_main", 0)
        if _fell:
            print(f"⚠ {_fell} cheap-leg call(s) fell through to the MAIN "
                  f"model ({payload['route_health'].get('route_timeouts', 0)} "
                  f"timeouts). Those trials were judged by the strong model, "
                  f"so this baseline is a BLEND — re-record when the judge "
                  f"node is healthy before gating against it.",
                  file=sys.stderr)
        print(f"baseline written: {out}")
        board.event(phase="done", verdict=(
            f"incumbent baseline recorded: {baseline_bal:.4f} "
            f"({gate_arm['arm']})"))
        return 0

    iterations = min(args.max_iterations, MAX_OPT_ITERATIONS)
    max_metric_calls = iterations * len(pub_trials)

    import gepa
    # The length constraint must be visible where proposals are BORN:
    # gepa's reflector composes new text from the PARENT's eval traces,
    # so a cap enforced only by zero-scoring failures is invisible to it
    # (run 3a: 37 straight ~5KB proposals, all silently zeroed). Append
    # the constraint to the proposal prompt itself.
    reflection_template = None
    if caps:
        from gepa.strategies.instruction_proposal import (
            InstructionProposalSignature,
        )
        cap_min = min(caps.values())
        reflection_template = (
            InstructionProposalSignature.default_prompt_template
            + f"\n\nHARD CONSTRAINT: your new instruction MUST be UNDER "
              f"{cap_min} characters (aim for ~{int(cap_min * 0.75)}). The "
              "judge model that follows it is SMALL and cannot reliably "
              "apply long rule lists — that failure is the very disease "
              "being treated. Compress: keep the decision procedure and "
              "EVERY false-alarm protection, cut redundancy, repetition "
              f"and worked examples. Proposals over {cap_min} characters "
              "score ZERO."
        )
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
        reflection_prompt_template=reflection_template,
        logger=_GepaLogger(board),
    )
    board.event(phase=f"candidate private eval (ship gate, {gate_arm['arm']})")
    best = dict(result.best_candidate)

    with gate_arm_active(adapter, args.escalate):
        cand_eval = adapter.evaluate(priv_trials, best, capture_traces=True)
    cand_raw = [t["score"] for t in cand_eval.trajectories]
    cand_verdicts = [t["verdict"] for t in cand_eval.trajectories]
    cand_bal = balanced_score(priv_trials, cand_raw, cand_verdicts)
    _cand_unjudged = sum(1 for v in cand_verdicts if v is None)
    if _cand_unjudged:
        print(f"⚠ candidate eval: {_cand_unjudged} unjudged trials "
              f"EXCLUDED from the gate")
    delta = cand_bal - baseline_bal
    valid = all(verifier_mod._validate_stage_template(n, t)
                for n, t in best.items())
    fits = all(best[n] == seed_candidate.get(n) or len(best[n]) <= cap
               for n, cap in caps.items() if n in best)
    ships = valid and fits and delta > args.min_delta
    print(f"A/B (PRIVATE trials, n={len(priv_trials)}, BALANCED metric, "
          f"arm={gate_arm['arm']}): "
          f"incumbent={baseline_bal:.3f} candidate={cand_bal:.3f} "
          f"delta={delta:+.3f} valid={valid} fits_caps={fits} "
          f"ships={ships}")
    board.event(phase="done", verdict=(
        f"{'SHIPPED' if ships else 'REJECTED'} — candidate {cand_bal:.3f} "
        f"vs incumbent {baseline_bal:.3f} (delta {delta:+.3f}, "
        f"fits_caps={fits})"))

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
            # WHICH PIPELINE decided this promotion. Two artifacts whose
            # `gate_arm` differs were judged by different systems and
            # their balanced scores are not comparable — the same reason
            # verify_bench reports `fpr_raw_judge` vs `fpr_escalated`.
            "gate_arm": gate_arm["arm"],
            "train_arm": train_arm["arm"],
            "gate_judge": gate_arm.get("judge"),
            "gate_main": gate_arm.get("main"),
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
