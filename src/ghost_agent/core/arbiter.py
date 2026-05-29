"""Dual-solver arbitration — roadmap phase 3.

The doc's fourth proposal is asymmetric dual-solver arbitration: when
prospective confidence is low, generate TWO independent candidate
plans (low-temp + varied-prompt), compare their semantic divergence,
and either execute the consensus or escalate to a rule-based validator
or the user. Heavy multi-agent debate at every turn is impractical on
edge hardware (the doc's whole 8GB framing) — but a two-shot fallback
on the small fraction of turns where the composite confidence is
already low is affordable.

This module wires four pieces:

  1. ``DualSolverArbiter`` — orchestrates two candidate generations
     using an injected ``runner`` callable. Re-uses the temperature-
     variation pattern from ``distill/self_consistency.py`` but with
     a hard ``n=2`` and a tight per-sample timeout.
  2. ``SemanticDivergence`` — cosine-similarity check between the two
     candidates. Embedding source is injected: production passes
     ``LLMClient.get_embeddings``; tests pass a deterministic stub.
  3. ``ArbitrationDecision`` — the structured result the caller acts
     on: ``execute`` / ``validate`` / ``ask_user``.
  4. ``confidence_gated`` — convenience helper that bridges
     ``CompositeConfidence`` to the arbiter so callers don't have to
     hand-thread the threshold check.

Cost discipline:
  * Max one arbitration round per turn (configurable).
  * Per-sample timeout (default 10 s).
  * On either timeout, fall through to whichever candidate did finish;
    if BOTH timed out, return an ``ask_user`` decision rather than
    blocking the turn.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("GhostAgent")


# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """One arbitration candidate. ``output`` is the model's full
    response; ``temperature`` is the parameter that produced it (used
    by debug logs to spot the diverging axis)."""

    output: str
    temperature: float
    duration_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and bool(self.output)


@dataclass(frozen=True)
class ArbitrationDecision:
    """Structured arbitration result.

    ``action`` ∈ {``execute``, ``validate``, ``ask_user``, ``skipped``}.
    ``chosen`` is the candidate the caller should use when ``action ==
    execute``; for the other actions it's the higher-confidence
    candidate (caller may still want to surface it to the user).
    """

    action: str
    chosen: Optional[Candidate]
    other: Optional[Candidate]
    similarity: float
    reason: str
    candidates: List[Candidate] = field(default_factory=list)


RunnerInput = Dict[str, Any]
RunnerOutput = Union[str, Dict[str, Any]]
Runner = Callable[[RunnerInput], Union[RunnerOutput, Awaitable[RunnerOutput]]]
EmbedFn = Callable[[Sequence[str]], Union[List[List[float]],
                                          Awaitable[List[List[float]]]]]


# ──────────────────────────────────────────────────────────────────────
# Semantic divergence
# ──────────────────────────────────────────────────────────────────────

class SemanticDivergence:
    """Embedding-based cosine-similarity comparator.

    The embedder is injected: a production caller passes
    ``LLMClient.get_embeddings``, tests pass a deterministic stub
    (e.g. token-hash → bag-of-words vector) so the comparator can be
    exercised without a model server.

    Fallback when no embedder is available: token-overlap Jaccard.
    It under-clusters paraphrases (so the arbiter is slightly more
    likely to escalate) but never crashes, which is the right
    failure mode for an advisory signal.
    """

    DEFAULT_THRESHOLD = 0.85

    def __init__(self, embedder: Optional[EmbedFn] = None,
                 *, threshold: float = DEFAULT_THRESHOLD):
        self.embedder = embedder
        self.threshold = float(threshold)

    async def similarity(self, a: str, b: str) -> float:
        """Return cosine similarity in [-1, 1] (typically [0, 1] for
        normalised embeddings). On failure, falls back to Jaccard.
        Empty input → similarity 0."""
        # Strip first: a whitespace-only candidate is empty in substance,
        # so it scores 0 (per the docstring) and must NOT slip through to
        # _jaccard, where two blank outputs tokenise to empty sets and
        # score 1.0 — a spurious "converged" the arbiter would act on.
        if not (a or "").strip() or not (b or "").strip():
            return 0.0
        if self.embedder is None:
            return _jaccard(a, b)
        try:
            embeddings = self.embedder([a, b])
            if inspect.isawaitable(embeddings):
                embeddings = await embeddings
            if (not embeddings or len(embeddings) != 2
                    or not embeddings[0] or not embeddings[1]):
                return _jaccard(a, b)
            return _cosine(embeddings[0], embeddings[1])
        except Exception as exc:
            logger.debug("SemanticDivergence embed failed: %s", exc)
            return _jaccard(a, b)

    def diverged(self, similarity: float) -> bool:
        return similarity < self.threshold


# ──────────────────────────────────────────────────────────────────────
# Arbiter
# ──────────────────────────────────────────────────────────────────────

class DualSolverArbiter:
    """Two-sample candidate generator + divergence comparator.

    Caller workflow::

        arbiter = DualSolverArbiter(runner=my_runner, embedder=my_embedder)
        decision = await arbiter.arbitrate(
            prompt="<the user request>",
            validator=my_sql_validator,   # optional rule-based fallback
        )

    The runner is invoked twice with the same ``prompt`` but different
    sampling configs (a low-temperature deterministic candidate plus
    a higher-temperature varied one). Decisions:

      * Both candidates fail/time-out → ``ask_user``
      * One candidate fails           → ``execute`` the survivor
      * Candidates converge (sim ≥ τ) → ``execute`` lower-temp one
      * Candidates diverge:
          - validator passes one      → ``execute`` that one
          - validator passes neither  → ``ask_user``
          - no validator              → ``validate`` (caller chooses)
    """

    def __init__(
        self,
        runner: Runner,
        *,
        embedder: Optional[EmbedFn] = None,
        temperatures: Tuple[float, float] = (0.2, 0.7),
        per_sample_timeout_s: float = 10.0,
        divergence_threshold: float = SemanticDivergence.DEFAULT_THRESHOLD,
    ):
        self.runner = runner
        self.temperatures = (float(temperatures[0]), float(temperatures[1]))
        self.per_sample_timeout_s = float(per_sample_timeout_s)
        self.divergence = SemanticDivergence(
            embedder, threshold=divergence_threshold,
        )

    async def arbitrate(
        self,
        prompt: str,
        *,
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ArbitrationDecision:
        """Run the two-shot arbitration. Returns an ``ArbitrationDecision``
        the caller can act on directly. Never raises.

        ``validator`` is the optional rule-based checker — same shape as
        the ``validators`` module. When given, divergent candidates are
        routed through it; consensus candidates bypass.
        """
        if not prompt or not prompt.strip():
            return ArbitrationDecision(
                action="skipped", chosen=None, other=None,
                similarity=0.0, reason="empty prompt", candidates=[],
            )

        cands = await self._run_pair(prompt, extra=extra or {})
        ok_cands = [c for c in cands if c.ok]
        if not ok_cands:
            return ArbitrationDecision(
                action="ask_user", chosen=None, other=None,
                similarity=0.0,
                reason="both candidates failed: " + "; ".join(
                    f"T={c.temperature}: {c.error}" for c in cands),
                candidates=cands,
            )
        if len(ok_cands) == 1:
            survivor = ok_cands[0]
            dead = next((c for c in cands if not c.ok), None)
            return ArbitrationDecision(
                action="execute", chosen=survivor, other=dead,
                similarity=0.0,
                reason=f"only T={survivor.temperature} completed",
                candidates=cands,
            )

        # Both ok — compare
        a, b = ok_cands
        sim = await self.divergence.similarity(a.output, b.output)
        # Always prefer the lower-temperature candidate as the chosen
        # one on convergence (it's the deterministic baseline).
        chosen = a if a.temperature <= b.temperature else b
        other = b if chosen is a else a

        if not self.divergence.diverged(sim):
            return ArbitrationDecision(
                action="execute", chosen=chosen, other=other,
                similarity=sim,
                reason=f"converged (sim={sim:.2f} ≥ "
                       f"τ={self.divergence.threshold:.2f})",
                candidates=cands,
            )

        # Diverged — route to validator or user
        if validator is None:
            return ArbitrationDecision(
                action="validate", chosen=chosen, other=other,
                similarity=sim,
                reason=f"diverged (sim={sim:.2f}); no validator provided",
                candidates=cands,
            )
        # Try the validator on both, pick the one that passes
        passers = []
        failures: List[str] = []
        for cand in (chosen, other):
            try:
                ok, why = validator(cand.output)
            except Exception as exc:
                ok, why = False, f"validator-error:{type(exc).__name__}"
            if ok:
                passers.append(cand)
            else:
                failures.append(f"T={cand.temperature}: {why}")
        if len(passers) == 1:
            return ArbitrationDecision(
                action="execute", chosen=passers[0],
                other=next(c for c in (chosen, other) if c is not passers[0]),
                similarity=sim,
                reason=f"diverged but validator picked T={passers[0].temperature}",
                candidates=cands,
            )
        if len(passers) >= 2:
            # Both pass validation but disagree semantically — caller
            # has to decide which to surface; fall through to ask_user.
            return ArbitrationDecision(
                action="ask_user", chosen=chosen, other=other,
                similarity=sim,
                reason=f"diverged and validator passed both (sim={sim:.2f})",
                candidates=cands,
            )
        return ArbitrationDecision(
            action="ask_user", chosen=chosen, other=other,
            similarity=sim,
            reason="diverged and validator rejected both: " + "; ".join(failures),
            candidates=cands,
        )

    # ---------------------------------------------------------- internal

    async def _run_pair(self, prompt: str,
                        *, extra: Dict[str, Any]) -> List[Candidate]:
        tasks = [self._run_one(prompt, t, extra) for t in self.temperatures]
        return list(await asyncio.gather(*tasks))

    async def _run_one(self, prompt: str, temperature: float,
                       extra: Dict[str, Any]) -> Candidate:
        start = time.monotonic()
        payload: Dict[str, Any] = {"prompt": prompt, "temperature": temperature}
        payload.update(extra)
        output_text = ""
        err = ""
        try:
            result = self.runner(payload)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(
                    result, timeout=self.per_sample_timeout_s,
                )
            if isinstance(result, dict):
                output_text = str(result.get("output") or "")
            else:
                output_text = str(result or "")
        except asyncio.TimeoutError:
            err = f"timeout after {self.per_sample_timeout_s:.1f}s"
        except Exception as e:
            err = f"runner raised {type(e).__name__}: {e}"
        return Candidate(
            output=output_text,
            temperature=temperature,
            duration_s=time.monotonic() - start,
            error=err,
        )


# ──────────────────────────────────────────────────────────────────────
# Confidence-gated convenience
# ──────────────────────────────────────────────────────────────────────

async def confidence_gated(
    *,
    confidence_below_threshold: bool,
    arbiter: DualSolverArbiter,
    prompt: str,
    validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
) -> Optional[ArbitrationDecision]:
    """Convenience: only invoke the arbiter when composite confidence
    flagged the turn as low. Returns ``None`` when the gate is open
    (caller should execute the original plan unchanged)."""
    if not confidence_below_threshold:
        return None
    return await arbiter.arbitrate(prompt, validator=validator)


# ──────────────────────────────────────────────────────────────────────
# Vector helpers
# ──────────────────────────────────────────────────────────────────────

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # Pad the shorter one with zeros so dot product is still defined
        n = max(len(a), len(b))
        a = list(a) + [0.0] * (n - len(a))
        b = list(b) + [0.0] * (n - len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        try:
            xv = float(x); yv = float(y)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(xv) and math.isfinite(yv)):
            continue
        dot += xv * yv
        na += xv * xv
        nb += yv * yv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(na) * math.sqrt(nb))))


def _jaccard(a: str, b: str) -> float:
    sa = set(_tokenise(a))
    sb = set(_tokenise(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _tokenise(s: str) -> List[str]:
    return [tok for tok in (s or "").lower().split() if tok]


__all__ = [
    "DualSolverArbiter",
    "SemanticDivergence",
    "ArbitrationDecision",
    "Candidate",
    "confidence_gated",
]
