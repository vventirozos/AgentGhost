"""GEPA optimizer runner.

Depends on `dspy` (listed in requirements.txt). Import is deferred to
call site via `_require_dspy()` so `from ghost_agent.optim import ...`
still works if dspy's own transitive deps (litellm, gepa, etc.) hit
install trouble — you get a clear error at the first GEPA call
instead of a cryptic ImportError at module load.

The wrapper does three things:

  1. Maps our `OptimizableSignature` onto a `dspy.Signature`.
  2. Wraps Ghost's own LLMClient as a dspy `LM` so the optimizer uses
     only the local upstream (no teacher, no external call).
  3. Runs DSPy's optimizer (GEPA by default, with MIPROv2 as a
     fallback), persists the winning instruction to disk, and returns
     it for the A/B harness to evaluate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .signatures import OptimizableSignature

logger = logging.getLogger("GhostOptim")


def _require_dspy():
    try:
        import dspy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "dspy import failed — it is a declared requirement in "
            "requirements.txt, so this points at a broken install. "
            "Try `pip install -U dspy-ai>=3.2.0`."
        ) from e


@dataclass
class GEPAResult:
    """Output of `run_gepa`. `optimized_instruction` is the new
    instruction text suitable for writing back to an
    OptimizableSignature."""

    signature_name: str
    baseline_instruction: str
    optimized_instruction: str
    train_score: float = 0.0
    eval_score: float = 0.0
    iterations: int = 0
    candidate_history: List[str] = field(default_factory=list)
    optimizer: str = "GEPA"


def _build_dspy_signature(sig: OptimizableSignature):
    """Produce a `dspy.Signature` subclass from our OptimizableSignature.
    Isolated for testability (pure function given dspy is importable)."""
    _require_dspy()
    import dspy

    fields = {}
    for name, desc in sig.inputs.items():
        fields[name] = dspy.InputField(desc=desc)
    for name, desc in sig.outputs.items():
        fields[name] = dspy.OutputField(desc=desc)
    cls = type(sig.name.replace(".", "_"), (dspy.Signature,), {
        "__doc__": sig.instruction,
        **fields,
    })
    return cls


class _GhostLMAdapter:
    """Wraps Ghost's LLMClient as a dspy-compatible LM.

    dspy's LM protocol: instances are callable with `(prompt,
    **kwargs)` and return a list of completion strings. We translate
    that into our chat_completion shape.

    The wrapper is the ONLY place in this module that talks to the
    upstream — keeping it thin makes it easy to audit that no other
    endpoint is contacted.
    """

    def __init__(self, llm_client, model: str):
        self.llm_client = llm_client
        self.model = model
        # dspy.LM-compatible metadata.
        self.kwargs: Dict[str, Any] = {"model": model}
        self.history: List[Dict[str, Any]] = []

    async def _acall(self, prompt: str, **kwargs) -> List[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(kwargs.get("temperature", 0.2)),
            "max_tokens": int(kwargs.get("max_tokens", 512)),
            "stream": False,
        }
        res = await self.llm_client.chat_completion(payload)
        text = (
            (res or {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return [text]

    def __call__(self, prompt: str, **kwargs) -> List[str]:
        import asyncio
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if running:
            # We're inside a running event loop (GEPA driven from the
            # agent's async runtime). The old code tried run_until_complete
            # (raises "loop already running") and then fell through to
            # asyncio.run (raises "cannot be called from a running event
            # loop") — uncaught, so GEPA died. We can't block the running
            # loop from within it, so run the coroutine to completion on a
            # SEPARATE thread with its own loop and block on the result.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(self._acall(prompt, **kwargs)))
                return fut.result()
        # No running loop on this thread (e.g. driven via asyncio.to_thread)
        # — safe to spin up one directly.
        return asyncio.run(self._acall(prompt, **kwargs))


def run_gepa(
    signature: OptimizableSignature,
    trainset: List[Any],
    *,
    llm_client,
    model: str,
    metric: Callable[[Any, Any], float],
    max_iterations: int = 8,
    optimizer: str = "GEPA",
    output_path: Optional[Path] = None,
) -> GEPAResult:
    """Run GEPA (or MIPROv2 as fallback) on `signature` using `trainset`.

    The caller supplies the metric because signature-specific scoring
    (exact-match? BLEU? validator-pass?) is domain-specific — we don't
    pick for them.
    """
    _require_dspy()
    import dspy

    sig_cls = _build_dspy_signature(signature)
    lm = _GhostLMAdapter(llm_client, model=model)
    dspy.configure(lm=lm)

    module = dspy.Predict(sig_cls)

    # GEPA was introduced in dspy 2.5+; fall back if unavailable.
    if optimizer == "GEPA" and hasattr(dspy, "GEPA"):
        tuner = dspy.GEPA(metric=metric, max_iterations=max_iterations)
    elif hasattr(dspy, "MIPROv2"):
        logger.info("GEPA unavailable; falling back to MIPROv2")
        tuner = dspy.MIPROv2(metric=metric, num_trials=max_iterations)
    else:
        # Last-resort: BootstrapFewShot is always present in recent dspy.
        logger.info("GEPA/MIPROv2 unavailable; falling back to BootstrapFewShot")
        tuner = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)

    compiled = tuner.compile(module, trainset=trainset)

    # Best-effort extraction of the new instruction. DSPy exposes it on
    # the compiled predictor's signature.
    new_instruction = getattr(
        getattr(compiled, "signature", None),
        "instructions",
        signature.instruction,
    )

    result = GEPAResult(
        signature_name=signature.name,
        baseline_instruction=signature.instruction,
        optimized_instruction=str(new_instruction or signature.instruction),
        train_score=0.0,  # metric doesn't expose post-compile score uniformly
        eval_score=0.0,
        iterations=max_iterations,
        candidate_history=[],
        optimizer=optimizer,
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "signature_name": result.signature_name,
            "baseline_instruction": result.baseline_instruction,
            "optimized_instruction": result.optimized_instruction,
            "optimizer": result.optimizer,
            "iterations": result.iterations,
        }, indent=2))
    return result
