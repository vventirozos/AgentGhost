"""Ghost Agent prompt optimization (DSPy / GEPA integration).

Scoped **tightly** to prompts that are safe to auto-tune:

    * planning prompt           — how the agent decomposes requests
    * tool-selection prompt     — how the agent picks between tools
    * reflection prompt         — the self-critique template used by the
                                  reflection loop (see ghost_agent.reflection)

Deliberately NOT in scope:

    * dream / REM prompt        — load-bearing idempotency invariants
    * biological-watchdog copy  — keyed off literal strings elsewhere
    * safety / sandbox prompts  — auto-tuning could silently weaken guards

DSPy (3.2+) is a hard dependency — listed in `requirements.txt`. The
`run_gepa` module still calls `_require_dspy()` before import-time
use of `dspy`, so a broken / partial install produces a clear error
instead of a cryptic ImportError deep in the module tree.

The optimizer LM is always the agent's own upstream — there's no
teacher. GEPA's reflective-mutation step uses the same model that will
ultimately follow the prompt; this is weaker than an external teacher
but still typically beats hand-tuned baselines, and it preserves the
single-LLM / local-only invariant.
"""

from .signatures import (
    OptimizableSignature,
    PLANNING_SIGNATURE,
    TOOL_SELECTION_SIGNATURE,
    REFLECTION_SIGNATURE,
    SIGNATURES,
)
from .trainset import (
    TrainExample,
    build_trainset,
    filter_by_outcome,
    split_train_eval,
)
from .ab_eval import compare_prompts, PromptComparison

__all__ = [
    "OptimizableSignature",
    "PLANNING_SIGNATURE",
    "TOOL_SELECTION_SIGNATURE",
    "REFLECTION_SIGNATURE",
    "SIGNATURES",
    "TrainExample",
    "build_trainset",
    "filter_by_outcome",
    "split_train_eval",
    "compare_prompts",
    "PromptComparison",
]
