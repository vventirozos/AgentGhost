"""Harness-dimension failure attribution.

Classifies a failure record by WHICH layer of the agent harness caused
it, adapting the six control surfaces of MemoHarness (arXiv:2607.14159)
plus two escape values:

* **context_assembly** — hydration, briefing, evidence packing, context
  pruning, prompt construction (e.g. the verifier evidence-packer
  truncation incident).
* **tool_interaction** — tool routing/args, browser, search, network,
  sandbox (e.g. native-tools arg corruption).
* **generation_control** — sampling, token budgets, LLM timeouts.
* **orchestration** — verifier route, loop-breaker, strikes, dispatch,
  planning.
* **memory** — recall, lessons, knowledge base, project store (wrong or
  stale retrieval, poisoned lessons).
* **output_processing** — parsers, scrub, schema validation (e.g. the
  fail-open SEARCH/REPLACE parser).
* **model** — a genuine base-model reasoning error, i.e. NOT the harness.
* **unknown** — no signal matched; candidates for offline adjudication.

Rationale: this project's debugging history repeatedly misattributed
harness failures to the model (evidence truncation read as a bad judge,
parser fail-open read as bad code edits). An explicit dimension on every
failure record institutionalizes "audit the harness before blaming the
model", and gives the dream-side distillation pass its group key.

This module is a dependency-free leaf: ``memory/skills.py`` imports it
inside the write chokepoint, so it must never import from ``ghost_agent``
at module level (the one lazy import lives inside
:func:`adjudicate_dimension`).
"""

import logging
import os
import re
from typing import Tuple

logger = logging.getLogger("GhostAgent")


DIM_CONTEXT_ASSEMBLY = "context_assembly"
DIM_TOOL_INTERACTION = "tool_interaction"
DIM_GENERATION_CONTROL = "generation_control"
DIM_ORCHESTRATION = "orchestration"
DIM_MEMORY = "memory"
DIM_OUTPUT_PROCESSING = "output_processing"
DIM_MODEL = "model"
DIM_UNKNOWN = "unknown"

DIMENSIONS = (
    DIM_CONTEXT_ASSEMBLY,
    DIM_TOOL_INTERACTION,
    DIM_GENERATION_CONTROL,
    DIM_ORCHESTRATION,
    DIM_MEMORY,
    DIM_OUTPUT_PROCESSING,
    DIM_MODEL,
    DIM_UNKNOWN,
)

# One-line definitions, shared by the adjudication prompt and the docs.
DIMENSION_DEFINITIONS = {
    DIM_CONTEXT_ASSEMBLY: "input construction: hydration, briefing, evidence packing, context pruning/overflow, prompt assembly",
    DIM_TOOL_INTERACTION: "tool routing/arguments, browser, search, network, sandbox access",
    DIM_GENERATION_CONTROL: "decoding: sampling params, token budgets, LLM call timeouts, empty/cut-off completions",
    DIM_ORCHESTRATION: "workflow control: verifier route, loop-breaker, strikes, dispatch, planning/replanning",
    DIM_MEMORY: "cross-call state: recall, lessons/playbook, knowledge base, project store (wrong/stale/poisoned retrieval)",
    DIM_OUTPUT_PROCESSING: "post-call handling: parsers, SEARCH/REPLACE blocks, scrub/redaction, schema validation",
    DIM_MODEL: "a base-model reasoning error despite correct context and tooling (hallucination, miscalculation, refusal)",
    DIM_UNKNOWN: "no attributable signal",
}


# Pattern tables. First match wins; the table order below deliberately
# checks harness dimensions BEFORE `model` so ambiguous records land on
# the harness side — the whole point is to stop blaming the model first.
# Within the ordering, specific beats generic: a network timeout
# (ETIMEDOUT, connection reset — tool_interaction) must not be swallowed
# by the generic LLM-timeout patterns (ReadTimeout, generation stall —
# generation_control), which is why tool_interaction is checked earlier.

_OUTPUT_PROCESSING_PATTERNS = [
    re.compile(r"SEARCH/REPLACE|search.?replace block", re.IGNORECASE),
    re.compile(r"invalid or contained broken JSON", re.IGNORECASE),
    re.compile(r"JSONDecodeError|json\.decoder", re.IGNORECASE),
    re.compile(r"parse (error|failure)|failed to parse|parser (rejected|truncat\w*)", re.IGNORECASE),
    re.compile(r"tool.?call pars\w*", re.IGNORECASE),
    re.compile(r"schema validation|failed validation|pydantic", re.IGNORECASE),
    re.compile(r"malformed (json|xml|output|response)", re.IGNORECASE),
    re.compile(r"unclosed (tag|fence|bracket)|unbalanced quote", re.IGNORECASE),
    re.compile(r"CDATA|</parameter>", re.IGNORECASE),
    re.compile(r"scrubbed|redact\w*", re.IGNORECASE),
    re.compile(r"markdown fence|stray fence", re.IGNORECASE),
    re.compile(r"ESCAPE HATCH", re.IGNORECASE),
]

_MEMORY_PATTERNS = [
    re.compile(r"wrong (recall|memory)|irrelevant (lesson|memory|recall)", re.IGNORECASE),
    re.compile(r"playbook|skill.?lesson", re.IGNORECASE),
    re.compile(r"quarantin\w*", re.IGNORECASE),
    re.compile(r"poisoned lesson|retract\w*", re.IGNORECASE),
    re.compile(r"stale (context|memory|state|hint)", re.IGNORECASE),
    re.compile(r"counterfactual regression", re.IGNORECASE),
    re.compile(r"recalled .{0,40}(wrong|unrelated|stale)", re.IGNORECASE),
    re.compile(r"knowledge.?base|episodic memory|graph memory", re.IGNORECASE),
    re.compile(r"memory (said|claimed|hint)", re.IGNORECASE),
    re.compile(r"hallucinated memory|false memory", re.IGNORECASE),
]

_CONTEXT_ASSEMBLY_PATTERNS = [
    re.compile(r"evidence .{0,30}truncat\w*|truncat\w* .{0,30}evidence", re.IGNORECASE),
    re.compile(r"context (overflow|window|length|limit)", re.IGNORECASE),
    re.compile(r"exceeds? .{0,20}context", re.IGNORECASE),
    re.compile(r"prompt too (long|large)|prompt (overflow|truncat\w*)", re.IGNORECASE),
    re.compile(r"rolling window|context prun\w*|_prune_context", re.IGNORECASE),
    re.compile(r"char.?budget|token budget exceeded", re.IGNORECASE),
    re.compile(r"context.?pressure", re.IGNORECASE),
    re.compile(r"\[truncated", re.IGNORECASE),
    re.compile(r"hydration|hydrated context|briefing", re.IGNORECASE),
    re.compile(r"system prompt .{0,30}(missing|omitted|truncat\w*)", re.IGNORECASE),
    re.compile(r"evidence pack\w*", re.IGNORECASE),
    re.compile(r"missing (from|in) context|not in (the )?context", re.IGNORECASE),
]

_ORCHESTRATION_PATTERNS = [
    re.compile(r"verifier (disagre\w*|refut\w*|wrong|error)|wrong(ly)? refut\w*", re.IGNORECASE),
    re.compile(r"REFUTED"),
    re.compile(r"loop.?breaker|no.?progress loop|noprogress", re.IGNORECASE),
    re.compile(r"strike (cap|decay|ledger)", re.IGNORECASE),
    re.compile(r"force_final_response|forced final", re.IGNORECASE),
    re.compile(r"replan\w*|re-plan\w*|plan node|task tree", re.IGNORECASE),
    re.compile(r"dispatch(ed)? (wrong|to)|misrout\w*", re.IGNORECASE),
    re.compile(r"SYSTEM ALERT"),
    re.compile(r"auto.?repair", re.IGNORECASE),
    re.compile(r"turn budget|budget (spent|exhausted)", re.IGNORECASE),
    re.compile(r"futility|thrash\w*", re.IGNORECASE),
    re.compile(r"subagent|delegat\w*", re.IGNORECASE),
]

_TOOL_INTERACTION_PATTERNS = [
    re.compile(r"tool.{0,15}not found|not found.{0,15}tool", re.IGNORECASE),
    re.compile(r"invalid.?(arg|param|schema)", re.IGNORECASE),
    re.compile(r"arg(ument)?s? (corrupt\w*|mangl\w*|swap\w*|dropp\w*)", re.IGNORECASE),
    re.compile(r"MANDATORY"),
    re.compile(r"browser|selector|element not (found|visible)|navigation (fail\w*|timeout)|playwright", re.IGNORECASE),
    re.compile(r"CAPTCHA|rate.?limit|too many requests", re.IGNORECASE),
    re.compile(r"ECONNREFUSED|ECONNRESET|ETIMEDOUT", re.IGNORECASE),
    re.compile(r"connection.?(reset|refused|closed|error)", re.IGNORECASE),
    re.compile(r"\b(?:502|503|504)\b|service unavailable", re.IGNORECASE),
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"sandbox.?(busy|unavailable|blocked)|PermissionError", re.IGNORECASE),
    re.compile(r"web_search|ZERO results|search (fail\w*|returned nothing)", re.IGNORECASE),
    re.compile(r"download fail\w*|fetch fail\w*", re.IGNORECASE),
]

_GENERATION_CONTROL_PATTERNS = [
    re.compile(r"max.?tokens|token (limit|cap)", re.IGNORECASE),
    re.compile(r"finish_reason.{0,10}length", re.IGNORECASE),
    re.compile(r"(response|completion|output) (was )?(cut off|truncat\w*)", re.IGNORECASE),
    re.compile(r"empty (completion|response|generation)", re.IGNORECASE),
    re.compile(r"ReadTimeout|read timed? out", re.IGNORECASE),
    re.compile(r"upstream (fatal|timeout|error)", re.IGNORECASE),
    re.compile(r"temperature|sampling|presence.?penalty", re.IGNORECASE),
    re.compile(r"thinking (budget|loop|burn\w*)", re.IGNORECASE),
    re.compile(r"Kernel Timeout|timed out after \d+", re.IGNORECASE),
    re.compile(r"generation (stall\w*|timeout)", re.IGNORECASE),
]

# Deliberately thin: confident `model` assignment mostly comes from the
# offline adjudication pass, not the heuristics — a harness failure that
# *mentions* hallucination should still be inspected as a harness failure
# first (all harness tables run before this one).
_MODEL_PATTERNS = [
    re.compile(r"hallucinat\w*", re.IGNORECASE),
    re.compile(r"fabricat(ed|ion)", re.IGNORECASE),
    re.compile(r"made up|invented (a|the)", re.IGNORECASE),
    re.compile(r"arithmetic (error|mistake)|miscalculat\w*", re.IGNORECASE),
    re.compile(r"misread (the|a)|misinterpret\w*", re.IGNORECASE),
    re.compile(r"contradict(ed|s) (itself|the evidence)", re.IGNORECASE),
    re.compile(r"despite (correct|complete) (context|evidence)", re.IGNORECASE),
    re.compile(r"refus(ed|al)", re.IGNORECASE),
]

_DIMENSION_TABLES = (
    (DIM_OUTPUT_PROCESSING, _OUTPUT_PROCESSING_PATTERNS),
    (DIM_MEMORY, _MEMORY_PATTERNS),
    (DIM_CONTEXT_ASSEMBLY, _CONTEXT_ASSEMBLY_PATTERNS),
    (DIM_ORCHESTRATION, _ORCHESTRATION_PATTERNS),
    (DIM_TOOL_INTERACTION, _TOOL_INTERACTION_PATTERNS),
    (DIM_GENERATION_CONTROL, _GENERATION_CONTROL_PATTERNS),
    (DIM_MODEL, _MODEL_PATTERNS),
)


def classify_failure_dimension(text: str) -> Tuple[str, str]:
    """Classify a failure description into a harness dimension.

    Returns ``(dimension, matched_signal)``; ``(DIM_UNKNOWN, "empty")``
    for empty/non-string input, ``(DIM_UNKNOWN, "unclassified")`` when
    nothing matches.
    """
    if not text or not isinstance(text, str):
        return DIM_UNKNOWN, "empty"
    for dimension, table in _DIMENSION_TABLES:
        for pat in table:
            m = pat.search(text)
            if m:
                return dimension, m.group(0)
    return DIM_UNKNOWN, "unclassified"


async def adjudicate_dimension(llm_client, record_text: str, heuristic: str) -> str:
    """Worker-route LLM adjudication of a failure record's dimension.

    Used offline (dream distillation pass) for records the heuristics
    left ``unknown``/ambiguous. Falls back to ``heuristic`` on any
    failure, missing worker pool, or an invalid label — never raises.
    """
    fallback = heuristic if heuristic in DIMENSIONS else DIM_UNKNOWN
    if llm_client is None or not record_text:
        return fallback
    try:
        from .llm import RoutingTask  # lazy: keep this module a leaf

        labels = "\n".join(
            f"- {d}: {DIMENSION_DEFINITIONS[d]}" for d in DIMENSIONS)
        payload = {
            "model": "default",
            "messages": [
                {"role": "system",
                 "content": ("You classify agent failures by harness "
                             "dimension. Output exactly one label, "
                             "nothing else.")},
                {"role": "user",
                 "content": (f"LABELS:\n{labels}\n\n"
                             f"FAILURE RECORD:\n{record_text[:600]}\n"
                             f"HEURISTIC GUESS: {fallback}\nLABEL:")},
            ],
        }
        reply = await llm_client.route(
            task=RoutingTask.CLASSIFY_FAILURE, payload=payload,
            max_tokens=8, temperature=0.0, fallback=None)
        label = str(reply or "").strip().strip("`'\"").lower()
        if label in DIMENSIONS:
            return label
    except Exception as e:
        logger.debug(f"dimension adjudication failed: {type(e).__name__}: {e}")
    return fallback


# --- env toggles -------------------------------------------------------
# Read per call (not at import) so a flag flips without a restart — the
# _two_stage_enabled() idiom from core/verifier.py. Ship defaults: both
# features ON (they are fail-open, background, and bounded); kill
# switches documented in PROJECT_JOURNAL.md §3.

def failure_dim_enabled() -> bool:
    """Dimension attribution at record time. Kill: GHOST_FAILURE_DIM=0."""
    return os.getenv("GHOST_FAILURE_DIM", "1").strip().lower() not in (
        "0", "false", "no")


def distill_enabled() -> bool:
    """Dream-cycle failure-cluster distillation. Kill: GHOST_FAILURE_DISTILL=0."""
    return os.getenv("GHOST_FAILURE_DISTILL", "1").strip().lower() not in (
        "0", "false", "no")


def distill_max() -> int:
    """Max distilled pattern lessons per dream cycle (GHOST_FAILURE_DISTILL_MAX)."""
    try:
        return max(0, int(os.getenv("GHOST_FAILURE_DISTILL_MAX", "2") or 2))
    except ValueError:
        return 2


def adjudicate_enabled() -> bool:
    """LLM adjudication of unknowns in the dream pass. Kill: GHOST_FAILURE_ADJUDICATE=0."""
    return os.getenv("GHOST_FAILURE_ADJUDICATE", "1").strip().lower() not in (
        "0", "false", "no")
