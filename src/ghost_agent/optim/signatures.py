"""Optimizable prompt signatures.

We don't take the dspy.Signature type as a hard dependency because the
training/eval surface should work without dspy installed. Instead we
ship a lightweight OptimizableSignature dataclass with the fields
dspy's mapping needs:

  * name           — stable ID; used as the on-disk key for the tuned prompt
  * inputs/outputs — field names + brief descriptions
  * instruction    — the hand-written default prompt (the baseline GEPA
                     has to beat before we actually ship a tuned version)

Adapters for dspy itself live in run_gepa.py, which is allowed to take
dspy as a hard dep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OptimizableSignature:
    """A prompt signature we're willing to expose to automated
    optimization. The `scope` field is an explicit whitelist statement —
    if you want to add a new signature here, the scope must be one of
    the approved categories below; otherwise it stays hand-tuned."""

    name: str
    scope: str                         # "planning" | "tool_selection" | "reflection"
    inputs: Dict[str, str]             # field_name → description
    outputs: Dict[str, str]            # field_name → description
    instruction: str                   # baseline (hand-tuned) prompt body
    # Optional notes — visible in CLI / eval output, NOT part of the
    # compiled prompt itself. Prevents scope drift ("I'll just add one
    # more note") from accidentally rewriting the baseline.
    notes: str = ""

    _ALLOWED_SCOPES = frozenset({"planning", "tool_selection", "reflection"})

    def __post_init__(self) -> None:
        if self.scope not in self._ALLOWED_SCOPES:
            raise ValueError(
                f"scope {self.scope!r} not in allow-list "
                f"{sorted(self._ALLOWED_SCOPES)} — optimizer will not touch "
                "any prompt outside the approved scopes"
            )
        if not self.name:
            raise ValueError("signature must have a name")
        if not self.inputs:
            raise ValueError("signature must declare at least one input field")
        if not self.outputs:
            raise ValueError("signature must declare at least one output field")

    def compile_baseline(self) -> str:
        """Render the baseline prompt text exactly as it would ship to
        the model. Kept simple: instruction → input block → output
        directive. GEPA will mutate only the instruction portion.
        """
        lines: List[str] = [self.instruction.strip(), ""]
        if self.inputs:
            lines.append("Inputs:")
            for k, desc in self.inputs.items():
                lines.append(f"  - {k}: {desc}")
            lines.append("")
        if self.outputs:
            lines.append("Required outputs:")
            for k, desc in self.outputs.items():
                lines.append(f"  - {k}: {desc}")
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Concrete signatures (the allow-list).
# Keep each instruction brief and honest about what it wants — GEPA
# will produce variants, and starting short lets the optimizer add
# rather than remove.
# ---------------------------------------------------------------------------

PLANNING_SIGNATURE = OptimizableSignature(
    name="planning.decompose",
    scope="planning",
    inputs={
        "user_request": "The user's natural-language goal.",
        "available_tools": "Comma-separated tool names the agent may use.",
        "memory_snippets": "Relevant facts from long-term memory (may be empty).",
    },
    outputs={
        "plan": "Ordered list of 1-5 steps. Each step names a tool when relevant.",
        "rationale": "One-sentence justification for the overall plan.",
    },
    instruction=(
        "Decompose the user's request into a short, executable plan. "
        "Prefer fewer steps over many. Name a concrete tool for each "
        "step where a tool applies, and leave steps that are plain "
        "reasoning unannotated."
    ),
)

TOOL_SELECTION_SIGNATURE = OptimizableSignature(
    name="tool_selection.pick",
    scope="tool_selection",
    inputs={
        "step_description": "The current step we are trying to execute.",
        "tool_catalog": "Tool name → one-line purpose pairs.",
        "recent_tool_outcomes": "Last 3 tool calls and whether they succeeded.",
    },
    outputs={
        "tool_name": "Exact name of the tool to call.",
        "arguments": "JSON object of arguments for the tool.",
    },
    instruction=(
        "Pick the single tool best suited to the current step. Reuse a "
        "tool only if the recent outcomes show it is succeeding on "
        "similar inputs. Prefer tools that have validated outcomes in "
        "memory over untested ones."
    ),
)

REFLECTION_SIGNATURE = OptimizableSignature(
    name="reflection.critique",
    scope="reflection",
    inputs={
        "failed_attempt": "The trajectory that failed (planning + tools + output).",
        "failure_reason": "What the validator (or user) said was wrong.",
        "original_request": "The original user goal.",
    },
    outputs={
        "diagnosis": "One-sentence explanation of the root cause.",
        "revised_plan": "Short list of steps that correct the failure.",
    },
    instruction=(
        "Read the failed attempt, say precisely why it failed in one "
        "sentence, then propose a corrected plan. Do not restate the "
        "original request. Do not invent facts not present in the "
        "failed attempt."
    ),
    notes=(
        "Used by the reflection-loop biological phase. Kept brief so "
        "the critique step doesn't blow the context budget when the "
        "failed attempt is already long."
    ),
)


SIGNATURES: Dict[str, OptimizableSignature] = {
    s.name: s for s in (
        PLANNING_SIGNATURE,
        TOOL_SELECTION_SIGNATURE,
        REFLECTION_SIGNATURE,
    )
}
