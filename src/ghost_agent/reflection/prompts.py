"""Reflection prompt builder.

A single compact prompt that produces BOTH a diagnosis and a revised
plan in a structured format the Reflector can parse reliably. We keep
it short because failed trajectories already carry context (the failure
reason, tool calls, final output); padding the prompt just risks the
model re-reading the failure instead of proposing a fix.

The prompt is deliberately kept outside of `optim/signatures.py` —
those are *optimizable* prompts that GEPA may rewrite; this one is a
*runtime* prompt used by the biological watchdog. The reflection
SIGNATURE in `optim/signatures.py` covers the *critique shape*
abstractly, which the optimizer can tune; this template is the
concrete body the Reflector ships with out of the box.
"""

from __future__ import annotations

from typing import List, Optional

from ..distill.schema import Trajectory, ToolCall


REFLECTION_PROMPT_TEMPLATE = """\
You are reviewing a failed attempt by yourself at an earlier turn. Your \
job is to produce (a) a one-sentence diagnosis of what went wrong, and \
(b) a corrected plan of 1-5 steps that would succeed.

You must NOT:
- restate the original request
- invent tools or facts that weren't available in the failed attempt
- produce prose outside the two required sections

Required output format:

DIAGNOSIS: <single sentence>
REVISED PLAN:
1. <first concrete step, naming a tool if applicable>
2. <second step>
...

Original request:
{user_request}

Failure reason (from validator or runtime):
{failure_reason}

What was tried:
{tried_summary}
"""


def _truncate(s: str, limit: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[truncated {len(s) - limit} chars]"


def _summarize_tool_calls(tool_calls: List[ToolCall], per_call_limit: int = 220) -> str:
    """Compact human-readable trace of a trajectory's tool sequence.

    Each call is one bullet; its result is truncated so the prompt
    stays well under any reasonable context budget.
    """
    if not tool_calls:
        return "(no tool calls on this attempt)"
    lines = []
    for i, tc in enumerate(tool_calls, 1):
        if tc is None:
            continue
        name = (tc.name or "unknown").strip()
        args = ", ".join(
            f"{k}={_truncate(str(v), 60)}"
            for k, v in (tc.arguments or {}).items()
        )
        result = _truncate(tc.result or "", per_call_limit)
        error = _truncate(tc.error or "", 120)
        entry = f"  {i}. {name}({args})"
        if result:
            entry += f"\n     → {result}"
        if error:
            entry += f"\n     ERROR: {error}"
        lines.append(entry)
    return "\n".join(lines)


def build_reflection_prompt(
    trajectory: Trajectory,
    *,
    max_user_request: int = 1200,
    max_failure_reason: int = 800,
    max_per_call_output: int = 220,
) -> str:
    """Render the full reflection prompt for `trajectory`. All inputs
    are truncated to defensive defaults; the caller can override if a
    specific trajectory needs more context."""
    user_request = _truncate(trajectory.user_request or "(missing)", max_user_request)
    failure_reason = _truncate(
        trajectory.failure_reason or "(validator reported no reason)",
        max_failure_reason,
    )
    tried_summary = _summarize_tool_calls(
        trajectory.tool_calls or [],
        per_call_limit=max_per_call_output,
    )
    return REFLECTION_PROMPT_TEMPLATE.format(
        user_request=user_request,
        failure_reason=failure_reason,
        tried_summary=tried_summary,
    )


def parse_reflection_output(text: str) -> tuple[str, List[str]]:
    """Parse the model's reflection response into (diagnosis, plan steps).

    Lenient by design: open-weights models in the 30B-A3 class often
    add markdown fences, bold markers, or preamble prose. We accept:
      * `DIAGNOSIS:` / `Diagnosis:` / `**Diagnosis**:` / `## Diagnosis`
      * `REVISED PLAN:` / `Revised Plan:` / `**Revised Plan**:` /
        `## Revised Plan` / `## Plan`
      * step markers `1.`, `1)`, `-`, `*`, `•`
      * any amount of surrounding prose

    Empty return means neither a diagnosis header nor a plan section
    was detected — the caller treats that as an unparseable response.
    """
    if not text:
        return "", []
    import re

    diagnosis = ""
    steps: List[str] = []

    # Strip markdown bold (`**word**`) and header markers (`## `) so
    # the keyword regex can match uniformly — but we must NOT strip
    # line-leading `*` that functions as a bullet marker, or we lose
    # the plan steps we're about to parse.
    stripped_text = re.sub(r"\*{2,}", "", text)            # bold only
    stripped_text = re.sub(r"^#+\s*", "", stripped_text, flags=re.MULTILINE)
    lower = stripped_text.lower()

    # --- DIAGNOSIS ---
    # Colon-only separator: a bare `-` after the keyword risks eating a
    # following bullet marker. The prompt itself asks for `:`.
    d_match = re.search(r"\bdiagnosis\b\s*:?\s*", lower)
    if d_match:
        start = d_match.end()
        # End at either the plan section or the next blank line, whichever comes first.
        plan_hits = [
            m.start() for m in re.finditer(
                r"\b(?:revised\s+plan|corrected\s+plan|plan)\b\s*[:\-]", lower[start:]
            )
        ]
        plan_pos = (start + plan_hits[0]) if plan_hits else -1
        blank_pos = lower.find("\n\n", start)
        candidates = [p for p in (plan_pos, blank_pos) if p != -1]
        end = min(candidates) if candidates else len(stripped_text)
        diagnosis_block = stripped_text[start:end].strip()
        # Prefer the first non-empty line; fall back to the whole block
        # (some models answer in one long sentence without a linebreak).
        for line in diagnosis_block.splitlines():
            line = line.strip()
            if line:
                diagnosis = line
                break
        if not diagnosis and diagnosis_block:
            diagnosis = diagnosis_block[:400]

    # --- PLAN ---
    plan_match = re.search(
        r"\b(?:revised\s+plan|corrected\s+plan|plan)\b\s*:?\s*",
        lower,
    )
    if plan_match:
        remainder = stripped_text[plan_match.end():].strip()
        for line in remainder.splitlines():
            s = line.strip()
            if not s:
                if steps:
                    break
                continue
            # Stop at the next top-level section heading, if any.
            if re.match(r"^(?:diagnosis|summary|conclusion|note)\s*[:\-]",
                        s, flags=re.IGNORECASE):
                break
            m = re.match(r"^(?:\d+[.)\-]|[\-\*•])\s*(.+)$", s)
            if m:
                steps.append(m.group(1).strip())
            elif steps:
                # Continuation of the previous step if it's plain prose
                # right after a numbered item.
                steps[-1] = (steps[-1] + " " + s).strip()

    return diagnosis, steps
