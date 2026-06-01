"""Safety rails for long-term projects.

Four concerns live here:
  1. Budget enforcement (runtime + tool-call caps on top of the
     step-count cap already implemented in ``project_advancer``).
  2. Human-gate postconditions — postconditions prefixed with
     ``HUMAN_GATE:`` force the task into ``NEEDS_USER`` instead of
     ``DONE`` regardless of the advancer's own judgement.
  3. Contradiction routing — when a new artifact for a task disagrees
     with a DONE sibling's result, we surface it via
     ``memory/contradiction_log.py`` instead of silently overwriting.
  4. Suggestion heuristics — decide whether to *suggest* (never
     auto-promote) a free-chat session into a project.

Every helper is pure (no I/O beyond the explicit store it is passed),
which keeps the rails independently testable and lets the scheduler
decide whether to wire them in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("GhostAgent")


# ------------------------------------------------------------------ budgets

@dataclass
class BudgetDecision:
    allowed: bool
    reason: str
    remaining: Dict[str, Any]


def check_budget(proj_metadata: Dict[str, Any]) -> BudgetDecision:
    """Evaluate whether a project has any remaining budget across all
    configured knobs.

    Configurable keys (all optional):
      - ``steps_cap`` / ``steps_used``      — per-tick step counter.
      - ``runtime_cap_seconds`` / ``runtime_used_seconds`` — cumulative
        wall-clock spent inside autoadvance ticks.
      - ``tool_call_cap`` / ``tool_call_used`` — total tool invocations.

    Missing knobs are treated as unbounded. A DENY from any single
    dimension is enough to block the tick — we deliberately avoid
    "most restrictive wins" fallbacks so a forgotten knob can't
    silently re-enable spend.
    """
    meta = proj_metadata or {}
    remaining: Dict[str, Any] = {}

    for used_key, cap_key, label in [
        ("steps_used", "steps_cap", "steps"),
        ("runtime_used_seconds", "runtime_cap_seconds", "runtime"),
        ("tool_call_used", "tool_call_cap", "tool_calls"),
    ]:
        cap = meta.get(cap_key)
        if cap is None:
            continue
        used = float(meta.get(used_key, 0) or 0)
        if used >= float(cap):
            return BudgetDecision(
                allowed=False,
                reason=f"{label} budget exhausted ({used}/{cap})",
                remaining={"kind": label, "used": used, "cap": cap},
            )
        remaining[label] = {"used": used, "cap": cap}

    return BudgetDecision(True, "within budget", remaining)


def record_runtime(
    store, project_id: str, seconds: float,
    tool_calls: int = 0,
) -> None:
    """Merge a tick's runtime + tool-call usage into the project metadata."""
    if store is None or seconds < 0 or tool_calls < 0:
        return
    proj = store.get_project(project_id)
    if not proj:
        return
    meta = dict(proj.get("metadata") or {})
    meta["runtime_used_seconds"] = float(
        meta.get("runtime_used_seconds", 0) or 0
    ) + float(seconds)
    if tool_calls:
        meta["tool_call_used"] = int(
            meta.get("tool_call_used", 0) or 0
        ) + int(tool_calls)
    store.update_project(project_id, metadata=meta)


# ------------------------------------------------------------------ human gates

HUMAN_GATE_PREFIX = "HUMAN_GATE:"


def enforce_human_gate(task: Dict[str, Any]) -> Optional[str]:
    """Return the first HUMAN_GATE reason found on a task, or None.

    Callers check the return value BEFORE marking DONE; a non-None
    result means the task must transition to NEEDS_USER instead.
    """
    for pc in (task or {}).get("postconditions") or []:
        if not pc:
            continue
        s = str(pc).strip()
        if s.upper().startswith(HUMAN_GATE_PREFIX):
            return s[len(HUMAN_GATE_PREFIX):].strip() or "human approval required"
    return None


# ------------------------------------------------------------------ contradictions

_CONFLICT_MARKERS = [
    ("true", "false"),
    ("yes", "no"),
    ("supported", "not supported"),
    ("confirmed", "denied"),
    ("succeeded", "failed"),
    ("safe", "unsafe"),
    ("compatible", "incompatible"),
]


def detect_contradiction(new_summary: str, prior_summary: str) -> Optional[str]:
    """Lightweight textual check: does a new result contradict a prior
    result on the same underlying claim?

    Heuristic only — it catches obvious flips like "confirmed" vs
    "denied". For nuanced disagreement we'd need an LLM judge, and the
    caller can upgrade to one by wrapping this function.
    """
    if not new_summary or not prior_summary:
        return None
    ns = str(new_summary).lower()
    ps = str(prior_summary).lower()
    for a, b in _CONFLICT_MARKERS:
        if a in ns and b in ps:
            return f"new says '{a}', prior says '{b}'"
        if b in ns and a in ps:
            return f"new says '{b}', prior says '{a}'"
    return None


def route_contradiction(
    contradiction_log, new_fact: str, prior_facts: List[str], reason: str = "",
) -> bool:
    """Record a contradiction via the existing contradiction_log store.

    Returns True on success, False when the log isn't wired up — the
    caller can use that signal to fall back to plain logging.
    """
    if contradiction_log is None:
        return False
    try:
        contradiction_log.record(
            new_fact=new_fact, old_facts=prior_facts,
            deleted_ids=[], reason=reason,
        )
        return True
    except Exception:
        logger.debug("contradiction_log.record failed", exc_info=True)
        return False


# ------------------------------------------------------------------ suggestion heuristic

@dataclass
class PromotionSuggestion:
    should_suggest: bool
    suggested_title: str = ""
    reason: str = ""
    signals: Dict[str, Any] = None


# Threshold knobs are module-level so tests + operators can override
# without touching the callsite. The defaults are deliberate: 8 turns
# is roughly where a freeform session stops looking like chat; ≥1
# sandbox write means the user has started producing durable output;
# ≥3 plan nodes means the planner thought the work deserved
# decomposition.
MIN_TURNS_FOR_SUGGESTION = 8
MIN_PLAN_NODES_FOR_SUGGESTION = 3


def should_suggest_promotion(
    *,
    user_turns: Sequence[str],
    assistant_turns: Sequence[str],
    sandbox_writes: int,
    plan_node_count: int,
    already_in_project: bool,
    managing_projects: bool = False,
) -> PromotionSuggestion:
    """Decide whether to *offer* promoting the current chat into a project.

    The answer is advisory only — the agent surfaces it as a question
    and waits for the user to accept. ``tool_manage_projects
    action=promote_from_context`` is the only path that actually
    creates a project.

    Returns ``should_suggest=False`` in any of these cases:
      - already in a project (we never nag while a project is active)
      - the turn used the project tool itself (list/delete/switch/...):
        the user is administering projects, not doing promotable free
        chat — offering to "promote this to a project" right after a
        ``delete project`` is nonsensical (reported in the field)
      - too few turns AND no sandbox writes AND a small plan
      - the user already declined (caller tracks that in scratchpad)
    """
    if already_in_project:
        return PromotionSuggestion(False, reason="already in project mode",
                                   signals={})
    if managing_projects:
        return PromotionSuggestion(False, reason="user is managing projects",
                                   signals={})

    signals: Dict[str, Any] = {
        "user_turn_count": len(user_turns),
        "assistant_turn_count": len(assistant_turns),
        "sandbox_writes": sandbox_writes,
        "plan_node_count": plan_node_count,
    }

    strong_signal = (
        len(user_turns) >= MIN_TURNS_FOR_SUGGESTION
        or sandbox_writes >= 1
        or plan_node_count >= MIN_PLAN_NODES_FOR_SUGGESTION
    )
    if not strong_signal:
        return PromotionSuggestion(False, reason="below thresholds",
                                   signals=signals)

    # Derive a compact title from the first user turn (the original
    # goal) rather than the latest turn (which is usually a follow-up).
    title = ""
    for t in user_turns:
        if t and t.strip():
            title = t.strip()
            break
    if len(title) > 80:
        title = title[:77] + "…"

    reason_bits: List[str] = []
    if len(user_turns) >= MIN_TURNS_FOR_SUGGESTION:
        reason_bits.append(f"{len(user_turns)} turns")
    if sandbox_writes:
        reason_bits.append(f"{sandbox_writes} sandbox write(s)")
    if plan_node_count >= MIN_PLAN_NODES_FOR_SUGGESTION:
        reason_bits.append(f"{plan_node_count}-node plan")

    return PromotionSuggestion(
        should_suggest=True,
        suggested_title=title or "Untitled effort",
        reason=" + ".join(reason_bits) or "signals crossed threshold",
        signals=signals,
    )
