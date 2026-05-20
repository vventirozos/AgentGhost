"""flag_uncertainty tool — lets the agent register what it does not know.

The ``UncertaintyTracker`` (``core/uncertainty.py``) was orphaned: it
held a clean unknown/assumption API but nothing in the running agent
populated it, so the metacognitive gate, the prompt-injected context,
and the recurring-blind-spot detector all had an empty list to work
with.

This tool is the explicit populate path. When the agent realises it is
missing a fact it needs (an unknown) or is proceeding on an unverified
belief (an assumption), it flags it here. Critical unknowns drive the
clarification gate; everything flagged persists to the durable log so
recurring blind-spots become visible across sessions.
"""

from __future__ import annotations

import logging

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

_VALID_ACTIONS = frozenset({"unknown", "assumption", "list"})


def _render(tracker) -> str:
    lines = []
    crit = tracker.get_critical_unknowns(min_impact=3)
    unresolved = [u for u in tracker.unknowns if not u.resolved]
    if unresolved:
        lines.append("Unknowns this turn:")
        for u in unresolved:
            lines.append(f"  - [impact {u.impact}] {u.what} (resolve via: {u.resolution})")
    if tracker.assumptions:
        lines.append("Assumptions this turn:")
        for a in tracker.assumptions:
            lines.append(f"  - [conf {a.confidence:.0%}] {a.claim}")
    recurring = tracker.recurring_unknowns()
    if recurring:
        lines.append("Recurring blind-spots (across past turns):")
        for text, count in recurring[:5]:
            lines.append(f"  - {text} ({count}×)")
    if not lines:
        return "No uncertainties on file for this turn."
    return "\n".join(lines)


async def tool_flag_uncertainty(
    action: str = None,
    text: str = None,
    impact: int = 3,
    confidence: float = 0.5,
    basis: str = "",
    resolution: str = "ask user",
    uncertainty_tracker=None,
    **kwargs,
) -> str:
    """Record an unknown or an assumption with the metacognitive tracker."""
    action = (action or "").strip().lower()
    if action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' is mandatory and must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )
    if uncertainty_tracker is None:
        return "Uncertainty tracking is unavailable — nothing was recorded."

    try:
        if action == "list":
            return _render(uncertainty_tracker)

        if action == "unknown":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for action='unknown'."
            try:
                impact_int = int(impact)
            except (TypeError, ValueError):
                impact_int = 3
            u = uncertainty_tracker.flag_unknown(
                text, impact=impact_int, resolution=(resolution or "ask user").strip(),
            )
            pretty_log("Uncertainty", f"unknown flagged (impact {u.impact}): {text}",
                       icon=Icons.UNCERTAINTY_DIE)
            note = ""
            if u.impact >= 4 and u.resolution == "ask user":
                note = (" This is a critical unknown — you should ask the user "
                        "to clarify it before finalizing.")
            return f"Flagged unknown (impact {u.impact}/5): {u.what}.{note}"

        if action == "assumption":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for action='assumption'."
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = 0.5
            a = uncertainty_tracker.flag_assumption(
                text, confidence=conf, basis=(basis or "").strip(),
            )
            pretty_log("Uncertainty", f"assumption flagged (conf {a.confidence:.0%}): {text}",
                       icon=Icons.UNCERTAINTY_DIE)
            return f"Flagged assumption (confidence {a.confidence:.0%}): {a.claim}"
    except Exception as e:  # noqa: BLE001 — metacognition is secondary
        logger.warning("flag_uncertainty tool failed: %s: %s", type(e).__name__, e)
        return f"Uncertainty operation failed: {type(e).__name__}: {e}"

    return "SYSTEM ERROR: unreachable flag_uncertainty branch."
