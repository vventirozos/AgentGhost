"""introspect tool — read-only introspection over the agent's selfhood.

The selfhood wake-up prefix is already spliced into every system prompt, so
a casual "tell me about yourself" question is grounded in the diary by
default. This tool is the explicit handle the agent reaches for when it
wants a richer, deterministic snapshot rather than relying on whatever
the prefix happened to surface — useful for introspective questions
("what have you been working on?", "what do you remember about X?") and
for evaluation / probing scripts.

All actions are read-only. The writable counterpart is ``self_state``
(``tools/self_state.py``); the two are deliberately split so a single
tool description does not conflate "introspect myself" with "author my
forward-looking continuity slot".

Routes through ``context.self_model`` so the SelfModel facade owns the
read API and a disabled selfhood ("--no-self-model" / "--no-memory")
degrades to a clear message instead of crashing.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"summary", "stats", "narrative", "recent", "recall"})

_DEFAULT_RECENT = 5
_DEFAULT_RECALL = 5
_MAX_LIMIT = 25
_SUMMARY_RECENT_N = 5


def _format_experience(exp) -> str:
    line = f"  - {exp.summary}"
    outcome = getattr(exp, "outcome", "") or ""
    if outcome and outcome != "unknown":
        line += f" [{outcome}]"
    return line


def _clamp_limit(value, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, _MAX_LIMIT)


def _render_stats(stats: dict) -> str:
    if not stats:
        return "No self-state on file."
    if stats.get("enabled") is False:
        return "Selfhood is disabled."
    lines: List[str] = []
    lines.append(f"Experiences on file: {stats.get('experience_count', 0)}")
    lines.append(f"Open questions: {stats.get('open_questions', 0)}")
    lines.append(f"Unfinished threads: {stats.get('unfinished_threads', 0)}")
    mood = stats.get("last_mood") or ""
    if mood:
        lines.append(f"Last noted mood: {mood}")
    last = stats.get("last_session_at") or ""
    if last:
        lines.append(f"Last active: {last}")
    lines.append(
        f"Running narrative: {'present' if stats.get('narrative_present') else 'none yet'}"
    )
    clusters = stats.get("clusters") or {}
    if clusters:
        ranked = sorted(clusters.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k}={v}" for k, v in ranked[:5])
        lines.append(f"Topic clusters: {top}")
    return "\n".join(lines)


def _render_summary(self_model) -> str:
    stats = self_model.stats()
    parts: List[str] = []
    parts.append("Who I am — a snapshot of my self-state:")
    parts.append(_render_stats(stats))

    narrative = ""
    if self_model.narrative is not None:
        narrative = (self_model.narrative.latest() or "").strip()
    if narrative:
        parts.append("\nMy running first-person diary:")
        parts.append(narrative)

    recent = []
    if self_model.autobio is not None:
        try:
            recent = self_model.autobio.recent(limit=_SUMMARY_RECENT_N)
        except Exception as e:  # noqa: BLE001 — read path is secondary
            logger.debug("introspect recent() failed: %s", e)
            recent = []
    if recent:
        parts.append("\nRecent things I remember doing:")
        for exp in recent:
            parts.append(_format_experience(exp))

    return "\n".join(parts).rstrip()


def _render_recent(self_model, limit: int) -> str:
    if self_model.autobio is None:
        return "No autobiographical log on file."
    try:
        recent = self_model.autobio.recent(limit=limit)
    except Exception as e:  # noqa: BLE001
        logger.warning("introspect recent failed: %s", e)
        return f"Could not read the autobiographical log: {type(e).__name__}: {e}"
    if not recent:
        return "I have no experiences on file yet."
    lines = [f"My {len(recent)} most recent experiences:"]
    for exp in recent:
        lines.append(_format_experience(exp))
    return "\n".join(lines)


def _render_recall(self_model, query: str, limit: int) -> str:
    matches = self_model.recall_relevant(query, limit=limit) or []
    if not matches:
        return f"Nothing in my past matches '{query}'."
    lines = [f"What I remember about '{query}':"]
    for exp in matches:
        lines.append(_format_experience(exp))
    return "\n".join(lines)


async def tool_introspect(
    action: str = None,
    query: str = None,
    limit: int = None,
    self_model=None,
    **kwargs,
) -> str:
    """Read-only introspection over the agent's selfhood.

    Never raises — introspection is secondary to the user turn.
    """
    raw_action = (action or "summary").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    if self_model is None or not getattr(self_model, "enabled", False):
        return (
            "Introspection is unavailable — the selfhood module is "
            "disabled (--no-self-model / --no-memory)."
        )

    try:
        if raw_action == "stats":
            return _render_stats(self_model.stats())

        if raw_action == "narrative":
            if self_model.narrative is None:
                return "No narrative on file."
            text = (self_model.narrative.latest() or "").strip()
            return text or "No narrative on file yet."

        if raw_action == "recent":
            return _render_recent(
                self_model, _clamp_limit(limit, _DEFAULT_RECENT),
            )

        if raw_action == "recall":
            q = (query or "").strip()
            if not q:
                return "SYSTEM ERROR: 'query' is required for recall."
            return _render_recall(
                self_model, q, _clamp_limit(limit, _DEFAULT_RECALL),
            )

        # Default: summary.
        pretty_log("Introspect", "snapshot requested", icon=Icons.BRAIN_SUM)
        return _render_summary(self_model)
    except Exception as e:  # noqa: BLE001 — never break the turn
        logger.warning("introspect tool failed: %s: %s", type(e).__name__, e)
        return f"Introspection failed: {type(e).__name__}: {e}"
