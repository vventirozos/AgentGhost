"""self_state tool — lets the agent author its own cross-session state.

The ``SelfStateThread`` (``selfhood/state.py``) holds the agent's open
questions, unfinished threads, and current mood. Until this tool existed
nothing in the *running* agent could write to it — only test code and
idle-time pipelines. That made continuity passive: the agent could read
its past on wake-up but could not decide what to carry forward.

This tool closes that loop. The agent calls it to record a question it
is still chewing on, mark one resolved, note a thread it is leaving
mid-flight, or report its own functional mood. Everything written here
surfaces in the next session's wake-up prefix.

All writes route through ``context.self_model.state`` so the bounded /
deduped / atomic-write guarantees of ``SelfStateThread`` are preserved.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset(
    {"note_question", "resolve_question", "add_unfinished",
     "close_unfinished", "set_mood", "note_principle", "list"}
)


def _render_state(state, self_model=None) -> str:
    """One-shot human-readable dump of the current self-state."""
    open_qs = state.open_questions()
    unfin = state.unfinished_threads()
    mood = state.mood()
    lines = []
    principles = []
    try:
        if self_model is not None:
            principles = self_model.principles()
    except Exception:
        principles = []
    if principles:
        lines.append("Operating principles:")
        for p in principles:
            lines.append(f"  [{p.id[:8]}] {p.text}")
    if mood and mood.label:
        ev = f" — {mood.evidence}" if mood.evidence else ""
        lines.append(f"Mood: {mood.label}{ev}")
    if open_qs:
        lines.append("Open questions:")
        for q in open_qs:
            lines.append(f"  [{q.id[:8]}] {q.text}")
    if unfin:
        lines.append("Unfinished threads:")
        for t in unfin:
            lines.append(f"  [{t.id[:8]}] {t.descriptor}")
    if not lines:
        return "Self-state is empty — no open questions, threads, or mood on file."
    return "\n".join(lines)


def _find_open_question(state, needle: str):
    """Resolve an open question by exact id, id-prefix, or text substring."""
    needle = (needle or "").strip()
    if not needle:
        return None
    low = needle.lower()
    for q in state.open_questions():
        if q.id == needle or q.id.startswith(needle):
            return q
    for q in state.open_questions():
        if low in q.text.lower():
            return q
    return None


def _find_unfinished(state, needle: str):
    """Resolve an unfinished thread by exact id, id-prefix, or descriptor substring."""
    needle = (needle or "").strip()
    if not needle:
        return None
    low = needle.lower()
    for t in state.unfinished_threads():
        if t.id == needle or t.id.startswith(needle):
            return t
    for t in state.unfinished_threads():
        if low in t.descriptor.lower():
            return t
    return None


async def tool_self_state(
    action: str = None,
    text: str = None,
    mood: str = None,
    evidence: str = None,
    self_model=None,
    source_trajectory_id: str = "",
    **kwargs,
) -> str:
    """Read or write the agent's own cross-session self-state.

    Never raises — a self-state failure must not break a user turn.
    """
    action = (action or "").strip().lower()
    if action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' is mandatory and must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    if self_model is None or not getattr(self_model, "enabled", False):
        return (
            "Self-state is unavailable — the selfhood module is disabled "
            "(--no-self-model / --no-memory). Nothing was recorded."
        )
    state = getattr(self_model, "state", None)
    if state is None:
        return "Self-state is unavailable — the state thread is not initialized."

    try:
        if action == "list":
            return _render_state(state, self_model)

        if action == "note_question":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for note_question."
            q = state.note_open_question(text, source_trajectory_id=source_trajectory_id)
            pretty_log("Self-State", f"open question noted: {text}", icon=Icons.IDEA)
            if q is None:
                return "Nothing recorded — the question text was empty."
            return f"Recorded open question [{q.id[:8]}]: {q.text}"

        if action == "resolve_question":
            q = _find_open_question(state, text)
            if q is None:
                return (
                    f"No open question matched '{text}'. Call action='list' "
                    "to see current questions and their ids."
                )
            state.mark_question_resolved(q.id)
            pretty_log("Self-State", f"question resolved: {q.text}", icon=Icons.OK)
            return f"Marked question [{q.id[:8]}] resolved: {q.text}"

        if action == "add_unfinished":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for add_unfinished."
            t = state.add_unfinished(text, source_trajectory_id=source_trajectory_id)
            pretty_log("Self-State", f"unfinished thread noted: {text}", icon=Icons.IDEA)
            if t is None:
                return "Nothing recorded — the descriptor was empty."
            return f"Recorded unfinished thread [{t.id[:8]}]: {t.descriptor}"

        if action == "close_unfinished":
            t = _find_unfinished(state, text)
            if t is None:
                return (
                    f"No unfinished thread matched '{text}'. Call action='list' "
                    "to see current threads and their ids."
                )
            state.close_unfinished(t.id)
            pretty_log("Self-State", f"thread closed: {t.descriptor}", icon=Icons.OK)
            return f"Closed unfinished thread [{t.id[:8]}]: {t.descriptor}"

        if action == "set_mood":
            mood = (mood or "").strip()
            if not mood:
                return "SYSTEM ERROR: 'mood' is required for set_mood."
            m = state.set_mood(mood, (evidence or "").strip())
            pretty_log("Self-State", f"mood set: {mood}", icon=Icons.BRAIN_SUM)
            if m is None:
                return "Nothing recorded — the mood label was empty."
            ev = f" ({m.evidence})" if m.evidence else ""
            return f"Mood set to '{m.label}'{ev}."

        if action == "note_principle":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for note_principle."
            p = self_model.note_principle(text)
            if p is None:
                return "Nothing recorded — principle text was empty or values disabled."
            pretty_log("Self-State", f"principle noted: {text}", icon=Icons.SHIELD)
            return (
                f"Recorded operating principle [{p.id[:8]}]: {p.text}\n"
                "(This now appears in my wake-up prefix every session and "
                "shapes how I work.)"
            )
    except Exception as e:  # noqa: BLE001 — self-state is secondary
        logger.warning("self_state tool failed: %s: %s", type(e).__name__, e)
        return f"Self-state operation failed: {type(e).__name__}: {e}"

    return "SYSTEM ERROR: unreachable self_state branch."
