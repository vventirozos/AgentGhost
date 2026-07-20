"""Workspace narrative — periodic "state of the workspace" summary.

Companion to ``selfhood.narrative``. Re-generated during the idle phase
(biological watchdog phase 2.9 — runs after the selfhood phase 2.8).
Persisted to a single file so the wake-up prefix can splice it in
without a re-run.

Falls back to a deterministic template when no LLM critique function
is supplied — same pattern as the selfhood summariser. The template
path is what tests / disabled-LLM deployments exercise.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

from .activity import WorkspaceActivity
from .schema import WorkspaceEvent, derive_event_project_id
from .state import WorkspaceStateThread

logger = logging.getLogger("GhostWorkspace")

NARRATIVE_FILENAME = "narrative.txt"


CritiqueFn = Callable[[str], Awaitable[str]]


def _summarise_events(events: Iterable[WorkspaceEvent]) -> List[str]:
    """One-line-per-event human-readable rendering for the prompt or
    the template fallback."""
    out: List[str] = []
    for ev in events:
        s = (ev.summary or "").strip()
        if not s:
            s = ev.kind
        out.append(f"- [{ev.kind}] {s}")
    return out


def _template_narrative(
    *,
    tracked_count: int,
    events: List[WorkspaceEvent],
    file_changes: List[dict],
) -> str:
    lines: List[str] = []
    if tracked_count:
        lines.append(
            f"I'm watching {tracked_count} file(s) across this workspace."
        )
    if file_changes:
        lines.append(
            f"Since I last looked, {len(file_changes)} of them changed:"
        )
        for ch in file_changes[:8]:
            lines.append(f"  - {ch.get('path')}: {ch.get('change')}")
    if events:
        lines.append("Recent workspace events:")
        for line in _summarise_events(events[-8:]):
            lines.append(f"  {line}")
    if not lines:
        return ""
    return "\n".join(lines)


def render_changelog(
    events: Iterable[WorkspaceEvent],
    *,
    active_project_id: Optional[str] = None,
    title: str = "Workspace changelog",
) -> str:
    """Roll the activity log into a dated, grouped markdown changelog
    (feature 2B).

    Events are scoped to ``active_project_id`` (when given), grouped by
    calendar day (descending), and rendered one bullet per event. Returns
    '' when there is nothing to report. Pure — no I/O.
    """
    active = (active_project_id or "").strip().lower()
    by_day: dict = {}
    order: List[str] = []
    for ev in events:
        if active:
            # Match the wake-up-prefix scoping (filter_events_for_project):
            # keep project-agnostic events AND events whose project can be
            # DERIVED from a projects/<id>/ path — not only the explicit
            # project_id stamp. The bare stamp check dropped legacy events
            # and every project-agnostic note from the changelog.
            owner = derive_event_project_id(ev)
            if owner and owner != active:
                continue
        # ISO timestamp → date prefix (YYYY-MM-DD); fall back to the whole
        # string if it isn't shaped as expected.
        ts = (ev.timestamp or "").strip()
        day = ts[:10] if len(ts) >= 10 else (ts or "undated")
        summary = (ev.summary or "").strip() or ev.kind
        if day not in by_day:
            by_day[day] = []
            order.append(day)
        by_day[day].append(f"- [{ev.kind}] {summary}")
    if not order:
        return ""
    # Real dates newest-first; the "undated" bucket last (a descending
    # string sort otherwise floats "undated" above every real date since
    # 'u' > '2').
    real_days = sorted((d for d in order if d != "undated"), reverse=True)
    if "undated" in order:
        real_days.append("undated")
    lines: List[str] = [f"# {title}", ""]
    for day in real_days:
        lines.append(f"## {day}")
        lines.extend(by_day[day])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class WorkspaceNarrative:
    """Persistent first-person workspace summary."""

    def __init__(
        self,
        root: Path,
        *,
        critique_fn: Optional[CritiqueFn] = None,
        max_recent_events: int = 12,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.path = self.root / NARRATIVE_FILENAME
        self.critique_fn = critique_fn
        self.max_recent_events = int(max_recent_events)
        self.enabled = bool(enabled)
        # Fingerprint of the last successfully-persisted regeneration's
        # INPUT (the deterministic template). Same idempotency discipline
        # as the selfhood narrative: identical workspace state → skip the
        # LLM round-trip and the redundant persist. In-memory on purpose
        # (fresh boot regenerates once).
        self._last_input_key = ""

    def latest(self) -> str:
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("workspace narrative read failed: %s", e)
            return ""

    def _persist(self, text: str) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".txt.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative persist failed: %s", e)
            return False

    async def regenerate(
        self,
        *,
        activity: WorkspaceActivity,
        state: WorkspaceStateThread,
        file_changes: Optional[List[dict]] = None,
    ) -> str:
        if not self.enabled:
            return ""
        events = activity.recent(limit=self.max_recent_events)
        tracked_count = len(state.tracked_files()) if state else 0
        changes = list(file_changes or [])

        template = _template_narrative(
            tracked_count=tracked_count,
            events=events,
            file_changes=changes,
        )
        if not template:
            return ""

        # Idempotency guard: the template is a deterministic render of
        # ALL inputs (tracked files, events, changes), so an unchanged
        # template means an unchanged workspace — skip the regeneration.
        # Never skip the very first write (no narrative on disk yet).
        input_key = hashlib.sha1(template.encode("utf-8")).hexdigest()
        if input_key == self._last_input_key and self.latest():
            logger.debug(
                "workspace narrative input unchanged; skipping regeneration"
            )
            return ""

        if self.critique_fn is None:
            self._persist(template)
            self._last_input_key = input_key
            return template

        prompt = (
            "You are the agent's own workspace narrator. Rewrite the "
            "following workspace activity as a short first-person "
            "paragraph (3-5 sentences) describing what is happening "
            "in the user's workspace. Do not invent details. Keep it "
            "tight and concrete.\n\n"
            f"{template}\n"
        )
        try:
            text = await self.critique_fn(prompt)
            if not (text or "").strip():
                # Make the degraded mode visible — a whole night of raw
                # template output went unnoticed because this was silent.
                logger.warning(
                    "workspace narrative critique returned empty content; "
                    "persisting raw template"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative LLM critique failed: %s", e)
            text = ""
        text = (text or "").strip() or template
        # Commit the idempotency key only on a successful persist — a
        # transient disk error otherwise becomes persistent staleness
        # (the unchanged-input guard keeps serving the stale file).
        if self._persist(text):
            self._last_input_key = input_key
        return text
