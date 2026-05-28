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

import logging
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

from .activity import WorkspaceActivity
from .schema import WorkspaceEvent
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

    def latest(self) -> str:
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("workspace narrative read failed: %s", e)
            return ""

    def _persist(self, text: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".txt.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative persist failed: %s", e)

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

        if self.critique_fn is None:
            self._persist(template)
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
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative LLM critique failed: %s", e)
            text = ""
        text = (text or "").strip() or template
        self._persist(text)
        return text
