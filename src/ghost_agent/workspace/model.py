"""WorkspaceModel — facade the rest of the agent talks to.

Symmetric with ``selfhood.SelfModel``. One attribute on the context
(``context.workspace_model``) gives:
  * a wake-up prefix renderer for the prompt assembly path,
  * a post-event capture hook (task outcomes, research artifacts,
    file-watch results, command outcomes),
  * an idle-phase narrative consolidator.

Disabled mode: when ``enabled=False`` (e.g. ``--no-memory`` or
``--no-workspace-model``), every method is a no-op. Lifespan always
attaches a facade so call sites don't have to branch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from .activity import WorkspaceActivity
from .narrative import WorkspaceNarrative
from .reactions import check_changed_python_files
from .recognition import build_workspace_prefix
from .schema import (
    CommandOutcome,
    ResearchArtifact,
    TaskOutcome,
    WorkspaceEvent,
    _PROJECTS_PATH_RE,
)
from .state import WorkspaceStateThread

logger = logging.getLogger("GhostWorkspace")


CritiqueFn = Callable[[str], Awaitable[str]]


class WorkspaceModel:
    """Top-level workspace facade. Cheap to construct."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = True,
        narrative_critique_fn: Optional[CritiqueFn] = None,
        max_recent_events_for_narrative: int = 12,
    ):
        self.root = Path(root)
        self.enabled = bool(enabled)
        # The project the agent is currently working in ("" = none/free
        # chat). Kept in sync by the agent each request so (a) recorded
        # events are stamped with their owning project and (b) the wake-up
        # prefix scopes events to THIS project — preventing a prior
        # project's file:// pulls from bleeding into a new one.
        self.current_project_id: str = ""
        # Per-session navigation counter (feature 2C): URL → visit count,
        # used to suggest caching / a strategy switch when the agent keeps
        # re-fetching the same page. In-memory, bounded, reset each boot.
        self._nav_counts: dict = {}
        if self.enabled:
            self.activity: Optional[WorkspaceActivity] = WorkspaceActivity(
                self.root, enabled=True,
            )
            self.state: Optional[WorkspaceStateThread] = WorkspaceStateThread(
                self.root, enabled=True,
            )
            self.narrative: Optional[WorkspaceNarrative] = WorkspaceNarrative(
                self.root,
                critique_fn=narrative_critique_fn,
                max_recent_events=max_recent_events_for_narrative,
                enabled=True,
            )
        else:
            self.activity = None
            self.state = None
            self.narrative = None

    # -----------------------------------------------------------------
    # Wake-up path
    # -----------------------------------------------------------------

    def build_wakeup_prefix(self, active_project_id: Optional[str] = None) -> str:
        """Compose the workspace prefix the prompt assembly path
        splices into the system prompt. Empty when there's nothing to
        say.

        ``active_project_id`` (defaults to ``self.current_project_id``)
        scopes the rendered events to the current project so a prior
        project's research artifacts / narrative don't leak in.

        We route through the model's own ``scan_tracked()`` (not the
        state thread's bare scan) so any detected changes are MIRRORED
        into the activity log. That keeps the on-disk record
        consistent regardless of who triggered the scan, and lets the
        ``workspace action=changes`` tool fall back to the recent
        file_changed events when the prefix scan already consumed
        them in this turn."""
        if not self.enabled:
            return ""
        active = active_project_id if active_project_id is not None else self.current_project_id
        narrative_text = self.narrative.latest() if self.narrative else ""
        file_changes: List[dict] = []
        if self.state is not None and self.state.tracked_files():
            try:
                file_changes = self.scan_tracked()
            except Exception as e:  # noqa: BLE001
                logger.debug("workspace scan_tracked failed: %s", e)
        # Scope tracked-file changes the same way: a change to a file under
        # a DIFFERENT project's directory isn't relevant to this turn.
        if active and file_changes:
            _a = str(active).strip().lower()
            scoped = []
            for ch in file_changes:
                m = _PROJECTS_PATH_RE.search(str(ch.get("path", "")))
                if not m or m.group(1).lower() == _a:
                    scoped.append(ch)
            file_changes = scoped
        # React to the (scoped) changes: flag any changed .py file that no
        # longer parses, so the wake-up prefix warns about it (feature 2A).
        file_warnings: List[dict] = []
        if file_changes:
            try:
                file_warnings = check_changed_python_files(file_changes)
            except Exception as e:  # noqa: BLE001
                logger.debug("react_to_changes failed: %s", e)
        return build_workspace_prefix(
            activity=self.activity,
            state=self.state,
            narrative=narrative_text,
            file_changes=file_changes,
            file_warnings=file_warnings,
            active_project_id=active or None,
        )

    # -----------------------------------------------------------------
    # Hot-path capture APIs
    # -----------------------------------------------------------------

    def record_task_outcome(
        self,
        *,
        job_id: str,
        task_name: str = "",
        outcome: str = "unknown",
        duration_seconds: float = 0.0,
        summary: str = "",
        error: str = "",
    ) -> Optional[TaskOutcome]:
        """Append a scheduled-task outcome to the activity log."""
        if not self.enabled or self.activity is None:
            return None
        try:
            t = TaskOutcome(
                job_id=str(job_id or ""),
                task_name=str(task_name or ""),
                outcome=str(outcome or "unknown"),
                duration_seconds=float(duration_seconds or 0.0),
                summary=str(summary or "")[:600],
                error=str(error or "")[:400],
            )
            self.activity.append(WorkspaceEvent(
                kind="task_outcome",
                payload=t.to_dict(),
                summary=(
                    f"task {task_name or job_id}: {outcome}"
                    if not error else f"task {task_name or job_id}: failed ({error[:80]})"
                ),
                project_id=self.current_project_id,
            ))
            return t
        except Exception as e:  # noqa: BLE001
            logger.debug("record_task_outcome skipped: %s", e)
            return None

    def record_research_artifact(
        self,
        *,
        url: str,
        title: str = "",
        source: str = "",
        note: str = "",
    ) -> Optional[ResearchArtifact]:
        """Record a URL we pulled. Idempotent — dedupes against the
        seen-URL set on the state thread so repeated researches don't
        flood the log."""
        if not self.enabled or self.activity is None or self.state is None:
            return None
        if not (url or "").strip():
            return None
        try:
            already_seen = not self.state.mark_url_seen(url)
            if already_seen:
                return None  # silent dedup
            art = ResearchArtifact(
                url=str(url),
                title=str(title or "")[:200],
                source=str(source or ""),
                note=str(note or "")[:200],
            )
            self.activity.append(WorkspaceEvent(
                kind="research",
                payload=art.to_dict(),
                summary=f"pulled {url}"[:200],
                project_id=self.current_project_id,
            ))
            return art
        except Exception as e:  # noqa: BLE001
            logger.debug("record_research_artifact skipped: %s", e)
            return None

    def record_navigation(self, url: str, *, threshold: int = 3) -> Optional[str]:
        """Count visits to ``url`` this session; on EXACTLY the
        ``threshold``-th identical visit, return a one-line suggestion to
        cache the result or switch strategy (feature 2C), and record it as a
        workspace note so it also surfaces in the wake-up prefix. Returns
        None otherwise. Fires once (at the threshold) to avoid spamming every
        subsequent re-fetch.
        """
        if not self.enabled:
            return None
        u = (url or "").strip()
        if not u:
            return None
        n = self._nav_counts.get(u, 0) + 1
        self._nav_counts[u] = n
        # Bound the map so a long crawl can't grow it without limit.
        if len(self._nav_counts) > 512:
            self._nav_counts.pop(next(iter(self._nav_counts)), None)
        if n == int(threshold):
            suggestion = (
                f"You've navigated to {u} {n} times this session. If you're "
                f"re-checking the same page for a change or an error, cache "
                f"the last result or switch strategy instead of re-fetching it."
            )
            try:
                self.note(suggestion, url=u, count=n, repeat=True)
            except Exception:  # noqa: BLE001
                pass
            return suggestion
        return None

    def has_seen_url(self, url: str) -> bool:
        if not self.enabled or self.state is None:
            return False
        try:
            return self.state.has_seen_url(url)
        except Exception:  # noqa: BLE001
            return False

    def record_command_outcome(
        self,
        *,
        command: str,
        exit_code: int = 0,
        duration_seconds: float = 0.0,
        note: str = "",
    ) -> Optional[CommandOutcome]:
        """Capture a significant command outcome (long, failed, or
        mutating). Caller decides what's significant — we just write."""
        if not self.enabled or self.activity is None:
            return None
        try:
            c = CommandOutcome(
                command=str(command or "")[:400],
                exit_code=int(exit_code),
                duration_seconds=float(duration_seconds or 0.0),
                note=str(note or "")[:200],
            )
            self.activity.append(WorkspaceEvent(
                kind="command",
                payload=c.to_dict(),
                summary=(
                    f"ran `{(command or '')[:80]}` exit={exit_code}"
                ),
                project_id=self.current_project_id,
            ))
            return c
        except Exception as e:  # noqa: BLE001
            logger.debug("record_command_outcome skipped: %s", e)
            return None

    def note(self, summary: str, **payload) -> Optional[WorkspaceEvent]:
        """Generic free-form workspace event. Used by the workspace_track
        tool when the agent wants to record something that isn't a
        file / task / research / command outcome."""
        if not self.enabled or self.activity is None:
            return None
        summary = (summary or "").strip()
        if not summary:
            return None
        try:
            ev = WorkspaceEvent(
                kind="note", payload=dict(payload), summary=summary[:600],
                project_id=self.current_project_id,
            )
            self.activity.append(ev)
            return ev
        except Exception as e:  # noqa: BLE001
            logger.debug("note skipped: %s", e)
            return None

    # -----------------------------------------------------------------
    # File watching
    # -----------------------------------------------------------------

    def track_file(self, path: str, *, label: str = ""):
        if not self.enabled or self.state is None:
            return None
        return self.state.track_file(path, label=label)

    def untrack_file(self, path: str) -> bool:
        if not self.enabled or self.state is None:
            return False
        return self.state.untrack_file(path)

    def scan_tracked(self) -> List[dict]:
        if not self.enabled or self.state is None:
            return []
        try:
            changes = self.state.scan_tracked()
        except Exception as e:  # noqa: BLE001
            logger.debug("scan_tracked failed: %s", e)
            return []
        # Mirror each change into the activity log so the narrative
        # consolidation can reason over it.
        if changes and self.activity is not None:
            for ch in changes:
                try:
                    # Derive the owning project from the changed path when
                    # possible (a tracked file lives under projects/<id>/),
                    # else fall back to the active project.
                    _m = _PROJECTS_PATH_RE.search(str(ch.get("path", "")))
                    _pid = _m.group(1).lower() if _m else self.current_project_id
                    self.activity.append(WorkspaceEvent(
                        kind="file_changed",
                        payload=ch,
                        summary=f"{ch.get('path')}: {ch.get('change')}",
                        project_id=_pid,
                    ))
                except Exception:  # noqa: BLE001
                    continue
        return changes

    def mark_session_boot(self) -> None:
        if not self.enabled or self.state is None:
            return
        try:
            self.state.touch_session()
        except Exception as e:  # noqa: BLE001
            logger.debug("mark_session_boot skipped: %s", e)

    # -----------------------------------------------------------------
    # Idle path
    # -----------------------------------------------------------------

    async def consolidate_narrative(self) -> str:
        if not self.enabled or self.activity is None or self.narrative is None:
            return ""
        # Roll recent activity into a dated CHANGELOG alongside the prose
        # narrative (feature 2B). Best-effort — never blocks the narrative.
        try:
            self.write_changelog()
        except Exception as e:  # noqa: BLE001
            logger.debug("workspace changelog write skipped: %s", e)
        try:
            return await self.narrative.regenerate(
                activity=self.activity, state=self.state,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative consolidation failed: %s", e)
            return ""

    def write_changelog(self, *, max_events: int = 200) -> Optional[Path]:
        """Render recent activity into a per-project CHANGELOG.md under the
        workspace root and return its path (feature 2B). No-op (None) when
        disabled, no activity, or nothing to report.
        """
        if not self.enabled or self.activity is None:
            return None
        from .narrative import render_changelog
        events = self.activity.recent(limit=max_events)
        active = self.current_project_id or ""
        body = render_changelog(events, active_project_id=active or None)
        if not body:
            return None
        # Scope the file per active project so projects don't share one log.
        name = f"CHANGELOG.{active}.md" if active else "CHANGELOG.md"
        path = self.root / name
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(path)
            return path
        except Exception as e:  # noqa: BLE001
            logger.debug("changelog persist failed: %s", e)
            return None

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "root": str(self.root),
            "tracked_files": len(self.state.tracked_files()) if self.state else 0,
            "seen_urls": (
                len(self.state.state.seen_urls) if self.state else 0
            ),
            "event_count": self.activity.count() if self.activity else 0,
            "event_kinds": self.activity.kinds() if self.activity else {},
            "narrative_present": (
                bool(self.narrative.latest()) if self.narrative else False
            ),
            "last_session_at": (
                self.state.state.last_session_at if self.state else ""
            ),
        }
