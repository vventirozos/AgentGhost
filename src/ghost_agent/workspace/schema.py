"""Dataclasses for the workspace module.

Keeping the schema flat and JSON-serialisable mirrors the selfhood
choice — explicit migrations, clean diffs, no pydantic.
"""

from __future__ import annotations

import datetime
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "v1"


def _utcnow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


@dataclass
class FileSnapshot:
    """A single stat reading of a tracked file. Cached on disk so the
    next session can diff against it.

    ``digest`` is optional because not every caller can afford to hash
    a large file on every check — the watchdog falls back to (mtime,
    size) when digest is empty."""

    path: str = ""
    size: int = 0
    mtime_ns: int = 0
    digest: str = ""  # short content hash; empty when not computed
    captured_at: str = field(default_factory=_utcnow_iso)
    exists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class TrackedFile:
    """A path the agent (or user) wants to watch across sessions.

    The watcher reads the current stat on demand and compares against
    ``last_snapshot``. Empty ``last_snapshot.path`` means "never seen
    before"; the first read primes it."""

    path: str = ""
    label: str = ""            # optional human descriptor
    added_at: str = field(default_factory=_utcnow_iso)
    last_seen_at: str = ""
    last_snapshot: Optional[FileSnapshot] = None


@dataclass
class TaskOutcome:
    """One execution of a scheduled task. Appended to the activity
    log every time a scheduler-fired prompt finishes."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str = ""
    task_name: str = ""
    fired_at: str = field(default_factory=_utcnow_iso)
    finished_at: str = ""
    duration_seconds: float = 0.0
    outcome: str = "unknown"  # passed | failed | unknown
    summary: str = ""         # short prose, agent-supplied or auto
    error: str = ""           # exception class + message when failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskOutcome":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class ResearchArtifact:
    """A URL / document the agent pulled during research. Persisted so
    a later research query can dedup against it."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    url: str = ""
    title: str = ""
    source: str = ""           # "deep_research" | "browser" | "web_search"
    captured_at: str = field(default_factory=_utcnow_iso)
    note: str = ""             # one-line agent-supplied descriptor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchArtifact":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class CommandOutcome:
    """A significant execute-tool command outcome we want to remember
    across sessions — long runs, failures, mutation-heavy commands."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    command: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    fired_at: str = field(default_factory=_utcnow_iso)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandOutcome":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class WorkspaceEvent:
    """A generic envelope written to the activity log. The ``kind``
    field plus ``payload`` lets one JSONL file hold heterogeneous
    events without a dedicated file per kind — same pattern as
    distill / journal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kind: str = "note"  # file_changed | task_outcome | research | command | note
    timestamp: str = field(default_factory=_utcnow_iso)
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    # Which project this event belongs to ("" = project-agnostic, e.g. a
    # web research pull or a generic note). Stamped at record time so the
    # wake-up prefix can scope events to the ACTIVE project and not bleed a
    # PRIOR project's files into a new one (the cross-project file:// pull
    # that trapped a fresh build in an infinite "read missing file" loop).
    project_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceEvent":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex),
            kind=str(d.get("kind") or "note"),
            timestamp=str(d.get("timestamp") or _utcnow_iso()),
            payload=dict(d.get("payload") or {}),
            summary=str(d.get("summary") or ""),
            project_id=str(d.get("project_id") or ""),
        )


# Project ids are hex hashes (e.g. ``991e52bce912``); the canonical
# on-disk layout scopes a project's files under ``projects/<id>/``. We
# anchor on that segment to recover an event's project from a file path
# even for LEGACY events written before ``project_id`` stamping existed.
_PROJECTS_PATH_RE = re.compile(r"projects/([0-9a-fA-F]{6,})\b")


def derive_event_project_id(ev: "WorkspaceEvent") -> str:
    """Best-effort: which project does this workspace event belong to?

    Returns the lowercased project id, or ``""`` when the event is
    project-agnostic (a web research pull, a generic note/command with no
    project path). Prefers the explicit ``project_id`` stamp; falls back
    to parsing a ``projects/<id>/`` segment out of the event's URL / path /
    summary so events logged before stamping existed are still scoped."""
    pid = (getattr(ev, "project_id", "") or "").strip()
    if pid:
        return pid.lower()
    payload = getattr(ev, "payload", None) or {}
    hay = " ".join(
        str(payload.get(k, "")) for k in ("url", "path", "command", "filename")
    ) + " " + (getattr(ev, "summary", "") or "")
    m = _PROJECTS_PATH_RE.search(hay)
    return m.group(1).lower() if m else ""


def filter_events_for_project(
    events: List["WorkspaceEvent"], active_project_id: Optional[str],
) -> List["WorkspaceEvent"]:
    """Keep events that are project-agnostic OR belong to the active
    project; drop events that clearly belong to a DIFFERENT project.

    When ``active_project_id`` is falsy (no project in context), keep
    everything — a free-chat / cross-project view is correct there."""
    active = (active_project_id or "").strip().lower()
    if not active:
        return list(events)
    kept: List["WorkspaceEvent"] = []
    for ev in events:
        owner = derive_event_project_id(ev)
        if not owner or owner == active:
            kept.append(ev)
    return kept


@dataclass
class WorkspaceState:
    """Cross-session state thread. Persisted as one JSON file. Bounded
    so the wake-up prefix stays compact."""

    schema_version: str = SCHEMA_VERSION
    tracked_files: List[TrackedFile] = field(default_factory=list)
    last_session_at: str = ""
    # When the workspace was touched BEFORE the current boot. touch_session
    # rolls last_session_at into this at startup so the cross-session
    # "last touched on …" line survives (last_session_at itself becomes the
    # current boot time immediately at boot).
    prior_session_at: str = ""
    # Set of URLs the agent has already pulled — kept on the state
    # thread (rather than re-scanning the activity log every dedup
    # call) so research dedup is O(1).
    seen_urls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tracked_files": [
                {
                    "path": tf.path,
                    "label": tf.label,
                    "added_at": tf.added_at,
                    "last_seen_at": tf.last_seen_at,
                    "last_snapshot": (
                        tf.last_snapshot.to_dict()
                        if tf.last_snapshot is not None else None
                    ),
                }
                for tf in self.tracked_files
            ],
            "last_session_at": self.last_session_at,
            "prior_session_at": self.prior_session_at,
            "seen_urls": list(self.seen_urls),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceState":
        tfs: List[TrackedFile] = []
        for raw in (d.get("tracked_files") or []):
            # Skip a malformed entry rather than letting it throw and
            # discard the WHOLE state (tracked files + seen_urls +
            # timestamps) — one bad row must not wipe everything.
            if not isinstance(raw, dict):
                continue
            try:
                snap_raw = raw.get("last_snapshot")
                snap = FileSnapshot.from_dict(snap_raw) if isinstance(snap_raw, dict) else None
                tfs.append(TrackedFile(
                    path=str(raw.get("path") or ""),
                    label=str(raw.get("label") or ""),
                    added_at=str(raw.get("added_at") or _utcnow_iso()),
                    last_seen_at=str(raw.get("last_seen_at") or ""),
                    last_snapshot=snap,
                ))
            except Exception:
                continue
        return cls(
            schema_version=str(d.get("schema_version") or SCHEMA_VERSION),
            tracked_files=tfs,
            last_session_at=str(d.get("last_session_at") or ""),
            prior_session_at=str(d.get("prior_session_at") or ""),
            seen_urls=[str(u) for u in (d.get("seen_urls") or []) if u],
        )
