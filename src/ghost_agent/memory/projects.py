"""Persistent store for long-term projects.

Projects are the top-level container for multi-session work. Each project
owns a tree of tasks (mirroring `core.planning.TaskNode` fields so a
`ProjectPlan` wrapper can rehydrate a `TaskTree` from rows), a stream of
artifacts produced by tasks, and an append-only event log used for
audit, resumption briefings, and the "go back and forth" UX.

The store is deliberately schema-first and framework-free: SQLite only,
no ORM, so it can be opened by the API server, the Slack bot, and the
dream consolidator without cross-imports.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


class ProjectKind(str, Enum):
    CODING = "CODING"
    GENERAL = "GENERAL"


class ProjectStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DONE = "DONE"
    ARCHIVED = "ARCHIVED"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    workspace_dir TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    parent_id TEXT,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    dependency_type TEXT NOT NULL DEFAULT 'ALL',
    alternatives_json TEXT NOT NULL DEFAULT '[]',
    postconditions_json TEXT NOT NULL DEFAULT '[]',
    result_summary TEXT NOT NULL DEFAULT '',
    failure_reason TEXT NOT NULL DEFAULT '',
    revision_count INTEGER NOT NULL DEFAULT 0,
    actual_tool_used TEXT,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    actual_cost REAL NOT NULL DEFAULT 0.0,
    depth INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(project_id, status);

CREATE TABLE IF NOT EXISTS task_artifacts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_task ON task_artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON task_artifacts(project_id);

CREATE TABLE IF NOT EXISTS project_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    task_id TEXT,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    ts REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_project ON project_events(project_id, ts);
"""


_ARTIFACT_KINDS = {"file", "url", "note", "tool_call"}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> float:
    return time.time()


class ProjectStore:
    """SQLite-backed store for projects, tasks, artifacts, and events.

    Single-writer discipline is enforced with an RLock. The store does
    not cache rows — callers that need repeated access should hold the
    returned dicts. Schema migrations are handled by additive
    ``ALTER TABLE`` calls in ``_init_db``.
    """

    def __init__(self, memory_dir: Path, sandbox_root: Optional[Path] = None,
                 db_name: str = "projects.db"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / db_name
        self.sandbox_root = Path(sandbox_root) if sandbox_root else None
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------ projects

    def create_project(self, title: str, kind: str = "GENERAL",
                       goal: str = "", metadata: Optional[Dict[str, Any]] = None,
                       workspace_dir: Optional[str] = None) -> str:
        if not title or not title.strip():
            raise ValueError("title must be non-empty")
        kind_norm = ProjectKind(kind.upper()).value
        project_id = _new_id()
        now = _now()
        meta_json = json.dumps(metadata or {})
        workspace = workspace_dir or self._default_workspace(project_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO projects(id, title, kind, goal, status, workspace_dir, "
                "metadata_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (project_id, title.strip(), kind_norm, goal, ProjectStatus.ACTIVE.value,
                 workspace, meta_json, now, now),
            )
            conn.commit()
        self.log_event(project_id, None, "project_created",
                       {"title": title, "kind": kind_norm})
        if workspace:
            try:
                Path(workspace).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning("Could not create workspace dir %s: %s", workspace, e)
        return project_id

    def _default_workspace(self, project_id: str) -> Optional[str]:
        if not self.sandbox_root:
            return None
        return str(self.sandbox_root / "projects" / project_id)

    def list_projects(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY updated_at DESC",
                    (status_filter.upper(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM projects ORDER BY updated_at DESC"
                ).fetchall()
            return [self._row_to_project(r) for r in rows]

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            return self._row_to_project(row) if row else None

    def update_project(self, project_id: str, **fields) -> bool:
        if not fields:
            return False
        allowed = {"title", "kind", "goal", "status", "workspace_dir", "metadata"}
        sets = []
        values: List[Any] = []
        for key, val in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown project field: {key}")
            if key == "metadata":
                sets.append("metadata_json = ?")
                values.append(json.dumps(val or {}))
            elif key == "status":
                sets.append("status = ?")
                values.append(ProjectStatus(val.upper()).value)
            elif key == "kind":
                sets.append("kind = ?")
                values.append(ProjectKind(val.upper()).value)
            else:
                sets.append(f"{key} = ?")
                values.append(val)
        sets.append("updated_at = ?")
        values.append(_now())
        values.append(project_id)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", values
            )
            conn.commit()
            updated = cur.rowcount > 0
        if updated:
            self.log_event(project_id, None, "project_updated", {"fields": list(fields.keys())})
        return updated

    def delete_project(self, project_id: str, hard: bool = False) -> bool:
        """Archive (soft) or delete (hard) a project.

        Soft-delete is the default: it flips status to ARCHIVED so the
        project remains resumable. Hard delete removes the rows (and
        cascades to tasks/artifacts/events via FK).
        """
        if hard:
            with self._lock, self._connect() as conn:
                cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                conn.commit()
                return cur.rowcount > 0
        return self.update_project(project_id, status=ProjectStatus.ARCHIVED.value)

    def _row_to_project(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        except Exception:
            d["metadata"] = {}
        return d

    def ensure_workspace(self, project_id: str) -> Optional[Path]:
        """Return the workspace Path for a project, creating it if missing.

        Returns None when the store has no sandbox_root configured and the
        project has no workspace_dir set — callers should treat that as
        "no isolated workspace available" rather than an error.
        """
        proj = self.get_project(project_id)
        if not proj:
            return None
        path_str = proj.get("workspace_dir")
        if not path_str:
            return None
        path = Path(path_str)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not ensure workspace %s: %s", path, e)
            return None
        return path

    # ------------------------------------------------------------------ tasks

    def add_task(self, project_id: str, description: str,
                 parent_id: Optional[str] = None,
                 status: str = "PENDING",
                 dependency_type: str = "ALL",
                 alternatives: Optional[List[str]] = None,
                 postconditions: Optional[List[str]] = None,
                 estimated_cost: float = 0.0,
                 position: Optional[int] = None) -> str:
        if not description or not description.strip():
            raise ValueError("description must be non-empty")
        task_id = _new_id()
        now = _now()
        depth = 0
        if parent_id:
            parent = self.get_task(parent_id)
            if not parent:
                raise ValueError(f"parent task not found: {parent_id}")
            if parent["project_id"] != project_id:
                raise ValueError("parent task belongs to a different project")
            depth = int(parent["depth"]) + 1
        with self._lock, self._connect() as conn:
            if position is None:
                if parent_id is None:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) AS m FROM tasks "
                        "WHERE project_id = ? AND parent_id IS NULL",
                        (project_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) AS m FROM tasks "
                        "WHERE parent_id = ?", (parent_id,),
                    ).fetchone()
                position = int(row["m"]) + 1
            conn.execute(
                "INSERT INTO tasks(id, project_id, parent_id, description, status, "
                "dependency_type, alternatives_json, postconditions_json, "
                "result_summary, failure_reason, revision_count, actual_tool_used, "
                "estimated_cost, actual_cost, depth, position, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, project_id, parent_id, description.strip(),
                 status.upper(), dependency_type.upper(),
                 json.dumps(alternatives or []), json.dumps(postconditions or []),
                 "", "", 0, None, estimated_cost, 0.0, depth, position, now, now),
            )
            conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            conn.commit()
        self.log_event(project_id, task_id, "task_added",
                       {"description": description, "parent_id": parent_id})
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return self._row_to_task(row) if row else None

    def list_tasks(self, project_id: str,
                   status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE project_id = ? AND status = ? "
                    "ORDER BY depth ASC, position ASC",
                    (project_id, status_filter.upper()),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE project_id = ? "
                    "ORDER BY depth ASC, position ASC",
                    (project_id,),
                ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def update_task(self, task_id: str, **fields) -> bool:
        if not fields:
            return False
        allowed = {"description", "status", "dependency_type", "alternatives",
                   "postconditions", "result_summary", "failure_reason",
                   "revision_count", "actual_tool_used", "estimated_cost",
                   "actual_cost", "parent_id", "position"}
        sets = []
        values: List[Any] = []
        for key, val in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown task field: {key}")
            if key in ("alternatives", "postconditions"):
                sets.append(f"{key}_json = ?")
                values.append(json.dumps(val or []))
            elif key == "status":
                sets.append("status = ?")
                values.append(str(val).upper())
            elif key == "dependency_type":
                sets.append("dependency_type = ?")
                values.append(str(val).upper())
            else:
                sets.append(f"{key} = ?")
                values.append(val)
        now = _now()
        sets.append("updated_at = ?")
        values.append(now)
        values.append(task_id)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", values
            )
            row = conn.execute(
                "SELECT project_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE projects SET updated_at = ? WHERE id = ?",
                    (now, row["project_id"]),
                )
            conn.commit()
            updated = cur.rowcount > 0
        if updated and row:
            self.log_event(row["project_id"], task_id, "task_updated",
                           {"fields": list(fields.keys())})
            # Auto-roll-up: when a task status changes, the project as a
            # whole may have finished. If every task is in a terminal
            # state, transition the project. (DONE if all DONE; FAILED
            # if any task ended in FAILED.) Skip if the project is
            # already in a terminal state — never auto-undo a manual
            # ARCHIVE.
            if "status" in fields:
                self._maybe_rollup_project_status(row["project_id"])
        return updated

    def _maybe_rollup_project_status(self, project_id: str) -> None:
        """Transition `project_id` to a terminal status if all its tasks
        have reached one. No-op if any task is still open or if the
        project is already terminal.

        Rules:
          * any task FAILED → project FAILED (mapped to DONE-with-fail-
            note since ProjectStatus has no FAILED member; we leave
            status DONE but emit an event so the rollup is observable).
          * all tasks DONE → project DONE.
          * otherwise → no-op.
        """
        proj = self.get_project(project_id)
        if not proj:
            return
        current = (proj.get("status") or "").upper()
        if current in {ProjectStatus.DONE.value, ProjectStatus.ARCHIVED.value}:
            return
        tasks = self.list_tasks(project_id)
        if not tasks:
            return
        terminal = {"DONE", "FAILED", "CANCELLED"}
        statuses = [str(t.get("status", "")).upper() for t in tasks]
        if not all(s in terminal for s in statuses):
            return
        # All terminal. Roll the project up.
        new_status = ProjectStatus.DONE.value
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, _now(), project_id),
            )
            conn.commit()
        self.log_event(
            project_id, None, "project_auto_rollup",
            {"new_status": new_status,
             "had_failures": any(s == "FAILED" for s in statuses)},
        )

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its descendants (via FK cascade on parent_id=NULL
        we can't rely on cascade, so we delete manually)."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return False
            project_id = row["project_id"]
            to_delete = [task_id]
            frontier = [task_id]
            while frontier:
                nxt: List[str] = []
                for tid in frontier:
                    child_rows = conn.execute(
                        "SELECT id FROM tasks WHERE parent_id = ?", (tid,)
                    ).fetchall()
                    for cr in child_rows:
                        to_delete.append(cr["id"])
                        nxt.append(cr["id"])
                frontier = nxt
            conn.executemany(
                "DELETE FROM tasks WHERE id = ?",
                [(tid,) for tid in to_delete],
            )
            conn.commit()
        self.log_event(project_id, task_id, "task_deleted",
                       {"cascaded": len(to_delete) - 1})
        return True

    def _row_to_task(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["alternatives"] = json.loads(d.pop("alternatives_json") or "[]")
        except Exception:
            d["alternatives"] = []
        try:
            d["postconditions"] = json.loads(d.pop("postconditions_json") or "[]")
        except Exception:
            d["postconditions"] = []
        return d

    # ------------------------------------------------------------------ artifacts

    def add_artifact(self, task_id: str, kind: str, payload: str) -> str:
        if kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unknown artifact kind: {kind}")
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"task not found: {task_id}")
        art_id = _new_id()
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO task_artifacts(id, task_id, project_id, kind, payload, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (art_id, task_id, task["project_id"], kind, payload, now),
            )
            conn.commit()
        self.log_event(task["project_id"], task_id, "artifact_added",
                       {"kind": kind, "artifact_id": art_id})
        return art_id

    def list_artifacts(self, project_id: Optional[str] = None,
                       task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT * FROM task_artifacts WHERE task_id = ? ORDER BY created_at ASC",
                    (task_id,),
                ).fetchall()
            elif project_id:
                rows = conn.execute(
                    "SELECT * FROM task_artifacts WHERE project_id = ? ORDER BY created_at ASC",
                    (project_id,),
                ).fetchall()
            else:
                raise ValueError("must provide project_id or task_id")
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ events

    def log_event(self, project_id: str, task_id: Optional[str], event_type: str,
                  payload: Optional[Dict[str, Any]] = None) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO project_events(project_id, task_id, type, payload_json, ts) "
                "VALUES (?,?,?,?,?)",
                (project_id, task_id, event_type, json.dumps(payload or {}), _now()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_events(self, project_id: str, limit: int = 50,
                    event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM project_events WHERE project_id = ? AND type = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (project_id, event_type, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM project_events WHERE project_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (project_id, int(limit)),
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["payload"] = json.loads(d.pop("payload_json") or "{}")
                except Exception:
                    d["payload"] = {}
                out.append(d)
            return out
