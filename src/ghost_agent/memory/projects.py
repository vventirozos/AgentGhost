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
import shutil
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
    # Terminal-with-failure and waiting states. Before these existed the
    # status rollup collapsed *every* terminal outcome to DONE — a project
    # whose tasks all FAILED reported as "done". FAILED/BLOCKED record a
    # genuinely unsuccessful project; NEEDS_USER marks a project parked on
    # human input (not terminal — it re-rolls forward once the task moves).
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NEEDS_USER = "NEEDS_USER"


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
    depends_on_json TEXT NOT NULL DEFAULT '[]',
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


def _canon_id(value: Optional[str]) -> str:
    """Canonical form of a project / task id.

    IDs are generated as lowercase hex (``uuid4().hex[:12]``), but an LLM
    that echoes one back in a tool call routinely mangles the case of an
    opaque hex token — e.g. ``9b5bd5cd812b`` → ``9B5Bd5Cd812B`` — which
    made the case-sensitive ``WHERE id = ?`` lookups miss with
    "project not found". Normalising every id the store accepts (strip +
    lowercase) makes generation and resolution always agree, regardless
    of how the id was transmitted. Idempotent on already-canonical ids."""
    return (value or "").strip().lower()


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
        # Optional hook fired exactly when a project *transitions* to DONE
        # (see _fire_project_done). main.py wires this to the workspace
        # cleanup sweep so a finished project's scratch files are removed
        # automatically. Left None in tests / headless contexts that don't
        # care about cleanup. Signature: (project_id: str) -> None.
        self.on_project_done = None
        self._lock = threading.RLock()
        self._init_db()

    def _fire_project_done(self, project_id: str) -> None:
        """Invoke the ``on_project_done`` hook for a just-completed project.

        Called *outside* the DB lock (filesystem cleanup must not run under
        the SQLite writer lock) and fully guarded — a cleanup failure can
        never propagate back into the status-transition path that triggered
        it. No-op when no hook is wired.
        """
        cb = getattr(self, "on_project_done", None)
        if cb is None:
            return
        try:
            cb(project_id)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "on_project_done hook failed for %s", project_id, exc_info=True
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL + a busy timeout so the cross-process readers/writers the
        # module is designed for (API server, Slack bot, dream
        # consolidator — each opens its own connection, often its own
        # process, so the in-process RLock can't serialize them) don't
        # immediately fail with "database is locked". WAL lets readers and
        # a single writer proceed concurrently; busy_timeout makes a
        # contended writer wait up to 5s instead of raising. journal_mode
        # is persistent (a no-op once set); busy_timeout is per-connection.
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
        except sqlite3.Error as e:  # pragma: no cover - exotic FS / :memory:
            logger.debug("Could not set WAL/busy_timeout pragmas: %s", e)
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive, idempotent column migrations for older DBs.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a
        DB created before a column was added to ``_SCHEMA`` lacks it. Each
        entry here is an ``ALTER TABLE ... ADD COLUMN`` guarded by a
        column-presence check, so re-running is safe.
        """
        wanted = {
            "tasks": [
                ("depends_on_json", "TEXT NOT NULL DEFAULT '[]'"),
            ],
        }
        for table, columns in wanted.items():
            try:
                existing = {
                    r["name"] for r in conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
            except sqlite3.Error:
                continue
            for name, decl in columns:
                if name not in existing:
                    try:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {decl}"
                        )
                    except sqlite3.Error as e:  # pragma: no cover
                        logger.warning(
                            "migration: could not add %s.%s: %s", table, name, e
                        )

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
        project_id = _canon_id(project_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            return self._row_to_project(row) if row else None

    def update_project(self, project_id: str, **fields) -> bool:
        project_id = _canon_id(project_id)
        if not fields:
            return False
        # Capture the prior status so a manual transition *into* DONE can
        # fire the cleanup hook exactly once (and not on a DONE→DONE no-op).
        prev_status = None
        if "status" in fields:
            existing = self.get_project(project_id)
            prev_status = (existing or {}).get("status")
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
            if "status" in fields:
                new_norm = ProjectStatus(fields["status"].upper()).value
                if (new_norm == ProjectStatus.DONE.value
                        and (prev_status or "").upper() != ProjectStatus.DONE.value):
                    self._fire_project_done(project_id)
        return updated

    def delete_project(self, project_id: str, hard: bool = False) -> bool:
        """Archive (soft) or delete (hard) a project.

        Soft-delete (``hard=False``) flips status to ARCHIVED so the
        project remains resumable — this is what ``action=archive`` uses.

        Hard delete (``hard=True``, what ``action=delete`` uses) removes
        the project COMPLETELY: the DB row plus all tasks/artifacts/events
        (FK ``ON DELETE CASCADE`` + ``PRAGMA foreign_keys=ON``, which
        includes the scratchpad-snapshot events), AND the project's
        workspace directory on disk (``<sandbox>/projects/<id>/``) so no
        files are left behind. The workspace is removed only when it
        resolves to a path strictly inside the configured sandbox root, so
        a stray/custom ``workspace_dir`` can never delete an arbitrary dir.
        """
        project_id = _canon_id(project_id)
        if not hard:
            return self.update_project(project_id, status=ProjectStatus.ARCHIVED.value)

        # Resolve the workspace path BEFORE deleting the row.
        proj = self.get_project(project_id)
        ws_str = (proj or {}).get("workspace_dir")
        if not ws_str and self.sandbox_root:
            ws_str = str(self.sandbox_root / "projects" / project_id)

        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            deleted = cur.rowcount > 0

        # Remove the on-disk workspace, but ONLY if it's safely contained in
        # the sandbox root (never the root itself, never an outside path).
        if ws_str and self.sandbox_root:
            try:
                ws_p = Path(ws_str).resolve()
                root = Path(self.sandbox_root).resolve()
                contained = ws_p != root and (
                    ws_p.is_relative_to(root) if hasattr(ws_p, "is_relative_to")
                    else str(ws_p).startswith(str(root) + "/")
                )
                if contained and ws_p.exists():
                    shutil.rmtree(ws_p, ignore_errors=True)
            except Exception as e:
                logger.warning("Could not remove workspace for %s: %s", project_id, e)
        return deleted

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
                 depends_on: Optional[List[str]] = None,
                 estimated_cost: float = 0.0,
                 position: Optional[int] = None) -> str:
        if not description or not description.strip():
            raise ValueError("description must be non-empty")
        project_id = _canon_id(project_id)
        parent_id = _canon_id(parent_id) or None
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
            depends_on_canon = [_canon_id(d) for d in (depends_on or []) if _canon_id(d)]
            conn.execute(
                "INSERT INTO tasks(id, project_id, parent_id, description, status, "
                "dependency_type, alternatives_json, postconditions_json, "
                "depends_on_json, "
                "result_summary, failure_reason, revision_count, actual_tool_used, "
                "estimated_cost, actual_cost, depth, position, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, project_id, parent_id, description.strip(),
                 status.upper(), dependency_type.upper(),
                 json.dumps(alternatives or []), json.dumps(postconditions or []),
                 json.dumps(depends_on_canon),
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
        task_id = _canon_id(task_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return self._row_to_task(row) if row else None

    def list_tasks(self, project_id: str,
                   status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        project_id = _canon_id(project_id)
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
        task_id = _canon_id(task_id)
        if not fields:
            return False
        allowed = {"description", "status", "dependency_type", "alternatives",
                   "postconditions", "depends_on", "result_summary",
                   "failure_reason", "revision_count", "actual_tool_used",
                   "estimated_cost", "actual_cost", "parent_id", "position"}
        sets = []
        values: List[Any] = []
        for key, val in fields.items():
            if key not in allowed:
                raise ValueError(f"unknown task field: {key}")
            if key == "depends_on":
                sets.append("depends_on_json = ?")
                values.append(json.dumps(
                    [_canon_id(d) for d in (val or []) if _canon_id(d)]
                ))
            elif key in ("alternatives", "postconditions"):
                sets.append(f"{key}_json = ?")
                values.append(json.dumps(val or []))
            elif key == "status":
                sets.append("status = ?")
                values.append(str(val).upper())
            elif key == "dependency_type":
                sets.append("dependency_type = ?")
                values.append(str(val).upper())
            elif key == "parent_id":
                # Canonicalize like add_task does — LLM-echoed ids arrive
                # case-mangled, and a raw value here breaks every
                # `WHERE parent_id = ?` lookup (position calc, cascade delete).
                sets.append("parent_id = ?")
                values.append(_canon_id(val) or None)
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
        """Transition `project_id` to its correct aggregate status when its
        tasks settle. No-op if work is still open or the project is locked.

        Rules (task terminal set = DONE / FAILED / BLOCKED):
          * all tasks DONE                       → project DONE
          * all tasks terminal, ≥1 FAILED/BLOCKED → project FAILED
          * all remaining-open tasks are NEEDS_USER (≥1 of them, rest
            terminal)                            → project NEEDS_USER
          * otherwise (real open work remains)   → no-op

        DONE and ARCHIVED are *locked*: once reached we never auto-undo
        them (a manual archive or a genuine completion stays put). FAILED
        and NEEDS_USER are NOT locked — a revised/answered task can roll
        the project forward to DONE on a later update.
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
        statuses = [str(t.get("status", "")).upper() for t in tasks]
        terminal = {"DONE", "FAILED", "BLOCKED"}
        failure = {"FAILED", "BLOCKED"}

        all_terminal = all(s in terminal for s in statuses)
        if all_terminal:
            if any(s in failure for s in statuses):
                new_status = ProjectStatus.FAILED.value
            else:
                new_status = ProjectStatus.DONE.value
        else:
            # Not all terminal — only roll up if the *only* non-terminal
            # work is waiting on the user. Anything else means there is
            # still autonomous work to do, so leave the project ACTIVE.
            open_states = {s for s in statuses if s not in terminal}
            if open_states and open_states == {"NEEDS_USER"}:
                new_status = ProjectStatus.NEEDS_USER.value
            else:
                return

        if new_status == current:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, _now(), project_id),
            )
            conn.commit()
        self.log_event(
            project_id, None, "project_auto_rollup",
            {"new_status": new_status,
             "had_failures": any(s in failure for s in statuses)},
        )
        # A genuine completion is the cleanup trigger. Fire only for DONE —
        # FAILED / NEEDS_USER stay resumable (a revised task can roll the
        # project forward to DONE later), so their workspace must survive.
        if new_status == ProjectStatus.DONE.value:
            self._fire_project_done(project_id)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its descendants (via FK cascade on parent_id=NULL
        we can't rely on cascade, so we delete manually)."""
        task_id = _canon_id(task_id)
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
        try:
            d["depends_on"] = json.loads(d.pop("depends_on_json") or "[]")
        except Exception:
            d["depends_on"] = []
        return d

    # ------------------------------------------------------------------ artifacts

    def add_artifact(self, task_id: str, kind: str, payload: str) -> str:
        task_id = _canon_id(task_id)
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

    def register_file_artifact(self, task_id: str, rel_path: str) -> Optional[str]:
        """Register a deliverable file (``kind='file'``) for ``task_id``,
        deduplicated within the project.

        This is the durable "keep me" marker the workspace cleanup sweep
        reads: any file path registered here survives a project's
        end-of-life sweep; everything else under the project workspace is
        deleted. Idempotent — re-registering the same project-relative path
        returns the existing artifact id instead of creating a duplicate, so
        callers can register on every DONE without accumulating rows.

        Returns the artifact id, or ``None`` when the path is blank or the
        task is unknown.
        """
        task_id = _canon_id(task_id)
        rel = (rel_path or "").strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.strip()
        if not rel:
            return None
        task = self.get_task(task_id)
        if not task:
            return None
        project_id = task["project_id"]
        for art in self.list_artifacts(project_id=project_id):
            if (art.get("kind") == "file"
                    and (art.get("payload") or "").strip() == rel):
                return art.get("id")
        return self.add_artifact(task_id, "file", rel)

    def list_artifacts(self, project_id: Optional[str] = None,
                       task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        project_id = _canon_id(project_id) or None
        task_id = _canon_id(task_id) or None
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

    # ------------------------------------------------------------------ design ledger

    # The ledger is the project's durable, compact working memory — file
    # layout, key function/API names, conventions, "what exists and where".
    # It lives in project metadata and is surfaced in the briefing every
    # turn so a fresh turn doesn't re-derive the project's shape by
    # re-reading files (the dominant cost observed on long projects).
    LEDGER_MAX_CHARS = 2400
    LEDGER_MAX_LINES = 30

    def _write_metadata(self, project_id: str, meta: Dict[str, Any]) -> None:
        """Persist a project's metadata WITHOUT logging a project_updated
        event. Ledger writes happen often; routing them through
        ``update_project`` would spam the event log (and the briefing's
        RECENT EVENTS) with bookkeeping noise and bloat project_events.
        Metadata never triggers a status transition, so skipping the event
        is safe."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE projects SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(meta or {}), _now(), _canon_id(project_id)),
            )
            conn.commit()

    def get_ledger(self, project_id: str) -> str:
        proj = self.get_project(project_id)
        return ((proj or {}).get("metadata") or {}).get("design_ledger") or ""

    def append_ledger(self, project_id: str, line: str) -> str:
        """Append one line to the project's design ledger (bounded, dedup'd
        against an identical trailing line). Returns the new ledger text."""
        line = " ".join((line or "").split())  # collapse whitespace/newlines
        proj = self.get_project(project_id)
        if not proj:
            return ""
        if not line:
            return ((proj.get("metadata") or {}).get("design_ledger") or "")
        meta = dict(proj.get("metadata") or {})
        existing = [l for l in (meta.get("design_ledger") or "").splitlines() if l.strip()]
        if not existing or existing[-1].strip() != line:
            existing.append(line)
        existing = existing[-self.LEDGER_MAX_LINES:]
        text = "\n".join(existing)
        if len(text) > self.LEDGER_MAX_CHARS:
            # Drop whole lines from the front until under the char budget.
            while existing and len("\n".join(existing)) > self.LEDGER_MAX_CHARS:
                existing.pop(0)
            text = "\n".join(existing)
        meta["design_ledger"] = text
        self._write_metadata(project_id, meta)
        return text

    def set_ledger(self, project_id: str, text: str) -> str:
        """Replace the project's design ledger wholesale (bounded)."""
        proj = self.get_project(project_id)
        if not proj:
            return ""
        meta = dict(proj.get("metadata") or {})
        meta["design_ledger"] = (text or "")[: self.LEDGER_MAX_CHARS]
        self._write_metadata(project_id, meta)
        return meta["design_ledger"]

    # The config slot is the project's durable record of settings that shape
    # how it builds/runs — env vars, key flags, dependency versions, the
    # model, ports, DB URIs — kept as a small bounded key→value map in
    # project metadata and surfaced in the briefing every turn. The design
    # ledger answers "what exists and where"; the config slot answers "under
    # what settings" — the things a fresh turn would otherwise re-discover by
    # re-reading requirements.txt / env / argv.
    CONFIG_MAX_KEYS = 30
    CONFIG_MAX_VALUE_CHARS = 200
    CONFIG_MAX_CHARS = 2000

    def get_config(self, project_id: str) -> Dict[str, str]:
        """Return the project's config map (possibly empty)."""
        proj = self.get_project(project_id)
        cfg = ((proj or {}).get("metadata") or {}).get("config") or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def set_config_value(self, project_id: str, key: str, value: str) -> Dict[str, str]:
        """Upsert one ``key → value`` config entry (bounded, last-write-wins).

        Keys are normalised (trimmed, whitespace-collapsed); an empty value
        deletes the key. The map is capped at ``CONFIG_MAX_KEYS`` (oldest
        insertion dropped first) and ``CONFIG_MAX_CHARS`` total. Returns the
        updated config map."""
        proj = self.get_project(project_id)
        if not proj:
            return {}
        key = " ".join((key or "").split())
        if not key:
            return self.get_config(project_id)
        meta = dict(proj.get("metadata") or {})
        cfg = dict(meta.get("config") or {}) if isinstance(meta.get("config"), dict) else {}
        value = " ".join((value or "").split())[: self.CONFIG_MAX_VALUE_CHARS]
        if not value:
            cfg.pop(key, None)
        else:
            # Re-insert at the end so the oldest key is dropped first on
            # overflow (dict preserves insertion order).
            cfg.pop(key, None)
            cfg[key] = value
        # Enforce key count, then total-char budget, dropping oldest first.
        while len(cfg) > self.CONFIG_MAX_KEYS:
            cfg.pop(next(iter(cfg)))
        while cfg and len(json.dumps(cfg)) > self.CONFIG_MAX_CHARS:
            cfg.pop(next(iter(cfg)))
        meta["config"] = cfg
        self._write_metadata(project_id, meta)
        return cfg

    # ------------------------------------------------------------------ events

    def log_event(self, project_id: str, task_id: Optional[str], event_type: str,
                  payload: Optional[Dict[str, Any]] = None) -> int:
        project_id = _canon_id(project_id)
        task_id = _canon_id(task_id) or None
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO project_events(project_id, task_id, type, payload_json, ts) "
                "VALUES (?,?,?,?,?)",
                (project_id, task_id, event_type, json.dumps(payload or {}), _now()),
            )
            new_id = int(cur.lastrowid)
            # Prune superseded scratchpad snapshots. Only the most recent
            # snapshot is ever read back (_hydrate_scratchpad uses limit=1),
            # but each one carries the full free-chat key/value dict, so on a
            # long-lived project they were the dominant source of unbounded
            # project_events growth. Keep just the row we wrote.
            if event_type == "scratchpad_snapshot":
                conn.execute(
                    "DELETE FROM project_events WHERE project_id = ? "
                    "AND type = 'scratchpad_snapshot' AND id < ?",
                    (project_id, new_id),
                )
            conn.commit()
            return new_id

    def list_events(self, project_id: str, limit: int = 50,
                    event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        project_id = _canon_id(project_id)
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
