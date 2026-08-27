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
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

# Words that carry no identity when comparing project titles ("Chess Game
# project" vs "chess game") — used by find_deleted_similar.
_TITLE_STOPWORDS = {"a", "an", "the", "project", "new", "my", "our", "game"}


class ProjectKind(str, Enum):
    CODING = "CODING"
    GENERAL = "GENERAL"


class ProjectStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DONE = "DONE"
    ARCHIVED = "ARCHIVED"
    # Human-attested terminal state (2026-07-25): only the release action can
    # set it (after a rehearsal that cold-starts the project from its own
    # release directions), never the rollup and never a generic update. A
    # RELEASED project is immutable — changes fork a new version via
    # create_version. The release dossier lives in metadata.release and is
    # rendered as RELEASE.md in the workspace.
    RELEASED = "RELEASED"
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
    constraints_json TEXT NOT NULL DEFAULT '[]',
    depends_on_json TEXT NOT NULL DEFAULT '[]',
    result_summary TEXT NOT NULL DEFAULT '',
    failure_reason TEXT NOT NULL DEFAULT '',
    revision_count INTEGER NOT NULL DEFAULT 0,
    actual_tool_used TEXT,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    actual_cost REAL NOT NULL DEFAULT 0.0,
    depth INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    closed_req_id TEXT NOT NULL DEFAULT '',
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

CREATE TABLE IF NOT EXISTS deleted_projects (
    id TEXT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'GENERAL',
    goal TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    deleted_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deleted_projects_ts ON deleted_projects(deleted_at);
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


def _constraint_list(value: Any) -> List[str]:
    """Normalise a metadata constraints value to a list of strings.

    Metadata is model-written JSON: a bare string
    (``{"constraints": "no pandas"}``) passes the dict boundary check and
    used to be iterated CHAR-BY-CHAR — the retirement path then persisted
    the shredded single-character list, destroying the record (review
    catch, 2026-08-01). Strings wrap to a one-element list; other
    non-list scalars stringify likewise; lists/tuples stringify per item."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(c) for c in value if str(c).strip()]
    return [str(value)]


class ProjectStore:
    """SQLite-backed store for projects, tasks, artifacts, and events.

    Single-writer discipline is enforced with an RLock. The store does
    not cache rows — callers that need repeated access should hold the
    returned dicts. Schema migrations are handled by additive
    ``ALTER TABLE`` calls in ``_init_db``.
    """

    #: Per-(project, type) retention for high-churn bookkeeping events
    #: (task_updated / project_updated / work_log). Readers use windows of
    #: ≤ 20; 300 keeps generous forensic depth while bounding the table.
    _EVENTS_RETAIN_PER_TYPE = 300

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
        # Optional hook fired when a TASK transitions DONE -> open (§4E
        # Tier 3, 2026-08-01). main.py wires this to the calibration
        # retro-negative writer. Left None in tests/headless contexts.
        # Signature: (project_id, task_id, from_status, closed_req_id).
        self.on_task_reopened = None
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

    def _fire_task_reopened(self, project_id: str, task_id: str,
                            from_status: str, closed_req_id: str) -> None:
        """Invoke the ``on_task_reopened`` hook (§4E Tier 3). Same discipline
        as ``_fire_project_done``: outside the DB lock, fully guarded — a
        calibration write failure must never break the status transition."""
        cb = getattr(self, "on_task_reopened", None)
        if cb is None:
            return
        try:
            cb(project_id, task_id, from_status, closed_req_id)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "on_task_reopened hook failed for task %s", task_id,
                exc_info=True,
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
                ("constraints_json", "TEXT NOT NULL DEFAULT '[]'"),
                # §4E Tier 3 (2026-08-01): the request id of the turn that
                # last closed this task DONE — the join key for retroactive
                # calibration negatives when the task is later reopened.
                ("closed_req_id", "TEXT NOT NULL DEFAULT ''"),
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

    def update_project(self, project_id: str, metadata_replace: bool = False,
                       **fields) -> bool:
        """Update project fields. ``metadata`` is MERGED (shallow) into the
        existing metadata by default — the blob carries system state
        (design_ledger, config, steps_used/cap, research index, runtime
        counters) that a whole-dict replace silently destroyed, e.g. the
        documented budget-raise ``metadata={"steps_cap": 100}``. Pass
        ``metadata_replace=True`` for a deliberate full replacement."""
        project_id = _canon_id(project_id)
        if not fields:
            return False
        allowed = {"title", "kind", "goal", "status", "workspace_dir", "metadata"}
        for key in fields:
            if key not in allowed:
                raise ValueError(f"unknown project field: {key}")
        # Normalise enums up front so a bad value raises before the txn opens.
        status_norm = (ProjectStatus(fields["status"].upper()).value
                       if "status" in fields else None)
        kind_norm = (ProjectKind(fields["kind"].upper()).value
                     if "kind" in fields else None)
        # Single BEGIN IMMEDIATE txn: the prior-status read, the metadata
        # read+merge, and the UPDATE are atomic across processes (mirrors
        # _atomic_metadata_update), so a concurrent writer can't interleave
        # between our read and write — no metadata clobber, and the DONE
        # hook below can't double-fire from two racing status updates.
        prev_status = None
        with self._lock, self._connect() as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status, metadata_json FROM projects WHERE id = ?",
                    (project_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return False
                prev_status = row["status"]
                sets = []
                values: List[Any] = []
                for key, val in fields.items():
                    if key == "metadata":
                        if metadata_replace:
                            merged: Dict[str, Any] = dict(val or {})
                        else:
                            try:
                                merged = json.loads(row["metadata_json"] or "{}")
                            except Exception:
                                merged = {}
                            if not isinstance(merged, dict):
                                merged = {}
                            merged.update(val or {})
                        sets.append("metadata_json = ?")
                        values.append(json.dumps(merged))
                    elif key == "status":
                        sets.append("status = ?")
                        values.append(status_norm)
                    elif key == "kind":
                        sets.append("kind = ?")
                        values.append(kind_norm)
                    else:
                        sets.append(f"{key} = ?")
                        values.append(val)
                sets.append("updated_at = ?")
                values.append(_now())
                values.append(project_id)
                values.append(prev_status)
                cur = conn.execute(
                    f"UPDATE projects SET {', '.join(sets)} "
                    "WHERE id = ? AND status = ?", values
                )
                conn.execute("COMMIT")
                updated = cur.rowcount > 0
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        if updated:
            self.log_event(project_id, None, "project_updated", {"fields": list(fields.keys())})
            if (status_norm == ProjectStatus.DONE.value
                    and (prev_status or "").upper() != ProjectStatus.DONE.value):
                # Constraint lifecycle (2026-08-01, req 56221fad post-mortem):
                # stored constraints bind work while the project is IN FLIGHT.
                # A 07-28 deliverable constraint ("Start with: What it means
                # to BE ghost") kept replaying into every request for 4 days
                # after the work closed — polluting new artifacts and driving
                # verifier refutes whose follow-up tasks reopened the project,
                # a self-feeding loop. DONE retires them; a verifier-refute
                # reopen does NOT resurrect them; the user restating one
                # re-arms it (see the create-merge path in tools.projects).
                self.retire_constraints(project_id, reason="project completed")
                self._fire_project_done(project_id)
        return updated

    def retire_constraints(self, project_id: str, reason: str = "",
                           only: Optional[List[str]] = None) -> List[str]:
        """Move the project's active ``constraints`` to
        ``constraints_retired`` (deduped, bounded). ``only`` limits the
        move to the given texts (case-insensitive exact match); default is
        all of them. Returns the list that was retired (empty when nothing
        matched OR the write did not persist). Never raises — lifecycle
        bookkeeping must not break a status transition."""
        moved: List[str] = []
        only_keys = ({str(c).lower() for c in only}
                     if only is not None else None)
        try:
            def _mut(meta):
                active = _constraint_list(meta.get("constraints"))
                if not active:
                    return meta
                if only_keys is None:
                    moving, keeping = active, []
                else:
                    moving = [c for c in active if c.lower() in only_keys]
                    keeping = [c for c in active if c.lower() not in only_keys]
                if not moving:
                    return meta
                prior = _constraint_list(meta.get("constraints_retired"))
                seen = {c.lower() for c in prior}
                for c in moving:
                    if c.lower() not in seen:
                        prior.append(c)
                        seen.add(c.lower())
                moved.extend(moving)
                meta["constraints"] = keeping
                # Bounded: this is an audit trail, not a working set.
                meta["constraints_retired"] = prior[-20:]
                return meta

            # Success is the WRITE landing, not the mutator running: a
            # mid-transaction failure rolls the store back after `moved`
            # was populated, and reporting those as retired would tell the
            # caller "stops replaying immediately" about a constraint that
            # is still live (review catch, 2026-08-01).
            if self._atomic_metadata_update(project_id, _mut) is None:
                return []
            if moved:
                self.log_event(project_id, None, "constraints_retired",
                               {"constraints": moved[:10],
                                "reason": reason or "unspecified"})
        except Exception:
            logger.debug("retire_constraints skipped for %s",
                         project_id, exc_info=True)
            return []
        return moved

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
            # Remember what we archived FROM (2026-07-25): resume used to
            # flip ARCHIVED → ACTIVE unconditionally, silently STRIPPING a
            # RELEASED project's attestation (and with it every immutability
            # guard, while RELEASE.md still claimed otherwise). The resume
            # path restores this value.
            prev = str((self.get_project(project_id) or {})
                       .get("status") or "").upper()
            if prev and prev != ProjectStatus.ARCHIVED.value:
                try:
                    def _mut(meta):
                        meta["archived_from"] = prev
                        return meta
                    self._atomic_metadata_update(project_id, _mut)
                except Exception:
                    logger.debug("archived_from stash skipped", exc_info=True)
            return self.update_project(project_id, status=ProjectStatus.ARCHIVED.value)

        # Resolve the workspace path BEFORE deleting the row.
        proj = self.get_project(project_id)
        ws_str = (proj or {}).get("workspace_dir")
        if not ws_str and self.sandbox_root:
            ws_str = str(self.sandbox_root / "projects" / project_id)

        with self._lock, self._connect() as conn:
            # Tombstone BEFORE the row goes: the FK cascade wipes tasks,
            # artifacts and events, so this separate table is the only
            # durable record that the project ever existed. create_project
            # consults it to detect the delete-then-recreate correction
            # pattern (user rejects a build, deletes it, re-asks with
            # added constraints).
            if proj:
                conn.execute(
                    "INSERT INTO deleted_projects(id, title, kind, goal, "
                    "created_at, deleted_at) VALUES (?,?,?,?,?,?)",
                    (project_id, proj.get("title") or "",
                     proj.get("kind") or "GENERAL", proj.get("goal") or "",
                     float(proj.get("created_at") or 0.0), _now()),
                )
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
                    # Restore writability first — a RELEASED workspace is
                    # chmod'd read-only and rmtree(ignore_errors) would
                    # silently leave the tree behind. Path-direct: the DB
                    # row is already gone at this point, so the
                    # project-id-based helper would no-op.
                    try:
                        self._chmod_tree(ws_p, False)
                    except Exception:
                        pass
                    shutil.rmtree(ws_p, ignore_errors=True)
                    # ⚠ the whole tree just went — stamp the removal mark
                    # (deleted=None = "everything") so a verdict in flight
                    # for this project reports could-not-check instead of
                    # refuting files the delete removed. This was the one
                    # removal route with no mark: `manage_projects` is
                    # excluded from the verdict's removal-capable names ON
                    # THE CLAIM that its removal paths stamp the store.
                    try:
                        from ..core.workspace_cleanup import _mark_removal
                        _mark_removal(self, project_id, None)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Could not remove workspace for %s: %s", project_id, e)
        return deleted

    def find_deleted_similar(self, title: str,
                             within_secs: float = 86400.0) -> Optional[Dict[str, Any]]:
        """Most recent tombstone whose title resembles ``title``.

        Similarity is token overlap (Jaccard >= 0.5) on lowercased word
        sets — enough to match "Chess Game" against "chess game project"
        without a model call. Returns the tombstone dict (id, title, kind,
        goal, created_at, deleted_at) or None.
        """
        want = {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
                if w not in _TITLE_STOPWORDS}
        if not want:
            return None
        cutoff = _now() - max(0.0, within_secs)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM deleted_projects WHERE deleted_at >= ? "
                "ORDER BY deleted_at DESC LIMIT 50", (cutoff,),
            ).fetchall()
        for row in rows:
            have = {w for w in re.findall(r"[a-z0-9]+", (row["title"] or "").lower())
                    if w not in _TITLE_STOPWORDS}
            if not have:
                continue
            overlap = len(want & have) / max(1, len(want | have))
            # Correction-linking demands STRONG title evidence: ≥2 shared
            # significant tokens, or identical token sets (covers short
            # titles). A lone shared token at 0.5 Jaccard mislinked
            # distinct projects — delete "Chess Game" ({chess} after
            # stopwords), create "Chess Tutorial" → stamped correction_of
            # and re-planned as if the user rejected the previous build.
            if overlap >= 0.5 and (len(want & have) >= 2 or want == have):
                return dict(row)
        return None

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
                 constraints: Optional[List[str]] = None,
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
                "constraints_json, depends_on_json, "
                "result_summary, failure_reason, revision_count, actual_tool_used, "
                "estimated_cost, actual_cost, depth, position, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, project_id, parent_id, description.strip(),
                 status.upper(), dependency_type.upper(),
                 json.dumps(alternatives or []), json.dumps(postconditions or []),
                 json.dumps(constraints or []), json.dumps(depends_on_canon),
                 "", "", 0, None, estimated_cost, 0.0, depth, position, now, now),
            )
            # REOPEN a finished project when new work is added (2026-07-11).
            #
            # Previously this only bumped `updated_at`, so a project that had
            # rolled to DONE stayed DONE — while `advance_once` hard-refuses a
            # non-ACTIVE project ("project is DONE, not ACTIVE"). Adding tasks
            # to a completed project therefore created work that autoadvance
            # could NEVER reach: the tool reported "all tasks are complete"
            # while N tasks sat PENDING, and the model burned turns trying to
            # reconcile the contradiction (observed live 2026-07-11, project
            # 6051abfb21b8: 20 tasks added to a DONE project, 8 pending,
            # autoadvance returned 0). Adding work to a finished project
            # un-finishes it — that is the only coherent semantic.
            #
            # DONE reopens; so do FAILED and PAUSED (2026-07-20) — a project
            # in either state holding revived work had NO path back to
            # ACTIVE (the tool status enum omits ACTIVE, advance_once
            # refuses non-ACTIVE), so new tasks sat unreachable forever.
            # NEEDS_USER stays put (parked on human input by design) and
            # ARCHIVED is a deliberate end-state (the cleanup sweep has
            # already run); silently resurrecting either would be a
            # surprise — the caller can un-archive / answer explicitly.
            # The read and guarded UPDATE share the INSERT's write txn, so
            # a concurrent status change can't interleave.
            reopened_from = None
            if str(status).upper() != "DONE":
                prow = conn.execute(
                    "SELECT status FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
                prev = ((prow["status"] if prow else "") or "").upper()
                # NEEDS_USER/BLOCKED joined 2026-07-25: adding work to a
                # waiting/blocked project previously created tasks the
                # advancer could NEVER reach (advance_once refuses
                # non-ACTIVE) — the same unreachable-work trap fixed for
                # DONE on 2026-07-11. ARCHIVED and RELEASED stay excluded:
                # deliberate end-states, resurrect explicitly.
                if prev in ("DONE", "FAILED", "PAUSED", "NEEDS_USER",
                            "BLOCKED"):
                    cur = conn.execute(
                        "UPDATE projects SET updated_at = ?, status = 'ACTIVE' "
                        "WHERE id = ? AND status = ?",
                        (now, project_id, prev),
                    )
                    if cur.rowcount > 0:
                        reopened_from = prev
            if not reopened_from:
                conn.execute(
                    "UPDATE projects SET updated_at = ? WHERE id = ?",
                    (now, project_id),
                )
            conn.commit()
        self.log_event(project_id, task_id, "task_added",
                       {"description": description, "parent_id": parent_id})
        if reopened_from:
            self.log_event(project_id, task_id, "project_reopened",
                           {"reason": f"new task added to a {reopened_from} project",
                            "from_status": reopened_from})
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
                   "postconditions", "constraints", "depends_on", "result_summary",
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
            elif key in ("alternatives", "postconditions", "constraints"):
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
        new_status = (str(fields["status"]).upper()
                      if "status" in fields else None)
        task_reopened_args = None
        with self._lock, self._connect() as conn:
            # Read the task's PRE-update state: the DONE->open transition
            # (§4E Tier 3) and the closing-turn stamp both key off it.
            # SELECT * + dict.get, NOT a named-column SELECT: _migrate is
            # best-effort (a locked DB skips the ALTER with a warning), and
            # naming closed_req_id here would turn that survivable warning
            # into "no such column" on EVERY update_task until restart.
            pre_row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            pre = dict(pre_row) if pre_row is not None else {}
            has_closed_req_col = "closed_req_id" in pre
            prev_task_status = (pre.get("status") or "").upper()
            prev_closed_req = str(pre.get("closed_req_id") or "")
            if (has_closed_req_col and new_status == "DONE"
                    and prev_task_status != "DONE"):
                # Stamp WHICH turn closed this task so a later reopen can
                # retro-label that turn's calibration sample. ALWAYS write
                # on the transition into DONE: "SYSTEM" (no request context
                # — boot reaper, maintenance) writes blank, which also
                # clears a stale stamp from an earlier closing turn.
                try:
                    from ..utils.logging import request_id_context
                    req = request_id_context.get()
                except Exception:  # pragma: no cover - defensive
                    req = ""
                if not req or req == "SYSTEM":
                    req = ""
                sets.append("closed_req_id = ?")
                values.append(str(req))
            elif (has_closed_req_col
                    and new_status in ("PENDING", "READY", "IN_PROGRESS")
                    and prev_task_status == "DONE"):
                # Reopening consumes the stamp: the retro-negative for this
                # closing turn fires below exactly once, and a later
                # re-close re-stamps (or blanks) it fresh.
                sets.append("closed_req_id = ?")
                values.append("")
            sets.append("updated_at = ?")
            values.append(now)
            values.append(task_id)
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
            # Re-opening a task to an open state on a FAILED/PAUSED project
            # reactivates it (2026-07-20) — otherwise the revived work was
            # unreachable (advance_once refuses non-ACTIVE, the tool status
            # enum omits ACTIVE). NEEDS_USER and ARCHIVED are never
            # auto-undone. Same txn as the task UPDATE, guarded on the
            # observed status, so a concurrent transition makes it a no-op.
            reopened_from = None
            if (row and cur.rowcount > 0 and "status" in fields
                    and str(fields["status"]).upper()
                    in ("PENDING", "READY", "IN_PROGRESS")):
                prow = conn.execute(
                    "SELECT status FROM projects WHERE id = ?",
                    (row["project_id"],),
                ).fetchone()
                prev = ((prow["status"] if prow else "") or "").upper()
                # DONE joined 2026-07-26 to match add_task's tuple: reviving
                # an EXISTING task on a DONE project ("redo task X") left
                # the project locked DONE — the rollup refuses to leave
                # DONE, advance_once refuses non-ACTIVE, and the tool's
                # status enum omits ACTIVE, so the revived task was
                # permanently unreachable. ARCHIVED/RELEASED stay excluded
                # (deliberate end-states), same as add_task.
                if prev in ("DONE", "FAILED", "PAUSED", "NEEDS_USER", "BLOCKED"):
                    rcur = conn.execute(
                        "UPDATE projects SET status = 'ACTIVE', updated_at = ? "
                        "WHERE id = ? AND status = ?",
                        (now, row["project_id"], prev),
                    )
                    if rcur.rowcount > 0:
                        reopened_from = prev
            # §4E Tier 3: the TASK-level DONE -> open transition (regardless
            # of project status — a task revived on an ACTIVE project is
            # just as much a delayed negative on the turn that closed it).
            if (row and cur.rowcount > 0
                    and new_status in ("PENDING", "READY", "IN_PROGRESS")
                    and prev_task_status == "DONE"):
                task_reopened_args = (
                    row["project_id"], task_id, prev_task_status,
                    ((pre["closed_req_id"] if pre else "") or ""),
                )
            conn.commit()
            updated = cur.rowcount > 0
        if updated and row:
            self.log_event(row["project_id"], task_id, "task_updated",
                           {"fields": list(fields.keys())})
            if reopened_from:
                self.log_event(row["project_id"], task_id, "project_reopened",
                               {"reason": f"task re-opened on a {reopened_from} project",
                                "from_status": reopened_from})
            if task_reopened_args:
                self.log_event(row["project_id"], task_id, "task_reopened",
                               {"from_status": task_reopened_args[2],
                                "closed_req_id": task_reopened_args[3]})
                # Outside the DB txn (committed above), still guarded.
                self._fire_task_reopened(*task_reopened_args)
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

        DONE, ARCHIVED, RELEASED and PAUSED are *locked*: once reached we
        never auto-undo them. RELEASED/PAUSED joined the lock 2026-07-25 —
        a store-level task write on a RELEASED project could roll it to
        DONE and FIRE THE CLEANUP SWEEP on a human-attested workspace, and
        one task flip on a deliberately-PAUSED project could complete it
        under the operator. FAILED and NEEDS_USER are NOT locked — a
        revised/answered task rolls the project forward (or, since
        2026-07-25, BACK to ACTIVE when real open work reappears: the
        answered-question NEEDS_USER trap left projects stranded forever
        because nothing ever rolled back).
        """
        proj = self.get_project(project_id)
        if not proj:
            return
        current = (proj.get("status") or "").upper()
        if current in {ProjectStatus.DONE.value, ProjectStatus.ARCHIVED.value,
                       ProjectStatus.RELEASED.value, ProjectStatus.PAUSED.value}:
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
            # still autonomous work to do.
            open_states = {s for s in statuses if s not in terminal}
            if open_states and open_states == {"NEEDS_USER"}:
                new_status = ProjectStatus.NEEDS_USER.value
            elif current in (ProjectStatus.NEEDS_USER.value,
                             ProjectStatus.BLOCKED.value):
                # Real open work reappeared on a waiting/blocked project
                # (the user answered; a task was revised) — roll BACK to
                # ACTIVE so the advancer and the tool enum can reach it.
                # Without this branch the project stayed NEEDS_USER forever
                # (the trap: advance_once refuses non-ACTIVE).
                new_status = ProjectStatus.ACTIVE.value
            else:
                return

        if new_status == current:
            return
        # Guard on the status observed at read time (mirrors add_task's
        # reopen): the lock was released between get_project/list_tasks and
        # this write, so a concurrent transition (e.g. a manual ARCHIVE)
        # would otherwise be stomped back to DONE — and then fire the
        # destructive cleanup on an archived project. Interleave → no-op.
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? "
                "WHERE id = ? AND status = ?",
                (new_status, _now(), project_id, current),
            )
            conn.commit()
            if cur.rowcount == 0:
                return
        self.log_event(
            project_id, None, "project_auto_rollup",
            {"new_status": new_status,
             "had_failures": any(s in failure for s in statuses)},
        )
        # A genuine completion is the cleanup trigger. Fire only for DONE —
        # FAILED / NEEDS_USER stay resumable (a revised task can roll the
        # project forward to DONE later), so their workspace must survive.
        if new_status == ProjectStatus.DONE.value:
            # Constraint lifecycle rides EVERY DONE transition — this raw-SQL
            # rollup (last task closed via task_update/delete_task) is the
            # path projects normally finish on; retiring only in
            # update_project would miss the incident's exact shape (review
            # catch, 2026-08-01).
            self.retire_constraints(project_id, reason="project completed")
            self._fire_project_done(project_id)

    def reset_orphaned_in_progress(self, older_than_seconds: float = 900.0) -> int:
        """Reset stale IN_PROGRESS tasks back to READY. Returns the count.

        The advancer commits a leaf's IN_PROGRESS claim before several
        awaits; a crash or deploy-kill (plain SIGTERM) between claim and
        completion wedges the task forever — ``next_ready_leaf`` only
        considers PENDING/READY, so the project sits ACTIVE reporting
        "all complete". Called at boot (before the advancer starts) and
        safe to call any time: only tasks whose ``updated_at`` is older
        than ``older_than_seconds`` are touched, and each reset is guarded
        on the row still being IN_PROGRESS, so genuinely in-flight work in
        a live process is never yanked. Logs one ``task_reset_orphaned``
        project event per reset.
        """
        cutoff = _now() - max(0.0, float(older_than_seconds))
        reset: List[sqlite3.Row] = []
        with self._lock, self._connect() as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT id, project_id FROM tasks "
                    "WHERE status = 'IN_PROGRESS' AND updated_at < ?",
                    (cutoff,),
                ).fetchall()
                if rows:
                    now = _now()
                    conn.executemany(
                        "UPDATE tasks SET status = 'READY', updated_at = ? "
                        "WHERE id = ? AND status = 'IN_PROGRESS'",
                        [(now, r["id"]) for r in rows],
                    )
                    reset = list(rows)
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        for r in reset:
            self.log_event(r["project_id"], r["id"], "task_reset_orphaned",
                           {"from": "IN_PROGRESS", "to": "READY",
                            "older_than_seconds": float(older_than_seconds)})
        return len(reset)

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
            # `visited` guards against a parent_id CYCLE (e.g. A→B→A, or a
            # self-parent) — without it the BFS re-appends the same ids
            # forever and `to_delete` grows without bound (hang/OOM).
            visited = {task_id}
            to_delete = [task_id]
            frontier = [task_id]
            while frontier:
                nxt: List[str] = []
                for tid in frontier:
                    child_rows = conn.execute(
                        "SELECT id FROM tasks WHERE parent_id = ?", (tid,)
                    ).fetchall()
                    for cr in child_rows:
                        cid = cr["id"]
                        if cid in visited:
                            continue
                        visited.add(cid)
                        to_delete.append(cid)
                        nxt.append(cid)
                frontier = nxt
            conn.executemany(
                "DELETE FROM tasks WHERE id = ?",
                [(tid,) for tid in to_delete],
            )
            conn.commit()
        self.log_event(project_id, task_id, "task_deleted",
                       {"cascaded": len(to_delete) - 1})
        # Deleting the last open task can settle the project (all remaining
        # tasks DONE) — without this, an otherwise-finished project stayed
        # ACTIVE forever because nothing ever re-evaluated the aggregate.
        self._maybe_rollup_project_status(project_id)
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
            d["constraints"] = json.loads(d.pop("constraints_json") or "[]")
        except Exception:
            d["constraints"] = []
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
        # Include a path key when the payload carries one: file_history's
        # reader scans ("path","rel","rel_path","file","payload") keys, and
        # a pathless artifact_added payload made registration events
        # invisible to per-file history (live: all 60 events pathless).
        # For kind='file' the payload IS the normalized rel path
        # (register_file_artifact contract); other kinds may carry a JSON
        # object with a path-ish key.
        _evt = {"kind": kind, "artifact_id": art_id}
        try:
            if kind == "file" and isinstance(payload, str) and payload:
                _evt["path"] = payload
            else:
                _pl = json.loads(payload) if isinstance(payload, str) else payload
                if isinstance(_pl, dict):
                    for _pk in ("path", "rel_path", "rel", "file"):
                        if _pl.get(_pk):
                            _evt["path"] = str(_pl[_pk])
                            break
        except Exception:
            pass
        self.log_event(task["project_id"], task_id, "artifact_added", _evt)
        return art_id

    def register_file_artifact(self, task_id: str, rel_path: str,
                               description: str = "") -> Optional[str]:
        """Register a deliverable file (``kind='file'``) for ``task_id``,
        deduplicated within the project.

        ``description`` (optional, 2026-07-24): a one-line "what this file
        is/does". When given, it is upserted into the project's file
        manifest (see ``describe_file``) so the briefing can show
        ``path — purpose`` instead of a bare path. Best-effort — a manifest
        failure never blocks artifact registration.

        This is the durable "keep me" marker the workspace cleanup sweep
        reads: any file path registered here survives a project's
        end-of-life sweep; everything else under the project workspace is
        deleted. Idempotent — re-registering the same project-relative path
        returns the existing artifact id instead of creating a duplicate, so
        callers can register on every DONE without accumulating rows.

        Paths are stored in the same normalized project-relative POSIX form
        the cleanup's ``_normalize_rel`` reduces walked files to — a leading
        ``/workspace/`` / ``workspace/`` segment, then ``projects/<id>/``,
        then ``/`` and ``./`` are stripped; ``..`` traversal is rejected.
        Before this (2026-07-20) an absolute ``/workspace/projects/<id>/x``
        payload was stored verbatim, matched no walked rel, and the DONE
        sweep deleted the registered deliverable.

        Returns the artifact id, or ``None`` when the path is blank,
        rejected, or the task is unknown.
        """
        task_id = _canon_id(task_id)
        task = self.get_task(task_id)
        if not task:
            return None
        project_id = task["project_id"]
        rel = self._normalize_rel_path(project_id, rel_path)
        if rel is None:
            return None
        if description:
            try:
                self.describe_file(project_id, rel, description)
            except Exception as e:
                logger.debug("manifest upsert on register skipped: %s", e)
        for art in self.list_artifacts(project_id=project_id):
            if (art.get("kind") == "file"
                    and (art.get("payload") or "").strip() == rel):
                return art.get("id")
        return self.add_artifact(task_id, "file", rel)

    @staticmethod
    def _normalize_rel_path(project_id: str, rel_path: str) -> Optional[str]:
        """Reduce a path to the normalized project-relative POSIX form used
        by both the artifact keep-set and the file manifest (extracted
        verbatim from ``register_file_artifact`` — see its docstring for why
        this exact shape matters to the cleanup sweep). Returns ``None`` for
        blank / traversal-rejected paths."""
        rel = (rel_path or "").strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if rel.lower().startswith("workspace/"):
            rel = rel[len("workspace/"):].lstrip("/")
        pref = f"projects/{project_id}/"
        if rel.lower().startswith(pref.lower()):
            rel = rel[len(pref):]
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            return None
        return "/".join(parts)

    def list_deliverables(self, project_id: str) -> List[str]:
        """Deduped, insertion-ordered file paths registered as deliverable
        (`kind='file'`) artifacts across the project's tasks — the
        project's own manifest of what it built. Until 2026-07-18 this
        data was write-only: registered on every DONE task (it drives the
        end-of-project cleanup keep-set) but never readable as a list, so
        the model re-derived "what exists" from sandbox listings every
        time."""
        seen: set = set()
        out: List[str] = []
        for a in self.list_artifacts(project_id=project_id):
            if str(a.get("kind") or "") != "file":
                continue
            p = str(a.get("payload") or "").strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def missing_deliverables(self, project_id: str) -> set:
        """Registered deliverables whose file is NOT on disk.

        ⚠ WHY (queue #10, 2026-08-21). `register_file_artifact` records a
        path; it never checks that anything is there, and nothing reconciles
        afterwards. The list is a record of CLAIMS — but `core/prompts.py`
        renders it into every project briefing as "DELIVERABLES (N file(s)
        the project built)", so a claim is presented to the model as fact.
        Measured on the live store: **3 of 66 registered files do not exist**
        (`cascade_analysis.md`, `cascade_evidence.py`, `roms/sonic.md`),
        across 2 of 5 projects, and none of them was removed by the cleanup
        sweep — every `workspace_tidy` event on this box deleted only debris
        (`.browser_runner.py`, `__pycache__`, screenshots). So the agent is
        told it produced files it never produced, and will cite them.

        The record is NOT rewritten — the claim is audit data, and a file can
        legitimately be deleted after the fact. The READ path is what learns
        to be honest, the same shape as §4CC's mood staleness and §4CD's
        diary-follows-corpus: keep the record, age the presentation.

        Returns an empty set when there is no sandbox root configured (tests,
        `--no-sandbox`): unknown is not the same as missing, and marking every
        deliverable gone because the checker cannot see the disk would be a
        worse lie than the one being fixed.
        """
        if not self.sandbox_root:
            return set()
        try:
            root = Path(self._default_workspace(project_id) or "")
        except Exception:  # noqa: BLE001
            return set()
        if not root or not root.exists():
            # The whole workspace is gone (deleted project, moved sandbox).
            # That is not evidence about individual files, and flagging all
            # of them would bury a real single-file loss in noise.
            return set()
        gone = set()
        for rel in self.list_deliverables(project_id):
            # ⚠ NORMALISE BEFORE STATTING, through the SAME function
            # registration uses. Some stored payloads carry the redundant
            # `projects/<id>/` prefix (rows written before the 2026-07-20
            # H9 fix); the cleanup sweep already re-normalises defensively at
            # read time, which is why those files were never swept. Comparing
            # the RAW payload against disk reported three of them missing on
            # the live store — including WebOS's `index.html` and
            # `server.js`, the project's actual deliverables, which are right
            # there. Marking a present file MISSING is a worse lie than the
            # unverified claim this method exists to catch, so the check goes
            # through the path contract rather than around it.
            probe = self._normalize_rel_path(project_id, rel) or rel
            try:
                if not (root / probe).exists():
                    gone.add(rel)
            except (OSError, ValueError):  # noqa: PERF203 — per-path guard
                continue
        return gone

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

    def _atomic_metadata_update(self, project_id: str,
                                mutate) -> Optional[Dict[str, Any]]:
        """Read-modify-write a project's metadata atomically ACROSS PROCESSES.

        ``mutate`` receives the current metadata dict, mutates it in place
        (or returns a replacement dict), and the result is persisted. The
        whole read-modify-write runs inside a single ``BEGIN IMMEDIATE``
        transaction so a concurrent metadata writer in ANOTHER process can't
        interleave and clobber the update. The previous
        ``get_project`` (read, lock released) → ``_write_metadata`` (write,
        lock re-taken) shape let two processes both read the same snapshot
        and lose one side's edit — the in-process ``RLock`` only serialises
        writers *inside* one process, and Slack, web and CLI each run their
        own.

        ``BEGIN IMMEDIATE`` takes SQLite's write lock up front, before the
        read, so a competing writer waits on ``busy_timeout`` instead of
        racing; ``isolation_level=None`` hands us manual transaction control
        (Python's sqlite3 otherwise opens a DEFERRED transaction that would
        not lock until the first write — too late to protect the read).

        Returns the persisted metadata, or ``None`` when the project is
        missing. Logs no ``project_updated`` event, matching the ledger /
        config callers that use it.
        """
        project_id = _canon_id(project_id)
        with self._lock, self._connect() as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT metadata_json FROM projects WHERE id = ?",
                    (project_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                try:
                    meta = json.loads(row["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                new_meta = mutate(meta)
                if new_meta is None:
                    new_meta = meta
                conn.execute(
                    "UPDATE projects SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(new_meta or {}), _now(), project_id),
                )
                conn.execute("COMMIT")
                return new_meta
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def get_ledger(self, project_id: str) -> str:
        proj = self.get_project(project_id)
        return ((proj or {}).get("metadata") or {}).get("design_ledger") or ""

    def append_ledger(self, project_id: str, line: str) -> str:
        """Append one line to the project's design ledger (bounded, dedup'd
        against an identical trailing line). Returns the new ledger text.

        The read-modify-write runs inside a single ``BEGIN IMMEDIATE``
        transaction (via ``_atomic_metadata_update``) so a concurrent ledger
        / config write in another process can't interleave and drop the
        appended line."""
        line = " ".join((line or "").split())  # collapse whitespace/newlines
        if not line:
            # Nothing to append — return the current ledger without writing
            # (matches the pre-atomic behaviour: no updated_at bump).
            proj = self.get_project(project_id)
            if not proj:
                return ""
            return ((proj.get("metadata") or {}).get("design_ledger") or "")

        result = {"text": ""}

        def _mutate(meta):
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
            result["text"] = text
            return meta

        updated = self._atomic_metadata_update(project_id, _mutate)
        if updated is None:  # project not found
            return ""
        return result["text"]

    def set_ledger(self, project_id: str, text: str) -> str:
        """Replace the project's design ledger wholesale (bounded).

        Atomic across processes (see ``_atomic_metadata_update``) so a
        concurrent config write to the same project's metadata isn't lost."""
        new_text = (text or "")[: self.LEDGER_MAX_CHARS]

        def _mutate(meta):
            meta["design_ledger"] = new_text
            return meta

        updated = self._atomic_metadata_update(project_id, _mutate)
        if updated is None:
            return ""
        return new_text

    # ------------------------------------------------------------------ file manifest
    #
    # The manifest is the project's per-file "what is this and what does it
    # do" map — the piece the design ledger (free prose) and deliverables
    # (bare paths) both lacked. It exists so a resumed session, or a small
    # model on a large project, is DIRECTED to the right files instead of
    # re-deriving the layout by re-reading everything (observed live
    # 2026-07-24: ~80s of thinking re-deriving a 2-file app's architecture
    # that a previous session had already worked out).
    # Shape: metadata.file_manifest = {rel_path: {"desc": str, "role": str,
    # "ts": float}} — bounded, most-recently-updated entries win eviction.
    MANIFEST_MAX_FILES = 60
    MANIFEST_DESC_MAX = 200
    MANIFEST_ROLE_MAX = 40

    def get_file_manifest(self, project_id: str) -> Dict[str, Dict[str, Any]]:
        """The project's file manifest, ``{rel_path: {desc, role, ts}}``.
        Empty dict when absent/legacy."""
        proj = self.get_project(project_id)
        mf = ((proj or {}).get("metadata") or {}).get("file_manifest")
        return dict(mf) if isinstance(mf, dict) else {}

    def describe_file(self, project_id: str, rel_path: str,
                      description: str, role: str = "") -> bool:
        """Upsert one file's manifest entry (atomic across processes,
        bounded). ``rel_path`` is normalized to the same project-relative
        form the artifact keep-set uses, so manifest keys and deliverable
        payloads always agree. Re-describing a path replaces its entry.
        Returns False for a blank/rejected path or unknown project.

        Also re-renders ``PROJECT_MAP.md`` in the project workspace
        (best-effort) so the manifest is greppable in-sandbox, not only
        briefing-injected."""
        rel = self._normalize_rel_path(_canon_id(project_id), rel_path)
        desc = " ".join((description or "").split())[: self.MANIFEST_DESC_MAX]
        if rel is None or not desc:
            return False

        def _mutate(meta):
            mf = meta.get("file_manifest")
            mf = dict(mf) if isinstance(mf, dict) else {}
            mf[rel] = {
                "desc": desc,
                "role": " ".join((role or "").split())[: self.MANIFEST_ROLE_MAX],
                "ts": _now(),
            }
            if len(mf) > self.MANIFEST_MAX_FILES:
                # Evict oldest-updated entries beyond the cap.
                victims = sorted(
                    mf.items(), key=lambda kv: kv[1].get("ts") or 0.0,
                )[: len(mf) - self.MANIFEST_MAX_FILES]
                for k, _ in victims:
                    mf.pop(k, None)
            meta["file_manifest"] = mf
            return meta

        updated = self._atomic_metadata_update(project_id, _mutate)
        if updated is None:
            return False
        try:
            self.render_project_map(project_id)
        except Exception as e:
            logger.debug("PROJECT_MAP render skipped: %s", e)
        return True

    def render_project_map(self, project_id: str) -> Optional[str]:
        """Write ``PROJECT_MAP.md`` into the project workspace: goal head +
        every deliverable with its manifest description (undescribed ones
        marked, so the idle backfill and the model can see what's missing).
        Atomic (tmp + os.replace), best-effort. Returns the path written,
        or ``None`` when the project/workspace is unavailable."""
        proj = self.get_project(project_id)
        if not proj:
            return None
        ws = (proj.get("workspace_dir") or "").strip()
        if not ws:
            return None
        mf = self.get_file_manifest(project_id)
        deliverables = self.list_deliverables(project_id)
        # Union: manifest may describe files not (yet) registered and
        # vice versa — show both, deliverables first in registration order.
        ordered = deliverables + [p for p in sorted(mf) if p not in deliverables]
        lines = [
            f"# PROJECT MAP — {proj.get('title') or project_id}",
            "",
            f"Goal: {(proj.get('goal') or '').strip()[:300]}",
            "",
            "## Files",
        ]
        for p in ordered:
            e = mf.get(p) or {}
            role = f" [{e['role']}]" if e.get("role") else ""
            desc = e.get("desc") or "(no description yet — use manage_projects action='describe_file')"
            lines.append(f"- `{p}`{role} — {desc}")
        if not ordered:
            lines.append("- (no deliverables registered yet)")
        text = "\n".join(lines) + "\n"
        try:
            ws_path = Path(ws)
            ws_path.mkdir(parents=True, exist_ok=True)
            target = ws_path / "PROJECT_MAP.md"
            tmp = target.with_suffix(".md.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
            return str(target)
        except Exception as e:
            logger.debug("PROJECT_MAP write failed: %s", e)
            return None

    @staticmethod
    def _chmod_tree(ws: Path, readonly: bool) -> int:
        """Walk ``ws`` flipping the write bits. Path-direct so callers that
        have already deleted the DB row (hard delete) can still restore
        writability before rmtree."""
        import stat
        touched = 0
        try:
            for p in [ws, *ws.rglob("*")]:
                try:
                    mode = p.stat().st_mode
                    if readonly:
                        new_mode = mode & ~(stat.S_IWUSR | stat.S_IWGRP
                                            | stat.S_IWOTH)
                    else:
                        new_mode = mode | stat.S_IWUSR
                    if new_mode != mode:
                        p.chmod(new_mode)
                        touched += 1
                except Exception:
                    continue
        except Exception:
            return touched
        return touched

    def set_workspace_readonly(self, project_id: str, readonly: bool) -> int:
        """chmod the whole workspace tree a-w (or restore u+w). OS-level
        half of release immutability (2026-07-25 round 2): the file-tool
        guard can't see `execute` shell writes, but mode bits can stop the
        common case (container root via virtiofs still maps through the
        host user on macOS). Best-effort — never raises; returns paths
        touched. Callers: release (True), unrelease (False), hard delete
        (False, before rmtree — which fails on read-only dirs)."""
        proj = self.get_project(project_id)
        ws = Path(str((proj or {}).get("workspace_dir") or ""))
        if not ws or not ws.is_dir():
            return 0
        return self._chmod_tree(ws, readonly)

    def unregister_file(self, project_id: str, rel_path: str) -> Dict[str, int]:
        """Remove a file from the deliverable record AND the manifest.

        The missing repair path (2026-07-25 review H4): a renamed/deleted
        deliverable's stale artifact row made the release rehearsal fail
        PERMANENTLY ("deliverable missing on disk") with no tool-side fix.
        Removes matching ``kind='file'`` artifact rows, the manifest entry,
        re-renders PROJECT_MAP.md. Returns counts. NOTE: also removes the
        path from the cleanup keep-set — the file (if still on disk) becomes
        sweepable debris, which is exactly right for a renamed-away file.
        """
        project_id = _canon_id(project_id)
        rel = self._normalize_rel_path(project_id, rel_path)
        out = {"artifacts_removed": 0, "manifest_removed": 0}
        if rel is None:
            return out
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM task_artifacts WHERE project_id = ? "
                "AND kind = 'file' AND payload = ?",
                (project_id, rel),
            )
            conn.commit()
            out["artifacts_removed"] = cur.rowcount

        def _mut(meta):
            mf = meta.get("file_manifest")
            if isinstance(mf, dict) and rel in mf:
                mf = dict(mf)
                mf.pop(rel, None)
                meta["file_manifest"] = mf
                out["manifest_removed"] = 1
            return meta

        try:
            self._atomic_metadata_update(project_id, _mut)
        except Exception:
            logger.debug("manifest unregister skipped", exc_info=True)
        if out["artifacts_removed"] or out["manifest_removed"]:
            self.log_event(project_id, None, "file_unregistered",
                           {"path": rel, **out})
            try:
                self.render_project_map(project_id)
            except Exception:
                pass
        return out

    # Stopword set for cross-project search (mirrors the briefing slice's
    # philosophy: strip generic verbs so "which project used X" matches X,
    # not the question's furniture).
    _SEARCH_STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "with", "is", "are", "was", "it", "this", "that", "which", "what",
        "project", "projects", "file", "files", "used", "uses", "using",
        "did", "does", "have", "has", "where", "who", "touched",
    })

    def search_projects(self, query: str, limit: int = 5,
                        per_project_events: int = 40) -> List[Dict[str, Any]]:
        """Cross-project keyword search (2026-07-25 review missing-feature
        #7): "which project touched X / used technique Y" was unanswerable —
        the graph covers concepts only, and every other record was
        per-project. Scans titles/goals, deliverable paths, manifest
        descriptions, design ledgers, research topics, and recent work_log
        entries. Deterministic token overlap, bounded, no LLM. Returns
        ranked ``[{project_id, title, status, score, matches: [{kind,
        snippet}]}]``."""
        tokens = {
            t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+",
                                  (query or "").lower())
            if len(t) > 2 and t not in self._SEARCH_STOPWORDS
        }
        if not tokens:
            return []

        def _hits(text: str) -> int:
            tl = (text or "").lower()
            return sum(1 for t in tokens if t in tl)

        results = []
        for proj in self.list_projects():
            pid = proj.get("id")
            meta = proj.get("metadata") or {}
            matches: List[Dict[str, str]] = []
            score = 0

            def _add(kind: str, text: str, weight: int = 1):
                nonlocal score
                h = _hits(text)
                if h:
                    score += h * weight
                    matches.append({"kind": kind,
                                    "snippet": " ".join(text.split())[:120]})

            _add("title/goal", f"{proj.get('title')} {proj.get('goal')}", 3)
            for p in self.list_deliverables(pid)[:40]:
                _add("deliverable", p, 2)
            for p, e in (meta.get("file_manifest") or {}).items():
                _add("manifest", f"{p}: {e.get('desc', '')}", 2)
            _add("ledger", meta.get("design_ledger") or "", 1)
            for r in (meta.get("research_index") or [])[:10]:
                _add("research", str(r.get("topic") or ""), 1)
            for ev in self.list_events(pid, limit=per_project_events,
                                       event_type="work_log"):
                pl = ev.get("payload") or {}
                _add("work_log",
                     f"{pl.get('request', '')} {pl.get('note', '')} "
                     + " ".join(str(f) for f in (pl.get('files') or [])), 1)
            if score:
                results.append({
                    "project_id": pid, "title": proj.get("title"),
                    "status": proj.get("status"), "score": score,
                    "matches": matches[:5],
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max(1, int(limit))]

    def list_children(self, project_id: str) -> List[Dict[str, Any]]:
        """Projects forked FROM ``project_id`` via create_version
        (``metadata.parent_project_id``), any status. Lineage was one-way
        until 2026-07-25 — "what versions of X exist?" was unanswerable and
        a double-fork was undetectable."""
        project_id = _canon_id(project_id)
        out = []
        for p in self.list_projects():
            if str((p.get("metadata") or {})
                   .get("parent_project_id") or "") == project_id:
                out.append(p)
        return out

    # ------------------------------------------------------------------ release dossier

    def get_release(self, project_id: str) -> Dict[str, Any]:
        """The release dossier (``metadata.release``), or {} when absent."""
        proj = self.get_project(project_id)
        rel = ((proj or {}).get("metadata") or {}).get("release")
        return dict(rel) if isinstance(rel, dict) else {}

    def set_release(self, project_id: str, release: Dict[str, Any]) -> bool:
        """Persist the release dossier (atomic) and render RELEASE.md.

        The dossier is the RELEASED project's operational runbook — usage
        directions, verified service commands/ports, URLs, deliverables —
        composed at release time from REHEARSED facts (the tool cold-starts
        the services before writing this). Injected runbook-first by the
        briefing when the project is resumed/run.
        """
        if not isinstance(release, dict) or not release:
            return False

        def _mutate(meta):
            meta["release"] = release
            return meta

        if self._atomic_metadata_update(project_id, _mutate) is None:
            return False
        try:
            self.render_release_md(project_id)
        except Exception as e:
            logger.debug("RELEASE.md render skipped: %s", e)
        return True

    def render_release_md(self, project_id: str) -> Optional[str]:
        """Write RELEASE.md into the workspace from the stored dossier
        (atomic, best-effort) — the human/grep-readable twin of
        ``metadata.release``, same dual pattern as the file manifest's
        PROJECT_MAP.md."""
        proj = self.get_project(project_id)
        if not proj:
            return None
        rel = self.get_release(project_id)
        ws = (proj.get("workspace_dir") or "").strip()
        if not rel or not ws:
            return None
        lines = [
            f"# RELEASE — {proj.get('title') or project_id} "
            f"(v{rel.get('version', 1)})",
            "",
            f"Released: {rel.get('released_at', '')}",
            "",
            "## How to use",
            str(rel.get("directions") or "").strip() or "(no directions recorded)",
        ]
        if rel.get("services"):
            lines += ["", "## Services (rehearsed at release)"]
            for s in rel["services"]:
                lines.append(
                    f"- `{s.get('name')}` — `{s.get('command')}`"
                    + (f" (port {s['port']})" if s.get("port") else ""))
        if rel.get("urls"):
            lines += ["", "## URLs"]
            lines += [f"- {u}" for u in rel["urls"]]
        if rel.get("deliverables"):
            lines += ["", "## Files"]
            for d in rel["deliverables"]:
                lines.append(f"- `{d.get('path')}`"
                             + (f" — {d['desc']}" if d.get("desc") else ""))
        lines += ["", "_This project is RELEASED (immutable). "
                      "Changes go to a new version: "
                      "`manage_projects action=create_version`._", ""]
        try:
            ws_path = Path(ws)
            ws_path.mkdir(parents=True, exist_ok=True)
            target = ws_path / "RELEASE.md"
            tmp = target.with_suffix(".md.tmp")
            tmp.write_text("\n".join(lines), encoding="utf-8")
            os.replace(tmp, target)
            return str(target)
        except Exception as e:
            logger.debug("RELEASE.md write failed: %s", e)
            return None

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
        insertion dropped first) and ``CONFIG_MAX_CHARS`` total. The
        read-modify-write is atomic across processes (see
        ``_atomic_metadata_update``) so a concurrent config / ledger write
        can't clobber a sibling key. Returns the updated config map."""
        key = " ".join((key or "").split())
        if not key:
            # No key to upsert — return the current map without writing.
            # Also yields {} when the project is missing, as before.
            return self.get_config(project_id)
        value = " ".join((value or "").split())[: self.CONFIG_MAX_VALUE_CHARS]

        result = {"cfg": {}}

        def _mutate(meta):
            cfg = dict(meta.get("config") or {}) if isinstance(meta.get("config"), dict) else {}
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
            result["cfg"] = cfg
            return meta

        updated = self._atomic_metadata_update(project_id, _mutate)
        if updated is None:  # project not found
            return {}
        return result["cfg"]

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
            # Per-type retention for the high-churn bookkeeping types —
            # one row per status write / field update / request meant
            # monotonic unbounded growth on any long-lived project
            # (EVENTS_MAX_LIMIT caps reads only, never the table). The
            # newest _EVENTS_RETAIN_PER_TYPE rows comfortably exceed every
            # reader's window (briefing slices read ≤ 20).
            if event_type in ("task_updated", "project_updated", "work_log"):
                conn.execute(
                    "DELETE FROM project_events WHERE project_id = ? "
                    "AND type = ? AND id NOT IN ("
                    "  SELECT id FROM project_events "
                    "  WHERE project_id = ? AND type = ? "
                    "  ORDER BY id DESC LIMIT ?)",
                    (project_id, event_type, project_id, event_type,
                     self._EVENTS_RETAIN_PER_TYPE),
                )
            conn.commit()
            return new_id

    # ── work log (2026-07-18) ────────────────────────────────────────
    #
    # The automatic per-request record of interactive work on a project.
    # Before this, agent.py never wrote to the project store — everything
    # depended on the model voluntarily calling task_update, so any work
    # outside an open task (all post-completion debugging, notably) left
    # zero trace: the 2026-07-17 game-project session logged 6 debugging
    # requests and one root-cause fix, and the store recorded none of it
    # (last event 21:41, session ran to 06:20). The work log is the
    # deterministic write-back: one bounded `work_log` event per request
    # that did real work while the project was bound, written by the
    # finalize chain — no LLM cooperation required.

    #: hard caps so a work_log event stays a compact, injectable record.
    WORK_LOG_REQUEST_CHARS = 220
    WORK_LOG_NOTE_CHARS = 300
    WORK_LOG_MAX_FILES = 12

    def add_work_log(self, project_id: str, *, request: str = "",
                     files: Optional[List[str]] = None,
                     tools: Optional[Dict[str, int]] = None,
                     commands: Optional[List[str]] = None,
                     outcome: str = "",
                     note: str = "",
                     failure_dimension: str = "") -> int:
        """Append one bounded work-log event for a request that did real
        work on this project. ``files`` = project-relative paths written;
        ``tools`` = {tool_name: successful_call_count}; ``commands`` =
        heads of successful shell commands (execute-created state — clones,
        script outputs — is invisible to the file accumulator, and its
        absence caused a re-clone strike on 2026-07-18); ``outcome`` = a
        short label ("completed" / verifier outcome / "had_failures");
        ``note`` = the head of the final response (what was concluded);
        ``failure_dimension`` = harness dimension the turn's failures were
        attributed to (core/failure_dimension.py), empty on success."""
        file_list = sorted({str(f).strip() for f in (files or []) if str(f).strip()})
        extra = len(file_list) - self.WORK_LOG_MAX_FILES
        payload = {
            "request": " ".join(str(request or "").split())[: self.WORK_LOG_REQUEST_CHARS],
            "files": file_list[: self.WORK_LOG_MAX_FILES],
            "files_truncated": max(0, extra),
            "tools": {str(k): int(v) for k, v in list((tools or {}).items())[:8]},
            "commands": [" ".join(str(c).split())[:90]
                         for c in (commands or [])[:5] if str(c).strip()],
            "outcome": str(outcome or "")[:60],
            "note": " ".join(str(note or "").split())[: self.WORK_LOG_NOTE_CHARS],
            "failure_dimension": str(failure_dimension or "")[:24],
        }
        return self.log_event(project_id, None, "work_log", payload)

    def recent_work_logs(self, project_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Newest-first work-log events, for the briefing and status views."""
        return self.list_events(project_id, limit=limit, event_type="work_log")

    def file_history(self, project_id: str, rel_path: str,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Newest-first journal slice for ONE file: every work_log /
        artifact_added / autoadvance event whose payload touched it.

        Answers "what happened to X?" from the journal instead of forcing a
        re-read of the file (2026-07-24). Both sides are normalized through
        ``_normalize_rel_path`` because live payloads carry a mix of forms —
        work_log ``files`` holds bare names AND absolute
        ``/workspace/projects/<id>/…`` paths (observed on 6a471d630e81).
        Returns compact rows: {ts, type, outcome, request, note}.
        """
        project_id = _canon_id(project_id)
        target = self._normalize_rel_path(project_id, rel_path)
        if target is None:
            return []
        out: List[Dict[str, Any]] = []
        # Bounded scan over the newest events (EVENTS_MAX_LIMIT cap inside).
        for ev in self.list_events(project_id, limit=self.EVENTS_MAX_LIMIT):
            p = ev.get("payload") or {}
            candidates: List[str] = []
            files = p.get("files")
            if isinstance(files, list):
                candidates.extend(str(f) for f in files)
            for key in ("path", "rel", "rel_path", "file", "payload"):
                v = p.get(key)
                if isinstance(v, str) and v.strip():
                    candidates.append(v)
            hit = any(
                self._normalize_rel_path(project_id, c) == target
                for c in candidates
            )
            if not hit:
                continue
            out.append({
                "ts": ev.get("ts"),
                "type": ev.get("type"),
                "outcome": str(p.get("outcome") or p.get("status") or ""),
                "request": str(p.get("request") or p.get("description") or "")[:160],
                "note": str(p.get("note") or p.get("result_summary") or "")[:200],
            })
            if len(out) >= max(1, int(limit)):
                break
        return out

    #: hard ceiling for list_events — a negative LIMIT is "no limit" to
    #: SQLite, so an unclamped caller (the tool passes limit through
    #: verbatim) could dump the entire event log.
    EVENTS_MAX_LIMIT = 500

    def list_events(self, project_id: str, limit: int = 50,
                    event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        project_id = _canon_id(project_id)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, self.EVENTS_MAX_LIMIT))
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
