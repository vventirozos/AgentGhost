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
from collections import OrderedDict
from enum import Enum
from pathlib import Path

from ..tools.file_system import write_text_nofollow as _fs_write_nofollow
from typing import Any, Dict, List, Optional, Tuple

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
    revision_count INTEGER NOT NULL DEFAULT 0,  -- §4PV: legacy, always 0; nothing reads or writes it
    actual_tool_used TEXT,
    estimated_cost REAL NOT NULL DEFAULT 0.0,
    actual_cost REAL NOT NULL DEFAULT 0.0,
    depth INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    closed_req_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',  -- §4SU Phase 2: declared task kind (core/task_kind.py); '' = undeclared
    wait_on TEXT NOT NULL DEFAULT '',   -- §4SU Phase 3 (core/task_wait.py): owner | owner_turn | dependency
    wait_code TEXT NOT NULL DEFAULT '',
    wait_why TEXT NOT NULL DEFAULT '',
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

-- §4SX B0: each conversation's focus project (at most one per conversation;
-- many conversations may focus the same project). Keyed by the request's
-- conversation key (``session:<id>`` / ``slack:<...>``).
CREATE TABLE IF NOT EXISTS conversation_focus (
    conversation_key TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    bound_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_focus_project ON conversation_focus(project_id);

-- §4UM: the project's change log ("living projects"). kind 'change' = one owner
-- request that changed the project's files (numbered 1, 2, …); kind 'deploy' =
-- the files went live (a lab deploy / keep / switch / put). Mirrored to the
-- project folder's PROJECT_CHANGES.md.
CREATE TABLE IF NOT EXISTS project_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'change',
    num INTEGER,
    ts REAL NOT NULL,
    req_id TEXT NOT NULL DEFAULT '',
    ask TEXT NOT NULL DEFAULT '',
    added_json TEXT NOT NULL DEFAULT '[]',
    changed_json TEXT NOT NULL DEFAULT '[]',
    removed_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    deployed INTEGER NOT NULL DEFAULT 0,
    deployed_at REAL,
    url TEXT NOT NULL DEFAULT '',
    pushed_json TEXT NOT NULL DEFAULT '[]',
    meta_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_changes_project ON project_changes(project_id, ts);
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


def _contained_workspace(store, ws):
    """Resolve a project workspace and refuse one that escapes the sandbox.

    Returns the resolved Path, or None when it leaves `sandbox_root` (which
    a symlinked project directory does). `delete_project` already resolves
    and contains its workspace 1300 lines below; these writers did not.
    """
    from pathlib import Path as _P
    try:
        resolved = _P(ws).resolve()
    except (OSError, ValueError):
        return None
    root = getattr(store, "sandbox_root", None)
    if not root:
        return resolved          # unscoped store (tests) — nothing to contain
    try:
        root_r = _P(root).resolve()
    except (OSError, ValueError):
        return None
    if resolved != root_r and root_r not in resolved.parents:
        return None
    return resolved


#: files a released app WRITES at runtime (its database and sqlite's side
#: files): kept writable by the release's immutable flag
_RUNTIME_STATE_RE = re.compile(r"\.(?:db|sqlite3?|db3)(?:-journal|-wal|-shm)?$", re.IGNORECASE)


def _is_runtime_state(name: str) -> bool:
    return bool(_RUNTIME_STATE_RE.search(str(name)))


def _inside_sandbox(store, ws: Path) -> bool:
    """A workspace must lie under the store's sandbox root, when one is
    known (a store without one keeps the old behaviour)."""
    root = getattr(store, "sandbox_root", None)
    if not root:
        return True
    try:
        r = Path(root).resolve()
        w = ws.resolve()
        return w != r and r in w.parents
    except OSError:
        return False


# §4UL (2e): the DONE clean-up sweeps a request deferred to its end —
# {request_id: [(store, project_id, queued_at)]}. The agent's turn end calls
# flush_deferred_sweeps(request_id); a request that never flushes (an API
# route, a crash) is swept by the next deferral or flush once
# DEFERRED_SWEEP_MAX_AGE_S have passed.
_DEFERRED_SWEEPS: Dict[str, List[tuple]] = {}
_DEFERRED_LOCK = threading.Lock()
#: §4UL review m4: a deferral is swept without its request's flush only after this long — longer than any
#: request runs (the 180-min deadline cap), so it never fires under a request that is still working. A
#: request ends through handle_chat's finally (which flushes); this is only the net for one that did not. The
#: queue is in memory: a crash loses it (the project's scratch files then stay until its next DONE).
DEFERRED_SWEEP_MAX_AGE_S = 4 * 3600.0

#: §4UL review M5: a project task's turn (``sub-ptask-…``, `core/project_turns`) runs INSIDE the owner's request
#: (or alone, for idle autopilot). {child request id: the request it runs in} — the child's DONE sweeps wait for
#: that request's end, and a write the outer request makes after the child closed the project's last task is the
#: same work. Bounded (the oldest entries go); never cleared at the child's end (the outer request reads it later).
_REQUEST_PARENTS: "OrderedDict[str, str]" = OrderedDict()
_REQUEST_PARENTS_MAX = 512


def set_request_parent(child: str, parent: str) -> None:
    """Record that request ``child`` runs inside request ``parent`` (§4UL review M5). Never raises."""
    child, parent = str(child or ""), str(parent or "")
    if not child or not parent or child == parent or parent == "SYSTEM":
        return
    with _DEFERRED_LOCK:
        _REQUEST_PARENTS[child] = parent
        _REQUEST_PARENTS.move_to_end(child)
        while len(_REQUEST_PARENTS) > _REQUEST_PARENTS_MAX:
            _REQUEST_PARENTS.popitem(last=False)


def request_root(rid: str) -> str:
    """The outermost request ``rid`` runs in (itself when it runs alone)."""
    cur, seen = str(rid or ""), set()
    with _DEFERRED_LOCK:
        while cur in _REQUEST_PARENTS and cur not in seen and len(seen) < 8:
            seen.add(cur)
            cur = _REQUEST_PARENTS[cur]
    return cur


def _sweep_request_id() -> str:
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""
    return "" if rid in ("", "SYSTEM") else request_root(rid)


def _run_deferred(items) -> int:
    n = 0
    for store, pid, _t in items:
        try:
            # a later write in the same request may have reopened it — the
            # next DONE fires again. §4UM: a ready project (planned work
            # finished, open) is swept like a DONE one was.
            if (str((store.get_project(pid) or {}).get("status") or "").upper() != "DONE"
                    and not store.planned_work_done(pid)):
                continue
            store._run_project_done(pid)
            n += 1
        except Exception:  # noqa: BLE001
            logger.warning("deferred DONE sweep failed for %s", pid, exc_info=True)
    return n


def _flush_stale_sweeps(exclude: str = "") -> int:
    cutoff = time.time() - DEFERRED_SWEEP_MAX_AGE_S
    stale: List[tuple] = []
    with _DEFERRED_LOCK:
        for rid in list(_DEFERRED_SWEEPS):
            if rid == exclude:
                continue
            keep = [x for x in _DEFERRED_SWEEPS[rid] if x[2] >= cutoff]
            stale += [x for x in _DEFERRED_SWEEPS[rid] if x[2] < cutoff]
            if keep:
                _DEFERRED_SWEEPS[rid] = keep
            else:
                _DEFERRED_SWEEPS.pop(rid, None)
    return _run_deferred(stale)


def flush_deferred_sweeps(request_id: str) -> int:
    """§4UL (2e): run the DONE clean-up sweeps ``request_id`` deferred (a
    project that is no longer DONE is skipped). Called at the end of every
    agent turn; never raises. Returns how many sweeps ran."""
    try:
        with _DEFERRED_LOCK:
            items = _DEFERRED_SWEEPS.pop(str(request_id or ""), [])
        return _run_deferred(items) + _flush_stale_sweeps()
    except Exception:  # noqa: BLE001
        logger.warning("deferred sweeps not flushed for %s", request_id, exc_info=True)
        return 0


# §4UM: the change log's per-request accumulator. An OWNER request that changes a project's folder is one
# change; it collects here ({request key: {project: entry}}) and ONE entry is written at the request's end
# (`flush_change_log`, called from handle_chat's finally). The request key is the ROOT request (a project task's
# turn inside the owner's request writes for it); an unattended task turn after the plan was finished is its
# own key (one entry per task turn, ask "task: …"). What changed is found two ways (review M3/M4):
#   * a FOLDER DIFF: a bounded stat snapshot of the project folder (`snapshot_folder`) taken when the request
#     first has the project in play — the project focused at the request's start (`watch_project`), or before
#     the first file_system / execute write into another project — and diffed at the flush: added, changed,
#     removed, whatever made them (a shell heredoc, `sed -i`, `rm`, `mv`, a pinned `execute`);
#   * the NOTED paths (`note_change_write`, the write hooks): a project first touched mid-request with no
#     snapshot before the write, and a folder too large to snapshot.
# Pushes to the project's kept lab collect too (one ``deploy`` entry). Entries are persisted minimally (no
# snapshot) in ``pending_changes.json`` beside the store, so a kill mid-request still writes them at the next
# start (`recover_pending_changes`). A request that never flushes is written by a later flush after
# DEFERRED_SWEEP_MAX_AGE_S.
_PENDING_CHANGES: Dict[str, Dict[str, Dict[str, Any]]] = {}
_PENDING_SEQ = [0]
import weakref as _weakref  # noqa: E402
#: every live ProjectStore (a streamed reply's summary is filled in whichever store holds the entry)
_KNOWN_STORES: "_weakref.WeakSet" = _weakref.WeakSet()
#: the owner's message of each running root request (the ask of its entries — never a task turn's prompt)
_REQUEST_ASK: Dict[str, str] = {}
_CHANGE_SKIP_NAMES = {"PROJECT_CHANGES.md", "PROJECT_MAP.md"}
#: §4UM review M2: the agent-owned change log file's first line — a file of that name without it is the
#: project's own and is never overwritten (a project's own CHANGELOG.md is ordinary content: its edits ARE changes)
CHANGES_FILE_HEADER = "<!-- written by the agent: this project's change log (§4UM). Edits here are overwritten. -->"
#: entries rendered into the file (the store keeps them all)
CHANGES_FILE_MAX = 200
#: the folder snapshot: entries (files AND folders) seen at most, and seconds at most (more = "too large": the
#: noted paths alone count; remembered per project for TOO_LARGE_RECHECK_S), and folders never walked (build/
#: and dist/ are walked: they can be deliverables)
SNAPSHOT_MAX_FILES = 5000
SNAPSHOT_MAX_SECONDS = 1.0
TOO_LARGE_RECHECK_S = 3600.0
_SNAPSHOT_SKIP_DIRS = frozenset({".git", "node_modules", ".deps", "__pycache__", ".venv", "venv", ".tox",
                                 ".pytest_cache", ".mypy_cache", ".cache", ".next"})
#: round 3: skipped only when they ARE what the name suggests (a virtualenv; a cargo build dir) — else content
_SNAPSHOT_SKIP_IF = {"env": ("pyvenv.cfg",), "target": ("CACHEDIR.TAG", ".rustc_info.json")}
#: the walk at the flush (the one that decides the entry) may take this long
SNAPSHOT_FLUSH_SECONDS = 5.0
_TOO_LARGE: Dict[str, float] = {}
PENDING_FILE = "pending_changes.json"
#: §4UM round 2: a diffed path counts only when it moved at/after the request's first write tool (less this
#: slack: a sandbox clock behind a bind mount)
WRITE_TIME_SLACK_S = 1.0
#: §4UM round 2 M1/M2: files the SYSTEM wrote or moved away (a browser screenshot, a check file, the research
#: mirror, the scratch and workspace sweeps): absolute path → when. Never a change of the owner's.
_SYSTEM_PATHS: Dict[str, float] = {}
_SYSTEM_PATHS_MAX = 4000
#: round 3: the same, per request (its own registrations — another request's sweep cannot evict them; dropped at
#: its flush)
_SYSTEM_BY_REQ: Dict[str, set] = {}
_SYSTEM_BY_REQ_MAX = 200
#: §4UM round 2 m1: the last write-tool note of every path: absolute path → (entry key, when) — another
#: request's (a concurrent task turn's) write is not this request's change
_RECENT_NOTES: Dict[str, Tuple[str, float]] = {}
_PERSIST_LOCK = threading.Lock()
#: §4UM round 2 m4: an entry whose write raised is kept and retried at the next flush, this many times at most
FLUSH_MAX_ATTEMPTS = 3


def _abs_key(path) -> str:
    """One spelling of a path for the system/notes registries (its folder resolved; the file itself may be gone)."""
    p = str(path)
    return os.path.join(os.path.realpath(os.path.dirname(p)), os.path.basename(p))


def note_system_path(path, t: Optional[float] = None) -> None:
    """§4UM round 2: the system wrote, or moved away, ``path`` (absolute host path) — the folder diff never
    counts it as the owner's change. Never raises."""
    try:
        k = _abs_key(path)
        rid = _change_rid()
        with _DEFERRED_LOCK:
            _SYSTEM_PATHS.pop(k, None)
            _SYSTEM_PATHS[k] = float(t or time.time())
            while len(_SYSTEM_PATHS) > _SYSTEM_PATHS_MAX:
                _SYSTEM_PATHS.pop(next(iter(_SYSTEM_PATHS)))
            if rid:
                mine = _SYSTEM_BY_REQ.setdefault(rid, set())
                if len(mine) < _SYSTEM_PATHS_MAX:
                    mine.add(k)
                while len(_SYSTEM_BY_REQ) > _SYSTEM_BY_REQ_MAX:
                    _SYSTEM_BY_REQ.pop(next(iter(_SYSTEM_BY_REQ)))
    except Exception:  # noqa: BLE001
        pass


def push_covers(pushed, rel: str) -> bool:
    """§4UM: a push list covers ``rel`` when it names it, or names a folder above it (``"site/"``; ``""`` or
    ``"/"`` = the whole project folder)."""
    for f in pushed or ():
        f = str(f)
        if f == rel or f in ("", "/") or (f.endswith("/") and rel.startswith(f)):
            return True
    return False


def _next_seq() -> int:
    _PENDING_SEQ[0] += 1
    return _PENDING_SEQ[0]


def _change_rid(rid: Optional[str] = None) -> str:
    if rid is None:
        try:
            from ..utils.logging import request_id_context
            rid = str(request_id_context.get() or "")
        except Exception:  # noqa: BLE001
            rid = ""
    rid = str(rid or "")
    return "" if rid in ("", "SYSTEM") else request_root(rid)


def _skip_change_path(rel: str) -> bool:
    _name = rel.split("/")[-1] if rel else ""
    if not rel or _name in _CHANGE_SKIP_NAMES or rel.endswith(".md.tmp") \
            or (_name.startswith(".PROJECT_CHANGES.md.") and _name.endswith(".tmp")):
        return True
    try:
        from ..core.workspace_cleanup import _is_debris
        return bool(_is_debris(rel))
    except Exception:  # noqa: BLE001
        return False


def _project_root(store, project_id: str) -> Optional[Path]:
    try:
        ws = store._default_workspace(_canon_id(project_id))
        root = _contained_workspace(store, ws) if ws else None
        return root if root is not None and root.is_dir() else None
    except Exception:  # noqa: BLE001
        return None


def snapshot_folder(root: Optional[Path], max_files: int = SNAPSHOT_MAX_FILES,
                    max_seconds: float = SNAPSHOT_MAX_SECONDS) -> Optional[Dict[str, List[int]]]:
    """§4UM review M3: {project-relative path: [size, mtime_ns, ctime_ns]} of the regular files under ``root``
    (no link followed; .git, node_modules, .deps, caches, virtualenvs, target/ and dot-folders skipped; the
    system-written files and debris left out). None when more than ``max_files`` entries (every file and folder
    seen counts) or ``max_seconds`` would be needed, or the folder cannot be read; {} for a missing folder."""
    if root is None:
        return {}
    import stat as _stat
    out: Dict[str, List[int]] = {}
    seen = 0
    deadline = time.monotonic() + max_seconds

    def _skip_dir(dirpath: str, d: str) -> bool:
        if d in _SNAPSHOT_SKIP_DIRS or d.startswith("."):
            return True
        marks = _SNAPSHOT_SKIP_IF.get(d)
        return bool(marks) and any(os.path.lexists(os.path.join(dirpath, d, m)) for m in marks)
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not _skip_dir(dirpath, d)]
            seen += len(dirnames) + len(filenames)
            if seen > max_files or time.monotonic() > deadline:
                return None
            for f in filenames:
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                if _skip_change_path(rel):
                    continue
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                if not _stat.S_ISREG(st.st_mode):
                    continue
                out[rel] = [int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns)]
    except Exception:  # noqa: BLE001 — round 3: anything unreadable is "too large" (never a stuck "taking")
        return None
    return out


def _pending_entry(store, project_id: str, key: str, ask: str = "", source: str = "owner") -> Dict[str, Any]:
    bucket = _PENDING_CHANGES.setdefault(key, {})
    k = f"{id(store)}:{project_id}"
    e = bucket.get(k)
    ask = _REQUEST_ASK.get(key) or ask        # the owner's message of the request, when known
    if e is None:
        e = bucket[k] = {"store": store, "pid": project_id, "rid": key, "ask": str(ask or "")[:400],
                         "first": time.time(), "last": time.time(), "files": {}, "pushes": [],
                         "reopened_from": "", "source": source}
    elif ask and not e["ask"]:
        e["ask"] = str(ask)[:400]
    return e


def _persist(store) -> None:
    """Write this store's pending entries (no snapshots) to ``pending_changes.json`` — atomic, best effort. The
    rows are built under the pending lock; one writer at a time (a later state never loses to an older one)."""
    try:
        mem = getattr(store, "memory_dir", None)
        if mem is None:
            return
        with _PERSIST_LOCK:
            with _DEFERRED_LOCK:
                rows = [{k: v for k, v in e.items() if k not in ("store", "snap")}
                        for bucket in _PENDING_CHANGES.values() for e in bucket.values()
                        if e["store"] is store and (e["files"] or e["pushes"] or e.get("wrote"))]
                data = json.dumps(rows, default=str) if rows else ""
            path = Path(mem) / PENDING_FILE
            if not data:
                if path.exists():
                    path.unlink()
                return
            tmp = Path(mem) / f".{PENDING_FILE}.{uuid.uuid4().hex[:8]}.tmp"
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        logger.debug("pending change log not persisted", exc_info=True)


def note_request_ask(request_id: str, text: str) -> None:
    """§4UM review: the OWNER's message of a root request — the ask of every entry it writes (a project task's
    turn inside it has its own prompt in ``last_user_content``)."""
    rid = _change_rid(request_id)
    if rid and rid == str(request_id or ""):        # a root request only: a task turn's prompt is never the ask
        with _DEFERRED_LOCK:
            _REQUEST_ASK[rid] = " ".join(str(text or "").split())[:400]
            while len(_REQUEST_ASK) > 200:
                _REQUEST_ASK.pop(next(iter(_REQUEST_ASK)))


def watch_project(store, project_id: str, *, rid: Optional[str] = None, key: Optional[str] = None,
                  ask: str = "", source: str = "owner", task_id: str = "", write: bool = False,
                  evidence: bool = False) -> bool:
    """§4UM review M3: the request has ``project_id`` in play — take the folder snapshot its flush diffs
    against (once per request and project). ``key``: the entry's own key (an unattended task turn); default the
    root request. ``write``: a writing tool is about to run on the project (round 2: only such a request's
    folder is diffed at all). Returns True when a snapshot was taken now. Never raises."""
    try:
        project_id = _canon_id(project_id)
        k = key or _change_rid(rid)
        if not k or not project_id:
            return False
        ck = f"{id(store)}:{project_id}"
        with _DEFERRED_LOCK:
            e = _pending_entry(store, project_id, k, ask, source)
            if task_id:
                e["task_id"] = str(task_id)
            if write:
                _mark_wrote(e, evidence)
            if "snap" in e:
                return False
            if time.time() - _TOO_LARGE.get(ck, 0.0) < TOO_LARGE_RECHECK_S:
                e["snap"] = "too_large"         # walked and too large within the hour: not walked again
                return False
            e["snap"] = "taking"
            e["snap_seq"] = _next_seq()         # writes noted before it are not in its "before"
        snap = None
        try:
            snap = snapshot_folder(_project_root(store, project_id))
        finally:
            with _DEFERRED_LOCK:
                e["snap"] = snap if snap is not None else "too_large"
                if snap is None:
                    _TOO_LARGE[ck] = time.time()
                else:
                    _TOO_LARGE.pop(ck, None)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("project folder not watched for %s", project_id, exc_info=True)
        return False


def _mark_wrote(e: Dict[str, Any], evidence: bool = False) -> None:
    """(under the lock) a writing tool ran on the entry's project in this request: its folder is diffed. Round 4:
    ``evidence`` — the tool REALLY wrote (a reported path, a command with write targets): only then may the entry
    name no file ("files not all listed" on a folder too large to compare); a pinned `pytest` / `ls` never does."""
    if not e.get("wrote"):
        e["wrote"] = True
        e["wrote_t"] = time.time()
    if evidence:
        e["write_evidence"] = True


def mark_task_write(store, project_id: str, rid: Optional[str] = None, evidence: bool = False) -> bool:
    """§4UM round 2: an UNATTENDED task turn (its own entry, keyed by its request id) runs a writing tool on
    ``project_id`` — mark that entry (only one the task turn's start opened; never a new one). Never raises."""
    try:
        if rid is None:
            from ..utils.logging import request_id_context
            rid = str(request_id_context.get() or "")
        project_id = _canon_id(project_id)
        cur, chain = str(rid or ""), []
        with _DEFERRED_LOCK:
            while cur and cur not in chain and len(chain) < 8:
                chain.append(cur)
                cur = _REQUEST_PARENTS.get(cur, "")
            for k in chain:
                e = (_PENDING_CHANGES.get(k) or {}).get(f"{id(store)}:{project_id}")
                if e is not None and e.get("source") == "task":
                    _mark_wrote(e, evidence)
                    return True
        return False
    except Exception:  # noqa: BLE001
        return False


def note_change_write(store, project_id: str, rel: str, *, ask: str = "", rid: Optional[str] = None,
                      reopened_from: str = "", known: Optional[set] = None, persist: bool = True) -> Optional[str]:
    """§4UM: record that this request wrote ``rel`` (project-relative) in ``project_id``'s folder. Decides
    added / changed / removed from disk: gone → removed; a file the project never knew (not registered,
    described, or in an earlier change) and born during this request → added; else changed. (The folder
    diff at the flush overrides it for every path it saw.) Returns the kind, or None when nothing was
    recorded. Never raises."""
    try:
        rid = _change_rid(rid)
        project_id = _canon_id(project_id)
        rel = ProjectStore._normalize_rel_path(project_id, rel) or ""
        if not rid or not project_id or _skip_change_path(rel):
            return None
        ws = store._default_workspace(project_id)
        full = Path(ws) / rel if ws else None
        if full is not None and full.is_dir():
            return None
        with _DEFERRED_LOCK:
            e = _pending_entry(store, project_id, rid, ask)
            if reopened_from and not e["reopened_from"]:
                e["reopened_from"] = reopened_from
            prev = e["files"].get(rel)
            _mark_wrote(e, True)
        if full is not None and not full.exists():
            kind = "removed"
            if prev and prev["kind"] == "added":
                with _DEFERRED_LOCK:
                    e["files"].pop(rel, None)       # made and removed in the same request: never there
                if persist:
                    _persist(store)
                return "removed"
        elif prev:
            kind = "added" if prev["kind"] == "added" else "changed"
        else:
            if known is None:
                known = known_project_files(store, project_id)
            kind = "changed" if rel in known else "added"
            if kind == "added" and full is not None:
                try:
                    born = getattr(full.stat(), "st_birthtime", None)
                    if born is not None and float(born) < e["first"] - 5.0:
                        kind = "changed"            # an older file the records never listed
                except OSError:
                    pass
        with _DEFERRED_LOCK:
            e["files"][rel] = {"kind": kind, "seq": _next_seq(), "t": time.time()}
            e["last"] = time.time()
            _mark_wrote(e, True)
            if full is not None:
                _ak = _abs_key(full)
                _RECENT_NOTES.pop(_ak, None)
                _RECENT_NOTES[_ak] = (rid, time.time())
                while len(_RECENT_NOTES) > _SYSTEM_PATHS_MAX:
                    _RECENT_NOTES.pop(next(iter(_RECENT_NOTES)))
        if persist:
            _persist(store)
        return kind
    except Exception:  # noqa: BLE001
        logger.warning("change not noted for %s %s", project_id, rel, exc_info=True)
        return None


def known_project_files(store, project_id: str) -> set:
    """The files the project's records already name (deliverables, the manifest, earlier changes) — a write to one
    of them is "changed", not "added"."""
    known: set = set()
    try:
        known.update(store.list_deliverables(project_id))
        known.update(store.get_file_manifest(project_id))
        for c in store.list_changes(project_id, kind="change"):
            known.update(c["added"])
            known.update(c["changed"])
    except Exception:  # noqa: BLE001
        pass
    return known


def note_change_writes(store, project_id: str, rels, *, ask: str = "", reopened_from: str = "") -> int:
    """Round 4: a BATCH of written paths (a lab folder `get`) — the records read once, the pending file written
    once. Returns how many were noted."""
    known = known_project_files(store, _canon_id(project_id))
    n = sum(1 for r in rels if note_change_write(store, project_id, r, ask=ask, reopened_from=reopened_from,
                                                 known=known, persist=False))
    _persist(store)
    return n


def note_change_push(store, project_id: str, files=None, *, url: str = "", lab_id: str = "", app: str = "",
                     how: str = "deploy", rid: Optional[str] = None, before_t: Optional[float] = None,
                     key: Optional[str] = None) -> None:
    """§4UM: this request pushed the project's files to its lab (``files=None`` = the whole version: a deploy
    or a switch; else the project-relative paths a `lab put` sent). ``before_t``: only changes written up to
    then are in the version (a switch serves the version staged at that time). Outside any request it is
    recorded at once. Never raises."""
    try:
        project_id = _canon_id(project_id)
        k = key or _change_rid(rid)
        if not project_id:
            return
        if not k:
            store.record_deploy(project_id, url=url, files=files, how=how, lab_id=lab_id, app=app,
                                before_ts=before_t)
            return
        with _DEFERRED_LOCK:
            e = _pending_entry(store, project_id, k)
            e["pushes"].append({"seq": _next_seq(), "t": time.time(),
                                "files": None if files is None else [str(f) for f in files],
                                "url": str(url or ""), "lab_id": str(lab_id or ""), "app": str(app or ""),
                                "how": str(how or "deploy"), "before_t": before_t})
        _persist(store)
    except Exception:  # noqa: BLE001
        logger.warning("lab push not noted for %s", project_id, exc_info=True)


def note_change_stage(store, project_id: str, lab_id: str, at: Optional[float] = None) -> None:
    """§4UM review: a deploy into a STAGED version (not serving yet) — when it was made; `lab switch` of that
    lab serves the changes made up to then."""
    try:
        at = float(at or time.time())

        def _mut(meta):
            st = dict(meta.get("staged_deploys") or {})
            st[str(lab_id)] = at
            meta["staged_deploys"] = dict(sorted(st.items(), key=lambda kv: kv[1])[-8:])
            return meta
        store._atomic_metadata_update(_canon_id(project_id), _mut)
    except Exception:  # noqa: BLE001
        logger.debug("staged deploy not noted for %s", project_id, exc_info=True)


def staged_at(store, project_id: str, lab_id: str) -> Optional[float]:
    try:
        v = (((store.get_project(project_id) or {}).get("metadata") or {}).get("staged_deploys") or {}).get(lab_id)
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def pending_change_files(store, project_id: str) -> set:
    """Paths a running request added or changed in ``project_id`` (not yet in the log) — kept by a sweep."""
    out: set = set()
    with _DEFERRED_LOCK:
        for bucket in _PENDING_CHANGES.values():
            for e in bucket.values():
                if e["store"] is store and e["pid"] == _canon_id(project_id):
                    out.update(r for r, f in e["files"].items() if f["kind"] != "removed")
    return out


def _reply_summary(reply: str) -> str:
    """§4UM review: the final reply as plain text (markdown, links and tool banners stripped, whitespace
    collapsed), ≤300 chars — what the request said it did."""
    txt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", str(reply or ""))
    kept = []
    for ln in txt.splitlines():
        ln = re.sub(r"[*_`#>]+", "", ln).strip(" -•\t")
        if ln and not ln.startswith(("[exited", "<!--")):
            kept.append(ln)
    out = " ".join(" ".join(kept).split())
    return out[:300].rstrip() + ("…" if len(out) > 300 else "")


def _was_ready_before(store, project_id: str, rid: str, t: float) -> bool:
    """The project's first plan was finished by ANOTHER request (``metadata.first_ready_at`` /
    ``first_ready_req``) — or, finished before §4UM, an older roll-up event before ``t``."""
    proj = store.get_project(project_id) or {}
    m = proj.get("metadata") or {}
    if m.get("first_ready_at"):
        by = str(m.get("first_ready_req") or "")
        mine = {rid, request_root(rid)} if rid else set()
        return not (by and (by in mine or request_root(by) in mine))
    for ev in store.list_events(project_id, limit=20, event_type="project_auto_rollup") or []:
        try:
            if float(ev.get("ts") or 0) < t:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _planned_in_request(store, project_id: str, rid: str, first_t: Optional[float] = None) -> bool:
    """The request built the project's FIRST plan (created it, or closed one of its tasks before the plan
    was ever finished) — that is recorded by the tasks, not as an owner change. Once the plan has been
    finished, a request that changes the files is a change even when it adds and closes tasks for it."""
    try:
        if first_t is not None and _was_ready_before(store, project_id, rid, first_t):
            return False
        for t in store.list_tasks(project_id) or []:
            c = str(t.get("closed_req_id") or "")
            if c and (c == rid or request_root(c) == rid):
                return True
        for ev in store.list_events(project_id, limit=5, event_type="project_created") or []:
            c = str((ev.get("payload") or {}).get("req") or "")
            if c and (c == rid or request_root(c) == rid):
                return True
    except Exception:  # noqa: BLE001
        logger.debug("planned-work check skipped for %s", project_id, exc_info=True)
    return False


def _file_t(st: List[int]) -> float:
    """When a snapshotted file last moved: its content (mtime) or its name (ctime: a rename, a copy -p)."""
    return max(st[1], st[2] if len(st) > 2 else 0) / 1e9


def _change_t(st: List[int], since: float) -> float:
    """§4UM-s: when a changed file's change happened — its content time (mtime) when the content moved during
    the request (on or after ``since``); the name time (ctime) only for content older than that (a rename, a
    copy -p brought it in). ctime alone also moves on a metadata touch — an xattr written when the file is
    opened (macOS ``com.apple.lastuseddate#PS``), a chmod — so it never dates a fresh write: req 56dff518's
    entry read 12:19 (index.html's ctime) for files last written at 11:52."""
    m = st[1] / 1e9
    return m if m >= since else _file_t(st)


def _foreign(e: Dict[str, Any], root: Path, rel: str, t: float) -> bool:
    """§4UM round 2: a diffed path that is not this request's change — the system wrote or moved it (a
    screenshot, a check file, the research mirror, a sweep), another request's write tool noted it during this
    request, or it moved before this request ran its first writing tool."""
    ak = _abs_key(root / rel)
    since = float(e["first"]) - WRITE_TIME_SLACK_S
    with _DEFERRED_LOCK:
        if ak in _SYSTEM_BY_REQ.get(str(e["rid"]), ()):
            return True                         # this request's own system write (never evicted)
        sys_t = _SYSTEM_PATHS.get(ak)
        other = _RECENT_NOTES.get(ak)
    if sys_t is not None and sys_t >= since:
        return True
    if other is not None and other[0] != e["rid"] and other[1] >= since:
        return True
    return t < float(e.get("wrote_t") or e["first"]) - WRITE_TIME_SLACK_S


_UNREAD = object()


def _push_times(e: Dict[str, Any]) -> Optional[List[float]]:
    """§4UM-s review: every time a version of ``e``'s project was cut — a push (this request's, another request's
    still pending, a recorded deploy entry) and a staged deploy (``metadata.staged_deploys`` and a pending switch's
    ``before_t``: the version a switch serves). A recorded switch marked its changes when it ran. None when they
    cannot be read (the caller then keeps the later time)."""
    try:
        pid, store = e["pid"], e["store"]
        with _DEFERRED_LOCK:
            pend = [p for b in _PENDING_CHANGES.values() for x in b.values()
                    if x["store"] is store and x["pid"] == pid for p in x["pushes"]]
        out: List[float] = []
        for p in pend + list(e["pushes"]):
            out += [float(p["t"])] + ([float(p["before_t"])] if p.get("before_t") is not None else [])
        meta = (store.get_project(pid) or {}).get("metadata") or {}
        out += [float(v) for v in (meta.get("staged_deploys") or {}).values() if v]
        with store._lock, store._connect() as conn:
            out += [float(r[0]) for r in conn.execute("SELECT ts FROM project_changes WHERE project_id = ? "
                                                      "AND kind = 'deploy'", (pid,)).fetchall()]
        return out
    except Exception:  # noqa: BLE001
        return None


def _push_between(times: Optional[List[float]], lo: float, hi: float) -> bool:
    """A version cut at a time in (lo, hi] — unknown counts as one."""
    return times is None or any(lo < t <= hi for t in times)


def _diffed_files(e: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The entry's files: the paths its write tools noted, and the folder diff (when a snapshot was taken) less
    what is not this request's (`_foreign`). A path first noted as added stays added."""
    files = {r: dict(f) for r, f in e["files"].items() if not _skip_change_path(r)}
    root = _project_root(e["store"], e["pid"])
    snap = e.get("snap")
    now = snapshot_folder(root, max_seconds=SNAPSHOT_FLUSH_SECONDS) if isinstance(snap, dict) else None
    if isinstance(snap, dict) and now is None:
        # round 3 M2: the flush walk failed (too large, too slow, unreadable) — never a silent drop: the entry
        # carries folder_too_large and the paths the tools reported. Round 5: the folder was comparable at the
        # start and is not now — it grew past the cap during the request: that IS write evidence
        e["snap"] = "too_large"
        with _DEFERRED_LOCK:
            _TOO_LARGE[f"{id(e['store'])}:{e['pid']}"] = time.time()
            if e.get("wrote"):
                e["write_evidence"] = True
    for r in list(files):
        there = root is not None and (root / r).exists()
        if files[r]["kind"] != "removed" and not there:     # a noted file gone since (made, then removed)
            if files[r]["kind"] == "added":
                files.pop(r)
            else:
                files[r]["kind"] = "removed"
        elif files[r]["kind"] == "removed" and there:       # removed, then made again
            if now is not None and snap.get(r) is not None and now.get(r) is not None \
                    and snap[r][:2] == now[r][:2]:
                files.pop(r)                    # as it was: no change
            else:
                files[r]["kind"] = "changed"
    if now is None:
        return files
    since = float(e.get("wrote_t") or e["first"]) - WRITE_TIME_SLACK_S     # as `_foreign` reads it
    _cuts: Any = _UNREAD
    for rel in set(snap) | set(now):
        a, b = snap.get(rel), now.get(rel)
        if a is not None and b is not None and a[:2] == b[:2]:
            continue
        if a is None:
            kind, t = "added", _file_t(b)
        elif b is None:
            kind = "removed"
            try:                                # when it went: the folder it was in changed then
                _ps = os.stat((root / rel).parent) if (root / rel).parent.exists() else None
                t = max(_ps.st_mtime, _ps.st_ctime) if _ps is not None else time.time()
            except OSError:
                t = time.time()
        else:
            kind, t = "changed", _file_t(b)
        # §4UM-s: `t` (content OR name time) decides whether the path is this request's; the change is dated by
        # its content's own time — unless a push lies between that and `t` (content renamed into place after a
        # push is not in that push: the later time keeps it out of `_covered` and `record_deploy` alike)
        ct = _change_t(b, since) if b is not None else t
        if ct < t:
            if _cuts is _UNREAD:
                _cuts = _push_times(e)          # once per flush, only when a file's ctime ran ahead of its mtime
            if _push_between(_cuts, ct, t):
                ct = t
        prev = files.get(rel)
        if prev is None and _foreign(e, root, rel, t):
            continue
        if prev is not None and prev["kind"] == "added" and kind == "changed" \
                and int(prev.get("seq") or 0) < int(e.get("snap_seq") or 0):
            kind = "added"                      # m5: added (noted) before the snapshot saw it — still added
        if prev is not None and kind == "removed" and prev["kind"] == "added":
            files.pop(rel, None)
            continue
        files[rel] = {"kind": kind, "t": max(ct, prev["t"]) if prev else ct,
                      **({"seq": prev["seq"]} if prev and prev.get("seq") else {})}
    for r in [r for r, f in files.items() if f["kind"] == "removed" and r not in snap]:
        if any(k.startswith(r.rstrip("/") + "/") for k in snap):
            files.pop(r)                        # a removed FOLDER: its files are listed one by one
    return files


def _covered(rel: str, f: Dict[str, Any], pushes: List[Dict[str, Any]]) -> bool:
    """A push after the write carried ``rel`` (a noted write: by order; a diffed one: by time)."""
    for p in pushes:
        if not (p["files"] is None or push_covers(p["files"], rel)):
            continue
        if p.get("before_t") is not None and f["t"] > float(p["before_t"]) + 1e-6:
            continue                            # a switch: not in the version it serves
        if (p["seq"] > f["seq"]) if f.get("seq") else (p["t"] >= f["t"]):
            return True
    return False


def _write_pending(e: Dict[str, Any], reply: str = "") -> int:
    store, pid, rid = e["store"], e["pid"], e["rid"]
    proj = store.get_project(pid)
    if not proj:
        return 0
    n = 0
    # §4UM round 2: only a request that ran a writing tool on the project has its folder diffed — a read, a
    # search or a screenshot is never a change. A DONE project reopens only at a write tool's noted path
    # (`note_project_write`), never from the diff alone.
    files = _diffed_files(e) if e.get("wrote") and not e.get("files_done") else {}
    pushes = [] if e.get("pushes_done") else list(e["pushes"])
    # round 3 M2: a writing request on a folder that could not be compared is an entry even with no file named
    unlisted = bool(e.get("wrote") and e.get("write_evidence") and not e.get("files_done")
                    and e.get("snap") == "too_large" and not files and not e.get("recovered"))
    if (e.get("wrote") and not files and not unlisted and e.get("snap") == "too_large" and not e.get("files_done")
            and not e.get("recovered")):
        # round 5: a command with no visible write (a generator, `make`) on a folder too large to compare may
        # have changed files — said once, at INFO; no entry (a pinned `pytest` / `ls` must not make one)
        logger.info("change log: project %s is too large to compare; a command ran that may have changed files — "
                    "no change entry", pid)
    if (files or unlisted) and not _planned_in_request(store, pid, rid, e["first"]):
        if str(proj.get("status") or "").upper() == "RELEASED":
            files, unlisted = {}, False         # immutable: never a change entry (changes go to create_version)
    else:
        files, unlisted = {}, False
    if files or unlisted:
        added = sorted(r for r, f in files.items() if f["kind"] == "added")
        changed = sorted(r for r, f in files.items() if f["kind"] == "changed")
        removed = sorted(r for r, f in files.items() if f["kind"] == "removed")
        live = {r: f for r, f in files.items() if f["kind"] != "removed"}
        if unlisted and not files:              # what changed is unknown: live once a whole version went after it
            _wt = {"t": float(e.get("last") or time.time())}
            deployed = any(_covered("", _wt, [p]) for p in pushes if p["files"] is None)
        elif live:
            deployed = all(_covered(r, f, pushes) for r, f in live.items())
        else:                                   # removal only: live once a whole version went after it
            deployed = all(_covered(r, f, [p for p in pushes if p["files"] is None]) for r, f in files.items())
        last_t = max([f["t"] for f in files.values()] or [float(e.get("last") or time.time())])
        url = next((p["url"] for p in reversed(pushes) if p["url"]), "")
        meta: Dict[str, Any] = {}
        if e["reopened_from"]:
            meta["reopened_from"] = e["reopened_from"]
        if e.get("source") == "task":
            meta.update(source="task", task_id=e.get("task_id") or "")
        if e.get("recovered"):
            meta["recovered"] = True
        if e.get("snap") == "too_large":
            meta["folder_too_large"] = True
        entry = store.add_change(pid, ts=last_t, req_id=rid, ask=e["ask"], added=added, changed=changed,
                                 removed=removed, summary=_reply_summary(reply), deployed=deployed,
                                 deployed_at=(max(p["t"] for p in pushes) if deployed and pushes else None),
                                 url=url if deployed else "", render=not pushes, meta=meta or None)
        e["files_done"] = True                  # a retry after a failed deploy write adds no second entry
        if entry:
            n += 1
            if e["reopened_from"]:
                store.log_event(pid, None, "project_reopened",
                                {"reason": f"reopened by change #{entry['num']}",
                                 "from_status": e["reopened_from"], "req": rid})
    if pushes:
        whole = any(p["files"] is None for p in pushes)
        pfiles = None if whole else sorted({f for p in pushes for f in (p["files"] or [])})
        last = pushes[-1]
        _bt = [p["before_t"] for p in pushes if p["files"] is None and p.get("before_t") is not None]
        store.record_deploy(pid, url=next((p["url"] for p in reversed(pushes) if p["url"]), ""),
                            files=pfiles, how="+".join(dict.fromkeys(p["how"] for p in pushes)),
                            lab_id=last["lab_id"], app=last["app"], req_id=rid,
                            ts=max(p["t"] for p in pushes),
                            before_ts=(max(_bt) if _bt and len(_bt) == sum(1 for p in pushes
                                                                          if p["files"] is None) else None))
        e["pushes_done"] = True
        n += 1
    return n


def flush_change_log(request_id: str, reply: str = "") -> int:
    """§4UM: write the change log entries ``request_id`` collected (one per project it changed, plus one
    deploy entry when it pushed to the project's lab). Called at the end of every agent turn, BEFORE the
    deferred sweeps (the sweep keeps the change's files); also writes any other request's entries older
    than DEFERRED_SWEEP_MAX_AGE_S. Never raises. Returns how many entries were written."""
    n = 0
    try:
        rid = str(request_id or "")
        # entries are keyed by the ROOT request: a project task's turn inside the owner's request (its own
        # handle_chat ends first) flushes nothing of the owner's — the owner's request end writes them
        cutoff = time.time() - DEFERRED_SWEEP_MAX_AGE_S
        mine: List[Dict[str, Any]] = []
        stale: List[Dict[str, Any]] = []
        with _DEFERRED_LOCK:
            if rid:
                mine += list((_PENDING_CHANGES.pop(rid, {}) or {}).values())
                _REQUEST_ASK.pop(rid, None)
            _sys_mine = _SYSTEM_BY_REQ.get(rid) if rid else None
            for k in list(_PENDING_CHANGES):
                b = _PENDING_CHANGES[k]
                if b and all(e["last"] < cutoff and e["first"] < cutoff for e in b.values()):
                    stale += list(_PENDING_CHANGES.pop(k).values())
        stores = {}
        for e, rep in [(x, reply) for x in mine] + [(x, "") for x in stale]:
            stores[id(e["store"])] = e["store"]
            try:
                n += _write_pending(e, rep)
            except Exception:  # noqa: BLE001
                # §4UM round 2 m4: kept pending (and on disk) — the next flush retries it, a few times at most
                e["attempts"] = int(e.get("attempts") or 0) + 1
                if e["attempts"] < FLUSH_MAX_ATTEMPTS:
                    logger.warning("change log entry not written for %s (kept, retried at the next flush)",
                                   e.get("pid"), exc_info=True)
                    with _DEFERRED_LOCK:
                        _PENDING_CHANGES.setdefault(str(e["rid"]), {})[f"{id(e['store'])}:{e['pid']}"] = e
                else:
                    logger.warning("change log entry for %s dropped after %d failed writes", e.get("pid"),
                                   e["attempts"], exc_info=True)
        for st in stores.values():
            _persist(st)
        if rid:
            with _DEFERRED_LOCK:
                if _SYSTEM_BY_REQ.get(rid) is _sys_mine:
                    _SYSTEM_BY_REQ.pop(rid, None)
    except Exception:  # noqa: BLE001
        logger.warning("change log not flushed for %s", request_id, exc_info=True)
    return n


def set_change_summary(request_id: str, reply: str) -> int:
    """§4UM review: a STREAMED reply is complete only after handle_chat's finally wrote the request's entries —
    fill their empty summaries now (and re-render the file). Returns how many were set."""
    rid = _change_rid(request_id)
    summary = _reply_summary(reply)
    if not rid or not summary:
        return 0
    n = 0
    for store in list(_KNOWN_STORES):
        try:
            with store._lock, store._connect() as conn:
                rows = conn.execute("SELECT id, project_id FROM project_changes WHERE req_id = ? AND kind = 'change' "
                                    "AND summary = ''", (rid,)).fetchall()
                for r in rows:
                    conn.execute("UPDATE project_changes SET summary = ? WHERE id = ?", (summary, r["id"]))
                conn.commit()
            for pid in {r["project_id"] for r in rows}:
                store.render_changelog(pid)
            n += len(rows)
        except Exception:  # noqa: BLE001
            logger.debug("change summary not set for %s", rid, exc_info=True)
    return n


def _already_logged(store, project_id: str, rid: str) -> set:
    """The kinds ("change", "deploy") the log already holds for (project, request)."""
    if not rid:
        return set()
    try:
        with store._lock, store._connect() as conn:
            return {r[0] for r in conn.execute("SELECT DISTINCT kind FROM project_changes WHERE project_id = ? "
                                               "AND req_id = ?", (project_id, rid)).fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def recover_pending_changes(store) -> int:
    """§4UM review: entries a killed process left in ``pending_changes.json`` are written now (no reply, no
    folder diff — the noted paths and pushes), marked recovered. Call once at start. Returns entries written.
    Round 2 m3/m4: a (project, request) already in the log is not written twice; the file goes only once every
    row was written (a row that fails stays pending: retried at a later flush); a corrupt file is moved aside
    to ``pending_changes.json.bad`` with one warning."""
    n = 0
    try:
        path = Path(store.memory_dir) / PENDING_FILE
        if not path.exists():
            return 0
        try:
            rows = json.loads(path.read_text(encoding="utf-8") or "[]")
            if not isinstance(rows, list):
                raise ValueError("not a list")
        except (ValueError, UnicodeDecodeError) as exc:
            os.replace(path, path.with_name(PENDING_FILE + ".bad"))
            logger.warning("change log: %s is unreadable (%s) — moved aside to %s.bad", PENDING_FILE, exc,
                           PENDING_FILE)
            return 0
        for r in rows:
            if not isinstance(r, dict) or not r.get("pid"):
                continue
            _logged = _already_logged(store, str(r["pid"]), str(r.get("rid") or ""))
            e = dict(r, store=store, recovered=True)
            # round 3: what the flush already wrote (a kill between its two writes) is not written again; the rest is
            if "change" in _logged or not r.get("files"):
                e["files_done"] = True
            if "deploy" in _logged or not r.get("pushes"):
                e["pushes_done"] = True
            if e.get("files_done") and e.get("pushes_done"):
                continue
            e.setdefault("files", {})
            e.setdefault("pushes", [])
            e.setdefault("reopened_from", "")
            e.setdefault("rid", "")
            e.setdefault("first", time.time())
            e["last"] = 0.0                     # stale: a later flush retries it if this write fails
            try:
                n += _write_pending(e, "")
            except Exception:  # noqa: BLE001
                logger.warning("recovered change log entry not written for %s (kept pending)", r.get("pid"),
                               exc_info=True)
                with _DEFERRED_LOCK:
                    _PENDING_CHANGES.setdefault(str(e["rid"] or "recovered"), {})[f"{id(store)}:{e['pid']}"] = e
        _persist(store)                         # gone when every row was written; else the rows still pending
        if n:
            logger.info("change log: %d entr%s recovered from a request that never ended", n, "y" if n == 1 else "ies")
    except Exception:  # noqa: BLE001
        logger.warning("pending change log not recovered", exc_info=True)
    return n


def backfill_change(store, project_id: str, ask: str, added=(), changed=(), summary: str = "",
                    ts: Optional[float] = None, req: str = "", deployed: bool = False, url: str = "",
                    removed=(), lab_id: str = "", how: str = "put",
                    allow_duplicate: bool = False) -> Optional[Dict[str, Any]]:
    """§4UM: add a past change to the log by hand (work done before the log existed). Idempotent: an entry
    with the same ``req``, or the same ``ts`` (±60 s) and ask, is returned as it is (``meta.existing``). With
    ``deployed`` and a ``url`` it also records the push (a deploy entry at the same time; ``metadata.live``
    only when no newer deploy is recorded); ``deployed`` without a url marks only this entry."""
    norm_ask = " ".join(str(ask or "").split())[:300]
    for c in ([] if allow_duplicate else store.list_changes(project_id, kind="change")):
        if (req and c.get("req_id") == req) or (
                ts is not None and abs(float(c["ts"]) - float(ts)) <= 60 and c.get("ask") == norm_ask):
            return dict(c, meta=dict(c.get("meta") or {}, existing=True))
    e = store.add_change(project_id, ts=ts, req_id=req, ask=ask, added=added, changed=changed,
                         removed=removed, summary=summary, render=not (deployed and url),
                         deployed=bool(deployed and not url), meta={"backfilled": True})
    if e and deployed and url:
        store.record_deploy(project_id, url=url, files=list(added) + list(changed), how=how,
                            lab_id=lab_id, req_id=req, ts=float(e["ts"]) + 0.001)
    return e


def _when(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return "?"


def change_line(e: Dict[str, Any]) -> str:
    """One change-log entry as one line (§4UM)."""
    if e.get("kind") == "deploy":
        m = e.get("meta") or {}
        what = "whole folder" if m.get("whole") else ", ".join((e.get("changed") or [])[:6]) or "files"
        return (f"{_when(e['ts'])} LIVE ({m.get('how') or 'deploy'}: {what})"
                + (f" → {e['url']}" if e.get("url") else "")
                + (f", includes #{m['through']}" if m.get("through") else ""))
    files = []
    if e.get("added"):
        files.append("+" + ", +".join(e["added"][:6]))
    if e.get("changed"):
        files.append("~" + ", ~".join(e["changed"][:6]))
    if e.get("removed"):
        files.append("-" + ", -".join(e["removed"][:4]))
    return (f"#{e.get('num')} {_when(e['ts'])} · {'live' if e.get('deployed') else 'NOT deployed'}"
            + (f" · asked: \"{str(e.get('ask') or '')[:120]}\"" if e.get("ask") else "")
            + (f" · files: {'; '.join(files)}" if files else "")
            + (" · (files not all listed: folder too large)" if (e.get("meta") or {}).get("folder_too_large") else "")
            + (f" · {str(e.get('summary') or '')[:160]}" if e.get("summary") else ""))


def recent_change_lines(store, project_id: str, n: int = 5) -> List[str]:
    """The last ``n`` change-log entries (changes and deploys), newest first."""
    try:
        return [change_line(e) for e in store.list_changes(project_id, limit=n)]
    except Exception:  # noqa: BLE001
        return []


#: folders the generated layout never walks
_LAYOUT_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".deps", ".venv", "venv", ".pytest_cache",
                     ".mypy_cache", "dist", "build", ".next", ".cache"}


def folder_layout(store, project_id: str, max_files: int = 60) -> List[str]:
    """§4UM: the project's files as they are on disk now (project-relative, sorted; dotfiles, caches and
    dependency folders left out) — the briefing's layout, never a stale record. Last entry "(+N more)"."""
    try:
        ws = store._default_workspace(_canon_id(project_id))
        root = _contained_workspace(store, ws) if ws else None
        if root is None or not root.is_dir():
            return []
        out: List[str] = []
        extra = 0
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in _LAYOUT_SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                if f.startswith(".") or f.endswith((".pyc", ".tmp")):
                    continue
                rel = (Path(dirpath) / f).relative_to(root).as_posix()
                if len(out) < max_files:
                    out.append(rel)
                else:
                    extra += 1
            if len(out) >= max_files and extra > 500:
                break
        if extra:
            out.append(f"(+{extra} more)")
        return out
    except Exception:  # noqa: BLE001
        logger.debug("folder layout skipped for %s", project_id, exc_info=True)
        return []


def live_line(store, project_id: str) -> str:
    """"Live at <url> (last deployed <when>, includes change #N; behind: #a–#b not deployed)" or ""."""
    try:
        li = store.live_info(project_id)
    except Exception:  # noqa: BLE001
        li = None
    if not li:
        return ""
    behind = li.get("behind") or []
    tail = ""
    if behind:
        span = f"#{behind[0]}" if len(behind) == 1 else f"#{behind[0]}–#{behind[-1]}"
        tail = f"; BEHIND: change(s) {span} not deployed yet"
    return (f"Live at {li.get('url') or '(no url)'} (last deployed {_when(li.get('at'))}"
            + (f", includes change #{li['through']}" if li.get("through") else "") + tail + ")")


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
        self._lock = threading.RLock()
        self._init_db()
        _KNOWN_STORES.add(self)

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
        # §4UL (2e): inside a request the sweep waits for the request's end
        # (flush_deferred_sweeps) — a roll-up mid-turn swept files the turn
        # was still writing; outside any request it runs now
        rid = _sweep_request_id()
        if rid:
            with _DEFERRED_LOCK:
                _q = _DEFERRED_SWEEPS.setdefault(rid, [])
                if not any(s is self and p == project_id for s, p, _t in _q):
                    _q.append((self, project_id, time.time()))
            _flush_stale_sweeps(exclude=rid)
            return
        self._run_project_done(project_id)

    def _run_project_done(self, project_id: str) -> None:
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
                ("constraints_json", "TEXT NOT NULL DEFAULT '[]'"),
                # §4E Tier 3 (2026-08-01): the request id of the turn that
                # last closed this task DONE — the join key for retroactive
                # calibration negatives when the task is later reopened.
                ("closed_req_id", "TEXT NOT NULL DEFAULT ''"),
                # §4SU Phase 2: the task's declared kind ('' = undeclared)
                ("kind", "TEXT NOT NULL DEFAULT ''"),
                # §4SU Phase 3: why a task waits, as fields (core/task_wait.py)
                ("wait_on", "TEXT NOT NULL DEFAULT ''"),
                ("wait_code", "TEXT NOT NULL DEFAULT ''"),
                ("wait_why", "TEXT NOT NULL DEFAULT ''"),
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
        self._migrate_task_states(conn)

    def _has_wait_cols(self) -> bool:
        """§4SU Phase 3: the wait columns exist (cached once true)."""
        if getattr(self, "_wait_cols_ok", False):
            return True
        try:
            with self._lock, self._connect() as conn:
                ok = "wait_on" in {r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        except sqlite3.Error:
            ok = False
        if ok:
            self._wait_cols_ok = True
        return ok

    @staticmethod
    def _migrate_task_states(conn: sqlite3.Connection) -> None:
        """§4SU Phase 3, one-off and idempotent: READY / task PAUSED fold
        into PENDING; a waiting row with no wait fields gets them from its
        old text mark (task_wait.wait_from_marks — the marks' only reader)."""
        from ..core.task_wait import wait_from_marks, fold_status
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
            if "wait_on" not in cols:
                return
            conn.execute("UPDATE tasks SET status = 'PENDING' WHERE status = 'READY'")
            rows = conn.execute(
                "SELECT id, status, result_summary, postconditions_json FROM tasks "
                "WHERE status IN ('NEEDS_USER', 'BLOCKED', 'PAUSED') AND wait_on = ''").fetchall()
            for r in rows:
                on, code, why = wait_from_marks(r["status"], r["result_summary"])
                # r6: an old BLOCKED row is the cascade's block only when a child
                # failed and no reason of its own was written; otherwise it is
                # the model's own block, never lifted automatically
                if fold_status(r["status"]) == "BLOCKED":
                    _fr = conn.execute("SELECT failure_reason FROM tasks WHERE id = ?", (r["id"],)).fetchone()
                    _kid_failed = conn.execute(
                        "SELECT 1 FROM tasks WHERE parent_id = ? AND status IN ('FAILED', 'BLOCKED') LIMIT 1",
                        (r["id"],)).fetchone()
                    _own = str((_fr[0] if _fr else "") or "") not in ("", "All ANY-dependency children failed")
                    if not _kid_failed or _own:
                        code = "blocked"
                conn.execute("UPDATE tasks SET status = ?, wait_on = ?, wait_code = ?, wait_why = ? "
                             "WHERE id = ?", (fold_status(r["status"]), on, code, why, r["id"]))
            if rows:
                logger.info("migration §4SU Phase 3: %d waiting task(s) given wait fields", len(rows))
        except sqlite3.Error as e:  # pragma: no cover
            logger.warning("migration §4SU Phase 3 skipped: %s", e)

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
        try:      # §4UI: the briefing says "this message asked for it" only in that request
            from ..utils.logging import request_id_context
            _req = str(request_id_context.get() or "")
        except Exception:  # noqa: BLE001
            _req = ""
        self.log_event(project_id, None, "project_created",
                       {"title": title, "kind": kind_norm, "req": _req})
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

    # ------------------------------------------------ conversation focus (§4SX B0)

    #: a focus row unused this long is pruned (UNIFIED_DESIGN §10.3)
    FOCUS_IDLE_PRUNE_S = 30 * 86400

    def get_focus(self, conversation_key: str) -> Optional[str]:
        """The project this conversation focuses, or None. Refreshes the row's
        ``last_used_at``. A row whose project is gone is dropped. An archive
        clears every row on the project when it happens; a LATER explicit
        switch to an archived project is the owner's choice and is kept
        (review r1: dropping it on read made the switch vanish next turn)."""
        key = str(conversation_key or "")
        if not key:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT f.project_id, p.status FROM conversation_focus f "
                "LEFT JOIN projects p ON p.id = f.project_id "
                "WHERE f.conversation_key = ?", (key,)).fetchone()
            if row is None:
                return None
            if row["status"] is None:
                conn.execute("DELETE FROM conversation_focus WHERE conversation_key = ?", (key,))
                conn.commit()
                return None
            conn.execute("UPDATE conversation_focus SET last_used_at = ? WHERE conversation_key = ?",
                         (_now(), key))
            conn.commit()
            return row["project_id"]

    def set_focus(self, conversation_key: str, project_id: str) -> bool:
        """Point THIS conversation's focus at ``project_id`` (no other
        conversation's row changes). False when the key is empty or the
        project does not exist. Prunes rows idle longer than
        ``FOCUS_IDLE_PRUNE_S``."""
        key = str(conversation_key or "")
        pid = _canon_id(project_id)
        if not key or not pid:
            return False
        now = _now()
        with self._lock, self._connect() as conn:
            if conn.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone() is None:
                return False
            conn.execute(
                "INSERT INTO conversation_focus(conversation_key, project_id, bound_at, last_used_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(conversation_key) DO UPDATE SET "
                "project_id = excluded.project_id, bound_at = excluded.bound_at, "
                "last_used_at = excluded.last_used_at", (key, pid, now, now))
            conn.execute("DELETE FROM conversation_focus WHERE last_used_at < ?",
                         (now - self.FOCUS_IDLE_PRUNE_S,))
            conn.commit()
        return True

    def clear_focus(self, conversation_key: str) -> None:
        key = str(conversation_key or "")
        if not key:
            return
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM conversation_focus WHERE conversation_key = ?", (key,))
            conn.commit()

    def clear_focus_for_project(self, project_id: str) -> int:
        """Drop every conversation's focus on ``project_id`` (archive/delete)."""
        pid = _canon_id(project_id)
        if not pid:
            return 0
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM conversation_focus WHERE project_id = ?", (pid,))
            conn.commit()
            return cur.rowcount or 0

    def focus_rows(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if project_id:
                rows = conn.execute("SELECT * FROM conversation_focus WHERE project_id = ? "
                                    "ORDER BY last_used_at DESC", (_canon_id(project_id),)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM conversation_focus "
                                    "ORDER BY last_used_at DESC").fetchall()
            return [dict(r) for r in rows]

    def update_project(self, project_id: str, metadata_replace: bool = False,
                       **fields) -> bool:
        """Update project fields. ``metadata`` is MERGED (shallow) into the
        existing metadata by default — the blob carries system state
        (design_ledger, config, steps_used/cap, research index, runtime
        counters) that a whole-dict replace silently destroyed (a budget
        raise is now the owner's ``manage_projects action=budget``, §4MQ). Pass
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
            if status_norm == ProjectStatus.ARCHIVED.value:
                # §4SX B0: an archived project is nobody's focus (every route
                # that archives — the tool, PATCH status — passes here)
                try:
                    self.clear_focus_for_project(project_id)
                except Exception:  # noqa: BLE001
                    logger.debug("focus clear on archive skipped", exc_info=True)
            if (status_norm == ProjectStatus.DONE.value
                    and (prev_status or "").upper() != ProjectStatus.DONE.value):
                # Constraint lifecycle (2026-08-01, req 56221fad post-mortem):
                # stored constraints bind work while the project is IN FLIGHT.
                # A 07-28 deliverable constraint ("Start with: What it means
                # to BE ghost") kept replaying into every request for 4 days
                # after the work closed — polluting new artifacts and driving
                # verifier refutes whose follow-up tasks reopened the project,
                # a self-feeding loop. DONE retires them; since §4UL any
                # reopen of the project re-arms what DONE retired (the change
                # is still the owner's project); the user restating one
                # re-arms it too (see the create-merge path in tools.projects).
                self.retire_constraints(project_id, reason="project completed")
                self._fire_project_done(project_id)
            elif ((prev_status or "").upper() == ProjectStatus.DONE.value and status_norm
                  and status_norm not in (ProjectStatus.DONE.value, ProjectStatus.ARCHIVED.value,
                                          ProjectStatus.RELEASED.value)):
                self.rearm_done_constraints(project_id, reason="project reopened")   # §4UL
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
                # §4UL: what the DONE close retired comes back on a reopen
                # (rearm_done_constraints); a retire for any other reason
                # (the owner's constraint_retire) stays retired
                _auto = _constraint_list(meta.get("constraints_done_retired"))
                _mk = {c.lower() for c in moving}
                if reason in ("project completed", "planned work finished"):     # §4UM: a finished plan too
                    _auto += [c for c in moving if c.lower() not in {a.lower() for a in _auto}]
                else:
                    _auto = [c for c in _auto if c.lower() not in _mk]
                if _auto:
                    meta["constraints_done_retired"] = _auto[-20:]
                else:
                    meta.pop("constraints_done_retired", None)
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

    def rearm_done_constraints(self, project_id: str, reason: str = "reopened") -> List[str]:
        """§4UL: a DONE project that reopens (a task added or revived, a
        write into its folder, a status change) gets back the constraints
        its DONE close retired — the owner's requirements still hold for the
        change. Constraints the owner retired stay retired. Never raises."""
        back: List[str] = []
        try:
            def _mut(meta):
                auto = _constraint_list(meta.get("constraints_done_retired"))
                if not auto:
                    return meta
                active = _constraint_list(meta.get("constraints"))
                keys = {c.lower() for c in active}
                for c in auto:
                    if c.lower() not in keys:
                        active.append(c)
                        keys.add(c.lower())
                        back.append(c)
                ak = {c.lower() for c in auto}
                meta["constraints"] = active
                meta["constraints_retired"] = [
                    c for c in _constraint_list(meta.get("constraints_retired"))
                    if c.lower() not in ak]
                meta.pop("constraints_done_retired", None)
                return meta
            if self._atomic_metadata_update(project_id, _mut) is None:
                return []
            if back:
                self.log_event(project_id, None, "constraints_rearmed",
                               {"constraints": back[:10], "reason": reason})
        except Exception:  # noqa: BLE001
            logger.debug("rearm_done_constraints skipped for %s", project_id, exc_info=True)
            return []
        return back

    def delete_project(self, project_id: str, hard: bool = False) -> bool:
        """Archive (soft) or delete (hard) a project.

        Soft-delete (``hard=False``) flips status to ARCHIVED so the
        project remains resumable — this is what ``action=archive`` uses.

        Hard delete (``hard=True``, what ``action=delete`` uses) removes
        the project COMPLETELY: the DB row plus all tasks/artifacts/events
        (FK ``ON DELETE CASCADE`` + ``PRAGMA foreign_keys=ON``, which
        includes the scratchpad-snapshot events), AND the project's
        workspace directory on disk (``<sandbox>/projects/<id>/``) so no
        files are left behind (§4QA-e: nor the kept probe transcripts under
        ``<memory_dir>/probe_transcripts/<id>/``). The workspace is removed only when it
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

        # §4QA-e: the kept probe transcripts (model answers) go with it — only
        # once the row is gone (r34: a failed delete kept the project and lost them)
        if deleted:
            try:
                from ..core.probe_transcripts import project_dir as _pt_dir
                _ptd = _pt_dir(getattr(self, "memory_dir", None), project_id)
                if _ptd is not None and _ptd.is_dir() and not _ptd.is_symlink():
                    shutil.rmtree(_ptd, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not remove kept probe transcripts for %s: %s", project_id, e)

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
                 position: Optional[int] = None,
                 kind: str = "") -> str:
        if not description or not description.strip():
            raise ValueError("description must be non-empty")
        from ..core.task_kind import normalize_kind
        kind = normalize_kind(kind)          # §4SU Phase 2 (ValueError on an unknown kind)
        from ..core.task_wait import fold_status, WAITING_STATUSES, default_wait
        _raw_status = status                         # §4SU Phase 3: READY → PENDING,
        status = fold_status(status) or "PENDING"    # PAUSED → NEEDS_USER (owner, paused)
        _won, _wcode = default_wait(_raw_status) if status in WAITING_STATUSES else ("", "")
        _wcols = self._has_wait_cols()
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
                "estimated_cost, actual_cost, depth, position, kind, "
                + ("wait_on, wait_code, " if _wcols else "") +
                "created_at, updated_at) "
                "VALUES (" + ",".join("?" * (23 if _wcols else 21)) + ")",
                (task_id, project_id, parent_id, description.strip(),
                 status.upper(), dependency_type.upper(),
                 json.dumps(alternatives or []), json.dumps(postconditions or []),
                 json.dumps(constraints or []), json.dumps(depends_on_canon),
                 "", "", 0, None, estimated_cost, 0.0, depth, position, kind,
                 *((_won, _wcode) if _wcols else ()), now, now),
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
            self.rearm_done_constraints(project_id, reason="task added")      # §4UL
        if str(status).upper() != "DONE" and self._planned_done_at(project_id):
            # §4UM: new planned work on a ready project — it is "working" again, and the next finish fires the
            # clean-up again; the finish retired the constraints, the new work re-arms them
            self._clear_planned_done(project_id)
            if not reopened_from:
                self.rearm_done_constraints(project_id, reason="task added")
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
                   "failure_reason", "actual_tool_used",
                   "estimated_cost", "actual_cost", "parent_id", "position", "kind",
                   "wait_on", "wait_code", "wait_why"}
        sets = []
        values: List[Any] = []
        if "status" in fields:
            # §4SU Phase 3: READY / task PAUSED fold into PENDING; leaving a
            # waiting status clears the wait fields (every writer); entering
            # one with no wait_on keeps the row's own, else the safe default
            from ..core.task_wait import fold_status, WAITING_STATUSES, default_wait, WAIT_ON, PAUSED_WAIT
            _raw_st = str(fields["status"]).strip().upper()
            _st = fold_status(fields["status"])
            fields = {**fields, "status": _st}
            if _raw_st == "PAUSED" and not fields.get("wait_on"):
                fields.update(wait_on=PAUSED_WAIT[0], wait_code=PAUSED_WAIT[1],
                              wait_why=fields.get("wait_why") or PAUSED_WAIT[2])
            if _st not in WAITING_STATUSES:
                fields.update(wait_on="", wait_code="", wait_why="")
            elif not fields.get("wait_on"):
                # r1 MAJOR: a status write with no wait_on gets the safe
                # default — never the row's earlier wait (a hold survived a
                # re-park, a BLOCKED kept a human-gate code)
                _on, _code = default_wait(_raw_st)
                fields.update(wait_on=_on, wait_code=fields.get("wait_code") or _code,
                              wait_why=fields.get("wait_why") or str(fields.get("result_summary") or "")[:300])
        if not self._has_wait_cols():
            # r1 MINOR: a failed column migration (a locked DB) must not
            # break every task write ("no such column: wait_on")
            fields = {k: v for k, v in fields.items() if not k.startswith("wait_")}
        if fields.get("wait_on"):
            from ..core.task_wait import WAIT_ON
            if fields["wait_on"] not in WAIT_ON:
                raise ValueError(f"unknown wait_on {fields['wait_on']!r}")
        if "kind" in fields:
            from ..core.task_kind import normalize_kind
            fields = {**fields, "kind": normalize_kind(fields["kind"])}   # §4SU Phase 2
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
                # Stamp WHICH turn closed this task (carried on the
                # `task_reopened` event if it is reopened). ALWAYS write
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
                # Reopening consumes the stamp: the `task_reopened` event
                # below carries it exactly once, and a later re-close
                # re-stamps (or blanks) it fresh.
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
            # §4SU Phase 0 (audit D): a RELEASED CLAIM (IN_PROGRESS → open:
            # a step stopped, deferred or crashed) is not a revival — it
            # un-paused a project the owner had just paused
            if (row and cur.rowcount > 0 and "status" in fields
                    and str(fields["status"]).upper()
                    in ("PENDING", "READY", "IN_PROGRESS")
                    and prev_task_status != "IN_PROGRESS"):
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
        if (updated and row and new_status is not None and new_status != "IN_PROGRESS"):
            # §4SU Phase 3 r3/r4: an open claim ends with any status write
            # other than the claim itself, whoever writes it — also when the
            # claim write had failed and the task never left its hold (an old
            # claim must never come back later with its old summary)
            try:
                if self.claimed_hold(row["project_id"], task_id) is not None:
                    self.log_event(row["project_id"], task_id, "hold_claim_ended", {})
            except Exception:  # noqa: BLE001
                logger.warning("hold claim end not logged for %s", task_id, exc_info=True)
        if updated and row:
            self.log_event(row["project_id"], task_id, "task_updated",
                           {"fields": [k for k in fields.keys()          # §4SU Phase 3: the wait
                                       if not k.startswith("wait_")]})   # columns follow status
            if reopened_from:
                self.log_event(row["project_id"], task_id, "project_reopened",
                               {"reason": f"task re-opened on a {reopened_from} project",
                                "from_status": reopened_from})
                self.rearm_done_constraints(row["project_id"], reason="task re-opened")   # §4UL
            if task_reopened_args:
                # §4PJ: the event stays (project history); the hook that fed
                # it to the calibration corpus was removed with the corpus.
                self.log_event(row["project_id"], task_id, "task_reopened",
                               {"from_status": task_reopened_args[2],
                                "closed_req_id": task_reopened_args[3]})
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
          * all tasks DONE                       → project stays/returns ACTIVE,
            "ready" (§4UM: only the owner closes a project) — the
            clean-up hooks fire once (`_planned_work_finished`)
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
                # §4UM (operator 11-13: "keep the projects open unless i ask the agent to set it done … an open
                # project should be live, it's just not final"): every planned task is DONE — the project
                # STAYS OPEN (ACTIVE, shown as "ready"); only the owner closes it (update status=DONE). What a
                # DONE transition used to fire (clean-up sweep, notify promise, constraint retirement) fires
                # once on this "planned work finished" transition instead.
                self._planned_work_finished(project_id, current, statuses)
                return
        else:
            if self._planned_done_at(project_id):
                # open work came back (a task revised to PENDING) — the next finish fires again
                self._clear_planned_done(project_id)
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
        # §4UM: the roll-up never sets DONE any more (only the owner closes a project) — a finished plan is
        # `_planned_work_finished` above, which fires the clean-up; FAILED / NEEDS_USER stay resumable and fire
        # nothing.

    # ------------------------------------------------------------ §4UM ready
    def _planned_done_at(self, project_id: str) -> float:
        proj = self.get_project(project_id) or {}
        try:
            return float((proj.get("metadata") or {}).get("planned_done_at") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _clear_planned_done(self, project_id: str) -> None:
        try:
            def _mut(meta):
                meta.pop("planned_done_at", None)
                return meta
            self._atomic_metadata_update(project_id, _mut)
        except Exception:  # noqa: BLE001
            logger.debug("planned_done clear skipped for %s", project_id, exc_info=True)

    def _planned_work_finished(self, project_id: str, current: str, statuses: List[str]) -> None:
        """§4UM: every task is DONE. The project goes (back) to ACTIVE — it was ACTIVE, or waited on the
        owner / was blocked / had failed work that is now done — and is "ready". The hooks a DONE transition
        fired run once per finish (``metadata.planned_done_at`` marks it; open work clears it)."""
        # guarded on the status the roll-up read (an ACTIVE one too): a concurrent transition (an ARCHIVE)
        # in between wins — nothing is marked and no hook fires on it
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (ProjectStatus.ACTIVE.value, _now(), project_id, current))
            conn.commit()
            if cur.rowcount == 0:
                return
        holder: Dict[str, bool] = {}

        def _mark(meta):
            if meta.get("planned_done_at"):
                holder["already"] = True
                return meta
            meta["planned_done_at"] = _now()
            if not meta.get("first_ready_at"):           # never cleared: the first plan was built (and by which request)
                meta["first_ready_at"] = meta["planned_done_at"]
                meta["first_ready_req"] = _change_rid()
            return meta

        if self._atomic_metadata_update(project_id, _mark) is None or holder.get("already"):
            return
        self.log_event(project_id, None, "project_auto_rollup",
                       {"new_status": "READY", "had_failures": False,
                        "note": "all planned tasks are done; the project stays open"})
        # Constraint lifecycle (2026-08-01): stored constraints bind the PLANNED work; a finished plan retires
        # them (they replayed into every later request), new tasks re-arm them (add_task).
        self.retire_constraints(project_id, reason="planned work finished")
        self._fire_project_done(project_id)

    def project_state(self, project_id: str, proj: Optional[Dict[str, Any]] = None,
                      tasks: Optional[List[Dict[str, Any]]] = None) -> str:
        """§4UM: how a project reads to the owner — ``working`` (planned work open or not planned yet),
        ``ready`` (ACTIVE, every task DONE: live, open for changes), ``done`` (closed by the owner), or the
        status itself in lower case (needs_user, failed, archived, released, paused, blocked)."""
        proj = proj if proj is not None else (self.get_project(project_id) or {})
        st = str(proj.get("status") or "").upper()
        if st == ProjectStatus.DONE.value and self.closed_by_rollup(project_id):
            return "ready"                    # §4UM review: a legacy auto-close is not the owner's close
        if st != ProjectStatus.ACTIVE.value:
            return st.lower() or "unknown"
        if tasks is None:
            tasks = self.list_tasks(project_id) or []
        sts = [str(t.get("status") or "").upper() for t in tasks]
        if sts and all(s == "DONE" for s in sts):
            return "ready"
        if not sts and (proj.get("metadata") or {}).get("planned_done_at"):
            return "ready"
        return "working"

    def closed_by_rollup(self, project_id: str) -> bool:
        """§4UM review: a DONE that came from the old task roll-up (before §4UM: ``project_auto_rollup``
        ``new_status: DONE``) and not from the owner — the latest status event decides. Such a project reads
        "ready" everywhere (no data migration); a write or a task reopens it like any DONE."""
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT type, payload_json FROM project_events WHERE project_id = ? AND type IN "
                    "('project_auto_rollup', 'project_updated', 'project_reopened', 'project_closed') "
                    "AND (type != 'project_updated' OR payload_json LIKE '%\"status\"%') "
                    "ORDER BY id DESC LIMIT 1", (_canon_id(project_id),)).fetchone()
            if not row or row["type"] != "project_auto_rollup":
                return False
            return str((json.loads(row["payload_json"] or "{}") or {}).get("new_status") or "").upper() == "DONE"
        except Exception:  # noqa: BLE001
            return False

    def planned_work_done(self, project_id: str) -> bool:
        """§4UM: ACTIVE with every planned task DONE (the old DONE roll-up)."""
        try:
            return self.project_state(project_id) == "ready"
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------ §4UM change log
    CHANGELOG_NAME = "PROJECT_CHANGES.md"

    @staticmethod
    def _row_to_change(row) -> Dict[str, Any]:
        d = dict(row)
        for k in ("added", "changed", "removed", "pushed"):
            try:
                d[k] = list(json.loads(d.pop(f"{k}_json") or "[]"))
            except Exception:  # noqa: BLE001
                d[k] = []
        try:
            d["meta"] = dict(json.loads(d.pop("meta_json") or "{}"))
        except Exception:  # noqa: BLE001
            d["meta"] = {}
        d["deployed"] = bool(d.get("deployed"))
        return d

    def add_change(self, project_id: str, *, kind: str = "change", ts: Optional[float] = None,
                   req_id: str = "", ask: str = "", added=(), changed=(), removed=(), summary: str = "",
                   deployed: bool = False, deployed_at: Optional[float] = None, url: str = "",
                   meta: Optional[Dict[str, Any]] = None, render: bool = True) -> Optional[Dict[str, Any]]:
        """Append one change-log entry (§4UM). ``kind='change'`` entries are numbered 1, 2, … per project;
        ``kind='deploy'`` entries are not. Re-renders the folder's PROJECT_CHANGES.md. Returns the entry, or None
        for an unknown project."""
        project_id = _canon_id(project_id)
        if not self.get_project(project_id):
            return None
        kind = "deploy" if str(kind) == "deploy" else "change"
        ts = float(ts or _now())

        def _paths(xs):
            return [str(x)[:300] for x in list(dict.fromkeys(xs or []))[:200]]

        with self._lock, self._connect() as conn:
            num = None
            if kind == "change":
                row = conn.execute("SELECT MAX(num) AS m FROM project_changes "
                                   "WHERE project_id = ? AND kind = 'change'", (project_id,)).fetchone()
                num = int((row["m"] if row else 0) or 0) + 1
            cur = conn.execute(
                "INSERT INTO project_changes(project_id, kind, num, ts, req_id, ask, added_json, changed_json, "
                "removed_json, summary, deployed, deployed_at, url, pushed_json, meta_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (project_id, kind, num, ts, str(req_id or "")[:64], " ".join(str(ask or "").split())[:300],
                 json.dumps(_paths(added)), json.dumps(_paths(changed)), json.dumps(_paths(removed)),
                 " ".join(str(summary or "").split())[:300], 1 if deployed else 0,
                 (float(deployed_at or ts) if deployed else None), str(url or "")[:300], "[]",
                 json.dumps(dict(meta or {}))[:4000]))
            conn.commit()
            cid = cur.lastrowid
            row = conn.execute("SELECT * FROM project_changes WHERE id = ?", (cid,)).fetchone()
        entry = self._row_to_change(row) if row else None
        if render:
            self.render_changelog(project_id)
        return entry

    def list_changes(self, project_id: str, limit: Optional[int] = None, offset: int = 0,
                     kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """The change log, newest first (§4UM)."""
        project_id = _canon_id(project_id)
        q = "SELECT * FROM project_changes WHERE project_id = ?"
        args: List[Any] = [project_id]
        if kind:
            q += " AND kind = ?"
            args.append(str(kind))
        q += " ORDER BY ts DESC, id DESC"
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            args += [max(0, int(limit)), max(0, int(offset or 0))]
        with self._lock, self._connect() as conn:
            rows = conn.execute(q, args).fetchall()
        return [self._row_to_change(r) for r in rows]

    def mark_changes_deployed(self, project_id: str, *, files=None, before_ts: Optional[float] = None,
                              at: Optional[float] = None, url: str = "", render: bool = False) -> List[int]:
        """Mark undeployed change entries live (§4UM). ``files=None`` (a whole-folder deploy): every one made
        at or before ``before_ts``. ``files`` (a `lab put` of single files): each entry accumulates the files
        pushed after it; it is live once every file it added or changed has been pushed (a pushed ``"dir/"``
        covers every file under it — a folder put or a subfolder deploy). Returns the
        numbers newly marked."""
        project_id = _canon_id(project_id)
        at = float(at or _now())
        pushed = None if files is None else {str(f) for f in files if f}
        done: List[int] = []
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM project_changes WHERE project_id = ? AND kind = 'change' "
                                "AND deployed = 0", (project_id,)).fetchall()
            for r in rows:
                e = self._row_to_change(r)
                if before_ts is not None and float(e["ts"]) > float(before_ts) + 1e-6:
                    continue
                if pushed is None:
                    conn.execute("UPDATE project_changes SET deployed = 1, deployed_at = ?, url = ? WHERE id = ?",
                                 (at, str(url or "")[:300], e["id"]))
                    done.append(int(e["num"] or 0))
                    continue
                need = set(e["added"]) | set(e["changed"])
                hit = {r for r in need if push_covers(pushed, r)}
                have = (set(e["pushed"]) & need) | hit
                if not hit:
                    continue
                if need and need <= have:
                    conn.execute("UPDATE project_changes SET deployed = 1, deployed_at = ?, url = ?, "
                                 "pushed_json = ? WHERE id = ?",
                                 (at, str(url or "")[:300], json.dumps(sorted(have)), e["id"]))
                    done.append(int(e["num"] or 0))
                else:
                    conn.execute("UPDATE project_changes SET pushed_json = ? WHERE id = ?",
                                 (json.dumps(sorted(have)), e["id"]))
            conn.commit()
        if render and done:
            self.render_changelog(project_id)
        return sorted(n for n in done if n)

    def record_deploy(self, project_id: str, *, url: str = "", files=None, how: str = "deploy",
                      lab_id: str = "", app: str = "", req_id: str = "", ts: Optional[float] = None,
                      before_ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """§4UM: the project's files went live (a lab deploy / keep / switch / put). Marks the changes it
        carries, appends a ``deploy`` entry and records where the project is live
        (``metadata.live``)."""
        project_id = _canon_id(project_id)
        if not self.get_project(project_id):
            return None
        ts = float(ts or _now())
        marked = self.mark_changes_deployed(project_id, files=files, at=ts, url=url,
                                            before_ts=ts if before_ts is None else before_ts)
        through = 0
        for e in self.list_changes(project_id, kind="change"):
            if e["deployed"]:
                through = max(through, int(e["num"] or 0))

        def _mut(meta):
            try:
                if float((meta.get("live") or {}).get("at") or 0) > ts:
                    return meta                   # §4UM review: an older (backfilled) deploy never overwrites a newer one
            except (TypeError, ValueError):
                pass
            meta["live"] = {"url": str(url or (meta.get("live") or {}).get("url") or "")[:300],
                            "lab_id": str(lab_id or (meta.get("live") or {}).get("lab_id") or "")[:40],
                            "app": str(app or (meta.get("live") or {}).get("app") or "")[:80],
                            "at": ts, "how": str(how)[:40], "through": through}
            return meta

        self._atomic_metadata_update(project_id, _mut)
        flist = None if files is None else [str(f) for f in files][:100]
        return self.add_change(project_id, kind="deploy", ts=ts, req_id=req_id, url=url, deployed=True,
                               deployed_at=ts, changed=flist or [],
                               meta={"how": how, "lab_id": lab_id, "app": app, "through": through,
                                     "marked": marked, "whole": files is None})

    def live_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """§4UM: where the project is live and which changes are not there yet: ``{url, lab_id, app, at, how,
        through, behind: [nums]}``, or None when it was never deployed."""
        proj = self.get_project(project_id) or {}
        live = (proj.get("metadata") or {}).get("live")
        if not isinstance(live, dict) or not live.get("at"):
            return None
        behind = sorted(int(e["num"] or 0) for e in self.list_changes(project_id, kind="change")
                        if not e["deployed"])
        return dict(live, behind=[n for n in behind if n])

    def change_files(self, project_id: str) -> set:
        """§4UM: every path an owner change added or changed (recorded or still pending in a running
        request) plus PROJECT_CHANGES.md once the log has an entry — the clean-up sweep keeps them (a change
        registers its files)."""
        out: set = set()
        try:
            if self.list_changes(project_id, limit=1):
                out.add(self.CHANGELOG_NAME)              # only once there is a log (an empty keep-set means more)
            for e in self.list_changes(project_id, kind="change"):
                out.update(e["added"])
                out.update(e["changed"])
                out.difference_update(e["removed"])
        except Exception:  # noqa: BLE001
            logger.debug("change files unreadable for %s", project_id, exc_info=True)
        out.update(pending_change_files(self, project_id))
        return out

    def render_changelog(self, project_id: str) -> Optional[str]:
        """Write the change log to ``PROJECT_CHANGES.md`` in the project folder (§4UM; newest first, the last
        CHANGES_FILE_MAX entries; atomic through a unique tmp name, no symlink followed, best effort). A file of
        that name that does not start with CHANGES_FILE_HEADER is the project's own: nothing is written.
        Returns the path written or None."""
        try:
            proj = self.get_project(project_id)
            if not proj:
                return None
            ws = (proj.get("workspace_dir") or "").strip() or self._default_workspace(_canon_id(project_id))
            if not ws:
                return None
            ws_path = _contained_workspace(self, ws)
            if ws_path is None or not ws_path.is_dir():
                return None
            target = ws_path / self.CHANGELOG_NAME
            if target.is_symlink():
                logger.warning("%s: %s is a link — the change log file is not written", project_id, target.name)
                return None
            if target.exists():
                try:
                    with open(target, "r", encoding="utf-8", errors="replace") as fh:
                        first = fh.readline().strip()
                except OSError:
                    first = ""
                if first != CHANGES_FILE_HEADER:
                    logger.warning("%s: %s is not the agent's (no header) — left as it is; the change log "
                                   "stays in the store", project_id, target.name)
                    return None
            lines = [CHANGES_FILE_HEADER, f"# Changes — {proj.get('title') or project_id}", "",
                     "Every change the owner asked for, and every time the project went live. Newest first.", ""]
            entries = self.list_changes(project_id, limit=CHANGES_FILE_MAX)
            for e in entries:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(e["ts"])))
                if e["kind"] == "deploy":
                    m = e.get("meta") or {}
                    what = "the whole folder" if m.get("whole") else (", ".join(e["changed"][:12]) or "files")
                    lines.append(f"## Live · {when} · {m.get('how') or 'deploy'}"
                                 + (f" → {e['url']}" if e.get("url") else ""))
                    lines.append(f"- Pushed: {what}"
                                 + (f" (includes change #{m['through']})" if m.get("through") else ""))
                    lines.append("")
                    continue
                state = "live" if e["deployed"] else "not deployed yet"
                lines.append(f"## #{e['num']} · {when} · {state}")
                if e.get("ask"):
                    lines.append(f"- Asked: {e['ask']}")
                if e["added"]:
                    lines.append(f"- Added: {', '.join(e['added'][:30])}")
                if e["changed"]:
                    lines.append(f"- Changed: {', '.join(e['changed'][:30])}")
                if e["removed"]:
                    lines.append(f"- Removed: {', '.join(e['removed'][:30])}")
                if (e.get("meta") or {}).get("folder_too_large"):
                    lines.append("- Files: not all listed (the folder was too large to compare)")
                if e.get("summary"):
                    lines.append(f"- Summary: {e['summary']}")
                lines.append("")
            if len(entries) >= CHANGES_FILE_MAX:
                lines.append("(older entries: manage_projects action=changes)")
            import uuid as _uuid
            tmp = ws_path / f".{self.CHANGELOG_NAME}.{_uuid.uuid4().hex[:12]}.tmp"
            try:
                _fs_write_nofollow(tmp, "\n".join(lines).rstrip() + "\n")
                os.replace(tmp, target)
            finally:
                try:
                    if tmp.exists() or tmp.is_symlink():
                        tmp.unlink()
                except OSError:
                    pass
            return str(target)
        except Exception:  # noqa: BLE001
            logger.debug("change log file render skipped for %s", project_id, exc_info=True)
            return None

    def reset_orphaned_in_progress(self, older_than_seconds: float = 900.0) -> int:
        """Reset stale IN_PROGRESS tasks back to PENDING (a claimed hold: back to its hold). Returns the count.

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
        # §4SU Phase 3 r3: a hold the owner's run had claimed goes back to the
        # hold FIRST (through update_task: its project rolls up too); a failed
        # restore leaves the row IN_PROGRESS — never PENDING, which an
        # unattended tick would run — and the next boot tries again
        restored = 0
        with self._lock, self._connect() as conn:
            stale = [dict(r) for r in conn.execute(
                "SELECT id, project_id, result_summary FROM tasks "
                "WHERE status = 'IN_PROGRESS' AND updated_at < ?", (cutoff,)).fetchall()]
        keep_ids = set()
        for r in stale:
            try:
                hold = self.claimed_hold(r["project_id"], r["id"])
            except Exception:  # noqa: BLE001
                logger.warning("orphaned task %s: claim unreadable — left IN_PROGRESS", r["id"], exc_info=True)
                keep_ids.add(r["id"])
                continue
            if not hold:
                continue
            keep_ids.add(r["id"])
            try:
                self.update_task(r["id"], status="NEEDS_USER", wait_on="owner_turn",
                                 wait_code=str(hold.get("wait_code") or "held"),
                                 wait_why=str(hold.get("wait_why") or ""),
                                 result_summary=str(hold.get("summary") or r.get("result_summary") or ""))
                self.log_event(r["project_id"], r["id"], "task_reset_orphaned",
                               {"from": "IN_PROGRESS", "to": "NEEDS_USER",
                                "older_than_seconds": float(older_than_seconds)})
                restored += 1
            except Exception:  # noqa: BLE001
                logger.warning("orphaned hold %s not restored — left IN_PROGRESS", r["id"], exc_info=True)
        reset: List[sqlite3.Row] = []
        with self._lock, self._connect() as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = [r for r in conn.execute(
                    "SELECT id, project_id FROM tasks "
                    "WHERE status = 'IN_PROGRESS' AND updated_at < ?",
                    (cutoff,),
                ).fetchall() if r["id"] not in keep_ids]
                if rows:
                    now = _now()
                    conn.executemany(
                        "UPDATE tasks SET status = 'PENDING', updated_at = ? "     # §4SU Phase 3: READY folded
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
                           {"from": "IN_PROGRESS", "to": "PENDING",
                            "older_than_seconds": float(older_than_seconds)})
        return len(reset) + restored

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
            # ⚠ RESOLVE THE PARENT, not just the final component.
            # `write_text_nofollow` refuses a symlink at the FILE, and its
            # docstring says intermediate directories are the caller's
            # problem — which both callers then ignored. The default
            # workspace is `<sandbox_root>/projects/<id>`, a directory the
            # container can `rm -rf` and replace with a symlink to anywhere
            # on the host; `mkdir(parents=True, exist_ok=True)` follows it
            # happily and the write lands outside. Demonstrated 2026-08-30:
            # PROJECT_MAP.md written into the agent's own source tree.
            ws_path = _contained_workspace(self, ws)
            if ws_path is None:
                return None
            ws_path.mkdir(parents=True, exist_ok=True)
            target = ws_path / "PROJECT_MAP.md"
            tmp = target.with_suffix(".md.tmp")
            # ⚠ O_NOFOLLOW: a fixed `.tmp` name inside a directory the
            # model can write to. `os.replace` below does not follow a
            # link, but this write did — a symlink planted at
            # `PROJECT_MAP.md.tmp` redirected model-authored text onto an
            # arbitrary host file. Demonstrated §4DX round 2.
            _fs_write_nofollow(tmp, text)
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
        # §4KW fourth review: mode bits alone were undone from inside the
        # sandbox (`chmod -R u+w . && rm -rf *` — container root holds
        # FOWNER); the user-immutable flag cannot be cleared from the
        # container (measured: rm, write, create, mv, rmtree, chmod all
        # refused). macOS/BSD only; elsewhere the mode bits remain.
        _imm = getattr(stat, "UF_IMMUTABLE", 0) if hasattr(os, "chflags") else 0
        try:
            # §4LP: never through a link — `ln -s /Users/…/file x` inside the
            # workspace made release chmod a HOST file read-only (and delete
            # make it writable); rglob also descends symlinked directories
            _root = ws.resolve()
            entries = [e for e in [ws, *ws.rglob("*")]
                       if not e.is_symlink() and e.resolve().is_relative_to(_root)]
            # (fifth review) a released app's RUNTIME STATE stays writable:
            # the Jiu Jitsu Calendar keeps its sqlite db next to app.py, and
            # sqlite must also create its journal in that folder. The db and
            # its folder keep their flags off; every other file — the code —
            # stays immutable (measured: inserts work, removing, renaming or
            # overwriting app.py and an immutable subfolder is refused).
            _state = {p for p in entries if not p.is_symlink() and p.is_file() and _is_runtime_state(p.name)}
            _exempt = _state | {p.parent for p in _state}
            for p in entries:
                try:
                    if _imm and not p.is_symlink():
                        _fl = p.lstat().st_flags
                        if (not readonly or p in _exempt) and _fl & _imm:
                            os.chflags(p, _fl & ~_imm, follow_symlinks=False)
                    mode = p.stat().st_mode
                    if readonly and p in _exempt:
                        # writable for real: the a-w bits alone already made
                        # sqlite refuse ("attempt to write a readonly
                        # database") — the calendar could not save since its
                        # release (measured in the sandbox, fifth review)
                        if not mode & stat.S_IWUSR:
                            p.chmod(mode | stat.S_IWUSR)
                            touched += 1
                        continue
                    if readonly:
                        new_mode = mode & ~(stat.S_IWUSR | stat.S_IWGRP
                                            | stat.S_IWOTH)
                    else:
                        new_mode = mode | stat.S_IWUSR
                    if new_mode != mode:
                        p.chmod(new_mode)
                        touched += 1
                    if _imm and readonly and not p.is_symlink():
                        _fl = p.lstat().st_flags
                        if not _fl & _imm:
                            os.chflags(p, _fl | _imm, follow_symlinks=False)
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
        # (fifth review) `Path("")` is the CURRENT directory: a project with
        # no workspace path made the process's own folder immutable,
        # recursively. Only an ABSOLUTE path inside the sandbox is touched.
        ws = Path(str((proj or {}).get("workspace_dir") or "").strip())
        if not ws.is_absolute() or not ws.is_dir() or not _inside_sandbox(self, ws):
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
            # ⚠ RESOLVE THE PARENT, not just the final component.
            # `write_text_nofollow` refuses a symlink at the FILE, and its
            # docstring says intermediate directories are the caller's
            # problem — which both callers then ignored. The default
            # workspace is `<sandbox_root>/projects/<id>`, a directory the
            # container can `rm -rf` and replace with a symlink to anywhere
            # on the host; `mkdir(parents=True, exist_ok=True)` follows it
            # happily and the write lands outside. Demonstrated 2026-08-30:
            # PROJECT_MAP.md written into the agent's own source tree.
            ws_path = _contained_workspace(self, ws)
            if ws_path is None:
                return None
            ws_path.mkdir(parents=True, exist_ok=True)
            target = ws_path / "RELEASE.md"
            tmp = target.with_suffix(".md.tmp")
            # ⚠ O_NOFOLLOW: a fixed `.tmp` name inside a directory the
            # model can write to. `os.replace` below does not follow a
            # link, but this write did — a symlink planted at
            # `RELEASE.md.tmp` redirected model-authored text onto an
            # arbitrary host file. Demonstrated §4DX round 2.
            _fs_write_nofollow(tmp, "\n".join(lines))
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
        # §4MM turn 16: an event written by a PROBE turn says so — the
        # "While you were away" digest reported a probe's fork of an owner
        # project ("Chess Coach v4 → FAILED") as the agent's own work
        # §4PF: …and so does every other writer (owner | autopilot | scheduled |
        # sub-agent | …) — the digest shows only what the agent did on its own
        try:
            from ..utils.logging import request_origin_label
            payload = dict(payload or {})
            payload.setdefault("origin", request_origin_label())
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass
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

    def claimed_hold(self, project_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """§4SU Phase 3 r2: the hold the owner's run claimed for this task and
        has not finished with (``hold_claimed`` newer than ``hold_claim_ended``)
        — its payload {wait_on, wait_code, wait_why, summary}, else None."""
        ev = self.latest_task_event(project_id, task_id, ("hold_claimed", "hold_claim_ended"))
        return ev["payload"] if ev and ev.get("type") == "hold_claimed" else None

    def release_claim(self, task_id: str, **fields) -> str:
        """§4SU Phase 3 r2 (review: the owner's run claimed a hold, was stopped
        or timed out, and the release wrote PENDING — the next unattended tick
        rebuilt it): give a claimed task back. A task claimed from a HOLD goes
        back to that hold; any other goes to PENDING. Only while it is still
        IN_PROGRESS. Returns the status written ("" when nothing was)."""
        row = self.get_task(task_id)
        if not row or str(row.get("status") or "").upper() != "IN_PROGRESS":
            return ""
        fields.pop("status", None)
        for _k in ("wait_on", "wait_code", "wait_why"):
            fields.pop(_k, None)
        hold = self.claimed_hold(row["project_id"], task_id)
        if hold:
            _summary = fields.pop("result_summary", None)
            self.update_task(task_id, status="NEEDS_USER", wait_on="owner_turn",
                             wait_code=str(hold.get("wait_code") or "held"),
                             wait_why=str(hold.get("wait_why") or ""),
                             result_summary=str(hold.get("summary") or _summary or row.get("result_summary") or ""),
                             **fields)                  # (update_task ends the claim)
            return "NEEDS_USER"
        self.update_task(task_id, status="PENDING", **fields)
        return "PENDING"

    def latest_task_event(self, project_id: str, task_id: str,
                          event_types) -> Optional[Dict[str, Any]]:
        """§4SU Phase 0: the newest event of one of ``event_types`` for ONE
        task — exact in SQL (a newest-N window lost a human-gate park on a
        busy project, and every later owner yes re-parked it)."""
        types = [str(t) for t in (event_types or [])]
        if not types:
            return None
        with self._lock, self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM project_events WHERE project_id = ? AND task_id = ? AND type IN ("
                + ",".join("?" * len(types)) + ") ORDER BY id DESC LIMIT 1",
                (_canon_id(project_id), _canon_id(task_id), *types)).fetchone()
        if r is None:
            return None
        d = dict(r)
        try:
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
        except Exception:  # noqa: BLE001
            d["payload"] = {}
        return d

    def list_events(self, project_id: str, limit: int = 50,
                    event_type: Optional[str] = None, since_id: Optional[int] = None,
                    before_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Newest first. ``since_id`` (§4PG R3-3): only events with id > it —
        the digest scans everything since its watermark instead of a window;
        ``before_id``: only events with id < it (paging)."""
        project_id = _canon_id(project_id)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, self.EVENTS_MAX_LIMIT))
        _since = (" AND id > ?" if since_id is not None else "") + (" AND id < ?" if before_id is not None else "")
        _sp = ((int(since_id),) if since_id is not None else ()) + ((int(before_id),) if before_id is not None else ())
        with self._lock, self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM project_events WHERE project_id = ? AND type = ?" + _since +
                    " ORDER BY id DESC LIMIT ?",
                    (project_id, event_type, *_sp, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM project_events WHERE project_id = ?" + _since +
                    " ORDER BY id DESC LIMIT ?",
                    (project_id, *_sp, int(limit)),
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
