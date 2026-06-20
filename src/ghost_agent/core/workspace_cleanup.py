"""Mark-and-sweep cleanup of a finished project's workspace.

When a project reaches DONE, the files it leaves behind in
``<sandbox>/projects/<id>/`` split into two kinds: *deliverables* the
user actually wanted, and *scratch* — screenshots, helper scripts, temp
files, ``__pycache__`` — that only mattered while the work was in
flight. This module keeps the deliverables and deletes the scratch.

The contract is deliberately simple: **registered ``file`` artifacts are
kept, everything else is deleted.** Anything the agent wants to survive
cleanup must be registered as a deliverable (see
``ProjectStore.register_file_artifact`` and the ``deliverables=[...]``
argument of the ``manage_projects`` ``task_update`` action); the research
subsystem already registers its briefs. Whatever is not registered is,
by definition, scratch.

The sweep is conservative about *where* it operates so a cleanup bug can
never escape the project's own scratch space:

  * It only ever walks ``<sandbox_root>/projects/<project_id>/``.
  * It does not follow symlinks out of that tree.
  * If it cannot read the keep-set it deletes **nothing** (fail-safe).
  * It never raises into the completion path — a cleanup failure must
    not strand a project mid-transition.

Wired onto ``ProjectStore.on_project_done`` in ``main.py`` so it fires
automatically the moment a project's tasks all settle to DONE. It fires
only on the *transition to DONE* — FAILED / NEEDS_USER projects stay
resumable, so their files are left untouched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


def _project_dir(store, project_id: str) -> Optional[Path]:
    """``<sandbox_root>/projects/<id>/`` — the single source of truth for a
    project's working directory (mirrors ``file_system.project_scoped_sandbox``).
    Returns ``None`` when the store has no sandbox root or the id is blank."""
    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid:
        return None
    return Path(root) / "projects" / pid


def _normalize_rel(payload: str, project_id: str) -> Optional[str]:
    """Reduce an artifact ``file`` payload to a clean project-relative POSIX
    path, or ``None`` if it cannot be trusted.

    Payloads are stored relative to the project dir (e.g. ``research/x.md``),
    but a few producers prepend the redundant ``projects/<id>/`` prefix the
    model sees in directory listings. Strip that, drop a leading ``./``, and
    reject any ``..`` traversal so a malformed payload can never widen the
    keep-set to a path outside the project dir.
    """
    if not payload:
        return None
    rel = str(payload).strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    pref = f"projects/{project_id}/"
    if rel.lower().startswith(pref.lower()):
        rel = rel[len(pref):]
    rel = rel.strip()
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def _keep_set(store, project_id: str) -> Optional[Set[str]]:
    """The set of project-relative paths to preserve: every registered
    ``file`` artifact. Returns ``None`` if the artifact list can't be read —
    the caller treats that as "delete nothing"."""
    pid = str(project_id or "").strip().lower()
    try:
        artifacts = store.list_artifacts(project_id=project_id)
    except Exception:
        logger.warning(
            "workspace cleanup: could not read artifacts for %s; keeping all files",
            project_id, exc_info=True,
        )
        return None
    keep: Set[str] = set()
    for art in artifacts or []:
        if (art.get("kind") or "") != "file":
            continue
        norm = _normalize_rel(art.get("payload") or "", pid)
        if norm:
            keep.add(norm)
    return keep


def sweep_project_workspace(store, project_id: str, *,
                            dry_run: bool = False) -> Dict[str, Any]:
    """Delete every file under the project's workspace that is not a
    registered ``file`` artifact, then prune the directories left empty.

    Returns a summary dict — ``{project_id, status, deleted, kept,
    dirs_removed, freed_bytes}`` — and never raises. ``status`` is ``"ok"``
    on a real sweep, or ``"skipped: <reason>"`` when there is nothing safe
    to do (no sandbox root, missing dir, unreadable keep-set). With
    ``dry_run=True`` it reports what it *would* delete without touching disk.
    """
    summary: Dict[str, Any] = {
        "project_id": project_id, "status": "ok",
        "deleted": [], "kept": [], "dirs_removed": [], "freed_bytes": 0,
    }

    proj_dir = _project_dir(store, project_id)
    if proj_dir is None:
        summary["status"] = "skipped: no sandbox_root"
        return summary
    try:
        root = proj_dir.resolve()
    except OSError:
        summary["status"] = "skipped: unresolvable dir"
        return summary
    if not root.is_dir():
        summary["status"] = "skipped: no project dir"
        return summary

    keep = _keep_set(store, project_id)
    if keep is None:
        summary["status"] = "skipped: artifact read failed"
        return summary

    deleted: List[str] = []
    kept: List[str] = []
    freed = 0

    # Pass 1 — delete unregistered files. topdown so we never descend into a
    # dir we're about to drop; followlinks=False so a symlink can't lead the
    # walk out of the project tree.
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue  # defensive: outside the tree, never touch it
            if rel in keep:
                kept.append(rel)
                continue
            try:
                # A symlink is scratch too: remove the link, never its target,
                # and never charge its target's size to freed bytes.
                size = 0 if fpath.is_symlink() else fpath.stat().st_size
            except OSError:
                size = 0
            if dry_run:
                deleted.append(rel)
                freed += size
                continue
            try:
                fpath.unlink()
                deleted.append(rel)
                freed += size
            except OSError as e:
                logger.debug("workspace cleanup: could not unlink %s: %s", fpath, e)

    # Pass 2 — prune directories left empty (bottom-up), keeping the project
    # dir itself so a fresh resume still has a workspace.
    dirs_removed: List[str] = []
    if not dry_run:
        for dirpath, _dirnames, _filenames in os.walk(root, topdown=False, followlinks=False):
            d = Path(dirpath)
            if d == root:
                continue
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                    dirs_removed.append(d.relative_to(root).as_posix())
                except OSError:
                    pass
            except OSError:
                pass

    summary.update(deleted=deleted, kept=kept,
                   dirs_removed=dirs_removed, freed_bytes=freed)
    if deleted or dirs_removed:
        verb = "would remove" if dry_run else "removed"
        pretty_log(
            "Cleanup",
            f"project {project_id}: {verb} {len(deleted)} scratch file(s), "
            f"kept {len(kept)} deliverable(s), freed {freed:,} bytes",
            icon=Icons.CUT,
        )
    return summary
