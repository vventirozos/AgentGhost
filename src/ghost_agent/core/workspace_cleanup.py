"""Mark-and-sweep cleanup of a finished project's workspace.

When a project reaches DONE, the files it leaves behind in
``<sandbox>/projects/<id>/`` split into two kinds: *deliverables* the
user actually wanted, and *scratch* — screenshots, helper scripts, temp
files, ``__pycache__`` — that only mattered while the work was in
flight. This module keeps the deliverables and deletes the scratch.

The contract: **when deliverables are registered, registered ``file``
artifacts are kept and everything else is deleted.** Anything the agent
wants to survive cleanup should be registered as a deliverable (see
``ProjectStore.register_file_artifact`` and the ``deliverables=[...]``
argument of the ``manage_projects`` ``task_update`` action); the research
subsystem already registers its briefs.

But registration depends on the agent remembering to do it, and it doesn't
always — a one-turn single-file build wrote ``index.html``, marked its task
DONE without ``deliverables=[...]``, and the old "nothing registered ⇒
everything is scratch" rule wiped the project's only file (observed live).
So an **empty keep-set is treated as missing registration, not as
permission to delete the workspace**: the sweep recovers the deliverables by
keeping (and registering) every non-debris file and deletes only categorical
debris — bytecode caches, browser scaffolding, swap/backup/dot files (see
``_is_debris``). A workspace holding nothing but debris still nets an empty
keep-set, so a lone ``__pycache__`` is still swept.

A **partial keep-set is just as untrustworthy** (chess incident,
2026-07-02): the agent registered one of the four files it built, the
old rule took that sparse set at face value, and the sweep deleted the
game's ``index.html`` and two of its three JS modules at the moment of
completion. Registration is evidence of what IS a deliverable, never
proof of what ISN'T. So when the keep-set is non-empty, the sweep still
recovers (keeps + registers) every unregistered **source/document** file
(``_is_source_like`` — code, markup, docs, config, data-as-text), and
deletes only debris and unregistered media/binary scratch (screenshots
being the classic case). Losing a stray kept file is cheap; losing the
build is not.

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


# ── no-registration recovery ────────────────────────────────────────────
# Transient debris a build leaves behind that is NEVER a deliverable. Used
# ONLY when a project reaches DONE with an EMPTY keep-set (the agent never
# registered anything). There the old policy — "nothing registered ⇒
# everything is scratch ⇒ delete it all" — destroyed the deliverable: a
# one-turn single-file build wrote index.html, marked the task DONE without
# `deliverables=[...]`, and the sweep wiped the project's only file (observed
# live). In that case we delete just this debris and KEEP + register
# everything else, recovering the deliverable set the agent forgot to mark.
# Deliberately conservative: anything not obviously debris is kept, because
# losing a real deliverable is far worse than leaving a stray scratch file.
_SCRATCH_DIRS = {
    "__pycache__", ".browser_profile", ".pytest_cache", ".ipynb_checkpoints",
    ".cache", ".git", ".mypy_cache", ".ruff_cache",
}
_SCRATCH_NAMES = {".ds_store", "thumbs.db", ".browser_runner.py"}
_SCRATCH_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp", ".swo"}
_SCRATCH_PREFIXES = ("temp_", "tmp_", "scratch_", "debug_")

# Dotfiles that are legitimate project deliverables/config — NOT debris.
_KEEP_DOTFILES = frozenset({
    ".htaccess", ".env", ".env.example", ".gitignore", ".gitattributes",
    ".dockerignore", ".editorconfig", ".npmrc", ".nvmrc", ".babelrc",
    ".eslintrc", ".prettierrc", ".flaskenv",
})


def _is_debris(rel: str) -> bool:
    """True for a project-relative path that is categorically transient —
    bytecode caches, browser scaffolding, editor swap/backup files, dotfiles.
    Conservative on purpose (see module note): everything else is a candidate
    deliverable and is kept by the no-registration recovery path."""
    parts = rel.split("/")
    if any(p in _SCRATCH_DIRS for p in parts[:-1]):
        return True
    name = parts[-1].lower()
    if name in _SCRATCH_NAMES:
        return True
    # Dotfiles are debris EXCEPT well-known config/deliverable dotfiles — a web
    # build's `.htaccess`, a `.env`, or a `.gitignore` are real deliverables and
    # were being deleted by the no-registration recovery path.
    if name.startswith(".") and name not in _KEEP_DOTFILES:
        return True
    if name.startswith(_SCRATCH_PREFIXES):
        return True
    dot = name.rfind(".")
    suffix = name[dot:] if dot > 0 else ""
    return suffix in _SCRATCH_SUFFIXES


# Files that read as SOURCE or DOCUMENT — the shape of a build's actual
# substance. Used by the partial-keep-set recovery: an unregistered file of
# one of these kinds is treated as a forgotten deliverable, not as scratch.
# Media/binary files (e.g. .png screenshots) deliberately stay OUT of this
# set: they are the sweep's primary cleanup target, and a media deliverable
# is protected by registering it (or by the empty-keep-set recovery, which
# keeps everything non-debris when there is no registration signal at all).
_SOURCE_SUFFIXES = {
    ".html", ".htm", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".css", ".scss", ".py", ".rb", ".go", ".rs", ".java", ".c", ".h",
    ".cpp", ".hpp", ".sh", ".bash", ".zsh", ".sql", ".md", ".rst",
    ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv", ".xml", ".svg",
}
_SOURCE_NAMES = {"makefile", "dockerfile", "readme", "license", "notice"}


def _is_source_like(rel: str) -> bool:
    """True when a project-relative path looks like source/documentation —
    a candidate deliverable the agent plausibly forgot to register."""
    name = rel.split("/")[-1].lower()
    if name in _SOURCE_NAMES:
        return True
    dot = name.rfind(".")
    return dot > 0 and name[dot:] in _SOURCE_SUFFIXES


def _any_task_id(store, project_id: str) -> Optional[str]:
    """The most-recent task id for the project, to hang recovered-deliverable
    artifacts on. ``register_file_artifact`` needs a task; recovery is
    best-effort, so ``None`` (no tasks) just means we keep without persisting
    a row — a later sweep re-derives the same keep-set."""
    try:
        tasks = store.list_tasks(project_id) or []
    except Exception:
        return None
    if not tasks:
        return None
    tasks = sorted(tasks, key=lambda t: t.get("updated_at") or t.get("created_at") or "")
    return tasks[-1].get("id")


def _recover_deliverables(store, project_id: str, root: Path,
                          *, dry_run: bool = False) -> Set[str]:
    """Keep-set for a project that registered NOTHING: every non-debris,
    non-symlink file under the workspace. Registers what it finds (best-effort,
    skipped on dry_run) so the record becomes truthful and the files survive
    any future sweep. Symlinks are never recovered — a symlink is scratch and
    could point outside the tree, so it is left for the normal pass to delete."""
    keep: Set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                if fpath.is_symlink():
                    continue
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue
            if not _is_debris(rel):
                keep.add(rel)
    if keep:
        tid = None if dry_run else _any_task_id(store, project_id)
        if tid:
            for rel in sorted(keep):
                try:
                    store.register_file_artifact(tid, rel)
                except Exception:
                    logger.debug("recovery registration skipped: %s", rel, exc_info=True)
        pretty_log(
            "Cleanup",
            f"project {project_id}: no deliverables were registered — recovered "
            f"{len(keep)} file(s) from the workspace and kept them (only debris "
            f"deleted). The build path should register deliverables explicitly.",
            icon=Icons.WARN, level="WARNING",
        )
    return keep


def _recover_unregistered_sources(store, project_id: str, root: Path,
                                  keep: Set[str],
                                  *, dry_run: bool = False) -> Set[str]:
    """Partial-registration recovery (chess incident, 2026-07-02): the set of
    unregistered, non-debris, non-symlink SOURCE files under the workspace.

    A non-empty keep-set proves the agent engaged with registration, not
    that it finished the job — one registered file out of four built ones
    let the old sweep delete the build at the moment of completion.
    Source/document files are what builds are made of, so an unregistered
    one is treated as a forgotten deliverable: kept, and registered
    (best-effort, skipped on dry_run) so the record becomes truthful and a
    later sweep keeps it through the normal keep-set. Unregistered media/
    binary files stay deletable — they are the scratch (screenshots,
    renders) this sweep exists to remove."""
    found: Set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                if fpath.is_symlink():
                    continue
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel in keep or _is_debris(rel) or not _is_source_like(rel):
                continue
            found.add(rel)
    if found:
        tid = None if dry_run else _any_task_id(store, project_id)
        if tid:
            for rel in sorted(found):
                try:
                    store.register_file_artifact(tid, rel)
                except Exception:
                    logger.debug("partial-recovery registration skipped: %s",
                                 rel, exc_info=True)
        pretty_log(
            "Cleanup",
            f"project {project_id}: keep-set was PARTIAL — recovered "
            f"{len(found)} unregistered source file(s) and kept them "
            f"(registered as deliverables). Close tasks with "
            f"deliverables=[...] so recovery isn't needed.",
            icon=Icons.WARN, level="WARNING",
        )
    return found


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
    """Sweep a finished project's workspace: keep registered ``file``
    artifacts plus recovered deliverables (see the module note — every
    non-debris file when nothing was registered; every unregistered source
    file when registration was partial), delete the rest, then prune the
    directories left empty.

    Returns a summary dict — ``{project_id, status, deleted, kept,
    dirs_removed, freed_bytes}`` (plus ``recovered`` when a recovery pass
    kept unregistered files) — and never raises. ``status`` is ``"ok"``
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

    # An EMPTY keep-set on a finished project means the agent registered no
    # deliverables — NOT that the whole workspace is scratch. Recover the
    # deliverable set (keep + register every non-debris file) so the sweep
    # removes only true debris instead of wiping the build. A workspace that
    # holds nothing but debris yields an empty keep-set here too, and the
    # normal pass below then deletes that debris (e.g. a lone __pycache__).
    if not keep:
        keep = _recover_deliverables(store, project_id, root, dry_run=dry_run)
        summary["recovered"] = sorted(keep)
    else:
        # A PARTIAL keep-set is evidence of what IS a deliverable, never
        # proof of what isn't: recover unregistered source files instead of
        # deleting the build (media/binary scratch stays deletable).
        recovered = _recover_unregistered_sources(
            store, project_id, root, keep, dry_run=dry_run)
        if recovered:
            keep = keep | recovered
            summary["recovered"] = sorted(recovered)

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
