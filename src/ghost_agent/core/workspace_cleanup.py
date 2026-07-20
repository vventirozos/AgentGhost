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


def _escapes_projects_root(store, project_id: str, root: Path) -> bool:
    """Defense-in-depth containment check (2026-07-20 review, C1): the tool
    boundary validates project ids, but a ``..``-bearing id that slips
    through makes ``_project_dir`` resolve OUTSIDE the sandbox, and the walks
    below would then delete the parent tree. True ⇒ do not walk. Any doubt
    (including an unexpected exception) reads as an escape — this must never
    raise into the completion path."""
    try:
        pid = str(project_id or "").strip()
        if "/" in pid or "\\" in pid or ".." in Path(pid).parts:
            return True
        projects_root = (Path(store.sandbox_root) / "projects").resolve()
        # strictly inside: the projects root itself is never a project dir
        return root == projects_root or not root.is_relative_to(projects_root)
    except Exception:
        return True


def _normalize_rel(payload: str, project_id: str) -> Optional[str]:
    """Reduce an artifact ``file`` payload to a clean project-relative POSIX
    path, or ``None`` if it cannot be trusted.

    Payloads are stored relative to the project dir (e.g. ``research/x.md``),
    but a few producers prepend the redundant ``projects/<id>/`` prefix the
    model sees in directory listings, and some pass the absolute sandbox path
    (``/workspace/projects/<id>/x.png``). Both must collapse to the same
    project-relative key the sweep's walk produces, or the registered file
    reads as unprotected and gets deleted (2026-07-20 review, H9). The shared
    contract (``register_file_artifact`` stores the same form): project-
    relative POSIX, no leading ``/``, no ``workspace/`` segment, no
    ``projects/<id>/`` prefix, ``..`` rejected.
    """
    if not payload:
        return None
    rel = str(payload).strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    if rel.lower().startswith("workspace/"):
        rel = rel[len("workspace/"):]
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


def _is_kept_dotfile(name: str) -> bool:
    """True for well-known config/deliverable dotfiles, INCLUDING their
    format variants — real projects spell them ``.eslintrc.json``,
    ``.babelrc.js``, ``.env.local`` — which the exact-match set missed, so
    they were classified debris and deleted (2026-07-20 review, H8)."""
    return name in _KEEP_DOTFILES or any(
        name.startswith(k + ".") for k in _KEEP_DOTFILES)


def _in_git_tree(rel: str) -> bool:
    """True for ``.git`` itself or anything under a ``.git/`` directory
    (including a worktree/submodule ``.git`` pointer file). Version-control
    state is never cleanup's to delete — tidying an in-flight clone corrupts
    the whole repo (2026-07-20 review, H8) — so both the recurring tidy and
    the DONE sweep skip this subtree outright. ``.git`` stays in
    ``_SCRATCH_DIRS`` so the recovery passes don't register its internals
    as deliverables; this guard is what keeps the files on disk."""
    return ".git" in rel.lower().split("/")


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
    if name.startswith(".") and not _is_kept_dotfile(name):
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
    dirs_removed, freed_bytes, kept_referenced}`` (plus ``recovered`` when a
    recovery pass kept unregistered files) — and never raises. ``status`` is
    ``"ok"`` on a real sweep, or ``"skipped: <reason>"`` when there is
    nothing safe to do (no sandbox root, escaped/missing dir, unreadable
    keep-set). With
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
    if _escapes_projects_root(store, project_id, root):
        summary["status"] = "skipped: path escape"
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

    # Pass 0 — referenced-media protection (2026-07-20): an unregistered
    # media file that a kept source file points at (sprite sheet, texture,
    # favicon) is an asset of the build, not a stray screenshot — deleting
    # it breaks the deliverable the sweep just protected. Same scan the
    # recurring tidy runs; symlinks stay deletable (a symlink is scratch).
    media_candidates: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                if fpath.is_symlink():
                    continue
                rel = fpath.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            if rel in keep or _is_debris(rel):
                continue
            name = rel.split("/")[-1].lower()
            dot = name.rfind(".")
            if dot > 0 and name[dot:] in _MEDIA_SUFFIXES:
                media_candidates.append(rel)
    referenced = _referenced_media(root, media_candidates)

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
            if _in_git_tree(rel):
                continue  # version-control state survives even the DONE sweep
            if rel in keep or rel in referenced:
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
                drel = d.relative_to(root).as_posix()
            except ValueError:
                continue
            if _in_git_tree(drel):
                continue  # git keeps legitimately-empty dirs (refs/tags, …)
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                    dirs_removed.append(drel)
                except OSError:
                    pass
            except OSError:
                pass

    summary.update(deleted=deleted, kept=kept,
                   dirs_removed=dirs_removed, freed_bytes=freed,
                   kept_referenced=sorted(referenced))
    if deleted or dirs_removed:
        verb = "would remove" if dry_run else "removed"
        pretty_log(
            "Cleanup",
            f"project {project_id}: {verb} {len(deleted)} scratch file(s), "
            f"kept {len(kept)} deliverable(s), freed {freed:,} bytes",
            icon=Icons.CUT,
        )
    return summary


# ── recurring tidy (2026-07-18) ─────────────────────────────────────────
#
# The DONE sweep above fires exactly once, on the transition. But work on
# a project doesn't stop at DONE: verification and post-completion
# debugging keep producing screenshots and helper scripts, and those land
# AFTER the sweep — so they accumulate until the operator deletes them by
# hand. Live case: the game project rolled DONE at 21:41 and had SIX
# unswept screenshots by the next morning (debug_start.png at 21:55
# through screenshot.png at 08:13). The tidy pass below is the recurring
# counterpart: safe to run on ANY project in ANY status, repeatedly.
#
# It is deliberately MUCH narrower than the DONE sweep. It deletes only:
#   * categorical debris (`_is_debris` — caches, browser scaffolding,
#     swap/backup files) older than the age gate — the tidy runs on ACTIVE
#     projects, so even debris-shaped files get their grace period (a fresh
#     `.hidden_scratch` may be an in-flight tool's state), and
#   * unregistered MEDIA files (screenshot-shaped: .png/.jpg/…)
#     that are (a) older than the age gate, (b) not in the keep-set,
#     (c) not referenced by any of the project's source files — a sprite
#     sheet an index.html points at is an asset, not a screenshot.
# Source/document files are NEVER deleted here regardless of
# registration — on a non-terminal project, today's unregistered helper
# script may be tomorrow's deliverable; the DONE sweep is the place where
# unregistered scratch scripts get judged. The `.git/` subtree is never
# touched at all (`_in_git_tree`).

#: media suffixes the tidy treats as screenshot-shaped scratch candidates.
_MEDIA_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
}

#: how old (hours) an unregistered media file must be before the idle
#: tidy will delete it. In-flight verification screenshots stay put.
TIDY_MIN_AGE_HOURS = 24.0

#: bound on how much source text the referenced-media check will read
#: per file — plenty for any real index.html/css, keeps the scan cheap.
_REFERENCE_SCAN_MAX_BYTES = 512 * 1024


def _referenced_media(root: Path, media_rels: List[str]) -> Set[str]:
    """Return the subset of ``media_rels`` whose BASENAME appears in any
    source-like file of the project — i.e. media that is an asset the
    build points at (sprite sheet, texture, favicon), not a stray
    screenshot. Basename matching is deliberately loose: a false KEEP
    costs a few kilobytes, a false DELETE breaks the build."""
    if not media_rels:
        return set()
    basenames = {rel: rel.split("/")[-1] for rel in media_rels}
    hit: Set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue
            if not _is_source_like(rel):
                continue
            try:
                if fpath.is_symlink() or fpath.stat().st_size > _REFERENCE_SCAN_MAX_BYTES:
                    continue
                text = fpath.read_text(errors="replace")
            except OSError:
                continue
            for mrel, mname in basenames.items():
                if mrel not in hit and mname in text:
                    hit.add(mrel)
        if len(hit) == len(basenames):
            break
    return hit


def tidy_project_workspace(store, project_id: str, *,
                           min_age_hours: float = TIDY_MIN_AGE_HOURS,
                           dry_run: bool = False) -> Dict[str, Any]:
    """Recurring, status-agnostic debris tidy for one project workspace.

    Deletes categorical debris and unregistered, unreferenced media files —
    both only when older than ``min_age_hours`` (see module note above);
    never touches source/document files, the ``.git/`` subtree, or anything
    in the keep-set.
    Logs one ``workspace_tidy`` project event when something was removed.
    Returns the same summary shape as ``sweep_project_workspace``;
    never raises."""
    import time as _time

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
    if _escapes_projects_root(store, project_id, root):
        summary["status"] = "skipped: path escape"
        return summary
    if not root.is_dir():
        summary["status"] = "skipped: no project dir"
        return summary
    keep = _keep_set(store, project_id)
    if keep is None:
        # fail-safe, same contract as the sweep: unreadable keep-set ⇒
        # delete nothing.
        summary["status"] = "skipped: artifact read failed"
        return summary

    age_cutoff = _time.time() - max(0.0, float(min_age_hours)) * 3600.0

    # Collect candidates first so the referenced-media scan runs once.
    debris: List[str] = []
    media_candidates: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel in keep:
                continue
            if _in_git_tree(rel):
                continue  # version-control state is never the tidy's to delete
            if _is_debris(rel):
                # Same grace period media gets: an ACTIVE project's fresh
                # debris (a tool's live dotfile state, a log being written)
                # must not vanish under the build; symlinks stay deletable.
                try:
                    if fpath.is_symlink() or fpath.stat().st_mtime <= age_cutoff:
                        debris.append(rel)
                except OSError:
                    pass
                continue
            if _is_source_like(rel):
                continue  # never the tidy's business
            name = rel.split("/")[-1].lower()
            dot = name.rfind(".")
            if dot > 0 and name[dot:] in _MEDIA_SUFFIXES:
                try:
                    if fpath.is_symlink() or fpath.stat().st_mtime <= age_cutoff:
                        media_candidates.append(rel)
                except OSError:
                    continue

    referenced = _referenced_media(root, media_candidates)
    to_delete = debris + [m for m in media_candidates if m not in referenced]

    deleted: List[str] = []
    freed = 0
    for rel in to_delete:
        fpath = root / rel
        try:
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
            logger.debug("workspace tidy: could not unlink %s: %s", fpath, e)

    # Prune dirs left empty (never the project root), mirroring the sweep.
    dirs_removed: List[str] = []
    if not dry_run and deleted:
        for dirpath, _dirnames, _filenames in os.walk(root, topdown=False, followlinks=False):
            d = Path(dirpath)
            if d == root:
                continue
            try:
                drel = d.relative_to(root).as_posix()
            except ValueError:
                continue
            if _in_git_tree(drel):
                continue  # git keeps legitimately-empty dirs (refs/tags, …)
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                    dirs_removed.append(drel)
                except OSError:
                    pass
            except OSError:
                pass

    summary.update(deleted=deleted, dirs_removed=dirs_removed,
                   freed_bytes=freed,
                   kept_referenced=sorted(referenced))
    if deleted:
        verb = "would remove" if dry_run else "removed"
        pretty_log(
            "Workspace Tidy",
            f"project {project_id}: {verb} {len(deleted)} debris file(s) "
            f"({freed:,} bytes)"
            + (f", kept {len(referenced)} referenced asset(s)" if referenced else ""),
            icon=Icons.CUT,
        )
        if not dry_run:
            try:
                store.log_event(project_id, None, "workspace_tidy", {
                    "deleted": deleted[:20],
                    "deleted_count": len(deleted),
                    "freed_bytes": freed,
                })
            except Exception:
                logger.debug("workspace_tidy event skipped", exc_info=True)
    return summary
