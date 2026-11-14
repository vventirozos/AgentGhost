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
build is not. Rendered documents (``.pdf``, office files, e-books) are
recovered too, and so is any file the project's goal or a task's
description/result names (§4SS — a final ``report.pdf`` was swept 40 s
after the task that wrote it closed without ``deliverables=[...]``).

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
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


#: §4ST: refused model-run write-ups (core.experiment_gate.QUARANTINE_DIR) — kept on disk,
#: never registered as deliverables by a recovery
_QUARANTINE_DIR = "simulated"

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
    # ⚠ COLLAPSE REPEATED SEPARATORS, THEN STRIP EACH PREFIX EXACTLY ONCE
    # (2026-08-11, §4AT-D). The measured divergence was narrow: a DOUBLED
    # SLASH (`/workspace//projects/<id>/x.png`) left `projects/<id>/` in place
    # here while the store's `_normalize_rel_path` removed it — one file, two
    # keys, so the registration stopped protecting and the sweep deleted it.
    #
    # ⚠⚠ A FIXED-POINT LOOP WAS TRIED FIRST AND WAS WORSE. Stripping
    # repeatedly also eats a LEGITIMATE top-level `workspace/` directory
    # inside a project, so `projects/<id>/workspace/hero.png` keyed as
    # `hero.png` and the registered deliverable was deleted — a NEW loss path
    # with a wider blast radius than the bug being fixed. Collapsing
    # separators addresses the actual divergence and nothing else; a repeated
    # real directory name is DATA, not a prefix to peel.
    while "//" in rel:
        rel = rel.replace("//", "/")
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
    # CI/tooling config — deliverables an operator would be startled to lose.
    # `.gitlab-ci.yml` was measured being deleted from an ACTIVE project by the
    # idle tidy (2026-08-11, §4AT-D).
    ".gitlab-ci.yml", ".travis.yml", ".pre-commit-config.yaml",
    ".ruff.toml", ".python-version", ".tool-versions", ".nojekyll",
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

# Binary/data DELIVERABLES the agent commonly produces but forgets to
# register: app databases, trained model weights, datasets. Distinct from
# media SCRATCH (.png screenshots, .mp4 renders) which the sweep exists to
# remove. On a PARTIAL keep-set these were deleted at the moment of
# completion because the source-only recovery didn't cover them — a real
# completion-time data-loss path (a coding project's seeded data.db /
# model.pt vanished when app.py alone was registered).
_DATA_DELIVERABLE_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".parquet", ".feather", ".arrow",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pb", ".h5", ".hdf5",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".model",
}


def _is_data_deliverable(rel: str) -> bool:
    """True for a binary/data deliverable (database / model weights /
    dataset) — a plausibly-forgotten deliverable, NOT media scratch."""
    name = rel.split("/")[-1].lower()
    return any(name.endswith(s) for s in _DATA_DELIVERABLE_SUFFIXES)


# §4SS (req b45ec99f, 2026-11-01): office/document DELIVERABLES. The project's
# last task compiled `report.pdf` (+ a copy `report_pdf.pdf`), closed with a
# `result` but no `deliverables=[...]`, and the DONE sweep deleted both 40 s
# later — `.pdf` was neither source-like nor a data deliverable, so the
# partial-keep-set recovery treated the goal's own "final PDF report" as
# scratch. A rendered document is what a project hands over, never debris.
_DOCUMENT_DELIVERABLE_SUFFIXES = {
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".epub",
    ".pptx", ".ppt", ".odp", ".key",
    ".xlsx", ".xls", ".ods", ".pages", ".numbers",
}


def _is_document_deliverable(rel: str) -> bool:
    """True for a rendered document (PDF / office / e-book) — a deliverable
    the agent plausibly forgot to register, NOT media scratch."""
    name = rel.split("/")[-1].lower()
    dot = name.rfind(".")
    return dot > 0 and name[dot:] in _DOCUMENT_DELIVERABLE_SUFFIXES


def _project_texts(store, project_id: str) -> List[str]:
    """The project's goal plus every task's description and result summary —
    where the agent SAYS which files it made. Best-effort: [] on any read
    failure (the other protections still apply)."""
    texts: List[str] = []
    try:
        proj = store.get_project(project_id) or {}
        texts.append(str(proj.get("goal") or ""))
    except Exception:
        logger.debug("named-file scan: goal unreadable", exc_info=True)
    try:
        for t in store.list_tasks(project_id) or []:
            texts.append(str(t.get("description") or ""))
            texts.append(str(t.get("result_summary") or ""))
    except Exception:
        logger.debug("named-file scan: tasks unreadable", exc_info=True)
    return texts


#: one path-shaped run of text: `results/plot.png`, `/workspace/…/report.pdf`
_PATH_RUN_RE = re.compile(r"[\w.\-/]+")


def _named_tokens(texts) -> tuple:
    """(names, paths) the texts mention, lowercased, extracted ONCE.

    A run of path characters (letters, digits, ``_ . - /``) is split on
    ``/``; trailing sentence dots are dropped ("Wrote report.pdf."). Every
    segment is a NAME, and every trailing chain of segments is a PATH
    (`/workspace/projects/x/results/plot.png` → `plot.png`,
    `results/plot.png`, …). A longer token never names a shorter file:
    "old_report.pdf", "final.report.pdf", "report.pdf.bak" and
    "report.pdf-old" are their own tokens. §4SS review: a regex per file over
    the whole text took 9 s for a 10k-file tree and blocked the event loop."""
    names: Set[str] = set()
    paths: Set[str] = set()
    for text in texts:
        low = _nfc(str(text or "").lower()).replace("\\", "/")     # r2: Windows paths
        for m in _PATH_RUN_RE.finditer(low):
            segs = [g.rstrip(".") for g in m.group(0).split("/")]
            segs = [g for g in segs if g]
            names.update(segs)
            # the last _PATH_CHAIN_MAX segments only: r2 — every chain of a
            # 10k-segment run was 150 MB of strings
            for k in range(max(0, len(segs) - _PATH_CHAIN_MAX), len(segs)):
                paths.add("/".join(segs[k:]))
    return names, paths


#: deepest project-relative path a mention can name (§4SS r2)
_PATH_CHAIN_MAX = 32


def _nfc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", s)


def _named_by_project(store, project_id: str, root: Path) -> Set[str]:
    """§4SS: workspace files the project's own words name — a task result
    "Wrote report.pdf (81019 bytes)" or a description "→ results/plot.png".

    A file with an extension is named when the goal, a task description or a
    task result mentions it as a whole token (case-insensitive,
    `_named_tokens`): its basename is enough, EXCEPT for media — a nested
    screenshot-shaped file must be named by its path (`results/plot.png`),
    because "checked screenshot.png" in a verify step would otherwise keep
    every screenshot in the tree, the tidy's main target. Names are matched
    against files that EXIST, so text can only keep, never delete. Debris and
    symlinks are never kept this way."""
    names, paths = _named_tokens(_project_texts(store, project_id))
    found: Set[str] = set()
    if not names:
        return found
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
            # §4SU Phase 0: a .log the project NAMES ("wrote results/train.log")
            # is its output, not scratch
            if _is_debris(rel) and not rel.lower().endswith(".log"):
                continue
            low = _nfc(rel.lower())
            name = low.rsplit("/", 1)[-1]
            dot = name.rfind(".")
            if dot <= 0:
                continue                  # "data", "out": ordinary words, not names
            if name[dot:] in _MEDIA_SUFFIXES:
                if low in paths:
                    found.add(rel)
            elif name in names:
                found.add(rel)
    return found


def _tidy_is_debris(rel: str) -> bool:
    """Debris rules for the ACTIVE-project tidy.

    Identical to `_is_debris` EXCEPT that a scratch NAME PREFIX alone does not
    condemn a source file. Everything else — scratch dirs, scaffolding names,
    non-kept dotfiles, scratch suffixes — still outranks source-ness.

    ⚠ THE SCOPE TOOK TWO CORRECTIONS. Inverting the precedence inside
    `_is_debris` broke the DONE sweep (sweeping a finished project's
    `temp_probe.py` is deliberate and pinned). Then a blanket
    `if _is_source_like: continue` in the tidy kept `.browser_runner.py` —
    browser scaffolding the tidy exists to remove, which is source-shaped only
    because it ends in `.py`. The exemption belongs to ONE rule, not to the
    classifier and not to the caller.
    """
    if not _is_debris(rel):
        return False
    if not _is_source_like(rel):
        return True
    parts = rel.split("/")
    name = parts[-1].lower()
    if any(p in _SCRATCH_DIRS for p in parts[:-1]):
        return True
    if name in _SCRATCH_NAMES:
        return True
    if name.startswith(".") and not _is_kept_dotfile(name):
        return True
    dot = name.rfind(".")
    if dot > 0 and name[dot:] in _SCRATCH_SUFFIXES:
        return True
    # Only the scratch-name prefix fired, and the file is source: on a project
    # still being WORKED ON, a name is a guess about intent and source-ness is
    # a fact. `debug_utils.py` may well be imported by the build.
    return False


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
                if rel.split("/", 1)[0] == _QUARANTINE_DIR:
                    continue        # §4ST: kept, never promoted to a deliverable
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
    unregistered, non-debris, non-symlink SOURCE files — and binary/data
    deliverables (databases, model weights, datasets) — under the workspace.

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
    _keep_low = _lower_keys(keep)
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
            if rel.lower() in _keep_low or _is_debris(rel):
                continue
            # Rescue source-like files AND binary/data deliverables
            # (databases, model weights, datasets) — both are plausibly
            # forgotten deliverables. Media scratch (screenshots, renders)
            # is neither and stays deletable.
            if not (_is_source_like(rel) or _is_data_deliverable(rel)
                    or _is_document_deliverable(rel)):
                continue
            found.add(rel)
    if found:
        tid = None if dry_run else _any_task_id(store, project_id)
        if tid:
            for rel in sorted(found):
                if rel.split("/", 1)[0] == _QUARANTINE_DIR:
                    continue        # §4ST: kept, never promoted to a deliverable
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


def _lower_keys(items) -> Set[str]:
    """Lowercase projection of a protection set, built ONCE per sweep.

    Membership is then a plain `rel.lower() in keep_low` — O(1) per file, and
    obvious at the call site rather than hidden behind an overridden operator.

    ⚠ 2026-08-11 (§4AT-D). Protection was exact-case while the walk yields the
    ON-DISK spelling, so a deliverable registered ``assets/hero.png`` and
    written ``assets/Hero.png`` was unprotected and deleted — by the DONE
    sweep AND the tidy. On macOS those are ONE file, so nothing else in the
    stack ever surfaces the drift; the registration silently stops working.

    Case-INSENSITIVE is the right safety bias on both filesystem types: on a
    case-insensitive volume the two names ARE one file, and on a
    case-sensitive one the worst case is keeping a file that could have been
    deleted. A spare file costs disk; a deleted deliverable is irreversible
    and unlogged.

    ⚠⚠ A `set` SUBCLASS WAS TRIED FIRST AND WAS A LANDMINE. Overriding
    `__contains__`/`add`/`__or__` left `update`, `|=`, `discard`, `remove`,
    `pop`, `clear`, `copy`, `union`, `-`, `&`, `^` unguarded — several of
    which silently return a PLAIN set (protection lost) or desync the mirror
    (a present key reporting absent ⇒ deleting a kept file). It also missed
    `_recover_deliverables`, which returns an ordinary `Set[str]`, so the
    empty-registration recovery — the module's own safety net — stayed
    case-sensitive. A projection built at the call site takes whatever set it
    is handed and cannot be bypassed by an operation nobody thought to
    override.
    """
    return {str(x).lower() for x in (items or ())}


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
    # §4UM: the owner's changes register their files (PROJECT_CHANGES.md, every path a change added or changed,
    # and the ones a running request is still writing) — a sweep at "planned work finished" must keep them
    _cf = getattr(store, "change_files", None)
    if callable(_cf):
        try:
            for p in _cf(project_id) or ():
                norm = _normalize_rel(str(p), pid)
                if norm:
                    keep.add(norm)
        except Exception:
            logger.warning("workspace cleanup: change log unreadable for %s; keeping all files",
                           project_id, exc_info=True)
            return None
    return keep


#: deleted paths a removal mark keeps (§4SS r2: 64 made a big sweep's later
#: files read "not created" in the reply's link notice); 16 projects max
_MARK_MAX_PATHS = 512


def _mark_removal(store, project_id, deleted=None, *, dry_run=False) -> None:
    """Record that files under ``project_id`` were REALLY removed just now.

    ⚠ Mark it where you catch it: the verifier's FILE-ARTIFACT re-read can
    race a removal fired by ANOTHER request while the verdict is in flight —
    a missing ledger-written file then reads as a failed delivery when the
    truth is "this project was just swept/deleted". The verdict consults
    this (recency-gated) and reports could-not-check instead of refuting.

    The first version was a single raw tuple and an audit broke it four
    ways at once: a DRY-RUN armed it (status stays "ok" and `deleted` lists
    what WOULD go); a second sweep of any other project evicted the slot
    inside the window, re-opening the race for the first; the mark carried
    no file list, so a tidy that deleted one stale screenshot "explained"
    the absence of `app.py` — a file tidy is structurally incapable of
    deleting; and the pid was stamped raw while `_project_dir` lowercases,
    so a mixed-case id missed in the false-refute direction. Hence: a
    dry-run guard, a per-project DICT (capped), the deleted list itself
    (``None`` = the whole tree went — hard delete), and casefolded keys.
    """
    if dry_run:
        return
    try:
        import time as _t
        marks = getattr(store, "recent_workspace_removals", None)
        if not isinstance(marks, dict):
            marks = {}
        key = str(project_id or "").strip().casefold()
        if not key:
            return
        rels = (None if deleted is None
                else [str(d).replace("\\", "/").casefold()
                      for d in list(deleted)[:_MARK_MAX_PATHS]])
        # ⚠ UNION with a still-fresh earlier mark: a second sweep/tidy of
        # the same project inside the window used to overwrite the first
        # list, un-explaining files the first sweep really deleted.
        prev = marks.get(key)
        if (prev and rels is not None and prev[1] is not None
                and _t.time() - prev[0] < 900):
            rels = list(dict.fromkeys(list(prev[1]) + rels))[:2 * _MARK_MAX_PATHS]
        elif prev and prev[1] is None:
            rels = None                       # whole-tree removal absorbs
        marks[key] = (_t.time(), rels)
        if len(marks) > 16:                    # cap; drop the oldest
            for stale in sorted(marks, key=lambda k: marks[k][0])[:-16]:
                marks.pop(stale, None)
        setattr(store, "recent_workspace_removals", marks)
    except Exception:
        pass


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

    # §4SS: a file the project's goal or a task's description/result NAMES is
    # a deliverable by the agent's own account ("Wrote report.pdf (81019
    # bytes)"), whatever its suffix and whether or not it was registered.
    named = _named_by_project(store, project_id, root) - set(keep)
    if named:
        keep = keep | named
        summary["kept_named"] = sorted(named)

    deleted: List[str] = []
    kept: List[str] = []
    freed = 0

    # Pass 0 — referenced-media protection (2026-07-20): an unregistered
    # media file that a kept source file points at (sprite sheet, texture,
    # favicon) is an asset of the build, not a stray screenshot — deleting
    # it breaks the deliverable the sweep just protected. Same scan the
    # recurring tidy runs; symlinks stay deletable (a symlink is scratch).
    media_candidates: List[str] = []
    _keep_low = _lower_keys(keep)
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
            if rel.lower() in _keep_low or _is_debris(rel):
                continue
            name = rel.split("/")[-1].lower()
            dot = name.rfind(".")
            if dot > 0 and name[dot:] in _MEDIA_SUFFIXES:
                media_candidates.append(rel)
    referenced = _referenced_media(root, media_candidates)

    import time as _time
    _trash_stamp = _time.strftime("%Y%m%d")
    _moved: List[str] = []
    # Pass 1 — delete unregistered files. topdown so we never descend into a
    # dir we're about to drop; followlinks=False so a symlink can't lead the
    # walk out of the project tree.
    _keep_low = _lower_keys(keep)
    _ref_low = _lower_keys(referenced)
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
            if rel.lower() in _keep_low or rel.lower() in _ref_low:
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
                # §4SU Phase 0 (audit): MOVED to the host trash, never
                # unlinked — the sweep's suffix rules still lose real
                # deliverables (.zip, fonts, audio, weights); a symlink is
                # scratch and is only unlinked
                _note_swept(fpath)          # §4UM round 2: the sweep's removal is not the owner's change
                if not fpath.is_symlink() and _to_project_trash(store, project_id, root, fpath, rel, _trash_stamp):
                    _moved.append(rel)
                else:
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
    if summary.get("status") == "ok" and (summary.get("deleted")
                                          or summary.get("dirs_removed")):
        _mark_removal(store, project_id, summary.get("deleted"),
                      dry_run=dry_run)
    if not dry_run and summary.get("status") == "ok" and deleted:
        # §4SU Phase 0 (audit): the DONE sweep leaves a durable record (the
        # b45ec99f PDF loss needed log forensics) — where each file went
        try:
            _tr = project_trash_dir(store)
            _gone = [r for r in deleted if r not in set(_moved)]
            store.log_event(project_id, None, "workspace_sweep", {
                "moved": _moved[:200], "deleted": _gone[:200], "count": len(deleted),
                "freed_bytes": freed,
                "trash": str(_tr / _trash_stamp / re.sub(r"[^0-9A-Za-z_-]", "_", str(project_id))) if _tr else ""})
        except Exception:  # noqa: BLE001
            logger.warning("workspace sweep: event not recorded", exc_info=True)
        try:
            purge_project_trash(store)
        except Exception:  # noqa: BLE001
            logger.debug("project trash purge skipped", exc_info=True)
    return summary


#: §4SU Phase 0: where the DONE sweep moves files — the HOST system dir
#: (``<memory_dir>/../project_trash/<YYYYMMDD>/<project>/<rel>``), never the
#: sandbox; day folders older than this are purged
PROJECT_TRASH_DIRNAME = "project_trash"
PROJECT_TRASH_KEEP_DAYS = 30


def project_trash_dir(store) -> Optional[Path]:
    md = getattr(store, "memory_dir", None)
    return (Path(md).parent / PROJECT_TRASH_DIRNAME) if md else None


def _note_swept(fpath) -> None:
    try:
        from ..memory.projects import note_system_path
        note_system_path(fpath)
    except Exception:  # noqa: BLE001
        pass


def _to_project_trash(store, project_id: str, root: Path, fpath: Path, rel: str, stamp: str) -> bool:
    """Move one regular workspace file to the host project trash; False when
    it could not be moved (the caller then deletes it, as before)."""
    trash = project_trash_dir(store)
    if trash is None:
        return False
    try:
        if not fpath.is_file() or Path(root).resolve() not in fpath.resolve().parents:
            return False
        if trash.is_symlink():
            return False
        dest = trash / stamp / re.sub(r"[^0-9A-Za-z_-]", "_", str(project_id)) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 1
        while dest.exists():
            n += 1
            dest = dest.with_name(f"{Path(rel).stem}.{n}{Path(rel).suffix}")
        shutil.move(str(fpath), str(dest))       # r1: a cross-device trash (EXDEV) still moves
        return True
    except (OSError, shutil.Error) as e:
        logger.warning("workspace sweep: could not move %s to the trash (deleted instead): %s", rel, e)
        return False


def purge_project_trash(store, now: Optional[float] = None) -> int:
    """Delete project-trash day folders older than PROJECT_TRASH_KEEP_DAYS
    (a symlinked trash or day folder is never followed)."""
    import shutil
    import time as _time
    trash = project_trash_dir(store)
    if trash is None or trash.is_symlink() or not trash.is_dir():
        return 0
    now = _time.time() if now is None else now
    n = 0
    for d in trash.iterdir():
        if d.is_symlink() or not d.is_dir() or not re.fullmatch(r"\d{8}", d.name):
            continue
        try:
            day = _time.mktime(_time.strptime(d.name, "%Y%m%d"))
        except ValueError:
            continue
        if now - day > (PROJECT_TRASH_KEEP_DAYS + 1) * 86400:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


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

#: Bound on how much source text the referenced-media check will read per
#: file.
#:
#: ⚠ RAISED FROM 512 KB IN §4GK ROUND 5. The old bound was chosen as "plenty
#: for any real index.html/css", which is true of hand-written sources and
#: false of the build output that sits beside them — a bundled `app.js`, a
#: sprite sheet manifest, a minified vendor file. Once round 5 made an
#: over-cap file inconclusive rather than silently half-scanned, a single
#: 600 KB bundle disabled the media tidy for that workspace permanently. The
#: work this cap was protecting against is a substring search for a handful
#: of basenames — microseconds per megabyte, no parsing — so the cap only
#: ever needed to stop a pathological file, not an ordinary bundle.
_REFERENCE_SCAN_MAX_BYTES = 8 * 1024 * 1024

#: One chunk of the streamed basename scan (§4GK round 6).
_REFERENCE_SCAN_CHUNK_BYTES = 1 * 1024 * 1024


def _scan_for_basenames(fpath: Path, basenames: Dict[str, str],
                        already: Set[str]) -> Set[str]:
    """Which of ``basenames`` appear anywhere in ``fpath``.

    Streamed in bounded chunks so a large bundle is answered COMPLETELY
    rather than partly (§4GK round 6). A basename split across a chunk
    boundary is caught by carrying an overlap of the longest name.
    Raises the same `OSError`/`ValueError` the caller already handles.
    """
    wanted = {rel: name.lower() for rel, name in basenames.items()
              if rel not in already}
    if not wanted:
        return set()
    # ⚠ THE OVERLAP IS IN BYTES, BECAUSE THE BOUNDARY IS (§4GK round 7). The
    # first version took `overlap` in CHARACTERS from decoded text while the
    # chunk boundary is a BYTE offset — so a non-ASCII basename straddling it
    # was mangled by `errors="replace"` in BOTH windows and found in neither.
    # Measured through the real scan on this box (a Greek locale, so the case
    # is not exotic): `εικόνα.png`, referenced at the 1 MB boundary, came back
    # unreferenced — and the caller turns that straight into `to_delete`,
    # which is the false DELETE this function's own contract forbids. Reading
    # the overlap back as BYTES makes the seam decode as one piece.
    overlap = max(len(n.encode("utf-8")) for n in wanted.values())
    found: Set[str] = set()
    from ..tools.file_system import read_text_nofollow
    size = fpath.stat().st_size
    pos = 0
    while pos < size and wanted:
        # Step back by the overlap in BYTES and re-read that seam, rather than
        # carrying decoded characters across a boundary that split a codepoint.
        _start = pos if pos == 0 else max(0, pos - overlap)
        chunk = read_text_nofollow(fpath, errors="replace",
                                   max_bytes=_REFERENCE_SCAN_CHUNK_BYTES
                                   + (pos - _start),
                                   offset=_start)
        if not chunk:
            break
        pos += _REFERENCE_SCAN_CHUNK_BYTES
        hay = chunk.lower()
        for rel, name in list(wanted.items()):
            if name in hay:
                found.add(rel)
                wanted.pop(rel, None)
    return found


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
    # ⚠ AN INCOMPLETE SCAN MUST NOT AUTHORISE A DELETE (§4GK round 4). Every
    # source-like file this loop cannot read — unreadable, a symlink the
    # hardened reader refuses, or over the size cap — used to be skipped
    # silently, and its references vanished with it. The caller turns "not in
    # the returned set" straight into `to_delete`, so one unreadable
    # `index.html` was enough to delete the assets it points at. This
    # function's own contract is that a false KEEP costs kilobytes and a
    # false DELETE breaks the build, so an unscanned file makes the answer
    # inconclusive and everything is kept.
    unscanned: List[str] = []
    #: Source files the scan could not read at all. Their silence is not
    #: evidence, so an unmatched candidate is kept when any exist.
    partial: List[str] = []          # kept for the report shape; no longer filled
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
                # §4GJ round 3: the `is_symlink()` pre-check left a
                # check-then-read window on a tree the model can rewrite.
                # `read_text_nofollow` makes the refusal part of the open —
                # the streaming scan below opens it that way per chunk.
                # ⚠ SCANNED WHOLE, IN BOUNDED CHUNKS (§4GK round 6).
                #
                # Round 4 refused any file over the cap, which made the answer
                # inconclusive and permanently disabled the media tidy for any
                # workspace holding one ordinary bundle. Round 5 scanned the
                # HEAD instead — better, but an unmatched candidate still fell
                # back to keep-everything, so the tidy still could not DELETE
                # anything in such a workspace. Both were fighting a cap that
                # never needed to exist here: the question is "does any source
                # file MENTION this basename", which is a substring search, not
                # a parse. Streaming it costs one chunk of memory and answers
                # completely, so there is no partial state left to reason about.
                #
                # The overlap carries the longest basename across a chunk
                # boundary, so a name split by the read is still found.
                _hits_here = _scan_for_basenames(fpath, basenames, hit)
                if _hits_here:
                    hit.update(_hits_here)
                continue
            except (OSError, ValueError):
                unscanned.append(rel)
                continue
            # (The CASE-INSENSITIVE match of §4AT-D — an `index.html` writing
            # `src="assets/hero.png"` against a disk file `assets/Hero.png`
            # found no reference and the asset was deleted — now lives in
            # `_scan_for_basenames`, which lowercases both the needle and every
            # chunk. The inline loop that used to do it here became dead code
            # after the streaming rewrite and referenced a name that no longer
            # exists; the lint gate caught it, §4GK round 6.)
        if len(hit) == len(basenames):
            break
    _unmatched = [m for m in media_rels if m not in hit]
    if (unscanned or partial) and _unmatched:
        # Only inconclusive about the candidates NOTHING matched: a file that
        # was matched is referenced whatever the unread tail says.
        _why = []
        if unscanned:
            _why.append(f"{len(unscanned)} unreadable ({', '.join(unscanned[:3])})")
        if partial:
            _why.append(f"{len(partial)} read only to the {_REFERENCE_SCAN_MAX_BYTES // (1024 * 1024)} MB "
                        f"cap ({', '.join(partial[:3])})")
        pretty_log(
            "Workspace Tidy",
            f"referenced-media scan is INCOMPLETE — {'; '.join(_why)} — so "
            f"{len(_unmatched)} unmatched candidate(s) are KEPT this pass "
            "rather than deleted on an answer the scan could not give.",
            level="WARNING", icon=Icons.WARN)
        return set(media_rels)
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
    _keep_low = _lower_keys(keep)
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            fpath = base / fname
            try:
                rel = fpath.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel.lower() in _keep_low:
                continue
            if _in_git_tree(rel):
                continue  # version-control state is never the tidy's to delete
            # ⚠ ON AN ACTIVE PROJECT, A NAME NEVER OUTRANKS BEING SOURCE
            # (2026-08-11, §4AT-D). This checked `_is_debris` FIRST, so
            # `debug_utils.py`, `temp_loader.py`, `scratch_notes.py` and
            # `tmp_fix.py` were deleted out from under a build that imports
            # them — while this module's contract says "Source/document files
            # are NEVER deleted here regardless of registration".
            #
            # ⚠⚠ SCOPED TO THE TIDY ONLY, and the scope is the whole point.
            # The first version of this fix inverted the precedence inside
            # `_is_debris` itself, which also changed the DONE SWEEP — where
            # sweeping `temp_probe.py` is deliberate and pinned by
            # `test_partial_keepset_debris_named_sources_still_swept`. That
            # test went red and was RIGHT: a finished project's scratch should
            # go. The harm is deleting source from a project still being
            # WORKED ON. Same word, two lifecycles, opposite correct answers.
            if _tidy_is_debris(rel):
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
    _ref_low_tidy = _lower_keys(referenced)
    # §4SS: same rule as the DONE sweep — a media file a task names
    # ("→ results/plot.png") is the task's deliverable, not a screenshot
    _named_low = (_lower_keys(_named_by_project(store, project_id, root))
                  if media_candidates else set())
    to_delete = debris + [m for m in media_candidates
                          if m.lower() not in _ref_low_tidy
                          and m.lower() not in _named_low]

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
            _note_swept(fpath)              # §4UM round 2: the tidy's removal is not the owner's change
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
    if summary.get("status") == "ok" and (summary.get("deleted")
                                          or summary.get("dirs_removed")):
        _mark_removal(store, project_id, summary.get("deleted"),
                      dry_run=dry_run)
    return summary
