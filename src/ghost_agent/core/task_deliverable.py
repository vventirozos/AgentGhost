"""What a project task description declares it will produce (§4QA, 2026-10-18).

Decomposed tasks are written like ``Task 1.3: Analyze recursion depth models
→ research/recursion-depth-models.md`` — a numbering prefix, the topic, and a
declared deliverable after the arrow. Project 448d5afa86f0 showed three ways
the raw string hurt:

- autoadvance web-searched the WHOLE description, so ``task 1.3`` and the
  ``→ path`` tail went into the query; the off-topic filter found neither in
  any result and task 1.3 FAILED (``task 6.2`` passed as "low relevance");
- the research brief was written to ``research/<slug of the title>.md``
  (``task-6-2-compare-…md``) while the task declared
  ``theory/comparison-with-existing-frameworks.md`` — nothing ever read the
  declared path, so the final reply linked a file that never existed;
- tasks closed DONE with the declared file missing (written one folder
  deeper) or holding three bytes (``...``).

:func:`task_topic` is what to search for, :func:`declared_deliverable` is the
path the task promised, :func:`hollow_document_reason` says a document has no
content, and :func:`deliverable_problem` checks a workspace for the promised
file — the gate ``manage_projects task_update`` applies before a DONE.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

# "Task 1.3:", "1.3.", "Step 2 -", "Phase 4 —", "#7)" at the very start.
_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:(?:task|step|phase)\s*#?\s*\d+(?:\.\d+)*|#?\d+(?:\.\d+)+|#?\d+)"
    r"\s*(?:[:.)]|[-–—](?=\s))\s*", re.I)    # "5-7 year outlook" is no prefix
# A trailing "→ path" (also -> and =>), optionally in backticks and closing
# the sentence ("→ `report/final.md`.").
_ARROW_TAIL_RE = re.compile(r"\s*(?:→|->|=>)\s*`?(?P<path>[^\s`]+)`?[.,;:]?\s*$")
# A workspace-relative file path with an extension; no spaces, no "..".
_PATH_RE = re.compile(r"^[\w.\-]+(?:/[\w.\-]+)*\.[A-Za-z0-9]{1,8}$")
# A bare name (no folder) is a file only with an extension a deliverable has:
# "→ Vue.js" / "-> socket.io" / "→ Python3.12" / "→ e.g." name a technology.
_FILE_SUFFIXES = frozenset({
    "md", "markdown", "rst", "txt", "pdf", "html", "htm", "css", "csv", "tsv", "json", "jsonl",
    "yaml", "yml", "toml", "xml", "ini", "cfg", "py", "ipynb", "js", "mjs", "ts", "tsx", "jsx",
    "sh", "sql", "r", "jl", "tex", "bib", "svg", "png", "jpg", "jpeg", "gif", "webp", "mp3",
    "wav", "mp4", "docx", "xlsx", "pptx", "zip", "c", "h", "cpp", "hpp", "go", "rs", "java",
    "kt", "swift", "rb", "php", "lua", "dart", "vue", "svelte", "log"})
_JS_LIBRARY_RE = re.compile(
    r"^(?:node|vue|next|nuxt|react|three|d3|chart|express|socket|ember|backbone|angular|alpine"
    r"|p5|ml5|tensorflow|brain|moment|underscore|nest|solid|preact|ionic|meteor|deno)\.js$", re.I)

#: Documents judged by :func:`hollow_document_reason`. Code and ``.txt`` are not
#: (``requirements.txt`` is two words); an empty
#: ``__init__.py`` is a real file.
DOC_SUFFIXES = (".md", ".markdown", ".rst")
#: Fewer words than this and a document has no content.
MIN_DOC_WORDS = 8

_ELISION_ONLY_RE = re.compile(r"^\s*(?:[-*>#]\s*)*(?:\.\.\.|…|tbd|todo|tba)?\s*$", re.I)
_WORD_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)
# Chinese/Japanese/Korean text has no spaces: a paragraph is one "word" to
# _WORD_RE. Two such characters count as one word.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


_PROJECT_PREFIX_RE = re.compile(r"^projects/[0-9a-f]{12}/")


def _normalise(path: str) -> Optional[str]:
    p = (path or "").strip().strip("`'\"").rstrip(".,;:)")
    for head in ("/workspace/", "./"):
        if p.startswith(head):
            p = p[len(head):]
    # "/workspace/projects/<id>/results/x.md" is the project's results/x.md
    # (the writer resolves it so) — r9: the gate looked for ws/projects/<id>/…
    p = _PROJECT_PREFIX_RE.sub("", p, count=1)
    if p.startswith("/") or not _PATH_RE.match(p):
        return None
    if any(part in ("", ".", "..") for part in p.split("/")):
        return None
    if "/" not in p and (p.rsplit(".", 1)[-1].lower() not in _FILE_SUFFIXES
                         or _JS_LIBRARY_RE.match(p)):
        return None
    return p


def declared_deliverable(description) -> Optional[str]:
    """The workspace-relative path after a trailing arrow, or None.

    Only a PATH counts: ``… → get actual scores`` is prose, not a file."""
    m = _ARROW_TAIL_RE.search(str(description or ""))
    if not m:
        return None
    return _normalise(m.group("path"))


def task_topic(description) -> str:
    """The description without its numbering prefix and declared-path tail —
    what a web search or a research brief should be about."""
    s = str(description or "")
    if declared_deliverable(s):
        s = _ARROW_TAIL_RE.sub("", s)
    s = _NUM_PREFIX_RE.sub("", s, count=1)
    return s.strip() or str(description or "").strip()


def hollow_document_reason(text, path) -> Optional[str]:
    """Why ``text`` is no document for ``path`` — or None.

    A ``.md``/``.rst`` whose lines are only elision markers
    (``...``, ``…``, TODO) or which holds fewer than :data:`MIN_DOC_WORDS`
    words. Other suffixes are never judged."""
    if not str(path or "").lower().endswith(DOC_SUFFIXES):
        return None
    lines = [ln for ln in str(text or "").splitlines() if not _ELISION_ONLY_RE.match(ln)]
    joined = "\n".join(lines)
    words = len(_WORD_RE.findall(_CJK_RE.sub(" ", joined))) + len(_CJK_RE.findall(joined)) // 2
    if words >= MIN_DOC_WORDS:
        return None
    return (f"{path} has no content ({words} word(s); a document needs at least "
            f"{MIN_DOC_WORDS}) — write the actual text, not a placeholder")


def elided_document_reason(text, path) -> Optional[str]:
    """Why ``text`` is only a placeholder for document ``path`` — every line
    blank or an elision marker (``...``, ``…``, TODO). The coding executor's
    floor for ANY document it writes; the word floor of
    :func:`hollow_document_reason` is for a document a task PROMISED (a
    one-line README is a weak file, not a refusal)."""
    if not str(path or "").lower().endswith(DOC_SUFFIXES):
        return None
    if any(not _ELISION_ONLY_RE.match(ln) for ln in str(text or "").splitlines()):
        return None
    return (f"{path} is only a placeholder (blank or '...' lines) — write the "
            f"actual text")


def _read_head(path: Path, limit: int = 65536) -> Optional[str]:
    """First ``limit`` bytes of a REGULAR file that is not itself a symlink
    (a symlinked FOLDER on the way is refused by :func:`deliverable_problem`)."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return None
    try:
        from ..tools.file_system import _block_if_regular
        if not _stat.S_ISREG(_block_if_regular(fd).st_mode):   # §4SD: virtiofs EAGAIN; the fd, not the path
            return None
        return os.read(fd, limit).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def _find_elsewhere(ws: Path, rel: str, max_files: int = 5000) -> Optional[str]:
    """A file under ``ws`` whose path ENDS with ``/rel`` (written one or more
    folders deeper — the stray-prefix case). Symlinked dirs are not entered."""
    seen = 0
    for root, dirs, files in os.walk(ws, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            seen += 1
            if seen > max_files:
                return None
            full = Path(root) / f
            r = full.relative_to(ws).as_posix()
            if r != rel and r.endswith("/" + rel):
                return r
    return None


def deliverable_problem(ws, description) -> Optional[str]:
    """Why the task's declared deliverable is not in workspace ``ws`` — None
    when the task declares none, or the file exists with content. (Its quotes
    are checked apart, advisory: :func:`note_misquotes`.)"""
    rel = declared_deliverable(description)
    if not rel or ws is None:
        return None
    ws = Path(ws)
    target = ws / rel
    if os.path.islink(target):
        return f"the declared deliverable {rel} is a symbolic link, not a file"
    try:
        # O_NOFOLLOW guards the last component only: a symlinked FOLDER on
        # the way ("linked/z.md", linked → /elsewhere) leaves the workspace
        _real_ws = os.path.realpath(ws)
        if os.path.lexists(target) and not os.path.realpath(target).startswith(_real_ws + os.sep):
            return f"the declared deliverable {rel} is reached through a symbolic link out of the workspace"
    except (OSError, ValueError):
        return None
    text = _read_head(target)
    if text is None:
        other = _find_elsewhere(ws, rel)
        if other:
            return (f"the declared deliverable {rel} does not exist — a file with that "
                    f"name is at {other} (one folder off). Move it to {rel} "
                    f"(file_system operation=move) or write {rel}")
        return f"the declared deliverable {rel} does not exist — write it first"
    return hollow_document_reason(text, rel)


#: the owner's ledger line — fixed text, no transcript words (r32: the ledger
#: is shown as trusted every turn and keeps only 2400 characters), and ONE
#: line per project, rewritten in place (r33: a line per task — 12 rerun
#: tasks — pushed the owner's design lines out)
_NOTE_PREFIX = "§4QA-e quote check (advisory)"
_NOTE_SHOWN = 5


def _note_ledger(store, project_id, task_id) -> None:
    def _mutate(meta):
        raw = meta.get("quote_check_tasks")
        ids = [str(t) for t in raw] if isinstance(raw, list) else []
        if str(task_id) not in ids:
            ids.append(str(task_id))
        shown = ", ".join(ids[-_NOTE_SHOWN:]) + (f" (+{len(ids) - _NOTE_SHOWN} earlier)"
                                                   if len(ids) > _NOTE_SHOWN else "")
        line = (f"{_NOTE_PREFIX}: the write-ups of {len(ids)} closed task(s) — {shown} — seem to quote "
                "the probe answers with words changed; the closes stood, the quotes in those files "
                "should be fixed.")
        lines = [l for l in (meta.get("design_ledger") or "").splitlines()
                 if l.strip() and not l.startswith(_NOTE_PREFIX)]
        lines.append(line)
        lines = lines[-store.LEDGER_MAX_LINES:]
        while len(lines) > 1 and len("\n".join(lines)) > store.LEDGER_MAX_CHARS:
            lines.pop(0)
        meta["design_ledger"] = "\n".join(lines)
        meta["quote_check_tasks"] = ids
        return meta
    store._atomic_metadata_update(project_id, _mutate)


async def note_misquotes(store, project_id, task_id, description, stored_description=None) -> Optional[str]:
    """§4QA-e (operator 2026-10-19, "advisory"): the quote check never holds a
    close. Reviews r27–r31 each found 3–9 MAJOR defects in a BLOCKING fuzzy
    judge — honest write-ups refused, close rewordings let through — so a
    write-up that seems to reword the model's answers is NOTED: one ledger
    line for the owner, and the text returned for the model to fix. None when
    nothing looks reworded (or the check could not run). Call it only after
    the task really closed DONE (r32: a later gate's refusal was logged as
    "closed"). The check runs in a worker thread (r32: seconds on the event
    loop); the store is touched only here, on the caller's thread."""
    import asyncio
    try:
        ws = store.ensure_workspace(project_id)
        args = (ws, description, getattr(store, "memory_dir", None), project_id,
                stored_description, project_quote_text(store, project_id))
        why = await asyncio.to_thread(quote_problem_of, *args)
    except Exception:  # noqa: BLE001 — advisory: a failed check never touches the close
        logger.warning("quote check skipped for task %s", task_id, exc_info=True)
        return None
    if why:
        try:
            _note_ledger(store, project_id, task_id)
        except Exception:  # noqa: BLE001
            logger.warning("quote check: ledger line for task %s not written", task_id, exc_info=True)
    return why


def project_quote_text(store, project_id) -> str:
    """The project's title and goal — words a write-up may quote (§4QA-e r29)."""
    try:
        p = store.get_project(project_id) or {}
    except Exception:  # noqa: BLE001
        return ""
    return f"{p.get('title') or ''}\n{p.get('goal') or ''}"


#: §4QA-e r30: the quote check reads this much of a write-up (the 64 KB head
#: cut a verbatim answer mid-word and refused it)
_QUOTE_READ = 1 << 20


def quote_problem_of(ws, description, memory_dir, project_id=None, stored_description=None,
                     project_text="") -> Optional[str]:
    """§4QA-e: the declared deliverable rewords a quote of a probe transcript
    it cites. The task's STORED description is the quote source, and the file
    it declared is checked too (r27 M3: a description rewritten in the
    closing call — a quote added to it, or the deliverable re-pointed at the
    transcript itself — let the reworded write-up close DONE)."""
    if not memory_dir or ws is None:
        return None
    from .probe_transcripts import quote_problem
    ws = Path(ws)
    pid = str(project_id or ws.name)
    stored = str(stored_description if stored_description is not None else description or "")
    rels = [r for r in dict.fromkeys([declared_deliverable(description), declared_deliverable(stored)]) if r]
    for rel in rels:
        target = ws / rel
        if os.path.islink(target):
            continue
        text = _read_head(target, _QUOTE_READ)
        if text and len(text.encode("utf-8", errors="replace")) >= _QUOTE_READ:
            text = text.rsplit("\n", 1)[0]       # r30: never a word cut in half
        if text:
            why = quote_problem(memory_dir, pid, text, rel, stored, project_text)
            if why:
                return why
    return None
