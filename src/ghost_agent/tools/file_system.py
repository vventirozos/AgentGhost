import asyncio
import hashlib
import os
import re
import urllib.parse
import json
import shlex
from pathlib import Path
from typing import Any
import httpx
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import request_new_tor_identity

def _read_head(path: Path, max_bytes: int = 8192) -> bytes:
    """Read the first `max_bytes` of a file. Used by `_looks_like_binary`
    to sniff before committing to a full text-mode read."""
    with open(path, "rb") as f:
        return f.read(max_bytes)


# Hard ceiling on a single line-range read so an absurd range (1..1e9) can't
# materialise a whole huge file. Generous — a surgical re-read is normally a
# few dozen lines — but bounded.
_LINE_RANGE_MAX_LINES = 2000


def _read_line_range(path: Path, filename: str,
                     start_line: int = None, end_line: int = None) -> str:
    """Return a 1-based, line-number-prefixed slice ``[start_line, end_line]``.

    Streams the file and stops at ``end_line`` (or the line cap), so cost is
    proportional to the requested range, not the file size. Defaults: start=1,
    end=start+cap. Clamps/validates the bounds and reports what actually came
    back so the model can widen or shift the window."""
    try:
        s = 1 if start_line is None else int(start_line)
        e = None if end_line is None else int(end_line)
    except (TypeError, ValueError):
        return ("Error: start_line/end_line must be integers, e.g. "
                f"file_system(operation='read', path='{filename}', start_line=200, end_line=260).")
    if s < 1:
        s = 1
    if e is not None and e < s:
        return (f"Error: end_line ({e}) is before start_line ({s}). "
                "Give a range where end_line >= start_line.")
    # Cap the span so a runaway range can't dump the whole file.
    cap_end = s + _LINE_RANGE_MAX_LINES - 1
    hard_end = cap_end if e is None else min(e, cap_end)

    # Binary sniff first (mirrors the whole-file read path).
    try:
        if _looks_like_binary(_read_head(path, 8192)):
            return (f"Error: '{filename}' appears to be a binary file. You "
                    "cannot read it as text. If it is an image, use "
                    "'vision_analysis'.")
    except OSError as oe:
        return f"Error: failed to read '{filename}': {oe}"

    picked = []
    last_lineno = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                last_lineno = lineno
                if lineno < s:
                    continue
                if lineno > hard_end:
                    break
                picked.append(f"{lineno}\t{line.rstrip(chr(10))}")
    except OSError as oe:
        return f"Error: failed to read '{filename}': {oe}"

    if not picked:
        if last_lineno == 0:
            return f"--- {filename} (lines {s}-{hard_end}) ---\n(file is empty)"
        return (f"Error: start_line {s} is past the end of '{filename}' "
                f"(only {last_lineno} lines). Re-read with a start_line "
                f"between 1 and {last_lineno}.")

    body = "\n".join(picked)
    shown_first = s
    shown_last = s + len(picked) - 1
    truncated = ""
    if e is not None and hard_end < e:
        truncated = (f"\n[... range capped at {_LINE_RANGE_MAX_LINES} lines; "
                     f"re-read from start_line={hard_end + 1} for more]")
    elif e is None and len(picked) == _LINE_RANGE_MAX_LINES:
        truncated = (f"\n[... {_LINE_RANGE_MAX_LINES}-line cap; re-read from "
                     f"start_line={hard_end + 1} for more]")
    return (f"--- {filename} (lines {shown_first}-{shown_last}) ---\n"
            f"{body}{truncated}")


# Allowable text-control characters (TAB, LF, CR, FF) — anything else in the
# 0..31 range that ISN'T one of these is a strong binary signal.
_TEXT_CTRL_OK = frozenset({0x09, 0x0A, 0x0C, 0x0D})


def _looks_like_complete_python_module(text: str) -> bool:
    """Return True when `text` looks like a self-contained Python module.

    Used by the `replace` auto-promote path: if the caller asked for a
    targeted replace but forgot `replace_with`, and their `content`
    parses as real Python with top-level imports/definitions, we treat
    the call as if they'd said operation='write' — the most common
    misfire mode observed in self-play logs.

    Conservative signal — all of the following must be true:
      1. Parses via `ast.parse` (no syntax errors).
      2. Has at least one top-level `import` / `from ... import`.
      3. Has at least one top-level `def` / `class` OR an `if __name__`.
      4. At least 5 non-blank lines.

    A snippet of a function body will fail (no import at top level); a
    single-line file will fail (line count); malformed code will fail
    (syntax)."""
    if not text or len(text) < 60:
        return False
    non_blank = [ln for ln in text.splitlines() if ln.strip()]
    if len(non_blank) < 4:
        return False
    try:
        import ast
        tree = ast.parse(text)
    except Exception:
        return False
    has_import = False
    has_def_or_main = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            has_import = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_def_or_main = True
        elif isinstance(node, ast.If):
            # Detect `if __name__ == "__main__":`
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                has_def_or_main = True
    return has_import and has_def_or_main


def _looks_like_binary(head: bytes) -> bool:
    """Return True when `head` looks like a binary file.

    Heuristic: any NUL byte → binary; otherwise count non-text control bytes
    and treat the file as binary if they exceed 5% of the sample. This is
    the same approach `git diff` uses internally and catches images, PDFs,
    archives, executables, and most font/binary blob formats.
    """
    if not head:
        return False
    if b"\x00" in head:
        return True
    suspicious = sum(1 for b in head if b < 0x20 and b not in _TEXT_CTRL_OK)
    return (suspicious / len(head)) > 0.05


def _get_safe_path(sandbox_dir: Path, filename: str, *, allow_root: bool = True) -> Path:
    """
    Safely resolve a path relative to the sandbox root, preventing
    traversal attacks.

    ``allow_root=False`` additionally REJECTS a path that resolves to the
    sandbox/project root itself. Reads/lists of the root are fine, but a
    DESTRUCTIVE op (delete / rename-move / copy-source) must never target
    it: ``.``, ``/``, ``workspace``, ``/workspace``, an empty string, and
    (when project-scoped) ``projects/<active-id>`` all collapse to the
    root here, so without this guard ``file_system(operation='delete',
    path='/workspace')`` would ``rmtree`` the entire workspace.

    Prior behavior stripped a leading ``sandbox/`` prefix to "heal"
    agent hallucinations. That silently broke the invariant that a
    file written at path ``P`` can be read back at path ``P`` by a
    script running in the sandbox container: the write healed the
    prefix (file landed at ``<sandbox>/foo``) but a container-side
    ``open("sandbox/foo")`` (relative to WORKDIR ``/workspace``) saw
    ``/workspace/sandbox/foo`` and failed. See 2026-04-19 trace for
    the exact blast radius — two debug cycles burned on an illusion.

    We keep the leading-slash fix (so ``/foo.txt`` is treated as
    ``foo.txt`` at sandbox root) but stop rewriting ``sandbox/``
    prefixes: whatever path the agent names, we honor it literally.
    Write-then-read symmetry is more valuable than heal-the-confused.

    Exception: ``/workspace/`` (and bare ``workspace/``) IS stripped.
    ``/workspace`` is the container's WORKDIR where the sandbox root
    is bind-mounted, so ``/workspace/foo.py`` inside the container is
    the same file as ``foo.py`` at the host's sandbox root. The LLM
    sees ``/workspace/...`` in every shell command it emits (the
    prompt tells it to), so when it calls ``file_system(path=
    "/workspace/foo.py")`` the intent is unambiguous: "sandbox root /
    foo.py". Without this strip, the path resolved to ``$SANDBOX_DIR
    /workspace/foo.py`` — a phantom ``workspace/`` subdir on host —
    and later ``execute`` calls using the same ``/workspace/foo.py``
    path inside the container got ENOENT because the real file was
    at ``/workspace/workspace/foo.py``. Confirmed 2026-04-24 in an
    in_gr_news skill session: the LLM burned ~60 s rediscovering the
    mismatch ("python3: can't open file '/workspace/skills/in_gr_news.py'").
    Write-then-read symmetry is still preserved: both tools now
    interpret ``/workspace/...`` as sandbox-root-relative.
    """
    raw = str(filename).strip()

    # 0a. Expand a leading "~" first. The model echoes host-style paths it
    # saw in conversation (e.g. the run command "python3 ~/Data/AI/Data/
    # sandbox/chess_client.py" it just gave the user — 2026-07-05 request
    # 9F burned a strike on exactly this). expanduser yields the host-
    # absolute form, which step 0 below maps back inside the sandbox; a
    # "~" that expands OUTSIDE the sandbox falls through to the normal
    # relative-path handling like any other foreign absolute path.
    if raw == "~" or raw.startswith("~/"):
        try:
            raw = str(Path(raw).expanduser())
        except Exception:
            pass

    # 0. Host-absolute sandbox path → sandbox-relative. The LLM sometimes
    # echoes the full HOST path of the sandbox (it sees it in a prior
    # sandbox-tree listing / file-read error and passes it back), e.g.
    # "/Users/x/.../sandbox/report.md". Joining that onto sandbox_dir
    # produced a phantom nested path ("<sandbox>/Users/x/.../sandbox/
    # report.md") → ENOENT, then a wasted strike + a retry with the
    # relative form (2026 report-regeneration trace burned ~5 turns on
    # this). If the given absolute path is already inside the sandbox
    # root, strip the prefix and honour the remainder as sandbox-relative.
    # Does NOT touch "/workspace/..." (handled below) — that host path is
    # not under the sandbox dir, so the relative_to check skips it.
    if raw.startswith("/"):
        try:
            _sb = sandbox_dir.resolve()
            _abs = Path(raw).resolve()
            try:
                inside = (_abs == _sb) or _abs.is_relative_to(_sb)
                rel = "" if _abs == _sb else str(_abs.relative_to(_sb))
            except AttributeError:  # Python < 3.9
                s_abs, s_sb = str(_abs), str(_sb).rstrip("/")
                inside = s_abs == s_sb or s_abs.startswith(s_sb + "/")
                rel = "" if s_abs == s_sb else s_abs[len(s_sb) + 1:]
            if inside:
                raw = rel
        except Exception:
            pass

    # 1. Strip leading slashes to treat as relative
    clean_name = raw.lstrip("/")

    # 1a. Strip a leading ``workspace/`` prefix (container WORKDIR).
    # Case-sensitive and exact-segment: ``workspaces/`` or
    # ``workspace_xyz/`` are untouched. Applied AFTER the leading
    # slash strip so ``/workspace/foo``, ``workspace/foo``, and
    # ``//workspace/foo`` all collapse identically.
    if clean_name == "workspace":
        clean_name = ""
    elif clean_name.startswith("workspace/"):
        clean_name = clean_name[len("workspace/"):]

    # 1b. Redundant project-prefix heal (project-scoped sandbox only).
    # When a project is active the sandbox root IS <sandbox>/projects/<id>
    # (so a project's files clean up with one `rm -rf`). The model reaches
    # for a file via every route it has seen it referenced, and each one
    # would double-nest under the already-scoped root:
    #   * ``projects/<id>/X``          (container-root-relative listing)
    #   * ``sandbox/projects/<id>/X``  (HOST-root-relative — the sandbox dir
    #     is literally named ``sandbox`` on disk and surfaces that way in
    #     the project's workspace_dir and some listings; the model then
    #     builds e.g. ``file:///workspace/sandbox/projects/<id>/X`` and the
    #     browser 404s on the doubled path — observed live, webOS build).
    # Strip the longest matching redundant prefix so the file lands once.
    # Case-insensitive on the id (project ids are canonicalised to lowercase
    # hex). Untouched for the normal unscoped root, whose parent dir is not
    # literally named ``projects``.
    if sandbox_dir.parent.name == "projects":
        _pid = sandbox_dir.name
        _root_name = sandbox_dir.parent.parent.name
        # Longest (root-qualified) form first so it wins over the bare one.
        _redundant = []
        if _root_name:
            _redundant.append(f"{_root_name}/projects/{_pid}")
        _redundant.append(f"projects/{_pid}")
        for _p in _redundant:
            _pl = _p.lower()
            if clean_name.lower().startswith(_pl + "/"):
                clean_name = clean_name[len(_p) + 1:]
                break
            if clean_name.lower() == _pl:
                clean_name = ""
                break
        # Generic guard: any OTHER ``projects/<slug>/`` prefix is the model
        # guessing a project path by name/title rather than the active id —
        # e.g. it wrote ``projects/DeskMiniX3/index.html`` (title-derived) the
        # same turn it created the project. Left literal, that spawns an orphan
        # ``projects/<title>/`` tree no project record owns: the file, the
        # store's workspace_dir, and the cleanup sweep then key on three
        # different ids (observed live — the deliverable survived only because
        # the sweep looked at the canonical id's empty dir). Collapse it into
        # the active project's dir so they all agree. (Trade-off: a literal
        # ``projects/`` SUBDIR inside a project workspace is not addressable
        # this way — acceptable; the model should not nest one.)
        _m = re.match(r"(?i)projects/[^/]+/", clean_name)
        if _m:
            clean_name = clean_name[_m.end():]

    # 2. Resolve to absolute path inside the sandbox root
    target_path = (sandbox_dir / clean_name).resolve()

    # 3. Ensure it's still inside sandbox (Robust Pathlib Check)
    _sb_resolved = sandbox_dir.resolve()
    try:
        if not target_path.is_relative_to(_sb_resolved):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")
    except AttributeError:
        # Fallback for Python < 3.9. Compare against the root WITH a trailing
        # separator so a sibling dir ("/sandbox-evil") can't prefix-match
        # "/sandbox" (the classic str.startswith containment bug).
        _sb_str = str(_sb_resolved)
        _t_str = str(target_path.resolve())
        if _t_str != _sb_str and not _t_str.startswith(_sb_str + os.sep):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")

    # 4. Destructive ops must not target the root itself.
    if not allow_root and target_path == _sb_resolved:
        raise ValueError(
            f"Security Error: refusing to run a destructive operation on the "
            f"sandbox/project root ('{filename}'). Target a specific file or "
            f"subdirectory instead."
        )

    return target_path


# Sentinel keys the projects tool uses to bind the active project to the
# conversation that activated it (kept in sync with tools/projects.py). Read
# here so file scoping can recover the project when the process-global
# ``current_project_id`` is transiently cleared by a concurrent conversation.
_PROJECT_BIND_PID = "__current_project__"
_PROJECT_BIND_CONV = "__current_project_conv__"


def _conversation_bound_project(context) -> str:
    """The project bound to THIS conversation (via the projects-tool scratchpad
    sentinels), or "" if none / it belongs to another conversation. Used as a
    fallback when ``current_project_id`` was cleared mid-request."""
    sp = getattr(context, "scratchpad", None)
    conv = getattr(context, "conversation_key", None)
    if sp is None or not conv:
        return ""
    try:
        bound_pid = sp.get(_PROJECT_BIND_PID)
        bound_conv = sp.get(_PROJECT_BIND_CONV)
    except Exception:
        return ""
    if isinstance(bound_pid, str) and bound_pid and bound_conv == conv:
        return bound_pid.strip().lower()
    return ""


def project_scoped_sandbox(context, stateful: bool = False):
    """Return ``(host_dir, container_workdir)`` scoped to the active project's
    workspace (``<sandbox>/projects/<id>/``), creating the dir on demand; or
    ``(sandbox_root, None)`` when no project is active (or ``stateful=True``,
    since the Jupyter kernel is pinned to ``/workspace``).

    This is the SINGLE SOURCE OF TRUTH for per-project sandbox scoping. Every
    surface that reads or writes the agent's working files routes through it
    so they all agree on "where the working directory is": the tool registry
    (file_system/execute/report_pdf/vision/image_generation/knowledge_base),
    the ambient sandbox-state listing injected into each turn, the
    ``/api/upload`` route, and the swarm bridge. Keeping them in sync is what
    prevents the model from being shown a root listing while its file ops land
    in the project dir (the source of the "file is at sandbox root" confusion).

    Scoping activates only for a genuine string project id (a MagicMock
    context attribute can't trigger it).
    """
    sb = getattr(context, "sandbox_dir", None)
    base = Path(sb) if sb is not None else None
    pid = getattr(context, "current_project_id", None)
    pid = pid.strip().lower() if isinstance(pid, str) else ""
    if not pid and not stateful:
        # ``current_project_id`` is process-global and can be cleared MID-REQUEST
        # by a concurrent conversation's reconcile (observed live: a 700s
        # autoadvance+manual build had its scoping wiped, so the agent's file
        # writes landed in the sandbox ROOT and it thrashed for minutes). The
        # per-conversation binding is NOT stomped that way, so re-derive this
        # conversation's project from it.
        pid = _conversation_bound_project(context)
    if pid and not stateful and base is not None:
        sub = base / "projects" / pid
        try:
            sub.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return sub, f"{_CONTAINER_WORKDIR}/projects/{pid}"
    return base, None


def project_download_prefix(sandbox_dir: Path) -> str:
    """Return the ``projects/<id>/`` path prefix when ``sandbox_dir`` is a
    project-scoped sandbox (its parent dir is literally ``projects``), else
    "".

    Artefact tools (report_pdf, image_generation) write into the scoped
    dir but advertise a ``/api/download/<path>`` link; that route resolves
    against the sandbox ROOT, so the link must carry this prefix to stay
    reachable. Detected the same way as the file_system path heal and
    ``_to_container_path`` un-scoping, so all three agree.
    """
    if sandbox_dir is not None and Path(sandbox_dir).parent.name == "projects":
        return f"projects/{Path(sandbox_dir).name}/"
    return ""


# Container WORKDIR — kept in sync with sandbox.docker.CONTAINER_WORKDIR.
# Duplicated here as a literal so this module can be imported without a
# hard dep on the sandbox package (e.g. in unit tests that exercise path
# translation without spinning Docker).
_CONTAINER_WORKDIR = "/workspace"


def _to_container_path(sandbox_dir: Path, host_path: Path) -> str:
    """Translate a HOST absolute path into the path the same file has
    INSIDE the sandbox container.

    The container bind-mounts ``sandbox_dir`` at ``/workspace``, so a
    host file at ``<sandbox_dir>/foo/bar.py`` is visible at
    ``/workspace/foo/bar.py`` from inside the container — and ONLY at
    that path, since the host filesystem is otherwise opaque to the
    container.

    Used by sandbox-exec callers (``rg``, ``find``, ...) that take a
    path argument: passing the host-absolute path would make those
    tools report "no matches" silently because the container can't see
    that path. The bug was load-bearing in a 2026-04-26 webOS session
    where six consecutive ``rg`` searches all returned empty against a
    file the agent had just successfully edited.

    Pre-condition: ``host_path`` already passed ``_get_safe_path``
    (i.e. is rooted under ``sandbox_dir``). Falls back to a simple
    ``.`` ("workspace root") when relative_to() raises, so a
    pathological caller still gets a usable command rather than an
    exception.
    """
    # The bind mount is fixed at the sandbox ROOT, even when a project is
    # active and file ops are scoped to <root>/projects/<id>. So the
    # container path must be computed relative to the ROOT, not the scoped
    # sub-dir — otherwise a file at <root>/projects/<id>/foo.py would be
    # reported as /workspace/foo.py when it actually lives at
    # /workspace/projects/<id>/foo.py. Un-scope before translating.
    root = sandbox_dir
    if sandbox_dir.parent.name == "projects":
        root = sandbox_dir.parent.parent
    try:
        rel = host_path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        # Should never happen post-_get_safe_path, but be defensive
        # rather than emit a malformed command.
        return _CONTAINER_WORKDIR
    rel_str = rel.as_posix()
    if rel_str in ("", "."):
        return _CONTAINER_WORKDIR
    return f"{_CONTAINER_WORKDIR}/{rel_str}"

def _missing_file_message(filename, sandbox_dir) -> str:
    """Loop-breaking 'not found' message for a missing read target.

    The bare ``Error: '<f>' not found.`` caused a documented infinite
    loop: when stale cross-project context (a workspace narrative about a
    PRIOR project's files, a memory hint, or a DYNAMIC SYSTEM STATE line)
    asserted a file existed, the model re-read the same missing path every
    turn — the tool said "no", the injected state said "yes", and nothing
    reconciled the two. This message resolves the contradiction in the
    model's favour-of-reality: the live sandbox is authoritative, lists
    what actually exists, and gives a concrete exit (create it, or pick a
    real file). Best-effort directory scan; never raises."""
    existing = []
    try:
        import os
        for root, dirs, files in os.walk(sandbox_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("__pycache__", "node_modules", "venv", "env")]
            for f in sorted(files):
                if f.startswith("."):
                    continue
                existing.append(os.path.relpath(os.path.join(root, f), sandbox_dir))
                if len(existing) >= 20:
                    break
            if len(existing) >= 20:
                break
    except Exception:
        pass
    if existing:
        listing = "Files that DO exist here: " + ", ".join(existing[:20]) + "."
    else:
        listing = "This project's sandbox is currently EMPTY (no files yet)."
    return (
        f"Error: '{filename}' does not exist in the current project's sandbox. "
        f"{listing} The live sandbox is AUTHORITATIVE — if a workspace narrative, "
        f"memory, prior session, or DYNAMIC SYSTEM STATE hint referenced "
        f"'{filename}', that was a DIFFERENT project/session and does NOT apply "
        f"here. Do NOT read this path again (it will keep failing). To proceed: "
        f"CREATE it with file_system(operation='write', filename='{filename}', "
        f"content=…), or pick a file from the list above."
    )


def read_byte_budget(max_context: int) -> int:
    """Bytes of raw file content allowed into the model's context — used for
    BOTH the per-file read cap and the per-batch cumulative allowance.

    Sized to a fraction of the window (chars ≈ tokens * 3.5) so raw reads
    can't crowd out the system prompt, the model's reasoning, the other tool
    outputs in the turn, and the response itself. The old per-file factor was
    0.5 (≈70% of the window for a single file); two such reads in one turn
    overflowed a 131 K window at 136 K tokens. 0.40 leaves headroom and, paired
    with the cumulative ReadBudget, makes parallel whole-file reads safe."""
    return max(150000, int(max_context * 3.5 * 0.40))


class ReadBudget:
    """Per-batch cumulative cap on bytes pulled into context by raw file reads.

    A single read may clear the per-file limit yet, combined with other reads
    dispatched from the SAME assistant message, overflow the model's context
    window — observed live: two 170+ KB experiment JSONs read in parallel
    produced a 136 K-token request against a 131 K window, a hard HTTP 400 that
    then crashed emergency recovery. The budget refuses the read that would
    breach the batch allowance and steers the model to read_chunked / search /
    execute instead of silently overflowing.

    Scope is one batch of tool calls (the agent resets it before dispatching
    each assistant message's tool calls); cross-turn accumulation is handled
    separately by context compaction, so it deliberately does not persist."""
    __slots__ = ("limit", "spent")

    def __init__(self, limit_bytes: int):
        self.limit = max(0, int(limit_bytes))
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def charge(self, n_bytes: int) -> None:
        self.spent += max(0, int(n_bytes))


async def tool_read_file(filename: str, sandbox_dir: Path, max_context: int = 8192,
                         read_budget: "ReadBudget | None" = None,
                         start_line: int = None, end_line: int = None):
    if start_line is not None or end_line is not None:
        pretty_log("File Read", f"{filename} [{start_line or 1}:{end_line or 'end'}]",
                   icon=Icons.TOOL_FILE_R)
    else:
        pretty_log("File Read", filename, icon=Icons.TOOL_FILE_R)
    # GUARD 1: Stop model from trying to read URLs as files. The error
    # surface PRIMES the model's next tool choice, so we lead with `browser`
    # (the right answer when the user says "open this page" or "what's on
    # example.com") and only mention `knowledge_base` as the path for
    # ingesting a document into long-term memory. The previous one-line
    # advice misrouted every "use the browser tool to open URL" request to
    # `knowledge_base(ingest_document, url=...)` because the error didn't
    # mention `browser` at all.
    if str(filename).startswith("http"):
        return (
            f"Error: file_system cannot read URLs. "
            f"To VIEW a webpage right now, use the `browser` tool "
            f"(operation='navigate' or 'extract_text', url='{filename}'). "
            f"To INGEST a document URL into long-term memory, use "
            f"`knowledge_base(action='ingest_document', filename='{filename}')`. "
            f"For a one-off page read pick `browser` — that's almost always what you want."
        )

    # GUARD 2: PDF files must be handled by the knowledge base
    if str(filename).lower().endswith(".pdf"):
        return f"Error: '{filename}' is a PDF. You cannot use read_file on PDFs. To permanently index it into your vector memory, use knowledge_base(action='ingest_document', filename='{filename}'). To just read a specific page into your immediate context, use file_system(operation='read_chunked', path='{filename}', page=1)."

    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists():
            return _missing_file_message(filename, sandbox_dir)

        # LINE-RANGE READ (#11): a bounded slice, EXEMPT from the whole-file
        # size cap — this is the cheap recovery path after a failed replace or
        # a too-large-file error. The range is streamed (we stop reading at
        # end_line) so a 300 KB file costs only the requested lines, and the
        # slice is returned with line-number prefixes so it chains directly
        # from `rg --line-number` output and the replace-failure snippet.
        if start_line is not None or end_line is not None:
            return await asyncio.to_thread(
                _read_line_range, path, filename, start_line, end_line)

        file_size = path.stat().st_size
        max_bytes = read_byte_budget(max_context)
        if file_size > max_bytes: # dynamic limit for raw reads
            return f"Error: File '{filename}' is too large to read entirely ({file_size / 1024:.1f} KB) into your chat context window. Limit is {max_bytes / 1024:.1f} KB. Note: This limit only applies to a WHOLE-file 'read'. Best option: read only the region you need — file_system(operation='read', path='{filename}', start_line=200, end_line=260) (line-ranged, exempt from this cap, chains directly from 'search' line numbers). Also: operation='read_chunked' to page through it, operation='search' to find specific lines, operation='inspect' for the first few lines, knowledge_base(action='ingest_document') (NO size limit) to index it, or a Python script via 'execute'."

        # Per-BATCH cumulative budget: even when THIS file clears the per-file
        # cap above, earlier reads dispatched from the SAME assistant message may
        # have already consumed most of the window. Refuse the read that would
        # tip the batch over (the FIRST read always proceeds — `spent == 0` —
        # since the per-file cap already bounds it). This is the guard that
        # stops parallel whole-file reads from overflowing: each passes alone,
        # together they don't fit.
        if (read_budget is not None and read_budget.spent > 0
                and file_size > read_budget.remaining):
            return (
                f"Error: Reading '{filename}' ({file_size / 1024:.1f} KB) now would "
                f"overflow the context window. {read_budget.spent / 1024:.1f} KB of file "
                f"data was already read into THIS turn and only "
                f"{read_budget.remaining / 1024:.1f} KB of the read budget remains. "
                f"Do NOT read more whole files this turn. Instead: process the data with "
                f"a Python script via the 'execute' tool (load the file, compute a compact "
                f"summary, print ~50 lines), or use file_system(operation='read_chunked', "
                f"filename='{filename}') / operation='search' / operation='inspect' to pull "
                f"only the parts you actually need."
            )

        # Sniff the first 8 KB for binary signatures BEFORE attempting the
        # full read. If we used `errors='replace'` directly on a binary file
        # we'd return garbled �-laced output to the model; conversely a pure
        # `read_text` aborts the whole read on a single non-UTF-8 byte. The
        # sniff lets us pick the right branch.
        try:
            head = await asyncio.to_thread(_read_head, path, 8192)
        except OSError as oe:
            return f"Error: failed to read '{filename}': {oe}"
        if _looks_like_binary(head):
            return f"Error: '{filename}' appears to be a binary file. You cannot read it as text. If it is an image, use the 'vision_analysis' tool."

        try:
            # `errors='replace'` keeps content readable when a TEXT file has
            # the occasional non-UTF-8 byte (mixed encodings, Windows source,
            # log-with-binary-noise) instead of aborting the entire read.
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, LookupError):
            return f"Error: '{filename}' appears to be a binary file. You cannot read it as text. If it is an image, use the 'vision_analysis' tool."
        except OSError as oe:
            return f"Error: failed to read '{filename}': {oe}"
        # Charge the batch budget by what we actually pulled in, so the NEXT
        # read in this same turn sees a smaller remaining allowance.
        if read_budget is not None:
            read_budget.charge(len(content))
        return f"--- {filename} CONTENTS ---\n{content}"
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_replace_text(filename: str, old_text: str, new_text: str, sandbox_dir: Path):
    pretty_log("File Replace", filename, icon=Icons.TOOL_FILE_W)
    if not old_text: return "Error: You must specify the exact 'content' to be replaced."

    has_aider_blocks = "<<<< SEARCH" in str(old_text)

    # IDENTICAL-ARGS CORRUPTION GUARD (2026-07-05). Measured over 5 days of
    # trajectories: 50 of 99 replace calls arrived with content ==
    # replace_with byte-identical — 27 of the 34 "search block NOT found"
    # failures, plus silent no-op "successes" where the identical text still
    # matched the file and was replaced with itself (the model then believed
    # a fix was applied that never happened). Cause is upstream tool-call
    # argument transport (the same native-tools parser that merged
    # multi-tool replies into one call's args — see the introspect
    # incident), so no matcher improvement can help: the search text is
    # usually the NEW block, which does not exist in the file yet.
    # Replacing a block with itself is never intentional → reject fast and
    # teach the single-argument SEARCH/REPLACE form, which cannot be
    # cross-merged.
    if (not has_aider_blocks and new_text is not None
            and str(old_text) == str(new_text)):
        pretty_log(
            "File Replace Corruption Guard",
            f"content == replace_with ({len(str(old_text))} chars) on "
            f"'{filename}' — suspected upstream tool-call argument "
            f"corruption; rejected before any file change",
            level="WARNING",
            icon=Icons.WARN,
        )
        return (
            "SYSTEM INSTRUCTION: REPLACE REJECTED — your 'content' and "
            f"'replace_with' arguments arrived BYTE-IDENTICAL "
            f"({len(str(old_text))} chars). Replacing a block with itself is "
            "always a mistake; this usually means the tool-call transport "
            "merged your two arguments, and the text that arrived is "
            "probably your NEW version (the old block was lost in transit). "
            f"The file '{filename}' was NOT modified. Do NOT resend the same "
            "two-argument call. Re-issue ONE file_system call with "
            "operation='replace', NO 'replace_with' argument, and 'content' "
            "containing exactly this structure:\n"
            "<<<< SEARCH\n"
            "<the exact CURRENT text from the file — re-read the file if "
            "unsure>\n"
            "====\n"
            "<your NEW text>\n"
            ">>>>\n"
            "This single-argument form is immune to the argument-merge "
            "corruption."
        )

    if not has_aider_blocks and new_text is None:
        # Auto-promote path: if the caller forgot `replace_with` but their
        # `content` is a complete, parseable Python module, they almost
        # always meant operation='write' (the most common LLM misfire
        # observed in self-play logs). Overwrite the file with `old_text`
        # as the new content and return a SUCCESS with a loud warning so
        # the model learns the correct shape for next time. Only triggers
        # for .py files — other extensions get the original error.
        ext = str(filename).split('.')[-1].lower()
        if ext == "py" and _looks_like_complete_python_module(str(old_text)):
            try:
                path = _get_safe_path(sandbox_dir, filename)
                # Only strip markdown fences if they're actually present;
                # otherwise preserve `old_text` byte-for-byte (trailing
                # newlines, indentation, everything). The sanitizer's
                # unconditional `.strip()` would drop the trailing newline
                # which matters for clean file writes.
                if "```" in str(old_text):
                    from ..utils.sanitizer import extract_code_from_markdown
                    clean_content = extract_code_from_markdown(str(old_text))
                else:
                    clean_content = str(old_text)
                await asyncio.to_thread(path.write_text, clean_content, encoding="utf-8")
                pretty_log(
                    "File Replace Auto-Promote",
                    f"replace→write on '{filename}' (content looked like a complete module)",
                    level="WARNING",
                    icon=Icons.WARN,
                )
                return (
                    f"SUCCESS: auto-promoted operation='replace' to 'write' for "
                    f"'{filename}' because your 'content' was a complete Python "
                    f"module and 'replace_with' was missing. The file has been "
                    f"overwritten. Next time, use operation='write' directly "
                    f"when you intend to rewrite the whole file."
                    + await _syntax_feedback(path, filename)
                )
            except Exception as e:
                return (
                    f"SYSTEM INSTRUCTION: Attempted to auto-promote replace→write "
                    f"but the write failed: {e}. Retry with operation='write'."
                )
        return "SYSTEM INSTRUCTION: You used operation='replace' but forgot to specify 'replace_with'. If you want to rewrite the entire file, use operation='write'. Otherwise, provide 'replace_with'."
        
    ext = str(filename).split('.')[-1].lower()
    if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
        from ..utils.sanitizer import extract_code_from_markdown
        # Only un-fence when fences are ACTUALLY present. The extractor's
        # no-fence fallback `.strip()`s the text, which dedents the FIRST
        # line of an indented replacement block (and only the first) —
        # corrupting the block's relative indentation before the matcher
        # re-anchors it, which then lands continuation lines at the wrong
        # column. Same guard the replace→write auto-promote path already
        # uses. When fences are absent, preserve old_text/new_text
        # byte-for-byte (indentation included).
        if "```" in str(old_text):
            old_text = extract_code_from_markdown(str(old_text))
        if new_text is not None and "```" in str(new_text):
            new_text = extract_code_from_markdown(str(new_text))
            
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."

        try:
            file_size = path.stat().st_size
        except OSError as se:
            return f"Error: failed to stat '{filename}': {se}"
        REPLACE_MAX_BYTES = 50 * 1024 * 1024
        STREAMING_THRESHOLD = 1 * 1024 * 1024  # 1 MB

        if file_size > REPLACE_MAX_BYTES:
            return f"Error: '{filename}' is {file_size // (1024*1024)} MB; the 'replace' operation refuses files larger than {REPLACE_MAX_BYTES // (1024*1024)} MB. Use 'execute' with a Python streaming script instead."

        # For large files (>1 MB), use streaming line-by-line replace
        # to avoid loading the entire file into memory at once.
        # Skip streaming when the search spans multiple lines — the
        # line-by-line loop can never match a block that crosses a newline,
        # so we'd silently return 0 replacements instead of falling back
        # to the full-file heuristic. For multi-line replacements we have
        # no choice but to read the whole file.
        old_text_has_newline = "\n" in str(old_text)
        if file_size > STREAMING_THRESHOLD and not has_aider_blocks and not old_text_has_newline:
            try:
                import tempfile
                def _streaming_replace():
                    replaced = 0
                    tmp_path = None
                    try:
                        # errors="replace" for the same tolerance as the
                        # non-streaming path above (bad-byte write-back is a
                        # deferred finding, not fixed here).
                        with open(path, 'r', encoding='utf-8', errors='replace') as f_in:
                            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                                            suffix='.tmp', delete=False,
                                                            encoding='utf-8') as f_out:
                                tmp_path = Path(f_out.name)
                                for line in f_in:
                                    if old_text in line:
                                        line = line.replace(old_text, new_text)
                                        replaced += 1
                                    f_out.write(line)
                        if replaced > 0:
                            import os
                            os.replace(tmp_path, path)
                            tmp_path = None  # consumed by the rename
                        return replaced
                    finally:
                        # Drop the temp file if it wasn't consumed by the
                        # rename (no matches, or an exception mid-stream) —
                        # otherwise failed/no-op replaces leak .tmp orphans.
                        if tmp_path is not None:
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                replaced = await asyncio.to_thread(_streaming_replace)
                if replaced > 0:
                    return f"SUCCESS: Streaming replace applied to '{filename}' ({replaced} line(s) modified)."
                # Fall through to heuristic match below
            except Exception as e:
                return f"Error: Streaming replace failed: {e}"

        try:
            # NOTE: errors="replace" is deliberately tolerant so a mostly-text
            # file with a few stray bad bytes can still be edited. The known
            # downside (untouched bad bytes are persisted as U+FFFD on
            # write-back) is tracked as a deferred finding in BUGHUNT.md; the
            # correct fix is a surrogateescape round-trip through the shared
            # write path, which is out of scope for this pass.
            file_content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, LookupError):
            return f"Error: '{filename}' appears to be a binary file and cannot be text-replaced."
        except OSError as oe:
            return f"Error: failed to read '{filename}' for replace: {oe}"
        
        if has_aider_blocks:
            import re
            # Indentation-preserving parse FIRST: capture block bodies
            # starting on the line AFTER each marker, so the first line's
            # leading whitespace survives. The loose fallback's `\s*` used
            # to swallow it, which (a) made exact matches impossible for
            # indented blocks and (b) fed _reindent_replacement a new_text
            # whose first line was dedented, mis-computing its indent base
            # — the fuzzy/anchor rungs then inserted a broken first line.
            blocks = re.findall(
                r'<<<<[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n====[ \t]*\r?\n(.*?)\r?\n?>>>>',
                str(old_text), re.DOTALL)
            if not blocks:
                # Loose fallback: tolerates content on the marker lines and
                # missing newlines, at the cost of stripping the blocks'
                # leading/trailing whitespace.
                blocks = re.findall(r'(?s)<<<<\s*SEARCH\s*(.*?)\s*====\s*(.*?)\s*>>>>', str(old_text))

            if not blocks:
                return "SYSTEM INSTRUCTION: Found SEARCH/REPLACE markers but failed to parse them. Ensure you use <<<< SEARCH, ====, and >>>> correctly."
            
            success_count = 0
            errors = []
            strategies = []
            prev_content = file_content

            for search_str, replace_str in blocks:
                # Same corruption signature as the two-argument form: a
                # block that "replaces" text with itself is a no-op and
                # means the old block was lost in transit.
                if search_str.strip() and search_str == replace_str:
                    errors.append(
                        "Block is a NO-OP (SEARCH == REPLACE) — the file "
                        "would be unchanged. Re-emit it with the exact "
                        "CURRENT text in SEARCH and your NEW text after "
                        "====.")
                    continue
                # Full matching ladder (exact → whitespace-flexible →
                # fuzzy → anchor), same rescue chain the two-argument form
                # gets — the SEARCH-block form is the one we now steer the
                # model toward, so it must not be the weaker matcher.
                located = _locate_block(file_content, search_str)
                if located is None:
                    errors.append(f"Could not find block:\n{search_str[:50]}...")
                    continue
                matched_text, strategy = located
                if strategy == "flexible":
                    # Re-anchor the replacement's indentation to the matched
                    # region so the whitespace-flexible match doesn't corrupt
                    # block indentation.
                    replacement = _reindent_replacement(
                        file_content, matched_text, replace_str)
                else:
                    replacement = replace_str
                    # Fuzzy/anchor windows are sliced with keepends; preserve
                    # a trailing newline the model's replacement omits, else
                    # the following line is welded onto the edited one.
                    if (matched_text.endswith("\n")
                            and not replacement.endswith("\n")):
                        replacement += "\n"
                file_content = file_content.replace(
                    matched_text, replacement, 1)
                success_count += 1
                strategies.append(strategy)

            if success_count > 0:
                msg = f"SUCCESS: Applied {success_count} SEARCH/REPLACE blocks to '{filename}'."
                non_exact = [s for s in strategies if s != "exact"]
                if non_exact:
                    msg += (" NOTE: " + str(len(non_exact)) + " block(s) did "
                            "not byte-match and were rescued by tolerant "
                            "matching (" + ", ".join(non_exact) + ") — "
                            "VERIFY the result is what you intended.")
                if errors:
                    msg += f" SYSTEM INSTRUCTION: {len(errors)} blocks failed:\n" + "\n".join(errors)
                return await _write_replace_guarded(path, prev_content, file_content, filename, msg)
            else:
                return f"SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in '{filename}'.\n" + "\n".join(errors)

        # 1. Exact match attempt
        if old_text in file_content:
            occurrences = file_content.count(old_text)
            new_file_content = file_content.replace(old_text, new_text)
            msg = f"SUCCESS: Exact match found and replaced in '{filename}'."
            if occurrences > 1: msg += f" WARNING: Replaced {occurrences} identical occurrences."
            return await _write_replace_guarded(path, file_content, new_file_content, filename, msg)

        # 2. Heuristic match (ignore arbitrary whitespace & newlines). The
        # match starts at the first token, so re-anchor the replacement's
        # indentation to the matched region before substituting (otherwise a
        # multi-line block's continuation lines land at the wrong column).
        import re
        words = [re.escape(w) for w in str(old_text).split()]
        flexible_old = r'\s+'.join(words)

        matches = re.findall(flexible_old, file_content)
        if len(matches) == 1:
            reindented = _reindent_replacement(file_content, matches[0], str(new_text))
            # Replace only the FIRST occurrence (count=1), matching the
            # exact/fuzzy/anchor/aider paths — an unbounded replace would
            # clobber every later copy of an identical block.
            new_file_content = file_content.replace(matches[0], reindented, 1)
            return await _write_replace_guarded(
                path, file_content, new_file_content, filename,
                f"SUCCESS: Flexible match found and replaced in '{filename}'.")
        elif len(matches) > 1:
            return "SYSTEM INSTRUCTION: Multiple instances of this text block found. Please provide a larger, more unique block of code in 'content' to ensure we replace the correct one."

        # 2.5 Fuzzy contiguous-block match. The flexible matcher above
        # only forgives whitespace; a single misremembered character or a
        # stray/missing token makes it miss, and the model's documented
        # next move is to rewrite the WHOLE file (a 200s+ regeneration of a
        # large index.html, and a fresh chance to introduce new bugs). A
        # high-confidence, UNIQUE difflib window match rescues exactly that
        # case surgically. Conservative on purpose: ratio >= 0.92 AND a
        # clear margin over the runner-up, so we never silently patch the
        # wrong region.
        fuzzy = _fuzzy_block_match(file_content, str(old_text))
        if fuzzy is not None:
            matched_text, ratio = fuzzy
            replacement = str(new_text)
            # The matched block was sliced with keepends, so it may carry a
            # trailing newline the model's replacement omits. Preserve it,
            # else a single-line fuzzy replace would weld the following line
            # onto the edited one.
            if matched_text.endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            new_file_content = file_content.replace(matched_text, replacement, 1)
            res = await _write_replace_guarded(
                path, file_content, new_file_content, filename,
                f"SUCCESS: Fuzzy match ({ratio:.0%} similar) found and replaced "
                f"in '{filename}'. Your `old_text` did not byte-match, but a single "
                f"near-identical block was unambiguous.")
            if res.startswith("SUCCESS"):
                res += (f" VERIFY the change is what you "
                        f"intended:\n--- REPLACED BLOCK (was) ---\n{matched_text[:600]}")
            return res

        # 2.7 Anchor-block match. When the block's BOUNDARIES are stable but
        # its middle drifted (the model remembers the signature + shape but
        # mis-transcribes the body), fuzzy's whole-block ratio falls below
        # threshold and misses. Anchoring on the unique first+last lines (or
        # a brace-balanced block from a unique signature) rescues exactly the
        # live `addFace` case that otherwise thrashed for minutes and forced
        # a full-file rewrite. Safe: both anchors must be unique and the span
        # is capped relative to the target size.
        anchor = _anchor_block_match(file_content, str(old_text))
        if anchor is not None:
            matched_text, info = anchor
            replacement = str(new_text)
            if matched_text.endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            new_file_content = file_content.replace(matched_text, replacement, 1)
            res = await _write_replace_guarded(
                path, file_content, new_file_content, filename,
                f"SUCCESS: Anchor match — replaced the block spanning lines "
                f"{info['start_line']}–{info['end_line']} in '{filename}' "
                f"(matched on its unique {'first+last lines' if info['strategy']=='first_last' else 'signature + balanced braces'}; "
                f"the middle differed from your old_text).")
            if res.startswith("SUCCESS"):
                res += (f" VERIFY the change:\n"
                        f"--- REPLACED BLOCK (was) ---\n{matched_text[:600]}")
            return res

        # 3. Neither exact nor flexible nor fuzzy nor anchor match. Return a
        # snippet of the file at the closest-looking neighborhood so the
        # model can see what the text actually is. Common cause: the model's
        # remembered `old_text` has slightly different whitespace,
        # indentation, or a missing/extra line relative to what is on
        # disk — the flexible-whitespace matcher handles most cases
        # but multi-line blocks with decorators / docstrings / nested
        # classes are brittle. Historical note: a blanket `black`
        # auto-formatter used to run after writes and the model would
        # blame reformatting for every failure, but that path is gone
        # — so do NOT tell the model the file was reformatted (false
        # hint that sends it into a debugging spiral).
        snippet = _nearest_snippet(file_content, str(old_text))
        # Scale the recovery advice to file size. A full rewrite is fine
        # for a small file but catastrophic for a large one: regenerating
        # a 400+ line index.html costs minutes of model output AND risks
        # introducing brand-new bugs, which is exactly the 200s-per-edit
        # spiral seen in production. For large files, forbid the rewrite
        # and force a surgical single-line replace anchored on the exact
        # current text we just handed back (with line numbers).
        line_count = file_content.count("\n") + 1
        is_large = file_size > 16 * 1024 or line_count > 250
        if is_large:
            primary = (
                f"  1. This file is LARGE ({line_count} lines). DO NOT use "
                "operation='write' to rewrite it — a full regeneration is slow "
                "and routinely introduces NEW bugs. Instead, copy the exact "
                "current text from the CLOSEST MATCH below (line numbers shown) "
                "and emit a tight single-line SEARCH block for the surgical edit.\n"
                "  2. If the snippet below isn't enough, re-read ONLY that "
                f"region — file_system(operation='read', path='{filename}', "
                "start_line=<N>, end_line=<M>) — do NOT re-read the whole file.\n"
                "  3. Only if the edit genuinely spans many lines, replace each "
                "line as its own one-line SEARCH/REPLACE rather than rewriting "
                "the file.\n"
            )
        else:
            primary = (
                "  1. If the edit is >3 lines or touches a block with "
                "decorators/docstrings/nested classes, use operation='write' "
                f"to overwrite this small file ({line_count} lines) — that's "
                "byte-exact and eliminates all matching issues.\n"
                "  2. Otherwise, READ the file first to get the current exact "
                "text, then emit a tighter SEARCH block (ideally a single line) "
                "for the surgical edit.\n"
            )
        return (
            "SYSTEM INSTRUCTION: The search block was NOT found in '"
            + filename
            + "'. Your remembered `old_text` does NOT byte-match the "
            "current file — either (a) your SEARCH block has an off-by-"
            "one whitespace/indentation issue, (b) your SEARCH block "
            "spans >3 lines and the flexible matcher can't resolve it "
            "uniquely, or (c) the file changed since you last read it. "
            "DO NOT retry replace with the same old_text. Options in "
            "priority order:\n"
            + primary +
            "  3. If two replace attempts have already failed on this "
            "file, STOP retrying the SAME old_text — copy the exact text from "
            "the snippet below instead. Do not loop.\n"
            "CLOSEST MATCH IN THE FILE (with line numbers):\n" + snippet
        )
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

def _locate_block(file_content: str, search_str: str):
    """Full matching ladder shared by the SEARCH/REPLACE-block loop:
    exact → whitespace-flexible (unique) → fuzzy window → anchor block.

    Returns ``(matched_text, strategy)`` where ``matched_text`` is the
    ORIGINAL bytes in the file to replace and ``strategy`` is one of
    ``"exact"``, ``"flexible"``, ``"fuzzy:NN%"``, ``"anchor:L1-L2"`` — or
    ``None`` when no matcher produces a unique, high-confidence hit."""
    if search_str in file_content:
        return search_str, "exact"
    words = [re.escape(w) for w in str(search_str).split()]
    if words:
        flexible = r'\s+'.join(words)
        matches = re.findall(flexible, file_content)
        if len(matches) == 1:
            return matches[0], "flexible"
    fuzzy = _fuzzy_block_match(file_content, str(search_str))
    if fuzzy is not None:
        return fuzzy[0], f"fuzzy:{fuzzy[1]:.0%}"
    anchor = _anchor_block_match(file_content, str(search_str))
    if anchor is not None:
        info = anchor[1]
        return anchor[0], f"anchor:{info['start_line']}-{info['end_line']}"
    return None


def _nearest_snippet(file_content: str, target: str,
                     context_lines: int = 3,
                     max_len: int = 800) -> str:
    """Return a short block of lines around the file's best textual
    overlap with ``target``.

    We tokenize both the file (line by line) and the target, then
    score each line by the count of shared non-trivial tokens. The
    best-scoring line is the anchor; we return it plus ``context_lines``
    lines of surrounding context. When nothing overlaps meaningfully,
    return the first few lines as a fallback so the model still sees
    what the file looks like.
    """
    if not file_content:
        return "(file is empty)"
    lines = file_content.splitlines()
    target_tokens = {
        t for t in re.findall(r"\w+", target.lower()) if len(t) > 2
    }
    best_idx = 0
    best_score = -1
    if target_tokens:
        for i, line in enumerate(lines):
            line_tokens = {
                t for t in re.findall(r"\w+", line.lower()) if len(t) > 2
            }
            score = len(line_tokens & target_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
    start = max(0, best_idx - context_lines)
    end = min(len(lines), best_idx + context_lines + 1)
    rendered_lines = []
    for i in range(start, end):
        marker = ">>>" if i == best_idx and best_score > 0 else "   "
        rendered_lines.append(f"{marker} {i+1:4d}: {lines[i]}")
    out = "\n".join(rendered_lines)
    if len(out) > max_len:
        out = out[:max_len] + "\n   ... [truncated]"
    return out


def _fuzzy_block_match(file_content: str, target: str,
                       min_ratio: float = 0.92,
                       margin: float = 0.04,
                       max_target_chars: int = 4000):
    """Find a UNIQUE contiguous line-block in ``file_content`` that is a
    near-identical match to ``target`` once per-line whitespace is ignored.

    Returns ``(matched_text, ratio)`` — ``matched_text`` is the ORIGINAL
    (un-normalised) bytes to replace — or ``None`` when there is no high-
    confidence, unambiguous match. Runs only AFTER the exact and flexible-
    whitespace matchers miss; it rescues a single misremembered character
    or stray token surgically instead of letting the model fall back to a
    slow, bug-prone full-file rewrite.

    Deliberately conservative: the best window must clear ``min_ratio`` AND
    beat the runner-up by ``margin``, so an ambiguous edit never silently
    lands in the wrong place. Cost is O(file_lines) difflib calls for the
    common single-line target (``quick_ratio`` prefilters before the exact
    ``ratio``); bounded by the window size for multi-line targets. Very
    large targets are skipped to keep the cost predictable.
    """
    import difflib
    if not target or not file_content:
        return None
    if len(target) > max_target_chars:
        return None
    target_lines = target.splitlines()
    n = len(target_lines)
    if n == 0:
        return None
    norm_target = "\n".join(line.strip() for line in target_lines)
    if not norm_target.strip():
        return None  # all-whitespace target is too ambiguous to anchor
    file_lines = file_content.splitlines(keepends=True)
    if n > len(file_lines):
        return None
    best_ratio = -1.0
    second_ratio = -1.0
    best_idx = -1
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(norm_target)
    for i in range(0, len(file_lines) - n + 1):
        norm_w = "\n".join(line.strip() for line in file_lines[i:i + n])
        sm.set_seq1(norm_w)
        # quick_ratio is a cheap upper bound on ratio; skip the exact
        # (more expensive) computation for windows that can't qualify.
        if sm.quick_ratio() < min_ratio:
            continue
        ratio = sm.ratio()
        if ratio > best_ratio:
            second_ratio = best_ratio
            best_ratio = ratio
            best_idx = i
        elif ratio > second_ratio:
            second_ratio = ratio
    if best_idx < 0 or best_ratio < min_ratio:
        return None
    if best_ratio - second_ratio < margin:
        return None  # two windows nearly equally good — refuse to guess
    matched_text = "".join(file_lines[best_idx:best_idx + n])
    return matched_text, best_ratio


def _anchor_block_match(file_content: str, target: str,
                        min_anchor_len: int = 10, max_span_factor: int = 3):
    """Locate a block whose BOUNDARIES are stable even though its middle
    drifted from ``target`` — the exact failure mode that made the live
    ``addFace`` edit thrash for ~4 minutes and then force a full-file
    rewrite. The model remembers the method signature and the overall shape
    but mis-transcribes the body, so exact/flex/fuzzy all miss.

    Two safe-by-construction strategies, tried in order:

      1. **first+last anchor** — the first AND last non-blank lines of
         ``target`` each occur EXACTLY ONCE in the file; replace the exact
         file bytes spanning them.
      2. **brace-balanced anchor** — ``target`` opens a ``{ … }`` block and
         its first non-blank line is unique; replace from that line through
         the matching closing brace (balanced count).

    Both require a UNIQUE anchor and bound the matched span to
    ``max_span_factor`` × the target's line count, so a bad anchor can never
    silently swallow a huge region. Returns ``(matched_text, info)`` or
    ``None``.
    """
    if not target or not file_content:
        return None
    tgt_nonblank = [ln for ln in target.splitlines() if ln.strip()]
    if len(tgt_nonblank) < 2:
        return None
    target_n = target.count("\n") + 1
    span_cap = max(target_n * max_span_factor, target_n + 8)
    file_lines = file_content.splitlines(keepends=True)
    stripped = [ln.strip() for ln in file_lines]

    # Strategy 1: unique first + last non-blank line.
    first, last = tgt_nonblank[0].strip(), tgt_nonblank[-1].strip()
    if len(first) >= min_anchor_len and len(last) >= min_anchor_len:
        starts = [i for i, s in enumerate(stripped) if s == first]
        ends = [i for i, s in enumerate(stripped) if s == last]
        if len(starts) == 1 and len(ends) == 1 and ends[0] >= starts[0]:
            span = ends[0] - starts[0] + 1
            if span <= span_cap:
                matched = "".join(file_lines[starts[0]:ends[0] + 1])
                return matched, {"strategy": "first_last",
                                 "start_line": starts[0] + 1,
                                 "end_line": ends[0] + 1}

    # Strategy 2: brace-balanced block from a unique block-opener line.
    # Anchor on the FIRST line of `target` that ends with "{" (a block
    # opener like a function/method signature) AND occurs exactly once in
    # the file — not merely target's first non-blank line, which may be a
    # comment or leading context the model included. Brace-balancing is
    # self-validating, so a shorter unique opener is safe.
    if "{" in target:
        opener = None
        for ln in tgt_nonblank:
            s = ln.strip()
            if s.endswith("{") and len(s) >= 8 and stripped.count(s) == 1:
                opener = s
                break
        if opener is not None:
            start = stripped.index(opener)
            depth, started, end = 0, False, None
            for i in range(start, len(file_lines)):
                for ch in file_lines[i]:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if started and depth <= 0:
                    end = i
                    break
            if end is not None and end >= start and (end - start + 1) <= span_cap:
                matched = "".join(file_lines[start:end + 1])
                return matched, {"strategy": "brace",
                                 "start_line": start + 1,
                                 "end_line": end + 1}
    return None


_DATA_FILE_EXTS = frozenset({
    "log", "csv", "tsv", "txt", "jsonl", "ndjson",
})


def _fixture_summary(content: str, ext: str) -> str:
    """Return a one-line summary of record counts for self-generated
    fixture files.

    Purpose: when the agent writes a test fixture and then writes an
    assertion like ``assert count == 4``, the number is usually wrong
    because it was guessed from the model's loose recollection of what
    it just generated. Reporting a concrete count in the write
    response gives the next turn a ground-truth anchor to cite.
    Only runs for obvious data formats — source files don't need it.
    """
    if ext not in _DATA_FILE_EXTS:
        return ""
    if not content:
        return ""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    parts: list = [f"{len(lines)} non-empty lines"]
    if ext in {"jsonl", "ndjson"}:
        parts.append(f"{len(lines)} JSON records")
    return " | FIXTURE-COUNT: " + ", ".join(parts)


_SYNTAX_CHECK_TIMEOUT_S = 10.0

# <script>…</script> blocks; group(1)=attributes, group(2)=body.
_HTML_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
# Inline script types we can syntax-check as classic JS. A `type=module`
# (import/export, top-level await) or `text/babel` (JSX) block would trip
# `node --check` with a FALSE positive, so we skip those rather than send
# the model chasing a non-bug.
_CHECKABLE_JS_TYPES = {"", "text/javascript", "application/javascript"}


def _inline_js_blocks(html: str):
    """Yield (start_line, source) for each inline, classic-JS <script> block.

    start_line is the 1-based line in the HTML file where the block body
    begins, so a parser error reported at body-line N maps back to the HTML
    line the model actually has to edit. External (`src=`) and non-JS-typed
    blocks are skipped.
    """
    blocks = []
    for m in _HTML_SCRIPT_RE.finditer(html):
        attrs = (m.group(1) or "").lower()
        if "src=" in attrs:
            continue
        tm = re.search(r'type\s*=\s*["\']?([^"\'\s>]+)', attrs)
        if tm and tm.group(1) not in _CHECKABLE_JS_TYPES:
            continue
        body = m.group(2)
        if not body.strip():
            continue
        start_line = html.count("\n", 0, m.start(2)) + 1
        blocks.append((start_line, body))
    return blocks


async def _node_check_source(source: str) -> str:
    """`node --check` a JS source string. Returns node's raw diagnostic
    (stderr) when it does NOT parse, else "" (also "" when node is absent)."""
    import shutil
    import tempfile
    node = shutil.which("node")
    if not node:
        return ""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8")
    try:
        tmp.write(source)
        tmp.close()
        proc = await asyncio.create_subprocess_exec(
            node, "--check", tmp.name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=_SYNTAX_CHECK_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ""
        if proc.returncode == 0:
            return ""
        return (stderr_b or b"").decode("utf-8", "replace")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _remap_node_diag(diag: str, start_line: int) -> str:
    """Turn node's temp-file diagnostic into an HTML-line-anchored message.

    node prints `<tmp>:<lineno>`, the offending source line, a caret, then
    `SyntaxError: …`. We recover the line, add the block's HTML offset, and
    keep the source line + message — dropping the module-loader stack noise.
    """
    lines = diag.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    node_line = None
    if nonempty:
        m = re.search(r":(\d+)\s*$", nonempty[0])
        if m:
            node_line = int(m.group(1))
    msg = ""
    for ln in lines:
        s = ln.strip()
        if re.match(r"^[\w.]*Error:", s):
            msg = s
            break
    src_line = ""
    for i, ln in enumerate(lines):
        if ln.strip() and set(ln.strip()) <= {"^", "~"} and i > 0:
            src_line = lines[i - 1].strip()
            break
    if node_line is not None:
        head = f"line {start_line + node_line - 1}: {msg or 'SyntaxError'}"
    else:
        head = msg or (nonempty[0][:200] if nonempty else "syntax error")
    return head + (f"\n    {src_line[:160]}" if src_line else "")


async def _syntax_feedback(path: Path, filename: str) -> str:
    """Best-effort post-write syntax check. Returns "" when clean or unknown.

    Write time is the cheapest place to catch a parse error: the same tool
    result that says SUCCESS also names the exact broken line, so the fix
    lands on the very next turn. (Production failure this prevents: data.js
    shipped with unescaped apostrophes in single-quoted strings, the
    verifier confirmed the build without loading it, and the user found the
    bug by clicking a dead button.)

    .py / .json parse in-process; .js / .mjs / .cjs shell out to
    ``node --check`` when a node binary exists on PATH (skipped silently
    otherwise — a missing checker must never fail a successful write).
    """
    ext = str(filename).split(".")[-1].lower()
    err = ""
    try:
        if ext == "py":
            import ast
            try:
                ast.parse(await asyncio.to_thread(path.read_text))
            except SyntaxError as se:
                err = f"{se.msg} (line {se.lineno}, col {se.offset})"
        elif ext == "json":
            try:
                json.loads(await asyncio.to_thread(path.read_text))
            except json.JSONDecodeError as je:
                err = f"{je.msg} (line {je.lineno}, col {je.colno})"
        elif ext in ("js", "mjs", "cjs"):
            import shutil
            node = shutil.which("node")
            if not node:
                return ""
            proc = await asyncio.create_subprocess_exec(
                node, "--check", str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=_SYNTAX_CHECK_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ""
            if proc.returncode != 0:
                # node prints "path:line", the offending source line, a
                # caret marker, then "SyntaxError: ..." — keep those, drop
                # the trailing module-loader noise.
                lines = (stderr_b or b"").decode("utf-8", "replace").splitlines()
                err = "\n".join(ln[:200] for ln in lines if ln.strip())[:800]
        elif ext in ("html", "htm"):
            # A single-file deliverable hides its JS in inline <script> blocks;
            # check each so a typo there (e.g. `content='…'` written `content=`)
            # is named with its HTML line instead of surfacing as a silent
            # blank page the model can only find by re-reading the whole file.
            html = await asyncio.to_thread(path.read_text)
            for start_line, src in _inline_js_blocks(html):
                diag = await _node_check_source(src)
                if diag:
                    err = _remap_node_diag(diag, start_line)
                    break
    except Exception:
        return ""
    if not err:
        return ""
    pretty_log(
        "syntax check",
        f"{filename}: {err.splitlines()[0] if err.splitlines() else err}",
        icon=Icons.FAIL, level="WARNING",
    )
    return (
        f"\n⚠ SYNTAX CHECK FAILED: '{filename}' was written but does NOT "
        f"parse. Fix this BEFORE any other step — a browser loads a broken "
        f"script silently and every downstream symptom (dead buttons, blank "
        f"page) traces back here:\n{err}"
    )


def _reindent_replacement(file_content: str, matched_text: str, new_text: str) -> str:
    """Re-anchor a multi-line replacement block to the indentation of the
    region it is replacing.

    The whitespace-flexible matcher (`r'\\s+'.join(words)`) matches starting
    at the first non-whitespace token, so the matched span does NOT include
    the leading indentation of its first line — that indent stays in the
    file, before the insertion point. A raw ``str.replace(match, new_text)``
    therefore puts line 1 of ``new_text`` after the file's indent but every
    continuation line exactly as the model typed it. Whatever absolute
    indentation the model chose, the result is broken: dedented → "unindent
    does not match any outer indentation level"; absolute → "unexpected
    indent". This was the cause of a ~10-turn replace/rewrite spiral in
    production (a `set_input_size` guard removal that broke the file four
    times in a row).

    Fix: drop the model's absolute indentation entirely and rebuild it from
    the anchor — line 1 gets no indent (the file supplies it), every
    continuation line gets ``anchor_indent`` plus its own indentation
    *relative* to the block's first line. Nesting inside the block is
    preserved; the block as a whole lands at the right column.

    Only fires when the match begins at the start of a line (so the anchor
    is genuine indentation, not code) and the replacement is multi-line.
    Returns ``new_text`` unchanged otherwise.
    """
    if "\n" not in new_text:
        return new_text
    idx = file_content.find(matched_text)
    if idx < 0:
        return new_text
    line_start = file_content.rfind("\n", 0, idx) + 1
    anchor_indent = file_content[line_start:idx]
    if anchor_indent.strip():
        # Match did not start at column 0 of its line (there's code before
        # it) — re-indentation doesn't apply; insert verbatim.
        return new_text

    lines = new_text.split("\n")
    base = None
    for ln in lines:
        if ln.strip():
            base = len(ln) - len(ln.lstrip())
            break
    if base is None:
        return new_text

    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            out.append("")
            continue
        rel = max(0, (len(ln) - len(ln.lstrip())) - base)
        prefix = (" " * rel) if i == 0 else (anchor_indent + " " * rel)
        out.append(prefix + ln.lstrip())
    return "\n".join(out)


def _syntax_regression(prev_content: str, new_content: str, filename: str) -> str:
    """Return a one-line error when ``new_content`` would INTRODUCE a syntax
    error that ``prev_content`` did not have, else "".

    Used to refuse a destructive replace *before* it touches disk. A replace
    that turns a parsing file into a non-parsing one destroys the
    known-good anchor state the model needs for its next surgical edit —
    that's what compounded a single broken edit into a multi-turn rewrite
    spiral (each subsequent SEARCH block failed to match the now-corrupted
    file). If the file was ALREADY broken, the edit is allowed (it may be a
    partial fix). Only .py / .json are checked in-process; other types pass
    through (best-effort, unchanged behaviour).
    """
    ext = str(filename).split(".")[-1].lower()
    if ext == "py":
        import ast
        try:
            ast.parse(new_content)
            return ""
        except SyntaxError as se:
            try:
                ast.parse(prev_content)
            except SyntaxError:
                return ""  # already broken — not a regression
            return f"{se.msg} (line {se.lineno}, col {se.offset})"
    elif ext == "json":
        try:
            json.loads(new_content)
            return ""
        except json.JSONDecodeError as je:
            try:
                json.loads(prev_content)
            except json.JSONDecodeError:
                return ""
            return f"{je.msg} (line {je.lineno}, col {je.colno})"
    return ""


async def _write_replace_guarded(path: Path, prev_content: str, new_content: str,
                                 filename: str, success_msg: str) -> str:
    """Apply a replace result with a syntax-regression rollback guard.

    If ``new_content`` would introduce a NEW syntax error (see
    `_syntax_regression`), the file is left UNCHANGED and a REJECTED message
    is returned steering the model to a tight surgical edit instead of a
    full-file rewrite. Otherwise the content is written and normal
    post-write syntax feedback is appended.
    """
    regression = _syntax_regression(prev_content, new_content, filename)
    if regression:
        pretty_log("Replace Rejected",
                   f"{filename}: edit would break syntax — file left unchanged",
                   icon=Icons.WARN, level="WARNING")
        return (
            f"REJECTED: that replace would introduce a syntax error and was "
            f"NOT applied — '{filename}' is unchanged on disk: {regression}. "
            f"Your replacement block's indentation or structure is off. "
            f"Re-read the file, then emit a TIGHT single-line SEARCH/REPLACE "
            f"for the surgical edit. Do NOT rewrite the whole file."
        )
    await asyncio.to_thread(path.write_text, new_content, encoding="utf-8")
    return success_msg + await _syntax_feedback(path, filename)


async def tool_write_file(filename: str, content: Any, sandbox_dir: Path):
    pretty_log("File Write", filename, icon=Icons.TOOL_FILE_W)
    try:
        # Reject a missing/empty payload, or the LITERAL Python None the LLM
        # sometimes emits as a string. Do NOT reject a legitimate file whose
        # content is the word "none"/"None" or is only whitespace — match the
        # bare token exactly, not any content that lowercases to "none".
        _c_str = "" if content is None else str(content)
        if content is None or _c_str.strip() == "" or _c_str.strip() == "None":
            return f"Error: The 'content' you provided for '{filename}' is empty or 'None'. You MUST provide the actual text to write. If you intended to use data from a previous tool, ensure that tool succeeded and produced output."

        # Auto-serialize if the LLM sends a JSON object/list instead of a string
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        elif not isinstance(content, str):
            content = str(content)

        # Strip accidental markdown wrappers for code/data files.
        # We pass `filename` so the extractor can skip the strip when
        # the payload is already valid code for the target language —
        # this protects raw .py files whose docstrings embed fenced
        # examples (sphinx / mkdocs style) from having their content
        # replaced by the inner example snippet.
        ext = str(filename).split('.')[-1].lower()
        if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
            from ..utils.sanitizer import extract_code_from_markdown
            content = extract_code_from_markdown(content, filename=filename)

        path = _get_safe_path(sandbox_dir, filename)

        # SELF-HEALING: Auto-create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always UTF-8: without an explicit encoding, write_text uses the
        # process locale (comma-decimal Greek Macs, LANG unset), which both
        # mojibakes non-ASCII content and — on UnicodeEncodeError — leaves
        # the file truncated (open('w') truncates before the failing encode).
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        # Report the resolved sandbox-relative path so scripts running
        # in the container (cwd=/workspace) know exactly where to find
        # it — e.g. if the model wrote "sandbox/foo.txt" and we honored
        # it literally, the container sees /workspace/sandbox/foo.txt
        # and the Python `open("sandbox/foo.txt")` works.
        try:
            rel = path.resolve().relative_to(sandbox_dir.resolve())
            rel_str = str(rel)
        except Exception:
            rel_str = str(filename)
        summary = _fixture_summary(content, ext)
        syntax_note = await _syntax_feedback(path, filename)
        return (
            f"SUCCESS: Wrote {len(content)} chars to '{filename}'. "
            f"Script-side path (from sandbox cwd): '{rel_str}'."
            f"{summary}{syntax_note}"
        )
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_list_files(sandbox_dir: Path, memory_system=None):
    pretty_log("Sandbox Tree", "Listing workspace files & mapping repo", icon=Icons.TOOL_FILE_I)
    try:
        def _build_map():
            import ast
            import os
            tree_lines = []
            
            for root, dirs, files in os.walk(sandbox_dir):
                # Ignore hidden and virtual env directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
                
                rel_root = Path(root).relative_to(sandbox_dir)
                root_prefix = "" if rel_root == Path(".") else f"{rel_root}/"
                
                for f in sorted(files):
                    if f.startswith('.'): continue
                    path = Path(root) / f
                    line = f"  {root_prefix}{f}"
                    
                    # --- REPO MAP: Extract AST Signatures for Python files ---
                    if f.endswith('.py'):
                        try:
                            # stat() must stay inside the try: a broken
                            # symlink or a file deleted between os.walk and
                            # here raises OSError, which would otherwise
                            # abort the ENTIRE listing.
                            if path.stat().st_size < 100000:
                                code = path.read_text(errors='ignore')
                                parsed = ast.parse(code)
                                sigs = []
                                for node in parsed.body:
                                    if isinstance(node, ast.FunctionDef):
                                        sigs.append(f"def {node.name}()")
                                    elif isinstance(node, ast.ClassDef):
                                        sigs.append(f"class {node.name}")
                                if sigs:
                                    line += f"  [{', '.join(sigs[:5])}{'...' if len(sigs)>5 else ''}]"
                        except Exception:
                            pass
                    tree_lines.append(line)
                    
            return "\n".join(tree_lines[:200]) if tree_lines else "[Empty]"
            
        sandbox_tree = await asyncio.to_thread(_build_map)
        if len(sandbox_tree.splitlines()) >= 200:
            sandbox_tree += "\n  ... [Truncated for length]"
            
        return f"CURRENT SANDBOX DIRECTORY STRUCTURE:\n{sandbox_tree}\n\n(Use these filenames for all file tools)"
    except Exception as e: return f"Error scanning sandbox: {e}"

async def tool_download_file(url: str, sandbox_dir: Path, tor_proxy: str, filename: str = None):
    # --- SSRF guard (shared) ---
    # Block file:// and internal/metadata hosts BEFORE any fetch, so the
    # LLM can't make the host read local files or reach 169.254.169.254 /
    # loopback services. file:// is reachable regardless of the proxy.
    from ..utils.helpers import url_ssrf_reason as _url_ssrf_reason
    _ssrf = _url_ssrf_reason(url)
    if _ssrf:
        return f"Error: {_ssrf}"

    # 1. Clean Proxy URL
    proxy_url = tor_proxy
    mode = "TOR" if proxy_url and "127.0.0.1" in proxy_url else "WEB"

    pretty_log(f"Download [{mode}]", f"{url[:35]}..", icon=Icons.TOOL_DOWN)
    
    if proxy_url and proxy_url.startswith("socks5://"): 
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for attempt in range(3):
        try:
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                try:
                    target_path = _get_safe_path(sandbox_dir, filename)
                except ValueError as ve: return str(ve)
                
                async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=60.0) as client:
                    resp = await client.get(url, stream=True)
                    if resp.status_code != 200:
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        return f"Error {resp.status_code} - Failed to download from {url}"
                    
                    clength = resp.headers.get("Content-Length")
                    if clength and int(clength) > 50000000:
                        return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."
                    
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    _MAX_DL = 50_000_000
                    _total = 0
                    _overflow = False
                    with open(target_path, "wb") as f:
                        buffer = bytearray()
                        async for chunk in resp.aiter_content():
                            if chunk:
                                buffer.extend(chunk)
                                _total += len(chunk)
                                if _total > _MAX_DL:
                                    _overflow = True
                                    break
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                        if buffer and not _overflow:
                            await asyncio.to_thread(f.write, buffer)
                    if _overflow:
                        try: target_path.unlink()
                        except Exception: pass
                        return "Error: download exceeded the 50MB cap (server omitted or exceeded Content-Length)."
                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
            else:
                async with httpx.AsyncClient(proxy=proxy_url, headers=headers, follow_redirects=True, timeout=60.0) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            if resp.status_code in [401, 403, 503] and mode == "TOR":
                                await asyncio.to_thread(request_new_tor_identity)
                                await asyncio.sleep(5)
                                continue
                            return f"Error {resp.status_code} - Failed to download from {url}"
                        
                        clength = resp.headers.get("Content-Length")
                        if clength and int(clength) > 50000000:
                            return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."

                        try:
                            target_path = _get_safe_path(sandbox_dir, filename)
                        except ValueError as ve: return str(ve)

                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        _MAX_DL = 50_000_000
                        _total = 0
                        _overflow = False
                        with open(target_path, "wb") as f:
                            buffer = bytearray()
                            async for chunk in resp.aiter_bytes():
                                buffer.extend(chunk)
                                _total += len(chunk)
                                if _total > _MAX_DL:
                                    _overflow = True
                                    break
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                            if buffer and not _overflow:
                                await asyncio.to_thread(f.write, buffer)
                        if _overflow:
                            try: target_path.unlink()
                            except Exception: pass
                            return "Error: download exceeded the 50MB cap (server omitted or exceeded Content-Length)."

                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            
    return f"Error: Failed after 3 attempts. Last error: {last_error}"

async def tool_file_search(pattern: str, sandbox_dir: Path, filename: str = None, sandbox_manager=None):
    # 1. Safety check for None
    if not pattern: return "Error: 'content' (search pattern) is required."

    try:
        # 2. Clean filename and pattern from model-injected artifacts.
        # ``_get_safe_path`` resolves to a HOST absolute path (e.g.
        # ``/Users/me/sandbox/webos/app.js``), but ``sandbox_manager.
        # execute`` runs the rg command INSIDE the Docker container at
        # workdir=/workspace where the host sandbox is bind-mounted. The
        # container has no visibility into the host filesystem, so a
        # host-absolute path makes rg report "no matches" silently —
        # confirmed in a 2026-04-26 webOS session where the agent burned
        # six search turns chasing the same empty result. Translate to a
        # container-visible /workspace/<rel> path here so the command rg
        # sees actually exists. Path safety / traversal protection still
        # comes from ``_get_safe_path`` rejecting anything that resolves
        # outside ``sandbox_dir``.
        if filename:
            search_path = _get_safe_path(sandbox_dir, filename)
            container_path = _to_container_path(sandbox_dir, search_path)
            escaped_path = shlex.quote(container_path)
        else:
            escaped_path = "."

        pattern = str(pattern).strip("'\"") # Strip accidental quotes
        escaped_pattern = shlex.quote(pattern)

        pretty_log("File Search", f"'{pattern}' in {filename or 'workspace'}", icon=Icons.TOOL_FILE_S)

        cmd = f"rg --line-number --no-heading --color=never --max-columns=300 {escaped_pattern} {escaped_path}"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, timeout=20)
        
        # Cap search output but keep BOTH head and tail. Without the tail
        # we'd silently hide the last matches in the file, which is the
        # opposite of helpful when the user is debugging a regression
        # whose root cause sits at the bottom of the output.
        # Cap: ~40 KB (10 KB head + 30 KB tail).
        try:
            HEAD_BYTES = 10 * 1024
            TAIL_BYTES = 30 * 1024
            MAX_TOTAL = HEAD_BYTES + TAIL_BYTES
            if isinstance(output, str) and len(output) > MAX_TOTAL:
                head = output[:HEAD_BYTES]
                tail = output[-TAIL_BYTES:]
                dropped = len(output) - HEAD_BYTES - TAIL_BYTES
                output = (
                    f"{head}\n\n"
                    f"... [TRUNCATED: {dropped} bytes dropped — keeping "
                    f"first {HEAD_BYTES // 1024} KB and last "
                    f"{TAIL_BYTES // 1024} KB. Make your search pattern "
                    f"more specific for a complete result.] ...\n\n"
                    f"{tail}"
                )
        except Exception:
            # Truncation is best-effort; never crash the search.
            pass

        return output if output.strip() else "Report: No matches found. (Tip: Use list_files to verify the path)"

    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_find_files(pattern: str, sandbox_manager, path: str = ".", sandbox_dir: Path = None):
    if not pattern: return "Error: 'pattern' is required for find operation."
    pretty_log("Find Files", f"'{pattern}' in {path}", icon=Icons.TOOL_FILE_S)
    try:
        # Normalize the path the same way `tool_file_search` does: if
        # the caller hands us a HOST-absolute path that points inside
        # the sandbox, translate it to its container-visible
        # /workspace/... form so `find` (running inside the container)
        # can actually see it. A relative path or a /workspace path is
        # left alone. ``sandbox_dir`` is required for translation; when
        # missing we fall through to the legacy behavior.
        search_path = path
        if sandbox_dir is not None and path and path != ".":
            try:
                resolved = _get_safe_path(sandbox_dir, path)
                search_path = _to_container_path(sandbox_dir, resolved)
            except ValueError:
                # Outside-sandbox traversal — surface clearly rather than
                # silently searching the wrong tree.
                return f"Error: path '{path}' resolves outside the sandbox."
        # The sandbox runs commands WITHOUT a shell (docker exec splits the
        # string into argv), so a bare `| head` would be passed as literal
        # args to `find`. Wrap the whole pipeline in `sh -c` so the pipe is
        # interpreted (mirrors the `bash -c` pattern in execute.py).
        _inner = f"find {shlex.quote(search_path)} -type f -name {shlex.quote(pattern)} -not -path '*/\\.*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*' | head -n 100"
        cmd = f"sh -c {shlex.quote(_inner)}"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, timeout=15)
        return output if output.strip() else "Report: No files found matching that pattern."
    except Exception as e: return f"Error: {e}"

async def tool_inspect_file(filename: str, sandbox_dir: Path, lines: int = 10):
    if not filename: return "Error: 'path' (filename) is required for inspection."
    pretty_log("File Peek", filename, icon=Icons.TOOL_FILE_I)
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists():
            return _missing_file_message(filename, sandbox_dir)
        def _read_peek():
            content = []
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for _ in range(lines):
                    line = f.readline()
                    if not line: break
                    content.append(line.strip())
            return "\n".join(content)
        
        return await asyncio.to_thread(_read_peek)
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_copy_file(src_name: str, dest_name: str, sandbox_dir: Path):
    """Copy a file or directory within the sandbox.

    The operation is additive (the source is preserved). Uses ``shutil.copy2``
    for files (metadata preserved) and ``shutil.copytree`` for directories.
    Refuses to overwrite an existing destination to avoid accidental clobber —
    the caller should delete or rename first if overwrite is intentional.
    """
    pretty_log("File Copy", f"{src_name} -> {dest_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        src_path = _get_safe_path(sandbox_dir, src_name, allow_root=False)
        dest_path = _get_safe_path(sandbox_dir, dest_name, allow_root=False)
        if not src_path.exists():
            return f"Error: '{src_name}' not found."
        if dest_path.exists():
            return (
                f"Error: destination '{dest_name}' already exists. Delete or "
                f"rename it first if you intended to overwrite."
            )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            await asyncio.to_thread(shutil.copytree, str(src_path), str(dest_path))
        else:
            await asyncio.to_thread(shutil.copy2, str(src_path), str(dest_path))
        return f"SUCCESS: Copied '{src_name}' to '{dest_name}'."
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error: {e}"


async def tool_rename_file(old_name: str, new_name: str, sandbox_dir: Path):
    pretty_log("File Rename", f"{old_name} -> {new_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        old_path = _get_safe_path(sandbox_dir, old_name, allow_root=False)
        new_path = _get_safe_path(sandbox_dir, new_name, allow_root=False)
        if not old_path.exists(): return f"Error: '{old_name}' not found."
        new_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
        return f"SUCCESS: Renamed/Moved '{old_name}' to '{new_name}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_delete_file(filename: str, sandbox_dir: Path):
    pretty_log("File Delete", filename, icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        path = _get_safe_path(sandbox_dir, filename, allow_root=False)
        if not path.exists(): return f"Error: '{filename}' not found."
        if path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await asyncio.to_thread(path.unlink)
        return f"SUCCESS: Deleted '{filename}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_read_document_chunked(filename: str, sandbox_dir: Path, page: int = 1, chunk_size: int = 32000, max_context: int = 8192) -> str:
    """
    Robust reader for large files. Supports PDFs via PyMuPDF or plain text chunked extraction.
    """
    pretty_log("Chunked Read", f"{filename} [Page {page}]", icon=Icons.TOOL_FILE_R)
    
    # GUARD 1: Stop model from trying to read URLs as files. Same logic
    # as `tool_read_file` — surface `browser` first, `knowledge_base`
    # only when the user wants to ingest a document.
    if str(filename).startswith("http"):
        return (
            f"Error: file_system cannot read URLs. "
            f"To VIEW a webpage right now, use the `browser` tool "
            f"(operation='navigate' or 'extract_text', url='{filename}'). "
            f"To INGEST a document URL into long-term memory, use "
            f"`knowledge_base(action='ingest_document', filename='{filename}')`. "
            f"For a one-off page read pick `browser` — that's almost always what you want."
        )

    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."

        # Ensure page and chunk_size are integers
        try:
            page = int(page)
            if page < 1: page = 1
        except:
            page = 1
            
        try:
            chunk_size = int(chunk_size)
            if chunk_size < 1000: chunk_size = 1000
            max_chunk = max(80000, int(max_context * 3.5 * 0.2))
            if chunk_size > max_chunk: chunk_size = max_chunk # dynamic cap sanity
        except:
            chunk_size = 32000

        def _extract_chunk():
            # `page` is rebound below (out-of-range single-section files are
            # served as page 1); declare nonlocal so that assignment doesn't
            # shadow the enclosing parameter and UnboundLocalError the reads.
            nonlocal page
            if filename.lower().endswith(".pdf"):
                try:
                    import fitz # PyMuPDF
                except ImportError:
                    return "Error: PyMuPDF (fitz) is not installed. PDF chunked reading requires it."
                    
                doc = fitz.open(path)
                total_pages = len(doc)
                
                if page > total_pages:
                    doc.close()
                    # Non-striking, terminal guidance (same rationale as the
                    # text path): an out-of-range page is the model over-
                    # estimating, and a higher-page retry will never succeed.
                    return (
                        f"NOTE: '{path.name}' has only {total_pages} page"
                        f"{'s' if total_pages != 1 else ''} (valid: 1-{total_pages}); "
                        f"page {page} does not exist. Do NOT request a higher "
                        f"page — re-read with page=1..{total_pages} if needed."
                    )

                # 1-indexed to 0-indexed for fitz
                text = doc[page - 1].get_text()
                last = page >= total_pages
                doc.close()
                tail = (
                    f"\n[End of PDF — page {page} is the last of {total_pages}; do not request page {page+1}.]"
                    if last else
                    f"\n[Page {page} of {total_pages}. Use page={page+1} to continue.]"
                )
                return f"[PDF Data - Page {page} of {total_pages}]\n{text}{tail}"
            else:
                # Text-based file reading with overlap
                file_size = path.stat().st_size
                overlap = min(200, chunk_size // 4)
                
                effective_chunk = chunk_size - overlap
                total_pages = max(1, (file_size + effective_chunk - 1) // effective_chunk)

                # Out-of-range page is almost always the model OVER-ESTIMATING
                # the file's size — guessing a page number instead of paginating
                # from 1. Never strike on it: make the turn progress instead.
                out_of_range = page > total_pages
                if out_of_range:
                    if total_pages == 1:
                        # The whole file is one section. The model never read it
                        # (it jumped straight to a high page), so serve the FULL
                        # file as section 1 rather than erroring on a page that
                        # was never going to exist.
                        page = 1
                    else:
                        # Genuinely chunked and the model overshot the end. This
                        # is terminal, not retryable — give actionable guidance
                        # and DON'T prefix with "Error" (which would strike and
                        # invite a higher-page retry loop).
                        return (
                            f"NOTE: '{path.name}' is only {total_pages} sections "
                            f"long (valid pages: 1-{total_pages}); section {page} "
                            f"does not exist and you have reached the END of the "
                            f"file. There is nothing further to read — do NOT "
                            f"request a higher page number. Re-read with "
                            f"page=1..{total_pages} if you need earlier content."
                        )

                start_byte = (page - 1) * effective_chunk
                # Read slightly more for overlap and to find a clean break if needed
                read_amount = chunk_size

                with open(path, "rb") as f:
                    f.seek(start_byte)
                    raw_bytes = f.read(read_amount)
                text = raw_bytes.decode("utf-8", errors="ignore")

                # End-aware footer: on the LAST section, telling the model to
                # "use page=N+1 to continue" is exactly what produced the
                # out-of-range requests — there is no N+1. Say so explicitly.
                if page >= total_pages:
                    if out_of_range:
                        footer = (
                            f"End of file — this is the COMPLETE content of "
                            f"'{path.name}' ({total_pages} section"
                            f"{'s' if total_pages != 1 else ''}). You requested a "
                            f"higher page; the file is smaller than you expected. "
                            f"Do not paginate further."
                        )
                    else:
                        footer = (
                            f"End of Section {page} — this is the LAST section "
                            f"({total_pages} total). You have now read the whole "
                            f"file; do NOT request page {page+1} (it does not exist)."
                        )
                else:
                    footer = f"End of Section {page}. Use page={page+1} to continue reading"
                return f"--- [TEXT DATA - Section {page} of {total_pages}] ---\n\n{text}\n\n--- [{footer}] ---"

        return await asyncio.to_thread(_extract_chunk)
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error reading document: {e}"

# Unified router
async def tool_file_system(operation: str = None, sandbox_dir: Path = None, path: str = None, content: str = None, replace_with: str = None, destination: str = None, pattern: str = None, max_context: int = 8192, read_budget: "ReadBudget | None" = None, **kwargs):
    if not operation:
        return "SYSTEM INSTRUCTION: The 'operation' parameter is MANDATORY. You must specify it (e.g., operation='read')."
    # Unified mapping for common parameter hallucinations
    url = kwargs.get("url")
    
    # --- HALLUCINATION HEALING ---
    # 1. If the LLM put the URL in the 'path' or 'filename' parameter
    if path and str(path).startswith("http"):
        if not url: url = str(path)
        path = None
    elif kwargs.get("filename") and str(kwargs.get("filename")).startswith("http"):
        if not url: url = str(kwargs.get("filename"))
        kwargs["filename"] = None

    target_path = path or kwargs.get("filename") or kwargs.get("path") or kwargs.get("file")
    final_content = content or kwargs.get("data") or kwargs.get("text") or kwargs.get("content")
    destination = destination or kwargs.get("new_name") or kwargs.get("target") or kwargs.get("new_path")
    pattern = pattern or kwargs.get("query") or kwargs.get("search_pattern")

    # 2. If the LLM used 'url' as a filename for a non-download operation
    if not target_path and url and operation != "download":
        target_path = url
        url = None
    
    # If the LLM put the content in 'path' but didn't provide 'content' (common for write)
    if operation == "write" and target_path and not final_content:
        # Check if the LLM accidentally sent the content as the only other parameter
        return "SYSTEM INSTRUCTION: The 'content' parameter is MANDATORY for write operations. You must provide the full text to write."

    sandbox_manager = kwargs.get("sandbox_manager")

    if operation in ["list", "list_files"]: return await tool_list_files(sandbox_dir)
    if operation == "search": 
        search_target = pattern or final_content
        if not search_target:
            return "SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for search operations."
        return await tool_file_search(search_target, sandbox_dir, target_path, sandbox_manager)
    if operation == "find":
        search_target = pattern or final_content
        if not search_target:
            return "SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for find operations (e.g. '*.py')."
        return await tool_find_files(search_target, sandbox_manager, target_path or ".", sandbox_dir=sandbox_dir)
    
    if operation == "download":
        if not url:
            return "SYSTEM INSTRUCTION: The 'url' parameter is MANDATORY for download operations."
            
        # Auto-heal missing or invalid target_path
        if not target_path or str(target_path).strip() == "" or str(target_path).startswith("http") or target_path == url:
            parsed = urllib.parse.urlparse(str(url))
            path_name = Path(parsed.path).name
            target_path = path_name if path_name else "download.bin"

        return await tool_download_file(url=str(url), sandbox_dir=sandbox_dir, tor_proxy=kwargs.get("tor_proxy"), filename=target_path)

    if not target_path: 
        return f"SYSTEM INSTRUCTION: The 'path' (target filename) is missing for the '{operation}' operation. You MUST specify WHICH file to {operation}."
    
    if operation == "read":
        # Optional line-range (#11). Accept the common aliases the model
        # reaches for (start/from/line_start ...). None → whole-file read.
        def _as_int(v):
            try:
                return int(v) if v is not None and str(v).strip() != "" else None
            except (TypeError, ValueError):
                return None
        _start = _as_int(kwargs.get("start_line", kwargs.get("start",
                         kwargs.get("from_line", kwargs.get("line_start")))))
        _end = _as_int(kwargs.get("end_line", kwargs.get("end",
                       kwargs.get("to_line", kwargs.get("line_end")))))
        return await tool_read_file(target_path, sandbox_dir, max_context=max_context,
                                    read_budget=read_budget,
                                    start_line=_start, end_line=_end)
    elif operation == "read_chunked":
        page = kwargs.get("page", 1)
        chunk_size = kwargs.get("chunk_size", 32000)
        return await tool_read_document_chunked(target_path, sandbox_dir, page=page, chunk_size=chunk_size, max_context=max_context)
    elif operation == "inspect": return await tool_inspect_file(target_path, sandbox_dir)
    elif operation == "write": return await tool_write_file(target_path, final_content, sandbox_dir)
    elif operation == "replace":
        # Accept the param-name variants the model routinely reaches for
        # (the live run's FIRST replace failed purely because it passed
        # `old_text=`/`replace_with=`, and `old_text` was not an alias for
        # `content`). Old text: content/data/text → old_text/old_string/
        # search/old. New text: replace_with → new_text/new_string/new/
        # replacement.
        _old = (final_content or kwargs.get("old_text") or kwargs.get("old_string")
                or kwargs.get("search") or kwargs.get("old"))
        _new = (replace_with if replace_with is not None else kwargs.get("replace_with"))
        if _new is None:
            _new = (kwargs.get("new_text") or kwargs.get("new_string")
                    or kwargs.get("new") or kwargs.get("replacement"))
        return await tool_replace_text(target_path, _old, _new, sandbox_dir)
    
    if operation == "copy":
        copy_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not copy_target or copy_target == target_path):
            copy_target = rw
        if not copy_target:
            return "SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for copy operations (must contain the destination filename/path)."
        return await tool_copy_file(target_path, copy_target, sandbox_dir)

    if operation in ["rename", "move"]:
        # Hallucination healing: catch intuitive parameter guesses like new_name or target
        rename_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not rename_target or rename_target == target_path):
            rename_target = rw

        if not rename_target:
            return "SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for rename/move operations (must contain the new filename/path)."
        return await tool_rename_file(target_path, rename_target, sandbox_dir)
        
    if operation == "delete":
        return await tool_delete_file(target_path, sandbox_dir)
    
    return f"Unknown operation: {operation}"