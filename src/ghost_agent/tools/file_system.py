import asyncio
import errno as _errno
import functools
import io
import os
import stat as stat_mod
import re
import sys
import urllib.parse
import json
import shlex
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import httpx
from .outcome import ToolOutcome, append_note
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log, request_id_context
from ..utils.edit_ledger import record_edit
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
_LINE_RANGE_MAX_BYTES = 96 * 1024   # review B: 2,000 lines of minified JS/CSV came back as 3 MB


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
    picked_bytes, byte_capped = 0, False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                last_lineno = lineno
                if lineno < s:
                    continue
                if lineno > hard_end:
                    break
                _lb = len(line.encode("utf-8", "replace"))
                if picked and picked_bytes + _lb > _LINE_RANGE_MAX_BYTES:
                    byte_capped = True    # review B: stop at the last line that fits, say where
                    hard_end = lineno - 1
                    break
                if not picked and _lb > _LINE_RANGE_MAX_BYTES:
                    # r2: one 5 MB line — show its head, say how long it really is
                    _cut = line.encode("utf-8", "replace")[:_LINE_RANGE_MAX_BYTES].decode("utf-8", "ignore")
                    picked.append(f"{lineno}\t{_cut.rstrip(chr(10))} [... line truncated: {_lb:,} bytes; "
                                  f"use operation='search' or an 'execute' script for the rest]")
                    byte_capped = True
                    hard_end = lineno
                    break
                picked.append(f"{lineno}\t{line.rstrip(chr(10))}")
                picked_bytes += _lb
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
    if byte_capped:
        truncated = (f"\n[... range capped at {_LINE_RANGE_MAX_BYTES // 1024} KB — showed lines "
                     f"{shown_first}-{shown_last}; re-read from start_line={shown_last + 1} for more]")
    elif e is not None and hard_end < e:
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


def _close_quietly(fd) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def read_bytes_nofollow(path: Path, *, max_bytes: int = 0, from_start: bool = False) -> bytes:
    """Read `path`, REFUSING to follow a symlink at the final component.

    The read counterpart of `write_text_nofollow`, and needed for the same
    reason: a FIXED filename inside a directory the model can write to. The
    service log tail (`<host_dir>/<stem>.log`) is exactly that — `host_dir`
    is bind-mounted into the container as `/workspace/.services`, so one
    `ln -s /Users/…/.ghost_api_key /workspace/.services/leak.log` followed
    by `manage_services(action='logs', name='leak')` returned the master key
    straight into the model's context. Demonstrated 2026-08-30 (§4DX r4).

    `max_bytes > 0` reads at most that many bytes from the END of the file
    (from the START with `from_start=True`).
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as e:
        raise ValueError(
            f"Security Error: refusing to read '{path}' — it is a symbolic "
            f"link or unreadable ({e.__class__.__name__}: {e})") from e
    try:
        import stat as _stat
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(
                f"Security Error: refusing to read '{path}' — not a regular file.")
        _block_if_regular(fd)                                   # §4SD
        if max_bytes > 0 and not from_start:
            size = os.fstat(fd).st_size
            if size > max_bytes:
                os.lseek(fd, size - max_bytes, os.SEEK_SET)
        chunks = []
        got = 0
        while True:
            want = (1 << 20) if not (from_start and max_bytes > 0) else min(1 << 20, max_bytes - got)
            if want <= 0:
                break
            b = os.read(fd, want)
            if not b:
                break
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def write_text_nofollow(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path`, REFUSING to follow a symlink at the final
    component.

    ⚠ WHY THIS EXISTS (§4DX round 2, 2026-08-30). Three host-side writers
    used a FIXED filename inside a directory the model can write to:

        acquired_skills.py   <sandbox>/test_skill.py     (create_skill)
        memory/projects.py   <project>/RELEASE.md.tmp    (manage_projects)
        memory/projects.py   <project>/PROJECT_MAP.md.tmp

    None called `_get_safe_path`, because the name is a constant — there is
    no model-supplied path to validate. But the model does not need to
    supply a path: it plants a SYMLINK at that constant name with one
    in-container call (`ln -s /Users/…/agent.py /workspace/test_skill.py`),
    and the next write follows it. Demonstrated end to end: the agent's own
    source file was overwritten with model-authored Python, and the
    cleanup path then unlinked the symlink so the sandbox looked untouched.

    `O_NOFOLLOW` is checked by the KERNEL at open time on the final
    component, so this is atomic — an `is_symlink()` test followed by a
    write is a TOCTOU race the model can win, because it controls the
    directory. On a symlink the open fails with ELOOP and this raises.

    Intermediate directories are NOT covered by O_NOFOLLOW; callers whose
    parent directory is itself model-controlled must additionally resolve
    the parent through `_get_safe_path`.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        # ⚠ 0o666, NOT 0o600 — the umask still applies, so this reproduces
        # exactly what `write_text` produced. The security property here is
        # O_NOFOLLOW, not the mode. Creating at 0o600 would also travel:
        # `os.replace(tmp, target)` carries the tmp's inode onto an EXISTING
        # artifact, so `PROJECT_MAP.md` silently went 0644 -> 0600 on its
        # next render. Harmless on this box (the container runs as root and
        # every daemon is the same user) and a latent breakage the moment
        # anything is served by a different uid.
        # ⚠ O_NONBLOCK. `O_NOFOLLOW` refuses a SYMLINK; it says nothing
        # about a FIFO. Opening a reader-less FIFO for writing blocks
        # FOREVER, and both callers invoke this synchronously from inside an
        # `async def` — so `mkfifo /workspace/test_skill.py` freezes the
        # asyncio event loop, i.e. the entire agent, with no exception for
        # the caller's `try` to catch. O_NONBLOCK makes that open fail with
        # ENXIO instead, and the `_refuse_non_regular` check below rejects
        # every non-regular target explicitly.
        fd = os.open(str(path), flags | getattr(os, "O_NONBLOCK", 0), 0o666)
    except OSError as e:
        # ELOOP (symlink) / ENOTDIR — refuse loudly rather than write through.
        raise ValueError(
            f"Security Error: refusing to write '{path}' — it is a symbolic "
            f"link or an unwritable path ({e.__class__.__name__}: {e})"
        ) from e
    # ⚠ NO MANUAL os.close IN AN EXCEPT BRANCH. `os.fdopen` takes ownership
    # of the fd, and the `with` closes it on the way out of BOTH the normal
    # and the exception path. The first version added `os.close(fd)` to an
    # `except BaseException`, which double-closed on any write error
    # (ENOSPC, EIO, UnicodeEncodeError) — and this process runs
    # `asyncio.to_thread` pools, so in the window between the two closes
    # another thread can be handed the same fd number and have it closed
    # underneath it. `except OSError: pass` made the victim silent.
    #
    # If `fdopen` itself raises (it can, on MemoryError) the fd leaks — one
    # fd, once, on a path that is already failing, which is strictly better
    # than closing a descriptor someone else now owns.
    # The target must be a REGULAR file. A FIFO, socket or device node at
    # this path is not something a text write should ever land on, and
    # fstat-ing the OPEN fd (not the path) means there is no second lookup
    # to race.
    try:
        import stat as _stat
        _regular = _stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError as e:
        _close_quietly(fd)
        raise ValueError(f"Security Error: cannot stat '{path}': {e}") from e
    if not _regular:
        # ⚠ ONE close on this path. The first version closed inside the
        # `try` AND again in `except OSError`, re-introducing the very
        # double-close the write path had just been cleaned of.
        _close_quietly(fd)
        raise ValueError(
            f"Security Error: refusing to write '{path}' — it is not a "
            "regular file (FIFO, socket or device node).")
    try:
        _block_if_regular(fd)                                   # §4SD
    except OSError:
        _close_quietly(fd)                                      # review: not yet owned by fdopen
        raise
    with os.fdopen(fd, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        # durable before the caller's os.replace publishes it: a registry torn
        # by a power loss read as EMPTY and the next save dropped every row
        # (§4MD M9)
        os.fsync(fh.fileno())


def write_text_nofollow_in_dir(dir_path: Path, name: str, text: str, *,
                               encoding: str = "utf-8") -> None:
    """Write ``text`` to ``dir_path/name`` refusing to follow a symlink at
    EITHER component (§4GI, 2026-09-13).

    `write_text_nofollow` covers the final component only; its docstring
    says a model-controlled PARENT must be resolved separately. The
    run-output spill (`docker._spill_run_output`) writes a fixed name
    (`run_N.log`, the counter is announced to the model) into a fixed
    directory (`.ghost_runs`) — both under the bind mount, so the model
    can replace the DIRECTORY with a symlink and every later spill lands
    wherever it points. Opening the directory with O_DIRECTORY|O_NOFOLLOW
    and the file relative to that fd (`dir_fd`) makes both checks atomic:
    a symlink at the directory fails with ELOOP, a non-directory with
    ENOTDIR, a symlink at the name with ELOOP, and a FIFO/device is
    refused after fstat of the open fd. ``name`` must be a bare filename.
    """
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"Security Error: refusing to write '{name}' — not a bare filename")
    dflags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        dfd = os.open(str(dir_path), dflags)
    except OSError as e:
        raise ValueError(
            f"Security Error: refusing to write into '{dir_path}' — it is a "
            f"symbolic link or not a directory ({e.__class__.__name__}: {e})") from e
    try:
        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            fd = os.open(name, flags, 0o666, dir_fd=dfd)
        except OSError as e:
            raise ValueError(
                f"Security Error: refusing to write '{dir_path / name}' — it is a "
                f"symbolic link or an unwritable path ({e.__class__.__name__}: {e})") from e
    finally:
        _close_quietly(dfd)
    try:
        import stat as _stat
        _regular = _stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError as e:
        _close_quietly(fd)
        raise ValueError(f"Security Error: cannot stat '{dir_path / name}': {e}") from e
    if not _regular:
        _close_quietly(fd)
        raise ValueError(
            f"Security Error: refusing to write '{dir_path / name}' — it is not a "
            "regular file (FIFO, socket or device node).")
    try:
        _block_if_regular(fd)                                   # §4SD
    except OSError:
        _close_quietly(fd)                                      # review: not yet owned by fdopen
        raise
    with os.fdopen(fd, "w", encoding=encoding) as fh:
        fh.write(text)


#: Every traversal below is `dir_fd`-relative. Without these the fallback
#: would be a path-based walk, i.e. the very TOCTOU this module exists to
#: close — so the helpers REFUSE rather than degrade (fail closed).
#: `os.lstat` is NOT in `supports_dir_fd` — the dir_fd-relative lstat is
#: `os.stat(..., follow_symlinks=False)`, which is. Checking the wrong
#: name made every call refuse; the FIFO test caught it, and the TOCTOU
#: repro had been passing VACUOUSLY on the refusal until then.
_DIR_FD_OK = ({os.open, os.stat, os.readlink, os.symlink}
              <= getattr(os, "supports_dir_fd", set()))


def _require_dir_fd(what: str) -> None:
    if not _DIR_FD_OK:
        raise ValueError(
            f"Security Error: refusing to {what} — this platform has no "
            "dir_fd support, so a symlink swapped in mid-walk could not be "
            "refused atomically.")


#: Conditions that are never about ONE entry: the filesystem is full, the
#: destination is read-only, or the quota is spent. A per-entry guard that
#: swallows these turns a total failure into a PARTIAL success (§4GK round 6).
#: ⚠ DESCRIPTOR PRESSURE IS TRANSIENT AND SELF-INFLICTED (§4GK round 7).
#: EMFILE/ENFILE were added here in round 6 alongside the genuinely fatal
#: conditions — but they are not the same kind of thing. The filesystem being
#: full or read-only will not clear on its own; running short of descriptors
#: clears the moment the walk that consumed them finishes. Treating it as
#: fatal made `tool_copy_file` answer a bare `"Error: [Errno 24] …"` string
#: (not even a `ToolOutcome`, so no world-changed or idempotency accounting)
#: over a destination tree with 0 of 10 files — and the overwrite guard then
#: refused the retry with "destination already exists", which is exactly the
#: half-copy-blocks-the-retry failure rounds 4 and 5 closed. It also raises
#: BEFORE the stub-unlink, bypassing round 5's "NOT COPIED MUST MEAN NOT
#: PRESENT" rule. They belong with the per-entry conditions, recorded and
#: retried, not with the ones that end the copy.
_COPY_FATAL_ERRNOS = frozenset({_errno.ENOSPC, _errno.EROFS, _errno.EDQUOT})

#: Descriptor exhaustion is not a "this entry was swapped" condition: a guard
#: that treats it as one silently truncates the traversal (§4GK round 4).
_FD_EXHAUSTED_ERRNOS = frozenset({_errno.EMFILE, _errno.ENFILE})


def _open_dir_nofollow(name, dir_fd=None):
    """Open a directory, REFUSING a symlink at that component. Returns the
    fd. Relative to ``dir_fd`` when given, which is what makes the check and
    the use the SAME syscall: the parent is already an open fd, so the path
    to it cannot be re-pointed between them."""
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    return os.open(name, flags, dir_fd=dir_fd) if dir_fd is not None else os.open(name, flags)


def read_text_nofollow(name, *, dir_fd=None, encoding: str = "utf-8",
                       errors: str = "replace", max_bytes: int = 0,
                       offset: int = 0) -> str:
    """`Path.read_text` that refuses a symlink at the final component, and —
    given ``dir_fd`` — at every component (§4GJ round 3).

    `os.walk(followlinks=False)` only declines to DESCEND a linked
    directory; a linked FILE is still listed and `read_text` follows it, so
    the idle project readers were reading whatever the model pointed at with
    no race at all. Pass the ``dir_fd`` yielded by `walk_nofollow` and the
    read is atomic with respect to the whole path.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if dir_fd is not None:
        _require_dir_fd("read a file inside a model-writable tree")
        fd = os.open(name, flags, dir_fd=dir_fd)
    else:
        fd = os.open(str(name), flags)
    try:
        import stat as _stat
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise ValueError(
                f"Security Error: refusing to read '{name}' — not a regular file.")
        _block_if_regular(fd, st)                               # §4SD
        # ⚠ HEAD, NOT TAIL (§4GK round 4). This used to `lseek` to
        # `st.st_size - max_bytes` and return the END of the file. Its only
        # capped caller — the workspace tidy's referenced-media scan — uses
        # the head idiom ("read cap+1, skip if longer"), so for any file over
        # the cap it scanned the last 512 KB and never saw an `<img src=…>`
        # in the head: the asset counted as unreferenced and the idle sweep
        # DELETED it, against that function's own "a false DELETE breaks the
        # build". Nothing wanted the tail; `read_bytes_nofollow_fd`, the
        # sibling written later, already reads the head.
        #
        # `offset` (§4GK round 6) lets a caller stream a large file in bounded
        # windows — the workspace tidy's basename scan reads a whole bundle
        # that way rather than being forced to choose between an incomplete
        # answer and an unbounded read.
        if offset:
            os.lseek(fd, int(offset), os.SEEK_SET)
        chunks = []
        _left = int(max_bytes) if max_bytes > 0 else -1
        while True:
            b = os.read(fd, (1 << 20) if _left < 0 else min(1 << 20, _left))
            if not b:
                break
            chunks.append(b)
            if _left >= 0:
                _left -= len(b)
                if _left <= 0:
                    break
    finally:
        _close_quietly(fd)
    return b"".join(chunks).decode(encoding, errors)


def read_bytes_nofollow_fd(name, *, dir_fd=None, max_bytes: int = 0) -> bytes:
    """`read_text_nofollow`'s bytes twin: open `name` relative to `dir_fd`
    with ``O_NOFOLLOW`` and return its bytes. Used by consumers that hash
    rather than decode (the leaf loop's workspace snapshot). Raises OSError
    when the final component is a symlink — the caller skips it.
    """
    import stat as _stat
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if dir_fd is None:
        fd = os.open(str(name), flags)
    else:
        _require_dir_fd("read a model-writable tree")
        fd = os.open(name, flags, dir_fd=dir_fd)
    # ⚠ OWNERSHIP, not a try/finally (§4GK round 4). The first version wrapped
    # the `with os.fdopen(fd, closefd=True)` in `except BaseException: os.close(fd)`.
    # A read that raises mid-stream (EIO) then closed the SAME fd twice — the
    # exact defect the sibling writer documents at length above: between the
    # two closes another thread can be handed that fd number and the second
    # close shuts down an unrelated file. Ownership transfers at `fdopen`, so
    # the manual close must cover only the window BEFORE it.
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise ValueError(f"{name}: not a regular file")
        _block_if_regular(fd, st)                               # §4SD
        fh = os.fdopen(fd, "rb", closefd=True)
    except BaseException:
        _close_quietly(fd)
        raise
    with fh:
        return fh.read(max_bytes) if max_bytes else fh.read()


def write_bytes_nofollow_rel(root: Path, rel: str, data: bytes) -> None:
    """Write ``data`` to ``root/rel`` without following a symlink at ANY
    component of ``rel``, creating intermediate directories as needed
    (§4GK round 5).

    `write_text_nofollow_in_dir` covers a fixed single-level name; this is
    its nested twin, for a caller that restores a whole relative path.

    ⚠ WHY IT EXISTS. `dream._restore_mocks` rewrote a self-play snapshot
    with `target.parent.mkdir(parents=True); target.write_bytes(blob)` on a
    tree the SOLVER controls between the snapshot and the restore. Replace
    `data/x.csv` with a link to any host file and the restore overwrites
    that file with bytes the challenge chose — a write through a planted
    link, the class §4GJ closed for readers and left open for this writer
    (`the-sibling-one-revision-behind`). The pre-validator restore runs with
    no purge at all, so nothing removes the link first.

    Each component is opened ``O_DIRECTORY|O_NOFOLLOW`` relative to its
    parent's descriptor, so check and use are one syscall and the parent
    cannot be re-pointed underneath it. Raises ``OSError`` when any
    component is a link — the caller skips that entry.
    """
    _require_dir_fd("write into a model-writable tree")
    parts = [p for p in str(rel).replace(os.sep, "/").split("/") if p and p != "."]
    if not parts or ".." in parts:
        raise ValueError(f"refusing to restore {rel!r} — not a contained relative path")
    dfd = _open_dir_nofollow(str(root))
    try:
        for comp in parts[:-1]:
            try:
                os.mkdir(comp, 0o755, dir_fd=dfd)
            except FileExistsError:
                pass
            try:
                nxt = _open_dir_nofollow(comp, dir_fd=dfd)
            except OSError as exc:
                if exc.errno not in (_errno.ELOOP, _errno.ENOTDIR):
                    raise
                # ⚠ AN INTERMEDIATE COMPONENT IS A PLANTED LINK (§4GK round 6).
                # The first version applied the unlink-and-retry to the FINAL
                # component only, so `rm -rf data && ln -s /host/dir data` left
                # the link in place and the entry simply unrestored — and the
                # pre-validator restore runs with NO purge, so the validator
                # then read HOST bytes the solver chose, presented as the
                # pristine mock. The fix had converted a write-through into a
                # read-through and still reported success. Unlinking a symlink
                # removes the link, never its target; the snapshot is the truth
                # for everything under this root.
                os.unlink(comp, dir_fd=dfd)
                os.mkdir(comp, 0o755, dir_fd=dfd)
                nxt = _open_dir_nofollow(comp, dir_fd=dfd)
            _close_quietly(dfd)
            dfd = nxt
        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            fd = os.open(parts[-1], flags, 0o644, dir_fd=dfd)
        except OSError as exc:
            if exc.errno != _errno.ELOOP:
                raise
            # The final component is a LINK the model planted. Unlinking a
            # symlink removes the link, never its target, and the snapshot
            # is by definition the truth here — so drop it and write the
            # real file, rather than failing the restore and leaving the
            # planted link in place for the next attempt to write through.
            os.unlink(parts[-1], dir_fd=dfd)
            fd = os.open(parts[-1], flags, 0o644, dir_fd=dfd)
        # ⚠ OWNERSHIP, NOT A CATCH-ALL (§4GK round 6). This writer was born
        # with the exact defect its sibling `read_bytes_nofollow_fd`
        # documents twenty lines above: `os.fdopen(closefd=True)` takes the
        # descriptor, and the `with` closes it on the way out even when the
        # body raised — so a manual close in the handler closes that
        # descriptor NUMBER a second time, landing on whatever the kernel has
        # since reissued. Measured: 4 writers x 400 restores against 4
        # unrelated readers closed 172 of their open files out from under
        # them. The manual close covers only the window BEFORE ownership
        # transfers.
        try:
            fh = os.fdopen(fd, "wb", closefd=True)
        except BaseException:
            _close_quietly(fd)
            raise
        with fh:
            fh.write(data)
    finally:
        _close_quietly(dfd)


def walk_nofollow(base: Path):
    """`os.walk` for a tree the sandbox controls: never descends through a
    symlinked directory and never LISTS a symlinked file (§4GJ round 3).

    Yields ``(dirpath, filenames, dir_fd)``. ``dirpath`` is the logical path
    (every component was opened ``O_NOFOLLOW``, so it is also the real one)
    and ``filenames`` holds regular files only. ``dir_fd`` is an open
    descriptor for ``dirpath`` — pass it to `read_text_nofollow` to make the
    read atomic. **It is closed when the generator advances**, so read what
    you need before the next iteration; a consumer that stashes it is using
    a closed fd, not a stale one.

    Depth-first, sorted, and it never follows a mount-point loop because a
    symlinked directory is simply not descended.
    """
    _require_dir_fd("walk a model-writable tree")
    # Yield paths under the CALLER's `base`, not its realpath: callers do
    # `.relative_to(base)` and on macOS `/var/…` resolves to `/private/var/…`,
    # which raised ValueError. Safety does not depend on the reported string —
    # every component below is opened O_NOFOLLOW through the parent's fd.
    base = Path(base)
    try:
        # ⚠ THE TOP IS NOT RESOLVED, AND A REFUSAL IS LOUD (§4GK round 5).
        #
        # Round 4 opened `os.path.realpath(str(base))` here, borrowing
        # `copytree_nofollow`'s justification — but not its PRECONDITION. That
        # helper's caller ran the path through `_get_safe_path`, which
        # resolves AND contains; `walk_nofollow`'s callers do not.
        # `project_advancer._gather_project_files` and
        # `project_research.reconcile_research_dir` build
        # `<sandbox>/projects/<pid>` themselves and gate only on `is_dir()`,
        # which follows a link. So the model could `rm -rf` its project
        # directory, replace it with a link to the host's home, and the walk
        # would read host files and report their paths as if they were inside
        # the project. Reproduced: an `id_rsa` outside the sandbox came back
        # through the idle project reader. Resolve-then-contain is the pattern
        # for a trusted root; a helper that is not told the root must simply
        # not follow.
        #
        # The problem round 4 was fixing was real though: an O_NOFOLLOW
        # refusal used to `return`, yielding an EMPTY generator that every
        # consumer read as "this tree has no files". Refusing LOUDLY keeps
        # the containment and removes the ambiguity.
        top = _open_dir_nofollow(str(base))
    except OSError as exc:
        if exc.errno in _FD_EXHAUSTED_ERRNOS:
            raise
        if exc.errno in (_errno.ELOOP, _errno.ENOTDIR):
            raise ValueError(
                f"Security Error: refusing to walk {base!r} — its final "
                "component is a symlink, so the tree it names is not the "
                "one the caller asked for.") from exc
        return
    stack = [(base, top)]
    try:
        while stack:
            dirpath, dfd = stack.pop()
            try:
                names = sorted(os.listdir(dfd))
                files, subdirs = [], []
                for name in names:
                    try:
                        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                    except OSError:
                        continue
                    import stat as _stat
                    # The stat above is `follow_symlinks=False`, so a symlink is
                    # NEITHER S_ISDIR nor S_ISREG: it is dropped by the positive
                    # tests below and never listed or entered. An explicit
                    # `S_ISLNK: continue` stood here and was deleted as dead —
                    # its mutant survived the §4GJ round-3 battery (R2).
                    if _stat.S_ISDIR(st.st_mode):
                        subdirs.append(name)
                    elif _stat.S_ISREG(st.st_mode):
                        files.append(name)
                yield dirpath, files, dfd
                for name in reversed(subdirs):
                    try:
                        stack.append((dirpath / name, _open_dir_nofollow(name, dir_fd=dfd)))
                    except OSError as exc:
                        # ⚠ RUNNING OUT OF DESCRIPTORS IS NOT "IT WAS SWAPPED"
                        # (§4GK round 4). EMFILE/ENFILE are OSError too, so the
                        # bare `continue` below turned exhaustion into a walk that
                        # returned a SUBSET of the tree with no exception and no
                        # log — the snapshot then reported "the attempt changed no
                        # files" and the ZIP shipped incomplete as a success.
                        if exc.errno in _FD_EXHAUSTED_ERRNOS:
                            raise
                        continue                      # swapped for a link/file mid-walk: skip it
            finally:
                _close_quietly(dfd)
    finally:
        # ⚠ THE FRONTIER, NOT JUST THE CURRENT FRAME (§4GK round 4). Sub-
        # directories are opened and queued BEFORE the generator suspends, so
        # a consumer that leaves the loop early — `break` at a max_files cap,
        # `return`, or a raise — ran only the INNER finally and orphaned every
        # descriptor still queued here, for the life of the process. Measured:
        # 240 descriptors leaked across five capped walks of a 60-directory
        # tree, and the idle project reader breaks at 12 files on every tick.
        for _dirpath, _fd in stack:
            _close_quietly(_fd)
        stack.clear()


def _clear_immutable(path) -> None:
    """Drop the user-immutable flag a release sets (macOS/BSD) from a COPY.
    Never raises."""
    try:
        import stat as _st
        _imm = getattr(_st, "UF_IMMUTABLE", 0)
        if _imm and hasattr(os, "chflags"):
            _fl = os.lstat(path).st_flags
            if _fl & _imm:
                os.chflags(path, _fl & ~_imm, follow_symlinks=False)
    except Exception:                                       # noqa: BLE001
        pass


def copytree_nofollow(src: Path, dest: Path, root: Path, *, ignore=None,
                      dirs_exist_ok: bool = False) -> list:
    """Copy a tree the sandbox controls without ever following a symlink.
    Returns the list of skipped entries (each a short reason string).

    §4GI used `shutil.copytree(symlinks=True, ignore=…)`. That is TOCTOU by
    construction and was reproduced: `copytree` decides symlink-vs-directory
    from the **cached `os.DirEntry`** of the parent's `scandir`, and the
    `ignore` callback runs once per directory BEFORE its entries are copied.
    Swap a directory for a symlink in that window — a `nohup`'d loop or a
    promoted job is the model's version — and the recursive call re-scans
    the path, follows the link, and copies the host tree behind it. A host
    private key landed in the copy on the second attempt while the function
    returned normally.

    This walk is `dir_fd`-relative instead: each child is opened
    ``O_NOFOLLOW`` **relative to its parent's open fd**, so the check and
    the use are one syscall and the parent cannot be re-pointed underneath
    it. An entry swapped between `lstat` and `open` fails the open and is
    SKIPPED, never followed. Non-regular files (FIFO, socket, device) are
    skipped with a reason instead of raising `shutil.Error` mid-tree.

    ``root`` bounds which symlinks are RECREATED: a link resolving outside
    it is dropped, so the copy carries no escape hatch either. ``ignore``
    has `shutil.copytree`'s signature ``(dirpath, names) -> set``.

    Residual, accepted: the TOP-level ``src`` is opened without
    ``O_NOFOLLOW`` because the caller resolved it through `_get_safe_path`,
    which deliberately follows a link that stays inside the sandbox (pinned
    by the top-level control test). Everything below it is refused.
    """
    _require_dir_fd("copy a model-writable tree")
    import stat as _stat
    src_real = Path(os.path.realpath(str(src)))
    dest = Path(dest)
    root_real = Path(os.path.realpath(str(root)))
    skipped: list = []

    def _escapes(dirpath: Path, name: str, dfd: int) -> bool:
        try:
            target = os.readlink(name, dir_fd=dfd)
        except OSError:
            return True
        real = Path(os.path.realpath(str(dirpath / target) if not os.path.isabs(target)
                                     else target))
        return not (real == root_real or root_real in real.parents)

    def _copy_dir(dirpath: Path, dfd: int, dst: Path) -> None:
        names = sorted(os.listdir(dfd))
        ignored = set(ignore(str(dirpath), list(names))) if ignore is not None else set()
        # ⚠ THE DESTINATION-SIDE GUARD MUST COVER THE DIRECTORY TOO (§4GK
        # round 5). Round 4 wrapped only the regular-file write, so
        # `os.makedirs` here and the recursive descent below still aborted
        # the whole tree on the first unexpected OSError — which is what the
        # fix claimed to stop. Reproduced with a 120-level deep workspace
        # (`mkdir d; cd d` in a loop, trivially model-reachable, and the walk
        # handles it fine): ENAMETOOLONG killed every copy of that workspace
        # at 45 levels, `tool_copy_file` returned a bare error over the
        # half-copy, and its own overwrite guard then refused the retry.
        try:
            os.makedirs(dst, exist_ok=dirs_exist_ok)
        except OSError as e:
            # ⚠ THE TOP-LEVEL DIRECTORY IS NOT "AN ENTRY" (§4GK round 6).
            # Round 5 moved this inside a per-entry guard so one unreachable
            # path could not end the whole copy — correct for a CHILD, wrong
            # for the destination root and wrong for a whole-filesystem fault.
            # Measured through the real tool: nothing was copied, the
            # destination did not exist, and the model was told "Copied 'tree'
            # to 'out/copy'" as a PARTIAL with world-changed credit and an
            # idempotency record. A copy that produced no destination is a
            # failure, and so is ENOSPC/EROFS/EDQUOT at any depth.
            if dst == dest or e.errno in _COPY_FATAL_ERRNOS:
                raise
            skipped.append(f"{dirpath}: directory could not be created in the "
                           f"copy ({e.__class__.__name__}) — its contents were "
                           "not copied")
            return
        for name in names:
            if name in ignored:
                continue
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except OSError:
                skipped.append(f"{dirpath / name}: vanished")
                continue
            if _stat.S_ISLNK(st.st_mode):
                if _escapes(dirpath, name, dfd):
                    skipped.append(f"{dirpath / name}: symlink escaping the sandbox")
                    continue
                try:
                    _target = os.readlink(name, dir_fd=dfd)
                except OSError as e:
                    skipped.append(f"{dirpath / name}: {e.__class__.__name__}")
                    continue
                if os.path.isabs(_target):
                    # ⚠ AN ABSOLUTE TARGET INSIDE `root` STILL NAMES THE
                    # SOURCE (§4GK round 4). `_escapes` asks only "does this
                    # resolve inside root", which an absolute link within the
                    # tree satisfies — so it was recreated VERBATIM, and the
                    # copy held a live write channel back into the original.
                    # Reproduced through `_fork_memory_dir`, whose docstring
                    # promises the replay cannot write to the real store: a
                    # write through the copied alias changed the production
                    # skill file. A relative target is safe because it re-
                    # resolves against its own directory inside the COPY.
                    skipped.append(f"{dirpath / name}: absolute symlink "
                                   "(recreating it would point back at the source)")
                    continue
                try:
                    os.symlink(_target, dst / name)
                except OSError as e:
                    skipped.append(f"{dirpath / name}: {e.__class__.__name__}")
                continue
            if _stat.S_ISDIR(st.st_mode):
                try:
                    cfd = _open_dir_nofollow(name, dir_fd=dfd)
                except OSError:
                    # swapped for a symlink (or a file) since the lstat
                    skipped.append(f"{dirpath / name}: not a directory when opened")
                    continue
                try:
                    # ⚠ NO `except OSError` HERE, AND THAT IS DELIBERATE
                    # (§4GK round 6, R2). Round 5 added one as a belt beside
                    # the `os.makedirs` guard above — and no mutant could kill
                    # it, because nothing reaches it: a non-fatal failure at
                    # any depth is already recorded and RETURNED by that
                    # guard, and a fatal errno is re-raised by both. A guard
                    # that cannot be distinguished from its own absence is
                    # dead code, and the protocol's rule for a proven
                    # equivalent mutant is to delete the code rather than
                    # carry a line the battery can only ever report green on.
                    _copy_dir(dirpath / name, cfd, dst / name)
                finally:
                    _close_quietly(cfd)
                continue
            if not _stat.S_ISREG(st.st_mode):
                skipped.append(f"{dirpath / name}: not a regular file "
                               "(FIFO, socket or device node)")
                continue
            try:
                fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0), dir_fd=dfd)
            except OSError:
                skipped.append(f"{dirpath / name}: not a regular file when opened")
                continue
            try:
                fst = os.fstat(fd)
                if not _stat.S_ISREG(fst.st_mode):
                    skipped.append(f"{dirpath / name}: not a regular file when opened")
                    continue
                _block_if_regular(fd, fst)                      # §4SD
                # ⚠ ONE BAD DESTINATION ENTRY MUST NOT ABORT THE TREE
                # (§4GK round 4). The destination-side `open`/`chmod` sat
                # outside any per-entry guard, unlike the source-side opens,
                # so a single unwritable target or an over-long name raised
                # straight out — where `shutil.copytree` collected such
                # errors and carried on. `tool_copy_file` then left a HALF
                # COPY on disk and its own overwrite guard refused the
                # retry: "destination already exists. Delete or rename it."
                try:
                    with open(dst / name, "wb") as out:
                        while True:
                            b = os.read(fd, 1 << 20)
                            if not b:
                                break
                            out.write(b)
                    os.chmod(dst / name, _stat.S_IMODE(fst.st_mode))
                    os.utime(dst / name, (fst.st_atime, fst.st_mtime))
                except OSError as e:
                    if e.errno in _COPY_FATAL_ERRNOS:
                        raise
                    # ⚠ "NOT COPIED" MUST MEAN NOT PRESENT (§4GK round 5). The
                    # guard wraps the source read AND the destination write, so
                    # a failure part-way through left a TRUNCATED file at the
                    # destination while reporting the entry as not copied — and
                    # the fork/clone `copied = sum(... is_file())` counted it.
                    # Reproduced with an EIO injected on the second chunk: a
                    # 1 MB file where the source had 3 MB. Remove the stub, so
                    # the report and the destination agree.
                    try:
                        os.unlink(dst / name)
                    except OSError:
                        pass
                    skipped.append(f"{dirpath / name}: could not be written to "
                                   f"the copy ({e.__class__.__name__})")
                    continue
            finally:
                _close_quietly(fd)
        # ⚠ THE DIRECTORY'S OWN MODE IS RESTORED LAST (§4GK round 5).
        # `shutil.copytree` preserved directory mode bits and mtimes; the
        # dir-fd rewrite took the umask default, so a fork of a chmod'd tree
        # came back with different permissions than its source. The first
        # version of that fix stamped the mode right after `os.makedirs` —
        # which chmods the destination to the source's mode BEFORE writing
        # its children, so copying a RELEASED (read-only, a-w) project
        # workspace created 0o500 directories and then could not write a
        # single file into them: the fork came out empty. Post-order is the
        # only correct order, and `shutil` does the same. Best-effort: a
        # destination we cannot stamp is not a reason to fail the copy.
        try:
            _src_stat = os.fstat(dfd)
            os.chmod(dst, _stat.S_IMODE(_src_stat.st_mode))
            os.utime(dst, (_src_stat.st_atime, _src_stat.st_mtime))
        except OSError:
            pass

    top = os.open(str(src_real), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _copy_dir(src_real, top, dest)
    finally:
        _close_quietly(top)
    return skipped


def _get_safe_path(sandbox_dir: Path, filename: str, *, allow_root: bool = True) -> Path:
    """§4PE (F7): a NUL byte in the name made `.resolve()` raise
    ValueError("embedded null byte"), which twelve handlers returned as that
    bare string — no `Error:` head, no filename, classified OK and
    world-changing. Refuse it in the same voice as every other containment
    refusal, before any path work."""
    if "\x00" in str(filename or ""):
        raise ValueError("Security Error: the path contains a NUL byte and cannot name a file. "
                         "Nothing was changed — send a plain filename.")
    return _get_safe_path_inner(sandbox_dir, filename, allow_root=allow_root)


def _confined_write_block(sandbox_dir, target):
    """§4PG (R2-4): an autonomous leaf (autopilot tick) writes only inside
    its pinned project workspace. `_get_safe_path` re-anchors
    `projects/<other-id>/…` and `/workspace/<root-file>` to the OUTER root —
    legitimate for a conversation, not for a build nobody is watching. None
    when the target resolves under ``sandbox_dir``; a declared refusal
    otherwise. Never raises."""
    try:
        sd = Path(str(sandbox_dir)).resolve()
        t = _get_safe_path(Path(str(sandbox_dir)), str(target)).resolve()
        if t == sd or sd in t.parents:
            return None
    except Exception:  # noqa: BLE001
        return None
    return ToolOutcome.rejected(
        f"SYSTEM BLOCK: this build runs unattended inside its own project workspace — "
        f"'{target}' resolves outside it, so the write was NOT applied. Use a path inside "
        f"the project (a bare name or a sub-folder); nothing was changed.",
        world_changed=False, reason_code="leaf_write_confined")


def _parent_swapped(sandbox_dir, path: Path):
    """§4PF r4 (F15): the containment decision was made on a resolved path;
    if the directory has since become a link that leaves the sandbox, the
    temp file would land outside. Re-resolve the parent NOW and refuse when
    it no longer lies under a sandbox root. None when it is fine."""
    try:
        real = Path(path).parent.resolve()
        roots = _edit_roots(Path(str(sandbox_dir)))
        if any(real == r or r in real.parents for r in roots):
            return None
    except Exception:  # noqa: BLE001
        return None
    return ("Security Error: the directory of this file now resolves outside the sandbox "
            "(it was swapped for a link) — the write was NOT applied (nothing was changed).")


def _fold(name) -> str:
    """§4PE (F2): the comparison form of a path component. APFS folds case
    AND Unicode compatibility forms — `projectſ` (LATIN SMALL LETTER LONG S)
    opens `projects/` — so every name-based guard compares NFKC + casefold,
    which maps both to `projects`; `str.lower()` maps neither."""
    import unicodedata
    return unicodedata.normalize("NFKC", str(name)).casefold()


def _outer_root(sandbox_dir) -> Path:
    """The OUTER sandbox root: from an active project the scoped dir is
    `<root>/projects/<id>`."""
    sd = Path(str(sandbox_dir))
    # §4PF: a project workspace has the shape <root>/projects/<12-hex id> —
    # a sandbox whose parent merely happens to be called "projects" is a root
    if _fold(sd.parent.name) == "projects" and re.fullmatch(r"[0-9a-f]{12}", _fold(sd.name)):
        return sd.parent.parent
    return sd


def _get_safe_path_inner(sandbox_dir: Path, filename: str, *, allow_root: bool = True) -> Path:
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

    PROJECT-SCOPED sandboxes (``sandbox_dir`` = ``<root>/projects/<id>``)
    add one more wrinkle: the container bind-mounts the OUTER ``<root>`` at
    ``/workspace`` even while file ops are scoped to the project dir, so a
    ``/workspace/X`` reference may name a real root-level file (e.g. a repo
    ``execute`` cloned with an absolute path) that the scoped heal made
    unreachable — every file_system op reported the project sandbox EMPTY
    while the shell listed the files fine (2026-07-14 log). Step 1c below
    resolves the ambiguity by existence: prefer the scoped candidate when
    it exists (the historical heal, and the default for brand-new files),
    fall back to the outer root when the file genuinely lives there.
    Host-absolute paths under the outer root anchor the same way in step 0.
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

    # Project-scoping context. When a project is active, ``sandbox_dir`` is
    # ``<root>/projects/<id>`` but the container STILL bind-mounts ``<root>``
    # at ``/workspace`` (see ``_to_container_path``) — so an explicitly
    # root-anchored reference (``/workspace/...`` or a host-absolute path
    # under ``<root>``) can legitimately name a file OUTSIDE the scoped dir,
    # e.g. one ``execute`` created with an absolute ``/workspace/...`` path
    # (a git clone). ``anchor`` is the directory the cleaned name is joined
    # to AND the containment root for the final safety check.
    _outer_root = (sandbox_dir.parent.parent
                   if sandbox_dir.parent.name == "projects" else None)
    anchor = sandbox_dir

    # 0. Host-absolute sandbox path → sandbox-relative. The LLM sometimes
    # echoes the full HOST path of the sandbox (it sees it in a prior
    # sandbox-tree listing / file-read error and passes it back), e.g.
    # "/Users/x/.../sandbox/report.md". Joining that onto sandbox_dir
    # produced a phantom nested path ("<sandbox>/Users/x/.../sandbox/
    # report.md") → ENOENT, then a wasted strike + a retry with the
    # relative form (2026 report-regeneration trace burned ~5 turns on
    # this). If the given absolute path is already inside the sandbox
    # root, strip the prefix and honour the remainder as sandbox-relative.
    # When project-scoped, a host-absolute path under the OUTER root (but
    # outside the scoped dir) anchors to the outer root — it names a real
    # root-level file, not a scoped one. Does NOT touch "/workspace/..."
    # (handled below) — that host path is not under either dir, so the
    # relative_to checks skip it.
    if raw.startswith("/"):
        try:
            _abs = Path(raw).resolve()
            _bases = [sandbox_dir] + ([_outer_root] if _outer_root is not None else [])
            for _base in _bases:
                _sb = _base.resolve()
                try:
                    # relative_to() RAISES when not inside — compute it only
                    # after the containment test, or a non-matching FIRST
                    # base would abort the whole loop through the outer
                    # except before the outer root ever got tried.
                    if _abs == _sb:
                        inside, rel = True, ""
                    elif _abs.is_relative_to(_sb):
                        inside, rel = True, str(_abs.relative_to(_sb))
                    else:
                        inside, rel = False, ""
                except AttributeError:  # Python < 3.9
                    s_abs, s_sb = str(_abs), str(_sb).rstrip("/")
                    inside = s_abs == s_sb or s_abs.startswith(s_sb + "/")
                    rel = "" if s_abs == s_sb else s_abs[len(s_sb) + 1:]
                if inside:
                    raw = rel
                    anchor = _base
                    break
        except Exception:
            pass

    # 1. Strip leading slashes to treat as relative
    clean_name = raw.lstrip("/")

    # 1a. Strip a leading ``workspace/`` prefix (container WORKDIR).
    # Case-sensitive and exact-segment: ``workspaces/`` or
    # ``workspace_xyz/`` are untouched. Applied AFTER the leading
    # slash strip so ``/workspace/foo``, ``workspace/foo``, and
    # ``//workspace/foo`` all collapse identically.
    _had_workspace_prefix = False
    if clean_name == "workspace":
        clean_name = ""
        _had_workspace_prefix = True
    elif clean_name.startswith("workspace/"):
        clean_name = clean_name[len("workspace/"):]
        _had_workspace_prefix = True
    # Pre-heal remainder — the literal container-relative path, needed by
    # step 1c to compute the ROOT candidate before 1b rewrites clean_name.
    _ws_remainder = clean_name

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
    if sandbox_dir.parent.name == "projects" and anchor is sandbox_dir:
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
        # EXCEPTION: a slug whose ``projects/<slug>`` dir actually EXISTS
        # under the outer root is an EXPLICIT reference to another project —
        # collapsing it silently read/wrote the ACTIVE project's file instead.
        # Those re-anchor to the outer root so the path resolves to THAT
        # project's dir; only a slug with no backing dir (the title-guess
        # case) is healed into the active project.
        _m = re.match(r"(?i)projects/([^/]+)/", clean_name)
        if _m:
            _slug = _m.group(1)
            _foreign = (_outer_root / "projects" / _slug
                        if _outer_root is not None and _slug not in (".", "..")
                        else None)
            if _foreign is not None and _foreign.is_dir():
                anchor = _outer_root
            else:
                clean_name = clean_name[_m.end():]

    # 1c. Existence-aware ROOT re-anchor for ``/workspace/...`` paths under
    # project scoping. Such a path is ambiguous: it may be the model's usual
    # shorthand for "my working file X" (healed into the scoped dir — the
    # common write case, kept as the default), or a REAL container-absolute
    # path quoted from ``execute`` output (ls / find / git clone) whose file
    # lives at the sandbox ROOT, outside the project dir. The 2026-07-14 log
    # shows the cost of always healing: a repo cloned to /workspace/analysis
    # was invisible to every file_system op ("sandbox is EMPTY") while the
    # shell saw it fine. Disambiguate by what actually exists on disk:
    #   1. scoped candidate exists                    → scoped (heal wins)
    #   2. ROOT candidate exists                      → root (the clone case)
    #   3. neither exists, but the ROOT candidate's parent dir does and the
    #      scoped one's doesn't                       → root (a NEW file
    #      inside an execute-created tree)
    #   4. otherwise                                  → scoped (heal default)
    if (_had_workspace_prefix and _outer_root is not None
            and anchor is sandbox_dir and _ws_remainder):
        try:
            _scoped_cand = (sandbox_dir / clean_name).resolve()
            _root_cand = (_outer_root / _ws_remainder).resolve()
            if _root_cand != _scoped_cand and not _scoped_cand.exists():
                if _root_cand.exists() or (
                        _root_cand.parent.is_dir()
                        and not _scoped_cand.parent.is_dir()):
                    anchor = _outer_root
                    clean_name = _ws_remainder
        except OSError:
            pass

    # 2. Resolve to absolute path inside the anchor root
    target_path = (anchor / clean_name).resolve()

    # 3. Ensure it's still inside the anchor root (Robust Pathlib Check).
    # ``anchor`` is the scoped sandbox dir normally, or the OUTER sandbox
    # root for explicitly root-anchored paths — both are inside the true
    # sandbox boundary, so traversal is blocked either way.
    _sb_resolved = anchor.resolve()
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

    # 4. Destructive ops must not target a root itself — neither the scoped
    # project root NOR (when scoped) the outer sandbox root, which root-
    # anchored paths can now name directly.
    if not allow_root:
        _roots = {sandbox_dir.resolve()}
        if _outer_root is not None:
            _roots.add(_outer_root.resolve())
        if target_path in _roots:
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
    # §4QA: a GUEST conversation (it named the project) resolves it too
    if isinstance(bound_pid, str) and bound_pid:
        from .projects import project_guests
        if conv in project_guests(sp, bound_pid):
            return bound_pid.strip().lower()
    return ""


#: A project id is an opaque generated token (12 hex chars in practice). It
#: is NOT a path fragment, and it reaches this function straight from a
#: client query parameter on `/api/upload?project_id=`.
_SAFE_PROJECT_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,63}\Z")


def _safe_project_id(pid) -> str:
    """Normalise a project id, or refuse to scope at all.

    ⚠ SECURITY. This used to be `pid.strip().lower()` with no path handling,
    and the value arrives from a client-supplied query parameter. Because the
    scoped directory is built as ``base / "projects" / pid``, pathlib's join
    semantics meant an ABSOLUTE id replaced the base outright and a
    `../..` id walked out of the sandbox — and the upload route's
    `_is_within(sandbox_dir, file_path)` check validated the file against the
    ALREADY-ESCAPED `sandbox_dir`, so it always passed. `mkdir(parents=True)`
    then created the directory. Verified: `project_id=/tmp/abs_pid` yielded
    `/tmp/abs_pid`, and `project_id=../../` left the sandbox entirely.

    Refusing (returning "") falls back to the unscoped sandbox root, which is
    always inside the sandbox — fail CLOSED, and never silently onto a path
    the caller chose.
    """
    if not isinstance(pid, str):
        return ""
    cleaned = pid.strip().lower()
    if not cleaned:
        return ""
    if not _SAFE_PROJECT_ID_RE.match(cleaned):
        import logging as _lg
        _lg.getLogger("GhostAgent").warning(
            "rejected unsafe project id %r — not scoping", pid[:80])
        return ""
    return cleaned


def project_scoped_sandbox(context, stateful: bool = False, explicit_project_id=None):
    """Return ``(host_dir, container_workdir)`` scoped to the active project's
    workspace (``<sandbox>/projects/<id>/``), creating the dir on demand; or
    ``(sandbox_root, None)`` when no project is active (or ``stateful=True``,
    since the Jupyter kernel is pinned to ``/workspace``).

    ``explicit_project_id`` (when a non-empty string) scopes to THAT project
    directly, bypassing the process-global ``current_project_id``. The
    stateless ``/api/upload`` / ``/api/download`` endpoints carry no
    conversation context, so they read the global — which a concurrent
    conversation's switch/reconcile can point at the wrong project, landing an
    upload in another conversation's sandbox. A client that passes its own
    ``project_id`` gets a race-free scope (the global stays the fallback).

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
    # An explicit caller-supplied project id wins over the racy global.
    if isinstance(explicit_project_id, str) and explicit_project_id.strip():
        pid = _safe_project_id(explicit_project_id)
    else:
        pid = _safe_project_id(getattr(context, "current_project_id", None))
    if not pid and not stateful and explicit_project_id is None:
        # ``current_project_id`` is process-global and can be cleared MID-REQUEST
        # by a concurrent conversation's reconcile (observed live: a 700s
        # autoadvance+manual build had its scoping wiped, so the agent's file
        # writes landed in the sandbox ROOT and it thrashed for minutes). The
        # per-conversation binding is NOT stomped that way, so re-derive this
        # conversation's project from it.
        pid = _safe_project_id(_conversation_bound_project(context))
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

def _scoped_root_fallback(sandbox_dir: Path, filename):
    """Resolve ``filename`` against the OUTER sandbox root when a project-
    scoped lookup missed. Returns the resolved root path when it exists
    there, else ``None`` (always ``None`` for an unscoped sandbox).

    READ-ONLY callers use this to serve files ``execute`` created via
    absolute ``/workspace/...`` paths — those land at the root, not the
    project dir, and were previously invisible to every file_system op
    (the 2026-07-14 "sandbox is EMPTY" log). Destructive ops must NOT use
    it: cross-scope edits stay explicit via a ``/workspace/...`` path."""
    sd = Path(sandbox_dir)
    if sd.parent.name != "projects":
        return None
    root = sd.parent.parent
    try:
        cand = _get_safe_path(root, filename)
        if cand.exists() and cand != root.resolve():
            return cand
    except (ValueError, OSError):
        pass
    return None


def _outer_root_files_hint(sandbox_dir: Path, limit: int = 8) -> str:
    """When a project-scoped workspace lookup comes up empty, name a few
    entries that DO exist at the sandbox ROOT (outside ``projects/``) so the
    model learns the ``/workspace/...`` route instead of concluding its files
    vanished. Empty string when unscoped or nothing relevant exists."""
    sd = Path(sandbox_dir)
    if sd.parent.name != "projects":
        return ""
    root = sd.parent.parent
    names = []
    try:
        for entry in sorted(os.listdir(root)):
            if entry.startswith(".") or entry in (
                    "projects", "__pycache__", "node_modules", "venv", "env"):
                continue
            names.append(entry + ("/" if (root / entry).is_dir() else ""))
            if len(names) >= limit:
                break
    except OSError:
        return ""
    if not names:
        return ""
    return (
        " NOTE: the sandbox ROOT (outside this project's dir) DOES contain: "
        + ", ".join(names)
        + ". Files created by `execute` with absolute /workspace/... paths "
        "land there, not in this project's workspace. Reach them with "
        "container-absolute paths, e.g. file_system(operation='list_files', "
        f"path='/workspace/{names[0].rstrip('/')}') or operation='read', "
        "path='/workspace/<file>'."
    )


def _sandbox_where(sandbox_dir) -> str:
    """Name the directory a missing-file message is about, as the model
    addresses it: the project workspace when ``sandbox_dir`` is scoped
    (``<root>/projects/<id>``), else the sandbox root.

    §4HD (2026-09-15, req 9b6b8757): the message said "the current
    project's sandbox" for EVERY directory. After the operator cleared the
    root sandbox, an unscoped read came back "does not exist in the
    current project's sandbox … a DIFFERENT project/session", and the
    model — which had no project open — spent a turn reconciling a
    project it was not in. Same detection as `project_download_prefix`,
    `_outer_root_files_hint` and the path heal, so all of them agree."""
    pfx = project_download_prefix(sandbox_dir)
    if pfx:
        return f"the project workspace '{pfx}'"
    return "the sandbox root (/workspace, no project workspace is active)"


#: §4IB — absolute container paths this tool CAN reach. Anything else that
#: starts with "/" is a real file `execute` may have just listed, and this
#: tool must say "outside", not "does not exist" (req 21b295ef: a header
#: under /usr/local/lib/python3.11/site-packages was reported non-existent
#: one turn after `find` printed it — a strike and a wasted turn).
_INSIDE_ABS_PREFIXES = ("/workspace", "/sandbox")


#: paths that name the USER's machine (macOS host), not the container —
#: neither file_system nor execute can reach them (§4LL review)
_HOST_MACHINE_PREFIXES = ("/users/", "/volumes/", "/private/", "/library/", "/applications/")
#: …and absolute paths that a MUTATION must never take as "the sandbox root
#: plus this path" (the leading-slash heal silently wrote ~/Desktop/notes.txt
#: to <sandbox>/Users/…/Desktop/notes.txt and reported it written to the Desktop)
_HOST_MUTATION_PREFIXES = _HOST_MACHINE_PREFIXES + ("/home/", "/tmp/", "/var/", "/etc/", "/opt/",
                                                    "/usr/", "/root/", "/system/")


def _host_path_block(sandbox_dir, target):
    """A refusal for a mutation whose path names a place outside the sandbox
    (the user's machine or the container's system dirs), else None. A path
    that resolves inside the sandbox or its outer root is fine."""
    raw = str(target or "").strip()
    if not raw:
        return None
    norm = raw.replace("\\", "/")
    if not (norm == "~" or norm.startswith("~/") or norm.lower().startswith(_HOST_MUTATION_PREFIXES)):
        return None
    try:
        # APFS is case-insensitive (review); normcase folds only on Windows
        _nc = (lambda p: str(p).lower()) if sys.platform == "darwin" else (lambda p: os.path.normcase(str(p)))
        r = _nc(Path(norm).expanduser().resolve())
        sb = Path(sandbox_dir).resolve()
        outer = sb.parent.parent if sb.parent.name == "projects" else sb
        for base in {_nc(sb), _nc(outer)}:
            if r == base or r.startswith(base.rstrip(os.sep) + os.sep):
                return None
    except Exception:  # noqa: BLE001
        pass
    return ToolOutcome.rejected(
        f"Error: '{raw}' is outside your sandbox — on the user's own machine or in the container's system "
        f"folders — so you cannot write, move or delete it. Use a path inside your sandbox instead (e.g. "
        f"'{Path(norm).name or 'file.txt'}'), and tell the user the file is in your sandbox, where they can "
        f"download it.", reason_code="host_path_refused")


def _download_error(code, url) -> str:
    """§4LL: "Error 429 - Failed to download" gave no next step; live the
    model fetched the same host with curl, saved a 166-byte error page as the
    "photo", and told the user it used the real photo."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        c = 0
    why = ("the site is rate-limiting this exit (429)" if c == 429 else
           "the site refused the download (403)" if c == 403 else
           "the file is not at that address (404)" if c == 404 else
           f"the server answered {c}")
    return (f"Error {c} - Failed to download from {url}: {why}. Nothing was saved. Do not retry the "
            f"same URL (with execute/curl either — it fails the same way); use a different source, "
            f"and if none works, tell the user the file could not be downloaded.")


def outside_workspace_message(filename) -> str:
    """The message for an absolute path that lies outside the sandbox
    workspace, or "" when the path is relative / inside. One home: every
    missing-file branch reads it through `_missing_file_message`."""
    raw = str(filename or "").strip()
    if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.IGNORECASE):
        # a URL is not a file (§4ME F8: URLs passed to file_system read
        # "not found" and the model kept guessing paths)
        return (
            f"Error: '{raw}' is a URL, not a file in your sandbox. To READ a web page use the "
            f"`browser` tool (or web_search); to SAVE a file from the web use file_system "
            f"operation='download' with url='{raw}'."
        )
    if not raw.startswith("/"):
        return ""
    norm = raw.replace("\\", "/")
    for pfx in _INSIDE_ABS_PREFIXES:
        if norm == pfx or norm.startswith(pfx + "/"):
            return ""
    if norm.lower().startswith(_HOST_MACHINE_PREFIXES):
        # the container has no /Users: `cat` there fails too (§4LL review)
        return (
            f"Error: '{raw}' is a path on the user's own computer, not in your sandbox. Neither "
            f"file_system nor execute can reach it. Ask the user to upload the file (or paste its "
            f"text) and then read it from your sandbox."
        )
    return (
        f"Error: '{raw}' is OUTSIDE the sandbox workspace. file_system reads "
        f"only files under /workspace (and the project workspaces inside it); "
        f"a path that `execute`, `ls` or `find` printed under /usr, /etc, "
        f"/opt, /home, /root or site-packages is real in the container but "
        f"this tool cannot open it. Read it with execute instead: "
        f"`cat '{raw}'` or `sed -n '1,120p' '{raw}'` (or `grep -n <pattern> "
        f"'{raw}'` to find the part you need)."
    )


_HINT_SCAN_MAX = 5000      # files walked when looking for a same-named file
_MAX_TAIL_PARTS = 32       # longest tail tried (one resolve() per tail)


def _suffix_match(norm: str, sandbox_dir) -> Optional[str]:
    """§4QB: the longest tail of ``norm`` (leading directories dropped) that
    exists inside the sandbox — a file or a directory — else None. A tail
    with a '..' or hidden part, or resolving outside the sandbox (a symlink),
    never matches."""
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if len(parts) < 2:
        return None
    try:
        root = Path(sandbox_dir).resolve()
    except Exception:  # noqa: BLE001 — best-effort hint
        return None
    for k in range(max(1, len(parts) - _MAX_TAIL_PARTS), len(parts)):
        if any(p == ".." or p.startswith(".") for p in parts[k:]):
            continue
        cand = "/".join(parts[k:])
        try:
            # r2: per tail — a symlink loop on a long tail must not hide a short one
            target = (root / cand).resolve()
            if target.is_relative_to(root) and target.exists():
                return cand
        except Exception:  # noqa: BLE001 — best-effort hint
            continue
    return None


def _basename_match(base: str, sandbox_dir) -> Optional[str]:
    """§4QB: the ONE file named ``base`` in the sandbox (bounded walk, same
    skipped directories as the listing), else None — a name found twice
    (__init__.py, README.md) names no file, and a hidden name is never
    suggested. "Twice" folds case and Unicode form (r6: on APFS 'readme.md'
    also opens README.md) and counts a symlinked copy, which is itself never
    the one suggested."""
    if not base or base.startswith("."):
        return None
    import unicodedata

    def _key(name):
        return unicodedata.normalize("NFC", name).casefold()
    want = _key(base)
    found = None
    copies = 0
    try:
        import os
        seen = 0
        for root, dirs, files in os.walk(sandbox_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                             and d not in ("__pycache__", "node_modules", "venv", "env"))
            for f in files:
                if _key(f) != want:
                    continue
                copies += 1
                if copies > 1:
                    return None
                if f == base and not os.path.islink(os.path.join(root, f)):
                    found = os.path.relpath(os.path.join(root, f), sandbox_dir)
            seen += len(files) + len(dirs)
            if seen >= _HINT_SCAN_MAX:
                return None
    except Exception:  # noqa: BLE001 — best-effort hint
        return None
    return found


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
    _outside = outside_workspace_message(filename)
    if _outside:
        return _outside
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
    where = _sandbox_where(sandbox_dir)
    if existing:
        listing = "Files that DO exist here: " + ", ".join(existing[:20]) + "."
    else:
        # Reads directly after "does not exist in <where>." — the place is
        # already named, so this line must not name it a second time.
        listing = ("It is currently EMPTY (no files)."
                   + _outer_root_files_hint(sandbox_dir))
    # Title-as-directory catch (2026-08-01): two consecutive live requests
    # read "Mini AI/…" / "Mini AI v2/…" — the model prefixes the PROJECT
    # TITLE as a directory even though the workspace root IS the project.
    # When the requested path has a directory prefix and a file with the
    # same basename exists, name the exact correction up front (the generic
    # message steered toward CREATING a duplicate instead).
    _norm = str(filename or "").replace("\\", "/").strip("/")
    _base = _norm.rsplit("/", 1)[-1] if "/" in _norm else ""
    prefix_hint = ""
    try:
        prefix_hint, _hinted = _prefix_hint(_norm, _base, sandbox_dir) if _base else ("", "")
    except Exception:  # noqa: BLE001 — the hint is best-effort; the message is not
        prefix_hint, _hinted = "", ""
    return (
        f"Error: '{filename}' does not exist in {where}."
        f"{prefix_hint} "
        f"{listing} The live sandbox is AUTHORITATIVE — if a workspace narrative, "
        f"memory, prior session, or DYNAMIC SYSTEM STATE hint referenced "
        f"'{filename}', it was written in a DIFFERENT project or session (or has "
        f"since been removed) and does NOT exist here now. Do NOT read this path "
        f"again (it will keep failing). To proceed: " + (
            f"use the relative path '{_hinted}' (see the NOTE), or "
            if prefix_hint else "") +
        f"CREATE it with file_system(operation='write', filename='{filename}', "
        f"content=…), or pick a file from the list above."
    )


def _round_trips(rel: str, sandbox_dir) -> bool:
    """§4QB r5: ``rel`` is offered only when the tool's own path resolver
    opens exactly that entry — _get_safe_path rewrites a leading
    'workspace/' or 'projects/<x>/', so 'workspace/notes.md' would open the
    ROOT notes.md (a different file)."""
    try:
        root = Path(sandbox_dir).resolve()
        return _get_safe_path(sandbox_dir, rel).resolve() == (root / rel).resolve()
    except Exception:  # noqa: BLE001
        return False


def _prefix_hint(norm: str, base: str, sandbox_dir) -> Tuple[str, str]:
    """§4QB: (NOTE, suggested path) for a wrong path prefix, or ("", "").

    The path minus its leading directories is checked directly (a live
    "448d5afa86f0/results/harness-scores.md" got no hint: the 20-file
    listing filled with research/ before results/, and a directory never
    matched a FILE listing). CONFIDENT only when the dropped part is exactly
    this project's id and no such directory really exists; anything else may
    be a different file of the same name (review rounds r1–r4 each found
    one), so it is HEDGED — the model gets the path either way. A path is
    offered only if the tool would open exactly that entry (r5)."""
    sb = Path(sandbox_dir)
    tail = _suffix_match(norm, sandbox_dir)
    if tail and not _round_trips(tail, sandbox_dir):
        tail = None
    clean = "/".join(p for p in norm.split("/") if p not in ("", "."))
    dropped = clean[:-len(tail) - 1] if tail else ""
    # (r4: a write with the id prefix makes a REAL <id>/ directory — then
    # "not a directory here" is false, so that case is hedged too; r5: a
    # project workspace is <root>/projects/<12-hex id>, as _outer_root says)
    if (tail and _outer_root(sb) != sb and dropped == sb.name
            and not os.path.lexists(sb / dropped)):
        return (f" NOTE: '{tail}' DOES exist — you are ALREADY inside this project's "
                f"workspace, so '{dropped}/' (the project's id) is not a directory here. "
                f"Use the relative path '{tail}'.", tail)
    if tail and "/" not in tail and not (sb / tail).is_dir():
        tail = None          # a bare root FILE is a name match: only a unique name counts
    match = tail or _basename_match(base, sandbox_dir)
    if not match or (match != tail and not _round_trips(match, sandbox_dir)):
        return "", ""
    return (f" NOTE: '{match}' exists in this workspace — if that is the one you mean, "
            f"use the relative path '{match}'.", match)


def read_byte_budget(max_context: int) -> int:
    """Bytes of raw file content allowed into the model's context — used for
    BOTH the per-file read cap and the per-batch cumulative allowance.

    Sized to a fraction of the window (chars ≈ tokens * 3.5) so raw reads
    can't crowd out the system prompt, the model's reasoning, the other tool
    outputs in the turn, and the response itself. The old per-file factor was
    0.5 (≈70% of the window for a single file); two such reads in one turn
    overflowed a 131 K window at 136 K tokens. 0.40 leaves headroom and, paired
    with the cumulative ReadBudget, makes parallel whole-file reads safe."""
    # § context R1 B2: the 150 KB floor is BOUNDED by 0.8x the window —
    # for small windows the unconditional floor INVERTED the cap (a 100 KB
    # file returned whole into an 8k-token window). Large windows unchanged.
    return min(max(150000, int(max_context * 3.5 * 0.40)),
               max(1, int(max_context * 3.5 * 0.80)))


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


# (§4NN: the `fs_batch` BATCH macro — multi-path `paths` reads and the
# post-edit view — shipped only as an experiment arm and was removed with the
# experiments registry; `paths` is ignored as it always was in the control arm.)


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
    if re.match(r"(?i)https?://", str(filename).strip()):        # §4LL: not http_log.txt
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
        _fb_note = ""
        if not path.exists():
            # Project-scoped miss, but the file exists at the sandbox ROOT
            # (created by `execute` via an absolute /workspace/... path):
            # serve it read-only with a note, instead of the dead-end
            # "sandbox is EMPTY" error the 2026-07-14 session looped on.
            fb = _scoped_root_fallback(sandbox_dir, filename)
            if fb is None:
                return _missing_file_message(filename, sandbox_dir)
            path = fb
            _fb_note = (
                f"NOTE: '{filename}' is not in the current project's "
                f"workspace; serving the file at the sandbox ROOT "
                f"(container path '{_to_container_path(sandbox_dir, fb)}'). "
                f"Use that /workspace/... path to reference it in other "
                f"file/shell operations.\n")

        # ⚠ TYPE CHECK BEFORE ANY BRANCH. `read` is the repair every gate
        # refusal names, so a wedge here is reachable from the guard's own
        # prescribed recovery. The ranged read, the >96 KB sampler and the
        # whole-file read all open the path below.
        _nr = _nonregular_refusal(path, filename)
        if _nr:
            return _nr

        # LINE-RANGE READ (#11): a bounded slice, EXEMPT from the whole-file
        # size cap — this is the cheap recovery path after a failed replace or
        # a too-large-file error. The range is streamed (we stop reading at
        # end_line) so a 300 KB file costs only the requested lines, and the
        # slice is returned with line-number prefixes so it chains directly
        # from `rg --line-number` output and the replace-failure snippet.
        if start_line is not None or end_line is not None:
            _rng = await asyncio.to_thread(
                _read_line_range, path, filename, start_line, end_line)
            # §4KB: a ranged read grounds an edit. Success is detected by
            # this module's OWN error protocol — every failure return in
            # file_system.py begins with "Error:", the same convention
            # `tools.tool_failure.result_is_failure` reads project-wide, and
            # `test_edit_read_receipts.py` enumerates _read_line_range's
            # failure modes against it. Not an invented proxy.
            if _fb_note and not _rng.startswith("Error"):
                _rng = _fb_note + _rng
            return _rng

        file_size = path.stat().st_size

        # Generated/data-shaped file sampling (2026-07-18): embedded data
        # tables (dat_tilesPC.c-style hex arrays, giant JSON blobs, minified
        # bundles) carry thousands of tokens with near-zero analytical value
        # per line — reading them whole is what blew the xrick feasibility
        # session to 398k tokens. Runs BEFORE the per-file cap so an
        # over-cap data table returns a USEFUL sample instead of the generic
        # too-large error. Heuristics on the 8 KB head: dense "0x" literals
        # (a C data table) or very long average lines (minified/single-line
        # data). Ranged reads and search remain available for specifics.
        # § context R1 B6: under a ZERO budget (context-pressure lockdown)
        # the sampler must not run — "whole-file reads DISABLED" has to mean
        # the sample too; the refusal below explains the state honestly.
        if file_size > 96 * 1024 and not (
                read_budget is not None and read_budget.remaining <= 0):
            try:
                _s_head = await asyncio.to_thread(_read_head, path, 8192)
            except OSError as oe:
                return f"Error: failed to read '{filename}': {oe}"
            if not _looks_like_binary(_s_head):
                _s_txt = _s_head.decode("utf-8", "replace") if isinstance(
                    _s_head, (bytes, bytearray)) else str(_s_head)
                _head_lines = _s_txt.split("\n")
                _avg_line = len(_s_txt) / max(1, len(_head_lines))
                _hex_density = _s_txt.count("0x")
                if _avg_line > 240 or _hex_density > 300:
                    _shape = ("dense hex data table" if _hex_density > 300
                              else f"very long lines (avg ≈{_avg_line:.0f} chars)")
                    sample = _s_txt[:4096]
                    if read_budget is not None:
                        read_budget.charge(len(sample))
                    return (
                        f"{_fb_note}--- {filename} SAMPLE ONLY ({file_size / 1024:.0f} KB, "
                        f"looks machine-generated: {_shape}) ---\n{sample}\n"
                        f"--- END SAMPLE (first 4 KB of {file_size / 1024:.0f} KB) ---\n"
                        f"Reading this file whole would waste the context window on "
                        f"generated data. Use operation='search' for specific symbols, "
                        f"a ranged read (start_line/end_line), or an 'execute' script "
                        f"to compute whatever digest you actually need."
                    )

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
        # § context R1 B2: NO first-read exemption. The old qualifier
        # `(spent > 0 or remaining <= 0)` let the FIRST read of a batch
        # ignore the occupancy-shrunk budget entirely (170 KB returned
        # against a 3 KB remaining budget, 57x) — the shrink only ever bit
        # the second read. Any read that exceeds the remaining budget is
        # refused; ranged reads/search stay exempt by design.
        if read_budget is not None and file_size > read_budget.remaining:
            if read_budget.spent == 0 and read_budget.remaining > 0:
                return (
                    f"Error: Reading '{filename}' ({file_size / 1024:.1f} KB) is "
                    f"refused — the conversation is near the context ceiling and "
                    f"only {read_budget.remaining / 1024:.1f} KB of whole-file "
                    f"read budget remains this turn. Pull only what you need: "
                    f"operation='search', a ranged read (start_line/end_line — "
                    f"exempt from this cap), or an 'execute' script that prints "
                    f"a compact digest."
                )
            if read_budget.spent == 0:
                # Zero capacity BEFORE any read: the conversation itself is
                # already near the context ceiling (occupancy-aware budget,
                # 2026-07-18) — bulk gathering must stop, not just pause.
                return (
                    f"Error: Reading '{filename}' is refused — the conversation is "
                    f"already near the context ceiling, so there is NO whole-file "
                    f"read budget left this turn. STOP bulk-reading. Instead: "
                    f"(1) write what you have learned so far into a compact notes "
                    f"file in the project (file_system write, e.g. 'analysis_notes.md') "
                    f"so it survives compaction; (2) pull only targeted evidence — "
                    f"operation='search', a ranged read (start_line/end_line — exempt "
                    f"from this cap), or an 'execute' script that prints a ~50-line "
                    f"digest; (3) if you already have enough, produce the deliverable NOW."
                )
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
        return f"{_fb_note}--- {filename} CONTENTS ---\n{content}"
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

# §4HB — the payload of a rejected `replace` is HELD so the corrective
# `write` does not regenerate it. Live (req 6a7882f5): operation='replace'
# arrived with the whole 13.7 k-char report as `content` and no
# `replace_with`; rejected (correctly — 8 of 13 such payloads in the corpus
# were FRAGMENTS, so promoting them would overwrite whole files), then the
# model produced the same 13.6 k chars again as a write. 124 s, one fifth of
# the turn, spent on a payload the tool had already received. The hold is
# keyed by the resolved path, single-slot per path, and expires; the model
# redeems it by name, so nothing is written the model did not ask to write.
_HELD_CONTENT_TOKEN = "<<HELD>>"
def _nonregular_refusal(path: Path, filename: str) -> Optional[str]:
    """Refusal when ``path`` exists and is not a regular file, else None.

    ONE helper, because this wedge was closed three times and reopened three
    times: `open()` on a FIFO blocks forever waiting for a writer, so every
    site that opens a model-named path needs the check BEFORE the open, and
    every site that forgot it reopened the hole — the binary sniff, `read`'s
    own sniff, the auto-promote's `write_text`, and the symbol indexer.
    `TimeoutError` is an `OSError`, so nothing can rescue it afterwards.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None                     # missing: the caller's own problem
    if stat_mod.S_ISREG(st.st_mode):
        return None
    if stat_mod.S_ISDIR(st.st_mode):
        # §4LL: name the next step for the common case
        return (f"Error: '{filename}' is a directory, not a file — name a file inside it "
                f"(file_system(operation='list_files', path='{filename}') shows them). "
                f"'{filename}' is unchanged.")
    return (f"Error: '{filename}' is not a regular file (it is a FIFO, "
            f"device, socket or directory), so it cannot be read or written "
            f"as text. '{filename}' is unchanged.")


_HELD_CONTENT_TTL_S = 600.0
_HELD_CONTENT_MAX = 8
_HELD_REPLACE_CONTENT: Dict[str, Tuple[float, str]] = {}


def _hold_rejected_content(sandbox_dir: Path, filename: str, payload: str) -> str:
    """Remember `payload` for `filename`; returns the hold key or ""."""
    try:
        key = str(_get_safe_path(sandbox_dir, filename))
    except Exception:  # noqa: BLE001 — a hold must never mask the rejection
        return ""
    if not payload:
        return ""
    now = time.monotonic()
    # Sweep expired and bound the map: a runaway loop of rejections must
    # not keep an unbounded number of 10-minute payloads alive.
    for k, (t, _) in list(_HELD_REPLACE_CONTENT.items()):
        if now - t >= _HELD_CONTENT_TTL_S:
            _HELD_REPLACE_CONTENT.pop(k, None)
    _HELD_REPLACE_CONTENT.pop(key, None)
    _HELD_REPLACE_CONTENT[key] = (now, str(payload))
    while len(_HELD_REPLACE_CONTENT) > _HELD_CONTENT_MAX:
        oldest = next(iter(_HELD_REPLACE_CONTENT))
        if oldest == key:
            break
        _HELD_REPLACE_CONTENT.pop(oldest, None)
    return key


def held_content_chars(sandbox_dir, filename: str) -> int:
    """Size of the un-redeemed hold for ``filename``, 0 if none or expired.
    A PEEK — nothing is consumed. §4HE (2026-09-15, req 5fa6aa97): after a
    rejected `replace` the model re-READ the file instead of redeeming,
    the no-progress breaker steered it to "the mutating action" in the
    abstract, and it delivered without ever writing. The breaker can now
    name the exact write that is pending."""
    try:
        key = str(_get_safe_path(sandbox_dir, filename))
    except Exception:  # noqa: BLE001
        return 0
    rec = _HELD_REPLACE_CONTENT.get(key)
    if rec is None or time.monotonic() - rec[0] >= _HELD_CONTENT_TTL_S:
        return 0
    return len(rec[1])


def _redeem_held_content(sandbox_dir: Path, filename: str) -> Optional[str]:
    """The held payload for `filename`, consumed on read; None if absent
    or expired. A hold is redeemed ONCE — a second `<<HELD>>` write for the
    same path gets a rejection, never a stale replay."""
    try:
        key = str(_get_safe_path(sandbox_dir, filename))
    except Exception:  # noqa: BLE001
        return None
    rec = _HELD_REPLACE_CONTENT.pop(key, None)
    if rec is None:
        return None
    t, content = rec
    if time.monotonic() - t >= _HELD_CONTENT_TTL_S:
        return None
    return content


async def tool_replace_text(filename: str, old_text: str, new_text: str,
                            sandbox_dir: Path, *, project_store=None):
    """Targeted edit — thin wrapper whose only job is to record ONE ladder row.

    §4KB. The edit itself lives in `_replace_text_impl`, which has ~20 return
    paths; recording at each of them would be the instance-fix this project's
    R1 forbids, and the next path added would silently go unrecorded. So the
    impl writes what it did into a mutable ``telem`` dict and this wrapper —
    the single place every path must pass through — writes the row.

    The ledger never changes the outcome: a failed write is swallowed, and the
    result object is returned untouched whether or not the row landed.
    """
    telem: Dict[str, Any] = {"strategies": [], "blocks_total": 0,
                             "blocks_applied": 0, "file_len": 0}
    try:
        result = await _replace_text_impl(
            filename, old_text, new_text, sandbox_dir,
            telem=telem, project_store=project_store)
    except BaseException as exc:                            # noqa: BLE001
        # An exception is an outcome too, and the one most worth counting.
        # Recorded, then re-raised unchanged — this wrapper must not swallow
        # a failure the caller is written to see.
        record_edit(path=str(filename), op="replace", applied=False,
                    reason=f"exception:{type(exc).__name__}",
                    search_len=len(str(old_text or "")),
                    file_len=int(telem.get("file_len") or 0),
                    req_id=str(request_id_context.get() or ""))
        raise
    # A matched rung is NOT an applied edit: `_write_replace_guarded` rolls
    # back on a marker leak or a syntax regression and returns REJECTED. The
    # ladder row must say "matched fuzzy, did NOT land", or the rollback
    # guards would read as tolerance successes.
    # A matched rung is NOT an applied edit. Two ways it is not:
    #   * the guard rolled the write back and returned REJECTED;
    #   * the write RAISED and unwound to the generic `except` that returns a
    #     plain "Error: ..." string — measured live on a 0444 file, which
    #     recorded `applied: True, strategies: ['exact']` over a file that was
    #     byte-identical on disk.
    # "Error:" is this module's own failure protocol (the same one
    # `tools.tool_failure.result_is_failure` reads project-wide), not an
    # invented text proxy.
    applied = _edit_applied(result, telem)
    # A TOLERANT apply is the event this whole ledger exists to price, so it
    # is also the one the operator should be able to SEE happening rather
    # than reconstruct from a JSONL afterwards. Exact applies stay quiet —
    # they are the common case and logging them would be noise.
    _applied_rungs = [str(x) for x in (telem.get("strategies") or [])]
    _tolerant = [x for x in _applied_rungs if not x.startswith("exact")]
    if applied and _tolerant:
        pretty_log(
            "Tolerant Match",
            f"{filename}: applied on {', '.join(_tolerant)} — did NOT "
            f"byte-match; verify the edited region is the intended one",
            icon=Icons.WARN, level="WARNING")
    record_edit(
        path=str(filename), op=str(telem.get("op") or "replace"),
        applied=applied,
        strategies=list(telem.get("strategies") or []),
        # The ledger's contract is that `reason` explains a NON-applied row.
        # A mixed batch (one block landed, one ambiguous) wrote applied=True
        # WITH reason="ambiguous_block", corrupting any reason-keyed analysis
        # of applied edits.
        # L1: the OUTCOME's reason_code wins. `telem["reason"]` is set
        # per-block inside the loop, so a batch whose guard then ROLLED BACK
        # logged "ambiguous_block" while the real verdict was
        # `syntax_regression_rolled_back`, mis-attributing rollbacks to
        # ambiguity in the report's histogram.
        reason=("" if applied else
                str(_outcome_reason(result) or telem.get("reason") or "")),
        search_len=len(str(old_text or "")),
        file_len=int(telem.get("file_len") or 0),
        blocks_total=int(telem.get("blocks_total") or 0),
        blocks_applied=int(telem.get("blocks_applied") or 0),
        req_id=str(request_id_context.get() or ""),
    )
    # §4PF (F13): every replace result names its own call too — parallel
    # replaces on one file were misbindable (the edit wrapper had this tag)
    _txt = str(result)
    if "This call's old_string" not in _txt:
        from .outcome import with_text as _with_text
        result = _with_text(result, _txt.rstrip() + f" (This call's old_string {_old_tag(old_text)}.)")
    return result


def _outcome_reason(result: Any) -> str:
    """The `reason_code` a ToolOutcome carries, else "" — never the message.

    Deliberately NOT parsed out of the result TEXT. This project has a
    standing lesson about lexical proxies for semantic properties; a ledger
    that greps its own success strings would break the moment a message is
    reworded, and would do it silently.
    """
    try:
        rc = getattr(result, "reason_code", "")
        return str(rc or "")
    except Exception:                                       # noqa: BLE001
        return ""


def _edit_applied(result: Any, telem: Dict[str, Any]) -> bool:
    """The ONE rule that decides whether a ledger row says `applied`.

    Shared by `tool_replace_text` and `tool_edit_text` so the two operations
    are measured by the same yardstick — a second copy of this rule would be
    the first place the `edit` vs `replace` comparison drifts.

    A matched rung is NOT an applied edit. Two ways it is not:
      * the guard rolled the write back and returned REJECTED;
      * the write RAISED and unwound to a handler that returns a plain
        "Error: ..." string — measured live on a 0444 file, which recorded
        `applied: True, strategies: ['exact']` over a byte-identical file.
    "Error:" is this module's own failure protocol (the same one
    `tools.tool_failure.result_is_failure` reads project-wide), not an
    invented text proxy.
    """
    _failed = str(result).lstrip().startswith("Error:")
    return (bool(telem.get("strategies")) or bool(telem.get("blocks_applied"))) \
        and not bool(getattr(result, "is_rejection", False)) \
        and not _failed


# ══════════════════════════════════════════════════════════════════════
# operation='edit' — §4KC (2026-09-23)
#
# What 102 real `replace` calls in the §4KB ledger showed: 46% rejected,
# and the rejections were INTERFACE failures, not matching failures —
# `missing_replace_with` 13/47, whole files pushed through the envelope
# (27% of applies were `write_promote`), not-found 11/47. The envelope's
# original justification (survive native tool-call argument corruption)
# died on 2026-07-31 when the transport was fixed; the ladder's tolerance
# rungs applied 4/61 times and are the ONLY edits no result-guard can
# see land in the wrong place.
#
# So `edit` is the claude-code shape and nothing else: `old_string` +
# `new_string`, both REQUIRED, exact and unique or REJECTED with the
# evidence a model needs to fix the call (where the nearest text is, what
# differs). No aliases, no envelope, no ladder, no auto-promote. `replace`
# stays for one measurement cycle; the ledger records `op` per row, so
# the two are comparable on the same yardstick (`_edit_applied`).
# ══════════════════════════════════════════════════════════════════════

#: A file this long, with `old_string` covering this share of it BY BOTH
#: measures (chars AND lines), is a rewrite wearing an edit's clothes:
#: refused, naming `write`. Short files are exempt — replacing all three
#: lines of a three-line file is an edit. Both axes, because a minified
#: bundle or a data URI puts most of a 30-line file's CHARS on one line and
#: a one-line edit of it is not a rewrite (fresh-eye review, §4KC round 2).
EDIT_WHOLE_FILE_MIN_LINES = 30
EDIT_WHOLE_FILE_SHARE = 0.8

#: How many occurrence lines an ambiguity rejection names.
_EDIT_AMBIGUITY_LINE_CAP = 8

#: The unambiguous envelope signature. NOT a bare `====` or `>>>>` line:
#: `====` is a Markdown setext underline and an edit must be free to write
#: one — that over-match was caught by this operation's own battery.
_ENVELOPE_MARKER_RE = re.compile(r"^<{4,}[ \t]*SEARCH[ \t]*$", re.MULTILINE)

#: The miss diagnosis scans the file several times; on a 50 MB file that is
#: seconds of blocked loop. It runs in a thread AND only over this much of
#: the head — a diagnosis is a hint, and the nearest-snippet fallback still
#: covers the rest.
_EDIT_DIAG_MAX_CHARS = 4 * 1024 * 1024

_EDIT_MAX_BYTES = 50 * 1024 * 1024


class _FileChangedUnderneath(Exception):
    """The target's identity or contents moved between the read and the
    write — raised BEFORE anything is truncated."""


def _first_line_at(file_content: str, text: str) -> int:
    """1-based line where the first non-blank line of ``text`` first occurs
    in the file, else 0. The cheapest anchor a miss diagnosis can offer."""
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not first:
        return 0
    at = _occurrence_lines(file_content, first, cap=1)
    return at[0] if at else 0


def _snippet_at(file_content: str, line_no: int, context_lines: int = 3) -> str:
    """``context_lines`` around a 1-based line, rendered like
    `_nearest_snippet` (the `>>>` marks the anchor)."""
    lines = file_content.splitlines()
    if not lines:
        return "(file is empty)"
    i = max(0, min(len(lines) - 1, line_no - 1))
    start, end = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
    return "\n".join(f"{'>>>' if k == i else '   '} {k + 1:4d}: {lines[k]}"
                     for k in range(start, end))


def _ws_line(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())


def _edit_miss_diagnosis(file_content: str, old: str) -> Tuple[str, int]:
    """WHY an exact match missed, when a cheap check can tell.

    Returns ``(why, line)``: a sentence naming the concrete difference, and
    the 1-based line the evidence sits at (0 = unknown). Diagnostic only —
    none of these variants is ever APPLIED; that would be the tolerance
    ladder under another name. The point is that the model's retry is a
    copy, not a guess.

    Each verdict is checked against the file before it is uttered (fresh-eye
    review, §4KC round 2: the first version said "different indentation"
    for a diverging tail, for a missing EOF newline and for a BOM). The
    whitespace verdict compares normalised LINE LISTS as a contiguous
    slice, so a line that merely CONTAINS old_string's line is not a hit.
    """
    if not old.strip():
        return "old_string is blank.", 0
    head = file_content[:_EDIT_DIAG_MAX_CHARS]
    line = _first_line_at(head, old)
    if "\r" in old and "\r" not in head:
        if old.replace("\r\n", "\n").replace("\r", "\n") in head:
            return ("old_string carries CR characters the file does not "
                    "have; the text is otherwise present.", line)
    # A trailing newline the file's LAST line does not have. A read cannot
    # show this either; say it rather than blame indentation.
    if old.endswith("\n") and not head.endswith("\n") and head.endswith(old[:-1]):
        return ("old_string ends with a newline but the file's last line has "
                "no trailing newline; drop the final newline from old_string.",
                head.count("\n") + 1)
    # Mixed endings: some lines CRLF, some LF — the all-CRLF inverse does
    # not apply, and blaming indentation would be false.
    if "\r" in head and "\r" not in old and old in head.replace("\r\n", "\n"):
        return ("The file mixes CRLF and LF line endings and the lines you "
                "named carry a CR the read could not show; edit those lines "
                "one at a time, or use `execute` to normalise the endings "
                "first.", line)
    old_lines = [_ws_line(x) for x in old.splitlines()]
    _popped = 0
    while old_lines and not old_lines[-1]:
        old_lines.pop()                       # a trailing blank line is not a line to match
        _popped += 1
    file_lines = [_ws_line(x) for x in head.splitlines()]
    raw_file_lines = head.splitlines()
    raw_old_lines = old.splitlines()[:len(old_lines)]
    n = len(old_lines)
    # The slice compare runs BEFORE the ends-at sentence: a block present
    # earlier in the file must not be reported as running past EOF (r5).
    if old_lines and n <= len(file_lines):
        for i in range(len(file_lines) - n + 1):
            if file_lines[i:i + n] == old_lines:
                if _popped and raw_file_lines[i:i + n] == raw_old_lines:
                    _tail_old = old.splitlines()[n:n + _popped]
                    _tail_file = raw_file_lines[i + n:i + n + _popped]
                    _blank_after = sum(1 for x in _tail_file if not x.strip())
                    if _blank_after < _popped:
                        return (f"old_string ends with {_popped} blank line(s) "
                                f"but the file has only {_blank_after} after "
                                f"line {i + n}; drop the extra one(s).", i + 1)
                    if _tail_old != _tail_file[:len(_tail_old)]:
                        return (f"old_string's trailing blank line(s) carry "
                                f"whitespace the file's blank lines after line "
                                f"{i + n} do not; make them empty.", i + 1)
                    return (f"old_string ends with {_popped} blank line(s) the "
                            f"file does not have after line {i + n}; drop "
                            f"them.", i + 1)
                return ("The text IS in the file but with different "
                        "indentation or spacing (tabs vs spaces, trailing "
                        "spaces, or a wrapped line). Copy it byte-for-byte "
                        "from a read; do not retype it.", i + 1)
    if old_lines and line and line + n - 1 > len(file_lines):
        return (f"old_string has {n} lines but the file ends at line "
                f"{len(file_lines)} — it is longer than what remains.", line)
    m = re.search(re.escape(old), head, re.IGNORECASE)
    if m:
        return ("The text matches only if letter case is ignored.",
                head.count("\n", 0, m.start()) + 1)
    if line:
        _first = next(ln for ln in old.splitlines() if ln.strip())
        at = list(dict.fromkeys(_occurrence_lines(head, _first, cap=3)))
        _where = ", ".join(str(k) for k in at)
        # (the contiguous-slice compare above already failed, so for a
        # one-line old_string no line EQUALS the text — it can only be part
        # of a longer one)
        if n == 1:
            return (f"old_string's text occurs at line(s) {_where} only as "
                    f"PART of a longer line — the line continues beyond what "
                    f"you named (a trailing comment or argument, perhaps). "
                    f"Include the whole line.", line)
        return (f"The first line of old_string occurs at line(s) {_where} "
                f"but the block diverges after it — a later line was "
                f"misremembered.", line)
    if head.startswith("﻿") and old.lstrip("﻿") != old:
        return ("old_string starts with a BOM character; the file's BOM "
                "sits before line 1, include it only if you mean it.", 1)
    if head.startswith("﻿"):
        return ("(The file starts with a UTF-8 BOM; if your old_string "
                "begins at the very top of the file, the BOM precedes it.)", 1)
    return "", 0


def _edit_envelope_refusal(filename: str, which: str, text: str):
    """An envelope in an `edit` argument is always a habit from `replace`."""
    if not _ENVELOPE_MARKER_RE.search(text):
        return None
    return ToolOutcome.rejected(
        f"REJECTED: '{which}' for '{filename}' contains SEARCH/REPLACE "
        f"envelope marker lines. operation='edit' takes PLAIN TEXT: "
        f"old_string is the exact current text, new_string is its "
        f"replacement — no markers. '{filename}' is unchanged.",
        reason_code="envelope_on_edit")


def _edit_roots(sandbox_dir: Path) -> list:
    sd = Path(sandbox_dir)
    roots = [sd.resolve()]
    if _fold(sd.parent.name) == "projects":
        roots.append(sd.parent.parent.resolve())
    return roots


class _InodeDenied(Exception):
    """The opened inode is one the write guards refuse; carries the outcome."""
    def __init__(self, outcome):
        super().__init__(str(outcome))
        self.outcome = outcome


def _canonical_fd_path(fd: int) -> Optional[str]:
    """The kernel's own spelling of the path behind an open fd — on-disk
    case and Unicode form, not the caller's. macOS: F_GETPATH; Linux:
    /proc/self/fd. None when neither answers."""
    try:
        import fcntl as _fcntl
        if hasattr(_fcntl, "F_GETPATH"):
            raw = _fcntl.fcntl(fd, _fcntl.F_GETPATH, b"\0" * 1024)
            return raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
    except (OSError, ImportError, ValueError):
        pass
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return None


def _inode_write_refusal(fd: int, filename: str, sandbox_dir, project_store):
    """The write guards the dispatcher ran on the REQUESTED spelling, run
    again on the OPENED inode — the only thing an in-place write actually
    mutates (§4KC round 3).

    Three confirmed bypasses shared one root: a hardlink named `cfg.ini`
    into `.git/config`; a hardlink at the sandbox root into a RELEASED
    project's file; and `project\u017f/<id>/` — LATIN SMALL LETTER LONG S,
    which APFS folds to `projects` and `str.lower()` does not. A name-based
    guard cannot see any of them. So: a shared inode (`st_nlink > 1`) is
    refused outright for in-place writers (`write` is tmp+rename and breaks
    the link, so it needs no such rule), and the `.git` / released decisions
    are re-derived from the kernel's canonical path of the fd.
    Returns a ToolOutcome to hand back, or None.
    """
    st = os.fstat(fd)
    if getattr(st, "st_nlink", 1) > 1:
        return ToolOutcome.rejected(
            f"REJECTED: '{filename}' has {st.st_nlink} hard links — an "
            f"in-place edit would change every name that shares the inode, "
            f"including ones outside what this call was checked against. "
            f"Nothing was changed. Use operation='write' (which replaces "
            f"the directory entry) if you mean to rewrite THIS name only.",
            reason_code="hard_linked")
    canon = _canonical_fd_path(fd)
    if not canon:
        return None
    parts = Path(canon).parts
    # Components under the sandbox roots only: a `.git` ABOVE the root is
    # the host's business, not a write into repository internals (r4).
    try:
        for r in _edit_roots(Path(str(sandbox_dir))):
            if Path(canon) == r or r in Path(canon).parents:
                parts = Path(canon).relative_to(r).parts
                break
    except Exception:                                       # noqa: BLE001
        pass
    if any(str(p).lower() == ".git" for p in parts):
        return ToolOutcome.rejected(
            f"SYSTEM BLOCK: '{filename}' is a name for a file inside a "
            f"`.git` directory ({canon}) and cannot be modified. Repository "
            f"internals are never edited by hand. Nothing was changed.",
            reason_code="dotgit_write_blocked")
    try:
        rb = _released_write_block(project_store, sandbox_dir, canon)
    except Exception:                                       # noqa: BLE001
        rb = None
    return rb


def _read_regular_nofollow(real: Path, *, filename: str = "",
                           sandbox_dir=None, project_store=None):
    """Open ``real`` WITHOUT following a final-component symlink, prove it is
    a regular file on the OPEN fd, run the inode-aware write guards, read it
    whole. Returns ``(bytes, stat)`` or raises OSError / `_InodeDenied`.
    The stat is the identity the write later checks.

    O_NONBLOCK so a FIFO swapped in does not wedge the open (the check is
    on the fd, after). A size cap is enforced from the fd's own stat so a
    file grown after the path stat cannot be read past it.
    """
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = _retry_transient(os.open, str(real), flags)            # §4SD review: reads too
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise _NotRegular()
        _block_if_regular(fd, st)                               # §4SD
        if st.st_size > _EDIT_MAX_BYTES:
            raise _TooLarge(st.st_size)
        _deny = _inode_write_refusal(fd, filename, sandbox_dir, project_store)
        if _deny is not None:
            raise _InodeDenied(_deny)
        chunks, remaining = [], st.st_size
        while remaining > 0:
            chunk = _retry_transient(os.read, fd, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks), st
    finally:
        os.close(fd)


class _NotRegular(Exception):
    pass


class _TooLarge(Exception):
    pass


def _read_text_guarded(real: Path, expect: os.stat_result, *,
                       encoding: str = "utf-8", errors: str = "surrogateescape",
                       newline=None) -> str:
    """`Path.read_text` semantics (universal newlines) through an O_NOFOLLOW
    fd that must be the inode the guard cleared (§4KC r5: `replace`'s
    by-path re-reads after the guard followed a planted symlink on the read
    side and a planted FIFO wedged the loop). Raises OSError on a swap or a
    non-regular file so `replace`'s existing read-error arms report it."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = _retry_transient(os.open, str(real), flags)            # §4SD review: reads too
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise OSError(_errno.EINVAL, "not a regular file")
        if (st.st_dev, st.st_ino) != (expect.st_dev, expect.st_ino):
            raise _FileChangedUnderneath()
        _block_if_regular(fd, st)                               # §4SD
        chunks, remaining = [], st.st_size
        while remaining > 0:
            chunk = _retry_transient(os.read, fd, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    return io.TextIOWrapper(io.BytesIO(b"".join(chunks)), encoding=encoding,
                            errors=errors, newline=newline).read()


#: §4SD: errnos the macOS /workspace bind mount (virtiofs) returns for a
#: moment while the host still holds a file it just wrote — req d886d047's
#: edit of index.html failed EAGAIN and the model spent a whole turn retrying
#: it by hand; the same edit landed 24 s later. EWOULDBLOCK == EAGAIN here.
_TRANSIENT_IO_ERRNOS = frozenset({_errno.EAGAIN, _errno.EBUSY})
#: sleeps between attempts (~3.9 s in all): the window has been seconds long
#: (§4NW: a PNG re-written a few seconds after it was created)
_TRANSIENT_IO_BACKOFF_S = (0.1, 0.25, 0.5, 1.0, 2.0)


def _block_if_regular(fd: int, st: Optional[os.stat_result] = None) -> os.stat_result:
    """§4SD: clear O_NONBLOCK once ``fd`` is known to be a REGULAR file. The
    flag is there only so a FIFO swapped in cannot wedge the open; on the
    macOS virtiofs mount it also makes a regular file's read or write fail
    EAGAIN — and a BUFFERED read (fdopen/TextIOWrapper) takes that EAGAIN for
    end-of-file (review: the streaming replace committed a truncated file
    with SUCCESS). Every O_NONBLOCK open calls this after its regular-file
    check (`tests/test_4sd_transient_write_retry.py` enumerates them).
    Returns the stat."""
    st = st if st is not None else os.fstat(fd)
    if stat_mod.S_ISREG(st.st_mode):
        os.set_blocking(fd, True)
    return st


def _retry_transient(fn, *args, **kwargs):
    """Call ``fn`` and retry it while it raises EAGAIN/EBUSY, then re-raise.
    Synchronous — callers run it in a worker thread. ``fn`` must be safe to
    repeat: a single syscall that did nothing when it failed, or a
    tmp-then-rename writer that removes its tmp on the way out. An error a
    nested retry already exhausted (`_commit_replace`) is not retried again:
    six outer runs of six inner retries were ~27 s and six tmp files (review)."""
    for _pause in _TRANSIENT_IO_BACKOFF_S:
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if e.errno not in _TRANSIENT_IO_ERRNOS or getattr(e, "_ghost_retried", False):
                raise
        time.sleep(_pause)
    return fn(*args, **kwargs)


def _commit_replace(tmp, dst) -> None:
    """§4SD: ``os.replace(tmp, dst)`` retried on EAGAIN/EBUSY. A rename the
    mount reported as failed but performed is DONE — a retry would fail
    ENOENT on an applied change, and a caller that re-ran its whole write
    would append the part twice. "Performed" is proved, not assumed: the
    tmp's inode now stands at ``dst`` (review: `lexists` is False on ANY
    lstat error, so a rename that did not happen plus a failed lstat
    reported SUCCESS with the target unchanged and the tmp leaked)."""
    try:
        tmp_id = _retry_transient(os.lstat, tmp)                 # review: a transient lstat of its own tmp
    except OSError as e:
        e._ghost_retried = True              # the caller's whole-step retry must not nest it (round 4)
        raise
    tmp_id = (tmp_id.st_dev, tmp_id.st_ino)

    def _landed() -> bool:
        try:
            os.lstat(tmp)
            return False                     # still there: not renamed
        except FileNotFoundError:
            pass
        except OSError:
            return False                     # cannot tell: not proved
        try:
            st = os.lstat(dst)
        except OSError:
            return False
        return (st.st_dev, st.st_ino) == tmp_id

    def _once():
        try:
            os.replace(tmp, dst)
        except OSError as e:
            # transient, or ENOENT on a retry whose earlier attempt landed
            # while the mount still showed the tmp
            if e.errno in _TRANSIENT_IO_ERRNOS | {_errno.ENOENT} and _landed():
                return
            raise
    try:
        _retry_transient(_once)
    except OSError as e:
        e._ghost_retried = True              # the caller's retry must not re-run the write
        raise


def _write_edit_nofollow(real: Path, data: bytes, expect: os.stat_result) -> None:
    """Write ``data`` over ``real`` through an O_NOFOLLOW fd whose identity
    AND contents-stamp match the read (`st_dev`/`st_ino`/`st_size`/
    `st_mtime_ns`). Raises `_FileChangedUnderneath` BEFORE truncating when
    they do not, so a symlink swapped in after the safe-path check, or a
    concurrent writer, never receives the edit. No O_TRUNC in the open
    flags: the truncate happens only after the identity check passes.

    Deliberately NOT tmp+rename: that would succeed on a read-only file
    (only the directory needs to be writable), and a 0444 bit is a
    statement this tool should respect. A failure after the truncate can
    leave a shorter file; the caller measures that by reading back.
    """
    flags = (os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        fd = _retry_transient(os.open, str(real), flags)
    except OSError as _oe:
        # ELOOP = the final component is now a symlink: swapped in after
        # the check. Nothing was opened, nothing written — same verdict as
        # a changed identity, and the caller says so.
        if _oe.errno in (_errno.ELOOP, getattr(_errno, "EMLINK", -1)):
            raise _FileChangedUnderneath() from _oe
        raise
    try:
        st = os.fstat(fd)
        # `st_ctime_ns` is in the stamp because `st_mtime_ns` alone is
        # forgeable: a same-size rewrite + `os.utime` with the original
        # nanoseconds restored every other field (§4KC r3); ctime cannot
        # be set from userland.
        if ((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                != (expect.st_dev, expect.st_ino, expect.st_size,
                    expect.st_mtime_ns, expect.st_ctime_ns)):
            raise _FileChangedUnderneath()
        _block_if_regular(fd, st)                               # §4SD
        # each step retried on its own (§4SD): a failed ftruncate/write did
        # nothing, while re-running the whole function after the truncate
        # would fail the stamp check above against its own truncate
        _retry_transient(os.ftruncate, fd, 0)
        view = memoryview(data)
        while view:
            n = _retry_transient(os.write, fd, view)
            view = view[n:]
    finally:
        os.close(fd)


async def tool_edit_text(filename: str, old_string: Any, new_string: Any,
                         sandbox_dir: Path, *, replace_all: bool = False,
                         project_store=None):
    """Exact, unique, in-place edit — the ledger wrapper (see §4KC above).

    Mirrors `tool_replace_text`: the impl writes into ``telem`` and this is
    the single place a row is written, so no return path can go unrecorded.
    """
    telem: Dict[str, Any] = {"strategies": [], "blocks_total": 0,
                             "blocks_applied": 0, "file_len": 0}
    _search_len = len(old_string) if isinstance(old_string, str) else 0
    try:
        result = await _edit_text_impl(
            filename, old_string, new_string, sandbox_dir,
            replace_all=replace_all, telem=telem,
            project_store=project_store)
    except BaseException as exc:                            # noqa: BLE001
        record_edit(path=str(filename), op="edit", applied=False,
                    reason=f"exception:{type(exc).__name__}",
                    search_len=_search_len,
                    file_len=int(telem.get("file_len") or 0),
                    req_id=str(request_id_context.get() or ""))
        raise
    applied = _edit_applied(result, telem)
    record_edit(
        path=str(filename), op="edit", applied=applied,
        strategies=list(telem.get("strategies") or []),
        reason=("" if applied else
                str(_outcome_reason(result) or telem.get("reason") or "")),
        search_len=_search_len,
        file_len=int(telem.get("file_len") or 0),
        blocks_total=int(telem.get("blocks_total") or 0),
        blocks_applied=int(telem.get("blocks_applied") or 0),
        req_id=str(request_id_context.get() or ""),
    )
    # review 13: EVERY edit result names its own call — the main three arms carry
    # the tag inline; the other refusals (changed on disk, syntax rollback, in
    # progress, duplicate…) get it here, the one exit every path passes through
    _txt = str(result)
    if "This call's old_string" not in _txt:         # §4PF (F13): the success arms too — parallel
        from .outcome import with_text as _with_text  # replaces on one file were misbindable
        result = _with_text(result, _txt.rstrip() + f" (This call's old_string {_old_tag(old_string)}.)")
    return result


async def _edit_text_impl(filename: str, old_string: Any, new_string: Any,
                          sandbox_dir: Path, *, replace_all: bool = False,
                          telem: Optional[Dict[str, Any]] = None,
                          project_store=None):
    """The edit. Every refusal leaves the file untouched and says so.

    Argument checks come FIRST and are done here, not only in the schema:
    the JSON schema's `required` is per-tool, not per-operation, and the
    native tool-call path does not enforce it anyway.

    Every refusal is a `ToolOutcome` with a `reason_code` (a bare "Error:"
    string is booked by the ledger with an empty reason — fresh-eye §4KC
    round 2), and nothing here lets an exception escape to the model: a
    traceback carries no "unchanged" statement.
    """
    if telem is None:
        telem = {}
    pretty_log("File Edit", filename, icon=Icons.TOOL_FILE_W)
    if old_string is None or (isinstance(old_string, str) and old_string == ""):
        return ToolOutcome.rejected(
            f"SYSTEM INSTRUCTION: operation='edit' needs 'old_string' — the "
            f"exact text that is in '{filename}' NOW, copied byte-for-byte "
            f"from a read. Nothing was changed. To create a file or rewrite "
            f"one completely, use operation='write' with the FULL file in "
            f"'content' instead.", reason_code="missing_old_string")
    if new_string is None:
        return ToolOutcome.rejected(
            f"SYSTEM INSTRUCTION: operation='edit' needs 'new_string' — the "
            f"text that takes the place of old_string in '{filename}'. Pass "
            f"an empty string to delete old_string. Nothing was changed.",
            reason_code="missing_new_string")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        # A dict/list/int here is a transport or generation fault, not text
        # to write; `str()`-ing it wrote a Python repr into the file.
        return ToolOutcome.rejected(
            f"SYSTEM INSTRUCTION: operation='edit' takes old_string and "
            f"new_string as TEXT; got {type(old_string).__name__} and "
            f"{type(new_string).__name__}. Nothing was changed in "
            f"'{filename}'.", reason_code="non_string_argument")
    old, new = old_string, new_string
    if old == new:
        return ToolOutcome.rejected(
            f"REJECTED: old_string and new_string are identical "
            f"({len(old)} chars) — there is nothing to change in "
            f"'{filename}'. Nothing was changed. (To ADD text at the END of the file use "
            f"operation='write' with append=true.)",
            reason_code="identical_edit")
    if "\x00" in new:
        # `ast.parse` refuses NUL with a ValueError the syntax guard treats
        # as "cannot judge", so a NUL landed in a .py with a clean SUCCESS.
        return ToolOutcome.rejected(
            f"REJECTED: new_string contains a NUL byte, which no text file "
            f"should carry. '{filename}' is unchanged.",
            reason_code="nul_in_new_string")
    for _which, _text in (("old_string", old), ("new_string", new)):
        _env = _edit_envelope_refusal(filename, _which, _text)
        if _env is not None:
            return _env

    # ── the target ────────────────────────────────────────────────────
    try:
        path = _get_safe_path(sandbox_dir, filename)
        real = path.resolve()
        _roots = _edit_roots(sandbox_dir)
        if not any(real == r or r in real.parents for r in _roots):
            raise ValueError(f"Security Error: Path '{filename}' attempts "
                             f"to access outside sandbox.")
    except ValueError as ve:
        # A ToolOutcome, not the bare string `replace` returns: a bare
        # string with no "Error:" prefix is booked as neither a failure nor
        # a rejection by the outcome classifier.
        return ToolOutcome.rejected(str(ve), reason_code="unsafe_path")
    except (OSError, RuntimeError) as _re:
        # ENAMETOOLONG from resolve(); RuntimeError("Symlink loop").
        return ToolOutcome.rejected(
            f"Security Error: cannot resolve '{filename}' ({_re}). Nothing "
            f"was changed.", reason_code="unsafe_path")
    try:
        _exists = real.exists()
    except OSError as _se:
        # Python < 3.12 `exists()` only swallows ENOENT-class errors;
        # EACCES on a parent dir and ENAMETOOLONG propagate.
        return ToolOutcome.failed(
            f"Error: cannot stat '{filename}' ({_se}). '{filename}' is "
            f"unchanged.", world_changed=False, reason_code="stat_failed")
    if not _exists:
        fb = _scoped_root_fallback(sandbox_dir, filename)
        if fb is not None:
            _cp = _to_container_path(sandbox_dir, fb)
            return ToolOutcome.rejected(
                f"REJECTED: '{filename}' is not in the current project's "
                f"workspace, but it DOES exist at the sandbox ROOT as "
                f"'{_cp}'. Re-issue the edit with path='{_cp}' to edit that "
                f"file. Nothing was changed.", reason_code="file_not_found")
        return ToolOutcome.rejected(
            _missing_file_message(filename, sandbox_dir)
            + " operation='edit' changes an EXISTING file and never creates "
              "one — to create it, use operation='write'.",
            reason_code="file_not_found")
    # ONE path-level check BEFORE any open (FIFO wedge — see
    # `_nonregular_refusal`), then the same check again on the OPEN fd,
    # which is the one a swap cannot beat.
    _nr = _nonregular_refusal(real, filename)
    if _nr:
        return ToolOutcome.failed(_nr, world_changed=False,
                                  reason_code="not_a_regular_file")
    try:
        _orig_bytes, _st0 = await asyncio.to_thread(
            _read_regular_nofollow, real, filename=filename,
            sandbox_dir=sandbox_dir, project_store=project_store)
    except _InodeDenied as _d:
        return _d.outcome
    except _NotRegular:
        return ToolOutcome.failed(
            f"Error: '{filename}' is not a regular file (it is a FIFO, "
            f"device, socket or directory), so it cannot be read or written "
            f"as text. '{filename}' is unchanged.",
            world_changed=False, reason_code="not_a_regular_file")
    except _TooLarge as _tl:
        return ToolOutcome.failed(
            f"Error: '{filename}' is {int(_tl.args[0]) // (1024*1024)} MB; "
            f"'edit' refuses files larger than 50 MB. Use 'execute' with a "
            f"Python streaming script instead.",
            world_changed=False, reason_code="too_large")
    except OSError as oe:
        # ELOOP here = a symlink was swapped in after the safe-path check.
        return ToolOutcome.failed(
            f"Error: failed to read '{filename}' for edit ({oe}). "
            f"'{filename}' is unchanged.",
            world_changed=False, reason_code="read_failed")
    if _looks_like_binary(_orig_bytes[:8192]):
        return ToolOutcome.failed(
            f"Error: '{filename}' appears to be a binary file and cannot be "
            f"text-edited.", world_changed=False, reason_code="binary")
    # BYTES, decoded — never `read_text`: universal newlines would turn
    # every CRLF into LF on read and the write would put LF back on every
    # line (`replace` has always done this). `surrogateescape` never
    # raises — a stray byte round-trips as a lone surrogate.
    file_content = _orig_bytes.decode("utf-8", "surrogateescape")
    telem["file_len"] = len(file_content)

    # ── CRLF: the ONE invertible normalisation ────────────────────────
    # The turn loop strips every `\r` from tool results before the model
    # sees them, so the model can never produce a CRLF old_string — a
    # byte-for-byte rule would make every CRLF file un-editable through
    # the documented route (fresh-eye §4KC round 2). This is not a
    # tolerant rung: it is the exact inverse of the transport's own strip,
    # applied only when the file is CRLF, old_string carries no CR, and the
    # CRLF form is what is present. Recorded as `exact:crlf` so the ledger
    # can see it.
    _crlf = False
    if "\r\n" in file_content and "\r" not in old and "\n" in old \
            and file_content.count(old) == 0:
        _old_c = old.replace("\n", "\r\n")
        if file_content.count(_old_c) > 0:
            old, new, _crlf = _old_c, new.replace("\r\n", "\n").replace("\n", "\r\n"), True
    # §4LL: a one-line old_string in a CRLF file, with a multi-line
    # new_string, inserted bare LF lines into the CRLF file
    if not _crlf and "\r\n" in file_content and "\r" not in new and "\n" in new:
        # the line ending of the REGION being edited, not a vote over the file
        # (review: a mixed file's LF region got CRLF lines)
        _at = file_content.find(old)
        _eol = file_content.find("\n", _at) if _at >= 0 else -1
        if _eol > 0 and file_content[_eol - 1] == "\r":
            new = new.replace("\n", "\r\n")

    _n_lines = len(file_content.splitlines())
    _old_n_lines = len(old.splitlines())
    if (_n_lines >= EDIT_WHOLE_FILE_MIN_LINES
            and len(old) >= EDIT_WHOLE_FILE_SHARE * len(file_content)
            and _old_n_lines >= EDIT_WHOLE_FILE_SHARE * _n_lines):
        return ToolOutcome.rejected(
            f"REJECTED: old_string is {_old_n_lines} of the {_n_lines} lines "
            f"in '{filename}' — that is a rewrite, not an edit, and nothing "
            f"was changed. Either narrow old_string to just the lines that "
            f"change (several small 'edit' calls are fine), or, if you "
            f"really mean to replace the whole file, use operation='write' "
            f"with the FULL new file in 'content'.",
            reason_code="whole_file_on_edit")

    n = file_content.count(old)
    if n == 0:
        _why, _near_line = await asyncio.to_thread(
            _edit_miss_diagnosis, file_content, old)
        _ol = [ln for ln in old.splitlines() if ln.strip()]
        if _ol and all(re.match(r"^\s*\d+\t", ln) for ln in _ol):
            # §4LL: copied from a ranged read, line numbers and all
            _why = ("old_string starts every line with a number and a tab ('12\\t…') — if you "
                    "copied it from a ranged read, those line numbers are the read's display and "
                    "are NOT in the file; remove them. " + (_why or ""))
        _near = (_snippet_at(file_content, _near_line) if _near_line
                 else await asyncio.to_thread(_nearest_snippet, file_content, old))
        return ToolOutcome.rejected(
            f"REJECTED: old_string was not found in '{filename}' — 'edit' "
            f"matches exactly, byte-for-byte, and nothing was changed."
            + (f" {_why}" if _why else "")
            + f"\nNearest region of the file (real line numbers):\n"
            f"{_near}\n"
            f"Copy the exact current text from a read (operation='read' "
            f"with start_line/end_line) into old_string and retry. Do NOT "
            f"rewrite the file with 'write'.",
            reason_code="old_string_not_found")
    # NON-overlapping, like `str.count`/`str.replace` — the overlapping
    # walk named lines that no replacement touched.
    _lines = _occurrence_lines(file_content, old, cap=_EDIT_AMBIGUITY_LINE_CAP,
                               overlapping=False)
    # §4NZ (operator: "verify that append works"): asked to add a chapter
    # at the end, the model grows the file with `edit` — old_string = the
    # closing line, new_string = that line + the new text — 5 of 8 times in
    # a live experiment, never with append. Story/markup endings repeat
    # ("The end.", "</section>"), so the anchor was ambiguous and refused
    # (probe-ap-d). An edit that only ADDS after text that ENDS the file is
    # exactly that: applied to the occurrence at the end.
    _adds = new.startswith(old) and len(new.strip()) > len(old.strip()) and not replace_all
    if _adds and len(new) - len(old) >= 20 and file_content.rstrip().endswith(new.rstrip()):
        # review: a re-sent add-at-end edit landed again at the end
        return ToolOutcome.rejected(
            f"SYSTEM INSTRUCTION: '{filename}' already ENDS with this edit's new text — it was most "
            f"likely applied before, so nothing was changed.", reason_code="duplicate_append")
    _at_eof = not file_content[file_content.rfind(old) + len(old):].strip()
    if _adds and not _at_eof and _is_in_progress(path):
        # review: a cut add-at-end edit was SAVED as a part (the file is in
        # progress); its complete re-send would land mid-file — a duplicate
        return ToolOutcome.rejected(
            f"SYSTEM INSTRUCTION: '{filename}' is being written in PARTS in this request and old_string "
            f"is no longer its end — the text after it was most likely already saved. Read the file's "
            f"last lines and add only what is missing, with operation='write' and append=true. "
            f"Nothing was changed.", reason_code="file_in_progress")
    _eof_add = n > 1 and _adds and _at_eof
    if n > 1 and not replace_all and not _eof_add:
        _where = ", ".join(str(x) for x in _lines)
        _more = "" if n <= len(_lines) else f" — first {len(_lines)} shown"
        return ToolOutcome.rejected(
            f"REJECTED: old_string occurs {n} times in '{filename}' (this call's old_string {_old_tag(old)}) — at "
            f"line(s) {_where}{_more} — so 'edit' cannot know which one you "
            f"mean; nothing was changed. Include more surrounding lines in "
            f"old_string so it is unique, or pass replace_all=true to "
            f"change every occurrence." + _append_instead_hint(file_content, old, new),
            reason_code="old_string_ambiguous")

    _count = n if replace_all else 1
    if replace_all:
        new_content = file_content.replace(old, new)
    else:
        _i = file_content.rfind(old) if _eof_add else file_content.find(old)
        new_content = file_content[:_i] + new + file_content[_i + len(old):]
        if _eof_add:
            _lines = [file_content.count("\n", 0, _i) + 1]
    telem["blocks_total"] = _count
    # Set BEFORE the write, like the ladder does: a rung that matched and
    # then failed to land must read "matched exact, did NOT apply".
    telem["strategies"] = ["exact:crlf" if _crlf else "exact"] * _count
    # Slot names carry `lines`/`count` on purpose and the PATH comes LAST:
    # the verifier's producer/parser parity check renders this f-string
    # and classifies each slot as a number or a path by its source text,
    # and `_fs_path_ledger` (core/agent.py) parses the rendered line with
    # an end-anchored pattern — a filename containing `' — replaced` could
    # otherwise plant a phantom path in the ledger.
    _at_lines = ", ".join(str(x) for x in _lines[:_count])
    _more_lines = "" if _count <= len(_lines) else f" — first {len(_lines)} shown"
    success = (f"SUCCESS: edited — replaced {_count} "
               f"occurrence{'s' if _count != 1 else ''} of old_string "
               f"(line{'s' if _count != 1 else ''} {_at_lines}{_more_lines}) "
               f"in '{filename}'. (This call's old_string {_old_tag(old)}.)")

    async def _writer(payload: bytes) -> None:
        await asyncio.to_thread(_write_edit_nofollow, real, payload, _st0)

    try:
        # `_FileChangedUnderneath` is handled INSIDE the guard (one site for
        # both operations) and comes back as a REJECTED outcome.
        result = await _write_replace_guarded(
            real, file_content, new_content, filename, success,
            marker_guard=False, writer=_writer)
    except OSError as _we:
        # The open had no O_TRUNC, so a failure at the OPEN leaves the file
        # intact (EACCES on a 0444 file); a failure AFTER the truncate can
        # leave a shorter — or same-length but different — file. MEASURE:
        # read back and compare BYTES, not sizes (a same-size partial write
        # of the new content read as "unchanged" under a size check).
        try:
            _now = await asyncio.to_thread(read_bytes_nofollow, real)
        except OSError:
            _now = None
        _intact = _now == _orig_bytes
        return ToolOutcome.failed(
            f"Error: the write to '{filename}' failed after the edit was "
            f"computed ({_we}); "
            + (f"'{filename}' is unchanged on disk."
               if _intact else
               f"'{filename}' may be TRUNCATED or PARTIALLY written "
               f"({len(_now) if _now is not None else 'unknown'} bytes now, "
               f"{len(_orig_bytes)} before) — re-read it before retrying."),
            world_changed=not _intact, reason_code="edit_write_failed")
    if not getattr(result, "is_rejection", False):
        telem["blocks_applied"] = _count
        if _eof_add:
            from .outcome import with_text as _with_text
            result = _with_text(result, str(result) + (
                f"\nNOTE: old_string also occurs earlier in the file; this edit only ADDS after the "
                f"file's last lines, so it was applied at the END. Next time add at the end with "
                f"operation='write', append=true and content = just the new text."))
    return result


async def _replace_text_impl(filename: str, old_text: str, new_text: str,
                             sandbox_dir: Path, *,
                             telem: Optional[Dict[str, Any]] = None,
                             project_store=None):
    """The actual edit. See `tool_replace_text` for why `telem` exists."""
    if telem is None:
        telem = {}
    pretty_log("File Replace", filename, icon=Icons.TOOL_FILE_W)
    if not old_text:
        # Name the legal alternative explicitly. Observed live 2026-07-08
        # (chess session): two content-less replace calls put the model in a
        # retry spiral because the error didn't tell it HOW to move forward —
        # either the upstream native-args transport dropped `content` (the
        # known corruption family below) or the model omitted it; either way
        # the escape is the same.
        return ("Error: You must specify the exact 'content' to be replaced. "
                "If you meant to rewrite or extend the whole file instead, "
                "call file_system with operation='write' and the FULL new "
                "file in 'content' — do NOT retry 'replace' without the "
                "exact existing text.")

    # Marker-run detection is width-tolerant (`<{4,}`): the model sometimes
    # emits the 7-char git-conflict dialect (`<<<<<<< SEARCH` / `=======` /
    # `>>>>>>> REPLACE`, observed live 2026-07-19) and a plain substring test
    # for "<<<< SEARCH" misses the no-space variant of either dialect.
    has_aider_blocks = re.search(r"<{4,}[ \t]*SEARCH", str(old_text)) is not None

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
            "Replace Guard Tripped",
            f"content == replace_with ({len(str(old_text))} chars) on "
            f"'{filename}' — suspected upstream tool-call argument "
            f"corruption; rejected before any file change",
            level="WARNING",
            icon=Icons.WARN,
        )
        return (
            ToolOutcome.rejected("SYSTEM INSTRUCTION: REPLACE REJECTED — your 'content' and "
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
            "corruption.")
        )

    if not has_aider_blocks and new_text is None:
        # C5: the auto-promote calls `write_text` directly, and the gate it
        # was given used `is_file()` — FALSE for a FIFO — so it was skipped
        # and the write blocked forever. The regular-file check further down
        # sits BELOW this branch and never ran for it.
        try:
            _nr_ap = _nonregular_refusal(
                _get_safe_path(sandbox_dir, filename), filename)
            if _nr_ap:
                return _nr_ap
        except ValueError:
            pass


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
            # ⚠ BOUND BEFORE THE TRY (§4GJ). The handler below reads
            # `_wrote` to decide whether the file was touched, but the two
            # steps that precede the assignment can raise: `_get_safe_path`
            # on a path escape, and `extract_code_from_markdown` on
            # malformed input. Either one turned the handler itself into a
            # `NameError` — the caller got an unhandled exception instead of
            # the SYSTEM INSTRUCTION the branch exists to return.
            _wrote = False
            try:
                path = _get_safe_path(sandbox_dir, filename)
                # Only strip markdown fences if they're actually present;
                # otherwise preserve `old_text` byte-for-byte (trailing
                # newlines, indentation, everything). The sanitizer's
                # unconditional `.strip()` would drop the trailing newline
                # which matters for clean file writes.
                if "```" in str(old_text):
                    from ..utils.sanitizer import extract_code_from_markdown
                    # filename= arms the extractor's "whole input already
                    # parses in the target language → keep it whole" guard
                    # (same as tool_write_file) — without it a complete
                    # module whose docstring embeds a ```python example was
                    # overwritten by just the inner snippet, SUCCESS.
                    clean_content = extract_code_from_markdown(
                        str(old_text), filename=filename)
                else:
                    clean_content = str(old_text)
                # ⚠ armed BEFORE the call, not after. `Path.write_text`
                # opens with 'w': it truncates first and writes second, so an
                # OSError raised by the write itself (ENOSPC, EIO, EDQUOT,
                # EFBIG) leaves a truncated file behind. Setting the flag
                # afterwards meant that case took the "nothing was touched"
                # arm — a DECLARED rejection over a half-written file, and
                # the `failed(world_changed=True)` arm written for it was
                # unreachable.
                _unread = unread_rewrite_refusal(path, filename)                 # §4OE-2: same gate as `write`
                if _unread:
                    return ToolOutcome.rejected("REJECTED: " + _unread, reason_code="unread_rewrite")
                _wrote = True
                # O_NOFOLLOW on the RESOLVED path: the promote is a whole-file
                # overwrite, and this was the one write in the module that
                # still followed a link planted after the check (§4KC r3).
                await asyncio.to_thread(write_text_nofollow, path.resolve(),
                                        clean_content, encoding="utf-8")
                _mark_written(path)
                pretty_log(
                    "Replace Auto-Promoted",
                    f"replace→write on '{filename}' (content looked like a complete module)",
                    level="WARNING",
                    icon=Icons.WARN,
                )
                _syn = await _syntax_feedback(path, filename)
                telem["op"] = "replace_promoted_to_write"
                telem["strategies"] = ["write_promote"]
                _text = (
                    f"SUCCESS: auto-promoted operation='replace' to 'write' for "
                    f"'{filename}' because your 'content' was a complete Python "
                    f"module and 'replace_with' was missing. The file has been "
                    f"overwritten. Next time, use operation='write' directly "
                    f"when you intend to rewrite the whole file."
                    + _syn
                )
                if _syn:
                    return ToolOutcome.partial(
                        _text, world_changed=True,
                        reason_code="written_but_syntax_failed")
                return _text
            except Exception as e:
                # ⚠ NOT a rejection once the write has been attempted.
                # `Path.write_text` opens with 'w': it TRUNCATES first and
                # writes second, so an OSError here (ENOSPC, EIO, EDQUOT,
                # EFBIG) leaves a truncated or half-written file. Reproduced
                # under RLIMIT_FSIZE: 5,434 bytes in, 8,192 bytes of a
                # different file out, original gone — and reported as
                # "changed nothing", which would have let the pre-flight
                # guard record it and the loop-breaker believe the file was
                # untouched. This is the one site `world_changed` exists
                # for. A failure BEFORE the write (markdown extraction) is
                # still a true refusal.
                _msg = (f"SYSTEM INSTRUCTION: Attempted to auto-promote "
                        f"replace→write but the write failed: {e}. Retry "
                        f"with operation='write'.")
                if _wrote:
                    _msg = (f"SYSTEM INSTRUCTION: auto-promoted "
                            f"replace→write on '{filename}' and the write "
                            f"LANDED, but a later step failed: {e}. The file "
                            f"has been overwritten — re-read it before "
                            f"acting on it.")
                return ToolOutcome.failed(
                    _msg, world_changed=True,
                    reason_code="auto_promote_write_failed") if _wrote else \
                    ToolOutcome.rejected(
                        _msg, reason_code="auto_promote_failed_before_write")
        # §4HB: hold the payload so the corrective write can redeem it by
        # name instead of regenerating it.
        _held_key = _hold_rejected_content(sandbox_dir, filename, str(old_text))
        _held_note = (
            f" Your {len(str(old_text))}-char 'content' is HELD for this path "
            f"for 10 minutes: to write it as the WHOLE file, call "
            f"file_system(operation='write', path='{filename}', "
            f"content='{_HELD_CONTENT_TOKEN}') — do NOT resend the text."
            if _held_key else "")
        return ToolOutcome.rejected("SYSTEM INSTRUCTION: You used operation='replace' but forgot to specify 'replace_with'. To change part of the file use operation='edit' with old_string (the exact text now) and new_string; to rewrite the entire file use operation='write'." + _held_note, reason_code="missing_replace_with")
        
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
        # filename= arms the "whole input already parses → keep it whole"
        # guard so a block that is itself valid code containing a fenced
        # docstring example is not collapsed to the inner snippet.
        if "```" in str(old_text):
            old_text = extract_code_from_markdown(str(old_text), filename=filename)
        if new_text is not None and "```" in str(new_text):
            new_text = extract_code_from_markdown(str(new_text), filename=filename)
            
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists():
            # NO silent cross-scope edit: when the file exists at the sandbox
            # ROOT instead (execute-created via an absolute path), require
            # the model to name it explicitly — an edit landing outside the
            # project dir must be intentional.
            fb = _scoped_root_fallback(sandbox_dir, filename)
            if fb is not None:
                _cp = _to_container_path(sandbox_dir, fb)
                return (f"Error: '{filename}' is not in the current project's "
                        f"workspace, but it DOES exist at the sandbox ROOT as "
                        f"'{_cp}'. Re-issue the replace with path='{_cp}' to "
                        f"edit that file.")
            return f"Error: '{filename}' not found."

        try:
            file_size = path.stat().st_size
        except OSError as se:
            return f"Error: failed to stat '{filename}': {se}"
        # ⚠ H2 (round 3): the SIZE and BINARY refusals run BEFORE the gate.
        # Hoisting the gate above them turned two actionable errors into
        # CLOSED LOOPS: a 60 MB file under strict said "you have not read it
        # — read it first", and the read then said "too large to read
        # entirely (limit 22.4 KB)"; a binary file said the same and the read
        # said "appears to be a binary file". Refuse → "read it" → "cannot
        # read it" → refuse, with no exit. A refusal must name the real
        # obstacle, and these two do not need a receipt to be decided.
        # ⚠ REGULAR FILES ONLY, and BEFORE anything opens the target.
        # The hashing helper got this guard first; the binary sniff two lines
        # below did NOT, and `_read_head` on a FIFO blocks forever exactly
        # like `open()` does — so the H2 fix re-introduced the very wedge H1
        # had just closed, one call earlier. One check, before any open.
        try:
            _st = os.stat(path)
        except OSError as _e:
            return f"Error: failed to stat '{filename}': {_e}"
        if not stat_mod.S_ISREG(_st.st_mode):
            return (f"Error: '{filename}' is not a regular file "
                    f"(it is a FIFO, device, socket or directory), so it "
                    f"cannot be text-replaced. '{filename}' is unchanged.")
        _pre_size = _st.st_size
        # §4KC round 3: the same TOCTOU closure `edit` got in round 2. The
        # stat above is the target's identity+stamp at read time; every
        # guarded write below goes through an O_NOFOLLOW fd that must match
        # it BEFORE truncating, so a symlink swapped in after the safe-path
        # check (a reviewer rewrote a file outside the sandbox through this
        # write site) or a concurrent writer is refused, nothing written.
        _real = path.resolve()
        # Inode-aware guards (§4KC r3): hardlinks and APFS folds the name
        # checks cannot see. A short-lived O_NOFOLLOW fd; the streaming
        # branch below re-opens its own.
        def _inode_check():
            _fd = _retry_transient(os.open, str(_real), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                   | getattr(os, "O_NONBLOCK", 0))
            try:
                _ist = _block_if_regular(_fd)                   # §4SD
                return (_inode_write_refusal(_fd, filename, sandbox_dir, project_store),
                        _ist, _retry_transient(os.read, _fd, 8192))
            finally:
                os.close(_fd)
        _deny, _st_checked, _head = await asyncio.to_thread(_inode_check)
        if _deny is not None:
            return _deny
        # §4KC r4: the identity the WRITE is bound to is the inode the GUARD
        # cleared — not the earlier `os.stat`. With the two unbound, a name
        # swapped away (guard sees a clean inode) and back (write lands on
        # the original) passed the guard on one inode and wrote another.
        # The pre-read stat and the guarded inode must also agree, or the
        # name already moved between them.
        if (_st_checked.st_dev, _st_checked.st_ino) != (_st.st_dev, _st.st_ino):
            return ToolOutcome.rejected(
                f"REJECTED: '{filename}' changed on disk while it was being "
                f"checked, so the replace was NOT applied — nothing was "
                f"written. Re-read the file and retry.",
                reason_code="file_changed_underneath")
        _st = _st_checked

        async def _writer(payload: bytes) -> None:
            await asyncio.to_thread(_write_edit_nofollow, _real, payload, _st)

        if _pre_size > 50 * 1024 * 1024:
            return (f"Error: '{filename}' is {_pre_size // (1024*1024)} MB; the "
                    f"'replace' operation refuses files larger than 50 MB. Use "
                    f"'execute' with a Python streaming script instead.")
        try:
            # the head the GUARD read, not a by-path re-open (§4KC r5)
            if _looks_like_binary(_head):
                return (f"Error: '{filename}' appears to be a binary file and "
                        f"cannot be text-replaced.")
        except OSError:
            pass


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
                    # Stage the result in a sibling tmp file WITHOUT
                    # committing — the guard chain below decides whether it
                    # ever replaces the original. Also collects the changed
                    # lines (before/after) so the marker-leak check can run
                    # without materialising the whole file.
                    replaced = 0
                    changed_old = []
                    changed_new = []
                    tmp_path = None
                    try:
                        # surrogateescape on BOTH read and write so untouched
                        # bad bytes round-trip to their exact originals (matches
                        # the non-streaming path); errors="replace" used to
                        # corrupt them to U+FFFD on write-back.
                        # O_NOFOLLOW + a regular-file check on the fd: this
                        # was the last read in the module that followed a
                        # link planted after the check (§4KC r3).
                        # O_RDWR: the KERNEL decides writability (uid, group,
                        # ACLs), exactly as the other two writers' O_WRONLY —
                        # an owner-bit test was the wrong proxy (r6)
                        _sfd = os.open(str(_real), os.O_RDWR
                                       | getattr(os, "O_NOFOLLOW", 0)
                                       | getattr(os, "O_NONBLOCK", 0))
                        try:                                     # §4SD review: no leak per retry
                            _sst = os.fstat(_sfd)
                        except BaseException:
                            os.close(_sfd)
                            raise
                        if not stat_mod.S_ISREG(_sst.st_mode):
                            os.close(_sfd)
                            raise OSError(_errno.EINVAL, "not a regular file")
                        if (_sst.st_dev, _sst.st_ino) != (_st.st_dev, _st.st_ino):
                            os.close(_sfd)
                            raise _FileChangedUnderneath()
                        try:
                            _block_if_regular(_sfd, _sst)       # §4SD: a buffered EAGAIN read = EOF
                        except BaseException:
                            os.close(_sfd)
                            raise
                        with open(_sfd, 'r', encoding='utf-8', errors='surrogateescape',
                                  newline='') as f_in:                      # endings KEPT (review B)
                            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                                            suffix='.tmp', delete=False,
                                                            encoding='utf-8', errors='surrogateescape',
                                                            newline='') as f_out:
                                tmp_path = Path(f_out.name)
                                for line in f_in:
                                    if old_text in line:
                                        _nt = new_text
                                        if line.endswith("\r\n") and "\r" not in _nt:
                                            _nt = _nt.replace("\n", "\r\n")   # r2: the line's own ending
                                        new_line = line.replace(old_text, _nt)
                                        changed_old.append(line)
                                        changed_new.append(new_line)
                                        line = new_line
                                        replaced += 1
                                    f_out.write(line)
                        if replaced == 0:
                            tmp_path.unlink(missing_ok=True)
                            tmp_path = None
                        return replaced, tmp_path, changed_old, changed_new
                    except BaseException:
                        if tmp_path is not None:
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                        raise
                replaced, _tmp, _chg_old, _chg_new = await asyncio.to_thread(
                    _retry_transient, _streaming_replace)                          # §4SD
                if replaced > 0 and _tmp is not None:
                    _committed = False
                    try:
                        # Same guard chain as _write_replace_guarded — this
                        # path used to bypass it entirely, so merged-args
                        # corruption (literal '<<<< SEARCH'/'====' markers in
                        # replace_with) was written verbatim with SUCCESS.
                        # The marker check runs on the changed-line delta
                        # only: unchanged lines cancel in the count-aware
                        # guard, so the delta equals the whole-file delta.
                        leak = _marker_leak("".join(_chg_old), "".join(_chg_new))
                        if leak:
                            pretty_log(
                                "Replace Rejected",
                                f"{filename}: streaming edit would write "
                                f"SEARCH/REPLACE marker line(s) into the file "
                                f"({leak}) — file left unchanged",
                                icon=Icons.WARN, level="WARNING")
                            return (
                                ToolOutcome.rejected(f"REJECTED: that replace would have written "
                                f"SEARCH/REPLACE marker line(s) ({leak}) into "
                                f"'{filename}' as literal file content — "
                                f"'{filename}' is unchanged on disk. Emit ONE "
                                f"complete envelope per edit, or use "
                                f"operation='write' if the file legitimately "
                                f"needs such a line."))
                        _g_ext = str(filename).split(".")[-1].lower()
                        if _g_ext in ("py", "json", "js", "mjs", "cjs", "html", "htm"):
                            # Syntax-checkable type: load both versions once
                            # for the regression check (bounded by the 50 MB
                            # replace cap, same as the non-streaming path).
                            _prev_c = await asyncio.to_thread(
                                _read_text_guarded, _real, _st)
                            _new_c = await asyncio.to_thread(
                                _tmp.read_text, encoding="utf-8", errors="surrogateescape")
                            regression = _syntax_regression(_prev_c, _new_c, filename)
                            if not regression:
                                regression = await _syntax_regression_js_html(
                                    _prev_c, _new_c, filename)
                            if regression:
                                pretty_log(
                                    "Replace Rejected",
                                    f"{filename}: streaming edit would break "
                                    f"syntax ({regression}) — file left unchanged",
                                    icon=Icons.WARN, level="WARNING")
                                return (
                                    ToolOutcome.rejected(f"REJECTED: that replace would introduce a "
                                    f"syntax error and was NOT applied — "
                                    f"'{filename}' is unchanged on disk: "
                                    f"{regression}. Re-read the file and emit a "
                                    f"corrected surgical edit."))
                        # keep the original's mode: a tmp file is 0600 and
                        # `os.replace` carried that onto a 0644 script (r4)
                        try:
                            os.chmod(_tmp, stat_mod.S_IMODE(_st.st_mode) & 0o777)
                        except OSError:
                            pass
                        await asyncio.to_thread(_commit_replace, _tmp, path)   # §4SD
                        _committed = True
                    finally:
                        if not _committed:
                            try:
                                _tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                    # §4KB fix (fresh-eye review): this path applied a real
                    # edit and recorded `applied: false` with no strategy,
                    # so every >1 MB edit landed in the ledger's REJECTED
                    # bucket and was invisible to the corrective-rate proxy.
                    telem["op"] = "replace_streaming"
                    telem["strategies"] = ["streaming"]
                    # CHARS, like every other path. This recorded BYTES
                    # (file_size), so one ledger column carried two units —
                    # 1,843,200 vs 1,228,800 for the same file.
                    telem["file_len"] = len(_chg_old) and sum(
                        len(x) for x in _chg_old) or file_size
                    telem["blocks_applied"] = replaced
                    return f"SUCCESS: Streaming replace applied to '{filename}' ({replaced} line(s) modified)."
                # Fall through to heuristic match below
            except _FileChangedUnderneath:
                # r4: the source fd's inode differed from the guarded one —
                # the same verdict the guarded writer gives, not a bare
                # "Error: Streaming replace failed:" with an empty message.
                return ToolOutcome.rejected(
                    f"REJECTED: '{filename}' changed on disk between the "
                    f"check and the read, so the replace was NOT applied — "
                    f"nothing was written. Re-read the file and retry.",
                    reason_code="file_changed_underneath")
            except Exception as e:
                return f"Error: Streaming replace failed: {e}"

        try:
            # errors="surrogateescape" (not "replace"): a mostly-text file with
            # a few stray non-UTF-8 bytes stays editable, AND the bad bytes
            # ROUND-TRIP losslessly — each becomes a lone surrogate on read and
            # is re-encoded to the exact original byte on write (see the
            # matching surrogateescape write in _write_replace_guarded and the
            # streaming path). Previously errors="replace" persisted every bad
            # byte as U+FFFD on write-back, corrupting regions the edit never
            # touched (was a deferred finding).
            file_content = await asyncio.to_thread(
                functools.partial(_read_text_guarded, _real, _st, newline=""))   # endings KEPT (review B)
            telem["file_len"] = len(file_content)
            if "\r\n" in file_content and "\r" not in str(old_text):
                # review B: `replace` used to rewrite every CRLF of the file as LF. The
                # transport strips CRs from what the model sees, so a CRLF file is
                # edited with LF-only old/new text — the same ONE invertible
                # normalisation `edit` applies (r2: only when the LF form is absent,
                # and the new text follows the ending of the REGION it lands in — a
                # mixed file keeps its LF region LF)
                _old_crlf = str(old_text).replace("\n", "\r\n")
                if "\n" in str(old_text) and str(old_text) not in file_content and _old_crlf in file_content:
                    old_text = _old_crlf
                _at = file_content.find(str(old_text))
                if _at >= 0 and "\r" not in str(new_text) and "\n" in str(new_text):
                    _nl = file_content.find("\n", _at)         # the FIRST break inside/after the match — `edit`'s rule
                    if _nl > 0 and file_content[_nl - 1] == "\r":
                        new_text = str(new_text).replace("\n", "\r\n")
        except _FileChangedUnderneath:
            # r6: the same verdict the guarded writer and the streaming
            # branch give — not a bare read-error string
            return ToolOutcome.rejected(
                f"REJECTED: '{filename}' changed on disk between the check "
                f"and the read, so the replace was NOT applied — nothing was "
                f"written. Re-read the file and retry.",
                reason_code="file_changed_underneath")
        except (UnicodeDecodeError, LookupError):
            return f"Error: '{filename}' appears to be a binary file and cannot be text-replaced."
        except OSError as oe:
            return f"Error: failed to read '{filename}' for replace: {oe}"
        
        if has_aider_blocks:
            # Indentation-preserving parse FIRST: capture block bodies
            # starting on the line AFTER each marker, so the first line's
            # leading whitespace survives. The loose fallback's `\s*` used
            # to swallow it, which (a) made exact matches impossible for
            # indented blocks and (b) fed _reindent_replacement a new_text
            # whose first line was dedented, mis-computing its indent base
            # — the fuzzy/anchor rungs then inserted a broken first line.
            # Both regexes accept marker runs of ANY width >= 4 (`<{4,}` /
            # `={4,}` / `>{4,}`, optional REPLACE suffix) so the 7-char
            # git-conflict dialect parses as intended, and both require the
            # separator to be a FULL LINE of equals. The old loose fallback
            # matched `====` ANYWHERE (`\s*====\s*`): a 7-char envelope fell
            # past the strict 4-char regex into it, and the first `=====`
            # inside a banner comment on the SEARCH text's first line
            # ("// ===== Minesweeper App =====") became the separator —
            # SEARCH parsed as `//`, the entire rest became the replacement,
            # and the first `//` in the file (inside the CSS
            # `@import url('https://...')`) was replaced with the whole
            # marker-stripped blob, reported as SUCCESS (live corruption
            # 2026-07-19, WebOS index.html).
            # The closer is line-anchored like the separator (a `>{4,}` run
            # mid-line — e.g. a `// >>>> end of patch` comment in the
            # replacement — used to terminate the envelope early, silently
            # truncating the replacement). The replacement capture is an
            # optional `(payload\n)?` group so a deletion envelope
            # (`====` directly followed by `>>>>`) still parses: the
            # separator's own trailing newline doubles as the closer's
            # anchor and findall yields '' for the non-participating group.
            blocks = re.findall(
                r'<{4,}[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n={4,}[ \t]*\r?\n(?:(.*?)\r?\n)?[ \t]*>{4,}[ \t]*(?:REPLACE)?',
                str(old_text), re.DOTALL)
            if not blocks:
                # Loose fallback: tolerates content on the OPENER line, at
                # the cost of stripping the blocks' leading/trailing
                # whitespace. The separator AND closer stay line-anchored —
                # that is the invariant that prevents content containing
                # `====`/`>>>>` runs from splitting the envelope.
                blocks = [
                    (s.strip(), r.strip()) for s, r in re.findall(
                        r'(?s)<{4,}[ \t]*SEARCH(.*?)\n[ \t]*={4,}[ \t]*\r?\n(?:(.*?)\n)?[ \t]*>{4,}[ \t]*(?:REPLACE)?',
                        str(old_text))
                ]
            if not blocks:
                # Last-resort rung: a closer GLUED to content (`…')>>>>`,
                # no newline before it). Only reached when neither anchored
                # parse found a single block — i.e. the envelope has NO
                # line-anchored closer at all — so a mid-payload `>{4,}`
                # run can never truncate a well-formed envelope (it used
                # to: `// >>>> end of patch` in the replacement ended the
                # match early and the truncated result was written with
                # SUCCESS). The per-block marker/tiny-search guards below
                # still screen whatever this rung yields.
                blocks = [
                    (s.strip(), r.strip()) for s, r in re.findall(
                        r'(?s)<{4,}[ \t]*SEARCH(.*?)\n[ \t]*={4,}[ \t]*\r?\n(.*?)>{4,}[ \t]*(?:REPLACE)?',
                        str(old_text))
                ]

            if not blocks:
                return ToolOutcome.rejected("SYSTEM INSTRUCTION: Found SEARCH/REPLACE markers but failed to parse them. Ensure you use <<<< SEARCH, ====, and >>>> correctly.", reason_code="unparsable_blocks")
            
            telem["op"] = "replace_block"
            telem["blocks_total"] = len(blocks)
            # STRUCTURAL. The all-ambiguous header used to be selected by
            # `e.startswith("Block REJECTED as AMBIGUOUS")` — a lexical match
            # on this module's own rejection prose, so rewording the message
            # silently fell back to the generic "None ... matched" header,
            # whose prescribed repair (widen the search) is the OPPOSITE of
            # the right one. Exactly the class §4KB claims to have avoided.
            ambiguous_count = 0
            success_count = 0
            errors = []
            strategies = []
            prev_content = file_content

            for search_str, replace_str in blocks:
                # MIS-SPLIT SANITY (2026-07-19). A parse that yields a
                # SEARCH text this small can only come from a mangled
                # envelope (the live corruption parsed SEARCH as `//`) or
                # an edit far too ambiguous to apply safely — a 1-2 char
                # needle matches essentially anywhere in the file.
                if len(search_str.strip()) < 3:
                    errors.append(
                        "Block REJECTED: its SEARCH text is "
                        f"{len(search_str.strip())} char(s) — too short to "
                        "locate safely (this usually means the envelope "
                        "markers were malformed and the parse split in the "
                        "wrong place). Re-emit the envelope with at least "
                        "one full line of the file's CURRENT text in the "
                        "SEARCH section.")
                    continue
                # A marker line INSIDE the parsed SEARCH section is the same
                # mangled-envelope signature on the other side of the
                # separator — never a legitimate edit.
                if any(_MARKER_LINE_RE.match(ln.rstrip("\r"))
                       for ln in search_str.splitlines()):
                    errors.append(
                        "Block REJECTED: its SEARCH section contains a "
                        "SEARCH/REPLACE marker line — the envelope was "
                        "malformed and the parse split in the wrong place. "
                        "Emit ONE clean envelope per edit.")
                    continue
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
                # MULTI-EDIT ENVELOPE GUARD (2026-07-14). Only the FIRST
                # `====` in an envelope is the separator; when the model
                # packs a second edit into the same envelope, that edit's
                # `====` lines and text all land in `replace_str` and would
                # be written to the file LITERALLY (observed live: index.html
                # shipped with `====` at lines 78/80/85 plus duplicated code,
                # and the tool said SUCCESS — the user's next message was the
                # browser's `Unexpected token '==='`). This syntax cannot
                # express that shape, so reject the block and teach
                # one-envelope-per-edit instead of failing open.
                if any(_MARKER_LINE_RE.match(ln.rstrip("\r"))
                       for ln in replace_str.splitlines()):
                    errors.append(
                        "Block REJECTED: its replacement section contains a "
                        "bare '====' (or '<<<< SEARCH'/'>>>>') marker line — "
                        "you packed MORE THAN ONE edit into a single "
                        "SEARCH/REPLACE envelope, and everything after the "
                        "first '====' would have been written into the file "
                        "literally. Emit ONE complete envelope per edit "
                        "(each with its own '<<<< SEARCH', '====' and "
                        "'>>>>'); multiple envelopes in one call are "
                        "supported.")
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
                if matched_text is None:
                    # AMBIGUOUS (§4KB). Reject the envelope, keep the rest of
                    # the batch going — a partial apply plus a named reason
                    # beats an all-or-nothing failure the model cannot act on.
                    _n = 0
                    try:
                        _n = int(str(strategy).split(":", 1)[1])
                    except Exception:                       # noqa: BLE001
                        _n = file_content.count(search_str)
                    telem["reason"] = "ambiguous_block"
                    ambiguous_count += 1
                    pretty_log(
                        "Replace Rejected",
                        f"{filename}: SEARCH block matched {_n}× (lines "
                        f"{', '.join(str(x) for x in _occurrence_lines(file_content, search_str))}) "
                        f"— ambiguous, this block NOT applied",
                        icon=Icons.WARN, level="WARNING")
                    errors.append(
                        _ambiguous_block_error(file_content, search_str, _n))
                    continue
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
                telem["strategies"] = list(strategies)
                telem["blocks_applied"] = success_count

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
                _res = await _write_replace_guarded(
                    path, prev_content, file_content, filename, msg,
                    writer=_writer)
                if errors and not getattr(_res, "is_rejection", False):
                    # Some blocks landed and some did not: PARTIAL.
                    # ⚠ Only when the write actually LANDED. The guard can
                    # roll the whole edit back (marker leak, syntax
                    # regression) and return REJECTED — relabelling that
                    # PARTIAL with `world_changed=True` claimed a mutation
                    # over a file that never changed, which fires
                    # `strikes.note_world_changed()` and wipes the
                    # loop-breaker's memory. Keep the guard's own verdict.
                    return ToolOutcome.partial(
                        str(_res),
                        world_changed=getattr(_res, "changed_the_world", True),
                        reason_code="some_replace_blocks_failed")
                return _res
            else:
                # "matched" would be a lie when the block matched N>1 times —
                # the model's repair for "not found" (widen the search) is the
                # OPPOSITE of the repair for "found twice" (add unique
                # context), so the header must name which one happened.
                if ambiguous_count and ambiguous_count == len(errors):
                    telem["reason"] = "ambiguous_block"
                    return ToolOutcome.rejected(
                        f"SYSTEM INSTRUCTION: every SEARCH/REPLACE block in "
                        f"'{filename}' matched MORE THAN ONCE — nothing was "
                        f"changed.\n" + "\n".join(errors),
                        reason_code="ambiguous_block")
                return ToolOutcome.rejected(f"SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in '{filename}'.\n" + "\n".join(errors), reason_code="no_blocks_matched")

        # 1. Exact match attempt
        if old_text in file_content:
            occurrences = file_content.count(old_text)
            new_file_content = file_content.replace(old_text, new_text)
            telem["strategies"] = ["exact"]
            msg = f"SUCCESS: Exact match found and replaced in '{filename}'."
            if occurrences > 1: msg += f" WARNING: Replaced {occurrences} identical occurrences."
            return await _write_replace_guarded(path, file_content, new_file_content, filename, msg, writer=_writer)

        # 2. Heuristic match (ignore arbitrary whitespace & newlines). The
        # match starts at the first token, so re-anchor the replacement's
        # indentation to the matched region before substituting (otherwise a
        # multi-line block's continuation lines land at the wrong column).
        words = [re.escape(w) for w in str(old_text).split()]
        flexible_old = r'\s+'.join(words)

        matches = re.findall(flexible_old, file_content)
        if len(matches) == 1:
            if "\r\n" in file_content and "\r" not in str(new_text) and "\n" in str(new_text):
                _fat = file_content.find(matches[0])          # review 14: the region's ending, as the exact rung
                _fnl = file_content.find("\n", _fat) if _fat >= 0 else -1
                if _fnl > 0 and file_content[_fnl - 1] == "\r":
                    new_text = str(new_text).replace("\n", "\r\n")
            reindented = _reindent_replacement(file_content, matches[0], str(new_text))
            # Replace only the FIRST occurrence (count=1), matching the
            # exact/fuzzy/anchor/aider paths — an unbounded replace would
            # clobber every later copy of an identical block.
            telem["strategies"] = ["flexible"]
            new_file_content = file_content.replace(matches[0], reindented, 1)
            return await _write_replace_guarded(
                path, file_content, new_file_content, filename,
                f"SUCCESS: Flexible match found and replaced in '{filename}'.",
                writer=_writer)
        elif len(matches) > 1:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: Multiple instances of this text block found. Please provide a larger, more unique block of code in 'content' to ensure we replace the correct one.", reason_code="ambiguous_block")

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
            telem["strategies"] = [f"fuzzy:{ratio:.0%}"]
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
                f"near-identical block was unambiguous.",
                writer=_writer)
            if res.startswith("SUCCESS"):
                res = append_note(res, f" VERIFY the change is what you "
                                  f"intended:\n--- REPLACED BLOCK (was) ---\n"
                                  f"{matched_text[:600]}")
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
            telem["strategies"] = [
                f"anchor:{info['start_line']}-{info['end_line']}"]
            replacement = str(new_text)
            if matched_text.endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            new_file_content = file_content.replace(matched_text, replacement, 1)
            res = await _write_replace_guarded(
                path, file_content, new_file_content, filename,
                f"SUCCESS: Anchor match — replaced the block spanning lines "
                f"{info['start_line']}–{info['end_line']} in '{filename}' "
                f"(matched on its unique {'first+last lines' if info['strategy']=='first_last' else 'signature + balanced braces'}; "
                f"the middle differed from your old_text).",
                writer=_writer)
            if res.startswith("SUCCESS"):
                res = append_note(res, f" VERIFY the change:\n"
                                  f"--- REPLACED BLOCK (was) ---\n"
                                  f"{matched_text[:600]}")
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
            ToolOutcome.rejected("SYSTEM INSTRUCTION: The search block was NOT found in '"
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
            "CLOSEST MATCH IN THE FILE (with line numbers):\n" + snippet, reason_code="search_block_not_found")
        )
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

#: Occurrence line numbers are capped in the rejection message: the model
#: needs enough to disambiguate, not a listing. Five is two more than the
#: duplicate count that actually occurs in practice.
_AMBIGUITY_LINE_CAP = 5


def _occurrence_lines(file_content: str, needle: str,
                      cap: int = _AMBIGUITY_LINE_CAP, *,
                      overlapping: bool = True):
    """1-based line numbers where ``needle`` STARTS, up to ``cap``.

    Used to build the ambiguity rejection. A model told "this matched
    3 times" still cannot act; told "lines 12, 48, 97" it can widen its
    SEARCH block with the right neighbouring context on the first retry.

    ``overlapping=False`` steps past each hit like `str.count`/`str.replace`
    do, so the lines named agree with the count and with what a
    replace_all actually touches (`x\nx\nx\nx` / `x\nx\n`: 2 hits at
    lines 1 and 3, not 3 hits). The default keeps `replace`'s wording.
    """
    out = []
    if not needle:
        return out
    step = 1 if overlapping else len(needle)
    idx = file_content.find(needle)
    while idx != -1 and len(out) < cap:
        out.append(file_content.count("\n", 0, idx) + 1)
        idx = file_content.find(needle, idx + step)
    return out


def _ambiguous_block_error(file_content: str, search_str: str,
                           occurrences: int) -> str:
    """The ONE message every rung emits for an ambiguous match (§4KB).

    Before this existed the ladder told three different stories about the
    same input: the whitespace-flexible rung REJECTED, the two-argument
    exact path replaced EVERY occurrence with a loud warning, and the
    block form's exact rung silently edited the FIRST one and reported
    ``exact`` — a wrong-region edit with a success message, which is the
    one outcome no downstream guard can detect (a first-occurrence edit is
    syntactically valid, leaks no markers, and regresses no parse).
    """
    lines = _occurrence_lines(file_content, search_str)
    where = ", ".join(str(n) for n in lines)
    more = "" if occurrences <= len(lines) else f" (first {len(lines)} shown)"
    head = str(search_str).strip().splitlines()[0][:60] if str(search_str).strip() else ""
    return (
        f"Block REJECTED as AMBIGUOUS: its SEARCH text occurs "
        f"{occurrences} times in the file — at line(s) {where}{more}"
        + (f" — starting `{head}`" if head else "")
        + ". THIS BLOCK was not applied. (Other blocks in the same call may "
        "have been — check the summary line.) Re-emit this envelope with "
        "MORE surrounding context (the line above and below the region you "
        "mean) so the SEARCH text appears exactly once."
    )


def _locate_block(file_content: str, search_str: str):
    """Full matching ladder shared by the SEARCH/REPLACE-block loop:
    exact → whitespace-flexible (unique) → fuzzy window → anchor block.

    Returns ``(matched_text, strategy)`` where ``matched_text`` is the
    ORIGINAL bytes in the file to replace and ``strategy`` is one of
    ``"exact"``, ``"flexible"``, ``"fuzzy:NN%"``, ``"anchor:L1-L2"`` — or
    ``None`` when no matcher produces a unique, high-confidence hit.

    ⚠ THIRD OUTCOME (§4KB): ``(None, "ambiguous:N")`` when the search text
    is found but N>1 times. Every caller must branch on a ``None``
    ``matched_text`` BEFORE using it — the previous contract had only two
    outcomes, so the exact rung resolved a 2-occurrence search to the first
    hit and called it ``exact``. Uniqueness is a property of the ladder, not
    of the caller: `flexible` already required ``len(matches) == 1``,
    `_fuzzy_block_match` requires a clear margin over the runner-up, and
    `_anchor_block_match` requires both anchors to be unique. The exact rung
    was the only one that did not check, and it is the rung that matches
    most often."""
    if search_str in file_content:
        n = file_content.count(search_str)
        if n > 1:
            return None, f"ambiguous:{n}"
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


def _find_node() -> "str | None":
    """Resolve the node binary even under launchd's minimal PATH.

    The agent runs as a LaunchDaemon whose PATH is /usr/bin:/bin:/usr/sbin:
    /sbin — no /opt/homebrew/bin — so a bare ``shutil.which("node")``
    returned None in production and every JS/HTML syntax check silently
    skipped. That's how a replace that pasted literal ``====`` marker lines
    into index.html still returned SUCCESS with no syntax warning
    (2026-07-14). Fall back to the common install locations before giving
    up; a missing node still fails open (checks are best-effort).
    """
    import shutil
    node = shutil.which("node")
    if node:
        return node
    for cand in ("/opt/homebrew/bin/node", "/usr/local/bin/node",
                 "/opt/local/bin/node"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


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
    import tempfile
    node = _find_node()
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


# Definition shapes the orphan guard recognizes: Python def/class, JS
# function declarations/assignments, and object-literal/class method
# shorthand. Keyword heads are excluded so `if (...) {` never reads as a
# method definition.
_SYMBOL_DEF_RES = (
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M),
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b", re.M),
    re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", re.M),
    re.compile(r"\b([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)", re.M),
    re.compile(r"^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", re.M),
)
_SYMBOL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "function", "return", "with",
    "else", "do", "try", "new", "await", "typeof", "constructor", "super",
})


def _defined_symbols(text: str) -> set:
    out = set()
    for rx in _SYMBOL_DEF_RES:
        for m in rx.finditer(text or ""):
            name = m.group(1)
            if name and name not in _SYMBOL_KEYWORDS:
                out.add(name)
    return out


def _orphaned_symbol_warning(old_text: str, new_text: str,
                             final_content: str) -> str:
    """Warn when a replace REMOVES a function/def whose references survive.

    The 2026-07-24 Router.init() regression: a refactor replaced ``init()``
    with ``bootstrap()`` but the nav-listener registration kept calling the
    now-orphaned name — all top navigation broke, undetected, and the NEXT
    session diagnosed it as its own 24-minute-old change. Deterministic and
    cheap: symbols defined in the removed block but not in the replacement,
    that still have references in the final file, get named in the tool
    result so the model fixes the callers in the SAME turn. Advisory only —
    never blocks the edit. Never raises."""
    try:
        removed = _defined_symbols(old_text) - _defined_symbols(new_text)
        if not removed:
            return ""
        warnings = []
        for sym in sorted(removed):
            refs = len(re.findall(rf"\b{re.escape(sym)}\b", final_content or ""))
            if refs > 0:
                warnings.append(f"`{sym}` ({refs} remaining reference(s))")
        if not warnings:
            return ""
        return (
            "\nWARNING: this replace REMOVED the definition of "
            + ", ".join(warnings)
            + " but the file still references it/them. If you renamed or "
              "replaced the function, update those call sites NOW (grep the "
              "file for each name) — orphaned wiring is how the last nav "
              "regression shipped."
        )
    except Exception:
        return ""


def _old_head(old: Any, n: int = 60) -> str:
    """«the first line of old_string…» — probe-big-edit (2026-10-12): three
    parallel edits on one file came back as two REJECTED + one SUCCESS with
    nothing to tell which was which; the model bound them wrong and reported
    a price change that never landed. Every edit result names its own call."""
    first = next((ln.strip() for ln in str(old or "").splitlines() if ln.strip()), "")
    first = first.replace("«", "<").replace("»", ">")      # the ledger regex ends on «…» — keep it unforgeable
    if len(first) > n:
        first = first[:n].rstrip() + "…"
    return f"«{first}»" if first else "«(empty)»"


def _old_tag(old: Any) -> str:
    """"began «…»", plus ", ended «…»" when the text is longer than the head
    shows — two catalogue rows share their first 60 chars (review 13)."""
    text = str(old or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tag = f"began {_old_head(old)}"
    if len(lines) > 1 or (lines and len(lines[0]) > 60):
        tail = lines[-1].replace("«", "<").replace("»", ">")
        if len(tail) > 30:
            tail = "…" + tail[-30:].lstrip()
        tag += f", ended «{tail}»"
    return tag


#: §4OI (iii): inline event handlers in a written page that call a function the
#: page never defines as a global — the census webOS calculator: `onclick=
#: "calcInput('7')"` while the script bound only `window['calcInput_'+id]`; every
#: button threw ReferenceError and the reply said "11 functional apps … verified".
_HANDLER_ATTR_RE = re.compile(r"""\son[a-z]+\s*=\s*(["'])\s*(?:return\s+)?([A-Za-z_$][\w$]*)\s*\(""", re.IGNORECASE)
_SCRIPT_BLOCK_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_JS_GLOBALS = frozenset("""alert confirm prompt parseInt parseFloat setTimeout setInterval clearTimeout clearInterval
    requestAnimationFrame fetch console document window location history navigator event this void eval encodeURIComponent
    decodeURIComponent Number String Boolean Array Object Math JSON Date isNaN open close print scrollTo
    if return new typeof""".split())


HANDLER_CHECK_MAX_BYTES = 600_000
HANDLER_CHECK_MAX_HANDLERS = 400
_CDN_HOST_RE = re.compile(r"(?i)https?://(?:cdn\.|cdnjs\.|unpkg\.com|cdn\.jsdelivr\.net|ajax\.googleapis\.com|"
                          r"code\.jquery\.com|[\w.-]*cloudflare[\w.-]*|esm\.sh|cdn\.skypack\.dev)")


def _script_blocks(t: str):
    """(attrs, body) of every <script>…</script>, by a linear scan (the
    regex was quadratic on 60k unclosed `<script`, r1 m2)."""
    low = t.lower()
    i = 0
    while True:
        a = low.find("<script", i)
        if a < 0:
            return
        gt = low.find(">", a)
        if gt < 0:
            return
        end = low.find("</script", gt)
        if end < 0:
            return
        yield t[a + 7:gt], t[gt + 1:end]
        i = end + 8


def _markup_without_comments(t: str) -> str:
    """The page with HTML comments and <pre>/<code>/<textarea>/<template>
    blocks blanked, script blocks kept verbatim — one left-to-right scan
    (§4OI r3 m4: the lazy-dot regexes took 15 s on 160 KB of unclosed `<!--`).
    An unclosed comment or block blanks the rest, as a browser would hide it."""
    low = t.lower()
    out, i, n = [], 0, len(t)
    tags = ("pre", "code", "textarea", "template")
    while i < n:
        lt = low.find("<", i)
        if lt < 0:
            break
        if low.startswith("<!--", lt):
            out.append(t[i:lt] + " ")
            end = low.find("-->", lt + 4)
            i = n if end < 0 else end + 3
            continue
        if low.startswith("<script", lt):
            end = low.find("</script", lt)
            stop = n if end < 0 else end + 8
            out.append(t[i:stop])
            i = stop
            continue
        name = next((g for g in tags if low.startswith("<" + g, lt)
                     and not (low[lt + 1 + len(g):lt + 2 + len(g)].isalnum())), None)
        if name:
            out.append(t[i:lt] + " ")
            end = low.find("</" + name, lt)
            if end < 0:
                i = n
            else:
                gt = low.find(">", end)
                i = n if gt < 0 else gt + 1
            continue
        out.append(t[i:lt + 1])
        i = lt + 1
    out.append(t[i:])
    return "".join(out)


def html_handler_note(text: str) -> str:
    """Advisory note (never a failure): names inline `on…="fn(…)"` handlers
    whose function no inline script of the page defines globally. Silent when
    the page loads scripts from files we cannot see (`<script src>` other than
    a CDN library), or when nothing is wrong."""
    t = str(text or "")
    if len(t) > HANDLER_CHECK_MAX_BYTES or "<script" not in t.lower() or " on" not in t.lower():
        return ""
    if re.search(r"Object\.assign\(\s*(?:window|globalThis)|self\s*\.\s*[A-Za-z_$][\w$]*\s*=", t):
        return ""                          # globals made dynamically — not judged (r1 m3)
    # tutorial code shown on the page is not markup (r1 m3: escaped `&lt;button onclick=…`); nor are
    # comments (r2 m4) — but only OUTSIDE scripts (r3 m3: `<script><!-- … //--></script>`), linearly (r3 m4)
    t = scan = _markup_without_comments(t)
    handlers = {m.group(2) for m in _HANDLER_ATTR_RE.finditer(scan)} - _JS_GLOBALS
    if len(handlers) > HANDLER_CHECK_MAX_HANDLERS:
        return ""
    if not handlers:
        return ""
    js = []
    for attrs, body in _script_blocks(t):
        src = re.search(r"""src\s*=\s*["']([^"']+)""", attrs or "", re.IGNORECASE)
        if src:
            if not _CDN_HOST_RE.match(src.group(1)):
                return ""                  # a script file we cannot see (local, or a first-party URL) — not judged
            continue
        if re.search(r"""type\s*=\s*["']?module\b""", attrs or "", re.IGNORECASE):
            # module scope is NOT global (r1 m3) — only its explicit window./globalThis. assignments count (r2 m4)
            js.append("\n".join(re.findall(r"(?:window|globalThis)\s*(?:\.\s*[\w$]+|\[\s*['\"][\w$]+['\"]\s*\])\s*=(?!=)", body)))
            continue
        js.append(body)
    code = "\n".join(js)
    missing = []
    for name in sorted(handlers):
        n = re.escape(name)
        if re.search(rf"(?:function\s*\*?\s*{n}\s*\(|(?:var|let|const)\s+{n}\s*=|(?<![\w$.]){n}\s*=(?!=)"
                     rf"|window\s*\.\s*{n}\s*=(?!=)|window\s*\[\s*['\"]{n}['\"]\s*\]\s*=(?!=)|class\s+{n}\b"
                     rf"|globalThis\s*\.\s*{n}\s*=)", code):
            continue
        missing.append(name)
    if not missing:
        return ""
    shown = ", ".join(f"`{m}`" for m in missing[:8]) + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
    return (f"\nHANDLER CHECK: the page's inline event handlers call {shown}, but no script in the page defines "
            f"{'it' if len(missing) == 1 else 'them'} as a global function — clicking those elements throws "
            f"ReferenceError. Define them at the top level of a <script> (or `window.name = …`), or attach the "
            f"listeners in code; do not call the page working until a click on them has been tested.")


#: §4SH: regions whose text is code or verbatim content BY DESIGN — blanked
#: (newlines kept, so line numbers hold) before looking for stray code. An
#: opener never closed runs to the end of the file, as it does in a browser.
#: r1: every other TAG is blanked too, quoted attribute values included — a
#: multi-line onclick is not stray code, and a `<!--` inside an attribute
#: opens no comment; a custom element (`<code-editor>`) is not `<code>`;
#: `[^<>]` keeps an opener without `>` from scanning to the end (quadratic).
_VERBATIM_REGION_RE = re.compile(
    r"<(script|style|pre|code|textarea|template|xmp)(?![\w-])[^<>]*>(?:.*?</\1\s*>|.*\Z)"
    r"|<[A-Za-z][\w:.-]*(?:\s+[^\s=<>\"'/]+(?:\s*=\s*(?:\"[^\"]{0,8000}\"|'[^']{0,8000}'|[^\s<>\"']+))?)*\s*/?>"
    r"|<!--.*?(?:-->|\Z)",
    re.IGNORECASE | re.DOTALL)
#: a JavaScript STATEMENT line — declarations and function assignments only,
#: never the prose a page shows
_STRAY_JS_LINE_RE = re.compile(
    r"^[ \t]*(?:(?:const|let|var)[ \t]+[\w$]+[ \t]*=|(?:async[ \t]+)?function[ \t]*\*?[ \t]*[\w$]+[ \t]*\("
    r"|[\w$]+(?:\.[\w$]+)+[ \t]*=[ \t]*(?:async[ \t]+)?(?:function\b|\([^)\n]*\)[ \t]*=>|[\w$]+[ \t]*=>))",
    re.MULTILINE)
STRAY_JS_MIN_LINES = 3


def html_stray_script_note(text: str) -> str:
    """§4SH (req cb27058e): JavaScript sitting OUTSIDE any <script> — the
    browser shows it as page text and never runs it. The shape a part
    boundary makes: part 3 ended with </script> and part 4 began with code,
    so the app never started and the model spent ~10 minutes finding why.
    Advisory, never a failure; "" when nothing is wrong."""
    t = str(text or "")
    if len(t) > HANDLER_CHECK_MAX_BYTES or "<" not in t:
        return ""
    bare = _VERBATIM_REGION_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), t)   # same length: offsets hold
    hits = list(_STRAY_JS_LINE_RE.finditer(bare))
    if len(hits) < STRAY_JS_MIN_LINES:
        return ""
    first = bare.count("\n", 0, hits[0].start()) + 1
    _close = [m for m in re.finditer(r"</script\s*>", t[:hits[0].start()], re.IGNORECASE)]
    after = (f" (the last </script> before it is at line {t.count(chr(10), 0, _close[-1].start()) + 1})"
             if _close else "")
    return (f"\nPAGE CHECK: from line {first}, {len(hits)} lines of JavaScript sit OUTSIDE any <script>{after} — "
            f"the browser shows them as page text and never runs them. A part most likely began after a "
            f"</script> without opening its own <script>: put <script> before line {first} and make sure "
            f"a </script> closes it.")


def html_page_notes(text: str) -> str:
    """Every whole-page HTML check (advisory): undefined inline handlers and
    code outside any <script>."""
    return html_handler_note(text) + html_stray_script_note(text)


async def _syntax_feedback(path: Path, filename: str, *, in_progress: bool = False,
                           last_rows: Optional[int] = None) -> str:
    """Best-effort post-write syntax check. Returns "" when clean or unknown.
    ``in_progress``: the file is a PART (more_parts) — not parseable yet by
    design, so the finding is logged as a deferred check, not an error.

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
        # Explicit UTF-8 on every read here: without it read_text() uses the
        # process locale (LANG unset under launchd on this box), and a non-
        # ASCII char in the file raised UnicodeDecodeError — swallowed by the
        # blanket except below, silently SKIPPING the whole syntax check.
        # Mirrors the same fix already applied to tool_write_file's write.
        if ext == "py":
            import ast
            _pytext = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="replace")
            try:
                ast.parse(_pytext)
            except SyntaxError as se:
                err = f"{se.msg} (line {se.lineno}, col {se.offset})"
                # Whole-file DOUBLE-ESCAPE shape (2026-08-01, req 50855398):
                # the content landed as ONE long line full of literal \n
                # two-char sequences (a JSON-escaped string written
                # verbatim). The generic SyntaxError ("unmatched '}' line 1
                # col 8370", "unexpected character after line continuation")
                # sent the model into line-patch spirals — three executor
                # retries and ~180s of interactive turns before it diagnosed
                # the shape itself. Name it outright.
                if _pytext.count("\n") <= 2 and _pytext.count("\\n") >= 5:
                    err += (
                        " — THE WHOLE FILE IS ONE LINE built of literal \\n "
                        "escape sequences: the content was written DOUBLE-"
                        "ESCAPED. Do NOT patch individual lines (there is "
                        "only one); REWRITE the entire file with real "
                        "newlines.")
        elif ext in ("sh", "bash"):
            # §4PH (R1-7): `bash -n` parses without running — cheap, and a
            # shell script is usually under the unverified-write size gate,
            # so this is the only check it gets at write time
            import shutil as _shutil
            _bash = _shutil.which("bash")
            # r3 (reader): the host's bash is 3.2 — a script using bash-4
            # constructs the sandbox accepts (`|&`, `;;&`, `${v,,}`) must not
            # be told it is broken
            _sh_text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
            if _bash and re.search(r"\|&|;;&|\$\{\w+(?:,,|\^\^)", _sh_text):
                _bash = None
            if _bash:
                proc = await asyncio.create_subprocess_exec(
                    _bash, "-n", str(path),
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
                    lines = (stderr_b or b"").decode("utf-8", "replace").splitlines()
                    err = "\n".join(ln[:200] for ln in lines if ln.strip())[:800]
        elif ext == "json":
            try:
                json.loads((await asyncio.to_thread(
                    path.read_text, encoding="utf-8", errors="replace")).lstrip("\ufeff"))   # §4SH r10: as the end check
            except json.JSONDecodeError as je:
                err = f"{je.msg} (line {je.lineno}, col {je.colno})"
        elif ext in ("js", "mjs", "cjs"):
            node = _find_node()
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
        elif ext in ("csv", "tsv"):
            # probe-big2 (2026-10-12): a 250-row CSV "looked" fine to the model while
            # 82 rows had an unquoted comma inside a field — nothing checked the shape
            err = await asyncio.to_thread(_csv_shape_error, path, ext, last_rows)
        elif ext in ("html", "htm"):
            # A single-file deliverable hides its JS in inline <script> blocks;
            # check each so a typo there (e.g. `content='…'` written `content=`)
            # is named with its HTML line instead of surfacing as a silent
            # blank page the model can only find by re-reading the whole file.
            html = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="replace")
            for start_line, src in _inline_js_blocks(html):
                diag = await _node_check_source(src)
                if diag:
                    err = _remap_node_diag(diag, start_line)
                    break
    except Exception:
        return ""
    if not err:
        return ""
    # §4RD: node/bash name the file by the HOST path they were given; the model
    # only knows the sandbox, so it ran `cd /Users/…/sandbox/webos` ("No such
    # file or directory") and then `find /` to locate it (req e2c7c7f4)
    # longest first: /private/tmp/x contains /tmp/x (r1)
    try:
        _resolved = str(path.resolve()) if path.is_absolute() else ""
    except (OSError, RuntimeError):      # a symlink loop (r2)
        _resolved = ""
    for _host in sorted({str(path), _resolved} - {""}, key=len, reverse=True):
        err = err.replace(_host, str(filename))
    _first = err.splitlines()[0] if err.splitlines() else err
    if in_progress:
        # operator (2026-10-12): a part of a file cannot parse until its last part
        # lands — informative, not an error line
        pretty_log("syntax check", f"{filename}: deferred — the file is still being written in parts "
                                   f"(so far: {_first}); it is checked when the last part lands",
                   icon=Icons.REQ_WAIT)
    else:
        pretty_log("syntax check", f"{filename}: {_first}", icon=Icons.FAIL, level="WARNING")
    # Error-keyed fix recipes (2026-07-18): for known traps the bare parser
    # message sends the model into a fix spiral — the xrick session failed
    # FOUR consecutive repairs of `if c == '\':` (unterminated string
    # literal: replace, replace, full rewrite with the SAME bug, sed) and a
    # later request rewrote the identical bug from scratch. Name the exact
    # cause and the exact fix in the same breath as the error.
    recipe = ""
    _err_low = err.lower()
    for _pat, _rec in _SYNTAX_FIX_RECIPES:
        if _pat in _err_low:
            recipe = f"\nRECIPE: {_rec}"
            break
    return (
        f"\n⚠ SYNTAX CHECK FAILED: '{filename}' was written but does NOT "
        f"parse. Fix this BEFORE any other step — a browser loads a broken "
        f"script silently and every downstream symptom (dead buttons, blank "
        f"page) traces back here:\n{err}{recipe}"
    )


# Known parse-error traps → the exact fix. Matched as substrings of the
# lower-cased checker message. Kept tiny and high-precision (same philosophy
# as tool_failure._FALLBACK_HINTS): a recipe on the wrong error is noise.
def _csv_shape_error(path: Path, ext: str, last_rows: Optional[int] = None) -> str:
    """"" when the rows have the header's field count; else which rows do not.
    Line numbers are PHYSICAL (a quoted field may span lines — review 14), so a
    ranged read / edit lands on the row. ``last_rows``: judge only the last N
    data rows (an append answers for what it appended, not for the file's
    earlier shape). A file whose EVERY data row differs from the header by
    the same count is ragged by design (logs, exports) — not an error."""
    import csv
    delim = "\t" if ext == "tsv" else ","
    rows = []                                  # (physical start line, field count)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        prev_end = 0
        for r in reader:
            rows.append((prev_end + 1, len(r), bool(r)))
            prev_end = reader.line_num
    if len(rows) < 2:
        return ""
    want = rows[0][1]
    data = [(ln, n) for ln, n, nonempty in rows[1:] if nonempty]
    if not data:
        return ""
    if all(n == data[0][1] for _, n in data) and data[0][1] != want and len(data) > 1:
        return ""                              # ragged by design: every row agrees, the header differs
    judged = data[-last_rows:] if last_rows else data
    bad = [(ln, n) for ln, n in judged if n != want]
    if not bad:
        return ""
    first_ln, first_n = bad[0]
    return (f"{len(bad)} of {len(judged)} rows have a different number of fields than the header "
            f"({want}): first at line {first_ln} with {first_n} fields")


_SYNTAX_FIX_RECIPES = (
    ("different number of fields than the header",
     "a comma (or tab) INSIDE a field splits it: put that field in double quotes "
     "(\"a fact, with a comma\") — or drop the comma. Rows with FEWER fields are "
     "missing a value. Fix the listed rows with 'edit'; do not rewrite the file."),
    ("unterminated string literal",
     "when the flagged line contains a backslash, the backslash is swallowing "
     "the closing quote (classic: `if c == '\\':`). A SINGLE literal backslash "
     "is written '\\\\' (two backslashes) in source, or use chr(92). Do NOT "
     "re-emit the same line unchanged — rewrite exactly that comparison. If "
     "there is no backslash, an earlier quote on the line is unclosed."),
    ("eol while scanning string literal",
     "same trap as 'unterminated string literal': a trailing backslash or an "
     "unclosed quote on that line. A single literal backslash is '\\\\' "
     "(doubled) or chr(92)."),
    ("unexpected character after line continuation",
     "a stray backslash sits outside any string on that line — delete it, or "
     "double it if a literal backslash was intended."),
    ("closing parenthesis '}' does not match",
     "an f-string brace holds a `[`, `(` or quote the parser can't nest — "
     "pre-compute the value into a plain variable first, then interpolate "
     "the bare name (v = d['k'][0]; f\"got {v}\")."),
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
    partial fix). Only .py / .json are checked in-process here; .js/.html
    get the same regression semantics via the async, node-backed
    `_syntax_regression_js_html` companion (both run in
    `_write_replace_guarded`); other types pass through (best-effort).
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
        except Exception:
            # ast.parse can raise UnicodeEncodeError/ValueError (not a
            # SyntaxError) on content carrying lone surrogates — which the
            # surrogateescape read preserves for a file with stray non-UTF-8
            # bytes. We can't judge a regression then, so FAIL OPEN (allow the
            # edit) rather than crash the replace.
            return ""
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
        except Exception:
            return ""
    return ""


# A line that IS a SEARCH/REPLACE marker: a full line of 4+ equals (the
# parser's separator — width-tolerant since 2026-07-19, when the 7-char
# git-conflict dialect `=======` leaked into a file UNDETECTED because this
# regex only knew the exact-4 form), a `<<<< SEARCH`-style opener, or a
# `>>>>`(` REPLACE`)-style closer. Used to catch marker text leaking INTO
# file content. An RST/Markdown `====` underline CAN now match, but the
# leak guard is count-aware — only lines ADDED relative to the previous
# content flag, and the reject message steers to operation='write' for the
# rare legitimate case.
# Leading [ \t]* tolerance matches the loose parser (which accepts indented
# separators) — without it an indented `    ====` sailed past the write
# backstop. Git-conflict marker lines (`<<<<<<< HEAD` / `>>>>>>> <ref>`)
# also count as markers: a SEARCH text carrying them (a conflict-resolution
# edit) is otherwise mis-split by the envelope parse — the conflict's own
# `=======` line hijacks the separator while its <<<<<<</>>>>>>> lines were
# invisible here (suffix != SEARCH/REPLACE), so the mangled block applied.
_MARKER_LINE_RE = re.compile(
    r"^[ \t]*(?:={4,}"
    r"|<{4,}[ \t]*(?:SEARCH.*)?"
    r"|>{4,}[ \t]*(?:REPLACE.*)?"
    r"|<{4,}[ \t]+\S.*"
    r"|>{4,}[ \t]+\S.*"
    r")[ \t]*$")


def _marker_leak(prev_content: str, new_content: str) -> str:
    """Return a short description of SEARCH/REPLACE marker lines that
    ``new_content`` would ADD relative to ``prev_content``, else "".

    A replace result must never contain MORE marker lines than the file
    already had: markers appearing in the output mean block syntax leaked
    into file content (observed live 2026-07-14 — a multi-edit envelope
    pasted its extra ``====`` separators plus the second edit's text into
    index.html and reported SUCCESS; the user's next message was the
    browser's `Unexpected token '==='`). Count-aware on purpose: a file
    that already carries leaked markers stays editable (the cleanup edit
    REMOVES markers, never adds them).
    """
    def _counts(text: str) -> dict:
        c: dict = {}
        for ln in text.splitlines():
            if _MARKER_LINE_RE.match(ln):
                key = ln.strip()
                c[key] = c.get(key, 0) + 1
        return c
    prev_counts = _counts(prev_content)
    added = [k for k, n in _counts(new_content).items()
             if n > prev_counts.get(k, 0)]
    return ", ".join(repr(k) for k in sorted(added))


async def _syntax_regression_js_html(prev_content: str, new_content: str,
                                     filename: str) -> str:
    """node-backed companion to `_syntax_regression` for .js/.mjs/.cjs files
    and inline <script> blocks in .html/.htm.

    Same contract: return a one-line error only when ``new_content``
    INTRODUCES a parse failure that ``prev_content`` did not have, else ""
    (fail open when node is unavailable). Kept separate from the sync
    `_syntax_regression` because `node --check` needs a subprocess; the
    write path awaits both. Before this existed, the syntax-rollback guard
    covered .py/.json only, so a replace could corrupt an HTML deliverable
    and still report SUCCESS (2026-07-14 marker-leak incident).
    """
    ext = str(filename).split(".")[-1].lower()
    try:
        if ext in ("js", "mjs", "cjs"):
            diag = await _node_check_source(new_content)
            if not diag:
                return ""
            if await _node_check_source(prev_content):
                return ""  # already broken — not a regression
            first = next((ln for ln in diag.splitlines() if ln.strip()), "syntax error")
            return first[:200]
        if ext in ("html", "htm"):
            new_err = ""
            for start_line, src in _inline_js_blocks(new_content):
                diag = await _node_check_source(src)
                if diag:
                    new_err = _remap_node_diag(diag, start_line)
                    break
            if not new_err:
                return ""
            for _start, src in _inline_js_blocks(prev_content):
                if await _node_check_source(src):
                    return ""  # already broken — not a regression
            return new_err
    except Exception:
        return ""  # best-effort: a broken checker must never block an edit
    return ""


def _would_be_snippet(new_content: str, regression: str,
                      context: int = 2) -> str:
    """Numbered lines of the REJECTED content around the error line named
    in ``regression`` ('… (line N, col M)'), '>' marking the line itself.
    Empty when no line number is present or it's out of range."""
    m = re.search(r"\(line (\d+)", regression or "")
    if not m:
        return ""
    ln = int(m.group(1))
    lines = (new_content or "").split("\n")
    if not (1 <= ln <= len(lines)):
        return ""
    lo, hi = max(1, ln - context), min(len(lines), ln + context)
    return "\n".join(
        f"{'>' if i == ln else ' '}{i:5d} | {lines[i - 1][:160]}"
        for i in range(lo, hi + 1))


async def _write_replace_guarded(path: Path, prev_content: str, new_content: str,
                                 filename: str, success_msg: str, *,
                                 marker_guard: bool = True,
                                 writer=None) -> str:
    """Apply a replace result with a syntax-regression rollback guard.

    If ``new_content`` would introduce a NEW syntax error (see
    `_syntax_regression` for .py/.json, `_syntax_regression_js_html` for
    .js/.html) or would ADD SEARCH/REPLACE marker lines to the file (see
    `_marker_leak`), the file is left UNCHANGED and a REJECTED message
    is returned steering the model to a tight surgical edit instead of a
    full-file rewrite. Otherwise the content is written and normal
    post-write syntax feedback is appended.
    """
    # The receipt follows the file: after a write lands, the request HAS
    # seen the new bytes, so a run of consecutive edits costs one read, not
    # one read per edit. Stamped on the post-guard content only — a rolled
    # back edit must not ground anything.
    # ``marker_guard`` is OFF for operation='edit': the leak this guard
    # catches is the ENVELOPE parser spilling its own markers into the file,
    # and `edit` has no envelope — while a Markdown setext underline
    # (`====`) is a line an edit must be allowed to write. `edit` refuses
    # marker lines in its ARGUMENTS instead (see `_edit_envelope_refusal`).
    leak = _marker_leak(prev_content, new_content) if marker_guard else ""
    if leak:
        pretty_log("Replace Rejected",
                   f"{filename}: edit would write SEARCH/REPLACE marker "
                   f"line(s) into the file ({leak}) — file left unchanged",
                   icon=Icons.WARN, level="WARNING")
        return (
            ToolOutcome.rejected(f"REJECTED: that replace would have written SEARCH/REPLACE "
            f"marker line(s) ({leak}) into '{filename}' as literal file "
            f"content — '{filename}' is unchanged on disk. This usually "
            f"means you packed more than one edit into a single "
            f"<<<< SEARCH block. Emit ONE complete envelope per edit "
            f"(each with its own '<<<< SEARCH', '====' and '>>>>' lines); "
            f"multiple envelopes in one call are fine. If the file "
            f"legitimately needs a line of exactly '====', use "
            f"operation='write' or an execute script instead.")
        )
    regression = _syntax_regression(prev_content, new_content, filename)
    if not regression:
        regression = await _syntax_regression_js_html(
            prev_content, new_content, filename)
    if regression:
        # Detail in the LOG LINE too (req A3, 2026-07-17): the tool result
        # always carried the error+line, but the operator-facing log said
        # only "would break syntax", so three straight rejections read as
        # an opaque mystery from the stream.
        pretty_log("Replace Rejected",
                   f"{filename}: edit would break syntax ({regression}) — "
                   f"file left unchanged",
                   icon=Icons.WARN, level="WARNING")
        # Show the model the WOULD-BE lines around the error. Knowing
        # "unexpected indent (line 238)" wasn't enough — the model blamed
        # "hidden characters" and escaped to an unguarded patch script
        # that corrupted the file. Seeing its own broken output makes the
        # mistake self-evident.
        _snippet = _would_be_snippet(new_content, regression)
        return (
            ToolOutcome.rejected(f"REJECTED: that replace would introduce a syntax error and was "
            f"NOT applied — '{filename}' is unchanged on disk: {regression}."
            + (f"\nThe REJECTED result around that line would have read:\n"
               f"{_snippet}\n" if _snippet else " ")
            + f"Your replacement's indentation or structure is off. "
            f"Re-read those lines, then make the change with operation='edit' "
            f"(old_string = the exact current lines, new_string = the corrected "
            f"lines, same indentation). Do NOT rewrite the whole file.", reason_code="syntax_regression_rolled_back")
        )
    # errors="surrogateescape": new_content is the replace result, which splices
    # the LLM's (clean) replacement into the untouched file body — that body may
    # carry lone surrogates from bytes the surrogateescape READ preserved. Write
    # them back as the exact original bytes; a default (strict) write would raise
    # UnicodeEncodeError on those surrogates.
    # ⚠ ENCODE BEFORE TRUNCATING. `write_text` opens with 'w': it truncates
    # first and encodes second, so a lone surrogate (U+D800-DBFF, which
    # `surrogateescape` does NOT handle) left the file at ZERO BYTES — and
    # the resulting UnicodeEncodeError is a ValueError, caught upstream and
    # returned WITHOUT an "Error:" prefix, so the turn booked a success with
    # world-changed credit and the wrapper stamped a receipt over the empty
    # file. `tool_write_file` already had this guard; this path did not —
    # "one-path fix ships half", and grounding was then hung off the
    # unfixed one.
    try:
        _payload_bytes = new_content.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError as _ue:
        return ToolOutcome.rejected(
            f"REJECTED: the replacement text for '{filename}' contains "
            f"characters that cannot be encoded as UTF-8 (lone surrogate "
            f"{new_content[_ue.start:_ue.end]!r} at offset {_ue.start}). "
            f"'{filename}' is UNCHANGED. Remove or re-escape those "
            f"characters and retry.", reason_code="lone_surrogate")
    # ``writer`` (operation='edit'): an fd-based O_NOFOLLOW write that
    # re-checks the target's identity against the read before truncating,
    # so a symlink swapped in after the safe-path check never receives the
    # bytes. `replace` keeps the plain path write (its own review item).
    if writer is not None:
        try:
            await writer(_payload_bytes)
        except _FileChangedUnderneath:
            pretty_log("Replace Rejected",
                       f"{filename}: target changed between the read and the "
                       f"write (another writer, or a swapped-in link) — "
                       f"nothing written",
                       icon=Icons.WARN, level="WARNING")
            return ToolOutcome.rejected(
                f"REJECTED: '{filename}' changed on disk between the read and "
                f"the write (another process, or the path now points "
                f"elsewhere), so the edit was NOT applied — nothing was "
                f"written. Re-read the file and retry.",
                reason_code="file_changed_underneath")
    else:
        await asyncio.to_thread(path.write_bytes, _payload_bytes)
    # Whole-file defs diff: symbols defined in the previous content but not
    # in the new content were removed by this edit; surviving references to
    # them get named in the result (see _orphaned_symbol_warning).
    _end = ""
    if _is_in_progress(path):
        # §4SH: the edit finished the document. r1: only when THIS edit made it end — a
        # typo fix in a part-1 skeleton that already ran through </html> closes nothing
        _end = (document_ends_here(filename, new_content)
                if not document_ends_here(filename, prev_content) else "")
        if _end:
            _mark_in_progress(path, False)
    _open = _is_in_progress(path)
    # §4SH r3: a file still open for parts gets the deferred verdict, as its write and
    # append parts do ("Fix this BEFORE any other step" told the model to close it early)
    _syn = await _syntax_feedback(path, filename, in_progress=_open)
    _text = (success_msg
             + _orphaned_symbol_warning(prev_content, new_content, new_content)
             + (_PARTS_PENDING_NOTE if (_open and _syn) else _syn))
    if _end:
        _text += _finished_by_end_note(filename, _end)
    elif _open:
        _text += OPEN_PART_NOTE                  # r4: as its write and append parts say
    if str(filename).lower().endswith((".html", ".htm")) and not _open:
        _text += await asyncio.to_thread(html_page_notes, new_content)   # §4OI (iii) + §4SH: advisory
    if _syn and not _open:
        return ToolOutcome.partial(_text, world_changed=True, reason_code="written_but_syntax_failed")
    return _text


# §4JP (2026-09-21, req fd89fd6d): the model rewrote a 15–30 KB HTML file
# from scratch SEVEN times in one request (126 KB generated, 90–256 s of
# generation each — half of a 30-minute budget) while `replace` was used
# twice. From the third full rewrite of a large existing file the success
# line says so and names the cheaper operation. Per request, per path.
REWRITE_HINT_AT = 3
REWRITE_HINT_MIN_BYTES = 8000
_REWRITES: "dict[tuple, int]" = {}
_REWRITES_MAX = 512


def note_full_rewrite(req_id: str, rel_path: str, prev_bytes: int) -> str:
    """Count this full write of an EXISTING file of prev_bytes and return the
    hint once the count reaches REWRITE_HINT_AT (and every time after)."""
    if prev_bytes < REWRITE_HINT_MIN_BYTES:
        return ""
    key = (str(req_id or ""), str(rel_path))
    if len(_REWRITES) >= _REWRITES_MAX and key not in _REWRITES:
        for k in list(_REWRITES)[: _REWRITES_MAX // 4]:
            _REWRITES.pop(k, None)
    n = _REWRITES.get(key, 0) + 1
    _REWRITES[key] = n
    if n < REWRITE_HINT_AT:
        return ""
    return (f" NOTE: this is full rewrite #{n} of an existing {prev_bytes:,}-byte file in this request — "
            "each rewrite regenerates the whole file (minutes of generation). For further changes use "
            "operation='edit' (old_string = just the lines that change, new_string = their replacement).")


def _stray_root_prefix_note(sandbox_dir, path) -> str:
    """§4QA: inside a project, a write that CREATES a first-level folder named
    like a folder at the sandbox ROOT (outside projects/) from before the
    project. Live (448d5afa86f0): tasks.md was written to
    /workspace/recursive-thinking-experiment/ before the project existed; once
    it did, the model kept that prefix, so "research/x.md" landed in
    projects/<id>/recursive-thinking-experiment/research/x.md and five later
    reads failed.

    A NOTE on the successful write, not a refusal (fresh-eye M1): a root
    ``src/`` or ``results/`` is just as old, and refusing the project's own
    first ``src/main.py`` broke builds and probe_model saves. Judged on the
    RESOLVED path, so ``/workspace/<head>/…`` and ``<head>/…`` agree. Call
    BEFORE the write: "" when the folder already exists in the project."""
    if not project_download_prefix(sandbox_dir):
        return ""
    try:
        ws = Path(str(sandbox_dir)).resolve()
        parts = Path(path).resolve().relative_to(ws).parts
        if len(parts) < 2 or Path(path).exists():
            return ""
        head = parts[0]
        if os.path.lexists(ws / head):
            return ""                        # the project has that folder already
        root_dir = ws.parent.parent / head
        if not root_dir.is_dir() or root_dir.is_symlink():
            return ""
        _ws_st = ws.stat()
        _born = getattr(_ws_st, "st_birthtime", None) or _ws_st.st_ctime
        if root_dir.stat().st_mtime > _born:
            return ""                        # the root folder is not from before the project
    except (OSError, ValueError):
        return ""
    rest = "/".join(parts[1:])
    return (f" NOTE: this created a '{head}/' folder inside the project. A '{head}/' folder also "
            f"exists at the sandbox ROOT from before this project — if '{head}/' was that old "
            f"folder's name, project paths do not need it (they are relative to "
            f"'{project_download_prefix(sandbox_dir)}'): the file belongs at '{rest}' "
            f"(file_system operation=move). If you meant a '{head}/' folder in the project, "
            f"nothing to do.")


async def tool_write_file(filename: str, content: Any, sandbox_dir: Path, more_parts: bool = False):
    # Include the size so the log says HOW MUCH was written, not just the path
    # (a path-only line can't tell an empty write from a 40 KB one).
    _n = len(content) if isinstance(content, (str, bytes)) else len(str(content or ""))
    pretty_log("File Write", f"{filename} ({_n} chars)", icon=Icons.TOOL_FILE_W)
    try:
        # Reject a missing/empty payload, or the LITERAL Python None the LLM
        # sometimes emits as a string. Do NOT reject a legitimate file whose
        # content is the word "none"/"None" or is only whitespace — match the
        # bare token exactly, not any content that lowercases to "none".
        _c_str = "" if content is None else str(content)
        # §4LL: an EMPTY marker file is a real request (`__init__.py`, `.gitkeep`)
        _marker = Path(str(filename)).name.lower() in _EMPTY_OK_NAMES
        if content is None or (_c_str.strip() == "" and not (_marker and content == "")) \
                or _c_str.strip() == "None":
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
        # §4NU r1: only a payload WRAPPED in a fence (one opening within its
        # first 200 chars, closing at its end) is unwrapped. An HTML page with
        # a ```js sample inside a <pre> was saved as just that sample — "Wrote
        # 10 chars" — and a part of a file never matches.
        _cs = content.strip()
        if (ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]
                and not more_parts and _cs.endswith("```") and 0 <= _cs.find("```") < 200):
            from ..utils.sanitizer import extract_code_from_markdown
            content = extract_code_from_markdown(content, filename=filename)

        path = _get_safe_path(sandbox_dir, filename)

        # §4KB: a whole-file WRITE over an EXISTING file is the most
        # destructive edit shape there is — it discards everything the model
        # did not re-emit. Gated on the same receipt as `replace`. A write
        # that CREATES a file is never gated: there are no bytes to have
        # read, and new-file creation is the leaf's normal first move.
        # ⚠ `is_file()` is FALSE for a FIFO/device, so this gate used to
        # be SKIPPED for exactly the targets `replace` refuses outright — and
        # the atomic write then replaced the FIFO with a regular file, ungated.
        # Refuse a non-regular existing target the same way, before the gate.
        if path.exists() and not path.is_file():
            return (f"Error: '{filename}' exists and is not a regular file "
                    f"(it is a FIFO, device, socket or directory), so it "
                    f"cannot be written as text. '{filename}' is unchanged.")

        _stray_note = _stray_root_prefix_note(sandbox_dir, path)
        _unread = unread_rewrite_refusal(path, filename, more_parts=bool(more_parts))   # §4OE-2
        if _unread:
            return ToolOutcome.rejected("REJECTED: " + _unread, reason_code="unread_rewrite")
        # §4NY: a NEW throwaway test/debug script is noted for the end-of-turn
        # cleanup (kept when the reply or the user names it, or tests were asked for)
        from .turn_scratch import is_scratch_test_name, note_created
        if is_scratch_test_name(path.name):
            note_created(path, "test")
        # SELF-HEALING: Auto-create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always UTF-8: without an explicit encoding, write_text uses the
        # process locale (comma-decimal Greek Macs, LANG unset), which
        # mojibakes non-ASCII content. Encode BEFORE touching the target:
        # write_text opens with truncate, so a strict-encode failure (a lone
        # surrogate like "\ud800" surviving json.loads) used to wipe the
        # file to 0 bytes AND surface as a bare UnicodeEncodeError string
        # with no 'Error:' prefix — recorded as written by callers keying
        # on that prefix.
        try:
            _payload = content.encode("utf-8")
        except UnicodeEncodeError as ue:
            return (f"Error: content for '{filename}' contains characters that "
                    f"cannot be encoded as UTF-8 (lone surrogate "
                    f"{content[ue.start:ue.end]!r} at offset {ue.start}). "
                    f"'{filename}' was NOT modified. Remove or re-escape those "
                    f"characters and retry.")

        def _atomic_write():
            _swap0 = _parent_swapped(sandbox_dir, path)      # §4PF r4 (F15): judged IN the thread, before mkstemp
            if _swap0:
                return _swap0
            # tmp-then-rename so a failure mid-write never leaves the target
            # truncated or half-written.
            import tempfile
            fd, _tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(_payload)
                # mkstemp creates 0600; keep the target's existing mode (or
                # a normal 0644) so container-side readers aren't locked out.
                try:
                    _mode = path.stat().st_mode & 0o7777
                except OSError:
                    _mode = 0o644
                os.chmod(_tmp, _mode)
                _swap1 = _parent_swapped(sandbox_dir, path)   # …and again right before the replace
                if _swap1:
                    os.unlink(_tmp)
                    return _swap1
                _commit_replace(_tmp, path)                     # §4SD
            except BaseException:
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
                raise
        try:
            _prev_bytes = path.stat().st_size if path.exists() else -1
        except OSError:
            _prev_bytes = -1
        _swapped = await asyncio.to_thread(_retry_transient, _atomic_write)      # §4SD; §4PF r4 (F15): a refusal string when the dir was swapped
        if _swapped:
            return _swapped
        try:
            _mark_written(path)             # §4OG-1 (h): the stamp follows THIS request's own write (direct callers too)
        except Exception:  # noqa: BLE001
            pass
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
        syntax_note = await _syntax_feedback(path, filename, in_progress=bool(more_parts))
        rewrite_note = note_full_rewrite(request_id_context.get() or "", rel_str, _prev_bytes) if _prev_bytes >= 0 else ""
        _text = (
            f"SUCCESS: Wrote {len(content)} chars to '{filename}'. "
            f"Script-side path (from sandbox cwd): '{rel_str}'."
            f"{summary}{syntax_note}{rewrite_note}{_stray_note}"
        )
        if not more_parts and ext in ("html", "htm"):
            _text += await asyncio.to_thread(html_page_notes, content)   # §4OI (iii) + §4SH: advisory, never a failure
        if syntax_note and more_parts:
            # §4NU: part 1 of a file written in parts — not parseable yet
            return (f"SUCCESS: Wrote {len(content)} chars to '{filename}'. "
                    f"Script-side path (from sandbox cwd): '{rel_str}'.{summary}{rewrite_note}{_stray_note}"
                    + _PARTS_PENDING_NOTE)
        if syntax_note:
            return ToolOutcome.partial(_text, world_changed=True,
                                       reason_code="written_but_syntax_failed")
        return _text
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

_DOC_OPENERS = (("html", re.compile(r"^\s*(?:<!doctype\s+html\b|<html\b)", re.IGNORECASE)),
                ("xml", re.compile(r"^\s*<\?xml\b", re.IGNORECASE)),
                ("shebang", re.compile(r"^#!")))


def _doc_kind(text) -> str:
    for kind, rx in _DOC_OPENERS:
        if rx.match(str(text or "")):
            return kind
    return ""


def restarts_file(existing_head, new_content) -> bool:
    """§4NW-b: does `new_content` start the file over? Only when BOTH open a
    document of the same kind (<!DOCTYPE html / <html, <?xml, a #! line) —
    a part can never legitimately do that. (Review: a "same first line"
    rule fired on a CSV part repeating its header, a repeated separator,
    `import os` — and replaced the earlier parts.)"""
    k = _doc_kind(existing_head)
    return bool(k) and k == _doc_kind(new_content)


#: §4NU r4: files THIS request is writing in parts — (req_id, real path).
#: A plain `write` to one replaced the part already written (probe-4nu-story:
#: "Part 2" sent as write+more_parts overwrote Part 1, three times).
_IN_PROGRESS: "dict[tuple, int]" = {}
_IN_PROGRESS_MAX = 512


def _progress_key(path: Path) -> tuple:
    try:
        rp = str(path.resolve())
    except Exception:  # noqa: BLE001
        rp = str(path)
    return (str(request_id_context.get() or ""), rp)


def _mark_in_progress(path: Path, on: bool) -> None:
    k = _progress_key(path)
    if not on:
        _IN_PROGRESS.pop(k, None)
        return
    if len(_IN_PROGRESS) >= _IN_PROGRESS_MAX and k not in _IN_PROGRESS:
        for old in list(_IN_PROGRESS)[: _IN_PROGRESS_MAX // 4]:
            _IN_PROGRESS.pop(old, None)
    _IN_PROGRESS[k] = 1


def _is_in_progress(path: Path) -> bool:
    return _progress_key(path) in _IN_PROGRESS


#: §4NZ: files THIS request wrote with a plain `write` (NOT an append, NOT a cut-salvaged part) — (req_id, real path)
_WRITTEN: "dict[tuple, int]" = {}


#: §4OE-2 (restores §4KB): files THIS request has READ — whole, ranged, or
#: by editing them (the receipt follows the file). A plain `write` over an
#: EXISTING file is refused until one of these stands: the edit ladder showed
#: 16 whole-file pushes (5.6 %) after the original gate had been lost (§4OD).
#: §4OG-1 (h): a receipt is COVERAGE + a VERSION — the §4OE-2 receipt was a
#: flag any 3-line ranged read set (a whole-file discard then passed, and the
#: refusal text pointed there), and nothing noticed the file changing between
#: the read and the write (read → own edit → whole write from the pre-edit
#: view succeeded). Value: {"ranges": [(s, e)], "bytes": [(a, b)], "cov": float,
#: "stamp": (mtime_ns, size)}.
_READ: "dict[tuple, dict]" = {}
#: the share of a file's lines a request must have seen before it may discard the rest
READ_COVERAGE_FLOOR = 0.80
#: a file this big is never line-counted for coverage (a whole read is capped long before)
_COVERAGE_MAX_BYTES = 8 * 1024 * 1024


def _file_stamp(path: Path):
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _count_lines(path: Path):
    """Line count for coverage, or None when the file is too big to bother."""
    try:
        if path.stat().st_size > _COVERAGE_MAX_BYTES:
            return None
        n = 0
        with open(path, "rb") as fh:
            for _ in fh:
                n += 1
        return n
    except OSError:
        return None


def _union_len(ranges, lo: int, hi: int) -> int:
    """Length of the union of closed integer ranges clipped to [lo, hi]."""
    pts = sorted((max(lo, int(s)), min(hi, int(e))) for s, e in ranges if s is not None and e is not None)
    total, cur_s, cur_e = 0, None, None
    for s, e in pts:
        if e < s:
            continue
        if cur_e is None or s > cur_e + 1:
            if cur_e is not None:
                total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s + 1
    return total


def _coverage(rec: dict, path: Path) -> float:
    if rec.get("whole"):
        return 1.0
    best = 0.0
    if rec.get("ranges"):
        n = _count_lines(path)
        if n:
            best = max(best, _union_len(rec["ranges"], 1, n) / float(n))
    if rec.get("bytes"):
        st = _file_stamp(path)
        if st and st[1] > 0:
            best = max(best, _union_len(rec["bytes"], 0, st[1] - 1) / float(st[1]))
    return min(1.0, best)


def _mark_read(path: Path, *, start=None, end=None, byte_from=None, byte_to=None, whole=None) -> None:
    """Record what this request SAW of `path`: a whole read (the default when
    no range is given — the bare call means "read whole"), a line range (as
    the tool actually showed it), or a byte range (a chunked page). The stamp
    is the file as it is now — a change by anything but this request's own
    tool calls invalidates it."""
    k = _progress_key(path)
    rec = _READ.get(k) or {"ranges": [], "bytes": [], "cov": 0.0, "stamp": None, "whole": False}
    if whole is None:
        whole = start is None and end is None and byte_from is None and byte_to is None
    if whole:
        rec["whole"] = True
    elif start is not None and end is not None:
        rec["ranges"].append((int(start), int(end)))
    elif byte_from is not None and byte_to is not None:
        rec["bytes"].append((int(byte_from), int(byte_to)))
    rec["cov"] = _coverage(rec, path)
    rec["stamp"] = _file_stamp(path)
    if len(_READ) >= _IN_PROGRESS_MAX and k not in _READ:
        for old in list(_READ)[: _IN_PROGRESS_MAX // 4]:
            _READ.pop(old, None)
    _READ[k] = rec


def _read_this_request(path: Path) -> bool:
    return _progress_key(path) in _READ


def read_coverage(path: Path) -> float:
    rec = _READ.get(_progress_key(path))
    return float(rec.get("cov") or 0.0) if isinstance(rec, dict) else 0.0


def _forget_parts_under(path: Path) -> None:
    """Drop every in-progress entry for `path` (or under it, for a directory)."""
    try:
        rp = str(Path(path).resolve())
    except Exception:  # noqa: BLE001
        rp = str(path)
    for k in [k for k in _IN_PROGRESS if k and (k[1] == rp or str(k[1]).startswith(rp + os.sep))]:
        _IN_PROGRESS.pop(k, None)


def _move_parts(old_rp: str, new_path: Path) -> None:
    try:
        new_rp = str(Path(new_path).resolve())
    except Exception:  # noqa: BLE001
        new_rp = str(new_path)
    for k in [k for k in _IN_PROGRESS if k and (k[1] == old_rp or str(k[1]).startswith(old_rp + os.sep))]:
        v = _IN_PROGRESS.pop(k)
        _IN_PROGRESS[(k[0], new_rp + str(k[1])[len(old_rp):])] = v


def files_left_in_progress(req_id) -> list:
    """§4OI (iii): files THIS request started writing in parts (`more_parts`)
    whose last part never came — incomplete and never syntax-checked. The
    census voxel game was one, reported "complete — open it and play"."""
    rid = str(req_id or "")
    out = []
    for k in list(_IN_PROGRESS):
        if k and k[0] == rid:
            if not os.path.exists(str(k[1])):
                _IN_PROGRESS.pop(k, None)       # r2 m6: gone (a case-variant delete, an `rm` in execute) — not "unfinished"
                continue
            out.append(str(k[1]))
    return out


#: §4SH: the bound on a JSON file parsed whole to see whether a part ended it
DOC_END_JSON_MAX_BYTES = 8 * 1024 * 1024
#: extensions `_syntax_feedback` can judge (.js/.mjs/.cjs only with node)
_CHECKED_EXTS = frozenset({"py", "sh", "bash", "json", "csv", "tsv", "html", "htm"})
_NODE_EXTS = frozenset({"js", "mjs", "cjs"})


#: r5: the closing tag at the end, optionally followed by comments (`</html>\n<!-- built -->`)
_ENDS_WITH_RE = {k: re.compile(r"</" + k + r"(?:[\t\n\f\r /][^>]*)?>(?:\s*<!--(?:(?!-->)[\s\S])*-->)*\s*\Z", re.IGNORECASE)
                 for k in ("html", "svg")}
_END_WINDOW = 20000


def document_ends_here(filename, text) -> str:
    """§4SH: the closing token when `text` is a WHOLE document of the kind
    the file's extension names, else "". A format with a definite end says
    itself when the last part has landed — nothing legitimately follows
    `</html>`, and a JSON value that parses is complete — so a part that ends
    the document is the last part whatever `more_parts` said (req cb27058e:
    the part ending `</body></html>` was sent with more_parts=true and the
    file stayed "unfinished", unchecked, for 17 turns). Code formats are not
    judged: a .py/.js split at function boundaries parses after every part."""
    ext = str(filename).rsplit(".", 1)[-1].lower() if "." in str(filename) else ""
    t = str(text or "")
    if ext in ("html", "htm", "xhtml"):
        # r1/r2: `</html>` inside a still-open <script> (a template literal) ends nothing
        # r4: or inside a comment, <style>, tag or quoted value — the page is whole only outside all of them
        if not _ENDS_WITH_RE["html"].search(t[-_END_WINDOW:]) or html_tail_state(t, xml=ext == "xhtml"):
            return ""
        return "</html>"
    if ext == "svg":
        # r4/r5: the outermost <svg> closed, read by the scan (an `<svg` in a comment, CDATA
        # or quoted value is not an opener); a trailing comment may follow it
        if not _ENDS_WITH_RE["svg"].search(t[-_END_WINDOW:]):
            return ""
        state, depth, _seen = _html_scan(t, xml=True)
        return "</svg>" if not state and depth == 0 else ""
    if ext == "json" and t.strip() and len(t) <= DOC_END_JSON_MAX_BYTES:
        try:
            json.loads(t.lstrip("\ufeff"))       # r9: a BOM is not the document
        except (ValueError, RecursionError):     # r8: nesting too deep is no proof of an end
            return ""
        return "a whole JSON value"
    return ""


def _ext_of(filename) -> str:
    n = Path(str(filename)).name
    return n.rsplit(".", 1)[-1].lower() if "." in n else ""


def _whole_check_phrase(filename, text=None) -> str:
    """r1: what the finish can truthfully say — "checked whole" only for a
    kind something here checks (an .svg/.md/.txt was claimed checked too).
    r3: an .svg/.xhtml is judged by its closing tag (as the reply-time settle
    does); ``text`` None = the caller already found the document's end."""
    ext = _ext_of(filename)
    if ext in _CHECKED_EXTS or (ext in _NODE_EXTS and _find_node()):
        return "was checked whole"
    if ext in ("svg", "xhtml"):
        if text is None or document_ends_here(filename, text):
            return "was checked: it ends with its closing tag"
        return "does NOT end with its closing tag — if a part is missing, append it"
    return f"nothing here checks a .{ext or 'extensionless'} file, so look at how it ends"


#: r3: the elements whose content is RAW text in the HTML tokenizer — a `<script>`
#: inside them opens nothing (`<textarea><script></textarea>`, `<title>… <script> …`)
_RAWTEXT_TAGS = frozenset({"script", "style", "textarea", "title", "noscript", "xmp", "iframe",
                           "noembed", "noframes"})
_TAG_NAME_RE = re.compile(r"<([A-Za-z][^\s/>]*)")
_TAG_STOP_RE = re.compile(r">|=[ \t\r\n\f]*([\"'])")     # a quote opens a value only after `=`
_END_NAME_RE = re.compile(r"</([A-Za-z][^\s/>]*)")
_FOREIGN_TAGS = frozenset({"svg", "math"})     # r4: nothing in them is raw text
_COMMENT_END_RE = re.compile(r"--!?>")
#: r5: HTML5 — these start tags end foreign (svg/math) content; these elements hold HTML inside it
_HTML_BREAKOUT_TAGS = frozenset("b big blockquote body br center code dd div dl dt em embed h1 h2 h3 h4 h5 h6 "
                                "head hr i img li listing menu meta nobr ol p pre ruby s small span strong "
                                "strike sub sup table tt u ul var".split())
_PAGE_MARKS = frozenset({"!doctype", "html", "head", "body"})
_INTEGRATION_TAGS = frozenset({"foreignobject", "desc", "title", "annotation-xml",
                               "mi", "mo", "mn", "ms", "mtext"})   # r10: the MathML text points too


def _start_tag_end(t: str, j: int) -> int:
    """Index after the `>` closing a start tag whose name ends at ``j`` — a
    `>` inside a quoted attribute value is text. -1: the file ends inside the
    tag. Linear (a regex with quoted alternatives rescanned to the end for
    every `<` in an unclosed quote)."""
    while True:
        m = _TAG_STOP_RE.search(t, j)
        if not m:
            return -1
        if not m.group(1):
            return m.end()
        # r5: inside an UNQUOTED value (`href=/s?q="`) a quote is text — the
        # word before this `=` already holds one (bounded by j: linear)
        eq = m.start()
        ws = max(t.rfind(c, j, eq) for c in " \t\n\r\f")
        if "=" in t[max(ws + 1, j):eq]:
            j = m.end()
            continue
        q = t.find(m.group(1), m.end())
        if q < 0:
            return -1
        j = q + 1


def _html_scan(text: str, *, xml: bool = False):
    """r2–r9: (where the page ENDS, the svg/math depth there, the html tags
    seen as tags), read with the
    HTML tokenizer's rules for what matters here — "" (outside everything: the page may be whole),
    "script" (its last <script> is never closed), "<name>" for another
    raw-text element (style, textarea, title, …), "comment", "tag" (inside a
    tag, a quoted value or a doctype). A raw-text element runs to its first
    `</name` + tab/LF/FF/CR/space, `/` or `>` (so `const t = "<script>"`
    inside a script is text); comments (`<!-->` and `<!--->` are whole, `--!>`
    ends one), CDATA inside <svg>/<math> (elsewhere a bogus comment to `>`),
    doctype/processing instructions and attribute values (`data-x="<script>"`)
    open nothing; inside <svg>/<math> nothing is raw text (an svg `<script/>`
    or `<title/>` closes itself) until an HTML breakout tag (`<div>`, `<p>`, …)
    ends it or an integration point (foreignObject, desc, title,
    annotation-xml, the MathML text points mi/mo/mn/ms/mtext) holds HTML;
    ``seen``: the doctype and html/head/body/`</html>` met as TAGS; ``xml`` (XHTML, .svg):
    `<script …/>` is a complete element. Indices are on the text itself
    (r4: a lowercased copy is longer after `İ`). Linear. Not modelled: the
    script "double-escape" (`<!--` + `<script>` inside a script)."""
    t = str(text or "")
    n, pos, foreign = len(t), 0, 0
    islands: list = []          # r5: (element, foreign depth) of an HTML island inside svg/math
    seen: set = set()           # r9: "html" / "/html" seen as TAGS (not inside a script string)
    while True:
        i = t.find("<", pos)
        if i < 0 or i + 1 >= n:
            return "", foreign, seen
        if t.startswith("<!--", i):
            if t.startswith("<!-->", i):
                pos = i + 5
                continue
            if t.startswith("<!--->", i):
                pos = i + 6
                continue
            end = _COMMENT_END_RE.search(t, i + 4)
            if not end:
                return "comment", foreign, seen
            pos = end.end()
            continue
        if foreign and t.startswith("<![CDATA[", i):
            end = t.find("]]>", i + 9)
            if end < 0:
                return "comment", foreign, seen
            pos = end + 3
            continue
        if t[i + 1] == "/":
            m = _END_NAME_RE.match(t, i)
            end = t.find(">", i + 2)
            if end < 0:
                return "tag", foreign, seen
            _en = m.group(1).lower() if m else ""
            if _en in ("html", "body"):
                seen.add("/" + _en)                  # r12: HTML5 may omit </html> after </body>
            if islands and _en == islands[-1][0]:
                foreign = islands.pop()[1]           # back inside the svg/math
            elif foreign and _en in _FOREIGN_TAGS:
                foreign -= 1
            pos = end + 1
            continue
        if t[i + 1] in "!?":                         # doctype, PI: to the next `>`
            end = t.find(">", i + 2)
            if end < 0:
                return "tag", foreign, seen
            if t[i + 1:i + 9].lower() == "!doctype":
                seen.add("!doctype")                 # r11: a document, not a fragment
            pos = end + 1
            continue
        m = _TAG_NAME_RE.match(t, i)
        if not m:
            pos = i + 1                              # a `<` that starts no tag is text
            continue
        end = _start_tag_end(t, m.end())
        if end < 0:
            return "tag", foreign, seen                    # the file ends inside a tag
        name = m.group(1).lower()
        pos = end
        if name in ("html", "head", "body"):
            seen.add(name)                           # r11: what makes it a whole page (not a fragment)
        closes_itself = t[m.end():end - 1].rstrip().endswith("/")
        if name in _FOREIGN_TAGS:
            foreign += not closes_itself
            continue
        if foreign and not xml:
            if name in _HTML_BREAKOUT_TAGS:
                foreign, islands = 0, []             # HTML5: this tag ends the foreign content
            elif name in _INTEGRATION_TAGS and not closes_itself:
                islands.append((name, foreign))      # foreignObject/desc/title: HTML inside
                foreign = 0
            continue
        if foreign or name not in _RAWTEXT_TAGS or (xml and closes_itself):
            continue
        close = re.compile(r"</" + re.escape(name) + r"[\t\n\f\r />]", re.IGNORECASE | re.ASCII).search(t, pos)
        if not close:
            return name, foreign, seen                     # a raw-text element runs to the end
        end = t.find(">", close.end() - 1)
        pos = n if end < 0 else end + 1


def html_tail_state(text: str, *, xml: bool = False) -> str:
    """Where the page ends (see `_html_scan`): "" outside everything, else
    "script", another raw-text tag name, "comment" or "tag"."""
    return _html_scan(text, xml=xml)[0]


def html_unclosed_script(text: str, *, xml: bool = False) -> bool:
    """r2: the page's last <script> is never closed (see `html_tail_state`)."""
    return html_tail_state(text, xml=xml) == "script"


def html_cut_finding(text: str) -> str:
    """r2: the one judgement of whether a page looks cut short — the finish
    note and the reply-time settle both say it (they disagreed: a
    skeleton-first page was "fine" at the reply and "missing a part" at the
    finish). "" when the page reads as whole."""
    t = str(text or "")
    state, _depth, seen = _html_scan(t)
    if state == "script":
        return "its last <script> is never closed, so it looks cut short"
    if state:                                    # r4: a comment, <style>, tag or quote left open
        what = {"comment": "a comment", "tag": "a tag (or a quoted value)"}.get(state, f"<{state}>")
        return f"it ends inside {what} that is never closed, so it looks cut short"
    # r9–r11: a `</html>` in a script string is not the end; a fragment (no doctype, <html>,
    # <head> or <body>) needs none — a page cut before its <html> tag still does
    if seen & _PAGE_MARKS and not seen & {"/html", "/body"}:
        return "it ends without </html>, so it looks cut short"
    return ""


def _finished_by_end_note(filename, end: str) -> str:
    return (f"\nNOTE: this ends the document ({end}), so it was taken as the LAST part — '{filename}' is "
            f"finished and {_whole_check_phrase(filename)}. (If more parts do follow, append them as usual: "
            f"the file reopens.)")


#: §4SH: said on every part that leaves the file open — how to close it is
#: always within reach (the model that mis-set the flag had no way out)
OPEN_PART_NOTE = (" The file stays OPEN for the next part (append=true). If this was the last part, finish it: "
                  "operation='write', append=true, content='' — nothing is added, the file is closed and the "
                  "finish says what was checked.")


def _first_line(note: str) -> str:
    return next((ln.strip() for ln in str(note or "").splitlines() if ln.strip()), "")[:240]


async def finish_parts_file(path: Path, filename: str):
    """§4SH: `write` + append=true + empty content on a file open for parts:
    nothing is added, the file is closed and checked whole."""
    _mark_in_progress(path, False)
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error: {e}"
    syn = await _syntax_feedback(path, filename)
    is_html = str(filename).lower().endswith((".html", ".htm"))
    page = await asyncio.to_thread(html_page_notes, text) if is_html else ""
    end = ""
    _cut = html_cut_finding(text) if is_html else ""
    if _cut:
        end = f"\nNOTE: {_cut} — if a part is missing, append it."
    out = (f"SUCCESS: Finished '{filename}' — written in parts, nothing added now; it is "
           f"{len(text.encode('utf-8')):,} bytes and {_whole_check_phrase(filename, text)}.{syn}{page}{end}")
    pretty_log("File Finished", f"{filename}: closed by an empty last part"
               + (f" — {_first_line(syn)}" if syn else ""), icon=Icons.TOOL_FILE_W)
    if syn:
        return ToolOutcome.partial(out, world_changed=False, reason_code="written_but_syntax_failed")
    return out


async def settle_open_parts(req_id) -> list:
    """§4SH: at the reply, the files THIS request left open for parts are
    closed by the system and judged by what is IN them — not by the flag.
    Returns [(name, finding)]; finding "" = the file passes its checks (the
    reply then says nothing: req cb27058e told the owner a complete page,
    tested three times in the browser, was "incomplete and never checked")."""
    rid = str(req_id or "")
    out = []
    for p in files_left_in_progress(rid):
        _IN_PROGRESS.pop((rid, p), None)
        path, name = Path(p), Path(p).name
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except OSError:
            continue
        finding = _first_line(await _syntax_feedback(path, name))
        if ext in ("html", "htm"):
            # r1: a page whose </html> came in an EARLIER part (the skeleton first, its
            # scripts appended after it) renders; r2: a never-closed <script> is cut short
            if not finding:
                finding = html_cut_finding(text)
            if not finding:
                finding = _first_line(await asyncio.to_thread(html_stray_script_note, text))
        elif ext in ("svg", "xhtml") and not finding and not document_ends_here(name, text):
            finding = "it ends without its closing tag, so it looks cut short"
        checked = ext in _CHECKED_EXTS or ext in ("svg", "xhtml") or (ext in _NODE_EXTS and _find_node())
        if not finding and not checked:
            tail = " ".join(text[-90:].split())
            finding = f"nothing here can check a .{ext or 'extensionless'} file, so look at how it ends: «…{tail}»"
        pretty_log("Parts File", f"{name}: left open by the last part — closed at the reply; "
                   + (finding or "it passes its checks (nothing said)"),
                   icon=Icons.WARN if finding else Icons.TOOL_FILE_W, level="WARNING" if finding else "INFO")
        out.append((name, finding))
    return out


def forget_request_receipts(req_id) -> int:
    """§4OG-1 (h): drop every receipt of a finished request (the tables were
    one 512-slot LRU across requests). Returns the number removed."""
    rid = str(req_id or "")
    n = 0
    for table in (_READ, _WRITTEN, _IN_PROGRESS):
        for k in [k for k in table if k and k[0] == rid]:
            table.pop(k, None)
            n += 1
    return n


def _note_mutation(path: Path) -> None:
    """A mutation by THIS request's own tool call (edit/replace/append):
    a file this request WROTE whole stays its own (the written stamp follows
    the file — it authored every byte, §4NU parts and appends); a file it
    only READ is now a different file than the one it saw, so the read
    receipt's stamp is voided and a whole-file write must follow a fresh
    whole read (reader M1: a whole write from the pre-edit view silently
    discarded the edit). An edit grants no COVERAGE either way."""
    k = _progress_key(path)
    wrec = _WRITTEN.get(k)
    if isinstance(wrec, dict):
        wrec["stamp"] = _file_stamp(path)
    rrec = _READ.get(k)
    if isinstance(rrec, dict):
        rrec["stamp"] = None


def unread_rewrite_refusal(path: Path, filename: str, **_ignored) -> str:
    """"" when a write over `path` is allowed, else the refusal TEXT (the
    caller wraps it in `ToolOutcome.rejected("REJECTED: …")`). A first PART
    over an existing unread file is as destructive as a plain write (the
    later parts pass: the file is then in progress / written this request).
    §4OG-1 (h): allowed when this request read ≥ READ_COVERAGE_FLOOR of the
    file (or wrote it whole) AND the file has not changed since."""
    if not isinstance(path, Path):
        return ""                                   # only a real path is judged (a mocked sandbox is not a file system)
    if not path.exists() or not path.is_file():
        return ""
    k = _progress_key(path)
    if k in _IN_PROGRESS:
        return ""
    now = _file_stamp(path)
    size = now[1] if now else 0
    tail = ("then change it with operation='edit' (old_string/new_string); write the whole file only "
            "after reading it whole — or, for a file too big to read whole, ranged reads "
            f"(start_line/end_line) or read_chunked pages covering at least {int(READ_COVERAGE_FLOOR * 100)}% of it. "
            "Nothing was changed.")
    rec = _READ.get(k)
    wrec = _WRITTEN.get(k)
    if isinstance(wrec, dict) and wrec.get("stamp") == now:
        return ""                                   # this request wrote it whole and nothing changed since
    if isinstance(rec, dict) and float(rec.get("cov") or 0.0) >= READ_COVERAGE_FLOOR:
        if rec.get("stamp") == now:
            return ""
        return (f"'{filename}' changed since this request read it ({size:,} bytes now) — a whole-file "
                f"write from that earlier view would discard the later changes (your own edits included). "
                f"Read it again (operation='read'), " + tail)
    if isinstance(wrec, dict):
        return (f"'{filename}' changed since this request wrote it ({size:,} bytes now) — a whole-file "
                f"write would discard the later changes (your own edits included). Read it again "
                f"(operation='read'), " + tail)
    if isinstance(rec, dict):
        pct = int(round(100 * float(rec.get("cov") or 0.0)))
        return (f"'{filename}' already exists ({size:,} bytes) and this request has read only about {pct}% of "
                f"it — a whole-file write would discard everything you did not re-emit. Read it whole "
                f"(operation='read'), " + tail)
    return (f"'{filename}' already exists ({size:,} bytes) and this request has not read it — a "
            f"whole-file write would discard everything you did not re-emit. Read it whole "
            f"(operation='read'), " + tail)


def _mark_written(path: Path) -> None:
    k = _progress_key(path)
    if len(_WRITTEN) >= _IN_PROGRESS_MAX and k not in _WRITTEN:
        for old in list(_WRITTEN)[: _IN_PROGRESS_MAX // 4]:
            _WRITTEN.pop(old, None)
    _WRITTEN[k] = {"stamp": _file_stamp(path)}


def _written_this_request(path: Path) -> bool:
    return _progress_key(path) in _WRITTEN


def _cut_saved_note(raw_len: int, path: Path, shown_path: str = "") -> str:
    """The note on a part SAVED from a write the output cap cut (§4NU r4)."""
    try:
        # r5: the END of the file (a story paragraph is one long line — the
        # first 200 chars of the last lines sent the model back mid-paragraph)
        shown = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")[-400:]
    except Exception:  # noqa: BLE001
        shown = ""
    return (f"\n⚠ Your write was CUT OFF by the output limit after {raw_len:,} characters — the text up to its "
            f"last complete line was SAVED as this part, and the file is IN PROGRESS. It now ends with:\n"
            f"<<<\n{shown}\n>>>\nContinue from exactly there with append=true — your next call is "
            f"{{\"operation\": \"write\", \"path\": \"{shown_path or path.name}\", \"append\": true, \"more_parts\": true, "
            f"\"content\": \"<only the text that comes after the lines above, ~15 KB>\"}} — in SMALLER parts; leave "
            f"more_parts off on the last part. Do not start the file over: what is saved stays.")


#: §4NU: the note on a part written with more_parts=true whose file does not
#: parse yet (instead of a syntax-failure verdict — part 1 of a .py/.json is
#: never complete)
_PARTS_PENDING_NOTE = " More parts follow (more_parts=true), so its syntax verdict waits for the last part."
#: §4NU: an append rewrites the file; beyond this the file is too big for it
APPEND_MAX_FILE_BYTES = 64 * 1024 * 1024
#: §4NU r1: a part this long that the file already ENDS with is a re-send
APPEND_DUP_MIN_CHARS = 200


def _append_instead_hint(file_content: str, old: str, new: str) -> str:
    """§4NU r6 (probe-4nu-story2): the model grew a story with `edit` —
    old_string = the closing line, new_string = that line + the next chapter
    — and story text repeats lines ("The end."), so the anchor was never
    unique; it detoured into a second file and then a script. An edit that
    only ADDS after text that ENDS the file is an append."""
    try:
        if (new.startswith(old) and len(new) > len(old)
                and file_content.rstrip().endswith(old.rstrip())):
            return (" This edit only ADDS text after the file's last lines — use "
                    "operation='write' with append=true and content = just the new text "
                    "(no anchor needed).")
    except Exception:  # noqa: BLE001
        pass
    return ""


async def tool_append_file(filename: str, content: Any, sandbox_dir: Path, more_parts: bool = False):
    """§4NU: `write` with append=true — add `content` to the END of an
    existing file, so a file larger than one reply's output (~16k tokens,
    7140be61) is written in parts. Same guards as `write` (the operation is
    still 'write'); the whole new file is written tmp-then-rename. A part
    the file already ends with is refused (a re-send after a cut-off would
    otherwise land twice). With more_parts=true a syntax error is noted,
    not booked; the LAST part (no more_parts) is checked like a write."""
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2)
    _c = "" if content is None else (content if isinstance(content, str) else str(content))
    pretty_log("File Append", f"{filename} (+{len(_c)} chars)", icon=Icons.TOOL_FILE_W)
    try:
        if content is None or _c == "" or _c.strip() in ("None", _HELD_CONTENT_TOKEN):
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: append=true needs the next part's actual text in 'content'. "
                f"'{filename}' is unchanged.", reason_code="missing_content")
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists():
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: '{filename}' does not exist yet — write the FIRST part "
                f"with operation='write' (no append), then append the rest.",
                reason_code="append_target_missing")
        if not path.is_file():
            return (f"Error: '{filename}' exists and is not a regular file "
                    f"(it is a FIFO, device, socket or directory), so it "
                    f"cannot be appended to. '{filename}' is unchanged.")
        try:
            _part = _c.encode("utf-8")
        except UnicodeEncodeError as ue:
            return (f"Error: the part for '{filename}' contains characters that "
                    f"cannot be encoded as UTF-8 (lone surrogate "
                    f"{_c[ue.start:ue.end]!r} at offset {ue.start}). "
                    f"'{filename}' was NOT modified.")
        if path.stat().st_size + len(_part) > APPEND_MAX_FILE_BYTES:
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: '{filename}' would exceed {APPEND_MAX_FILE_BYTES // (1024 * 1024)} MB — "
                f"too big to grow by appends; build it with a script (execute). Unchanged.",
                reason_code="append_too_big")
        _state = {}

        def _atomic_append():
            _swap0 = _parent_swapped(sandbox_dir, path)      # §4PF r4 (F15): judged IN the thread, before mkstemp
            if _swap0:
                return _swap0
            import tempfile
            _old = path.read_bytes()
            _new = _part
            if len(_c.strip()) >= APPEND_DUP_MIN_CHARS and _old.rstrip().endswith(_new.rstrip()):
                _state["dup"] = True
                return len(_old)
            # `write` strips the trailing newline of a code file, so part 1
            # ends without one: put the line break back at the boundary
            # (parts are split at line boundaries — the guidance says so), in
            # the file's own line-ending style
            if _old and not _old.endswith(b"\n") and not _new.startswith((b"\n", b"\r\n")):
                _new = (b"\r\n" if b"\r\n" in _old[-4096:] else b"\n") + _new
                _state["nl"] = True
            fd, _tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(_old)
                    f.write(_new)
                try:
                    _mode = path.stat().st_mode & 0o7777
                except OSError:
                    _mode = 0o644
                os.chmod(_tmp, _mode)
                _swap1 = _parent_swapped(sandbox_dir, path)   # …and again right before the replace
                if _swap1:
                    os.unlink(_tmp)
                    return _swap1
                _commit_replace(_tmp, path)                     # §4SD: the commit point — nothing after it re-runs
            except BaseException:
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
                raise
            return len(_old) + len(_new)
        def _append_and_note():
            r = _retry_transient(_atomic_append)                # §4SD: re-runs only before the commit
            if isinstance(r, int) and not _state.get("dup"):
                _note_mutation(path)        # §4OG-1 (h): a direct append (the §4OB continuation) keeps the file this request's own
            return r
        total = await asyncio.to_thread(_append_and_note)
        if isinstance(total, str):                                # §4PF r4 (F15): the dir was swapped
            return total
        if _state.get("dup"):
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: '{filename}' already ENDS with this exact part — it was most likely "
                f"appended before, so it was NOT appended again. If the file really needs this text "
                f"twice, append it together with the lines that follow it.",
                reason_code="duplicate_append")
        try:
            rel_str = str(path.resolve().relative_to(sandbox_dir.resolve()))
        except Exception:
            rel_str = str(filename)
        syntax_note = await _syntax_feedback(path, filename, in_progress=bool(more_parts),
                                             last_rows=sum(1 for ln in str(_c).splitlines() if ln.strip()) or None)
        _nl = " A line break was added before it (the file did not end with one)." if _state.get("nl") else ""
        _text = (f"SUCCESS: Appended {len(_c)} chars to '{filename}'. Script-side path "
                 f"(from sandbox cwd): '{rel_str}'. The file is now {total:,} bytes.{_nl}")
        _handler = ""
        if not more_parts and str(filename).lower().endswith((".html", ".htm")):
            try:                                                     # §4OI (iii): the finished page, checked whole
                _handler = await asyncio.to_thread(
                    html_page_notes, path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        if syntax_note and more_parts:
            return _text + _PARTS_PENDING_NOTE
        if syntax_note:
            return ToolOutcome.partial(_text + syntax_note + _handler, world_changed=True,   # r1 m4: syntax first
                                       reason_code="written_but_syntax_failed")
        return _text + _handler
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error: {e}"


# Hard cap on entries in one listing. Beyond it the tail is COUNTED and the
# message says how to see it (list a subdirectory) — the old silent [:200]
# slice made big workspaces look like their subdirectories were empty, which
# sent the model to `execute`+ls instead of file_system (2026-07-14 log).
_LIST_MAX_ENTRIES = 200


async def tool_list_files(sandbox_dir: Path, memory_system=None, path: str = None):
    """List the workspace tree (with Python AST signatures). ``path`` scopes
    the listing to one subdirectory — previously the dispatcher dropped it,
    so 'list analysis/src' silently re-listed the whole (truncated) root."""
    _want_sub = path is not None and str(path).strip() not in ("", ".", "./")
    pretty_log("Sandbox Tree",
               "Listing workspace files & mapping repo"
               + (f" [{path}]" if _want_sub else ""),
               icon=Icons.TOOL_FILE_I)
    try:
        base = Path(sandbox_dir)
        display_prefix = ""
        if _want_sub:
            try:
                base = _get_safe_path(sandbox_dir, path)
            except ValueError as ve:
                return str(ve)
            if not base.exists():
                fb = _scoped_root_fallback(sandbox_dir, path)
                if fb is None:
                    return _missing_file_message(path, sandbox_dir)
                base = fb
            if base.is_file():
                return (f"Note: '{path}' is a FILE, not a directory. Use "
                        f"file_system(operation='read', path='{path}') to read it.")
            # Prefix every entry with a path the model can feed straight back
            # into file tools: sandbox-relative when the dir is inside the
            # active workspace, container-absolute (/workspace/...) when it
            # lives at the outer root of a project-scoped sandbox.
            try:
                rel = base.resolve().relative_to(Path(sandbox_dir).resolve())
                display_prefix = "" if str(rel) in ("", ".") else rel.as_posix() + "/"
            except ValueError:
                display_prefix = _to_container_path(sandbox_dir, base).rstrip("/") + "/"

        def _build_map():
            import ast
            import os
            tree_lines = []
            total = 0

            for root, dirs, files in os.walk(base):
                # Ignore hidden and virtual env directories; sort so the
                # listing is deterministic across calls (os.walk's raw dir
                # order is filesystem-dependent, which made truncated
                # listings show a DIFFERENT subset each time).
                dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env'])

                rel_root = Path(root).relative_to(base)
                root_prefix = display_prefix if rel_root == Path(".") else f"{display_prefix}{rel_root.as_posix()}/"

                for f in sorted(files):
                    if f.startswith('.'): continue
                    total += 1
                    if len(tree_lines) >= _LIST_MAX_ENTRIES:
                        # Past the cap: keep COUNTING (for the truncation
                        # report) but skip the expensive AST work.
                        continue
                    path_ = Path(root) / f
                    line = f"  {root_prefix}{f}"

                    # --- REPO MAP: Extract AST Signatures for Python files ---
                    # §4GI: a symlinked .py is listed by name only — reading
                    # it would parse (and print the signatures of) whatever
                    # host file the sandbox pointed it at.
                    if f.endswith('.py') and not path_.is_symlink():
                        try:
                            # stat() must stay inside the try: a broken
                            # symlink or a file deleted between os.walk and
                            # here raises OSError, which would otherwise
                            # abort the ENTIRE listing.
                            if path_.stat().st_size < 100000:
                                # §4GJ round 3: O_NOFOLLOW rather than the
                                # `is_symlink()` pre-check §4GI added — the
                                # check and the read were two syscalls, so a
                                # swap between them still read the target.
                                code = read_text_nofollow(path_, errors='ignore')
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

            return tree_lines, total

        tree_lines, total = await asyncio.to_thread(_build_map)
        if tree_lines:
            # §4FD: the model counted 34 listed entries as "45 files" — 19
            # real turns contradicted their own tool output by counting.
            # State the count; never make the model count a list.
            sandbox_tree = (f"{total} entries under "
                            f"{('/' + str(path).strip('/')) if _want_sub else 'the workspace root'}"
                            f" (files, excluding dotfiles):\n" + "\n".join(tree_lines))
            if total > len(tree_lines):
                sandbox_tree += (
                    f"\n  ... [{total - len(tree_lines)} more files NOT shown "
                    f"({total} total). This listing is capped at "
                    f"{_LIST_MAX_ENTRIES} entries — list ONE subdirectory to "
                    f"see the rest: file_system(operation='list_files', "
                    f"path='<subdir>')]")
        else:
            sandbox_tree = "[Empty]"
            if not _want_sub:
                # The scoped workspace itself is empty — say so, but point at
                # root-level files (e.g. an execute-side git clone) instead of
                # letting the model conclude its files vanished.
                sandbox_tree += _outer_root_files_hint(sandbox_dir)

        return f"CURRENT SANDBOX DIRECTORY STRUCTURE:\n{sandbox_tree}\n\n(Use these filenames for all file tools)"
    except Exception as e: return f"Error scanning sandbox: {e}"

# Max HTTP redirect hops the downloader will follow. Each hop's Location is
# re-validated against the SSRF guard, so a public page 302-redirecting to
# 127.0.0.1 / 169.254.169.254 / a LAN host is blocked — the original-URL-only
# check missed this because both clients auto-follow redirects.
_MAX_DOWNLOAD_REDIRECTS = 5


def _download_redirect_target(status, headers, cur_url, ssrf_reason):
    """Redirect-hop decision for the downloader.

    Returns ``(target_url, error)``:
      * ``(url, None)`` — this is a redirect to a SAFE host; follow it.
      * ``(None, "Error: …")`` — redirect to an SSRF host; block.
      * ``(None, None)`` — not a redirect; this is the final response.
    """
    if status in (301, 302, 303, 307, 308):
        loc = headers.get("location") or headers.get("Location")
        if loc:
            target = urllib.parse.urljoin(str(cur_url), str(loc))
            from .quarantine import is_darkweb_url as _is_dark
            if _is_dark(target):
                # §4NL r1: a clearnet URL that 302s onto a hidden service would
                # land the file in the WORKSPACE — the quarantine's mirror rule.
                # r2: judged BEFORE the SSRF check, whose resolve=True mode
                # would send the onion name to the host resolver.
                return None, ("Error: download redirect blocked — the clearnet URL redirected onto the dark "
                              "web; a dark-web file goes to the quarantine: call download with the .onion URL directly.")
            reason = ssrf_reason(target)
            if reason:
                return None, f"Error: download redirect blocked (SSRF): {reason}"
            return target, None
    return None, None


#: A plain-curl identity for the retry a refused Tor request gets. Cloudflare
#: answers a Chrome-impersonating client from a Tor exit with a challenge
#: page (HTTP 403, text/html) while a plain curl over the SAME circuit gets
#: the file — measured three for three on 2026-09-09. The retry stays on
#: Tor: the agent's egress is Tor-only, and no message may suggest otherwise.
_PLAIN_PROFILE_UA = "curl/8.7.1"


def _server_name(resp) -> str:
    """The `Server:` header, bounded and printable — remote-controlled text
    that lands in a message the model reads (review, 2026-09-09)."""
    try:
        raw = str(resp.headers.get("server") or "").strip()
    except Exception:  # noqa: BLE001
        raw = ""
    raw = "".join(ch for ch in raw if ch.isprintable())[:40]
    return raw or "the server"


class _LocalSaveError(Exception):
    """§4PE (F5): the download body could not be SAVED (the target is a
    directory, the disk is full, the name is unwritable) — a local fault
    the caller reports as such, never as the site refusing Tor."""


_DOWNLOAD_MAX_BYTES = 50_000_000
_BINARY_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".ico", ".pdf", ".zip", ".gz",
    ".tgz", ".tar", ".7z", ".rar", ".mp4", ".mp3", ".wav", ".ogg", ".webm", ".mov", ".m4a", ".bin",
    ".exe", ".dmg", ".iso", ".whl", ".so", ".dylib", ".woff", ".woff2", ".ttf", ".epub", ".docx", ".xlsx"})


async def _save_download_stream(target_path: Path, chunks, _max: int = _DOWNLOAD_MAX_BYTES):
    """§4PE (F4/F5): the body goes to a temp file beside the target and is
    moved over it only when ALL of it has arrived — a symlink planted at the
    target during the transfer is replaced as an entry, never followed
    (`open(target, "wb")` followed it: URL-chosen bytes into any host file);
    a connection cut mid-stream leaves no partial file ("NOT COPIED MUST MEAN
    NOT PRESENT"); a LOCAL error is a `_LocalSaveError`. Returns
    ``(bytes, overflow, first 512 bytes)``."""
    import tempfile
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=str(target_path.parent))
    except OSError as oe:
        raise _LocalSaveError(f"{oe.strerror or oe} — '{target_path.name}'") from oe
    total, overflow, head = 0, False, b""
    try:
        with os.fdopen(fd, "wb") as f:
            buffer = bytearray()
            async for chunk in chunks:
                if not chunk:
                    continue
                buffer.extend(chunk)
                total += len(chunk)
                if len(head) < 512:
                    head += bytes(chunk)[: 512 - len(head)]
                if total > _max:
                    overflow = True
                    break
                if len(buffer) >= 1024 * 1024:
                    try:
                        await asyncio.to_thread(f.write, buffer)
                    except OSError as oe:
                        raise _LocalSaveError(f"{oe.strerror or oe} while writing '{target_path.name}'") from oe
                    buffer.clear()
            if buffer and not overflow:
                try:
                    await asyncio.to_thread(f.write, buffer)
                except OSError as oe:
                    raise _LocalSaveError(f"{oe.strerror or oe} while writing '{target_path.name}'") from oe
        if not overflow:
            try:
                os.replace(tmp, target_path)
            except OSError as oe:
                raise _LocalSaveError(f"{oe.strerror or oe} — '{target_path.name}'") from oe
            tmp = None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return total, overflow, head


def _download_receipt(url, filename, target_path: Path, total: int, ctype, head: bytes) -> str:
    """§4PE (F6): what landed — size and type — and a warning when a 200
    carried an HTML page under a binary name (the §4LL "saved the error page
    as the photo" incident). The write is stamped as this request's, so a
    retry under the same name is not refused as an unread rewrite."""
    try:
        _mark_written(target_path)
    except Exception:  # noqa: BLE001
        pass
    ct = str(ctype or "").split(";")[0].strip().lower() or "unknown type"
    suffix = Path(str(filename)).suffix.lower()
    body = f"Downloaded '{url}' to '{filename}' ({total:,} bytes, {ct})."
    # §4PE r3: a suspect body is PARTIAL with a "PARTIAL:" head — the first cut
    # said "treat as FAILED" under a SUCCESS head, which the verifier booked
    # as delivered
    if total == 0:
        return ToolOutcome.partial(
            f"PARTIAL: {body} ⚠ The file is EMPTY (0 bytes): the server sent no body — do not use it; "
            f"try another source or tell the user.", world_changed=True, reason_code="download_empty")
    if suffix in _BINARY_SUFFIXES and (re.search(rb"<!doctype\s+html|<html[\s>]", head[:512], re.I)
                                       or ct == "text/html"):
        return ToolOutcome.partial(
            f"PARTIAL: {body} ⚠ The body is an HTML page, not a {suffix} file — the site answered 200 "
            f"with a page (a challenge, login or error page). Do not use '{filename}' as the {suffix}; "
            f"try another source, and if none works, tell the user.", world_changed=True,
            reason_code="download_html_body")
    return f"SUCCESS: Downloaded '{url}' to '{filename}' ({total:,} bytes, {ct})."


async def tool_download_file(url: str, sandbox_dir: Path, tor_proxy: str, filename: str = None):
    # §4NL r2: the refusal lives HERE, not only in the dispatcher — a direct
    # caller (subject_photos, a future tool) must not put an onion file in the
    # workspace either
    from .quarantine import is_darkweb_url as _is_dark
    if _is_dark(str(url)):
        return ("Error: a dark-web (.onion) file never goes to the workspace — use "
                "file_system(operation='download', url=<the .onion URL>), which stores it in the quarantine.")
    # --- SSRF guard (shared) ---
    # Block file:// and internal/metadata hosts BEFORE any fetch, so the
    # LLM can't make the host read local files or reach 169.254.169.254 /
    # loopback services. file:// is reachable regardless of the proxy. Every
    # REDIRECT hop is re-validated below (both HTTP clients auto-followed
    # redirects, so a 302 → internal host used to bypass this original-URL
    # check on the non-Tor WEB path; under Tor the SOCKS proxy already can't
    # reach link-local/metadata).
    from ..utils.helpers import url_ssrf_reason as _url_ssrf_reason
    from ..utils.helpers import aclose_curl_response as _aclose_curl
    from ..utils.egress_guard import resolve_egress_proxy as _resolve_egress_proxy
    # 1. Clean Proxy URL. Fail-closed (§4P): under --mandatory-tor a falsy proxy
    # is replaced with the loopback Tor default so a public download never
    # connects cleartext; the socket guard can't backstop this curl_cffi
    # call. Outside mandatory-tor the proxy is unchanged (direct WEB mode).
    proxy_url = _resolve_egress_proxy(tor_proxy, url)
    # §4LM: the guard runs AFTER the proxy is known, with resolve=False over
    # Tor — the default host-side getaddrinfo sent the target's name to the
    # local resolver in cleartext before the socks5h fetch (the leak the
    # web fetch already avoids). Literal internal IPs are still refused.
    _ssrf_resolve = not bool(proxy_url)

    def _ssrf_check(u):
        return _url_ssrf_reason(u, resolve=_ssrf_resolve)
    _ssrf = _ssrf_check(url)
    if _ssrf:
        return f"Error: {_ssrf}"
    mode = "TOR" if proxy_url and "127.0.0.1" in proxy_url else "WEB"

    # Full URL — the [:35] pre-truncation also starved the durable mirror (it
    # saw 35 chars too). Let the stream truncate; the durable log keeps it whole.
    pretty_log(f"Download [{mode}]", url, icon=Icons.TOOL_DOWN)
    
    if proxy_url and proxy_url.startswith("socks5://"): 
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    # §4OE-2: a download over an EXISTING file this request has not read is a
    # whole-file replacement — refused BEFORE any network cost (review 3:
    # inside the hop loop it would have refused after a Tor rotation)
    if filename:
        try:
            _unread = unread_rewrite_refusal(_get_safe_path(sandbox_dir, filename), filename)
        except Exception:  # noqa: BLE001 — the per-hop safe-path check below answers escapes
            _unread = ""
        if _unread:
            return ToolOutcome.rejected("REJECTED: " + _unread, reason_code="unread_rewrite")
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    _plain = False           # switched on by the first Tor refusal (see _PLAIN_PROFILE_UA)
    for attempt in range(3):
        try:
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                try:
                    target_path = _get_safe_path(sandbox_dir, filename)
                except ValueError as ve: return str(ve)
                
                async with curl_requests.AsyncSession(
                        impersonate=(None if _plain else "chrome110"),
                        headers=({"User-Agent": _PLAIN_PROFILE_UA} if _plain else None),
                        proxies=proxies, timeout=60.0) as client:
                    # Manual redirect follow (auto-redirect OFF) so each hop's
                    # Location is SSRF-validated before we fetch it.
                    cur_url = url
                    resp = None
                    for _hop in range(_MAX_DOWNLOAD_REDIRECTS + 1):
                        resp = await client.get(cur_url, stream=True, allow_redirects=False)
                        _next, _rerr = _download_redirect_target(
                            resp.status_code, resp.headers, cur_url, _ssrf_check)
                        # AsyncSession streaming responses need the ASYNC
                        # close (quit_now + aclose): the sync close() reaps
                        # stream_task, which async responses never set, so it
                        # silently left every hop's transfer running.
                        if _rerr:
                            await _aclose_curl(resp)
                            return _rerr
                        if _next is None:
                            break  # final (non-redirect) response
                        await _aclose_curl(resp)
                        cur_url = _next
                    else:
                        return "Error: too many redirects (possible redirect loop)."
                    if resp.status_code != 200:
                        # Abort the transfer BEFORE the identity-renew sleep:
                        # an open streaming response pumps its body into the
                        # unbounded response queue in RAM the whole 5s.
                        await _aclose_curl(resp)
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            # ⚠ RECORD THE REFUSAL. This path re-tried with a fresh Tor
                            # identity and left `last_error` at None, so three Cloudflare
                            # 403s reported as "Failed after 3 attempts. Last error: None"
                            # — the model retried the same tool blind (§4FR, 2026-09-09).
                            last_error = (f"HTTP {resp.status_code} from {_server_name(resp)} "
                                          f"({'plain' if _plain else 'chrome110'} profile)")
                            if not _plain:
                                # The site refused the FINGERPRINT, not the exit: the
                                # next attempt uses a plain curl identity on the same
                                # circuit, without spending 5 s on a new one (review,
                                # 2026-09-09). `continue` — this branch sits AFTER the
                                # hop loop, inside the attempt loop.
                                _plain = True
                                continue
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        return _download_error(resp.status_code, url)

                    clength = resp.headers.get("Content-Length")
                    if clength and int(clength) > 50000000:
                        await _aclose_curl(resp)
                        return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."

                    _nr = _nonregular_refusal(target_path, filename)    # §4PE (F3): a FIFO target wedged the loop
                    if _nr:
                        await _aclose_curl(resp)
                        return _nr
                    try:
                        _total, _overflow, _head = await _save_download_stream(target_path, resp.aiter_content())
                    finally:
                        # On the overflow break the transfer is still live —
                        # abort it instead of letting it stream into the
                        # queue until session teardown.
                        await _aclose_curl(resp)
                    if _overflow:
                        return ("Error: download exceeded the 50MB cap (server omitted or exceeded "
                                "Content-Length). Nothing was saved.")
                    return _download_receipt(url, filename, target_path, _total, resp.headers.get("Content-Type"), _head)
            else:
                # follow_redirects=False + manual hop loop so each Location is
                # SSRF-validated before the fetch (see _download_redirect_target).
                async with httpx.AsyncClient(proxy=proxy_url, headers=headers, follow_redirects=False, timeout=60.0) as client:
                    cur_url = url
                    for _hop in range(_MAX_DOWNLOAD_REDIRECTS + 1):
                        async with client.stream("GET", cur_url) as resp:
                            _next, _rerr = _download_redirect_target(
                                resp.status_code, resp.headers, cur_url, _ssrf_check)
                            if _rerr:
                                return _rerr
                            if _next is not None:
                                cur_url = _next
                                continue  # closes this stream, re-requests the validated hop
                            if resp.status_code != 200:
                                if resp.status_code in [401, 403, 503] and mode == "TOR":
                                    # ⚠ RECORD THE REFUSAL. This path re-tried with a fresh Tor
                                    # identity and left `last_error` at None, so three Cloudflare
                                    # 403s reported as "Failed after 3 attempts. Last error: None"
                                    # — the model retried the same tool blind (§4FR, 2026-09-09).
                                    last_error = (f"HTTP {resp.status_code} from {_server_name(resp)}")
                                    # Rotation happens ONCE below the hop
                                    # loop — rotating here too doubled the
                                    # identity churn + 10s of sleep per retry.
                                    break  # break the hop loop → outer attempt retry
                                return _download_error(resp.status_code, url)

                            clength = resp.headers.get("Content-Length")
                            if clength and int(clength) > 50000000:
                                return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."

                            try:
                                target_path = _get_safe_path(sandbox_dir, filename)
                            except ValueError as ve: return str(ve)

                            _nr = _nonregular_refusal(target_path, filename)    # §4PE (F3)
                            if _nr:
                                return _nr
                            _total, _overflow, _head = await _save_download_stream(target_path, resp.aiter_bytes())
                            if _overflow:
                                return ("Error: download exceeded the 50MB cap (server omitted or exceeded "
                                        "Content-Length). Nothing was saved.")
                            return _download_receipt(url, filename, target_path, _total,
                                                     resp.headers.get("Content-Type"), _head)
                    else:
                        return "Error: too many redirects (possible redirect loop)."
                    # reached only via the 401/403/503 `break` → fall to the
                    # outer `for attempt` Tor-identity retry.
                    if mode == "TOR":
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
        except _LocalSaveError as le:
            # §4PE (F5): "[Errno 21] Is a directory" was reported, after three
            # Tor identity rotations, as the site refusing Tor exits
            return (f"Error: could not save '{filename}': {le}. Nothing was saved. This is a local "
                    f"path problem, not the site — fix the destination name and call again.")
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            
    if mode == "TOR":
        # The host's egress is Tor, and some sites answer a browser-shaped
        # Tor client with a challenge page (Cloudflare: HTTP 403, text/html)
        # while a plain curl from the sandbox — which has its own direct
        # egress by design, see sandbox/docker.py — gets the file. Say so,
        # and name that route: the model's only other option is to retry
        # this one, which it did twice on 2026-09-09 (request probe, §4FR).
        # §4FS review: a first version recommended `curl` from the sandbox.
        # The sandbox's egress is cleartext from the host IP (docker.py sets
        # no proxy; under host networking its Tor cannot bind), and the
        # agent's rule is Tor-only — a tool message must not steer the model
        # through that hole. The plain-profile retry above is the honest
        # remedy; past it, the file has to come from the user.
        return (
            f"Error: Failed after 3 attempts over Tor — {last_error or 'no usable response'}. "
            "Tried a browser profile and a plain curl profile over Tor; the site refuses this "
            "agent from Tor exits, and retrying this tool will not change that. The agent's "
            "egress is Tor-only by policy: ask the user to provide the file or a mirror that "
            "accepts Tor. Do NOT fetch it any other way."
        )
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
        # IDENTICAL-ARGS CORRUPTION GUARD (2026-07-17) — sibling of the
        # replace content==replace_with guard: upstream native tool-call
        # transport can duplicate one argument's value into the next slot.
        # Observed live (req AF): filename arrived == the search pattern, so
        # rg ran against '<workspace>/makeDraggable' — a path that cannot
        # exist — and the turn was wasted on "No such file". A file named
        # exactly like the pattern being searched IN it is never intentional;
        # heal by dropping the corrupt filename and searching the workspace.
        _healed_dup_filename = False
        if filename and str(filename).strip() == str(pattern).strip("'\""):
            pretty_log(
                "Search Guard Tripped",
                f"filename == search pattern ({str(pattern)[:60]!r}) — "
                "suspected upstream argument duplication; searching the "
                "whole workspace instead",
                level="WARNING", icon=Icons.WARN,
            )
            filename = None
            _healed_dup_filename = True
        if filename:
            search_path = _get_safe_path(sandbox_dir, filename)
            if not search_path.exists():
                # Read-only root fallback (see _scoped_root_fallback): the
                # target may live at the sandbox ROOT, outside the project.
                fb = _scoped_root_fallback(sandbox_dir, filename)
                if fb is not None:
                    search_path = fb
            container_path = _to_container_path(sandbox_dir, search_path)
            escaped_path = shlex.quote(container_path)
        else:
            # Scope the default search to the ACTIVE workspace. The rg
            # command runs at the container root (/workspace), so a bare "."
            # searched the WHOLE sandbox root even when a project was active
            # — matches from other projects came back with paths the scoped
            # read op couldn't resolve. "/workspace" when unscoped (same
            # tree as "." was), "/workspace/projects/<id>" when scoped.
            escaped_path = shlex.quote(
                _to_container_path(sandbox_dir, Path(sandbox_dir)))

        pattern = str(pattern).strip("'\"") # Strip accidental quotes
        escaped_pattern = shlex.quote(pattern)

        pretty_log("File Search", f"'{pattern}' in {filename or 'workspace'}", icon=Icons.TOOL_FILE_S)

        # -e: a pattern starting with '-' must not parse as an rg flag.
        # §4LL: hidden and git-ignored files are searched too (a string in a
        # gitignored dist/ or in .env was "No matches … the search did not
        # fail"), never inside .git itself
        # hidden files always (.env); git-ignored files only when the call
        # names a path — a whole-workspace search through node_modules/.venv
        # flooded the head+tail cap and dropped the project's own hits (review)
        _rg = ("rg --line-number --no-heading --color=never --max-columns=300 "
               "--hidden --glob '!.git' " + ("--no-ignore-vcs " if filename else ""))
        cmd = f"{_rg}-e {escaped_pattern} {escaped_path}"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, timeout=20)
        _as_text_note = ""
        if exit_code == 2 and "regex parse error" in str(output or ""):
            # the schema says "exact text": `print(` is a broken regex, not a
            # failed search — search it as plain text (§4LL review)
            output, exit_code = await asyncio.to_thread(
                sandbox_manager.execute, f"{_rg}-F -e {escaped_pattern} {escaped_path}", timeout=20)
            _as_text_note = (f" [searched as plain text: '{pattern}' is not a valid regular "
                             f"expression]")
        if exit_code == 2:
            # rg exits 2 when ANY file errored, even with matches printed —
            # keep the matches (review); fail only when nothing was found
            _hits = [ln for ln in str(output or "").splitlines() if re.match(r"^[^:\n]+:\d+:", ln)]
            if not _hits:
                return ToolOutcome.failed(
                    f"Error: the search failed (rg exit 2): {str(output or '').strip()[:400]}",
                    reason_code="search_error")
            output = "\n".join(_hits) + "\n[note: some files could not be read and were skipped]"
            exit_code = 0
        if _as_text_note and exit_code == 0 and isinstance(output, str) and output.strip():
            output = output.rstrip() + "\n" + _as_text_note.strip()

        # rg exits 1 to mean "no matches" — a successful query with an empty
        # result, not a failure. The docker layer substitutes "[SYSTEM
        # ERROR]: Process failed (Exit 1) with no output." for empty output
        # before we see it, so the friendly no-matches fallback below never
        # fired and the model read its own successful verification as a tool
        # failure (observed live 2026-07-14: three identical searches burned
        # verification turns because "artifacts gone" was indistinguishable
        # from "search broken"). Mirrors execute.py's grep-exit-1
        # normalization. rg signals real errors with exit 2 + stderr, which
        # still passes through untouched.
        _out_stripped = (output or "").strip() if isinstance(output, str) else ""
        if exit_code == 1 and (
                not _out_stripped
                or _out_stripped == "[SYSTEM ERROR]: Process failed (Exit 1) with no output."):
            _dup_note = (
                " [NOTE: your 'filename' arrived identical to the pattern — "
                "suspected upstream argument duplication; the whole "
                "workspace was searched.]" if _healed_dup_filename else "")
            return (f"Report: No matches found for '{pattern}'. (rg exited 1, "
                    f"which for a search means the pattern was NOT FOUND — "
                    f"the search itself did not fail.){_dup_note}{_as_text_note}")

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

        if _healed_dup_filename and isinstance(output, str) and output.strip():
            output = (
                "[NOTE: your 'filename' argument arrived IDENTICAL to the "
                "search pattern — suspected upstream argument duplication; "
                "the whole workspace was searched instead. Pass the real "
                "filename if you meant one file.]\n" + output
            )
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
        if sandbox_dir is not None:
            if not path or str(path).strip() in ("", "."):
                # Scope the default to the ACTIVE workspace: find runs at the
                # container root (/workspace), so a literal "." swept the
                # whole sandbox root even when a project was active.
                search_path = _to_container_path(sandbox_dir, Path(sandbox_dir))
            else:
                try:
                    resolved = _get_safe_path(sandbox_dir, path)
                    if not resolved.exists():
                        # Read-only root fallback — the dir may live at the
                        # sandbox ROOT (execute-created), outside the project.
                        fb = _scoped_root_fallback(sandbox_dir, path)
                        if fb is not None:
                            resolved = fb
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
        if not output.strip():
            return "Report: No files found matching that pattern."
        if len(output.strip().splitlines()) >= 100:       # §4PE: 100 names read as "all"
            output = output.rstrip("\n") + "\n[Note: the listing is capped at 100 names — there may be more; narrow the pattern or the path.]"
        return output
    except Exception as e: return f"Error: {e}"

_BINARY_MAGIC = (
    (b"\xff\xd8\xff", "JPEG image"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"GIF87a", "GIF image"), (b"GIF89a", "GIF image"),
    (b"RIFF", "RIFF container (WebP/WAV/AVI)"),
    (b"%PDF", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (or docx/xlsx/jar)"),
    (b"\x1f\x8b", "gzip archive"),
    (b"SQLite format 3", "SQLite database"),
    (b"\x7fELF", "ELF executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
)


def _binary_peek_summary(path: Path, filename: str) -> str:
    """What `inspect` says about a binary file instead of dumping its bytes.

    A factual, non-error line: the file exists, its kind (by magic number)
    and size, and where to go next (vision for images, read_chunked never —
    it is binary). Deliberately does not start with "Error" because the
    strike counter treats that prefix as an execution failure."""
    try:
        head = _read_head(path, 16)
    except OSError:
        head = b""
    kind = next((name for magic, name in _BINARY_MAGIC
                 if head.startswith(magic)), "binary file")
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    size_s = (f"{size:,} bytes" if size >= 0 else "size unknown")
    hint = (" Use 'vision_analysis' to look at it."
            if kind.endswith("image") else
            " It has no text preview; do not read it as text.")
    return (f"'{filename}' exists: {kind}, {size_s} — binary, so no text "
            f"lines to show.{hint}")


async def tool_inspect_file(filename: str, sandbox_dir: Path, lines: int = 10):
    if not filename: return "Error: 'path' (filename) is required for inspection."
    pretty_log("File Peek", filename, icon=Icons.TOOL_FILE_I)
    try:
        # Coerce: the dispatcher forwards the model-supplied value verbatim
        # (may be None, "20", or garbage) — a bad value must not crash the op.
        try:
            lines = max(1, int(lines))
        except (TypeError, ValueError):
            lines = 10
        path = _get_safe_path(sandbox_dir, filename)
        _fb_note = ""
        if not path.exists():
            fb = _scoped_root_fallback(sandbox_dir, filename)
            if fb is None:
                return _missing_file_message(filename, sandbox_dir)
            path = fb
            _fb_note = (f"NOTE: serving the sandbox-ROOT copy "
                        f"'{_to_container_path(sandbox_dir, fb)}' (not in this "
                        f"project's workspace).\n")
        # Binary sniff, as `read` / `read_chunked` already do. Without it a
        # JPEG "peek" returned its first ten newline-delimited runs of
        # JFIF/Exif/ICC bytes as text (req 2422eb25, 2026-09-06). This is NOT
        # an "Error:" reply on purpose: the model peeks images to confirm
        # they exist before vision, and an Error prefix is a strike.
        _nr = _nonregular_refusal(path, filename)       # §4PE (F3): a FIFO wedged the loop here
        if _nr:
            return _nr
        try:
            if _looks_like_binary(_read_head(path, 8192)):
                return _fb_note + _binary_peek_summary(path, filename)
        except OSError as oe:
            return f"Error: failed to read '{filename}': {oe}"
        def _read_peek():
            content = []
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for _ in range(lines):
                    line = f.readline()
                    if not line: break
                    content.append(line.strip())
            return "\n".join(content)

        return _fb_note + await asyncio.to_thread(_read_peek)
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
        # §4GI: never follow a symlink the sandbox planted — nested links
        # are recreated as links (or dropped when they escape the root),
        # never materialised as the host files behind them.
        _skipped = []
        if src_path.is_dir():
            # ⚠ THE MODEL IS TOLD WHAT DID NOT COME ACROSS (§4GK round 4).
            # `skipped` was discarded, so a copy that silently dropped an
            # escaping symlink, a FIFO or an unreadable subtree still answered
            # "SUCCESS: Copied 'tree' to 'copy'." — and the model then worked
            # from a destination it believes is complete. Verified: a source
            # holding one regular file, one escaping link and one FIFO copied
            # one file and reported plain success.
            _skipped = await asyncio.to_thread(
                copytree_nofollow, src_path, dest_path, sandbox_dir) or []
        else:
            # `src_path` already went through `_get_safe_path`, which RESOLVES
            # a link, so `follow_symlinks=False` is unobservable HERE (its
            # mutant is equivalent — §4GI battery); it stays because the AST
            # enumeration holds every copy in this file to the same rule, so a
            # future copy site that does not resolve first cannot follow.
            await asyncio.to_thread(shutil.copy2, str(src_path), str(dest_path),
                                    follow_symlinks=False)
            # copy2 carries BSD flags on macOS: a copy OUT of a released
            # workspace would be immutable too (§4KW fourth review)
            _clear_immutable(dest_path)
        if _skipped:
            # A half-landed write is `ToolOutcome.partial`, never a bare
            # string that merely starts with "PARTIAL:" — the package-wide
            # contract enumeration caught exactly that here (§4GK round 5):
            # a bare string coerces to OK, so the world-changed credit, the
            # idempotency record and the strike decay all read it as a
            # clean copy.
            return ToolOutcome.partial(
                f"Copied '{src_name}' to '{dest_name}', but "
                f"{len(_skipped)} entr(ies) were NOT copied — "
                f"{'; '.join(_skipped[:5])}"
                f"{'…' if len(_skipped) > 5 else ''}. The destination is "
                "incomplete; symlinks that leave the sandbox, device "
                "nodes and unreadable entries are never copied.",
                world_changed=True, reason_code="copy_incomplete")
        return f"SUCCESS: Copied '{src_name}' to '{dest_name}'."
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error: {e}"


async def tool_rename_file(old_name: str, new_name: str, sandbox_dir: Path):
    pretty_log("File Rename", f"{old_name} -> {new_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        _link = _symlink_itself(sandbox_dir, old_name)
        if _link is not None:
            new_path = _get_safe_path(sandbox_dir, new_name, allow_root=False)
            if new_path.exists() or new_path.is_symlink():
                return f"Error: destination '{new_name}' already exists."
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(_link), str(new_path))
            # the shape `_FS_MOVED_RE` parses (the verifier's ground truth)
            return f"SUCCESS: Renamed/Moved '{old_name}' to '{new_name}'. It was a link — its target was not touched."
        old_path = _get_safe_path(sandbox_dir, old_name, allow_root=False)
        # moving a whole project out of projects/ and deleting it there was
        # the same rmtree around manage_projects' confirm (§4MB review)
        if old_path.exists():
            _ref = _protected_dir_refusal(sandbox_dir, old_path, old_name)
            if _ref is not None:
                return _ref
        new_path = _get_safe_path(sandbox_dir, new_name, allow_root=False)
        if not old_path.exists(): return f"Error: '{old_name}' not found."
        if new_path.exists():
            # Refuse a silent clobber (mirrors tool_copy_file). shutil.move
            # also treats an existing-dir destination as "move INTO it" —
            # make that explicit instead of guessing. A case-only rename on
            # a case-insensitive FS (old and new are the SAME file) is
            # still allowed through.
            try:
                _same = new_path.samefile(old_path)
            except OSError:
                _same = False
            if new_path.is_dir() and not _same:
                return (f"Error: destination '{new_name}' is an existing "
                        f"directory. To move INTO it, pass the full "
                        f"destination path, e.g. "
                        f"'{str(new_name).rstrip('/')}/{old_path.name}'.")
            if not _same:
                return (f"Error: destination '{new_name}' already exists. "
                        f"Delete or rename it first if you intended to "
                        f"overwrite.")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        _old_rp = str(old_path.resolve())
        await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
        _move_parts(_old_rp, new_path)             # §4OI r1 m5: the parts state follows the file
        return f"SUCCESS: Renamed/Moved '{old_name}' to '{new_name}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

def _protected_dir_refusal(sandbox_dir: Path, path: Path, filename: str):
    """A whole PROJECT directory is deleted through manage_projects, which
    previews and asks the user — `file_system delete projects/<id>` was a
    one-call rmtree around that confirm (§4MB). After outside content entered
    the request, the uploads folder and the projects root are refused too."""
    if not path.is_dir():
        return None
    # §4PE (F1, CRIT): judged against the OUTER root — from an active project
    # the scoped dir is `<root>/projects/<cur>`, so `<root>/projects/<other>`
    # was not relative to it, the ValueError below answered None, and the
    # whole other project was rmtree'd with no preview; the compare is
    # folded (APFS: `Projects/<id>` is the same directory)
    try:
        rel = tuple(_fold(p) for p in path.resolve().relative_to(_outer_root(sandbox_dir).resolve()).parts)
    except (ValueError, OSError):
        return None
    from .outcome import ToolOutcome
    # a PROJECT is a directory the project store owns (its id is 12 hex
    # characters); a user folder named "projects/<name>" inside a project's
    # workspace is not one
    if len(rel) == 2 and rel[0] == "projects" and re.fullmatch(r"[0-9a-f]{12}", rel[1]):
        return ToolOutcome.rejected(
            f"Error: '{filename}' is a whole project — delete it with manage_projects "
            f"action='delete' (it shows the user a preview and asks). Nothing was deleted.",
            world_changed=False, reason_code="protected_dir")
    if rel in (("projects",), ("uploads",)):
        from ..utils.provenance import untrusted_seen, content_refusal
        _src = untrusted_seen()
        if _src:
            return ToolOutcome.rejected(
                content_refusal(f"the deletion of '{filename}'", _src).replace("was not created", "was not done"),
                world_changed=False, reason_code="untrusted_content")
    return None


async def tool_delete_file(filename: str, sandbox_dir: Path):
    pretty_log("File Delete", filename, icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        _link = _symlink_itself(sandbox_dir, filename)
        if _link is not None:
            await asyncio.to_thread(_link.unlink)
            # the shape `_FS_RETIRE_RE` parses (the verifier's ground truth)
            return f"SUCCESS: Deleted '{filename}'. It was a link — its target was not touched."
        path = _get_safe_path(sandbox_dir, filename, allow_root=False)
        if not path.exists(): return f"Error: '{filename}' not found."
        _ref = _protected_dir_refusal(sandbox_dir, path, filename)
        if _ref is not None:
            return _ref
        if path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await asyncio.to_thread(path.unlink)
        _forget_parts_under(path)                  # §4OI r1 m5: a deleted parts file is not "unfinished"
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
    if re.match(r"(?i)https?://", str(filename).strip()):        # §4LL: not http_log.txt
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
        if not path.exists():
            fb = _scoped_root_fallback(sandbox_dir, filename)
            if fb is None:
                return _missing_file_message(filename, sandbox_dir)
            path = fb

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

        _nr = _nonregular_refusal(path, filename)       # §4PE (F3): a FIFO hung the request's thread
        if _nr:
            return _nr

        def _extract_chunk():
            # `page` is rebound below (out-of-range single-section files are
            # served as page 1); declare nonlocal so that assignment doesn't
            # shadow the enclosing parameter and UnboundLocalError the reads.
            nonlocal page
            if str(filename).lower().endswith(".pdf"):
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
                if len(text) > chunk_size:                     # §4PF (F11): a page is bounded like a text chunk
                    text = text[:chunk_size] + (f"\n[… page {page} has {len(text):,} chars; the first "
                                                f"{chunk_size:,} are shown — raise chunk_size for the rest]")
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
_PROJECT_DIR_RE = re.compile(r"projects/([0-9a-f]{12})(?:/|$)")


#: Operations that CANNOT mutate the workspace. Release immutability is
#: enforced by guarding everything NOT in this set (queue #12, 2026-08-21).
#:
#: ⚠ IT USED TO BE THE OTHER WAY AROUND — a deny-list of guarded operations,
#: `("write", "replace", "delete", "move", "append")` — and it leaked exactly
#: as a deny-list does. Reproduced against a real RELEASED project:
#:   * `copy`     — not listed at all → wrote into the released workspace;
#:   * `rename`   — the dispatcher accepts ["rename", "move"] and only "move"
#:                  was listed, so the SAME code path was guarded under one
#:                  alias and open under the other;
#:   * `download` — not listed → would write a fetched file straight in;
#:   * `move`     — listed, but the guard only looked at the SOURCE path, so
#:                  moving a file INTO a released project sailed through.
#: Three of the four landed files in a workspace documented as
#: "human-attested, immutable". An allow-list of READ-ONLY operations cannot
#: miss a newly-added mutating one; a deny-list of guarded ones always can.
_READ_ONLY_OPS = frozenset({
    "read", "read_file", "read_chunked", "read_document", "inspect",
    "outline", "symbols",
    "search", "find", "list", "list_files", "ls", "dir", "tree", "list_dir",
    "list_directory",
})


def _released_under(project_store, sandbox_dir, target) -> Optional[str]:
    """The id of a RELEASED project whose workspace lies INSIDE ``target``
    (a directory), else None. Workspaces live at ``<sandbox>/projects/<id>``,
    so only that folder (or an ancestor of it) can contain one — a copy such
    as ``backup/projects/<id>`` is not a workspace (second review: it could
    never be cleaned up). A symlink is judged as itself, never by what it
    points to. Never raises."""
    try:
        sd = Path(str(sandbox_dir))
        if _symlink_itself(sd, target) is not None:
            return None
        t = _get_safe_path(sd, str(target)).resolve()
        # the OUTER sandbox's projects folder: from an active project the
        # scoped dir is `<root>/projects/<id>`, and `/workspace/projects`
        # names `<root>/projects` (fourth review: `sd / "projects"` did not
        # exist there, so the delete removed every project)
        root_proj = (_outer_root(sd) / "projects").resolve()
        # folded: APFS takes `PROJECTS` and `projectſ` for `projects` (§4PE)
        _low = _fold
        if not root_proj.is_dir() or not (_low(root_proj) == _low(t)
                                          or _low(t) in {_low(x) for x in root_proj.parents}):
            return None
        for child in root_proj.iterdir():
            if re.fullmatch(r"[0-9a-f]{12}", _fold(child.name)) and str(
                    (project_store.get_project(_fold(child.name)) or {}).get("status", "")).upper() == "RELEASED":
                return _fold(child.name)
    except Exception:                                       # noqa: BLE001
        return None
    return None


def _projects_folder_block(sandbox_dir, target) -> Optional[str]:
    """Refusal when removing or moving ``target`` would take the projects
    folder — every project's workspace — with it: the folder itself or an
    ancestor of it. Independent of any release (fifth review: with no
    released project, `delete path=/workspace/projects` removed them all).
    A symlink is judged as itself. Never raises."""
    try:
        sd = Path(str(sandbox_dir))
        if _symlink_itself(sd, target) is not None:
            return None
        t = _get_safe_path(sd, str(target)).resolve()
        root_proj = (_outer_root(sd) / "projects").resolve()
        _low = _fold                                        # §4PE: folded compare
        if root_proj.is_dir() and (_low(root_proj) == _low(t) or _low(t) in {_low(x) for x in root_proj.parents}):
            return ToolOutcome.rejected(
                "SYSTEM BLOCK: this would remove or move the projects folder, which holds every project's "
                "workspace — NOT applied. Remove one project's files explicitly, or use manage_projects to "
                "archive or delete a project.", reason_code="projects_folder_blocked")
    except Exception:                                       # noqa: BLE001
        return None
    return None


def _symlink_itself(sandbox_dir, name) -> Optional[Path]:
    """The path of ``name`` WITHOUT following its last component, when that
    is a symlink inside the sandbox, else None. Fresh review: delete and
    rename resolved the link and acted on its TARGET (another project's
    folder deleted, the link left dangling); a link is removed or moved as
    itself. Never raises."""
    try:
        raw = str(name or "").strip()
        if not raw:
            return None
        rel = Path(raw.lstrip("/"))
        parent = _get_safe_path(Path(str(sandbox_dir)), str(rel.parent) if str(rel.parent) not in ("", ".") else ".")
        cand = parent / rel.name
        return cand if cand.is_symlink() else None
    except Exception:                                       # noqa: BLE001
        return None


def _released_write_block(project_store, sandbox_dir, target, *, removes: bool = False,
                          include_scope: bool = True) -> Optional[str]:
    """Refusal message when a write would land inside a RELEASED project's
    workspace, else None. The hard half of release immutability (2026-07-25):
    the briefing steers, but the agent's measured failure mode is regressing
    working artifacts, so the write path itself must refuse. Derives the
    project id from the scoped sandbox_dir OR the target path; one cheap
    status lookup. Never raises — an unresolvable store never blocks a write.

    ``removes`` (§4KW): ``target`` is REMOVED or moved away (the source of a
    delete/rename/move) — then a parent folder of a released workspace is
    locked too, and a symlink is judged as the link itself.
    ``include_scope=False``: judge the target only, not the scoped sandbox
    (forget of a MEMORY must not be refused because the active project is
    released — second review).
    """
    try:
        if project_store is None:
            return None
        # ⚠ IDs from the RESOLVED path's components, every one of them, any
        # case — not the first regex hit on the raw spelling. The raw form
        # let `Projects/`, `AAAA…` (APFS is case-insensitive), `projects//`,
        # `projects/zz/../<id>` and, from an ACTIVE scope, `projects/<released>`
        # through, because the first hit was the active id (fresh-eye §4KC
        # round 2, seven confirmed bypasses).
        cands = set()
        sd = Path(str(sandbox_dir))
        part_lists = []
        if include_scope:
            try:
                part_lists.append(sd.resolve().parts)
            except Exception:                               # noqa: BLE001
                part_lists.append(sd.parts)
        _link = _symlink_itself(sd, target) if (removes and target is not None) else None
        if _link is not None:
            # the link's own location (its parent resolved, its name kept)
            part_lists.append(_link.parent.resolve().parts + (_link.name,))
        elif target is not None and str(target).strip():
            try:
                part_lists.append(_get_safe_path(sd, str(target)).resolve().parts)
            except Exception:                               # noqa: BLE001
                # unresolvable (a traversal the safe-path helper refused):
                # judge the NORMALISED raw spelling, so `zz/../<id>` counts
                part_lists.append(Path(os.path.normpath(f"{sd}/{target}")).parts)
        for parts in part_lists:
            low = [_fold(p) for p in parts]                 # §4PE: NFKC+casefold, not lower()
            for i in range(len(low) - 1):
                if low[i] == "projects" and re.fullmatch(r"[0-9a-f]{12}", low[i + 1]):
                    cands.add(low[i + 1])
        hay = (f"{sandbox_dir}/{'' if target is None else target}" if include_scope
               else ("" if target is None else str(target)))
        for m in _PROJECT_DIR_RE.finditer(_fold(str(hay).replace("\\", "/"))):
            cands.add(m.group(1))
        pid = next((c for c in sorted(cands)
                    if str((project_store.get_project(c) or {}).get("status", "")).upper() == "RELEASED"),
                   None)
        if pid is None and removes and target is not None and str(target).strip():
            # Fresh review: a PARENT of a released workspace — `delete
            # path=projects`, `rename projects` — named no project id and
            # removed every project, released ones included.
            pid = _released_under(project_store, sd, str(target))
        if pid is not None:
            return (
                ToolOutcome.rejected(f"SYSTEM BLOCK: project {pid} is RELEASED (human-attested, "
                f"immutable) — this write was NOT applied. To change it, "
                f"(only if the user asked for a change) fork a development copy first: manage_projects "
                f"action=create_version project_id={pid} "
                f"description=\"<the requested change>\", then edit the new "
                f"version's workspace. The released version keeps running "
                f"untouched.", reason_code="released_write_blocked"))
    except Exception:
        return None
    return None


#: files whose whole point is to exist empty (§4LL review: a write with
#: content="" was refused as a lost payload)
_EMPTY_OK_NAMES = frozenset({"__init__.py", ".gitkeep", ".keep", ".nojekyll", "py.typed"})


#: the operations the tool schema advertises (tools/registry.py enum) — the
#: unknown-operation message lists THESE (§4LL: it listed a nonexistent
#: "batch" and left out edit/search/copy…). Pinned equal to the enum in tests.
ADVERTISED_OPS = ("read", "read_chunked", "outline", "symbols", "inspect", "search", "find", "list_files",
                  "write", "edit", "replace", "download", "copy", "rename", "move", "delete",
                  "quarantine_list", "quarantine_inspect", "quarantine_delete")
#: §4NL: the quarantine operations — they never touch the workspace
QUARANTINE_OPS = frozenset({"quarantine_list", "quarantine_inspect", "quarantine_delete"})
#: …and every name the dispatch below accepts (the list aliases)
ACCEPTED_OPS = ADVERTISED_OPS + ("list", "ls", "dir", "tree", "list_dir", "list_directory")


async def tool_file_system(operation: str = None, sandbox_dir: Path = None, path: str = None, content: str = None, replace_with: str = None, destination: str = None, pattern: str = None, max_context: int = 8192, read_budget: "ReadBudget | None" = None, **kwargs):
    # §4LL: "Read" / " write " are the same operations (the read-only set
    # already lowercased; the dispatch did not)
    if isinstance(operation, str):
        operation = operation.strip().lower()
    if operation in ("append", "append_file", "write_append"):
        # §4NZ (operator: "verify that append works"): the natural name for
        # adding to the end was "Unknown operation" — a whole round trip for
        # an intent with one meaning. It IS write + append=true.
        operation = "write"
        kwargs["append"] = True
    if not operation:
        return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'operation' parameter is MANDATORY. You must specify it (e.g., operation='read').", reason_code="missing_operation")
    if operation in QUARANTINE_OPS:
        # §4NL: the quarantine lives OUTSIDE the sandbox; no path, no workspace guard applies
        from . import quarantine as _q
        _qid = kwargs.get("quarantine_id") or kwargs.get("id") or path or kwargs.get("filename") or ""
        if operation == "quarantine_list":
            return await _q.quarantine_list()
        if operation == "quarantine_inspect":
            return await _q.quarantine_inspect(str(_qid))
        return await _q.quarantine_delete(str(_qid))
    # Release immutability: block mutations into a RELEASED project's
    # workspace at the WRITE PATH (see _released_write_block).
    if str(operation or "").strip().lower() not in _READ_ONLY_OPS:
        # EVERY candidate target, not just the first one that happens to be
        # set: `move`/`copy` carry the source in `path` and the real write in
        # `destination`, and the old `path or filename or destination` chain
        # therefore inspected the SOURCE and let the destination through.
        _op = str(operation or "").strip().lower()
        _moving = _op in ("copy", "rename", "move")
        _removes_src = _op in ("delete", "remove", "rm", "rename", "move")
        # EVERY alias the healing step below reads (third review: `file=`,
        # `new_name=`, `target=`, `new_path=`, `data=`, `text=` reached the
        # write unchecked — `write file=projects/<released>/app.py` overwrote it)
        _srcs = (path, kwargs.get("filename"), kwargs.get("path"), kwargs.get("file"))
        # `url` becomes the target of any non-download operation below
        # (fourth review: `delete url=projects/<released>/app.py` deleted it)
        if _op != "download":
            _srcs += (kwargs.get("url"),)
        if _op == "copy":
            _srcs = ()      # a copy READS its source (fourth review: copying out was refused)
        _dsts = (destination, kwargs.get("destination"), kwargs.get("dest"), kwargs.get("new_name"),
                 kwargs.get("target"), kwargs.get("new_path"))
        _dsts += ((replace_with, content, kwargs.get("data"), kwargs.get("text")) if _moving else ())
        for _target, _rm in ([(t, _removes_src) for t in _srcs] + [(t, False) for t in _dsts]):
            if not isinstance(_target, str):
                continue
            _hb = _host_path_block(sandbox_dir, _target)           # §4LL: every write target
            if _hb:
                return _hb
            if kwargs.get("confine_to_sandbox"):
                _cb = _confined_write_block(sandbox_dir, _target)  # §4PG (R2-4): an autonomous leaf
                if _cb:
                    return _cb
            _pb = _projects_folder_block(sandbox_dir, _target) if _rm else None
            if _pb:
                return _pb
            _rb = _released_write_block(
                kwargs.get("project_store"), sandbox_dir, _target, removes=_rm)
            if _rb:
                return _rb
    # Unified mapping for common parameter hallucinations
    url = kwargs.get("url")
    
    # --- HALLUCINATION HEALING ---
    # 1. If the LLM put the URL in the 'path' or 'filename' parameter
    if path and re.match(r"https?://", str(path), re.I):     # §4PE: `httpd.conf` is a name
        if not url: url = str(path)
        path = None
    elif kwargs.get("filename") and re.match(r"https?://", str(kwargs.get("filename")), re.I):
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

    # §4HB: redeem a payload held by a rejected `replace` on this path.
    _held_redeemed = False
    if (operation == "write" and target_path
            and str(final_content or "").strip() == _HELD_CONTENT_TOKEN
            and not (kwargs.get("append") is True
                     or str(kwargs.get("append")).strip().lower() in ("true", "1", "yes"))):
        _held = _redeem_held_content(sandbox_dir, target_path)
        if _held is None:
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: no held content for '{target_path}' — "
                f"a hold lasts 10 minutes and is redeemed once. Send the full "
                f"'content' explicitly.", reason_code="no_held_content")
        pretty_log("File Write", f"{target_path}: redeeming held content "
                   f"({len(_held)} chars) from the rejected replace",
                   icon=Icons.TOOL_FILE_W)
        final_content = _held
        _held_redeemed = True

    # §4SH: an EMPTY last part finishes a file written in parts — the way out
    # for a model whose real last part went with more_parts=true
    def _flag(v):
        return v is True or str(v).strip().lower() in ("true", "1", "yes")
    if (operation == "write" and target_path and content == "" and not final_content
            and _flag(kwargs.get("append"))):     # r1: exactly '' (a "\n" append appends)
        try:
            _fin = _get_safe_path(sandbox_dir, target_path)
        except Exception:  # noqa: BLE001
            _fin = None
        if _fin is not None and _fin.is_file() and _is_in_progress(_fin):
            return await finish_parts_file(_fin, str(target_path))
        if _fin is not None and _fin.is_file():
            # r3: information, not a refusal — an extra finish after a part that already closed
            # the file is harmless (a REJECTED read as a failure of the file)
            return (f"NOTE: '{target_path}' has no part pending in this request (it is not open for parts), so an "
                    f"empty append has nothing to finish and nothing was changed. To add text at its end, send "
                    f"it in 'content' with append=true.")
        # r4: a finish on a missing file — the MANDATORY refusal below advised this very call
        _why = ("is not a usable path in the sandbox" if _fin is None else "is a directory" if _fin.is_dir()
                else "is not a regular file" if _fin.exists() else "does not exist")      # r5/r6
        return ToolOutcome.rejected(f"'{target_path}' {_why}, so there is nothing to finish — check the "
                                    f"path. A new file is written with operation='write' and its full content.",
                                    reason_code="missing_file")

    # If the LLM put the content in 'path' but didn't provide 'content' (common for write)
    if (operation == "write" and target_path and not final_content
            and not (content == "" and Path(str(target_path)).name.lower() in _EMPTY_OK_NAMES)):
        # Check if the LLM accidentally sent the content as the only other parameter
        _fin_hint = (" To FINISH a file written in parts, send content='' (an empty string) with append=true — "
                     "do not re-send the file." if _flag(kwargs.get("append")) else "")      # §4SH r3
        return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'content' parameter is MANDATORY for write operations. "
                                    "You must provide the full text to write." + _fin_hint, reason_code="missing_content")

    sandbox_manager = kwargs.get("sandbox_manager")

    # Accept the aliases the model reaches for; `path` (optional) scopes the
    # listing to one subdirectory — it used to be silently dropped, so every
    # "list <subdir>" call re-listed the (truncated) root and big workspaces
    # looked like their subdirectories were empty.
    if operation in ("list", "list_files", "ls", "dir", "tree", "list_dir",
                     "list_directory"):
        return await tool_list_files(sandbox_dir, path=target_path)
    if operation == "search": 
        search_target = pattern or final_content
        if not search_target:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for search operations.", reason_code="missing_pattern")
        return await tool_file_search(search_target, sandbox_dir, target_path, sandbox_manager)
    if operation == "find":
        search_target = pattern or final_content
        # §4HC: `find path="*.md"` with no pattern — every one of the 7
        # corpus rejections had the glob in `path`. A literal directory is
        # never spelled with `*` or `?`, so the intent is unambiguous: the
        # glob is the pattern, the directory is the root.
        _tp = str(target_path or "")
        if not search_target and _tp and any(ch in _tp for ch in "*?"):
            search_target, target_path = _tp, "."
            pretty_log("File Find", f"read the glob {_tp!r} from 'path' as the pattern",
                       icon=Icons.TOOL_FILE_R)
        if not search_target:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for find operations (e.g. '*.py').", reason_code="missing_pattern")
        return await tool_find_files(search_target, sandbox_manager, target_path or ".", sandbox_dir=sandbox_dir)
    
    if operation == "download":
        if not url:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'url' parameter is MANDATORY for download operations.", reason_code="missing_url")
        from .quarantine import is_darkweb_url as _is_darkweb_url, quarantine_download as _quarantine_download
        if _is_darkweb_url(str(url)):
            # §4NL: a dark-web file never lands in the workspace — the model
            # cannot choose a path for it
            return await _quarantine_download(str(url), kwargs.get("tor_proxy"))
            
        # Auto-heal missing or invalid target_path
        # `startswith("http")` also discarded legitimate filenames — httpd.conf,
        # http_log.txt, https_cert.pem — so test for an actual URL instead.
        _tp = str(target_path or "").strip()
        if not _tp or "://" in _tp or _tp == str(url):
            parsed = urllib.parse.urlparse(str(url))
            path_name = Path(parsed.path).name
            # PERCENT-DECODE the name (§4JZ). Saved raw, a Wikimedia URL lands
            # as `960px-Zoi_..._%28cropped%29.jpg` while every human-readable
            # reference to it — the model's, the page's — says `(cropped)`, so
            # a later `reference_images` / read by that name misses. Seen live.
            # ⚠ Re-take `.name` AFTER decoding: `%2F` decodes to a separator,
            # and a filename that smuggles one would escape this directory.
            if path_name:
                decoded = Path(urllib.parse.unquote(path_name)).name
                # ⚠ Drop CONTROL characters, not just surrounding whitespace.
                # `%00` put a real NUL in the name; `_get_safe_path` then raised
                # ValueError("embedded null byte") and the handler returned that
                # BARE STRING, which the outcome classifier reads as neither a
                # failure nor a rejection — a download that wrote nothing was
                # booked as a success. And `%0A` let the URL inject whole fake
                # result lines into the tool output the model reads.
                # (Deliberately NOT quoting the literal prefix here: the
                # producer/parser parity check scans this file for message
                # shapes, and a comment carrying one registers as a producer.)
                decoded = "".join(c for c in decoded if c.isprintable())
                decoded = decoded.replace("\\", "_").strip()
                # Leading dots are legitimate (`.env`, `.gitignore`); only a
                # name that is ENTIRELY dots is a traversal shape.
                if decoded.strip(".") == "":
                    decoded = ""
                if len(decoded) > 180:
                    stem, _, ext = decoded.rpartition(".")
                    decoded = (stem[:170] + "." + ext[:9]) if stem else decoded[:180]
                if decoded:
                    path_name = decoded
            target_path = path_name if path_name else "download.bin"

        return await tool_download_file(url=str(url), sandbox_dir=sandbox_dir, tor_proxy=kwargs.get("tor_proxy"), filename=target_path)

    #: §4KB: operations that act on the PROJECT, not on one file. They must
    #: be exempt from the path requirement below, which would otherwise
    #: reject them before their own argument check ever runs — the model
    #: would be told to name a file for a whole-project symbol lookup.
    _PATHLESS_OPS = frozenset({"symbols"})

    # ══════════════════════════════════════════════════════════════════
    # NO MUTATION INSIDE `.git`. §4KB round 4.
    #
    # Rounds 2, 3 and 4 each closed a git escape and each was an INSTANCE
    # fix: a `.git` FILE, then `core.worktree`/`core.hooksPath`/`commondir`/
    # `alternates`, then `diff.external` and `filter.*.clean` — and git has
    # many more keys that name a program to run (`core.pager`,
    # `sequence.editor`, `*.textconv`, `credential.helper`,
    # `core.sshCommand`, `alias.*`, `uploadpack.packObjectsHook`). Enumerating
    # them is unwinnable, and three rounds of trying is the evidence.
    #
    # Every one of those escapes needs the SAME precondition: the model
    # writing inside `.git`. Nothing legitimate does — git's own plumbing is
    # reached through the `git` tool, and a workspace's repository internals
    # are not a place an agent edits by hand. So the precondition is what
    # gets removed, at the one place every mutating operation passes through.
    #
    # This is deliberately a DENY on a path COMPONENT, not on a filename:
    # `.git/config`, `.git/objects/info/alternates`, `a/b/.git/hooks/pre-commit`
    # and a `.git` file itself are all refused. `_get_safe_path` has already
    # resolved traversal, so `../.git/config` cannot slip past.
    # The SOURCE and every DESTINATION (fresh review: copy/move INTO `.git`
    # passed — the same source-only bug the released lock had).
    for _git_target in ([target_path] + [d for d in (destination, kwargs.get("destination"),
                                                     kwargs.get("dest"), kwargs.get("new_name"),
                                                     kwargs.get("target"), kwargs.get("new_path"),
                                                     # healed destinations (second/third review)
                                                     *((replace_with, content, final_content)
                                                       if operation in ("copy", "rename", "move")
                                                       else ())) if d and isinstance(d, str)]
                        if (operation not in _READ_ONLY_OPS and target_path) else []):
            try:
                _probe = _get_safe_path(sandbox_dir, _git_target)
                # Lower-cased: the data volume is APFS (case-insensitive), so
                # `.GIT/config` IS `.git/config` on disk (fresh-eye §4KC r2).
                _parts = {str(p).lower() for p in _probe.parts} | {
                    p.lower() for p in
                    str(_git_target).replace("\\", "/").split("/")}
                if ".git" in _parts:
                    pretty_log(
                        "Write Blocked",
                        f"{_git_target}: refused — a repository's internals are "
                        f"not editable by hand",
                        icon=Icons.SHIELD, level="WARNING")
                    return ToolOutcome.rejected(
                        f"SYSTEM BLOCK: '{_git_target}' is inside a `.git` "
                        f"directory and cannot be created, modified or removed. "
                        f"Repository internals are never edited by hand: a "
                        f"hand-written git config can redirect the repository "
                        f"out of this workspace or name a program for git to "
                        f"execute. Use `execute` to run git commands. Nothing "
                        f"was changed.",
                        reason_code="dotgit_write_blocked")
            except ValueError:
                pass            # path escape: the operation's own guard reports it
            except (OSError, RuntimeError) as _pe:
                # ENAMETOOLONG, or a symlink LOOP (`resolve()` raises
                # RuntimeError). A path this check cannot resolve is one it
                # cannot CLEAR either — a `.git` symlink pointing out of the
                # sandbox landed here and the deny was silently skipped (§4KC
                # r3). Only ValueError was caught before r2, so a looped link
                # tracebacked out of every mutating op.
                return ToolOutcome.rejected(
                    f"SYSTEM BLOCK: '{_git_target}' could not be resolved for "
                    f"the write-safety checks ({_pe}), so nothing was changed. "
                    f"Name the file by a plain relative path.",
                    reason_code="unresolvable_path")

    if not target_path and operation not in _PATHLESS_OPS:
        return ToolOutcome.rejected(f"SYSTEM INSTRUCTION: The 'path' (target filename) is missing for the '{operation}' operation. You MUST specify WHICH file to {operation}.", reason_code="missing_path")
    
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
        _limit = _as_int(kwargs.get("limit"))
        if _start is None and _as_int(kwargs.get("offset")) is not None:
            _start = max(1, _as_int(kwargs.get("offset")))          # review B: offset/limit shape (offset 0 = line 1)
        if _end is None and _limit and _limit > 0:
            _end = (_start or 1) + _limit - 1
            _start = _start or 1
        _rd = await tool_read_file(target_path, sandbox_dir, max_context=max_context,
                                   read_budget=read_budget,
                                   start_line=_start, end_line=_end)
        if not str(_rd).startswith("Error") and "SAMPLE ONLY (" not in str(_rd)[:400]:
            try:                                                            # §4OE-2 receipt; §4OG-1 (h): coverage + stamp
                _shown = re.search(r"(?m)^--- .*? \(lines (\d+)-(\d+)\) ---", str(_rd)[:600])
                if _shown:
                    _mark_read(_get_safe_path(sandbox_dir, target_path),
                               start=int(_shown.group(1)), end=int(_shown.group(2)))
                elif _start is None and _end is None:
                    _mark_read(_get_safe_path(sandbox_dir, target_path), whole=True)
            except Exception:  # noqa: BLE001
                pass
        return _rd
    elif operation == "read_chunked":
        page = kwargs.get("page", 1)
        chunk_size = kwargs.get("chunk_size", 32000)
        # § context R1 B3: chunked reads were INVISIBLE to the batch budget —
        # N per-call-capped pages accumulate unbudgeted (the exact
        # "each call clears its own cap, together they don't fit" shape the
        # ReadBudget exists for), and the budget's own refusal text routed
        # the model into this uncharged bypass. Refuse at zero remaining;
        # charge whatever is returned.
        if read_budget is not None and read_budget.remaining <= 0:
            if read_budget.spent == 0:
                return ("Error: chunked reading is refused — the conversation "
                        "is already near the context ceiling (no read budget "
                        "this turn). Use operation='search', a ranged read "
                        "(start_line/end_line), or an 'execute' digest script.")
            return (f"Error: the read budget for THIS turn is exhausted "
                    f"({read_budget.spent / 1024:.1f} KB already read). Use "
                    f"operation='search', a ranged read, or an 'execute' "
                    f"digest script.")
        _ck = await tool_read_document_chunked(target_path, sandbox_dir, page=page, chunk_size=chunk_size, max_context=max_context)
        if isinstance(_ck, str) and not _ck.startswith("Error"):
            if read_budget is not None:
                read_budget.charge(len(_ck))
        if not str(_ck).startswith("Error"):
            try:                                                            # §4OE-2; §4OG-1 (h): one page = its byte range
                _pg = max(1, int(page or 1))
                try:
                    _cs = int(chunk_size)
                except (TypeError, ValueError):
                    _cs = 32000
                _cs = min(max(1000, _cs), max(80000, int(max_context * 3.5 * 0.2)))   # the reader's clamp (reader M2)
                _eff = _cs - min(200, _cs // 4)                                          # …and its stride
                _mark_read(_get_safe_path(sandbox_dir, target_path),
                           byte_from=(_pg - 1) * _eff, byte_to=(_pg - 1) * _eff + _cs - 1)
            except Exception:  # noqa: BLE001
                pass
        return _ck
    elif operation == "outline":
        # §4KB step 5a. A MAP, never grounding: an outline does not return
        # file content, so it deliberately does NOT stamp a read receipt —
        # knowing a function is at line 120 is not having seen line 120.
        from .outline import outline_source, render_outline
        pretty_log("File Outline", str(target_path), icon=Icons.TOOL_FILE_O)
        try:
            _p = _get_safe_path(sandbox_dir, target_path)
        except ValueError as ve:
            return str(ve)
        _nr = _nonregular_refusal(_p, target_path)      # §4PE: a FIFO was reported as nonexistent
        if _nr:
            return _nr
        if not _p.exists() or not _p.is_file():
            return _missing_file_message(target_path, sandbox_dir)
        try:
            _src = await asyncio.to_thread(
                _p.read_text, encoding="utf-8", errors="replace")
        except OSError as oe:
            return f"Error: failed to read '{target_path}': {oe}"
        _syms, _method = await asyncio.to_thread(
            outline_source, _src, str(target_path))
        return render_outline(_syms, str(target_path), _method)
    elif operation == "symbols":
        # §4KB step 5b. Project-wide DEFINITION lookup. Uses is a different
        # question with a better existing answer (`search` is ripgrep), and
        # the renderer says so rather than shipping a weaker duplicate.
        from .outline import build_symbol_index, render_definitions
        _name = str(kwargs.get("name") or pattern or "").strip()
        pretty_log("Symbol Lookup", _name or "(no name given)",
                   icon=Icons.TOOL_FILE_O)
        if not _name:
            return ToolOutcome.rejected(
                "SYSTEM INSTRUCTION: operation='symbols' needs `name` — the "
                "symbol you are looking for (a function, class or constant). "
                "To search for arbitrary TEXT, use operation='search'.",
                reason_code="missing_symbol_name")
        _idx = await asyncio.to_thread(build_symbol_index, Path(sandbox_dir))
        # M3: `render_definitions` calls difflib.get_close_matches, measured
        # at 24 ms on the real 32,398-name index — blocked loop, 20x the
        # substring scan it augments. The index build was already threaded;
        # the render was not.
        return await asyncio.to_thread(render_definitions, _idx, _name)
    elif operation == "inspect":
        # `lines` used to be dropped here, so the model could never widen the
        # peek; tool_inspect_file coerces bad values back to the default.
        return await tool_inspect_file(target_path, sandbox_dir,
                                       lines=kwargs.get("lines", 10))
    elif operation == "write":
        _app = kwargs.get("append")
        _app = _app is True or str(_app).strip().lower() in ("true", "1", "yes")
        _more = kwargs.get("more_parts")
        _more = _more is True or str(_more).strip().lower() in ("true", "1", "yes")
        _cut = kwargs.get("cut_off_saved")
        try:
            _wpath = _get_safe_path(sandbox_dir, target_path) if target_path else None
        except Exception:  # noqa: BLE001
            _wpath = None
        _after = kwargs.get("append_after")
        if _after is not None and _app and _wpath is not None:
            # §4NZ: a cut add-at-end EDIT turned into an append — only if the
            # file really ends with the edit's anchor
            try:
                _tail_ok = _wpath.read_text(encoding="utf-8", errors="replace").rstrip().endswith(
                    str(_after).rstrip())
            except Exception:  # noqa: BLE001
                _tail_ok = False
            if _tail_ok:
                # the edit replaced `anchor` with `anchor + extra`; the file
                # already has a line break after the anchor, so the extra's
                # own leading break would add a blank line (review)
                try:
                    _ftail = _wpath.read_text(encoding="utf-8", errors="replace")
                    _after_anchor = _ftail[_ftail.rstrip().rfind(str(_after).rstrip()) + len(str(_after).rstrip()):]
                except Exception:  # noqa: BLE001
                    _after_anchor = ""
                if "\n" in _after_anchor and isinstance(final_content, str):
                    final_content = final_content[2:] if final_content.startswith("\r\n") else (
                        final_content[1:] if final_content.startswith("\n") else final_content)
            if not _tail_ok:
                return ToolOutcome.failed(
                    f"SYSTEM ERROR: your edit to '{target_path}' was CUT OFF by the output limit and its "
                    f"old_string is not the end of the file, so NOTHING was changed. Add text at the END "
                    f"with operation='write', append=true (~15 KB per part); change the middle with "
                    f"smaller edits.", world_changed=False, reason_code="tool_call_cut_off")
        _restart, _head = False, ""
        if (_wpath is not None and _app and _wpath.exists() and not _is_in_progress(_wpath)
                and _wpath.suffix.lower() in (".html", ".htm", ".xhtml", ".xml", ".svg")):
            # verify: the same two-documents failure on a FINISHED file — an
            # append that opens a second document of the file's own kind
            try:
                _done_head = _wpath.read_text(encoding="utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                _done_head = ""
            if restarts_file(_done_head, final_content):
                return ToolOutcome.rejected(
                    f"SYSTEM INSTRUCTION: this part opens a new document, but '{target_path}' already "
                    f"is one — appending it would put two documents in one file. To add to it, append only "
                    f"the new text; to replace it, use a plain write. Nothing was changed.",
                    reason_code="second_document")
        if (_cut and _wpath is not None and not _app and _wpath.exists() and not _is_in_progress(_wpath)
                and _written_this_request(_wpath)):
            # §4NZ (probe-ap-b): the model wrote a small skeleton, then re-sent
            # the whole page from <!DOCTYPE, cut off — refused as "a partial
            # replacement of an existing file", 4 minutes of output thrown
            # away, though THIS request had just written that file. A restart
            # of a file this request wrote is the new version (saved as a part).
            try:
                _restart = restarts_file(_wpath.read_text(encoding="utf-8", errors="replace")[:400],
                                         final_content)
            except Exception:  # noqa: BLE001
                _restart = False
        if (_wpath is not None and _wpath.exists() and _is_in_progress(_wpath)):
            # §4NW-b (req c7f06c37): a plain write that OPENS the way the file
            # opens is a restart of the whole file — it REPLACES the parts
            # (the model re-emitted the file from <!DOCTYPE twice; the old
            # 200-char exact match missed the second and a cut restart was
            # appended onto the first copy: two documents in one file)
            try:
                _head = _wpath.read_text(encoding="utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                _head = ""
            _restart = restarts_file(_head, final_content)
            if _app and _restart:
                # the same two-documents failure through the append path
                return ToolOutcome.rejected(
                    f"SYSTEM INSTRUCTION: this part opens a new document, but '{target_path}' already "
                    f"starts one — appending it would put two documents in one file. To continue, append "
                    f"only the text that comes AFTER the end of the file. To start the file over, send the "
                    f"new opening part as a plain write (no append). Nothing was changed.",
                    reason_code="file_in_progress")
            try:
                _saved_n = _wpath.stat().st_size
            except OSError:
                _saved_n = 0
            if (_restart and _cut and not _app and _saved_n >= 4096
                    and len(str(final_content or "").encode("utf-8")) < _saved_n * 1.25):
                # §4OA (probe-oa-facts): the model re-sent gen_ocean.py from its
                # first line four times; each try was cut at the same length,
                # replaced the saved part and gained nothing — 13 minutes. A
                # cut restart that is not clearly longer is no progress.
                return ToolOutcome.failed(
                    f"SYSTEM ERROR: you started '{target_path}' over from its first line again and this "
                    f"reply was CUT OFF before it got past what is already saved ({_saved_n:,} bytes) — "
                    f"starting over can never get past that point. Nothing was "
                    f"changed; the saved part stays. Continue it: operation='write', append=true, content "
                    f"= ONLY the lines that come after its end (read its last lines if needed), ~15 KB "
                    f"per call. Or, to replace it with a much shorter version (e.g. one that generates "
                    f"the data with a loop instead of writing every item out), delete the file first, "
                    f"then write the new one.", world_changed=False,
                    reason_code="tool_call_cut_off")
            # (not for a CUT write: its content is a salvaged prefix, not the
            # model's intended version — review MAJOR)
            if (_restart and not _app and not _cut and _saved_n >= 4096
                    and len(str(final_content or "").encode("utf-8")) * 2 < _saved_n):
                # §4OA (probe-ap2-h): a 1.6 KB "restart" (a fresh skeleton)
                # replaced 72 KB of saved parts — 4 minutes of output gone
                return ToolOutcome.rejected(
                    f"SYSTEM INSTRUCTION: '{target_path}' already holds {_saved_n:,} bytes written in parts in "
                    f"this request; this write would REPLACE them with a much shorter version. Continue it "
                    f"with operation='write' and append=true. To really start over, delete the file first. "
                    f"Nothing was changed.", reason_code="file_in_progress")
        if (_cut and _wpath is not None and not _app and not _restart and _wpath.exists()
                and _is_in_progress(_wpath)):
            # r5: the model writes `append`/`more_parts` AFTER `content`, so
            # a cut APPEND arrives as a plain write. The file is being written
            # in parts in this request, so this is its next part — unless it
            # repeats the file's opening word for word (then nothing is written).
            if _head and str(final_content or "")[:200] == _head[:200]:
                return ToolOutcome.failed(
                    f"SYSTEM ERROR: your write to '{target_path}' was CUT OFF by the output limit and "
                    f"repeats the file's opening, so NOTHING was written — the parts already saved are "
                    f"kept. Add the NEXT part with append=true (~15 KB).", world_changed=False,
                    reason_code="tool_call_cut_off")
            _app = True
        if (_wpath is not None and not _app and not _restart and _wpath.exists()
                and _is_in_progress(_wpath)):
            # §4NU r4: "Part 2" sent as a plain write replaced Part 1
            _sz = _wpath.stat().st_size
            if _progress_key(_wpath) not in _WRITTEN:
                # review B/r2: the file existed BEFORE this request and the model has
                # been appending parts to it — "delete the file first" would aim at
                # the owner's data, so the advice names the parts instead
                return ToolOutcome.rejected(
                    f"SYSTEM INSTRUCTION: '{target_path}' existed before this request and you have been "
                    f"appending parts to it ({_sz:,} bytes now) — a write without append=true would REPLACE "
                    f"the owner's file. To add the next part use operation='write' with append=true (and "
                    f"more_parts=true unless it is the last); to undo your parts, remove them with "
                    f"operation='edit'. Nothing was changed.", reason_code="file_in_progress")
            return ToolOutcome.rejected(
                f"SYSTEM INSTRUCTION: '{target_path}' is being written in PARTS in this request "
                f"({_sz:,} bytes so far) — a write without append=true would REPLACE them. To add the "
                f"next part use operation='write' with append=true (and more_parts=true unless it is the "
                f"last). To really start over, delete the file first. Nothing was changed.",
                reason_code="file_in_progress")
        if (_cut and _wpath is not None and not _app and not _restart and _wpath.exists()):
            # a cut-off REPLACEMENT of an existing file is never saved as a
            # part: a partial replacement would lose the file's tail (a
            # restart of a file THIS request is building is the exception:
            # its old parts are the version being replaced)
            return ToolOutcome.failed(
                f"SYSTEM ERROR: your write to '{target_path}' was CUT OFF by the output limit, so "
                f"NOTHING was written ('{target_path}' already exists and was left unchanged). If you "
                f"were ADDING to it, send each part with append=true (~15 KB per part). Only to REPLACE "
                f"it: the first part with a plain write and more_parts=true, then append=true for the "
                f"rest.", world_changed=False, reason_code="tool_call_cut_off")
        _ends = ""
        if _more and _app and not _cut and _wpath is not None:
            # §4SH: an appended part that ENDS its document is the last part, whatever
            # more_parts says. Not a FIRST part: cb27058e's part 1 was the whole
            # skeleton through </html>, and its scripts were appended after it.
            try:
                _x = _wpath.suffix.lower()
                _big = (_x == ".json" and _wpath.is_file() and _wpath.stat().st_size > DOC_END_JSON_MAX_BYTES)
                # r1: only a format with a definite end is read, and off the event loop
                if _x in (".html", ".htm", ".xhtml", ".svg", ".json") and not _big and _wpath.is_file():
                    _prior = await asyncio.to_thread(_wpath.read_text, encoding="utf-8", errors="replace")
                    _ends = await asyncio.to_thread(document_ends_here, target_path,
                                                    _prior + str(final_content or ""))
            except Exception:  # noqa: BLE001
                _ends = ""
            if _ends:
                _more = False
        if _app:
            res = await tool_append_file(target_path, final_content, sandbox_dir, more_parts=_more)
        else:
            res = await tool_write_file(target_path, "" if (final_content is None and content == "")
                                        else final_content, sandbox_dir, more_parts=_more)
            if _held_redeemed and (str(getattr(res, "status", "")).lower().endswith("rejected")
                                   or str(res).startswith("REJECTED")):
                # §4PE (F8): the hold was consumed by a write the gate then
                # refused (unread rewrite) — the model regenerated the whole
                # payload. Re-hold it so the prescribed `<<HELD>>` retry works.
                _hold_rejected_content(sandbox_dir, target_path, str(final_content))
                res = append_note(res, "\nThe held content is KEPT for another 10 minutes: once the refusal "
                                       "above is satisfied (read the file), write again with content='<<HELD>>'.")
        if _wpath is not None and str(res).startswith("SUCCESS"):
            _mark_in_progress(_wpath, _more)
            if _ends:
                res = append_note(res, _finished_by_end_note(target_path, _ends))
            elif _more and not _cut:
                res = append_note(res, OPEN_PART_NOTE)
            if not _app:
                _mark_written(_wpath)       # review: an append to an owner's file is not "this request's file"
            else:
                _note_mutation(_wpath)      # §4OG-1 (h): this request's own append keeps its view current
            if _cut:
                try:
                    _cut_n = int(_cut)
                except (TypeError, ValueError):
                    _cut_n = 0
                _note = _cut_saved_note(_cut_n, _wpath, str(target_path)) + str(kwargs.get("cut_note_extra") or "")
                from .outcome import with_text as _with_text
                res = _with_text(res, str(res) + _note)
            if _restart:
                from .outcome import with_text as _with_text
                res = _with_text(res, str(res) + (
                    f"\nNOTE: this write starts '{target_path}' over from its opening, so it REPLACED the "
                    f"parts written before in this request. Continue THIS version with append=true — "
                    f"do not start the file over again."))
        return res
    elif operation == "edit":
        # §4KC: NO aliases, by design. The alias fan-out on `replace` is
        # where its interface failures lived (`content` holding a whole
        # file, `replace_with` missing). A call that reaches for the old
        # names is told the new ones, not healed into them.
        _old_s = kwargs.get("old_string")
        _new_s = kwargs.get("new_string")
        if _old_s is None and _new_s is None and (
                final_content is not None or replace_with is not None):
            return ToolOutcome.rejected(
                "SYSTEM INSTRUCTION: operation='edit' takes 'old_string' "
                "(the exact text in the file now) and 'new_string' (its "
                "replacement) — not 'content'/'replace_with'. Re-issue with "
                "those two parameters. Nothing was changed.",
                reason_code="edit_wrong_params")
        _ra = kwargs.get("replace_all")
        _ra = _ra is True or str(_ra or "").strip().lower() in ("true", "1", "yes")
        _ed = await tool_edit_text(target_path, _old_s, _new_s, sandbox_dir,
                                    replace_all=_ra,
                                    project_store=kwargs.get("project_store"))
        if str(_ed).startswith("SUCCESS"):
            try:
                _note_mutation(_get_safe_path(sandbox_dir, target_path))    # §4OG-1 (h): an edit is NOT a read — it invalidates the view
            except Exception:  # noqa: BLE001
                pass
        return _ed
    elif operation == "replace":
        # Accept the param-name variants the model routinely reaches for
        # (the live run's FIRST replace failed purely because it passed
        # `old_text=`/`replace_with=`, and `old_text` was not an alias for
        # `content`). Old text: content/data/text → old_text/old_string/
        # search/old. New text: replace_with → new_text/new_string/new/
        # replacement.
        def _first_set(*vals):
            # §4LL: "" is a value (an empty replacement DELETES the text); the
            # `or` chain read new_string="" as missing and offered to
            # overwrite the whole file with the fragment. A non-empty alias
            # still wins over an empty one (review: replace_with="" beside
            # new_string="Y = 42" deleted the text)
            vs = [v for v in vals if v is not None]
            return next((v for v in vs if v != ""), vs[0] if vs else None)
        _old = (final_content or kwargs.get("old_text") or kwargs.get("old_string")
                or kwargs.get("search") or kwargs.get("old"))
        _new = _first_set(replace_with, kwargs.get("replace_with"), kwargs.get("new_text"),
                          kwargs.get("new_string"), kwargs.get("new"), kwargs.get("replacement"))
        _rp = await tool_replace_text(target_path, _old, _new, sandbox_dir,
                                      project_store=kwargs.get("project_store"))
        if str(_rp).startswith("SUCCESS"):
            try:
                _note_mutation(_get_safe_path(sandbox_dir, target_path))    # §4OG-1 (h): the view is stale now
            except Exception:  # noqa: BLE001
                pass
        return _rp

    if operation == "copy":
        copy_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not copy_target or copy_target == target_path):
            copy_target = rw
        if not copy_target:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for copy operations (must contain the destination filename/path).", reason_code="missing_destination")
        return await tool_copy_file(target_path, copy_target, sandbox_dir)

    if operation in ["rename", "move"]:
        # Hallucination healing: catch intuitive parameter guesses like new_name or target
        rename_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not rename_target or rename_target == target_path):
            rename_target = rw

        if not rename_target:
            return ToolOutcome.rejected("SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for rename/move operations (must contain the new filename/path).")
        return await tool_rename_file(target_path, rename_target, sandbox_dir)
        
    if operation == "delete":
        return await tool_delete_file(target_path, sandbox_dir)
    
    return ToolOutcome.rejected(
        f"Unknown operation: {operation}. Valid operations are: {', '.join(ADVERTISED_OPS)} "
        f"(to change part of a file use 'edit').",
        reason_code="unknown_operation")
