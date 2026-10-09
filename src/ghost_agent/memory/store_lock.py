"""§4MN: one writer process per memory directory.

Chroma (and the other stores) assume a single writer; the contract lived in
repair-script docstrings ("run ONLY with the agent stopped") and nothing
enforced it. Two processes writing the same chroma store, the first one then
SIGTERMed (the deploy path), left it segfaulting on every open in 4 of 7
runs — the agent then crash-loops at boot under KeepAlive with no Python
error line, and the recovery script crashes on the same read.

`acquire_writer_lock(memory_dir)` takes an exclusive, non-blocking `flock` on
`<memory_dir>/store.writer.lock` and keeps it for the life of the process
(released by the OS on exit, crash or kill — no stale lock to clean up). A
second process gets `StoreLockedError`. Re-entrant within one process.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # POSIX only; on a platform without fcntl the lock is a no-op
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

LOCK_NAME = "store.writer.lock"
_HELD: dict = {}


class StoreLockedError(RuntimeError):
    """Another process holds the memory directory's writer lock."""


def _holder(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def acquire_writer_lock(memory_dir, who: str = "") -> Path:
    """Take (or confirm we already hold) the writer lock for ``memory_dir``.
    Raises `StoreLockedError` naming the holder when another process has it."""
    d = Path(memory_dir).resolve()
    key = str(d)
    path = d / LOCK_NAME
    if key in _HELD or fcntl is None:
        return path
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise StoreLockedError(
            f"{d} is being written by another process ({_holder(path)}). Stores are single-writer: "
            f"stop the agent (or the other script) first.") from None
    os.ftruncate(fd, 0)
    os.write(fd, f"pid {os.getpid()} {who}".strip().encode())
    _HELD[key] = fd
    return path


def assert_no_other_writer(memory_dir, who: str = "repair script") -> None:
    """For offline scripts: exit with a clear message unless this process can
    become the memory directory's only writer (i.e. the agent is stopped)."""
    try:
        acquire_writer_lock(memory_dir, who)
    except StoreLockedError as e:
        raise SystemExit(f"refusing to run: {e}")


def release_writer_lock(memory_dir) -> None:
    """Tests only — a running process keeps its lock until it exits."""
    key = str(Path(memory_dir).resolve())
    fd = _HELD.pop(key, None)
    if fd is not None and fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
