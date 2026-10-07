"""Shared helpers for small JSON stores (§4LZ C1).

A store that reads a damaged file as EMPTY and saves on its next write
replaces every row with the one new row — five stores did exactly that. The
policy the playbook, profile and auto-skills stores already follow, in ONE
place: set the unreadable file aside under a timestamped ``.corrupt-*`` name
before starting empty, and fsync before the atomic rename so a power loss
cannot promote a torn or empty file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("GhostAgent")


def preserve_corrupt(path, err, label: str = "", *, is_valid=None) -> None:
    """Move an unreadable store file aside so the next save cannot erase it.

    Re-checks first: another process may have saved a GOOD file since this
    one read the damaged one, and that file must not be set aside. The
    backup name is unique (microseconds + pid), so two set-asides in the
    same second never overwrite each other."""
    p = Path(path)
    try:
        _data = json.loads(p.read_text(encoding="utf-8"))
        if is_valid is not None and not is_valid(_data):
            raise ValueError("parses, but is not this store's shape")
        logger.info("%s is readable again (another writer saved it); not set aside", label or p.name)
        return
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        pass        # still damaged — set it aside
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    backup = p.with_name(p.name + f".corrupt-{ts}-{os.getpid()}")
    try:
        p.replace(backup)
        logger.warning("%s was unreadable (%s); preserved as %s", label or p.name, err, backup)
    except OSError as e:
        logger.error("%s unreadable AND could not be set aside: %s", label or p.name, e)


def write_json_atomic(path, data, *, indent: int = 2, default=None) -> None:
    """unique tmp + fsync + os.replace (a fixed tmp name let two processes
    interleave into one file and publish it torn)."""
    p = Path(path)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=indent, default=default))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def open_append(path, encoding: str = "utf-8"):
    """``open(path, "a")`` for a JSONL ledger — after repairing a TORN tail.

    A process killed mid-append leaves a last line without its newline; the
    next record was then glued onto it and BOTH were lost to every reader
    that skips unparseable lines (§4MD MINOR 2, measured: a SIGKILL lost the
    torn line and the one after it). One newline first, when the file does
    not already end with one, confines the loss to the torn line."""
    p = Path(path)
    try:
        if p.exists() and p.stat().st_size > 0:
            with p.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
            if last != b"\n":
                with p.open("ab") as fh:
                    fh.write(b"\n")
    except OSError:
        pass
    # Path.open, as the writers this replaces used — a test (or caller) that
    # patches builtins.open sees the same behaviour as before
    return p.open("a", encoding=encoding)


_ATOMIC_TMP_RE = re.compile(r".+\.(\d+)\.[0-9a-f]{8}\.tmp$")


def sweep_orphan_temps(root, max_age_s: float = 3600.0) -> int:
    """Remove `write_json_atomic` temp files a killed writer left behind
    (``<name>.<pid>.<hex8>.tmp``, §4MD MINOR 4: 14 left by 15 SIGKILLs):
    only that exact shape, only older than ``max_age_s``, only when the pid
    that wrote it is gone. Returns the count. Never raises."""
    import time
    n = 0
    try:
        now = time.time()
        for p in Path(root).rglob("*.tmp"):
            m = _ATOMIC_TMP_RE.fullmatch(p.name)
            if not m or not p.is_file() or p.is_symlink():
                continue
            try:
                if now - p.stat().st_mtime < max_age_s:
                    continue
                pid = int(m.group(1))
                try:
                    os.kill(pid, 0)
                    continue                      # its writer is still alive
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
                p.unlink()
                n += 1
            except (OSError, OverflowError, ValueError):
                continue                          # one odd file never stops the sweep
    except Exception as e:  # noqa: BLE001
        logger.debug("temp sweep skipped: %s", e)
    return n
