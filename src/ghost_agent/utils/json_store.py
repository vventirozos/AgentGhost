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
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("GhostAgent")


def preserve_corrupt(path, err, label: str = "") -> None:
    """Move an unreadable store file aside so the next save cannot erase it.

    Re-checks first: another process may have saved a GOOD file since this
    one read the damaged one, and that file must not be set aside. The
    backup name is unique (microseconds + pid), so two set-asides in the
    same second never overwrite each other."""
    p = Path(path)
    try:
        json.loads(p.read_text(encoding="utf-8"))
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
