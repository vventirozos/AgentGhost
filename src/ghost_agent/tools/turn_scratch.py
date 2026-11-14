"""§4NY (2026-10-12, operator): "the agent doesn't seem to clean up its own
tests, screenshots etc. look at sandbox/webos/ — I want it to clean up after
itself."

Live: request 1c343374 left seven verification screenshots next to the
`webos/index.html` it built (screenshot.png, calculator_result.png, …); its
reply named none of them.

The rule, deterministic and reversible:

* A file THIS interactive turn created only to check its own work — a
  browser screenshot, or a NEW `file_system` script named `test_*` /
  `debug_*` / `*_test.*` (code extensions only) — is noted when created.
* At the end of the turn a noted file is KEPT when the reply or the user's
  message names it (path or file name, URL-decoded — covers
  `/api/download/<rel>` links and "save it as x.png"). Screenshots are also
  kept when the user asked for a screenshot and the reply links none of
  them (when it links one, that is the one asked for); test scripts when the user
  asked for tests, or when they live under `projects/` (a project's tests
  are part of its codebase). Otherwise the file is MOVED — never deleted —
  to the HOST trash `<system>/scratch_trash/<YYYYMMDD>/<request>/<rel>`.
* Nothing moves at turn end: the late verifier still reads these files
  after the reply (the visual check judges the turn's screenshot; FILE-
  ARTIFACT and the audit's source read the scripts it wrote — review r2
  MAJOR). Each is queued and moved 30 min later, and only if it is
  unmodified (no later turn re-took or edited it) and no later turn named it.
* A path another turn writes again before this turn ends is no longer this
  turn's to move. Trash days older than 7 are purged.

The trash and the pending list live in the HOST data dir, never in the
sandbox: code running in the sandbox could otherwise plant a `.trash`
symlink that turns the move into a write outside the sandbox and the purge
into an `rmtree` of a host folder (review r1 CRITICAL), or list the owner's
files for removal (the §4NS forged-registry lesson).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat as _stat
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("GhostAgent")

TRASH_DIRNAME = "scratch_trash"
LEDGER_NAME = "turn_scratch_pending.json"
DELAY_S = 30 * 60
TEST_DELAY_S = DELAY_S          # (name kept for the tests' readability)
TRASH_KEEP_DAYS = 7
_MAX_OPEN_TURNS = 64

#: a NEW file with one of these names is a throwaway check, not a deliverable
#: (review: tmp_/check_/verify_/probe_/scratch_ and data extensions caught
#: real deliverables — check_site.py, temp_converter.py, test_data.json)
_SCRATCH_NAME_RE = re.compile(
    r"^(?:test|debug)_[\w.\-]*\.(?:py|js|mjs|ts|html?|sh)$"
    r"|^[\w.\-]+_test\.(?:py|js|mjs|ts)$",
    re.IGNORECASE)
_ASKS_FOR_TESTS_RE = re.compile(r"\b(?:tests?|testing|pytest|unit[\s-]?tests?|test\s*suite)\b", re.IGNORECASE)
_ASKS_FOR_SHOT_RE = re.compile(r"\b(?:screenshots?|screen\s*shots?|screen\s*caps?|capture|snapshot)\b",
                               re.IGNORECASE)

_LOCK = threading.Lock()
#: request id → {host path: kind}; filled during the turn
_NOTED: Dict[str, Dict[str, str]] = {}


def is_scratch_test_name(name: str) -> bool:
    return bool(_SCRATCH_NAME_RE.match(str(name or "")))


def _safe_id(req_id) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(req_id or ""))[:16] or "turn"


def note_created(host_path, kind: str, req_id: Optional[str] = None) -> None:
    """Record a file this request is about to write. A path that already
    exists is not created by this call — unless another open turn noted it
    as its check file: then THIS turn re-takes it and owns it from now on
    (verify: dropping it from the first turn left it uncleaned forever)."""
    try:
        p = Path(host_path)
        key = str(p)
        if req_id is None:
            from ..utils.logging import request_id_context
            req_id = str(request_id_context.get() or "")
        with _LOCK:
            if p.exists() or p.is_symlink():
                prior = None
                for rid, files in _NOTED.items():
                    if rid != req_id and key in files:
                        prior = files.pop(key)
                if prior and req_id and not p.is_symlink():
                    _NOTED.setdefault(req_id, {})[key] = prior
                return
            if not req_id:
                return
            _NOTED.setdefault(req_id, {})[key] = kind
            while len(_NOTED) > _MAX_OPEN_TURNS:      # a cancelled turn never finishes
                _NOTED.pop(next(iter(_NOTED)))
        try:    # §4UM round 2/3: a NEW screenshot / check file is the system's, never a project change
            from ..memory.projects import note_system_path
            note_system_path(p)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a tool
        pass


def drop_notes(req_id) -> None:
    with _LOCK:
        _NOTED.pop(str(req_id or ""), None)


def _haystack(texts) -> str:
    raw = "\n".join(str(t or "") for t in texts)
    return (raw + "\n" + urllib.parse.unquote(raw)).lower()


def _named(path: Path, root: Path, hay: str) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    return path.name.lower() in hay or rel.lower() in hay


def _inside(path: Path, root: Path) -> bool:
    try:
        return Path(root).resolve() in Path(path).resolve().parents
    except OSError:
        return False


def _move_to_trash(path: Path, root: Path, trash: Path, req_id: str) -> Optional[Path]:
    """Move one regular file from the sandbox into the host trash; None when
    it was not moved. Re-checks containment right before the move."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if not _stat.S_ISREG(st.st_mode) or not _inside(path, root):
        return None
    try:
        src = Path(path).resolve()
        rel = src.relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return None
    dest = Path(trash) / time.strftime("%Y%m%d") / _safe_id(req_id) / rel
    try:
        from ..memory.projects import note_system_path
        note_system_path(src)       # §4UM round 2: the sweep's move is not the owner's removal
    except Exception:  # noqa: BLE001
        pass
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}.{int(time.time() * 1000)}{dest.suffix}")
        os.replace(src, dest)
        return dest
    except OSError as e:
        logger.info("turn scratch: could not move %s: %s", path, e)
        return None


def purge_old_trash(trash, now: Optional[float] = None) -> int:
    """Delete trash day folders older than TRASH_KEEP_DAYS (host-only dir;
    a symlinked trash or day folder is never followed)."""
    trash = Path(trash)
    try:
        if _stat.S_ISLNK(os.lstat(trash).st_mode) or not trash.is_dir():
            return 0
    except OSError:
        return 0
    now = time.time() if now is None else now
    n = 0
    for d in trash.iterdir():
        if d.is_symlink() or not d.is_dir() or not re.fullmatch(r"\d{8}", d.name):
            continue
        try:
            day = time.mktime(time.strptime(d.name, "%Y%m%d"))
        except ValueError:
            continue
        if now - day > (TRASH_KEEP_DAYS + 1) * 86400:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def _load(ledger: Path) -> list:
    try:
        data = json.loads(Path(ledger).read_text(encoding="utf-8"))
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(ledger: Path, rows: list) -> None:
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_name(ledger.name + ".tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(tmp, ledger)


def finish_turn(req_id: str, reply_text, user_text, root, system_dir,
                now: Optional[float] = None) -> dict:
    """End of an interactive turn: keep what is named or asked for, queue the
    rest for the deferred move; release queued files this turn names.
    Returns {"kept", "queued"} (paths relative to `root`)."""
    out = {"kept": [], "queued": []}
    with _LOCK:
        noted = _NOTED.pop(str(req_id or ""), {})
    if not root or not system_dir:
        return out
    root, system_dir = Path(root), Path(system_dir)
    ledger = system_dir / LEDGER_NAME
    now = time.time() if now is None else now
    hay = _haystack((reply_text, user_text))
    wants_tests = bool(_ASKS_FOR_TESTS_RE.search(str(user_text or "")))
    wants_shot = bool(_ASKS_FOR_SHOT_RE.search(str(user_text or "")))
    queue = []
    # a screenshot the user asked for is the one the reply shows; only when
    # the reply shows none are they all kept (probe: the failed first try
    # stayed next to the linked one)
    if wants_shot and any(k == "screenshot" and _named(Path(ps), root, hay)
                          for ps, k in noted.items() if Path(ps).is_file()):
        wants_shot = False
    for p_str, kind in noted.items():
        p = Path(p_str)
        if not p.is_file() or not _inside(p, root):
            continue
        rel = p.resolve().relative_to(root.resolve()).as_posix()
        if (_named(p, root, hay)
                or (kind == "screenshot" and wants_shot)
                or (kind == "test" and (wants_tests or rel.startswith("projects/")))):
            out["kept"].append(rel)
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        queue.append({"path": str(p), "req": _safe_id(req_id), "kind": kind,
                      "due": now + DELAY_S, "mtime": mtime})
        out["queued"].append(rel)
    with _LOCK:
        rows = _load(ledger)
        # a later turn that NAMES a queued script is using it — release it
        kept_rows = [r for r in rows if not _named(Path(str(r.get("path") or "")), root, hay)]
        if queue or len(kept_rows) != len(rows):
            _save(ledger, (kept_rows + queue)[-500:])
    return out


def sweep_due(root, system_dir, now: Optional[float] = None) -> list:
    """Move queued check files whose delay has passed and that nobody has
    modified since; purge old trash. Returns the moved relative paths."""
    moved = []
    if not root or not system_dir:
        return moved
    root, system_dir = Path(root), Path(system_dir)
    trash, ledger = system_dir / TRASH_DIRNAME, system_dir / LEDGER_NAME
    now = time.time() if now is None else now
    with _LOCK:
        rows = _load(ledger)
        keep = []
        for r in rows:
            if float(r.get("due") or 0) > now:
                keep.append(r)
                continue
            p = Path(str(r.get("path") or ""))
            try:
                unchanged = p.is_file() and abs(p.stat().st_mtime - float(r.get("mtime") or 0)) < 1e-3
            except OSError:
                unchanged = False
            if unchanged and _move_to_trash(p, root, trash, str(r.get("req") or "")):
                try:
                    moved.append(p.resolve().relative_to(root.resolve()).as_posix())
                except (OSError, ValueError):
                    moved.append(p.name)
        if len(keep) != len(rows):
            _save(ledger, keep)
    try:
        purge_old_trash(trash, now)
    except OSError:
        pass
    return moved
