"""§4UY C6/C21 — the previous version of a workspace file is kept before a tool overwrites or deletes it.

One rule for every writer (§4UX C6: `file_system` refused to overwrite or delete a file the request had not
read, while `execute` — `>`, heredocs, `sed -i`, `rm`, `execute(filename=, content=)` — did not; and nothing
kept an old version, C21). The refusal was the over-restriction (§4UI) and its delete form asked the user for
a yes (§4UK). Now no ordinary coding step is refused: whatever replaces or removes an existing file first
copies it to the version store, and the result says so. Only a wipe of a whole project folder is still
refused (`shell_analysis._bulk_destructive`).

WHERE (§4UY review r1, MAJOR): the store lives OUTSIDE the sandbox bind mount, under
``<GHOST_HOME>/system/file_versions/<workspace key>/<workspace-relative path>@<UTC stamp>``. Inside the workspace a model
could plant a symlink in the store (or at a version's name) and turn "keep a version" into a write anywhere on
the host. Every destination is resolved and checked to stay inside the store, files are created with
O_EXCL|O_NOFOLLOW, and a source that is a symlink is kept AS a symlink — a source link is never followed (to
the host or anywhere). The model lists and restores versions through `file_system` (`versions`, `restore`).

What is versioned: what a TOOL overwrites or deletes — `file_system` write/replace/download/delete and the
paths a shell command names (`rm`, `>`, `sed -i`, `cp` onto, …). A program writing a file itself (python
`open(…, 'w')`) is not.

Bounds: a file larger than ``MAX_FILE_BYTES`` is not copied; one call keeps at most ``MAX_FILES_PER_CALL``
files / ``MAX_BYTES_PER_CALL`` bytes and STOPS walking when that budget is spent ("kept N of M"); a walk skips
generated folders (``SKIP_DIRS``: dist, build, target, .next, node_modules, venvs, caches); each file keeps its
newest ``KEEP_PER_FILE`` versions; versions older than ``MAX_AGE_S`` (14 days, operator decision) expire, and
the whole store is held under ``MAX_STORE_BYTES`` oldest-first. Never raises: a version that cannot be kept is
reported, never a reason to block the write.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

STORE_REL = Path("system") / "file_versions"
KEEP_PER_FILE = 5
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES_PER_CALL = 200
MAX_BYTES_PER_CALL = 50 * 1024 * 1024
MAX_AGE_S = 14 * 86400
MAX_STORE_BYTES = 500 * 1024 * 1024
#: how often (seconds) a keep sweeps the store for expired versions and the total cap
SWEEP_EVERY_S = 3600.0
#: never versioned: the store itself, version control, the services registry
_SKIP_PARTS = frozenset({".ghost_versions", ".git", ".services"})
#: a WALK (a deleted folder) does not descend into generated folders — their content is rebuilt, not written
SKIP_DIRS = frozenset({"node_modules", "dist", "build", "target", ".next", ".nuxt", "out", "__pycache__",
                       ".venv", "venv", "env", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
                       ".cache", ".gradle", "coverage", ".parcel-cache", ".turbo"})
_last_sweep = 0.0


def store_root() -> Path:
    return _home() / STORE_REL


def _store(root) -> Path:
    """One store per workspace root (keyed by its real path), so two sandboxes never share a version id."""
    import hashlib
    key = hashlib.sha1(os.path.realpath(root).encode("utf-8", "surrogateescape")).hexdigest()[:12]
    return store_root() / key


def _stamp() -> str:
    t = time.time()
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(t)) + f".{int((t % 1) * 1e6):06d}Z"


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def _relative(path: Path, root: Path) -> Optional[Path]:
    """The workspace-relative path of ``path`` — its PARENT resolved (a symlinked folder is judged by where it
    really is), the entry itself not (a link is versioned as a link). None when outside ``root`` or skipped."""
    try:
        real_root = os.path.realpath(root)
        parent = os.path.realpath(path.parent)
    except (OSError, ValueError):
        return None
    if not _inside(parent, real_root) or path.name in ("", ".", ".."):
        return None
    rel = Path(os.path.relpath(os.path.join(parent, path.name), real_root))
    if not rel.parts or rel.parts[0] == ".." or any(p in _SKIP_PARTS for p in rel.parts):
        return None
    return rel


def _home() -> Path:
    home = (os.environ.get("GHOST_HOME") or "").strip()
    return Path(home) if home else Path.home() / "Data" / "AI" / "Data"


def _real_store(store: Path) -> Optional[str]:
    """The real path of ``store`` when no component below GHOST_HOME is a link (else None): every READ of
    the store (list, restore, prune, sweep) checks this, so a planted link never reads or unlinks outside."""
    base = _home()
    try:
        below = store.relative_to(base)
        want = os.path.join(os.path.realpath(base), *below.parts)
        return want if os.path.realpath(store) == want else None
    except (ValueError, OSError):
        return None


def _safe_dir(store: Path, rel_parent: Path) -> Optional[Path]:
    """Create ``store/rel_parent`` one component at a time FROM GHOST_HOME DOWN (`system`, `file_versions`,
    the workspace key, then the file's folders), refusing any component that is a symlink or not a directory,
    and verify the result resolves inside the store. None when unsafe. (GHOST_HOME itself may be a link.)"""
    try:
        base = _home()
        try:
            below = store.relative_to(base)
        except ValueError:
            return None
        if not base.is_dir():
            return None
        real_base = os.path.realpath(base)
        cur = base
        for part in list(below.parts) + list(rel_parent.parts):
            if part in ("", ".", ".."):
                return None
            cur = cur / part
            try:
                st = os.lstat(cur)
                if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                    return None
            except FileNotFoundError:
                os.mkdir(cur, 0o700)
        real_store = os.path.join(real_base, *below.parts)
        if os.path.realpath(store) != real_store or not _inside(os.path.realpath(cur), real_store):
            return None
        return cur
    except OSError:
        return None


def _copy_one(src: Path, dest: Path) -> int:
    """Copy ``src`` (never following it if it is a link) to a NEW ``dest``. Returns bytes copied; raises
    OSError on failure."""
    st = os.lstat(src)
    if stat.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(src), dest)          # a link is kept as a link: its target is never read
        return 0
    if not stat.S_ISREG(st.st_mode):
        raise OSError(errno.EINVAL, "not a regular file")
    flags_r = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags_w = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fi = os.open(src, flags_r)
    try:
        fo = os.open(dest, flags_w, 0o600)
        n = 0
        try:
            while True:
                buf = os.read(fi, 1 << 20)
                if not buf:
                    break
                os.write(fo, buf)
                n += len(buf)
        except BaseException:
            os.close(fo)
            try:
                os.unlink(dest)
            except OSError:
                pass
            raise
        os.close(fo)
        try:
            os.utime(dest, (st.st_atime, st.st_mtime), follow_symlinks=False)
        except (OSError, NotImplementedError):
            pass
        return n
    finally:
        os.close(fi)


def _prune(folder: Path, name: str) -> None:
    store = store_root()
    try:
        real = _real_store(store) if store.exists() else None
        if real is None or folder.is_symlink() or not _inside(os.path.realpath(folder), real):
            return
        olds = sorted(p for p in folder.iterdir() if p.name.startswith(name + "@"))
    except OSError:
        return
    for p in olds[:-KEEP_PER_FILE]:
        try:
            os.unlink(p)                                # unlink never follows a link
        except OSError:
            pass


def sweep(now: Optional[float] = None, force: bool = False) -> int:
    """Expire versions older than MAX_AGE_S and hold the store under MAX_STORE_BYTES (oldest first). Returns
    how many were removed. Runs at most once per SWEEP_EVERY_S unless ``force``."""
    global _last_sweep
    now = time.time() if now is None else now
    if not force and now - _last_sweep < SWEEP_EVERY_S:
        return 0
    _last_sweep = now
    store = store_root()
    if not store.is_dir() or _real_store(store) is None:
        return 0
    entries = []
    for d, dirs, names in os.walk(store, followlinks=False):
        for n in names + [x for x in dirs if os.path.islink(os.path.join(d, x))]:
            p = os.path.join(d, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            entries.append((st.st_mtime if "@" not in n else _stamp_time(n, st.st_mtime), st.st_size, p))
    removed = 0
    keep = []
    for t, size, p in entries:
        if now - t > MAX_AGE_S:
            try:
                os.unlink(p)
                removed += 1
            except OSError:
                pass
        else:
            keep.append((t, size, p))
    total = sum(s for _, s, _ in keep)
    for t, size, p in sorted(keep):
        if total <= MAX_STORE_BYTES:
            break
        try:
            os.unlink(p)
            removed += 1
            total -= size
        except OSError:
            pass
    return removed


def _stamp_time(name: str, default: float) -> float:
    try:
        s = name.rsplit("@", 1)[1].rstrip("Z")
        base, frac = (s.split(".", 1) + ["0"])[:2]
        import calendar
        return calendar.timegm(time.strptime(base, "%Y%m%dT%H%M%S")) + float("0." + frac)
    except (IndexError, ValueError):
        return default


def _carry_outside_row(src, dst) -> None:
    """§4UY N6 (seam review): the outside-file ledger row of ``src`` follows its bytes to ``dst`` — a version
    kept of a downloaded file, and that version restored (`provenance.carry_outside_rows`). Never raises."""
    try:
        from ..utils.provenance import carry_outside_rows
        carry_outside_rows(src, dst)
    except Exception:  # noqa: BLE001
        pass


def keep_previous(paths: Iterable, root) -> Tuple[List[str], List[str]]:
    """Version each existing file in ``paths`` (a directory: its files, generated folders skipped) under the
    store. An item may be ``(path, match)``: a walk then keeps only the entries ``match(rel, is_dir)`` accepts
    (a matched folder: all of it) — a narrowed `find … -name '*.py' -delete` (§4UY r2 m4). Returns (kept, skipped): kept = version ids ("<rel>@<stamp>"); skipped = "<file>: why" lines, the
    last one "kept N of M …" when the budget stopped the walk."""
    kept: List[str] = []
    skipped: List[str] = []
    try:
        root = Path(root)
        if not root.is_dir():
            return kept, skipped
    except (TypeError, OSError):
        return kept, skipped
    try:
        sweep()
    except Exception as e:  # noqa: BLE001
        logger.debug("file_versions sweep: %s", e)
    store = _store(root)
    budget_files, budget_bytes = MAX_FILES_PER_CALL, MAX_BYTES_PER_CALL
    seen = 0
    spent = False
    stamp = _stamp()

    def _one(p: Path) -> None:
        nonlocal budget_files, budget_bytes, seen, spent
        rel = _relative(p, root)
        if rel is None:
            return
        try:
            st = os.lstat(p)
        except OSError:
            return
        is_link = stat.S_ISLNK(st.st_mode)
        if not is_link and (not stat.S_ISREG(st.st_mode) or st.st_size == 0):
            return                                    # an empty file loses nothing; a FIFO is never read
        seen += 1
        size = 0 if is_link else st.st_size
        if size > MAX_FILE_BYTES:
            skipped.append(f"{rel.as_posix()}: {size:,} bytes, over the {MAX_FILE_BYTES // (1024 * 1024)} MB "
                           "version limit")
            return
        if budget_files <= 0 or size > budget_bytes:
            spent = True
            return
        folder = _safe_dir(store, rel.parent)
        if folder is None:
            skipped.append(f"{rel.as_posix()}: the version folder is not safe to write")
            return
        dest = folder / f"{rel.name}@{stamp}"
        real = _real_store(store)
        if real is None or not _inside(os.path.realpath(dest.parent), real):
            skipped.append(f"{rel.as_posix()}: the version folder is not safe to write")
            return
        try:
            _copy_one(p, dest)
        except OSError as e:
            skipped.append(f"{rel.as_posix()}: {type(e).__name__}")
            return
        _carry_outside_row(p, dest)                     # §4UY N6: a kept version of outside bytes is outside
        budget_files -= 1
        budget_bytes -= size
        kept.append(f"{rel.as_posix()}@{stamp}")
        _prune(folder, rel.name)

    for raw in paths or ():
        if spent:
            break
        match = None
        if isinstance(raw, tuple):
            raw, match = raw                          # §4UY r2 m4: a narrowed find — only what it can match
        try:
            p = Path(raw)
            if not p.is_absolute():
                p = root / p
            if p.is_dir() and not p.is_symlink():
                if p.name in SKIP_DIRS:
                    continue                          # `rm -rf node_modules build`: rebuilt, never written
                taken: List[str] = []                 # folders the filter matched: everything under them goes
                for d, dirs, names in os.walk(p, followlinks=False):
                    dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and x not in _SKIP_PARTS
                                     and not os.path.islink(os.path.join(d, x)))
                    links = sorted(x for x in os.listdir(d) if os.path.islink(os.path.join(d, x))
                                   and x not in names)
                    drel = os.path.relpath(d, p)
                    drel = "" if drel == "." else drel
                    whole = match is None or any(d == t or d.startswith(t + os.sep) for t in taken)
                    if not whole:
                        taken += [os.path.join(d, x) for x in dirs if match(os.path.join(drel, x), True)]
                    for n in sorted(names) + links:
                        if not whole and not match(os.path.join(drel, n), False):
                            continue
                        _one(Path(d) / n)
                        if spent:
                            break
                    if spent:
                        break
            elif os.path.lexists(p) and (match is None or match("", False)):
                _one(p)
        except Exception as e:  # noqa: BLE001 — a version is never a reason to fail the write
            logger.debug("file_versions: %s not kept: %s", raw, e)
    if spent:
        skipped.append(f"kept {len(kept)} of at least {seen} files — the walk stopped at this call's version "
                       f"budget ({MAX_FILES_PER_CALL} files / {MAX_BYTES_PER_CALL // (1024 * 1024)} MB)")
    return kept, skipped


def versions_note(kept: List[str], skipped: List[str]) -> str:
    """The one sentence a result carries about kept versions ("" when there is nothing to say)."""
    if not kept and not skipped:
        return ""
    bits = []
    if kept:
        shown = ", ".join(kept[:3]) + (f" (+{len(kept) - 3} more)" if len(kept) > 3 else "")
        bits.append(f"previous version kept: {shown} — file_system(operation='restore', path=…) brings it back")
    if skipped:
        bits.append("NOT kept: " + "; ".join(skipped[:3]) + (" …" if len(skipped) > 3 else ""))
    return "[" + "; ".join(bits) + "]"


def _version_folder(path, root) -> Tuple[Optional[Path], Optional[Path]]:
    root = Path(root)
    p = Path(path)
    rel = _relative(p if p.is_absolute() else root / p, root)
    if rel is None:
        return None, None
    store = _store(root)
    folder = store / rel.parent
    try:
        real = _real_store(store)
        if (real is None or not folder.is_dir() or folder.is_symlink()
                or not _inside(os.path.realpath(folder), real)):
            return rel, None
    except OSError:
        return rel, None
    return rel, folder


def list_versions(path, root) -> List[str]:
    """The kept versions of ``path`` (workspace-relative ids "<rel>@<stamp>"), oldest first."""
    try:
        rel, folder = _version_folder(path, root)
        if folder is None:
            return []
        return sorted(f"{(rel.parent / p.name).as_posix()}" for p in folder.iterdir()
                      if p.name.startswith(rel.name + "@"))
    except OSError:
        return []


def version_path(version_id: str, root) -> Optional[Path]:
    """The store path of a version id of the workspace ``root``, verified inside the store (None otherwise)."""
    store = _store(root)
    v = str(version_id or "").strip().lstrip("/")
    if "@" not in v:
        return None
    p = store / v
    try:
        real = _real_store(store)
        if real is None or not _inside(os.path.realpath(p.parent), real) or not os.path.lexists(p):
            return None
    except OSError:
        return None
    return p


def restore_version(path, root, version: str = "") -> Tuple[bool, str]:
    """Put a kept version of ``path`` back (the newest, or ``version`` — a stamp or a full id). The file it
    replaces is kept as a version first. Returns (ok, message)."""
    root = Path(root)
    vs = list_versions(path, root)
    if not vs:
        return False, f"no kept versions of '{path}'"
    want = str(version or "").strip()
    pick = vs[-1] if not want else next((v for v in reversed(vs) if v == want or v.endswith("@" + want)
                                         or v.rsplit("@", 1)[1].startswith(want)), None)
    if pick is None:
        return False, f"no version '{want}' of '{path}' — kept: " + ", ".join(v.rsplit("@", 1)[1] for v in vs)
    src = version_path(pick, root)
    rel = _relative(Path(path) if Path(path).is_absolute() else root / path, root)
    if src is None or rel is None:
        return False, f"version '{pick}' is not readable"
    target = Path(os.path.realpath(root)) / rel      # rel came from _relative: its folder is inside root
    target.parent.mkdir(parents=True, exist_ok=True)
    kept, _sk = keep_previous([target], root) if os.path.lexists(target) else ([], [])
    tmp = target.parent / f".{target.name}.restore-{os.getpid()}"
    try:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        _copy_one(src, tmp)
        os.replace(tmp, target)
        _carry_outside_row(src, target)                 # §4UY N6: restored outside bytes keep their row
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, f"restore failed: {type(e).__name__}: {e}"
    return True, (f"restored '{rel.as_posix()}' from version {pick.rsplit('@', 1)[1]}"
                  + (f" (the content it replaced is kept as {kept[0]})" if kept else ""))
