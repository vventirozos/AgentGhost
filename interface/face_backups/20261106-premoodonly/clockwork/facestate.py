"""Remember which face form the operator last chose, across restarts.

**Why this is not just localStorage.** ``matrix_graph.js`` already persists the
form the ◈ button selects (``localStorage['ghost_face_form']``) and restores it
at module load — and on the handheld that mechanism can never work, for two
independent reasons:

  1. The client's ``QWebEngineView`` uses the default profile, which is
     OFF-THE-RECORD: there is no ``~/.local/share/**/QtWebEngine`` directory on
     the device, so web storage is discarded when the process exits.
  2. localStorage is keyed by ORIGIN, and the face is served from
     ``http://127.0.0.1:<free port>`` — a different port, hence a different
     origin, on every boot. Even a persistent profile would look the choice up
     under a key it had never written.

So the memory lives here instead: one line of text in a file the Python side
owns, written when the form changes and read back at startup. Nothing in this
module imports Qt, which is what makes it testable off the device.

**Precedence at startup:** ``GHOST_FACE_FORM`` (an explicit operator override)
> the remembered form > :data:`FALLBACK_FORM`.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Used when nothing has been remembered yet — a fresh device, or the first boot
# after this file was added. Operator pick, 2026-08-03.
FALLBACK_FORM = "descent"

STATE_PATH = os.environ.get("GHOST_FACE_STATE", "~/.ghost_face_form")

_FORMS_RE = re.compile(r"const FORMS = \[(.*?)\];", re.S)
_NAME_RE = re.compile(r"'([a-z]+)'")
# A form name is a bare lowercase word — bounded, because "lowercase letters"
# alone still admits a megabyte of them from a corrupted file. Anything else is
# treated as corruption rather than fed to the face.
_VALID_RE = re.compile(r"^[a-z]{1,24}$")


def _path(path=None) -> Path:
    return Path(os.path.expanduser(path or STATE_PATH))


def known_forms(face_dir) -> list:
    """The face's own FORMS list, parsed from the matrix_graph.js beside it.

    Empty list when it cannot be read — callers treat that as "cannot
    validate" and fall through, rather than rejecting a form that is probably
    fine. The alternative (hardcoding the list here) is a second copy that
    would drift the first time a form is added.
    """
    try:
        src = (Path(face_dir) / "matrix_graph.js").read_text()
    except OSError:
        return []
    m = _FORMS_RE.search(src)
    return _NAME_RE.findall(m.group(1)) if m else []


def load_form(face_dir=None, path=None):
    """The remembered form, or None if there is nothing usable to remember.

    Never raises: a missing, unreadable or corrupt state file just means "no
    memory", and the caller falls back to the default.
    """
    try:
        raw = _path(path).read_text().strip()
    except OSError:
        return None
    if not raw or not _VALID_RE.match(raw):
        return None
    forms = known_forms(face_dir) if face_dir else []
    # Only reject when we actually managed to read the list — a form dropped
    # from FORMS would otherwise leave the face on the JS default with nothing
    # said, which is the failure this check exists to prevent.
    if forms and raw not in forms:
        return None
    return raw


def save_form(name, path=None) -> bool:
    """Remember `name`. Returns whether it was written.

    Atomic (temp file + replace) so a power cut mid-write — which on a handheld
    is just "the battery ran out" — cannot leave a truncated file that reads as
    corruption on the next boot.
    """
    if not name or not _VALID_RE.match(str(name)):
        return False
    target = _path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".face-")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(f"{name}\n")
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        return False
    return True


def startup_form(face_dir=None, path=None, env=None) -> str:
    """Which form to open on: explicit override > remembered > fallback."""
    env = os.environ if env is None else env
    forced = (env.get("GHOST_FACE_FORM") or "").strip()
    if forced:
        return forced
    return load_form(face_dir=face_dir, path=path) or FALLBACK_FORM
