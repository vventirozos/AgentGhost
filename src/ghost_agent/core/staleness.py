"""Source-vs-process divergence guard (§4BN R28-R34).

Lives in its own module for two reasons, both of them defects this
section actually hit:

1. **`main.py` must not be imported from a submodule.** Production
   launches ``python -m src.ghost_agent.main``, so `main` runs as
   ``__main__``; a ``from ..main import …`` inside the watchdog tick
   loaded a SECOND copy of the module and re-executed its 3,144-line
   body inside the live process — R34 found three spurious boot banners
   in the log, printed after "system ready", and the two copies had
   separate baselines and separate dedup state.

2. **The package prefix must be derived, never typed.** R33 keyed
   `sys.modules` membership on a literal ``"ghost_agent."``; production
   modules are ``src.ghost_agent.*``, so the watched set was always
   empty and the auditor returned ``None`` on every tick of every
   production process — dead in the only environment it exists for.
   `utils/component_guard.py` records the same prefix bug leaving five
   idle subsystems inert on the live agent for weeks while their tests
   passed. `__name__` is available; use it.

WHY THIS EXISTS: R28 found the live agent had run pre-§4BN code for a
day, emitting the retracted §4BM framing every ~3h while the corrected
message had never executed. R30 found the same condition nine minutes
after the restart that fixed it. CPython does not reload, so an operator
reading a log line has no way to know the source disagrees — which is
§4BN's own defect class, one level up.

⚠ Do not "simplify" the digest comparison back to mtime, nor the
baseline back to import time. Both were tried; both false-fired in
production (a byte-identical `cp` restore, and lazily-imported modules
whose baseline predated their own load). A false-firing staleness
warning is worse than none — it trains the operator to ignore the one
line that matters.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import time
from typing import Dict, List, Optional

# Every module whose text §4BN's claims depend on. If the running process
# does not match these, an operator reading their output is reading
# something the box is not executing.
PRM_STALENESS_WATCHED = (
    "core/agent.py",            # the dispatch, phase 2.7, both cause helpers
    "core/learning_health.py",  # both wiring views
    "core/feedback.py",         # the fourth-inertness warning (went stale R30)
    "tools/memory.py",          # the TWIN skip log (original R3 drift site)
    "prm/scorer.py",            # the load-bearing "refuses to bootstrap"
    "main.py",                  # the boot warnings
)

# Digest of each watched module as of the first tick that saw it loaded.
_DIGESTS_AT_LOAD: Dict[str, str] = {}

# (relative path, digest-at-report) pairs already announced. Keyed on the
# DIGEST as well as the path (R34 MAJOR-3): keying on path alone meant a
# second, genuinely different divergence of the same file was silent for
# the life of the process — and this section edits `core/agent.py` most
# rounds, so edit → warn → restore → edit again was exactly the R28/R30
# condition, silently.
_REPORTED: set = set()
# §LOG-6b: per-file warn cooldown (seconds) + last-warned clock. See the
# audit function — per-digest dedup alone re-warned on every dev-box save.
_FILE_WARN_COOLDOWN = 3600.0
_FILE_LAST_WARNED: Dict[str, float] = {}


def _package_root() -> str:
    """`ghost_agent` or `src.ghost_agent`, whichever this process used."""
    # __name__ is "<root>.core.staleness" under either import shape.
    return __name__.rsplit(".", 2)[0]


def _module_name(rel: str) -> str:
    return _package_root() + "." + rel[:-3].replace("/", ".")


def loaded_watched_files() -> List[str]:
    """Watched modules actually present in `sys.modules` right now.

    `main.py` is special: under ``-m`` it is ``__main__``, so accept
    either name.
    """
    out = []
    for rel in PRM_STALENESS_WATCHED:
        name = _module_name(rel)
        if name in sys.modules or (rel == "main.py" and "__main__" in sys.modules):
            out.append(rel)
    return out


def read_digests(only=None) -> Dict[str, str]:
    """sha256 of each watched module as it is on disk right now."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for rel in (only if only is not None else PRM_STALENESS_WATCHED):
        try:
            with open(os.path.join(here, rel), "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            continue
    return out


def audit_source_newer_than_process(log) -> Optional[str]:
    """Warn when the code on disk no longer matches what is running.

    `log` is the `pretty_log` callable, injected so this module stays
    free of import cycles.
    """
    loaded = loaded_watched_files()
    if not loaded:
        return None
    current = read_digests(only=loaded)
    for rel, dig in current.items():
        _DIGESTS_AT_LOAD.setdefault(rel, dig)
    stale = [rel for rel in loaded
             if rel in current and current[rel] != _DIGESTS_AT_LOAD.get(rel)]
    fresh = [rel for rel in stale if (rel, current[rel]) not in _REPORTED]
    if not fresh:
        return None
    # §LOG-6b (2026-08-20): per-FILE warn-level cooldown ON TOP OF the
    # per-digest dedup — reconciled with two standing laws this module
    # already pins: (R34) every DISTINCT divergence must still be
    # announced (path-keyed suppression was tried and rejected — the
    # edit/restore/edit cycle hid real staleness), and (R33) nothing may
    # be marked before a SUCCESSFUL emit. So repeats within the window
    # still log and still return — but at INFO: during active development
    # every save mints a new digest, and 95 near-identical WARNINGs in 16
    # days (the #1 warning) trained the operator to ignore the color.
    # Only the FIRST divergence of a file per cooldown is a WARNING.
    _now = time.monotonic()
    _loud = [rel for rel in fresh
             if _now - _FILE_LAST_WARNED.get(rel, -1e9)
             >= _FILE_WARN_COOLDOWN]
    _level = "WARNING" if _loud else "INFO"
    msg = ("the running process no longer matches its source: "
           + ", ".join(fresh)
           + " changed on disk after this process loaded them. CPython does "
             "not reload, so what you are reading is not what is running — "
             "restart before trusting any log line from these modules.")
    # Pass the level only to sinks that accept it (the production lambdas
    # do; the R34 pin's bare `list.append` sink does not) — decided by
    # signature, never by a try/except that could double-emit.
    try:
        _params = inspect.signature(log).parameters.values()
        _takes_level = any(
            p.kind is inspect.Parameter.VAR_KEYWORD or p.name == "level"
            for p in _params)
    except (TypeError, ValueError):
        _takes_level = False
    if _takes_level:
        log(msg, level=_level)
    else:
        log(msg)
    # Marked AFTER emitting: the watchdog tick swallows exceptions, so
    # marking first meant one raising logger silenced the divergence for
    # the life of the process. (§LOG-6b: the warn-cooldown stamp obeys the
    # same law — a raising sink must leave the next audit LOUD.)
    for rel in fresh:
        _REPORTED.add((rel, current[rel]))
    for rel in _loud:
        _FILE_LAST_WARNED[rel] = _now
    return msg
