"""ONE implementation of the host-side rules for numbers the sandbox can
write (§4GI, 2026-09-13).

Both registries — ``sandbox/jobs.py`` (``/workspace/.jobs/registry.json``)
and ``sandbox/services.py`` (``/workspace/.services/registry.json``) —
live on the bind mount, so the very processes they supervise can write
them. Every pid, port and name read from them is attacker input. The rules
that make that safe were written once for jobs in §4DX (pid floor, row
validation, an INCONCLUSIVE probe state, a dash-safe process-tree kill)
and never reached the services twin: a planted ``{"web": {"pid": 1, "port":
8100}}`` row made ``stop web`` run ``kill -TERM -- -1 || kill -TERM 1`` —
the ``--`` form is a dash syntax error, so the fallback TERMed docker-init
and the whole container with it. Two files, each naming the other as "the
unsafe twin", each hardened in a different round. This module is the
single home; ``tests/test_registry_guard_4gi.py`` enumerates both files
from the AST and fails if either grows its own copy of any rule.
"""
from __future__ import annotations

import re
import shlex
from typing import Callable, Optional, Tuple

#: Linux ``pid_max`` ceiling (4194304 on 64-bit); anything at or above is
#: not a pid the container can hold.
PID_MAX = 4194304

#: A registry name: letters, digits, ``_`` and ``-``, no leading digit, no
#: ``--`` (which would read as an option to a shell tool), ≤ 32 chars.
NAME_RE = re.compile(r"^[A-Za-z](?!.*--)[A-Za-z0-9_-]{0,31}$")

Exec = Callable[..., Tuple[str, int]]

#: `kill_tree_script` prints exactly one of these as its last line. A caller
#: that sees NEITHER (a stub, a truncated read, an exec that never ran the
#: tail) must treat the outcome as unknown, not as survival.
SURVIVED_MARKER = "GHOST_KILL_SURVIVED"
KILLED_MARKER = "GHOST_KILL_DONE"


def valid_pid(pid) -> Optional[int]:
    """The pid as an int when it is a signal-safe target, else None.

    ⚠ THE FLOOR IS A SAFETY BOUNDARY. In POSIX ``kill -- -1`` means "every
    process the caller may signal", and ``-0`` is the caller's own group;
    pid 1 is docker-init. A pid the container cannot hold is a forged row.
    """
    if isinstance(pid, bool):
        return None
    # ⚠ A FRACTIONAL VALUE IS A MALFORMED ROW, NOT A PID (§4GK round 4).
    # `int(2.9)` is 2, so a tampered or corrupted `pid: 2.9` used to build a
    # real kill for pid 2. Anything that is not an exact integer is refused
    # rather than rounded into a neighbour's process.
    #
    # ⚠ AND IT MUST REFUSE, NOT RAISE (§4GK round 5). The first version wrote
    # the comparison OUTSIDE the try below, where `int(nan)` raises ValueError
    # and `int(inf)` raises OverflowError — and `json.loads` accepts both
    # spellings by default, on a registry that lives on the bind mount the
    # sandboxed process can write. One planted `{"pid": NaN}` row escaped
    # `_validated_rows`, hit `_load`'s outer `except`, and returned an EMPTY
    # map: every live service row vanished from the live map AND from the
    # quarantine, so the next save destroyed them permanently. That is the
    # exact denial the quarantine exists to prevent, caused by the guard.
    try:
        if isinstance(pid, float) and pid != int(pid):
            return None
        p = int(pid)
    except (TypeError, ValueError, OverflowError):
        return None
    if p <= 1 or p >= PID_MAX:
        return None
    return p


def valid_port(port, lo: int = 1, hi: int = 65535) -> Optional[int]:
    """The port as an int within ``[lo, hi]``, else None.

    ⚠ NON-INTEGRAL IS MALFORMED, NOT ROUNDABLE — the same rule `valid_pid`
    got, applied to its sibling one round later (§4GK round 5). `8100.9` used
    to truncate to `8100` and the row KEPT the float, which is invisible to
    the allocator's `isinstance(e.get("port"), int)` dead-claim filter while
    still comparing unequal to the int port everywhere else: a claim nothing
    can reclaim and nothing can see.
    """
    if isinstance(port, bool):
        return None
    try:
        if isinstance(port, float) and port != int(port):
            return None
        p = int(port)
    except (TypeError, ValueError, OverflowError):
        return None
    if p < lo or p > hi:
        return None
    return p


def valid_name(name) -> bool:
    return isinstance(name, str) and bool(NAME_RE.match(name))


def probe_inconclusive(out, code) -> bool:
    """True when a probe's non-zero exit says "the PROBE failed", not "the
    process is gone".

    Two shapes, both measured: ``sandbox.execute`` reports an infra fault
    (wedged daemon, container restart, provision backoff) as exit 1 with an
    ``[SANDBOX INFRA ERROR]`` body — and it wraps every probe in its own
    ``timeout -k 5s 15s``, whose expiry surfaces as exit 124 with the generic
    ``[SYSTEM ERROR]: Process failed`` line and NO infra marker. Reading
    either as death is how a live job got exit 137 with its log deleted, and
    how a service registry dropped a live row and re-issued its port.
    """
    text = str(out or "")
    return ("SANDBOX INFRA ERROR" in text
            or int(code or 0) in (124, 137, 143))


def pid_state_cmd(pid: int) -> str:
    """Liveness probe: ``kill -0`` AND not a zombie. State is the first field
    after the LAST ``)`` in ``/proc/<pid>/stat`` (comm may contain spaces
    and parens)."""
    return (f"sh -c 'kill -0 {int(pid)} 2>/dev/null && "
            f"[ \"$(sed \"s/^.*) //\" /proc/{int(pid)}/stat 2>/dev/null "
            f"| cut -d\" \" -f1)\" != Z ]'")


def pid_state(exec_fn: Exec, pid, *, timeout: int = 15) -> Optional[bool]:
    """``True`` alive, ``False`` dead (or not a valid target — nothing to
    signal), ``None`` when the probe itself failed (infra fault, probe
    timeout). Anything that would KILL, DISCARD a row, or RE-ISSUE a port
    on the answer must handle ``None`` as "unknown", never as "dead"."""
    p = valid_pid(pid)
    if p is None:
        return False
    out, code = exec_fn(pid_state_cmd(p), timeout=timeout)
    if code != 0 and probe_inconclusive(out, code):
        return None
    return code == 0


def kill_tree_script(pid: int) -> str:
    """The shell that TERMs, then KILLs, a supervised process's whole tree:
    its process group (``setsid`` made the recorded pid the leader), its
    session, and every ``/proc`` descendant by ppid ancestry (bounded depth
    32 so a hostile ``/proc`` cannot hang it), with a plain-pid fallback.

    ⚠ NO ``--`` BEFORE THE NEGATIVE PID. The container's ``/bin/sh`` is
    dash, whose builtin ``kill`` consumes ``-TERM`` and then parses ``--`` as
    a pid: ``kill -TERM -- -123`` fails with "Illegal number: -" and sends
    NOTHING (measured in python:3.11-slim-bookworm). ``kill -TERM -123``
    signals the group. Verbatim the §4DX jobs script (its executed pins
    in tests/test_sandbox_job_promotion.py spell it); a recorder sees the
    signal in the ``sig TERM`` / ``sig KILL`` calls and the target in
    ``S=<pid>;``.
    """
    # The guard runs FIRST (§4GK round 4): a bare `int(pid)` ahead of it
    # raised TypeError for None or a list, not the ValueError this function
    # documents, so callers guarding on ValueError did not catch it.
    p = valid_pid(pid)
    if p is None:
        raise ValueError(f"refusing to build a kill for pid {pid!r}")
    return (
        f'S={p}; '
        f'mine() {{ q=$1; d=0; while [ "$q" -gt 1 ] 2>/dev/null; do '
        f'[ "$q" = "$S" ] && return 0; '
        f'd=$((d+1)); [ "$d" -gt 32 ] && return 1; '
        f'st=$(sed "s/^.*) //" "/proc/$q/stat" 2>/dev/null); '
        f'[ -n "$st" ] || return 1; '
        f'sd=$(echo "$st" | cut -d" " -f4); '
        f'[ "$sd" = "$S" ] && return 0; '
        f'q=$(echo "$st" | cut -d" " -f2); done; return 1; }}; '
        f'sig() {{ kill -"$1" -$S 2>/dev/null || '
        f'kill -"$1" $S 2>/dev/null; '
        f'for f in /proc/[0-9]*; do p=${{f##*/}}; '
        f'case "$p" in *[!0-9]*) continue;; esac; '
        f'[ "$p" -gt 1 ] || continue; '
        f'mine "$p" && kill -"$1" "$p" 2>/dev/null; '
        f'done; true; }}; '
        # Poll for death instead of a flat `sleep 2`: a cancel or a TTL
        # expiry blocked a worker thread for a guaranteed two seconds
        # even when the tree died on the first TERM.
        f'sig TERM; i=0; while [ $i -lt 20 ]; do '
        f'kill -0 $S 2>/dev/null || break; '
        f'sleep 0.1; i=$((i+1)); done; '
        # ⚠ THE SCRIPT REPORTS WHETHER IT WORKED (§4GK round 5). `kill_tree`
        # used to answer "a signal was sent", and services.py believed it —
        # unlinking the pidfile and dropping the row over a process that was
        # still running and still holding its port. The verdict comes from
        # the SAME shell that did the killing, one exec, with no window for
        # the process to die (or be replaced) between a kill and a separate
        # probe. Absence of a marker is INCONCLUSIVE, never "it survived":
        # refusing to release a row on a probe that could not answer strands
        # the port forever.
        # ⚠ THE VERDICT MUST OUTLAST THE REAPER, AND A ZOMBIE IS NOT ALIVE
        # (§4GK round 6). The first version asked `kill -0` with NO wait after
        # the KILL and no zombie test — and `kill -0` succeeds on a zombie.
        # Measured 6/6: a TERM-ignoring orphan reparented to init (the ordinary
        # shape of a container service under an init shim) printed SURVIVED
        # while `ps` showed the pid gone 0.2s later. Round 5 made that verdict
        # consequential, so a stop that WORKED answered "did NOT stop", put the
        # registry row back and kept the port claimed. The TERM phase already
        # polls for death; the KILL phase gets the same courtesy, and the state
        # test is the one this module's own `pid_state_cmd` uses.
        f'sig KILL; j=0; while [ $j -lt 20 ]; do '
        f'kill -0 $S 2>/dev/null || break; '
        f'st=$(sed "s/^.*) //" "/proc/$S/stat" 2>/dev/null | cut -d" " -f1); '
        f'[ "$st" = Z ] && break; '
        f'sleep 0.1; j=$((j+1)); done; '
        f'if kill -0 $S 2>/dev/null && '
        f'[ "$(sed "s/^.*) //" "/proc/$S/stat" 2>/dev/null | cut -d" " -f1)" != Z ]; '
        f'then echo {SURVIVED_MARKER}; else echo {KILLED_MARKER}; fi; true'
    )


def kill_tree(exec_fn: Exec, pid, *, log=None, timeout: int = 30) -> bool:
    """Run :func:`kill_tree_script` for ``pid`` and report whether the tree is
    actually GONE. ``False`` means nothing was sent, the exec itself failed,
    or the process is still alive afterwards.

    ⚠ IT USED TO ANSWER "A SIGNAL WAS SENT" (§4GK round 4). The exec result
    was discarded and the return was an unconditional ``True`` once the pid
    passed the floor — so an exec that exited non-zero, or a process that
    simply survived TERM+KILL, both read as a successful stop. `jobs.py`
    re-probed and warned; `services.py` did not, so `manage_services stop`
    reported success, unlinked the pidfile and dropped the registry row while
    the process kept running and holding its port. That is the remaining half
    of the jobs/services divergence this module exists to close: the SCRIPT
    was shared, the VERIFICATION was not. An inconclusive probe (an infra
    fault, not an answer) counts as gone, because refusing to release the row
    on a probe failure strands the port forever — but it is logged.
    """
    p = valid_pid(pid)
    if p is None:
        if log is not None:
            log(f"refusing to signal pid {pid!r} — 'kill -{pid}' would target "
                f"the whole container, not one process (registry row is "
                f"malformed or tampered with)")
        return False
    try:
        _out, _code = exec_fn(f"sh -c {shlex.quote(kill_tree_script(p))}",
                              timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — an exec that raises sent nothing
        if log is not None:
            log(f"kill for pid {p} could not be issued: "
                f"{type(exc).__name__}: {exc}")
        return False
    if _code != 0 and not probe_inconclusive(_out, _code):
        if log is not None:
            log(f"kill for pid {p} exited {_code} — nothing was signalled")
        return False
    _text = _out.decode("utf-8", "replace") if isinstance(_out, bytes) else str(_out or "")
    if SURVIVED_MARKER in _text:
        if log is not None:
            log(f"pid {p} SURVIVED TERM+KILL — it still holds its port and "
                f"files; its row and pidfile must not be dropped as stopped")
        return False
    if KILLED_MARKER not in _text and log is not None:
        log(f"pid {p} was signalled but the kill script printed no verdict "
            f"— treating it as stopped (a row held on an unanswerable probe "
            f"strands the port forever)")
    return True


def validate_row(entry, *, require_pid: bool, port_range=None) -> Optional[str]:
    """Why a registry row must be dropped, or ``None`` when it is safe.

    ``pid`` must be a valid target (``None`` allowed only when
    ``require_pid`` is False — a service row exists before its launcher
    reports a pid); ``port``, when present, an int in ``port_range`` (a
    ``(lo, hi)`` pair, default the whole TCP range); ``name``, when present,
    must match :data:`NAME_RE`; ``container_id``, when present, a short
    string.
    """
    if not isinstance(entry, dict):
        return "not an object"
    pid = entry.get("pid")
    if pid is None:
        if require_pid:
            return "no pid"
    elif valid_pid(pid) is None:
        return f"pid {pid!r} is not a signal-safe target"
    port = entry.get("port")
    if port is not None:
        lo, hi = port_range or (1, 65535)
        if valid_port(port, lo, hi) is None:
            return f"port {port!r} outside {lo}-{hi}"
    name = entry.get("name")
    if name is not None and not valid_name(name):
        return f"name {name!r} malformed"
    cid = entry.get("container_id")
    if cid is not None and not (isinstance(cid, str) and len(cid) <= 128):
        return "container_id malformed"
    return None


__all__ = [
    "PID_MAX", "NAME_RE", "valid_pid", "valid_port", "valid_name",
    "probe_inconclusive", "pid_state_cmd", "pid_state", "kill_tree_script",
    "kill_tree", "validate_row",
]
