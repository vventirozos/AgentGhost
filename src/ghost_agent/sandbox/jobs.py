"""Promoted long-running sandbox commands — "jobs" (2026-08-11).

``execute`` wraps every sandboxed run in a hard budget (``_EXEC_TIMEOUT_S``,
600 s). Before this module, hitting that budget was pure loss:

* the process was KILLED at the budget and everything it had done was thrown
  away;
* the tool returned exit 124, which the turn loop scores as a FAILURE — so a
  legitimately-long task burned a transient strike, and strike accounting
  feeds outcome labels and calibration (a miscounted strike poisons the
  learning signal, not just the log);
* the model, seeing a failure with no other explanation, RE-RAN the identical
  command — turning 10 minutes of loss into 20, 30, 40 (observed live
  2026-08-11: a yt-dlp download over Tor, killed at 600 s and immediately
  retried).

The fix is to PROMOTE instead of kill. When the budget expires and the
process is *still alive and still making progress*, we detach it as a
supervised job and tell the caller so:

    ▎ Still running at 600s — promoted to background job job-1a2b3c4d.
      Do NOT re-run this command — it was NOT killed and is still working.
    ▎ Poll it with jobs(action='status', job_id='job-1a2b3c4d'); read its
      full result with jobs(action='collect', …) once it finishes. …

**The TOOL decides, not the model.** Promotion happens on OBSERVED duration
and OBSERVED progress; nothing depends on the model predicting how long
``yt-dlp`` takes over Tor.

Two rails keep this from being a slower timeout / an unbounded wait:

1. **Progress, not just liveness.** A wedged process still dies at the
   budget. The discriminator, within ``progress_window_s()``: has the command
   WRITTEN anything (stdout/stderr growth, read host-side off the bind mount),
   or have ITS OWN I/O counters moved (``rchar``/``wchar`` summed across its
   session in ``/proc``)? CPU time is deliberately NOT a signal — a spin-loop
   is exactly the wedge we must kill, and it burns CPU by definition.
   Attribution is the subtle part: a first implementation asked "did a file
   under the workdir change", and a live test promoted a genuinely wedged
   ``sleep 300`` because a DIFFERENT job was writing in the same shared
   ``/workspace``. Per-process counters attribute to the job that earned
   them.
2. **A separate lifetime cap.** A promoted job is reaped at ``job_ttl_s()``
   (default 45 min) whether it has finished or not, so the unbounded-wait
   problem is not merely moved into the job layer. Reaping runs from the
   ``jobs`` tool, from a periodic sweeper, and at boot (a promoted job
   outlives the agent process — the container does not).

Mechanics mirror :mod:`sandbox.services` (the proven pattern for detached
container processes): the command ships as a SCRIPT through the bind mount
(no quoting hazards), launches under ``setsid nohup`` so it re-parents to the
container's PID 1 and survives the exec, records its own pid, and writes its
exit code to a sentinel file. State lives in
``<host_workspace>/.jobs/registry.json`` — host side of the bind mount, so it
survives an agent restart alongside the container.

Distinct from ``manage_services``: a service is a long-lived LISTENER with a
port lease that the model starts deliberately; a job is a one-shot COMMAND
the model ran in the foreground that simply took longer than the budget.
Neither replaces the other, and jobs never get a port.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

JOBS_DIRNAME = ".jobs"
CONTAINER_JOBS_DIR = "/workspace/.jobs"
CONTAINER_WORKDIR = "/workspace"

# Job states. RUNNING is the only non-terminal one.
STATE_RUNNING = "running"
STATE_DONE = "done"            # exited on its own; exit_code carries the code
STATE_EXPIRED = "expired"      # killed at job_ttl_s()
STATE_CANCELLED = "cancelled"  # killed on request
STATE_LOST = "lost"            # pid gone with no exit sentinel (recreate/OOM)
_STATES = frozenset({STATE_RUNNING, STATE_DONE, STATE_EXPIRED,
                     STATE_CANCELLED, STATE_LOST})

# MACHINE-READABLE PROMOTION MARKER. A promoted result is deliberately
# success-SHAPED (`EXIT CODE: 0`) so the turn loop does not score a strike —
# but roughly a dozen other consumers read an execute result to decide
# whether work SUCCEEDED, and to them "exit 0" would mean "finished, fine".
# Two of those decisions are consequential: project_advancer's verify gate
# marks a task DONE, and acquired_skills' TDD gate installs a skill. Prose is
# not a contract, so the result carries this marker in its FIRST LINE — well
# inside the `[:120]`/`[:200]` head slices those consumers take — and every
# grader tests for it explicitly.
PROMOTED_RESULT_MARKER = "[SANDBOX JOB PROMOTED]"
# ⚠ MATCH THE COMPOSITE BANNER, NOT THE BARE MARKER. The marker alone is a
# string that appears in this repo's own source, its tests, its transcripts
# and its git history — so a genuinely SUCCESSFUL `grep -rn "SANDBOX JOB
# PROMOTED" src/` (or a `cat` of this very file) was being read by every
# consumer as an unfinished command: the verify gate returned "inconclusive",
# the self-play reward counted an error, a macro step was marked failed. That
# is not hypothetical here — the agent reviews its own source. Requiring the
# whole first line, which only `_promoted_result` emits, discriminates:
#   grep of jobs.py → False ✅   cat of execute.py → False ✅
#   pytest echoing an assertion → False ✅   a real promoted result → True ✅
# Deliberately a SUBSTRING test rather than first-line-anchored: the turn
# loop can PREPEND a `[FAILURE BANNER]` line to a result before the corpus
# sees it, and that is exactly the path where the verdict matters most.
PROMOTED_RESULT_BANNER = f"--- COMMAND RESULT --- {PROMOTED_RESULT_MARKER}"


def is_promoted_result(text) -> bool:
    """True when an execute result describes a command that was DETACHED and
    is still running — neither a success nor a failure, and never evidence
    that anything was verified.

    The residual false positive is reading a TRANSCRIPT that quotes a real
    promoted result; no textual marker can fix that, only out-of-band
    signalling the plain-string tool contract cannot express today."""
    return PROMOTED_RESULT_BANNER in str(text or "")


_LOG_TAIL_BYTES = 16 * 1024
# Ceiling on how much of a job log is pulled into RAM at once. docker's own
# exec buffer has no such bound, so this is strictly safer than the path it
# replaces; head+tail keeps both ends of a pathological log readable.
_LOG_READ_CAP = 32 * 1024 * 1024
# Terminal rows kept in the registry. It is a worklist, not an archive: the
# output of a collected job is already in the model's context.
_MAX_TERMINAL_ROWS = 20
# Registry cap on a stored command string. The re-run guard declines to match
# at or above this length, since the stored copy is a prefix.
_MAX_COMMAND_CHARS = 2000
_ID_RE = re.compile(r"^job-[0-9a-f]{8}$")
# The poll loop's container round-trip: liveness + the job's I/O counters in
# one exec. Without the liveness half, a job whose runner was SIGKILLed (so it
# never wrote its exit sentinel) would block the caller for the whole budget
# instead of returning the moment it died.
#
# The FIRST probe is deliberately late: a command that finishes in under two
# seconds — the overwhelming majority — must not pay a docker round-trip it
# will never need. After that, one probe per 30 s is ~20 execs across a
# 600 s run.
_PROBE_FIRST_AFTER_S = 2.0
_PROBE_EVERY_S = 30.0
# Grace for the exit sentinel to cross the bind mount after the process is
# gone. Without it a command that finished in the last instant would be
# reported as killed (exit 137) purely because of mount propagation lag.
_SENTINEL_GRACE_S = 3.0
# Slack added to the runner's last-resort `timeout` so the TTL reaper always
# fires first. Must exceed the sweeper's period (main._SANDBOX_JOB_REAP_EVERY_S,
# 60 s) — otherwise the ceiling lands the job as "exit 124" and the EXPIRED
# state, with its "hit its lifetime cap" message, becomes dead code.
# Sized against the WORST case, not the nominal one: promotion drift (launch
# RTT + probe + the inline reap) plus the sweeper's period plus a full reap
# walk, each of which is bounded by docker's own client deadlines (~75-90 s
# per round-trip) times the concurrency cap. Measured nominal ≈ 63 s; the
# slow-daemon bound is ≈ 840 s. Under-sizing it does not lose data — it makes
# the container ceiling fire first, landing the job as `exit 124` and turning
# STATE_EXPIRED into dead code.
_CEILING_MARGIN_S = 1200.0


def _probe_inconclusive(out, code) -> bool:
    """True when a probe's non-zero exit says "the PROBE failed", not "the
    process is gone".

    Two shapes, both measured: ``sandbox.execute`` reports an infra fault
    (wedged daemon, container restart, provision backoff) as exit 1 with an
    ``[SANDBOX INFRA ERROR]`` body — and it wraps every probe in its own
    ``timeout -k 5s 15s``, whose expiry surfaces as exit 124 with the generic
    ``[SYSTEM ERROR]: Process failed`` line and NO infra marker. Reading
    either as death is how a live job got exit 137 with its log deleted.
    """
    text = str(out or "")
    return ("SANDBOX INFRA ERROR" in text
            or int(code or 0) in (124, 137, 143))


def jobs_enabled() -> bool:
    """Kill switch. ``GHOST_SANDBOX_JOBS=0`` restores the pre-2026-08-11
    behaviour exactly: every sandboxed run goes through the plain
    ``timeout -k 5s`` wrapper and a budget overrun is killed."""
    return str(os.environ.get("GHOST_SANDBOX_JOBS", "1")).strip().lower() \
        not in ("0", "false", "no", "off")


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    """Read a bounded float from the environment. An unparseable or
    out-of-range value falls back to the default rather than disabling the
    rail it configures (a 0-second TTL would reap every job instantly)."""
    try:
        val = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return val if lo <= val <= hi else default


def promote_after_s() -> float:
    """How long a command must run before it is eligible for promotion.

    DELIBERATELY MUCH SHORTER THAN THE BUDGET (default 90 s vs 600 s), and
    they are different questions:

    * ``promote_after_s`` — "this is long-running; stop making the turn wait
      for it." Blocking a turn for ten minutes to reach a decision that was
      already obvious at ninety seconds is pure latency: the operator waits,
      and so does the model.
    * the exec BUDGET — "this has gone quiet; give up on it." A silent,
      purely-computational run still gets the full budget before it is
      killed, exactly as it did before promotion existed. Lowering the
      budget along with this would have made that class die 6.7× sooner —
      a regression dressed up as a speed-up.

    Promotion at 90 s is only safe because a landed job RE-ENGAGES the model
    (``main._reap_sandbox_jobs`` → an internal turn). Without that, an early
    promotion on a `pytest`/`pip install` would end the turn and strand the
    work until the operator spoke again. If auto-resume is ever disabled,
    raise this back toward the budget.

    Bounded to [5 s, 1 h].
    """
    return _env_float("GHOST_SANDBOX_JOB_PROMOTE_AFTER_S", 90.0, 5.0, 3600.0)


def job_ttl_s() -> float:
    """How long a PROMOTED job may keep running before the supervisor reaps
    it. The clock starts at promotion, so total wall time is at most the exec
    budget plus this. Default 45 min; bounded to [60 s, 6 h]."""
    return _env_float("GHOST_SANDBOX_JOB_TTL_S", 45 * 60.0, 60.0, 6 * 3600.0)


def job_max_log_bytes() -> int:
    """Hard ceiling on a SINGLE job's log file, in bytes.

    A promoted job writes its log for its whole TTL with no size cap; a fast
    writer (`yes`, a wget/curl loop) is "progressing", so it promotes and then
    fills the host disk (measured 100s of GB across budget + the 45-min TTL).
    Crossing this ceiling EXPIRES the job through the SAME _kill_pgroup +
    STATE_EXPIRED path the TTL uses. The live log is NEVER truncated in place —
    the writer holds the file offset, and deleting/rewinding an open file it is
    still writing to corrupts it; reads already head+tail via _LOG_READ_CAP.

    Default 1024 MB; a value of 0 disables the cap. Bounded to [0, 64 GB]."""
    mb = _env_float("GHOST_SANDBOX_JOB_MAX_LOG_MB", 1024.0, 0.0, 65536.0)
    return int(mb * 1024 * 1024) if mb > 0 else 0


def progress_window_s() -> float:
    """How recently the command must have shown progress to earn promotion.
    Default 120 s; bounded to [10 s, 1 h]."""
    return _env_float("GHOST_SANDBOX_JOB_PROGRESS_WINDOW_S",
                      120.0, 10.0, 3600.0)


def effective_window_s(budget: float) -> float:
    """The progress window actually used for a run of ``budget`` seconds.

    Clamped to half the budget. Configured alone the window is a foot-gun: a
    window >= the budget means no sample is ever older than it, the
    comparison falls back to the very first sample, and the rail silently
    degrades from "did it do anything RECENTLY" to "did it ever do anything
    at all" — i.e. off.

    The 10 s floor wins for budgets under 20 s, which is the test regime,
    not production's 600 s; there the fallback baseline is the t≈2 s sample,
    so the question becomes "anything since it started" — the most that can
    be asked of a run that short.

    A module-level function so it can be TESTED rather than reimplemented in
    a test: the previous pin computed the same arithmetic in its own body
    and stayed green when the clamp was deleted from production.
    """
    return max(10.0, min(progress_window_s(), max(1.0, float(budget)) / 2.0))


def max_concurrent_jobs() -> int:
    """Concurrent promoted jobs allowed. Without a cap, a loop of long
    commands piles up detached container processes nobody is watching."""
    return int(_env_float("GHOST_SANDBOX_JOB_MAX", 3.0, 1.0, 16.0))


def valid_job_id(job_id) -> bool:
    """Job ids reach this module from the MODEL (``jobs(job_id=…)``) and are
    interpolated into file paths, so the shape is enforced, not assumed."""
    return bool(_ID_RE.match(str(job_id or "")))


class SandboxJobSupervisor:
    """Promote, track, and reap detached long-running sandbox commands.

    All methods are SYNC (they shell into the container via
    ``sandbox_manager.execute``) — call them through ``asyncio.to_thread``
    from async tool code, exactly like :class:`services.ServiceSupervisor`.
    """

    def __init__(self, sandbox_manager):
        self.sandbox = sandbox_manager
        self._lock = threading.RLock()
        # jid -> the per-job sentinel nonce this process minted. In-memory
        # by design: a nonce that survived a restart would be readable from
        # the registry, which is the file the nonce defends against. A job
        # adopted after a restart simply falls back to the unauthenticated
        # read — it is already tracked, so the forgery cannot create an
        # untracked process, only land an existing row early.
        self._nonces: Dict[str, str] = {}
        # jids already warned about a bad sentinel (one line per job, not
        # one per poll tick).
        self._warned_nonce: set = set()

    # -- paths / registry ----------------------------------------------------

    @property
    def host_dir(self) -> Path:
        """``<host_workspace>/.jobs`` — verified to be a real directory
        inside the sandbox, not a symlink out of it.

        ⚠ SECURITY BOUNDARY, same class as ``_safe_cleanup_paths``. This
        path is inside the bind mount, so the supervised command can replace
        it with a symlink; every file this module writes and deletes derives
        from it. Measured: `ln -s /somewhere/else /workspace/.jobs` made the
        host side write the runner script into — and delete files from —
        an arbitrary directory at the agent's uid, with no race needed.
        Hardening cleanup_paths alone left this door open.
        """
        root = Path(self.sandbox.host_workspace)
        path = root / JOBS_DIRNAME
        try:
            if path.is_symlink() or (
                    path.exists()
                    and path.resolve().parent != root.resolve()):
                raise RuntimeError(
                    f"sandbox job dir {path} is not a real directory inside "
                    f"{root} (symlink or redirected) — refusing to use it")
        except RuntimeError:
            raise
        except OSError:
            pass
        return path

    @property
    def _registry_path(self) -> Path:
        return self.host_dir / "registry.json"

    def _load(self) -> Dict[str, dict]:
        """Registry rows, MALFORMED ONES DROPPED.

        This file lives on the bind mount at ``/workspace/.jobs/`` — inside
        the sandbox, writable by the very commands this module supervises. So
        it is untrusted input, and every field a KILL or a PATH is built from
        is validated here rather than at the point of use:

        * the key must be the ``job-<8 hex>`` shape (it becomes a filename);
        * ``pid`` must be an integer above 1 (``kill -- -1`` is "every
          process in the container" — see ``_kill_pgroup``);
        * every row must carry a numeric ``deadline_at``. Without this check
          a row with no deadline read as ``float(None or 0) == 0``, i.e.
          "expired in 1970" — instant expiry AND an instant kill on the next
          sweep, which is a fail-DANGEROUS default for a missing field.
        """
        try:
            data = json.loads(self._registry_path.read_text())
        except Exception:  # noqa: BLE001 — absent/corrupt → empty
            return {}
        if not isinstance(data, dict):
            return {}
        clean: Dict[str, dict] = {}
        for key, entry in data.items():
            if not (valid_job_id(key) and isinstance(entry, dict)):
                continue
            try:
                pid = int(entry.get("pid"))
                deadline = float(entry.get("deadline_at"))
            except (TypeError, ValueError):
                logger.warning(
                    "sandbox job registry: dropping malformed row %s "
                    "(pid=%r deadline_at=%r)", key,
                    entry.get("pid"), entry.get("deadline_at"))
                continue
            # NaN/Infinity parse as floats and pass a `<= 0` test, which
            # would make `time.time() >= deadline` permanently False — an
            # infinite TTL smuggled in as a number.
            if pid <= 1 or not math.isfinite(deadline) or deadline <= 0:
                logger.warning(
                    "sandbox job registry: dropping unsafe row %s "
                    "(pid=%s deadline=%r) — a pid of 0/1 would signal the "
                    "whole container, and a non-finite deadline is an "
                    "unbounded lifetime", key, pid, entry.get("deadline_at"))
                continue
            # An unknown `state` must not let a row opt OUT of reaping: a
            # single trailing space ("done ") made the sweep skip it, so its
            # pid was never probed and its lifetime cap never fired. Unknown
            # normalises to RUNNING — the state that gets supervised.
            if entry.get("state") not in _STATES:
                logger.warning(
                    "sandbox job registry: row %s has unknown state %r — "
                    "treating it as running so the lifetime cap still "
                    "applies", key, entry.get("state"))
                entry = {**entry, "state": STATE_RUNNING}
            clean[key] = entry
        return clean

    def _save(self, reg: Dict[str, dict]) -> None:
        self.host_dir.mkdir(parents=True, exist_ok=True)
        # UNIQUE temp name. A fixed `registry.tmp` (the shape inherited from
        # services.py) is not safe against a second process sharing this
        # workspace — a throwaway instance for an ablation, or the test
        # suite: B can truncate the file A is still writing, then rename it
        # into place, so A's remaining bytes land in what is now the live
        # registry. The result parses as garbage, `_load` returns {}, and
        # EVERY running job silently loses its row — never reaped, never
        # killed, never collectable.
        tmp = self._registry_path.with_suffix(
            f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(reg, indent=2))
            os.replace(tmp, self._registry_path)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def list_entries(self) -> List[dict]:
        """Read-only snapshot of every tracked job, oldest first."""
        return sorted(self._load().values(),
                      key=lambda e: float(e.get("promoted_at") or 0.0))

    def get(self, job_id) -> Optional[dict]:
        return self._load().get(str(job_id or ""))

    # -- container helpers ---------------------------------------------------

    def _exec(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """A bookkeeping round-trip: liveness, I/O counters, a kill. Always
        QUIET — these fire every 30 s per running job and once a minute per
        job from the reaper, and a subsystem whose purpose is to run
        unattended must not flood the stream the operator watches. Managers
        without the `quiet` kwarg (older stubs) still work."""
        try:
            out, code = self.sandbox.execute(cmd, timeout=timeout, quiet=True)
        except TypeError:
            out, code = self.sandbox.execute(cmd, timeout=timeout)
        return (out or ""), code

    def _container_generation(self) -> Optional[str]:
        """Id of the LIVE sandbox container, or None when unknown. Stamped on
        every entry: a container recreate invalidates every pid, and a
        same-numbered pid in the NEW container is an unrelated process — so a
        generation mismatch means the job is gone, never "kill that pid"."""
        # ⚠ id + StartedAt, not id alone — see the twin in services.py.
        # A graceful stop→resume keeps the id but resets the pid counter, so
        # a bare-id stamp let the TTL reaper aim `_kill_pgroup` (a
        # SESSION-wide kill) at a recycled pid within its 45-min window
        # (§4BW CRITICAL-1). StartedAt changes on every restart; a legacy
        # id-only stamp mismatches this form and `_generation_ok` reads it
        # dead — the safe direction for a kill primitive.
        try:
            container = getattr(self.sandbox, "container", None)
            cid = getattr(container, "id", None)
            if not cid:
                return None
            started = ""
            try:
                attrs = getattr(container, "attrs", None) or {}
                started = (attrs.get("State", {}) or {}).get("StartedAt", "")
                if not started and hasattr(container, "reload"):
                    container.reload()
                    attrs = getattr(container, "attrs", None) or {}
                    started = (attrs.get("State", {}) or {}).get("StartedAt", "")
            except Exception:  # noqa: BLE001 — attrs/reload may be stubbed
                started = ""
            return f"{cid}:{started}" if started else str(cid)
        except Exception:  # noqa: BLE001 — a mock may refuse attributes
            return None

    def _pid_state(self, pid) -> Optional[bool]:
        """True alive, False dead, **None UNKNOWN**.

        The third value is the important one. This probe runs through the
        container, and ``sandbox.execute`` reports an infra fault (wedged
        daemon, a container restart, the provision-backoff refusal) as exit 1
        with an ``[SANDBOX INFRA ERROR]`` body — indistinguishable, on the
        exit code alone, from "that pid is gone". Reading a docker hiccup as
        DEATH is how a healthy job got exit 137, its log deleted out from
        under it, and its process left running with no registry row: an
        orphan created by the very code meant to prevent orphans. Unknown
        stays unknown, and callers pick the safe direction for themselves.

        Live AND not a zombie: ``kill -0`` alone reports zombies as alive,
        and a finished job whose parent shell has exited IS a zombie until
        PID 1 reaps it — which would make a done job read as running forever
        (the trap services.py hit in 2026-07). State is the first field after
        the LAST ')' in /proc/<pid>/stat (comm may contain spaces/parens)."""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        cmd = (f"sh -c 'kill -0 {pid} 2>/dev/null && "
               f"[ \"$(sed \"s/^.*) //\" /proc/{pid}/stat 2>/dev/null "
               f"| cut -d\" \" -f1)\" != Z ]'")
        out, code = self._exec(cmd, timeout=15)
        if code != 0 and _probe_inconclusive(out, code):
            return None
        return code == 0

    def _pid_alive(self, pid) -> bool:
        """Strict liveness — UNKNOWN reads as not-alive. Only for callers
        where that is the safe direction (verifying a kill actually landed).
        Anything that would KILL or DISCARD on the answer must use
        :meth:`_pid_state` and handle None."""
        return self._pid_state(pid) is True

    def _generation_ok(self, entry: dict) -> bool:
        """False only on a POSITIVE container-generation mismatch: the pid
        recorded for this entry belonged to a container that no longer
        exists, so the number must never be trusted OR signalled. An
        unstamped entry, or an unknown current generation, gets the benefit
        of the doubt and falls through to the pid check."""
        if not isinstance(entry, dict):
            return False
        stamped = entry.get("container_id")
        gen = self._container_generation()
        if not stamped:
            # ⚠ NOT "benefit of the doubt" any more. The registry lives on
            # the bind mount, so a supervised command can write a row naming
            # ANY live pid; an unstamped row that reads as "same generation"
            # turned the TTL reaper into an arbitrary in-container
            # session-kill primitive (measured: a planted row with
            # deadline_at=1 killed an unrelated setsid'd service tree, and
            # services.py launches under setsid too). Every row THIS module
            # writes is stamped whenever the generation is knowable, so an
            # unstamped row is either forged or predates the stamp — neither
            # earns a signal. Only trusted when the generation is itself
            # unknown (a stub manager with no container).
            return gen is None
        return not (gen and gen != stamped)

    def _entry_alive(self, entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        if not self._generation_ok(entry):
            return False
        return self._pid_alive(entry.get("pid"))

    def _kill_pgroup(self, pid) -> bool:
        """TERM then KILL the whole process group (``setsid`` made the job's
        shell the leader), with a plain-pid fallback for the odd case where
        setsid was unavailable. Returns True when the pid is gone afterwards.

        ⚠ THE PID FLOOR IS A SAFETY BOUNDARY, NOT A TIDINESS CHECK. In POSIX
        ``kill -- -1`` means "every process the caller may signal" — NOT
        "process group 1" — so a pid of 1 here TERMs and then KILLs the whole
        container: every service, every other job, tor. Verified in a live
        container: the shell issuing it did not survive to print its next
        line. ``0`` is nearly as bad (``-0`` = the caller's own process
        group). The registry that supplies this number lives on the bind
        mount and is therefore writable by the sandboxed process itself, so
        the number is untrusted input; ``_load`` rejects malformed rows and
        this floor is the second gate.
        """
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 1:
            pretty_log(
                "Job Kill Refused",
                f"refusing to signal pid {pid} — 'kill -- -{pid}' targets "
                f"the whole container, not one job (registry row is "
                f"malformed or tampered with)",
                level="ERROR", icon=Icons.SHIELD)
            return False
        # SESSION-scoped, not just process-group-scoped. Measured live: the
        # runner's own last-resort `timeout` wrapper calls setpgid() on
        # itself (GNU timeout runs its child in a NEW process group unless
        # --foreground), so `kill -- -<runner pid>` killed the runner shell
        # and left `timeout` + the actual command alive — a wedged `sleep
        # 999` survived its own kill, and the leader dying made it look
        # successful. A session is inherited and cannot be changed by a
        # child, so scoping by session id reaches the whole tree however it
        # re-groups. The group kill is kept as the no-/proc fallback.
        # ⚠ NO `--` BEFORE THE NEGATIVE PID. The container's /bin/sh is dash,
        # whose builtin `kill` consumes `-TERM` and then parses `--` as a pid:
        # `kill -TERM -- -123` fails with "Illegal number: -" and sends
        # NOTHING (measured in python:3.11-slim-bookworm). `kill -TERM -123`
        # is the portable form and does signal the group. The plain-pid
        # fallback matters too: without it, a job whose /proc scan comes up
        # empty (no /proc, no sed, or setsid never ran so the recorded pid is
        # not a session leader) would be marked reaped while its process
        # kept running — the invisible orphan this module exists to prevent.
        # Membership is SESSION **or** PPID-ANCESTRY. Session alone is not
        # an ancestor-closed set: a descendant that changes session or
        # process group escapes the group kill and the session scan, and —
        # because the verification used that same scan — was certified as a
        # clean kill. The ancestry walk is bounded (depth 32) so a cycle in
        # a hostile /proc cannot hang it.
        #
        # ⚠ KNOWN LIMIT, measured, and NOT a regression: a descendant that
        # DAEMONIZES (`setsid sleep … &` and friends) both re-sessions AND
        # re-parents to PID 1, which erases every link back to us — so
        # neither scan can reach it. Verified in the real image that the
        # CLASSIC `timeout -k 5s` path leaks exactly the same process, so
        # this is a property of the sandbox, not of promotion. The
        # backgrounded-server guard in tools/execute.py is what actually
        # addresses that class, by refusing the command up front.
        script = (
            f'S={pid}; '
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
            f'sig KILL; true'
        )
        self._exec(f"sh -c {shlex.quote(script)}", timeout=30)
        # Verify against the SESSION too: _probe reports alive when ANY
        # member is still running, so a surviving grandchild cannot be
        # mistaken for a clean kill just because the leader is gone.
        alive, _io = self._probe(pid)
        return alive is False

    # -- job files -----------------------------------------------------------

    @staticmethod
    def _new_id() -> str:
        return f"job-{uuid.uuid4().hex[:8]}"

    def _paths(self, jid: str) -> dict:
        """Host- and container-side paths for one job's files."""
        if not valid_job_id(jid):
            raise ValueError(f"malformed job id {jid!r}")
        return {
            "script": self.host_dir / f"{jid}.cmd.sh",
            "log": self.host_dir / f"{jid}.log",
            "pid": self.host_dir / f"{jid}.pid",
            "exit": self.host_dir / f"{jid}.exit",
            "c_script": f"{CONTAINER_JOBS_DIR}/{jid}.cmd.sh",
            "c_log": f"{CONTAINER_JOBS_DIR}/{jid}.log",
            "c_pid": f"{CONTAINER_JOBS_DIR}/{jid}.pid",
            "c_exit": f"{CONTAINER_JOBS_DIR}/{jid}.exit",
        }

    def log_rel_path(self, jid: str) -> str:
        """Workspace-relative log path — readable with ``file_system``."""
        return f"{JOBS_DIRNAME}/{jid}.log"

    def _read_log(self, jid: str) -> bytes:
        try:
            path = self._paths(jid)["log"]
            size = path.stat().st_size
            if size <= _LOG_READ_CAP:
                return path.read_bytes()
            half = _LOG_READ_CAP // 2
            with open(path, "rb") as fh:
                head = fh.read(half)
                fh.seek(size - half)
                tail = fh.read(half)
            return (head
                    + f"\n\n[... {size - _LOG_READ_CAP} bytes of job output "
                      f"omitted — read {self.log_rel_path(jid)} ...]\n\n"
                      .encode()
                    + tail)
        except (OSError, ValueError):
            return b""

    def log_tail(self, jid: str, lines: int = 40) -> str:
        try:
            path = self._paths(jid)["log"]
            with open(path, "rb") as fh:
                size = path.stat().st_size
                fh.seek(max(0, size - _LOG_TAIL_BYTES))
                raw = fh.read()
        except (OSError, ValueError):
            return "(no output)"
        if not raw:
            return "(no output)"
        text = raw.decode("utf-8", "replace")
        return "\n".join(text.splitlines()[-max(1, int(lines)):])

    def _read_exit(self, jid: str) -> Optional[int]:
        """The command's own exit code once the sentinel has landed, else
        None.

        The sentinel is written tmp+rename inside the container, but a
        bind-mount read can still catch a partial/empty file — an unparseable
        read is treated as NOT-yet-there rather than as exit 0.

        ⚠ AND IT CARRIES A NONCE. The sentinel lives in a directory the
        supervised command can write, so a bare exit code was forgeable AND
        creatable by accident: one line —
        ``echo 0 > /workspace/.jobs/<jid>.exit`` — made `execute` return
        `EXIT CODE: 0` in 0.0 s for a command that kept running, with no
        registry row, no TTL and no reaper. Any command that merely wipes or
        writes into `.jobs/` produced the same result. The nonce is minted
        host-side per job and only the runner script quotes it back."""
        try:
            txt = self._paths(jid)["exit"].read_text().strip()
        except (OSError, ValueError):
            return None
        nonce = self._nonces.get(jid)
        if nonce:
            prefix = nonce + " "
            if not txt.startswith(prefix):
                # ONCE per job: this is read on every poll tick, so an
                # unauthenticated sentinel would otherwise emit ~45 warnings
                # per run — burying the one line that matters.
                if jid not in self._warned_nonce:
                    self._warned_nonce.add(jid)
                    pretty_log(
                        "Job Sentinel Rejected",
                        f"{jid}: exit sentinel without the expected nonce — "
                        f"ignoring it. The command wrote its own, or wiped "
                        f"{JOBS_DIRNAME}/; the job is supervised on its real "
                        f"process instead.",
                        level="WARNING", icon=Icons.SHIELD)
                return None
            txt = txt[len(prefix):].strip()
        return int(txt) if txt.lstrip("-").isdigit() else None

    def _read_pid(self, jid: str) -> Optional[int]:
        try:
            txt = self._paths(jid)["pid"].read_text().strip()
        except (OSError, ValueError):
            return None
        return int(txt) if txt.isdigit() else None

    def _log_size(self, jid: str) -> int:
        try:
            return self._paths(jid)["log"].stat().st_size
        except (OSError, ValueError):
            return 0

    def _cleanup_files(self, jid: str, drop_log: bool = False,
                       entry: Optional[dict] = None) -> None:
        """Drop a job's bookkeeping files. The LOG is kept by default — it is
        the job's output, and ``collect`` still has to return it.

        ``<jid>.exit.tmp`` is included: a job killed between the runner's
        ``echo`` and its ``mv`` leaves one behind, and nothing else would
        ever remove it.

        ``entry['cleanup_paths']`` are HOST paths the caller handed over —
        the ephemeral script an ``execute(content=…)`` run is still
        executing. It cannot be deleted while the job runs (the interpreter
        is reading it), and it is persisted on the row rather than held in
        memory so a restart still cleans it up."""
        try:
            paths = self._paths(jid)
        except ValueError:
            return
        keys = ["script", "pid", "exit"] + (["log"] if drop_log else [])
        for key in keys:
            try:
                paths[key].unlink()
            except OSError:
                pass
        try:
            (self.host_dir / f"{jid}.exit.tmp").unlink()
        except OSError:
            pass
        for extra in self._safe_cleanup_paths(
                (entry or {}).get("cleanup_paths")):
            try:
                extra.unlink()
            except OSError:
                pass

    def _safe_cleanup_paths(self, raw) -> List[Path]:
        """``cleanup_paths`` entries that are provably inside the sandbox.

        ⚠ THIS IS A SECURITY BOUNDARY, not tidiness. These paths come off the
        registry, which is bind-mounted read-write INTO the container — so a
        sandboxed command can write them, and unlinking them unvalidated
        would hand it an arbitrary HOST file-deletion primitive against
        anything the agent's uid can remove (its own source tree, GHOST_HOME
        databases, keys). ``_load`` already validates every field a KILL is
        built from for the same reason; this is the field a PATH is built
        from. Resolved (so ``..`` cannot escape) and required to sit under
        the sandbox root.
        """
        out: List[Path] = []
        try:
            root = Path(self.sandbox.host_workspace).resolve()
        except Exception:  # noqa: BLE001
            return out
        for item in raw or ():
            try:
                path = Path(str(item)).resolve()
            except (OSError, ValueError, TypeError):
                continue
            if path == root or root not in path.parents:
                logger.warning(
                    "sandbox job cleanup: refusing to delete %s — outside "
                    "the sandbox root %s (registry row is malformed or "
                    "tampered with)", path, root)
                continue
            if path.is_dir():
                continue          # unlink() only; never a tree
            out.append(path)
        return out

    # -- progress ------------------------------------------------------------

    def _probe(self, pid) -> tuple:
        """One container round-trip answering both questions the poll loop
        needs: is the job ALIVE, and how much I/O has it done in total?

        ``(alive, io_chars_or_None)``. ``io_chars`` sums ``rchar + wchar``
        across every process in the job's SESSION — ``setsid`` made the
        recorded pid the session leader, so descendants are included and
        nothing else is. None means the counters were unreadable.

        **Why not "did a file under the workdir change recently".** That was
        the first implementation and a live test killed it: a genuinely
        wedged ``sleep 300`` was PROMOTED because a *different* job writing
        in the same ``/workspace`` looked like its progress. The workdir is
        shared — by other jobs, other conversations, services — so mtimes
        there attribute to nobody. Per-process I/O counters attribute
        exactly, which is what "has IT made progress" actually means.

        rchar+wchar (not just writes): a job pulling a large file over the
        network, or reading its way through a big input, is working. A wedged
        process — blocked on a dead socket, deadlocked, or sleeping — moves
        neither, and neither does a syscall-free spin loop. Both remain
        killable, which is the point of the rail.
        """
        try:
            sid = int(pid)
        except (TypeError, ValueError):
            return False, None
        # Iterate /proc rather than shelling to `ps`: procps is not
        # guaranteed in the image, /proc always is. Session id is the 4th
        # field after the LAST ')' in stat (comm can contain spaces/parens);
        # state is the 1st, so liveness + zombie-rejection come from the same
        # read services.py's _pid_alive does.
        script = (
            f'S={sid}; N=0; A=0; T=0; '
            f'for f in /proc/[0-9]*; do '
            f'  s=$(sed "s/^.*) //" "$f/stat" 2>/dev/null); '
            f'  [ -n "$s" ] || continue; '
            f'  st=$(echo "$s" | cut -d" " -f1); '
            f'  sd=$(echo "$s" | cut -d" " -f4); '
            f'  [ "$sd" = "$S" ] || continue; '
            f'  N=$((N+1)); '
            f'  [ "$st" = "Z" ] || A=1; '
            f'  v=$(sed -n "s/^[rw]char:[[:space:]]*//p" "$f/io" 2>/dev/null); '
            f'  for n in $v; do T=$((T+n)); done; '
            f'done; echo "$N $A $T"'
        )
        out, code = self._exec(f"sh -c {shlex.quote(script)}", timeout=20)
        parts = (out or "").split()
        # N == 0 means either the session is gone or /proc is not readable
        # (a non-Linux host in a test). Those are indistinguishable from
        # here, so both defer to the plain pid check — authoritative for
        # liveness, and reporting NO io evidence rather than false evidence.
        # Degrading this way lands on the pre-2026-08-11 behaviour (kill at
        # the budget unless the log grew), never on "promote blindly".
        if code != 0 or len(parts) < 3 or parts[0] == "0":
            return self._pid_state(sid), None
        alive = parts[1] == "1"
        try:
            return alive, int(parts[2])
        except ValueError:
            return alive, None

    # -- launch / poll -------------------------------------------------------

    def _write_script(self, jid: str, cmd: str, hard_ceiling_s: float) -> None:
        """The runner script: record our pid, run the command, publish the
        exit code atomically. Written from the HOST through the bind mount so
        no shell quoting can corrupt the command."""
        paths = self._paths(jid)
        self.host_dir.mkdir(parents=True, exist_ok=True)
        # Clear BEFORE writing: a recycled id would otherwise inherit a stale
        # sentinel and "finish" instantly. Ids are random, but the cost of
        # being sure is one unlink — and doing it after the write would
        # delete the script we just laid down.
        self._cleanup_files(jid, drop_log=True)
        # LAST-RESORT CEILING, inside the container. The classic path wrapped
        # every run in `timeout -k 5s <budget>s`, which survived the agent
        # process dying; the job path replaces that with a HOST-side poll
        # loop plus `deadline_at`, and both die with the agent. A deploy is a
        # plain `kill` here (launchd respawns), so an agent killed mid-run
        # would otherwise leave a detached process with no registry row, no
        # TTL, and no wrapper — immortal until the container is recreated.
        # The ceiling must never pre-empt the normal reaper — it only catches
        # the case where nothing is left to reap. budget + TTL alone is NOT
        # enough: that fires at the same instant as `deadline_at`, and since
        # the sweeper only polls every 60 s the ceiling usually wins, landing
        # the job DONE/exit 124 instead of EXPIRED and telling the model its
        # command FAILED. The margin is the sweeper's period plus slack.
        ceiling = int(max(60.0, float(hard_ceiling_s)) + _CEILING_MARGIN_S)
        nonce = secrets.token_hex(8)
        self._nonces[jid] = nonce
        paths["script"].write_text(
            "#!/bin/sh\n"
            # $$ under setsid is the session/group leader, so `kill -- -$$`
            # reaps the whole tree (the command may fork children).
            f"echo $$ > {paths['c_pid']}\n"
            # Probed, not assumed: `timeout` is coreutils (present in the
            # Debian sandbox image, absent on a BSD host and on a minimal
            # image). Unset _T expands to nothing, so the command still runs
            # — losing only the last-resort ceiling, never the run.
            "_T=\"\"\n"
            "command -v timeout >/dev/null 2>&1 && "
            f"_T=\"timeout -k 5s {ceiling}s\"\n"
            f"$_T {cmd}\n"
            "_rc=$?\n"
            # tmp+rename: a reader on the host must never see a half-written
            # sentinel and conclude "exit 0".
            f"echo \"{nonce} $_rc\" > {paths['c_exit']}.tmp\n"
            f"mv {paths['c_exit']}.tmp {paths['c_exit']}\n"
            "exit \"$_rc\"\n"
        )

    def _launch_exec(self, cmd: str, exec_kwargs: dict) -> Tuple[str, int]:
        """Run a short command in the container, preferring the raw
        ``_exec_run`` (so the caller's user/workdir kwargs apply) and falling
        back to the manager's public ``execute`` for stub managers."""
        runner = getattr(self.sandbox, "_exec_run", None)
        if runner is None:
            out, code = self.sandbox.execute(cmd, timeout=60,
                                             **{k: v for k, v in exec_kwargs.items()
                                                if k == "workdir"})
            return (out or ""), code
        result = runner(cmd, deadline_s=60, **exec_kwargs)
        raw = getattr(result, "output", b"") or b""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        return raw, int(getattr(result, "exit_code", 0) or 0)

    def _launch(self, jid: str, exec_kwargs: dict) -> Optional[int]:
        """Start the runner detached; return the launcher pid (a fallback —
        the script's own ``$$`` is authoritative and read by the poll loop).

        No ``cd`` here: the exec already starts in ``exec_kwargs['workdir']``
        and the detached child inherits it, so a missing workdir fails the
        same way it does on the classic path instead of turning into a
        confusing shell error.
        """
        paths = self._paths(jid)
        inner = (f"setsid nohup sh {paths['c_script']} > {paths['c_log']} "
                 f"2>&1 < /dev/null & echo $!")
        out, code = self._launch_exec(f"sh -c {shlex.quote(inner)}",
                                      exec_kwargs)
        if code != 0:
            raise RuntimeError(
                f"job launcher failed (exit {code}): {out.strip()[:300]}")
        for tok in reversed(out.split()):
            if tok.isdigit():
                return int(tok)
        return None

    def _find_running(self, identity: str, workdir: str) -> Optional[dict]:
        """A RUNNING job with this exact IDENTITY + workdir, or None. Reaps
        first so a job that has since finished cannot masquerade as one.

        Identity is the command by default, but the caller may supply
        something narrower. It must, for a SCRIPT run: there the command is
        ``python3 -u build.py``, which is stable across every rewrite of
        build.py — so a model that reads the promotion banner, FIXES the
        script and re-runs it would be handed the old job back, told "already
        running", given the old job's log as its output, and blocked from
        running the fix for up to the whole TTL. The execute tool therefore
        folds a hash of the script's CONTENT into the identity.
        """
        with self._lock:
            self.reap()
            if len(str(identity)) >= _MAX_COMMAND_CHARS:
                # The stored command is TRUNCATED, so a comparison at this
                # length is prefix-only: two different long commands sharing
                # a prefix would collide, and the second would never run
                # while being told it was "already running". Decline to
                # match rather than guess.
                return None
            for entry in self._load().values():
                if (entry.get("state") == STATE_RUNNING
                        and entry.get("identity") == str(identity)
                        and str(entry.get("workdir") or "") == str(workdir)):
                    return dict(entry)
        return None

    def _wait_for_sentinel(self, jid: str, grace_s: float) -> Optional[int]:
        """Poll briefly for a late exit sentinel (bind-mount propagation)."""
        deadline = time.monotonic() + max(0.0, float(grace_s))
        while True:
            code = self._read_exit(jid)
            if code is not None or time.monotonic() >= deadline:
                return code
            time.sleep(0.1)

    def run(self, cmd: str, *, timeout: float, workdir: str = None,
            label: str = None, project_id=None, exec_kwargs: dict = None,
            cleanup_paths=None, identity: str = None):
        """Run ``cmd`` with a budget; promote instead of killing when it is
        still alive AND still progressing at the budget.

        Returns ``(stdout_bytes, exit_code, job_entry_or_None)``. When
        ``job_entry`` is not None the command is STILL RUNNING and
        ``exit_code`` is 0 by convention — a promotion is not a failure and
        must not burn a strike; ``stdout_bytes`` is the output so far.
        """
        wd = str(workdir or (exec_kwargs or {}).get("workdir")
                 or CONTAINER_WORKDIR)
        budget = max(1.0, float(timeout))
        window = effective_window_s(budget)

        # RE-RUN GUARD. The banner tells the model not to re-run a promoted
        # command; this makes it true whether or not the model complies. The
        # old exit-124 path had the strike counter as a brake — a promotion
        # deliberately removes that, so without this a model that re-runs
        # detaches a SECOND copy of a heavy command competing for the same
        # container CPU, and a third, until the concurrency cap. Identity is
        # the exact command + workdir, and only against a job that is still
        # RUNNING, so an ordinary repeated `ls` can never match (it finishes,
        # so it is never a job in the first place).
        existing = self._find_running(identity or cmd, wd)
        if existing is not None:
            pretty_log(
                "Job Already Running",
                f"{existing['id']} is this exact command, still running — "
                f"NOT starting a second copy: {str(cmd)[:70]}",
                level="WARNING", icon=Icons.JOB_PROMOTE)
            return (self._read_log(existing["id"]), 0,
                    {**existing, "duplicate": True})

        jid = self._new_id()
        self._write_script(jid, cmd, budget + job_ttl_s())
        launcher_pid = None
        try:
            # The LAUNCH is inside the guard: it is the docker round-trip
            # most likely to fail, and it can fail AFTER the container has
            # already spawned the detached runner (a client-deadline
            # abandonment, a socket error on the read). Outside the guard,
            # that leaves a live process with no row, no TTL and no reaper.
            launcher_pid = self._launch(jid, dict(exec_kwargs or {}))
            return self._supervise(jid, cmd, launcher_pid, budget=budget,
                                   window=window, workdir=wd, label=label,
                                   project_id=project_id,
                                   cleanup_paths=cleanup_paths,
                                   identity=identity)
        except BaseException:
            # ⚠ BaseException, NOT Exception (§4BW CRITICAL-2). This guard's
            # whole job is "kill the detached process on any exit from the
            # launch→registration window" — but the window is dominated by
            # `_supervise`'s poll-loop SLEEP, so an interrupt (KeyboardInterrupt
            # / SystemExit from a Ctrl-C, a pytest timeout, a killed batch)
            # lands here far more often than an ordinary Exception. Those are
            # BaseException and used to sail straight through, leaking an
            # immortal detached process with no row and no reaper — the source
            # of the week-old busy-loop orphans on this box. The re-raise
            # below preserves the interrupt.
            #
            # EVERYTHING from here to the registry write runs with a live
            # detached process and no row tracking it. A docker fault (wedged
            # daemon, container restart, a raise from the registry save) would
            # otherwise leave that process running with no TTL, no reaper, and
            # nothing that knows it exists. Kill it and re-raise — the caller
            # (docker._execute_impl) turns the exception into a normal
            # [SANDBOX INFRA ERROR] result.
            pid = self._read_pid(jid) or launcher_pid
            if pid:
                try:
                    self._kill_pgroup(pid)
                except Exception:  # noqa: BLE001 — already unwinding
                    pass
            self._cleanup_files(jid, drop_log=True,
                                entry={"cleanup_paths": cleanup_paths})
            raise

    def _supervise(self, jid: str, cmd: str, launcher_pid, *, budget: float,
                   window: float, workdir: str, label, project_id,
                   cleanup_paths=None, identity: str = None):
        """The poll loop, split out so :meth:`run` can guarantee the launched
        process is killed on ANY failure between launch and registration."""
        started = time.monotonic()
        pid = None
        last_size = 0
        # None until the log ACTUALLY grows. Seeding it with the start time
        # would mean "it wrote something at t=0" for a command that never
        # writes anything — and any budget shorter than the window would then
        # promote every silent process, which is precisely the wedge case
        # this discriminator exists to kill.
        last_growth = None
        # (monotonic_ts, io_chars) samples from the periodic probe. The
        # promotion test asks "did EITHER counter move within the window",
        # so we only need the newest sample plus the newest one older than
        # the window — but the list is tiny (one per 30 s) and keeping it
        # whole makes the comparison obvious.
        io_samples = []
        maxlog = job_max_log_bytes()
        promote_after = min(float(promote_after_s()), budget)
        # A probe lands exactly at the promotion threshold, so the decision
        # is taken at ~90 s rather than at the next 30 s tick after it.
        next_probe = min(started + _PROBE_FIRST_AFTER_S,
                         started + promote_after)
        # Poll cadence: tight at first so a 50 ms `ls` stays a 50 ms `ls`,
        # then relaxed so a 10-minute run costs ~600 stat calls, not 12,000.
        while True:
            code = self._read_exit(jid)
            if code is not None:
                return self._finish_completed(jid, code)
            if pid is None:
                pid = self._read_pid(jid)
            size = self._log_size(jid)
            now = time.monotonic()
            if size > last_size:
                last_size, last_growth = size, now
            if maxlog and size > maxlog:
                # Runaway writer within the budget window (before it even
                # promotes): kill now rather than let a fast producer fill the
                # host disk for the rest of the budget. Same cap as the reaper.
                pid = pid or self._read_pid(jid) or launcher_pid
                pretty_log(
                    "Job Log Cap",
                    f"{jid} exceeded the {maxlog // (1024 * 1024)}MB job-log "
                    f"cap mid-run — killed: {str(cmd)[:70]}",
                    level="WARNING", icon=Icons.STOP)
                return self._kill_and_return(jid, pid, 124,
                                             cleanup_paths=cleanup_paths)
            if now - started >= budget:
                break
            if now >= next_probe:
                # Before the threshold, the next probe lands ON it (so the
                # decision is taken at ~90 s, not at the next 30 s tick
                # after it); past it, the ordinary cadence resumes. `max`
                # here would have pushed the first decision out to 32 s.
                next_probe = (min(now + _PROBE_EVERY_S,
                                  started + promote_after)
                              if now - started < promote_after
                              else now + _PROBE_EVERY_S)
                probe_pid = pid or launcher_pid
                if probe_pid:
                    alive, io_chars = self._probe(probe_pid)
                    if io_chars is not None:
                        io_samples.append((now, io_chars))
                    # ── EARLY PROMOTION ──────────────────────────────────
                    # Past the threshold, a command that is demonstrably
                    # WORKING is detached immediately instead of making the
                    # turn wait out the rest of the budget. The budget keeps
                    # its other job: a command that is NOT progressing is
                    # left alone here and only killed when the full budget
                    # expires, so a silent pure-compute run gets exactly the
                    # patience it always had.
                    if (alive is not False
                            and now - started >= promote_after):
                        if self._progressing(now, last_growth, io_samples,
                                             io_chars, window):
                            entry = self._promote(
                                jid, cmd=cmd, pid=probe_pid, workdir=workdir,
                                label=label, project_id=project_id,
                                ran_s=now - started,
                                cleanup_paths=cleanup_paths,
                                identity=identity)
                            if entry is not None:
                                return self._read_log(jid), 0, entry
                            # Cap full — fall through and keep waiting; the
                            # budget still bounds it.
                    if alive is False:
                        # Runner gone with no sentinel — SIGKILL, OOM, or the
                        # container went away. Return NOW instead of blocking
                        # the caller for the rest of the budget. Only on a
                        # DEFINITE death: `None` means the probe itself
                        # failed, and treating that as death deleted a live
                        # job's log and orphaned its process.
                        code = self._wait_for_sentinel(jid, _SENTINEL_GRACE_S)
                        if code is not None:
                            return self._finish_completed(jid, code)
                        out = self._read_log(jid)
                        self._cleanup_files(jid, drop_log=True)
                        return out, 137, None
            elapsed = now - started
            time.sleep(0.05 if elapsed < 1.0
                       else 0.2 if elapsed < 10.0
                       else 1.0)

        # --- budget reached -------------------------------------------------
        pid = pid or self._read_pid(jid) or launcher_pid
        # One last look: the command may have landed in the final tick.
        code = self._wait_for_sentinel(jid, 0.2)
        if code is not None:
            return self._finish_completed(jid, code)

        alive, io_now = (self._probe(pid) if pid else (False, None))
        if alive is False:
            code = self._wait_for_sentinel(jid, _SENTINEL_GRACE_S)
            if code is not None:
                return self._finish_completed(jid, code)
            out = self._read_log(jid)
            self._cleanup_files(jid, drop_log=True)
            return out, 137, None
        # alive is None → the probe failed, not the job. Handled below, at
        # the promotion decision: it must not fall through to the progress
        # test, because the same fault zeroes the I/O evidence that test
        # reads.

        now = time.monotonic()
        # The probe failed rather than the job: `_probe` zeroes BOTH values
        # on a fault, so `did_io` would be False for want of evidence and
        # the "not progressing" branch below would KILL a healthy job — the
        # false failure this whole module exists to prevent, re-entering
        # through the diagnostics. Promote instead: the job is then TRACKED
        # and TTL-bounded, which is strictly better than an untracked kill
        # that (on the same fault) also fails.
        if alive is None:
            entry = self._promote(jid, cmd=cmd, pid=pid, workdir=workdir,
                                  label=label, project_id=project_id,
                                  ran_s=budget, cleanup_paths=cleanup_paths,
                                  identity=identity)
            if entry is not None:
                pretty_log(
                    "Job Probe Failed",
                    f"{jid} could not be probed at the budget (docker "
                    f"fault) — promoted rather than killed; the reaper "
                    f"settles it: {str(cmd)[:60]}",
                    level="WARNING", icon=Icons.JOB_PROMOTE)
                return self._read_log(jid), 0, entry
            return self._kill_and_return(jid, pid, 124,
                                         cleanup_paths=cleanup_paths)
        progressing = self._progressing(now, last_growth, io_samples,
                                        io_now, window)
        if not progressing:
            pretty_log(
                "Job Not Promoted",
                f"produced no output and did no I/O for {int(window)}s at "
                f"the {int(budget)}s budget — killed: {str(cmd)[:80]}",
                level="WARNING", icon=Icons.STOP)
            return self._kill_and_return(jid, pid, 124,
                                        cleanup_paths=cleanup_paths)

        entry = self._promote(jid, cmd=cmd, pid=pid, workdir=workdir,
                              label=label, project_id=project_id,
                              ran_s=budget, cleanup_paths=cleanup_paths,
                              identity=identity)
        if entry is None:
            # Cap reached — the job cannot be tracked, so it must not be left
            # running unsupervised.
            return self._kill_and_return(jid, pid, 124,
                                        cleanup_paths=cleanup_paths)
        return self._read_log(jid), 0, entry

    def _kill_and_return(self, jid: str, pid, code: int, cleanup_paths=None):
        """Kill the job and return its output. The log is dropped ONLY when
        the kill is confirmed — a surviving process is still writing to it,
        and deleting an open file leaves it writing to an unlinked inode
        nobody can read."""
        killed = self._kill_pgroup(pid)
        out = self._read_log(jid)
        self._cleanup_files(jid, drop_log=killed,
                            entry={"cleanup_paths": cleanup_paths})
        if not killed:
            logger.warning(
                "sandbox job %s: pid %s survived TERM+KILL — its log is "
                "kept at %s", jid, pid, self.log_rel_path(jid))
        return out, code, None

    @staticmethod
    def _progressing(now, last_growth, io_samples, io_now, window) -> bool:
        """Did this job produce output, or move its own I/O counters, within
        the window? Shared by the early-promotion check and the at-budget
        check so the two can never drift apart."""
        if last_growth is not None and (now - last_growth) <= window:
            return True
        if io_now is None or not io_samples:
            return False
        # Compare against the newest sample from BEFORE the window, so the
        # question is "did anything happen in the LAST `window` seconds".
        # When the run is shorter than the window there is no such sample and
        # the oldest one stands in — the comparison is then over the whole
        # run, which is the most that can be asked.
        older = [c for ts, c in io_samples if now - ts >= window]
        baseline = older[-1] if older else io_samples[0][1]
        # CHANGED, not INCREASED: the sum spans the job's whole session, so a
        # child exiting REMOVES its counters and the total can fall. Either
        # direction means the process tree did something; a total that has
        # not moved at all in `window` seconds is the wedge.
        return io_now != baseline

    def _finish_completed(self, jid: str, code: int):
        """Normal completion — read the output, then drop every file for this
        job (nothing will poll for it)."""
        out = self._read_log(jid)
        self._cleanup_files(jid, drop_log=True)
        return out, int(code), None

    def _promote(self, jid: str, *, cmd: str, pid: int, workdir: str,
                 label, project_id, ran_s: float,
                 cleanup_paths=None, identity: str = None) -> Optional[dict]:
        """Register a still-running command as a tracked job. Returns None
        when the concurrency cap is already taken (the caller kills)."""
        with self._lock:
            self.reap()
            reg = self._load()
            live = [e for e in reg.values() if e.get("state") == STATE_RUNNING]
            cap = max_concurrent_jobs()
            if len(live) >= cap:
                pretty_log(
                    "Job Cap Reached",
                    f"{len(live)}/{cap} promoted jobs already running — "
                    f"killing this one instead of detaching it (free a slot "
                    f"with jobs(action='cancel')): {str(cmd)[:60]}",
                    level="WARNING", icon=Icons.STOP)
                return None
            now = time.time()
            entry = {
                "id": jid,
                "command": str(cmd)[:_MAX_COMMAND_CHARS],
                # What the re-run guard compares. For a script run this
                # carries a content hash, so a REWRITTEN script is a
                # different job (see _find_running).
                "identity": str(identity or cmd)[:_MAX_COMMAND_CHARS],
                "label": str(label or cmd)[:200],
                "workdir": workdir,
                "pid": int(pid),
                "project_id": str(project_id) if project_id else None,
                "state": STATE_RUNNING,
                "exit_code": None,
                "started_at": now - float(ran_s),
                "promoted_at": now,
                "deadline_at": now + job_ttl_s(),
                "ran_before_promotion_s": round(float(ran_s), 1),
                "container_id": self._container_generation(),
                "log": self.log_rel_path(jid),
                # HOST paths this job owns until it ends — the ephemeral
                # script an `execute(content=…)` run is still executing.
                # Persisted (not held in memory) so a restart still cleans up.
                "cleanup_paths": [str(x) for x in (cleanup_paths or [])],
            }
            reg[jid] = entry
            self._save(reg)
        pretty_log(
            "Job Promoted",
            f"{jid} still running at {int(ran_s)}s (pid {pid}) — detached "
            f"instead of killed; reaped at {int(job_ttl_s() / 60)}min: "
            f"{str(cmd)[:70]}",
            icon=Icons.JOB_PROMOTE)
        return dict(entry)

    # -- lifecycle -----------------------------------------------------------

    def reap(self) -> List[dict]:
        """Land finished jobs and kill expired ones. Returns the entries whose
        state CHANGED, so a caller can report only the news.

        Safe to call from anywhere at any cadence: it is the only writer of
        terminal states and every transition is idempotent."""
        changed: List[dict] = []
        with self._lock:
            reg = self._load()
            dirty = False
            for jid, entry in list(reg.items()):
                if entry.get("state") != STATE_RUNNING:
                    continue
                code = self._read_exit(jid)
                if code is not None:
                    entry["state"] = STATE_DONE
                    entry["exit_code"] = int(code)
                    entry["finished_at"] = time.time()
                    self._cleanup_files(jid, entry=entry)
                    changed.append(entry)
                    dirty = True
                    continue
                # GENERATION FIRST, ALWAYS BEFORE ANY SIGNAL. A recreated
                # container invalidates every recorded pid, and fresh
                # containers hand out LOW pids — so the number in this row
                # very plausibly belongs to a supervised service, another
                # job, or tor. services.py had to learn this in 2026-07
                # ("PID recycling across a container recreate … pointed
                # stop() at innocents"); the expiry branch below used to
                # kill before asking, which is the same defect.
                gen_ok = self._generation_ok(entry)
                if not gen_ok:
                    entry["state"] = STATE_LOST
                    entry["finished_at"] = time.time()
                    self._cleanup_files(jid, entry=entry)
                    changed.append(entry)
                    dirty = True
                    continue
                maxlog = job_max_log_bytes()
                if maxlog and self._log_size(jid) > maxlog:
                    # Runaway writer: a fast producer promotes (it is
                    # "progressing") and would otherwise write for its whole
                    # TTL, filling the host disk (measured 100s of GB). Expire
                    # it exactly like the TTL does — probe BEFORE signalling (a
                    # generation match is not proof this pid is still the job's
                    # process). The live log is kept, NOT truncated in place:
                    # the writer holds its offset, and _read_log already
                    # head+tails it for the model.
                    if self._pid_state(entry.get("pid")) is not False:
                        self._kill_pgroup(entry.get("pid"))
                    entry["state"] = STATE_EXPIRED
                    entry["finished_at"] = time.time()
                    entry["expired_reason"] = "log_size_cap"
                    self._cleanup_files(jid, entry=entry)
                    changed.append(entry)
                    dirty = True
                    pretty_log(
                        "Job Log Cap",
                        f"{jid} exceeded the {maxlog // (1024 * 1024)}MB "
                        f"job-log cap and was killed: "
                        f"{str(entry.get('command'))[:70]}",
                        level="WARNING", icon=Icons.STOP)
                    continue
                if time.time() >= float(entry.get("deadline_at") or 0):
                    # Probe BEFORE signalling. A generation match only says
                    # "same container" — it does not say this pid is still
                    # the job's process. An expired row whose process died
                    # without a sentinel (OOM) while the agent was down would
                    # otherwise have its recycled pid signalled, and the blast
                    # radius is now a whole SESSION (services launch under
                    # setsid too, so that could be an entire service tree).
                    if self._pid_state(entry.get("pid")) is not False:
                        self._kill_pgroup(entry.get("pid"))
                    entry["state"] = STATE_EXPIRED
                    entry["finished_at"] = time.time()
                    self._cleanup_files(jid, entry=entry)
                    changed.append(entry)
                    dirty = True
                    pretty_log(
                        "Job Expired",
                        f"{jid} hit its {int(job_ttl_s() / 60)}min lifetime "
                        f"cap and was killed: "
                        f"{str(entry.get('command'))[:70]}",
                        level="WARNING", icon=Icons.STOP)
                    continue
                state = self._pid_state(entry.get("pid"))
                if state is not False:
                    # Alive, or UNKNOWN (the probe failed, not the job).
                    # Leaving the row RUNNING costs one more sweep; calling
                    # it LOST on a docker hiccup would stop tracking a live
                    # process for good — no TTL, no reaper, no collect.
                    continue
                # The pid is gone. It may have finished in the instant
                # between the sentinel read above and this probe — a real
                # race (measured: a job exiting mid-reap was marked LOST and
                # its exit code destroyed), so re-read before concluding.
                late = self._wait_for_sentinel(jid, _SENTINEL_GRACE_S)
                if late is not None:
                    entry["state"] = STATE_DONE
                    entry["exit_code"] = int(late)
                else:
                    entry["state"] = STATE_LOST
                entry["finished_at"] = time.time()
                self._cleanup_files(jid, entry=entry)
                changed.append(entry)
                dirty = True
            if self._trim_terminal(reg):
                dirty = True
            # Stale `registry.<pid>.<hex>.tmp` files from a process that died
            # between write and rename. Harmless but they accumulate in a
            # directory the model can list.
            for stale in self.host_dir.glob("registry.*.tmp"):
                try:
                    if time.time() - stale.stat().st_mtime > 3600:
                        stale.unlink()
                except OSError:
                    pass
            if dirty:
                self._save(reg)
        return changed

    def _trim_terminal(self, reg: Dict[str, dict]) -> bool:
        """Bound the registry: drop the oldest terminal rows (and their logs)
        past ``_MAX_TERMINAL_ROWS``. Running rows are never dropped."""
        # ⚠ Keyed by the REGISTRY KEY, never by the row's own `id` field. A
        # desynced (or planted) `id` aimed the pop at a DIFFERENT row: it
        # destroyed a RUNNING job and its log, left the intended row in
        # place — so the count never dropped and the next sweep ate another
        # live job, every minute, forever.
        terminal = sorted(
            ((k, e) for k, e in reg.items()
             if e.get("state") != STATE_RUNNING),
            key=lambda kv: float(kv[1].get("finished_at")
                                 or kv[1].get("promoted_at") or 0.0))
        excess = len(terminal) - _MAX_TERMINAL_ROWS
        if excess <= 0:
            return False
        for key, entry in terminal[:excess]:
            reg.pop(key, None)
            self._cleanup_files(key, drop_log=True, entry=entry)
        return True

    def cancel(self, job_id) -> Tuple[bool, str]:
        """Kill a running job's process group.

        Returns ``(killed, message)``. The BOOLEAN is the point: the caller
        marks its own row cancelled on it, and a message-sniffing caller got
        this wrong — "already done — nothing to cancel" carries no ``Error:``
        prefix, so it read as a successful kill and overwrote a COMPLETED
        job's result with CANCELLED. Only an actual kill returns True.
        """
        jid = str(job_id or "")
        if not valid_job_id(jid):
            return False, f"Error: malformed job id {jid!r}."
        with self._lock:
            reg = self._load()
            entry = reg.get(jid)
            if entry is None:
                return False, f"Error: no sandbox job {jid!r}."
            if entry.get("state") != STATE_RUNNING:
                return False, (f"Job {jid} is already {entry.get('state')} "
                               f"— nothing to cancel.")
            # Same rule as reap: never signal a pid from a dead container
            # generation — it belongs to an unrelated process now.
            if not self._generation_ok(entry):
                entry["state"] = STATE_LOST
                entry["finished_at"] = time.time()
                self._cleanup_files(jid, entry=entry)
                self._save(reg)
                return False, (
                    f"Job {jid} is gone — its container was recreated, so "
                    f"the process no longer exists and its pid must not be "
                    f"signalled. Marked lost.")
            killed = self._kill_pgroup(entry.get("pid"))
            if not killed:
                return False, (
                    f"Error: could not kill job {jid} (pid "
                    f"{entry.get('pid')} still alive after TERM+KILL). "
                    f"Left RUNNING — reporting a cancel that did not happen "
                    f"would hide a live process.")
            entry["state"] = STATE_CANCELLED
            entry["finished_at"] = time.time()
            self._cleanup_files(jid, entry=entry)
            self._save(reg)
        pretty_log("Job Cancelled", f"{jid} killed on request", icon=Icons.STOP)
        return True, (f"Job {jid} cancelled (process group killed). Output so "
                      f"far is in {self.log_rel_path(jid)}.")

    def forget(self, job_id) -> None:
        """Drop a terminal job's registry row AND its log.

        Deliberately has no production caller. Landing a job used to call it,
        which destroyed the full output before anything could read it — the
        `jobs` reconcile runs on every call, and the promotion banner had just
        pointed the model at that log. Bounding is `_trim_terminal`'s job
        instead. Kept as the explicit release path for tests and for an
        operator clearing one job by hand."""
        jid = str(job_id or "")
        if not valid_job_id(jid):
            return
        with self._lock:
            reg = self._load()
            entry = reg.get(jid)
            if entry is None or entry.get("state") == STATE_RUNNING:
                return
            reg.pop(jid, None)
            self._save(reg)
        self._cleanup_files(jid, drop_log=True, entry=entry)

    # -- reporting -----------------------------------------------------------

    def summary_line(self) -> Optional[str]:
        """One-line boot report, or None when nothing is tracked."""
        entries = self.list_entries()
        if not entries:
            return None
        running = [e for e in entries if e.get("state") == STATE_RUNNING]
        done = [e for e in entries if e.get("state") != STATE_RUNNING]
        now = time.time()
        bits = []
        for entry in running[:3]:
            age = int(now - float(entry.get("started_at") or now))
            bits.append(f"{entry.get('id')} running {age}s "
                        f"({str(entry.get('command'))[:40]})")
        for entry in done[:3]:
            code = entry.get("exit_code")
            bits.append(f"{entry.get('id')} {entry.get('state')}"
                        + (f" exit {code}" if code is not None else ""))
        return (f"{len(running)} promoted sandbox job(s) running, "
                f"{len(done)} finished — " + " · ".join(bits))


def get_job_supervisor(sandbox_manager) -> Optional[SandboxJobSupervisor]:
    """Get-or-create the supervisor bound to this sandbox manager. Returns
    None when there is no usable sandbox, so call sites degrade instead of
    raising."""
    if sandbox_manager is None:
        return None
    sup = getattr(sandbox_manager, "_job_supervisor", None)
    if isinstance(sup, SandboxJobSupervisor):
        return sup
    if not getattr(sandbox_manager, "host_workspace", None):
        return None
    sup = SandboxJobSupervisor(sandbox_manager)
    try:
        sandbox_manager._job_supervisor = sup
    except Exception:  # noqa: BLE001 — a mock may refuse attributes
        pass
    return sup


__all__ = [
    "SandboxJobSupervisor", "get_job_supervisor", "jobs_enabled",
    "job_ttl_s", "promote_after_s", "progress_window_s",
    "effective_window_s",
    "max_concurrent_jobs", "valid_job_id",
    "PROMOTED_RESULT_MARKER", "PROMOTED_RESULT_BANNER",
    "is_promoted_result",
    "STATE_RUNNING", "STATE_DONE", "STATE_EXPIRED", "STATE_CANCELLED",
    "STATE_LOST", "JOBS_DIRNAME", "CONTAINER_JOBS_DIR",
]
