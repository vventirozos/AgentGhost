"""§4DC Phase 0+1 — the first autonomous GEPA actors.

Two jobs, both riding the instruments' EXIT CONTRACTS (`gate_contract`),
which is the whole design: an autonomous caller acts on exit codes, and
§4DA spent eighteen rounds making those codes stop lying before anything
was allowed to act on them.

**Phase 0 — supply watch.** Re-mines the tool-choice fixture pool on a
wall-clock cadence and notifies ONCE when the supply gate flips from
parked to ready. The miner's own gates decide readiness (exit 0 = pool
written, 1 = parked at `.notready`, 2 = missing corpus); this job adds
only a clock and a transition-edge notification.

**Phase 1 — live judge.** Runs `gepa_live_check --revert` daily over
every signature with a live artifact. The judge's contract: 0 = KEEP,
1 = REVERT (retired on disk), 2 = could not measure, 3 = reported but
not acted on. The only action this grants the loop is the one that can
exclusively UNDO GEPA's own work — retiring a losing artifact back to
the hand-written baseline.

Design constraints, each from a recorded failure:

* **Wall-clock cadence, persisted** (`traffic-gated-clocks`): at ~3.5
  turns/day anything gated on turns starves. Due-ness is computed from a
  state file that survives restarts.
* **Notify on TRANSITIONS, not ticks** (`fire-once-notification`,
  `chat-noise-preference`): the parked->ready edge notifies once
  (re-armed if supply falls back below the gate); a REVERT notifies per
  retirement; KEEP/could-not-measure are log-only.
* **The subprocess IS the interface**: the scripts are the tested,
  mutation-hardened surface. Running them in-process would bypass the
  argv/env/exit-code layer these jobs exist to consume — and an
  UNDECLARED exit code is treated as an instrument failure, notified
  once per distinct code, never acted on
  (`instruments-fail-not-runtime`).
* **Kill switches** (`outcome-gated-learning-loop`):
  `GHOST_GEPA_AUTONOMY=0` disables both jobs; `GHOST_GEPA_AUTO_REVERT=0`
  demotes the judge to report-only (drops `--revert`).
* **Neither job touches the inference slot**: the miner and the judge
  are file readers. Timeouts are watchdogs against a wedged filesystem,
  not against model latency.
* **Retirement needs a restart to take effect** and restarts are the
  operator's (root launchd; see the respawn-loop incident). The REVERT
  notification says so explicitly — Phase 2 (loader hot-reload) removes
  this, not Phase 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .gate_contract import (
    JUDGE_RETIRED_MARKER,
    JUDGE_REVERT_MARKER,
    JUDGE_RUN_BANNER,
    MINER_DONE_MARKER,
    MINER_RUN_BANNER,
    JudgeExit,
)

#: Wall-clock cadences (seconds). Weekly mine, daily judgement — the
#: 7-day re-draw guard and MIN_PER_ARM-at-3.5-turns/day both make faster
#: clocks pointless.
SUPPLY_WATCH_INTERVAL_S = 7 * 86400
LIVE_JUDGE_INTERVAL_S = 86400

#: Watchdog deadlines for the subprocesses (file readers; generous).
SUPPLY_WATCH_TIMEOUT_S = 900
LIVE_JUDGE_TIMEOUT_S = 600

_STATE_NAME = "gepa_autonomy_state.json"

#: The miner's exit contract (scripts/mine_tool_fixtures.py).
MINER_READY = 0
MINER_PARKED = 1
MINER_NO_CORPUS = 2


def autonomy_enabled() -> bool:
    """Master kill switch. Default ON — the operator's stated goal is a
    fully autonomous loop; `GHOST_GEPA_AUTONOMY=0` turns both jobs off.
    Read per call (not at import) so the flag can be flipped without a
    restart; the off-set idiom matches `verifier.py`'s kill switches."""
    return os.getenv("GHOST_GEPA_AUTONOMY", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def auto_revert_enabled() -> bool:
    """`GHOST_GEPA_AUTO_REVERT=0` demotes the judge to report-only."""
    return os.getenv("GHOST_GEPA_AUTO_REVERT", "1").strip().lower() \
        not in ("0", "false", "no", "off")


#: §4U, trimmed to this job class. `replay_engine.preflight()` gates on
#: Docker because a replay batch spawns containers; these jobs are file
#: readers whose only real resource precondition is disk (the miner
#: writes one pool file; the state file is tiny). Importing the full
#: preflight would stand the supply watch down whenever OrbStack is off,
#: for no reason.
MIN_DISK_FREE_MB = 512


def _preflight(home: str) -> Optional[str]:
    """None when clear; a reason string when the job must stand down.
    A preflight that cannot read a precondition must REPORT that, not
    clear the launch (`replay_engine`'s psutil rule)."""
    try:
        import shutil as _sh
        free_mb = _sh.disk_usage(str(Path(home))).free / 1e6
    except Exception as e:  # noqa: BLE001
        return f"preflight could not read disk usage ({e})"
    if free_mb < MIN_DISK_FREE_MB:
        return (f"only {free_mb:.0f}MB free under GHOST_HOME "
                f"(floor {MIN_DISK_FREE_MB}MB)")
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _state_path(home: str) -> Path:
    # ⚠ NOT under `system/optim/`. Everything matching `*.json` there IS
    # a live artifact by convention, and parking the state file there
    # polluted three existing instruments at once (§4DC lens B, B1):
    # learning_health grew a phantom `{chars: 0, valid: False}` row,
    # liveness' RETIRED branch flipped to "1 invalid artifact minted",
    # and the applies-probe's HISTORICAL annotation was suppressed for
    # good. `mark-it-where-you-catch-it`: the file lives one level up.
    return Path(home) / "system" / _STATE_NAME


def _load_state(home: str) -> Dict[str, Any]:
    try:
        return json.loads(_state_path(home).read_text())
    except Exception:  # noqa: BLE001 — a torn/missing state file means
        return {}      # "never ran"; both jobs are safe to re-run.


def _save_state(home: str, state: Dict[str, Any]) -> None:
    """Atomic, and NEVER raises — the module contract says a job that
    can crash its caller stops being scheduled, and this sat outside
    every try (§4DC lens B, B3). The staging name is unique PER CALL —
    a per-PID name protected two processes but not the pair its own
    comment named (an orphaned outer-timeout thread beside the next
    tick SHARES the pid; driven, the interleave still raced `os.replace`
    into FileNotFoundError — lens A, B-2). Last-writer-wins on the
    final replace is accepted: the state degrades to a re-run or one
    duplicate notification, never a doubled disk action."""
    try:
        import uuid as _uuid
        path = _state_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(
            path.suffix + f".staging.{os.getpid()}.{_uuid.uuid4().hex[:8]}")
        staged.write_text(json.dumps(state, indent=1))
        os.replace(staged, path)
    except Exception:  # noqa: BLE001 — state loss degrades to a re-run
        pass


def _job_slot(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    """The job's dict, coerced — a hand-edited state file with a string
    where a dict belongs must mean "never ran", not a crash (B3)."""
    slot = state.get(key)
    if not isinstance(slot, dict):
        slot = {}
        state[key] = slot
    return slot


def _due(state: Dict[str, Any], key: str, interval_s: float,
         now: Optional[float] = None) -> bool:
    slot = state.get(key)
    last = slot.get("last_run_epoch") if isinstance(slot, dict) else None
    if not isinstance(last, (int, float)):
        return True
    _now = time.time() if now is None else now
    # A last-run in the FUTURE is a clock jump, not a recent run — the
    # same rule as the re-draw guard's promoted_utc handling.
    if last > _now:
        return True
    return (_now - last) >= interval_s


def _run_script(script: str, args: list, *, home: str,
                timeout_s: float) -> subprocess.CompletedProcess:
    """The subprocess boundary — argv, env, exit code. This is the layer
    the §4DA hardening made trustworthy; bypassing it (in-process import)
    would consume internals instead of the contract."""
    env = dict(os.environ)
    env["GHOST_HOME"] = home
    env["PYTHONPATH"] = str(_repo_root() / "src")
    return subprocess.run(
        [sys.executable, str(_repo_root() / script), *args],
        capture_output=True, text=True, timeout=timeout_s, env=env,
        cwd=str(_repo_root()))


def _notify_once(job: Dict[str, Any], condition: str,
                 notify: Callable[[str], None], message: str) -> None:
    """Fire-once per distinct condition; re-armed when the condition
    changes (`fire-once-notification`: a notifier that fires every tick
    is noise, one that never re-arms goes silent forever)."""
    if job.get("last_notified_condition") == condition:
        return
    notify(message)
    job["last_notified_condition"] = condition


# ─────────────────────────────────────────────────────────────────────
# Phase 0 — supply watch
# ─────────────────────────────────────────────────────────────────────

def run_supply_watch(home: str, *, notify: Callable[[str], None],
                     log: Callable[[str], None],
                     run_script: Callable = _run_script,
                     now: Optional[float] = None,
                     force: bool = False) -> Optional[int]:
    """One tick. Returns the miner's exit code, or None when not due /
    disabled. Never raises: an autonomy job that can crash the caller is
    a job the caller will stop scheduling."""
    if not autonomy_enabled():
        return None
    state = _load_state(home)
    if not force and not _due(state, "supply_watch",
                              SUPPLY_WATCH_INTERVAL_S, now):
        return None
    job = _job_slot(state, "supply_watch")
    job["last_run_epoch"] = time.time() if now is None else now
    _blocked = _preflight(home)
    if _blocked is not None:
        job["last_exit"] = None
        job["last_outcome"] = "stood_down"
        job["last_summary"] = f"stood down: {_blocked}"
        log(f"supply watch: stood down — {_blocked}")
        _save_state(home, state)
        return None
    job["last_outcome"] = "ran"
    try:
        proc = run_script("scripts/mine_tool_fixtures.py", [],
                          home=home, timeout_s=SUPPLY_WATCH_TIMEOUT_S)
        rc: Optional[int] = proc.returncode
        out_full = proc.stdout or ""
        tail = "\n".join(out_full.strip().splitlines()[-6:])
        if rc != 0 and (proc.stderr or "").strip():
            # The stderr tail is often the only clue — it rides along.
            tail += "\n[stderr] " + "\n".join(
                proc.stderr.strip().splitlines()[-4:])
        # ⚠ TWO MARKERS, TWO QUESTIONS. The BANNER (printed before any
        # I/O) says the SCRIPT ran at all — argparse, a moved script and
        # the interpreter all exit 2, the same code as "no corpus" (lens
        # A, A-1). The DONE marker — the miner's LAST line, printed
        # AFTER the pool write/park — says a mine COMPLETED: the first
        # version gated on `Labels:`, which prints before the gates and
        # the write, so a crash AT the write exited 1 with the marker
        # present and was filed as parked forever (round 2, F1 —
        # executed: `system/optim` as a file gave rc=1, zero
        # notifications, "still parked — FileExistsError"). Any code
        # without its marker is an instrument failure.
        _cause = f"rc-{rc}"
        if MINER_RUN_BANNER not in out_full:
            rc, _cause = None, "no-banner"
            tail = ("[no '" + MINER_RUN_BANNER + "' banner — the script "
                    "did not start: bad argv, moved script, or broken "
                    "interpreter]\n" + tail)
        elif rc in (MINER_READY, MINER_PARKED) \
                and MINER_DONE_MARKER not in out_full:
            rc, _cause = None, "no-done-marker"
            tail = ("[no '" + MINER_DONE_MARKER + "' marker — the miner "
                    "started but did not COMPLETE the mine (a crash at "
                    "the write is the known shape)]\n" + tail)
    except Exception as e:  # noqa: BLE001 — timeout / spawn failure
        rc, tail = None, f"{type(e).__name__}: {e}"
        _cause = type(e).__name__
    job["last_exit"] = rc
    job["last_summary"] = tail[-2000:]
    if rc == MINER_READY:
        _notify_once(
            job, "ready", notify,
            "GEPA SUPPLY GATE OPEN: the tool-choice fixture pool passed "
            "the miner's gates and was written live. "
            "scripts/optimize_tool_descriptions.py can now run.\n" + tail)
        log("supply watch: pool READY")
    elif rc == MINER_PARKED:
        # Parked is the steady state — log-only, and RE-ARM the ready
        # edge so a pool that later falls back below the gate (era
        # cutoff, corpus loss) notifies again when it recovers.
        job["last_notified_condition"] = "parked"
        _last = tail.splitlines()[-1] if tail else ""
        log("supply watch: still parked" + (f" — {_last}" if _last else ""))
    else:
        # MINER_NO_CORPUS, an undeclared code, a marker-less exit, or a
        # spawn failure: the INSTRUMENT failed, not the supply. Notify
        # once per condition.
        # ⚠ KEYED BY CAUSE, NOT BY rc. Banner-less exits, marker-less
        # exits, timeouts and spawn failures ALL coerce rc to None, so
        # `instrument:{rc}` collapsed every distinct failure into
        # `instrument:None` — a DIFFERENT later cause never re-notified
        # (round 3, F1: week-1 moved script, week-2 crash-at-write, one
        # notification). The same rule the exit-3 split applied, one
        # branch over.
        _notify_once(
            job, f"instrument:{_cause}", notify,
            f"GEPA SUPPLY WATCH: the miner did not run cleanly "
            f"(exit={rc}, cause={_cause}). This is an instrument "
            f"failure, not a supply verdict.\n{tail}")
        log(f"supply watch: miner exit={rc}")
    _save_state(home, state)
    return rc


# ─────────────────────────────────────────────────────────────────────
# Phase 1 — live judge
# ─────────────────────────────────────────────────────────────────────

def live_signatures(home: str) -> list:
    """Signatures with a LIVE artifact — exactly the loader's contract:
    `system/optim/<signature>.json`. The suffix conventions (`.prev`,
    `.candidate.rejected`, `.retired-*`, `.notready`, `.staging`) all
    fail the `*.json` glob by construction. (The autonomy state file
    lives OUTSIDE this directory precisely so nothing here needs an
    exemption — lens B, B1.)"""
    d = Path(home) / "system" / "optim"
    return sorted(p.stem for p in d.glob("*.json") if p.is_file())


def _artifact_file_sha(home: str, sig: str) -> str:
    """A short content hash of the artifact FILE, for keying the
    notify-once condition per ARTIFACT rather than per signature — a
    retire → re-promote → retire sequence is two distinct retirements
    and must notify twice (lens B, C2: the docstring claimed this and
    the signature-keyed condition did not deliver it)."""
    try:
        import hashlib as _hl
        raw = (Path(home) / "system" / "optim"
               / f"{sig}.json").read_bytes()
        return _hl.sha256(raw).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        return "unknown"


def run_live_judge(home: str, *, notify: Callable[[str], None],
                   log: Callable[[str], None],
                   run_script: Callable = _run_script,
                   now: Optional[float] = None,
                   force: bool = False) -> Optional[Dict[str, int]]:
    """One tick. Returns {signature: exit_code}, or None when not due /
    disabled. Acts ONLY on the judge's declared contract."""
    if not autonomy_enabled():
        return None
    state = _load_state(home)
    if not force and not _due(state, "live_judge",
                              LIVE_JUDGE_INTERVAL_S, now):
        return None
    job = _job_slot(state, "live_judge")
    job["last_run_epoch"] = time.time() if now is None else now
    _blocked = _preflight(home)
    if _blocked is not None:
        job["last_outcome"] = "stood_down"
        # The liveness probe's note points the operator at last_summary
        # — which only the supply watch wrote (round 2, F5).
        job["last_summary"] = f"stood down: {_blocked}"
        log(f"live judge: stood down — {_blocked}")
        _save_state(home, state)
        return None
    job["last_outcome"] = "ran"
    # ⚠ COERCED, like every level of this file — round 2 (F2) drove a
    # hand-edited `"per_signature": "oops"` into an AttributeError that
    # aborted the phase BEFORE the supply watch and never saved state,
    # so BOTH jobs stalled with hourly ERROR spam until hand repair:
    # the B3 rule, one level below where round 1 applied it.
    per_sig = job.get("per_signature")
    if not isinstance(per_sig, dict):
        per_sig = job["per_signature"] = {}
    results: Dict[str, int] = {}
    for sig in live_signatures(home):
        _art_sha = _artifact_file_sha(home, sig)
        args = ["--signature", sig, "--home", home]
        if auto_revert_enabled():
            args.append("--revert")
        try:
            proc = run_script("scripts/gepa_live_check.py", args,
                              home=home, timeout_s=LIVE_JUDGE_TIMEOUT_S)
            rc: Optional[int] = proc.returncode
            out_full = proc.stdout or ""
            tail = "\n".join(out_full.strip().splitlines()[-8:])
            if rc != 0 and (proc.stderr or "").strip():
                # The stderr tail is often the only clue — it rides.
                tail += "\n[stderr] " + "\n".join(
                    proc.stderr.strip().splitlines()[-6:])
            # ⚠ TWO MARKER CHECKS, TWO IMPERSONATIONS. The BANNER says
            # the SCRIPT ran at all — argparse, a moved script and the
            # interpreter all exit 2, the judge's benign "could not
            # measure yet", so a permanently dead judge read as thin
            # data forever while REVERTs that should happen never did
            # (lens A, A-1). The VERDICT marker says exit 1 was a
            # verdict — a crash also exits 1, and was notified as
            # "RETIRED on disk" about an untouched artifact, after which
            # the false condition swallowed the real retirement (lens B,
            # A1). Any code without its marker is an instrument failure.
            _cause = f"rc-{rc}"
            if JUDGE_RUN_BANNER not in out_full:
                rc, _cause = None, "no-banner"
                tail = ("[no '" + JUDGE_RUN_BANNER + "' banner — the "
                        "script did not start: bad argv, moved script, "
                        "or broken interpreter]\n" + tail)
            elif rc == JudgeExit.NO_LONGER_WINS:
                _want = (JUDGE_RETIRED_MARKER if auto_revert_enabled()
                         else JUDGE_REVERT_MARKER)
                if _want not in out_full:
                    rc, _cause = None, "no-verdict-marker"
                    tail = ("[exit 1 without the '" + _want
                            + "' marker — a crash, not a verdict]\n"
                            + tail)
        except Exception as e:  # noqa: BLE001
            rc, tail = None, f"{type(e).__name__}: {e}"
            _cause = type(e).__name__
        sig_state = per_sig.setdefault(sig, {})
        if not isinstance(sig_state, dict):
            sig_state = per_sig[sig] = {}
        sig_state["last_exit"] = rc
        results[sig] = rc
        if rc == JudgeExit.STILL_WINS:
            sig_state["last_notified_condition"] = "keep"
            log(f"live judge: {sig} KEEP")
        elif rc == JudgeExit.NO_LONGER_WINS:
            # The one autonomous ACTION: with --revert this code means
            # the artifact was retired on disk; report-only means the
            # verdict is REVERT and nothing was touched.
            if auto_revert_enabled():
                _notify_once(
                    sig_state, f"retired:{_art_sha}", notify,
                    f"GEPA LIVE JUDGE: {sig} measurably LOSES to its "
                    f"baseline on production turns and was RETIRED on "
                    f"disk. ⚠ The running agent still serves it until "
                    f"the next restart (operator action: "
                    f"`sudo launchctl kickstart -k "
                    f"system/com.local.ghost-agent` when convenient).\n"
                    + tail)
            else:
                _notify_once(
                    sig_state, f"revert-reported:{_art_sha}", notify,
                    f"GEPA LIVE JUDGE (report-only, "
                    f"GHOST_GEPA_AUTO_REVERT=0): {sig} measurably LOSES "
                    f"to its baseline. Nothing was retired.\n" + tail)
            log(f"live judge: {sig} REVERT"
                + (" (retired)" if auto_revert_enabled() else
                   " (report-only)"))
        elif rc == JudgeExit.COULD_NOT_MEASURE:
            sig_state["last_notified_condition"] = "insufficient"
            log(f"live judge: {sig} could not measure yet")
        elif rc == JudgeExit.REPORTED_NOT_ACTED:
            # ⚠ THREE CAUSES SHARE EXIT 3, and the notification said
            # "vanished mid-run" for all of them — including the sha
            # mismatch, where nothing vanished: a PROMOTION completed
            # (round 2, F3). The script's stderr says which; so does
            # the condition key, so a later different cause re-notifies.
            _all_out = tail
            if "no longer the one this verdict measured" in _all_out:
                _why = ("a PROMOTION completed mid-run — the judge "
                        "refused to retire the fresh artifact on the "
                        "old one's evidence")
                _cond = f"swapped:{_art_sha}"
            else:
                _why = ("the artifact vanished mid-run (another "
                        "retirement racing?)")
                _cond = f"race:{_art_sha}"
            _notify_once(
                sig_state, _cond, notify,
                f"GEPA LIVE JUDGE: {sig} verdict was REVERT but nothing "
                f"was renamed — {_why}. Worth a look.\n" + tail)
            log(f"live judge: {sig} REVERT reported, not acted")
        else:
            # Cause-keyed for the same reason as the supply watch's.
            _notify_once(
                sig_state, f"instrument:{_cause}", notify,
                f"GEPA LIVE JUDGE: gepa_live_check exited {rc} for "
                f"{sig} (cause={_cause}) — instrument failure; no "
                f"action taken.\n{tail}")
            log(f"live judge: {sig} instrument failure exit={rc}")
    if not results:
        log("live judge: no live artifacts to judge")
    _save_state(home, state)
    return results
