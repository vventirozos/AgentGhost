"""§4DC Phase 0+1 + §4DF Phase 3 — the autonomous GEPA actors.

Three jobs, all riding the instruments' EXIT CONTRACTS (`gate_contract`),
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

**Phase 3 — the optimizer launcher (§4DF).** Runs one gate script per
day, round-robin over `OPTIMIZER_TARGETS`, and consumes `GateExit`
(0 = promoted → the §4DE epoch swap deploys it, 1 = rejected,
2 = could not measure, 3 = no candidate). The GATES are the
decision-makers: every "should we run?" question is a cheap pre-flight
inside the gate that exits 2 in seconds, so this job re-implements none
of them and NEVER passes an override flag (no `--allow-*`, no
`--force-supply`, no `--no-ab-gate` — operator-only), and never runs
`scripts/optimize_verifier.py` (outside the contract, pinned perimeter).

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
  `GHOST_GEPA_AUTONOMY=0` disables all three jobs;
  `GHOST_GEPA_AUTO_REVERT=0` demotes the judge to report-only (drops
  `--revert`); `GHOST_GEPA_AUTO_OPTIMIZE=0` disables just the launcher.
* **Only the optimizer touches the inference slot**: the miner and the
  judge are file readers whose timeouts watchdog a wedged filesystem.
  The optimizer is hours of sequential main-slot replays by design,
  which is why it launches in DEEP idle only and carries §4U's RAM
  floor on top of the shared disk floor.
* **Retirement deploys itself** (§4DE): the epoch swap in the biological
  tick notices the rename within ~a minute and the loader serves the
  baseline again — no restart, no operator action. (Phase 1 shipped
  before Phase 2 and this line then said the opposite; a stale "restart
  needed" instruction is operator noise.)
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
    GATE_NO_CANDIDATE_MARKER,
    GATE_PROMOTED_MARKER_GEPA,
    GATE_PROMOTED_MARKER_OTD,
    GATE_REJECTED_MARKER,
    GATE_RUN_BANNER_GEPA,
    GATE_RUN_BANNER_OTD,
    JUDGE_RETIRED_MARKER,
    JUDGE_REVERT_MARKER,
    JUDGE_RUN_BANNER,
    MINER_DONE_MARKER,
    MINER_RUN_BANNER,
    GateExit,
    JudgeExit,
)

#: Wall-clock cadences (seconds). Weekly mine, daily judgement — the
#: 7-day re-draw guard and MIN_PER_ARM-at-3.5-turns/day both make faster
#: clocks pointless.
SUPPLY_WATCH_INTERVAL_S = 7 * 86400
LIVE_JUDGE_INTERVAL_S = 86400
#: §4DF Phase 3 — at most ONE optimizer target attempted per day, each
#: target at most once per 7 days. The 7d is a politeness mirror of the
#: gates' own `--min-promotion-age-days` re-draw guard, which remains
#: the authority (the gate refuses with exit 2 either way; ours only
#: avoids paying a subprocess spawn to be told no).
OPTIMIZER_INTERVAL_S = 86400
OPTIMIZER_TARGET_INTERVAL_S = 7 * 86400

#: Watchdog deadlines for the subprocesses (file readers; generous).
SUPPLY_WATCH_TIMEOUT_S = 900
LIVE_JUDGE_TIMEOUT_S = 600
#: The optimizer is NOT a file reader — it is hours of sequential
#: main-slot replays by design (the politeness the gates document).
#: 6h bounds a wedged run without truncating an honest slow one.
OPTIMIZER_TIMEOUT_S = 6 * 3600

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
    # `last != last` is NaN (§4DF round 1, MIN-6): every comparison
    # against it is False, so a hand-edited NaN read as "not in the
    # future, not yet due" FOREVER — a silently parked job. Non-finite
    # means "never ran", the same coercion rule as every other level.
    if not isinstance(last, (int, float)) or last != last:
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
        # §4DF round 3 (MIN-3): the harvest, applied to THIS handler
        # too — a timeout kill AFTER the pool write + DONE marker is a
        # COMPLETED mine, not a supply verdict failure. The exit code
        # died with the kill so READY/PARKED is unknown and nothing is
        # acted on, but the cause is named distinctly (a later
        # different failure re-notifies) and the tail carries the clue.
        _t_out = getattr(e, "stdout", None) or getattr(e, "output", None)
        if _t_out:
            if isinstance(_t_out, bytes):
                _t_out = _t_out.decode("utf-8", "replace")
            _t_out = str(_t_out)
            tail += "\n" + "\n".join(_t_out.strip().splitlines()[-6:])
            if MINER_RUN_BANNER in _t_out and MINER_DONE_MARKER in _t_out:
                _cause = "timeout-after-complete"
                tail = ("[the mine COMPLETED before the kill — the pool "
                        "on disk is authoritative; check whether it "
                        "went live or parked]\n" + tail)
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
            # ⚠ §4DF round 3 (MAJOR-1): the round-2 harvest, applied to
            # the one handler whose child is the autonomous ACTOR. A
            # timeout kill landing AFTER the rename + RETIRED marker
            # left the retirement ON DISK — the §4DE swap deploys it
            # within ~a minute — while the notification said "no action
            # taken": MAJOR-4's false claim back through the timeout
            # door, one sibling over. The banner+marker in the
            # harvested output ARE the verdict (same discipline as the
            # live path); rc is restored so the normal branch notifies
            # the retirement under its real `retired:{sha}` key
            # (`_art_sha` was computed before the run, so it is the
            # pre-retirement artifact's — correct).
            _t_out = (getattr(e, "stdout", None)
                      or getattr(e, "output", None))
            if _t_out:
                if isinstance(_t_out, bytes):
                    _t_out = _t_out.decode("utf-8", "replace")
                _t_out = str(_t_out)
                tail += "\n" + "\n".join(
                    _t_out.strip().splitlines()[-8:])
                _want = (JUDGE_RETIRED_MARKER if auto_revert_enabled()
                         else JUDGE_REVERT_MARKER)
                if JUDGE_RUN_BANNER in _t_out and _want in _t_out:
                    rc = JudgeExit.NO_LONGER_WINS
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
                    f"disk. The epoch swap deploys the retirement live "
                    f"within ~a minute (§4DE) — no restart needed; the "
                    f"swap announces itself when it lands.\n" + tail)
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


# ─────────────────────────────────────────────────────────────────────
# Phase 3 (§4DF) — the loop launches its own optimizer runs
# ─────────────────────────────────────────────────────────────────────

#: Round-robin targets. ⚠ scripts/optimize_verifier.py is DELIBERATELY
#: absent — outside the gate contract by operator decision, and the
#: perimeter is pinned. Adding a target here means adding its banner and
#: markers to `gate_contract` FIRST (the §4DF day-one rule).
OPTIMIZER_TARGETS = (
    "tool_descriptions",
    "gepa:planning.decompose",
    "gepa:tool_selection.pick",
    "gepa:reflection.critique",
)

#: §4U: an optimizer run is hours of unattended main-slot replays, so it
#: gets the RAM floor the file-reader jobs deliberately skip.
#: `replay_engine`'s precedent (1.5GB headroom above the llama-server
#: working set on the 16GB box).
MIN_RAM_FREE_MB = 1500


def auto_optimize_enabled() -> bool:
    """`GHOST_GEPA_AUTO_OPTIMIZE=0` disables just the optimizer job
    (default ON — the operator's stated goal is the closed loop)."""
    return os.getenv("GHOST_GEPA_AUTO_OPTIMIZE", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _optimizer_preflight(home: str) -> Optional[str]:
    """Disk floor (shared) + RAM floor, both fail-CLOSED: a preflight
    that cannot read a precondition reports that, never clears the
    launch (`replay_engine`'s psutil rule, §4U)."""
    blocked = _preflight(home)
    if blocked is not None:
        return blocked
    try:
        import psutil as _ps
        avail_mb = _ps.virtual_memory().available / 1e6
    except Exception as e:  # noqa: BLE001
        return f"preflight could not read available RAM ({e})"
    if avail_mb < MIN_RAM_FREE_MB:
        return (f"only {avail_mb:.0f}MB RAM available "
                f"(floor {MIN_RAM_FREE_MB}MB)")
    return None


def _target_command(target: str, home: str):
    """(script, args, banner, promoted_marker) for one target string.
    The `--fixtures` argv is the MINER'S OUTPUT PATH, built from the one
    shared basename — driven before this line existed: the launcher
    spawned the real gate with no argv and argparse exited 2 BEFORE the
    banner, filing every launch as an instrument failure."""
    if target == "tool_descriptions":
        from .gate_contract import TOOL_FIXTURES_BASENAME
        pool = str(Path(home) / "system" / "optim"
                   / TOOL_FIXTURES_BASENAME)
        return ("scripts/optimize_tool_descriptions.py",
                ["--fixtures", pool],
                GATE_RUN_BANNER_OTD, GATE_PROMOTED_MARKER_OTD)
    sig = target.split(":", 1)[1]
    return ("scripts/run_gepa.py", ["--signature", sig],
            GATE_RUN_BANNER_GEPA, GATE_PROMOTED_MARKER_GEPA)


def _pick_target(per_target: Dict[str, Any],
                 now: float) -> Optional[str]:
    """Staleness-ordered round-robin: never-attempted targets first in
    declared order, then the longest-since-attempted whose age clears
    `OPTIMIZER_TARGET_INTERVAL_S`. A last-attempt in the FUTURE is a
    clock jump, not a recent attempt (the `_due` rule). None when every
    target is fresh — a healthy state, not a stall."""
    best, best_age = None, None
    for target in OPTIMIZER_TARGETS:
        slot = per_target.get(target)
        last = slot.get("last_attempt_epoch") \
            if isinstance(slot, dict) else None
        # `last != last` is NaN (MIN-6): with it, `age` is NaN, every
        # eligibility comparison is False, and ONE target silently
        # starves forever while the job clock keeps advancing —
        # unalarmed. Non-finite means "never attempted".
        if not isinstance(last, (int, float)) or last != last \
                or last > now:
            return target
        age = now - last
        if age >= OPTIMIZER_TARGET_INTERVAL_S \
                and (best_age is None or age > best_age):
            best, best_age = target, age
    return best


def run_optimizer(home: str, *, notify: Callable[[str], None],
                  log: Callable[[str], None],
                  run_script: Callable = _run_script,
                  now: Optional[float] = None,
                  force: bool = False) -> Optional[tuple]:
    """One tick: launch at most ONE gate run and consume its exit
    contract. Returns (target, exit_code) — exit_code None on an
    instrument failure — or None when not due / disabled / every target
    fresh. Never raises.

    ⚠ THE GATES ARE THE DECISION-MAKERS. Every "should we run?"
    question — supply, corpus size, resolution, re-draw age, upstream
    health — is a cheap pre-flight INSIDE the gate that exits 2 in
    seconds without paying for the optimizer. This job deliberately
    re-implements none of them (`the-sibling-one-revision-behind`), and
    NEVER passes an override flag: no `--allow-*`, no `--force-supply`,
    no `--no-ab-gate` (operator-only; §4DA's lesson is that the one
    flag bypassing a gate carries its own bypass).
    """
    if not autonomy_enabled() or not auto_optimize_enabled():
        return None
    state = _load_state(home)
    if not force and not _due(state, "optimizer",
                              OPTIMIZER_INTERVAL_S, now):
        return None
    _now = time.time() if now is None else now
    job = _job_slot(state, "optimizer")
    job["last_run_epoch"] = _now
    per_target = job.get("per_target")
    if not isinstance(per_target, dict):  # the B3 coercion rule
        per_target = job["per_target"] = {}
    target = _pick_target(per_target, _now)
    if target is None:
        # Every target attempted within its 7d window: the day's
        # decision is "nothing to do", which is health, not a stall —
        # the outcome field says so for the liveness probe.
        job["last_outcome"] = "nothing_due"
        job["last_summary"] = "all targets fresh (7d/target)"
        log("optimizer: all targets fresh — nothing to launch")
        _save_state(home, state)
        return None
    _blocked = _optimizer_preflight(home)
    if _blocked is not None:
        job["last_outcome"] = "stood_down"
        job["last_summary"] = f"stood down: {_blocked}"
        log(f"optimizer: stood down — {_blocked}")
        _save_state(home, state)
        return None
    job["last_outcome"] = "ran"
    job["last_target"] = target
    tslot = per_target.setdefault(target, {})
    if not isinstance(tslot, dict):
        tslot = per_target[target] = {}
    tslot["last_attempt_epoch"] = _now
    script, args, banner, promoted_marker = _target_command(target, home)
    log(f"optimizer: launching {target} ({script}) — this can run for "
        f"hours on the main slot")
    try:
        proc = run_script(script, args, home=home,
                          timeout_s=OPTIMIZER_TIMEOUT_S)
        rc: Optional[int] = proc.returncode
        out_full = proc.stdout or ""
        tail = "\n".join(out_full.strip().splitlines()[-8:])
        if rc != 0 and (proc.stderr or "").strip():
            tail += "\n[stderr] " + "\n".join(
                proc.stderr.strip().splitlines()[-6:])
        # ⚠ AN EXIT CODE IS BELIEVED ONLY WITH ITS MARKER. The banner
        # catches the triply-overloaded exit 2 (argparse, missing file,
        # COULD_NOT_MEASURE); the per-code markers catch a crash
        # impersonating a verdict — and here REJECTED is LOG-ONLY, so a
        # permanently crashing optimizer without this check would read
        # as "rejected weekly" forever (the judge-A1 shape).
        _cause = f"rc-{rc}"
        if banner not in out_full:
            rc, _cause = None, "no-banner"
            tail = ("[no '" + banner + "' banner — the script did not "
                    "start: bad argv, moved script, or broken "
                    "interpreter]\n" + tail)
        elif rc != GateExit.PROMOTED and promoted_marker in out_full:
            # ⚠ A PROMOTED MARKER BESIDE A NON-ZERO EXIT IS A PARTIAL
            # PROMOTION (§4DF round 1, MAJOR-4): components that
            # reached disk ARE live (the epoch swap deploys whatever it
            # finds), and the generic "nothing was acted on" text was a
            # false claim about exactly this world. The gate now stages
            # all before swapping any, so this is the residual
            # crash-between-renames window — rare, and worth a look.
            rc, _cause = None, "partial-promotion"
            tail = ("[the '" + promoted_marker + "' marker is present "
                    "but the exit was " + str(proc.returncode)
                    + " — a PARTIAL promotion: the components named in "
                    "the PROMOTED lines ARE live via the epoch swap "
                    "while the rest of the run failed. Inspect "
                    "system/optim/.]\n" + tail)
        elif rc == GateExit.PROMOTED and promoted_marker not in out_full:
            rc, _cause = None, "no-promoted-marker"
            tail = ("[exit 0 without the '" + promoted_marker
                    + "' marker — not a promotion]\n" + tail)
        elif rc == GateExit.REJECTED \
                and GATE_REJECTED_MARKER not in out_full:
            rc, _cause = None, "no-rejected-marker"
            tail = ("[exit 1 without the '" + GATE_REJECTED_MARKER
                    + "' marker — a crash, not a verdict]\n" + tail)
        elif rc == GateExit.NO_CANDIDATE \
                and GATE_NO_CANDIDATE_MARKER not in out_full:
            rc, _cause = None, "no-no-candidate-marker"
            tail = ("[exit 3 without the '" + GATE_NO_CANDIDATE_MARKER
                    + "' marker — a crash, not a verdict]\n" + tail)
    except Exception as e:  # noqa: BLE001 — timeout / spawn failure
        rc, tail = None, f"{type(e).__name__}: {e}"
        _cause = type(e).__name__
        # ⚠ A TIMEOUT KILL CARRIES THE CHILD'S OUTPUT (§4DF round 2):
        # `TimeoutExpired.stdout` holds everything printed before the
        # kill, and discarding it filed a timeout AFTER the PROMOTED
        # lines as "nothing was believed or acted on" — the MAJOR-4
        # false claim, back through the timeout door.
        _t_out = getattr(e, "stdout", None) or getattr(e, "output", None)
        if _t_out:
            if isinstance(_t_out, bytes):
                _t_out = _t_out.decode("utf-8", "replace")
            tail += "\n" + "\n".join(str(_t_out).strip().splitlines()[-8:])
            # Banner required, like the live path — a marker without
            # proof the script started is believed nowhere (round 3).
            if banner in str(_t_out) and promoted_marker in str(_t_out):
                _cause = "partial-promotion"
    tslot["last_exit"] = rc
    tslot["last_summary"] = tail[-2000:]
    if rc == GateExit.PROMOTED:
        tslot["last_outcome"] = "promoted"
        # A promotion is at most one per target per 7 days — each one is
        # genuinely news, so this notifies directly rather than
        # per-condition. The §4DE epoch swap will separately announce
        # the DEPLOY when it lands (~a minute).
        tslot["last_notified_condition"] = "promoted"
        notify(f"GEPA OPTIMIZER: {target} ran the gate and a candidate "
               f"was PROMOTED. The epoch swap deploys it live within "
               f"~a minute (§4DE) and announces itself; the daily judge "
               f"now watches it.\n{tail}")
        log(f"optimizer: {target} PROMOTED")
    elif rc == GateExit.REJECTED:
        # The system working as designed — measured, and the incumbent
        # stands. The full record is on disk (`.candidate.rejected`).
        tslot["last_outcome"] = "rejected"
        tslot["last_notified_condition"] = "rejected"
        log(f"optimizer: {target} measured a candidate and REJECTED it "
            f"— incumbent stands")
    elif rc == GateExit.COULD_NOT_MEASURE:
        # The gate's own pre-flights refusing (supply, re-draw age,
        # upstream health) or a mid-run abort: routine, log-only.
        tslot["last_outcome"] = "could_not_measure"
        tslot["last_notified_condition"] = "insufficient"
        _last = tail.splitlines()[-1] if tail else ""
        log(f"optimizer: {target} could not measure"
            + (f" — {_last}" if _last else ""))
    elif rc == GateExit.NO_CANDIDATE:
        # A wasted run or a broken reflection LM — news ONCE; weekly
        # repeats of the same condition are not.
        tslot["last_outcome"] = "no_candidate"
        _notify_once(
            tslot, "no-candidate", notify,
            f"GEPA OPTIMIZER: {target} ran to completion but the "
            f"optimizer produced NO candidate (returned the seed "
            f"verbatim). One wasted run is luck; repeats suggest a "
            f"broken reflection LM.\n{tail}")
        log(f"optimizer: {target} produced no candidate")
    elif _cause == "partial-promotion":
        # The one instrument failure whose honest message is NOT
        # "nothing was acted on": promoted components ARE live — so it
        # notifies DIRECTLY, like the clean promotion arm (§4DF round
        # 2, MIN-2: MUT-G's law is "a deploy must always notify", and
        # routing this arm through `_notify_once` swallowed the second
        # of two consecutive partial deploys).
        tslot["last_outcome"] = "instrument_failure"
        tslot["last_notified_condition"] = "partial-promotion"
        notify(
            f"GEPA OPTIMIZER: {script} for {target} printed a PROMOTED "
            f"marker but did not exit 0 (a non-zero exit, or killed at "
            f"the deadline) — a PARTIAL promotion. The "
            f"components named in the PROMOTED lines are LIVE (the "
            f"epoch swap deploys them); the rest of the run failed. "
            f"Inspect system/optim/.\n{tail}")
        log(f"optimizer: {target} PARTIAL promotion — inspect "
            f"system/optim/")
    else:
        tslot["last_outcome"] = "instrument_failure"
        _notify_once(
            tslot, f"instrument:{_cause}", notify,
            f"GEPA OPTIMIZER: {script} did not run cleanly for {target} "
            f"(exit={rc}, cause={_cause}). Instrument failure — nothing "
            f"was believed or acted on.\n{tail}")
        log(f"optimizer: {target} instrument failure exit={rc}")
    _save_state(home, state)
    return (target, rc)
