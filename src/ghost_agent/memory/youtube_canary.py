"""YouTube route canary (§4KH, 2026-09-24).

YouTube periodically rotates the BotGuard challenge; when it does, the local
PO-token helper mints tokens YouTube no longer honours and every Tor circuit
is refused ("Sign in to confirm you're not a bot") until yt-dlp and the
helper are updated together. Until this module the agent only found that out
when someone next pasted a link. Now it checks on its own.

**What one probe does.** Helper ``/ping``, Tor liveness, then one
``fetch_info`` of a fixed public video (the 19-second "Me at the zoo") on
up to four fresh circuits — resolve only: no captions, no audio, no store
writes. Measured 2026-09-24: with a working token about 4 in 10 circuits
pass and failures take 2–9 s, so four circuits separate "one bad exit"
(≈13% all-fail) from "the wall" cheaply; a WALLED verdict is confirmed on
the next run before it is announced twice.

**States.** ``ok``, ``walled`` (every circuit bot-walled with the helper
up → rotation suspected), ``helper_down``, ``tor_down``, ``error`` (a
terminal fault: the canary video removed, yt-dlp broken).

**Fire once per transition.** Entering a failure state records ONE
``notify``-severity activity (→ Slack DM to the owner, webhook/ntfy); the
return to ``ok`` records one recovery notice; a steady state records nothing
— an alarm that repeats every day is one the operator learns to ignore. The
current state is always readable at ``/api/health`` (``youtube_route``) and
in ``$GHOST_HOME/system/youtube_canary.json``.

**Cadence.** ``GHOST_YT_CANARY_HOURS`` (default 24; ``0`` disables); the
first run ``GHOST_YT_CANARY_BOOT_DELAY_S`` (default 600) after boot so a
fresh deploy gets a baseline soon. Driven from the biological tick in a
worker thread — never on the event loop, never while a foreground turn is
in flight, never overlapping itself.

The remedy is deliberately NOT automatic: ``bin/update-youtube-stack.sh``
upgrades third-party code in the agent's venv and restarts a daemon, which
the operator chose to keep as a one-command manual step.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..utils.helpers import env_positive

logger = logging.getLogger(__name__)

CANARY_VIDEO_ID = "jNQXAC9IVRw"          # "Me at the zoo" — public since 2005, 19 s
CANARY_URL = f"https://www.youtube.com/watch?v={CANARY_VIDEO_ID}"
CANARY_ATTEMPTS = max(1, int(env_positive("GHOST_YT_CANARY_ATTEMPTS", 4)))


def _interval_from_env() -> float:
    """Hours between probes. ``0`` disables; a typo falls back to 24 (never
    raises at import — the `env_positive` lesson, but here zero is a choice)."""
    raw = str(os.environ.get("GHOST_YT_CANARY_HOURS", "24")).strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = 24.0
    return max(0.0, hours) * 3600.0


INTERVAL_S = _interval_from_env()
BOOT_DELAY_S = env_positive("GHOST_YT_CANARY_BOOT_DELAY_S", 600.0)
# A HELD verdict (all circuits failed once) is re-checked after this, not
# after the full interval: confirmation in ~15 min, not 24–48 h.
RECHECK_S = env_positive("GHOST_YT_CANARY_RECHECK_S", 900.0)
REMEDY = "bin/update-youtube-stack.sh"
PHASE = "youtube_canary"
HISTORY_KEEP = 30

STATE_OK, STATE_WALLED, STATE_UNSTABLE, STATE_HELPER_DOWN, STATE_TOR_DOWN, STATE_ERROR = (
    "ok", "walled", "unstable", "helper_down", "tor_down", "error")
FAILURE_STATES = (STATE_WALLED, STATE_UNSTABLE, STATE_HELPER_DOWN, STATE_TOR_DOWN, STATE_ERROR)
#: Verdicts an EXHAUSTED rotation produces. With ~4 in 10 circuits passing,
#: all four failing by chance is ≈13% per run whatever the mix of causes —
#: so both are held until a second consecutive run confirms them. A run that
#: stopped EARLY (fewer attempts than allowed) hit a terminal fault and is
#: `error` at once.
HELD_STATES = (STATE_WALLED, STATE_UNSTABLE)


def enabled() -> bool:
    return INTERVAL_S > 0


@dataclass
class CanaryRun:
    ts: float
    state: str
    attempts: int = 0
    bot_walls: int = 0
    reason: str = ""
    duration_s: float = 0.0
    title: str = ""


@dataclass
class CanaryState:
    state: str = ""            # "" = never run
    since: float = 0.0         # when the current state began
    last_run: float = 0.0
    last: Optional[dict] = None
    history: List[dict] = field(default_factory=list)
    announced: str = ""        # the last state an activity record was written for

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CanaryState":
        st = cls()
        if isinstance(d, dict):
            st.state = str(d.get("state") or "")
            st.since = float(d.get("since") or 0.0)
            st.last_run = float(d.get("last_run") or 0.0)
            st.last = d.get("last") if isinstance(d.get("last"), dict) else None
            st.history = [h for h in (d.get("history") or []) if isinstance(h, dict)][-HISTORY_KEEP:]
            st.announced = str(d.get("announced") or "")
        return st


# ── persistence ─────────────────────────────────────────────────────────────

def state_path(context) -> Optional[Path]:
    """``$GHOST_HOME/system/youtube_canary.json`` (beside scheduled_tasks.json)."""
    mem = getattr(context, "memory_dir", None)
    if not mem:
        return None
    return Path(str(mem)).parent / "youtube_canary.json"


def load_state(path: Optional[Path]) -> CanaryState:
    try:
        if path and path.exists():
            return CanaryState.from_dict(json.loads(path.read_text(encoding="utf-8") or "{}"))
    except Exception as exc:  # noqa: BLE001 — a corrupt file is "never run"
        logger.warning("youtube canary state unreadable (%s); starting fresh", exc)
    return CanaryState()


def save_state(path: Optional[Path], st: CanaryState) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(st.to_dict()), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("youtube canary state not saved: %s", exc)


# ── the probe ───────────────────────────────────────────────────────────────

def probe(tor_proxy: Optional[str], *, attempts: int = CANARY_ATTEMPTS,
          ping: Optional[Callable[[], Optional[str]]] = None,
          tor_ok: Optional[Callable[[str], bool]] = None,
          resolve: Optional[Callable] = None,
          now: Callable[[], float] = time.time) -> CanaryRun:
    """One measurement → a :class:`CanaryRun`. Every dependency is injectable
    (the helper ping, Tor liveness, the resolver) so tests never touch the
    network; production wires the real ones."""
    from . import youtube_ingest as yi
    t0 = now()
    ping = ping or yi.pot_ping
    if tor_ok is None:
        from ..utils.egress_guard import tor_liveness_ok
        tor_ok = tor_liveness_ok
    resolve = resolve or (lambda url, vid, proxy, stats, n: yi.fetch_info(
        url, vid, proxy, run=yi._run_subprocess, stats=stats, attempts=n))

    if not tor_proxy or not tor_ok(tor_proxy):
        return CanaryRun(ts=t0, state=STATE_TOR_DOWN, reason="Tor proxy not configured or not answering",
                         duration_s=now() - t0)
    if not ping():
        return CanaryRun(ts=t0, state=STATE_HELPER_DOWN,
                         reason=f"PO-token helper at {yi.POT_URL} does not answer /ping",
                         duration_s=now() - t0)
    stats = yi.FetchStats()
    info, reason, _proxy = resolve(CANARY_URL, CANARY_VIDEO_ID, tor_proxy, stats, attempts)
    if info:
        return CanaryRun(ts=t0, state=STATE_OK, attempts=stats.attempts, bot_walls=stats.bot_walls,
                         title=str(info.get("title") or ""), duration_s=now() - t0)
    exhausted = stats.attempts >= attempts
    if exhausted and stats.bot_walls == stats.attempts:
        return CanaryRun(ts=t0, state=STATE_WALLED, attempts=stats.attempts, bot_walls=stats.bot_walls,
                         reason=str(reason)[:200], duration_s=now() - t0)
    if exhausted:
        # every circuit failed, but not all by the bot wall (403s, timeouts,
        # a SABR-only session): a bad night for Tor, or the wall plus noise —
        # held and re-checked, never paged from one run.
        return CanaryRun(ts=t0, state=STATE_UNSTABLE, attempts=stats.attempts, bot_walls=stats.bot_walls,
                         reason=str(reason)[:200], duration_s=now() - t0)
    return CanaryRun(ts=t0, state=STATE_ERROR, attempts=stats.attempts, bot_walls=stats.bot_walls,
                     reason=str(reason)[:200], duration_s=now() - t0)


# ── the ledger: transitions, announced once ────────────────────────────────

def _message(run: CanaryRun, previous: str) -> str:
    if run.state == STATE_OK:
        return (f"YouTube route RECOVERED: the canary video resolved over Tor "
                f"(attempt {run.attempts}/{CANARY_ATTEMPTS}) after being {previous or 'unknown'}.")
    if run.state == STATE_WALLED:
        return (f"YouTube route WALLED: all {run.attempts} circuits were refused ('not a bot') even with a "
                f"PO token — YouTube has most likely rotated its challenge. Remedy (manual): {REMEDY} "
                f"(upgrades yt-dlp + the helper together and restarts com.local.ghost-pot).")
    if run.state == STATE_UNSTABLE:
        return (f"YouTube route UNSTABLE: two probes in a row exhausted all {run.attempts} circuits with mixed "
                f"faults ({run.bot_walls} bot walls, the rest 403/timeouts). Tor may be having a bad day; if it "
                f"persists, treat it as a rotation: {REMEDY}.")
    if run.state == STATE_HELPER_DOWN:
        return (f"YouTube route DOWN: the PO-token helper does not answer — check "
                f"`sudo launchctl print system/com.local.ghost-pot` and ~/Data/AI/Logs/ghost-pot.err.")
    if run.state == STATE_TOR_DOWN:
        return "YouTube route DOWN: Tor is not answering, so no YouTube link can be fetched."
    return (f"YouTube route ERROR: the canary video could not be resolved and it was not the bot wall — "
            f"{run.reason or 'no detail'}. If yt-dlp itself is broken the remedy is {REMEDY}.")


def pending_verdict(st: CanaryState) -> str:
    """The HELD verdict of the last run, if it differs from the settled state
    (``""`` when nothing is pending). What the re-check clock and the health
    view read."""
    last = (st.last or {}).get("state", "")
    return last if last in HELD_STATES and last != st.state else ""


def apply_run(st: CanaryState, run: CanaryRun, *, record: Optional[Callable[[str, str, str, dict], object]] = None,
              log: Optional[Callable[[str, str], None]] = None) -> CanaryState:
    """Fold one run into the persisted state; announce ONLY a transition.

    An exhausted-rotation verdict (``walled`` / ``unstable``) is announced on
    the run that CONFIRMS it (the second consecutive one): four circuits all
    failing by chance is ≈13% with a healthy token, whatever the mix of
    causes, so a single all-fail is noted, re-checked after RECHECK_S, and
    only then paged. Early-stopping faults (``error``, helper/Tor down) are
    announced at once. ``announced`` moves only when ``record`` accepted the
    message (a lost DM is retried on the next run, not forgotten).
    """
    prev = st.state
    entering = run.state != prev
    st.last_run = run.ts
    st.last = asdict(run)
    st.history = (st.history + [asdict(run)])[-HISTORY_KEEP:]
    confirmed = (run.state in HELD_STATES and len(st.history) >= 2
                 and st.history[-2].get("state") == run.state)
    if entering:
        if run.state in HELD_STATES and not confirmed:
            if log:
                log("info", f"canary: all {run.attempts} circuits failed once ({run.state}) — re-checking in "
                            f"{RECHECK_S / 60:.0f} min before announcing")
            return st
        st.state, st.since = run.state, run.ts
    if st.state != st.announced:
        if st.state in FAILURE_STATES or (st.state == STATE_OK and st.announced):
            msg = _message(run, prev if prev else st.announced)
            accepted = True
            if record:
                accepted = record(PHASE, msg, "notify", {"state": st.state, "attempts": run.attempts,
                                                        "bot_walls": run.bot_walls, "remedy": REMEDY}) is not False
            if log:
                log("WARNING" if st.state in FAILURE_STATES else "info", msg)
            if accepted:
                st.announced = st.state
        else:
            st.announced = st.state
    elif log:
        log("info", f"canary: {st.state} (attempt {run.attempts}/{CANARY_ATTEMPTS}, {run.duration_s:.0f}s)")
    return st


# ── scheduling from the biological tick ────────────────────────────────────

_lock = threading.Lock()
_running = False
# The boot clock lives on `app.state`, not on the context (neither branch
# below finds one in production), so the module's first-import time stands in:
# the tick imports this ~60 s after the loop starts, close enough to "boot" for
# a 10-minute delay whose purpose is to let Tor and the sandbox settle.
_process_start_monotonic = time.monotonic()


def due(st: CanaryState, *, boot_monotonic: Optional[float], now: float,
        monotonic_now: float, interval_s: float = INTERVAL_S, boot_delay_s: float = BOOT_DELAY_S,
        recheck_s: float = RECHECK_S) -> bool:
    if interval_s <= 0:
        return False
    if boot_monotonic is not None and monotonic_now - boot_monotonic < boot_delay_s:
        return False
    if st.last_run > now:
        return True   # a clock that went backwards / a restored file: do not wait it out
    wait = min(interval_s, recheck_s) if pending_verdict(st) else interval_s
    return (now - st.last_run) >= wait


def maybe_run(agent, *, force: bool = False) -> bool:
    """Called from the biological tick (~60 s). Cheap when nothing is due.
    Returns True when a probe was STARTED (in a worker thread)."""
    global _running
    if not enabled() and not force:
        return False
    ctx = getattr(agent, "context", None)
    if ctx is None:
        return False
    lc = getattr(ctx, "llm_client", None)
    if not force and (getattr(lc, "foreground_tasks", 0) > 0 or getattr(lc, "foreground_requests", 0) > 0):
        return False
    path = state_path(ctx)
    st = load_state(path)
    boot = getattr(ctx, "boot_monotonic", None)
    if boot is None:
        boot = getattr(getattr(agent, "app_state", None), "boot_monotonic", None)
    if boot is None:
        boot = _process_start_monotonic
    if not force and not due(st, boot_monotonic=boot, now=time.time(), monotonic_now=time.monotonic()):
        return False
    with _lock:
        if _running:
            return False
        _running = True

    def _worker():
        global _running
        try:
            from ..utils.logging import Icons, pretty_log
            try:
                run = probe(getattr(ctx, "tor_proxy", None))
            except Exception as exc:  # noqa: BLE001 — a broken probe is a verdict, not a retry-every-tick
                run = CanaryRun(ts=time.time(), state=STATE_ERROR,
                                reason=f"probe raised {type(exc).__name__}: {exc}"[:200])
            fresh = load_state(path)   # re-read: the file may have moved on

            def _record(phase, msg, severity, meta):
                log_obj = getattr(ctx, "activity_log", None)
                if log_obj is not None and hasattr(log_obj, "record"):
                    return log_obj.record(phase, msg, severity=severity, **meta)   # True/False
                rec = getattr(agent, "_record_autonomous_activity", None)
                if callable(rec):
                    rec(phase, msg, severity=severity, **meta)
                    return True
                return False

            def _log(level, msg):
                pretty_log("YouTube Canary", msg, level=level.upper() if level != "info" else "INFO",
                           icon=Icons.WARN if level.upper() == "WARNING" else Icons.OK)
            apply_run(fresh, run, record=_record, log=_log)
            save_state(path, fresh)
        except Exception as exc:  # noqa: BLE001 — the canary must never break the tick
            logger.warning("youtube canary run failed: %s", exc)
        finally:
            with _lock:
                _running = False

    threading.Thread(target=_worker, name="youtube-canary", daemon=True).start()
    return True


def _iso(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat() if ts else ""


def health_view(context) -> Dict[str, object]:
    """The ``youtube_route`` field of ``/api/health``: the settled state, the
    last run (its own verdict and reason — a HELD wall is visible here as
    ``pending``), and when. Keyed on ``last_run``, not on the settled state,
    so a first run that is being held still shows as a run."""
    try:
        st = load_state(state_path(context))
    except Exception:  # noqa: BLE001
        return {"state": "unknown"}
    base = {"enabled": enabled(), "interval_h": INTERVAL_S / 3600.0}
    if not st.last_run:
        return dict(base, state="not_yet_run")
    last = st.last or {}
    pending = pending_verdict(st)
    return dict(base, **{
        "state": st.state or "pending", "since": st.since, "since_iso": _iso(st.since),
        "last_run": st.last_run, "last_run_iso": _iso(st.last_run),
        "last_state": last.get("state"), "last_attempts": last.get("attempts"),
        "last_reason": (last.get("reason") or "")[:160],
        "pending": pending, "recheck_in_s": (RECHECK_S if pending else None),
        "remedy": REMEDY if (st.state in FAILURE_STATES or pending) else "",
    })
