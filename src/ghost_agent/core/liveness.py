"""Is each mechanism ALIVE? — a probe per subsystem, not a count per ledger.

⚠ WHY THIS EXISTS, and why `learning_health.activity_liveness` was not enough.

That function (§4AD) made absence a ROW rather than a silence — but only for the
18 phases that write to ``autonomous_activity.jsonl``. The most important dead
loop found on 2026-08-10 — metacog **arbitration, 0 firings against 118
below-threshold opportunities** — does not write there, and was found by
grepping the log BY HAND. A dead-loop detector blind to the exact class of
thing that justified building it is the defect it was built to remove, one
level up.

So: mechanisms declare WHERE THEIR OWN DURABLE EVIDENCE LIVES, and this reads
it. Non-invasive by design — no production write path changes, no new ledger
volume. A probe that cannot find its source says so.

THE THIRD STATE, which is the whole point:

    FIRED      evidence exists and the mechanism ran
    ZERO       evidence source EXISTS and genuinely shows nothing
    NO_SOURCE  there is no durable way to tell     <-- an instrumentation GAP
    GATED      switched off on purpose; a count would be meaningless

`activity_liveness` conflates the middle two: a phase absent from the ledger
reads as zero whether it is quiet or unobservable. That is the same
missing-vs-empty conflation removed from the response cache in §4AC, and it is
precisely how arbitration stayed invisible. NO_SOURCE is not a diagnostic
nicety — it is the finding.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

FIRED = "fired"
ZERO = "zero"
NO_SOURCE = "no_source"
GATED = "gated"

# What a ZERO means for this mechanism (mirrors autonomous_activity's vocabulary
# so an operator reads one scale across both views).
EXPECT_PERIODIC = "periodic"
EXPECT_ON_OUTPUT = "on_output"
EXPECT_ON_DEMAND = "on_demand"
EXPECT_GATED = "gated"

# WHICH DENOMINATOR EXPLAINS THIS MECHANISM'S SILENCE (2026-08-11). Separate
# from the expectation vocabulary above, which describes CADENCE for the
# reader; this one decides whether a zero is excusable, and the two are not the
# same axis. `router.decisions` is rendered "periodic" but routes on EVERY
# request including self-play, so a traffic-free box explains it while a quiet
# *user* day does not. Getting these backwards is how a guard silences the
# alarm it exists to raise.
DEN_USER_TURNS = "user_turns"   # simulation-gated: only REAL turns feed it
DEN_REQUESTS = "requests"       # any request, self-play included
DEN_NONE = "none"               # idle-clock: no amount of quiet excuses it


@dataclass
class ProbeResult:
    status: str
    count: Optional[int] = None
    last_ts: Optional[float] = None
    note: str = ""
    #: §4CS item E: an explicit alarm, for a mechanism whose BAD state is
    #: not a zero. A negative-control run that fired two of three controls
    #: has count=2 and is FIRED — and is exactly the thing an operator
    #: must be told about, because the one that did not fire is a guard
    #: presumed dead. `alarm_if_zero` cannot express that.
    alarm: bool = False


@dataclass
class Probe:
    name: str
    expectation: str
    # Human-readable path/description of the evidence. Printed, so an operator
    # can go look at the same thing the probe looked at.
    source: str
    fn: Callable[[Path], ProbeResult]
    # Only mechanisms whose zero is genuinely anomalous may alarm. A monitor
    # that cries on a benign zero is one the operator learns to scroll past —
    # the lesson §4AD already paid for.
    alarm_if_zero: bool = False
    # Which absence of traffic (if any) makes this mechanism's zero benign.
    # DEN_NONE is the deliberate default: a mechanism must EARN its excuse.
    denominator: str = DEN_NONE


def _home(ghost_home: Optional[Path] = None) -> Path:
    return Path(ghost_home or os.getenv("GHOST_HOME", "")) if (
        ghost_home or os.getenv("GHOST_HOME")) else Path.home()


def _parse_ts(v: Any) -> Optional[float]:
    """Epoch float from either a number OR an ISO-8601 string.

    ⚠ REVIEW ROUND 1. The first version did `float(v)` only. The live stores
    write ISO strings — `'2026-08-05T11:15:25.881934Z'` — so every record threw
    ValueError and was skipped, and both `foresight.predictions` and
    `rrf.observations` reported ZERO with a "no parseable ts" note. Two probes
    reading healthy stores as silent: the instrument-reads-as-dead failure this
    module exists to remove, committed inside it.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) or None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(
            s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _jsonl_probe(rel: str, *, window_h: float) -> Callable[[Path], ProbeResult]:
    """Count records in a JSONL whose `ts` falls inside the window."""
    def probe(home: Path) -> ProbeResult:
        p = home / rel
        if not p.exists():
            return ProbeResult(NO_SOURCE, note=f"{rel} absent")
        cutoff = time.time() - window_h * 3600.0
        n, last = 0, None
        try:
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:            # noqa: BLE001
                        continue
                    ts = _parse_ts(rec.get("ts") or rec.get("timestamp"))
                    if ts:
                        last = max(last or 0.0, ts)
                        if ts >= cutoff:
                            n += 1
        except OSError as e:
            return ProbeResult(NO_SOURCE, note=f"unreadable: {e}")
        # ⚠ mtime is the FALLBACK, never the primary: a file touched by a
        # backup or an editor would read as activity.
        if last is None:
            try:
                last = p.stat().st_mtime
                return ProbeResult(ZERO if n == 0 else FIRED, n, last,
                                   "no parseable ts — mtime shown, treat as weak")
            except OSError:
                return ProbeResult(NO_SOURCE, note="no ts and no mtime")
        return ProbeResult(FIRED if n else ZERO, n, last)
    return probe


def _mtime_probe(rel: str, *, stale_h: float) -> Callable[[Path], ProbeResult]:
    """For stores rewritten in place (params, checkpoints) where mtime IS the
    fire signal — there is no per-event record to count."""
    def probe(home: Path) -> ProbeResult:
        p = home / rel
        if not p.exists():
            return ProbeResult(NO_SOURCE, note=f"{rel} absent")
        try:
            m = p.stat().st_mtime
        except OSError as e:
            return ProbeResult(NO_SOURCE, note=f"unreadable: {e}")
        age_h = (time.time() - m) / 3600.0
        return ProbeResult(FIRED if age_h <= stale_h else ZERO, None, m,
                           "rewritten-in-place; mtime is the only signal")
    return probe


# ⚠ REVIEW ROUND 2: the log is 14 MB and THREE probes each read it end to end,
# costing ~1.9 s per `probe_all()`. Parsed once per (path, size, mtime) and
# shared. Keyed on size+mtime so an appended log invalidates the entry rather
# than serving a stale count — a monitor caching its own staleness would be a
# neat way to reinvent the defect this module exists to catch.
_LOG_CACHE: Dict[tuple, List[tuple]] = {}


def _log_entries(p: Path) -> Optional[List[tuple]]:
    """[(epoch_ts, line)] for timestamped mirror lines. One parse, shared."""
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, st.st_mtime)
    hit = _LOG_CACHE.get(key)
    if hit is not None:
        return hit
    _LOG_CACHE.clear()          # only ever hold the current log
    ts_rx = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    out: List[tuple] = []
    try:
        with p.open(errors="replace") as fh:
            for line in fh:
                m = ts_rx.match(line)
                if not m:
                    continue
                try:
                    out.append((time.mktime(time.strptime(
                        m.group(1), "%Y-%m-%d %H:%M:%S")), line))
                except ValueError:
                    continue
    except OSError:
        return None
    _LOG_CACHE[key] = out
    return out


def _log_probe(pattern: str, *, window_h: float,
               rel: str = "system/ghost-agent.log",
               exclude: Optional[str] = None
               ) -> Callable[[Path], ProbeResult]:
    """Count matching log lines inside the window.

    The log is the durable record for mechanisms that emit a line but keep no
    store of their own. Lines carry a leading 'YYYY-MM-DD HH:MM:SS' stamp.
    """
    rx = re.compile(pattern)
    # ⚠ FALSE-GREEN GUARD (review round 2). A mechanism's NAME appearing in a
    # log line is not evidence it RAN: the agent's own reasoning is mirrored at
    # DEBUG, so "thinking — the verifier refuted my claim …" matches a verifier
    # pattern while proving nothing. Measured: 12 of 1370 matches were prose.
    # Only 0.9% today — but if verification ever genuinely stopped, that prose
    # would keep the probe GREEN, which is the failure nobody investigates.
    ex = re.compile(exclude) if exclude else None

    def probe(home: Path) -> ProbeResult:
        p = home / rel
        if not p.exists():
            return ProbeResult(NO_SOURCE, note=f"{rel} absent")
        entries = _log_entries(p)
        if entries is None:
            return ProbeResult(NO_SOURCE, note=f"{rel} unreadable")
        cutoff = time.time() - window_h * 3600.0
        n, last = 0, None
        for ts, line in entries:
            if not rx.search(line):
                continue
            if ex is not None and ex.search(line):
                continue
            last = max(last or 0.0, ts)
            if ts >= cutoff:
                n += 1
        return ProbeResult(FIRED if n else ZERO, n, last)
    return probe


def _json_field_ts_probe(rel: str, field: str, *, stale_h: float
                         ) -> Callable[[Path], ProbeResult]:
    """Freshness from a timestamp INSIDE the file, not from its mtime.

    ⚠ REVIEW ROUND 2. `_mtime_probe` cannot tell a real write from a `touch`:
    a backup, an rsync, or an editor save makes a DEAD mechanism report FIRED.
    That is a false GREEN, which is worse than a false alarm — nobody
    investigates a green row. Where the store carries its own timestamp
    (`calibration_params.json` has `fitted_at`) the content is the honest
    signal and is immune to being touched.

    Reserved for stores that HAVE such a field. `router/checkpoint.json` does
    not, and its probe says so instead of pretending.
    """
    def probe(home: Path) -> ProbeResult:
        p = home / rel
        if not p.exists():
            return ProbeResult(NO_SOURCE, note=f"{rel} absent")
        try:
            d = json.loads(p.read_text())
        except Exception as e:                               # noqa: BLE001
            return ProbeResult(NO_SOURCE, note=f"unparseable: {type(e).__name__}")
        ts = _parse_ts(d.get(field))
        if ts is None:
            return ProbeResult(NO_SOURCE,
                               note=f"no usable '{field}' inside {rel}")
        age_h = (time.time() - ts) / 3600.0
        return ProbeResult(FIRED if age_h <= stale_h else ZERO, None, ts,
                           f"from '{field}' INSIDE the file — a touch cannot "
                           f"fake this")
    return probe


def _newest_child_probe(rel: str, *, stale_h: float
                        ) -> Callable[[Path], ProbeResult]:
    """Freshness of the newest child under a directory that grows by
    date-partition (one subdir per day) rather than by appending to a file."""
    def probe(home: Path) -> ProbeResult:
        d = home / rel
        if not d.is_dir():
            return ProbeResult(NO_SOURCE, note=f"{rel} absent")
        try:
            kids = list(d.iterdir())
        except OSError as e:
            return ProbeResult(NO_SOURCE, note=f"unreadable: {e}")
        if not kids:
            return ProbeResult(ZERO, 0, None, "directory empty")
        try:
            newest = max(k.stat().st_mtime for k in kids)
        except OSError:
            return ProbeResult(NO_SOURCE, note="children unreadable")
        age_h = (time.time() - newest) / 3600.0
        return ProbeResult(FIRED if age_h <= stale_h else ZERO,
                           len(kids), newest,
                           "date-partitioned; writes are SPARSE and selective "
                           "(observed 0-7 partitions/day against 24-48 turns), "
                           "so a quiet day is within range")
    return probe


def _count_user_turns(home: Path, window_h: float) -> tuple:
    """THE DENOMINATORS. Returns (user_turns, all_requests, reason).

    `all_requests` counts every request line and so is always knowable;
    `user_turns` needs the origin stamp and is None when the window cannot be
    classified. Two numbers because two different absences excuse two
    different silences — see DEN_* above.

    ⚠ ADDED IN REVIEW ROUND 1, after it nearly produced a false MAJOR. Three
    stores (trajectories, foresight, rrf) had all been quiet since ~02:00,
    which reads as a simultaneous triple failure. They are TURN-DRIVEN: with no
    user turns their silence is correct behaviour, and without this number a
    reader cannot tell "the mechanism broke" from "nothing asked it to run".
    It is the same shape as `activity_liveness`'s agent-down guard, one layer
    out.

    (In that instance the turns WERE present — 24 that day — and the real
    answer came from the base rate: writes are sparse, and a zero-day had
    already occurred on 08-07 with 29 turns. Two independent guards were needed
    to avoid the wrong call.)

    ⚠⚠ AND IT COUNTED THE WRONG THING UNTIL 2026-08-11, which is how it came
    to make the exact error it was built to prevent. Self-play/dream turns
    enter through the same `handle_chat` and emit the same `request started`
    line, so all of them landed in this count. Measured on the live log that
    day: 28 "user turns" in 24h of which **28 were self-play and 0 were real**
    — while foresight/rrf/trajectories were correctly silent because their
    simulation gates had excluded every one of those turns. The denominator
    did not merely fail to help, it argued FOR the false MAJOR, and it can
    reach 0 (the branch that withholds alarms) only on a box where the idle
    self-play loop is also dead. A guard whose triggering condition is
    unreachable in production is furniture.

    Now counts `origin=user` stamps only. A window containing request lines
    written by a pre-stamp build is reported as UNCLASSIFIED (None) rather
    than as a number that silently means something else — missing is UNKNOWN,
    never zero, and never an inflated count.
    """
    p = home / "system" / "ghost-agent.log"
    if not p.exists():
        return (None, 0, "no-log")
    cutoff = time.time() - window_h * 3600.0
    ts_rx = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*request started")
    origin_rx = re.compile(r"\borigin=(\w+)")
    n = 0
    total = 0
    unstamped = 0
    try:
        with p.open(errors="replace") as fh:
            for line in fh:
                m = ts_rx.match(line)
                if not m:
                    continue
                try:
                    ts = time.mktime(time.strptime(m.group(1),
                                                   "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                total += 1
                o = origin_rx.search(line)
                if o is None:
                    unstamped += 1
                elif o.group(1) == "user":
                    n += 1
    except OSError:
        return (None, 0, "no-log")
    if unstamped:
        # Mixed or wholly pre-stamp window: the stamped subset is a LOWER
        # bound on user turns, never the count. Say so instead of quoting it.
        return (None, total,
                f"unclassified ({unstamped} request line(s) predate the "
                f"origin stamp; {n} stamped user)")
    return (n, total, "")


def _literal_flag_from_source(path: Path, name: str) -> Optional[bool]:
    """Read a module-level boolean LITERAL without importing the module.

    Importing `core.agent` costs ~1.8 s; this costs milliseconds and executes
    nothing. Returns None when the name is absent OR is not a plain literal —
    if the flag ever becomes computed, this must decline rather than guess,
    because a wrong answer here reports a live mechanism as gated (or the
    reverse) and both are silent failures.
    """
    import ast
    try:
        tree = ast.parse(path.read_text())
    except Exception:                                        # noqa: BLE001
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, bool):
                    return node.value.value
                return None
    return None


def _trajectory_router_signal_probe(home: Path) -> ProbeResult:
    """Does the router's per-request decision reach the DURABLE corpus?

    Distinguishes three things that look alike from the outside:
      NO_SOURCE  no trajectory corpus at all — cannot tell;
      ZERO       trajectories exist and NONE carry a router field — the
                 signal is being computed and discarded (measured state);
      FIRED      the corpus carries it and downstream work is unblocked.

    Bounded to the newest few partitions: this runs in a report, and reading
    every trajectory ever written would make an operator screen cost seconds.
    """
    d = home / "system" / "trajectories"
    if not d.is_dir():
        return ProbeResult(NO_SOURCE, note="no trajectory corpus")
    try:
        parts = sorted((p for p in d.iterdir() if p.is_dir()),
                       key=lambda p: p.name)[-3:]
    except OSError as e:
        return ProbeResult(NO_SOURCE, note=f"unreadable: {e}")
    if not parts:
        return ProbeResult(NO_SOURCE, note="corpus has no date partitions")
    seen = withr = 0
    last = None
    for part in parts:
        try:
            files = list(part.iterdir())
        except OSError:
            continue
        for f in files:
            try:
                for line in f.open(errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:            # noqa: BLE001
                        continue
                    seen += 1
                    # ⚠ CHECK `extra` TOO — the router facts are NESTED there
                    # (agent.py merges turn_facts into `_extra`, which becomes
                    # `Trajectory.extra`). The first version of this probe
                    # scanned TOP-LEVEL KEYS ONLY, found nothing, and reported
                    # ZERO — and I filed that as a MAJOR ("the signal never
                    # reaches the corpus, router accuracy is UNMEASURABLE").
                    # It was a search bug: 65 of 1552 records carry it.
                    # Reading absence as evidence, inside the audit built to
                    # stop exactly that.
                    _extra = rec.get("extra")
                    _fields = list(rec) + (list(_extra)
                                           if isinstance(_extra, dict) else [])
                    if any("router" in k.lower() for k in _fields):
                        withr += 1
                        t = _parse_ts(rec.get("timestamp") or rec.get("ts"))
                        if t:
                            last = max(last or 0.0, t)
            except OSError:
                continue
    if seen == 0:
        return ProbeResult(NO_SOURCE, note="partitions present but no records")
    if withr == 0:
        return ProbeResult(
            ZERO, 0, None,
            f"0 of {seen} recent trajectory records carry a router field "
            f"(checked top level AND `extra`) — the per-request decision is "
            f"not reaching the corpus, so router accuracy cannot be measured "
            f"from it")
    return ProbeResult(
        FIRED, withr, last,
        f"{withr}/{seen} recent records carry the signal (under `extra`). "
        f"COVERAGE IS PARTIAL — trajectory writes are sparse and selective, "
        f"so check this subset for bias before validating the router on it")


def _arbitration_probe(home: Path) -> ProbeResult:
    """Metacog arbitration — the loop that was dead and invisible.

    Reports the GATE first. Its call site is hard-gated by
    ``_METACOG_ARBITER_ENABLED`` in core/agent.py — a module constant,
    independent of every flag — so a firing count would be answering a
    question nobody is asking: it CANNOT fire, and saying "0" would invite the
    operator to hunt for a scheduling bug that does not exist.

    Measured 2026-08-10 across 209 metacog summaries: confidence computed 865
    times, `below_threshold` fired 118 times, arbitrations **0**.
    """
    # ⚠ REVIEW ROUND 2: this used `from . import agent`, which costs ~1.8 s
    # cold and was 90% of probe_all()'s entire runtime. In the LIVE agent the
    # module is already imported and it is free; standalone it dominates.
    # Prefer the loaded module (authoritative), else read the LITERAL from
    # source without executing it.
    import sys as _sys
    _mod = _sys.modules.get("ghost_agent.core.agent")
    if _mod is not None:
        enabled = bool(getattr(_mod, "_METACOG_ARBITER_ENABLED", False))
    else:
        enabled = _literal_flag_from_source(
            Path(__file__).with_name("agent.py"), "_METACOG_ARBITER_ENABLED")
        if enabled is None:
            return ProbeResult(
                NO_SOURCE,
                note="_METACOG_ARBITER_ENABLED not found as a module-level "
                     "literal — it may have become computed; read it directly")
    if not enabled:
        return ProbeResult(
            GATED, 0, None,
            "hard-disabled by _METACOG_ARBITER_ENABLED in core/agent.py — "
            "the calibration threshold's ONLY consumer")
    # Enabled: the durable trace is the shutdown summary line, which is
    # emitted per PROCESS, not per firing. Say so rather than implying a
    # live count.
    return _log_probe(r"arbitrations=\d+", window_h=168.0)(home)


# ── THE REGISTRY ─────────────────────────────────────────────────────────────
# Everything here is DELIBERATELY outside autonomous_activity.jsonl. The phases
# inside it stay with `learning_health.activity_liveness`; duplicating them
# would create the two-implementations drift this repo keeps being bitten by.
def _gepa_autonomy_probe(home: Path) -> ProbeResult:
    """§4DC — is the autonomous GEPA loop's own clock still advancing?

    Reads `system/gepa_autonomy_state.json`'s `last_run_epoch`
    fields — the jobs' persisted wall clock — rather than the activity
    ledger, whose rows are transition-only by design (a quiet week is
    healthy; a state file that stopped advancing is a dead loop wearing
    a quiet week's clothes). Staleness bounds come from the module that
    owns the cadence, so probe and phase cannot drift apart.
    """
    import json as _js

    from ..optim.autonomy import (
        LIVE_JUDGE_INTERVAL_S,
        OPTIMIZER_TARGET_INTERVAL_S,
        SUPPLY_WATCH_INTERVAL_S,
    )
    path = Path(home) / "system" / "gepa_autonomy_state.json"
    try:
        st = _js.loads(path.read_text())
    except FileNotFoundError:
        return ProbeResult(
            NO_SOURCE,
            note="no state file yet — the loop has never ticked (the "
                 "first probe comes one cooldown after boot; if this "
                 "persists across days, the phase is not being reached)")
    except Exception as e:  # noqa: BLE001
        return ProbeResult(
            NO_SOURCE, note=f"state file unreadable ({type(e).__name__})")
    now = time.time()
    #: 3x the cadence: one missed window is idle-alignment luck, three
    #: is a stopped schedule (the negctrl STALE_AFTER_S reasoning).
    bounds = {"live_judge": 3 * LIVE_JUDGE_INTERVAL_S,
              "supply_watch": 3 * SUPPLY_WATCH_INTERVAL_S,
              # §4DF: bounded on the 7d TARGET interval, not the daily
              # job interval — the job legitimately decides "nothing
              # due" most days and deep idle can be scarce, so a 3×1d
              # bound would false-alarm on any busy week. (The state
              # stamps `last_run_epoch` on every due probe, including
              # `nothing_due`, so 21d of silence really is a stall.)
              "optimizer": 3 * OPTIMIZER_TARGET_INTERVAL_S}
    fresh, stale_jobs, standing, last_ts = [], [], [], None
    for job, bound in bounds.items():
        slot = st.get(job) if isinstance(st.get(job), dict) else {}
        _last = slot.get("last_run_epoch")
        # `_last != _last` is NaN (§4DF round 1, MIN-6): NaN IS a float,
        # every comparison below is False, and the row printed as
        # "(nand ago)" — a hand-edited NaN reads as "never", same as the
        # jobs' own coercion.
        if not isinstance(_last, (int, float)) or _last != _last:
            stale_jobs.append(f"{job} (never)")
            continue
        last_ts = max(last_ts or 0, _last)
        # ⚠ A STAND-DOWN ADVANCES THE CLOCK BUT IS NOT A RUN. The jobs
        # stamp `last_run_epoch` before the preflight (so a stood-down
        # job does not retry every tick), which made a box below the
        # disk floor read FIRED for months (lens B, B2). The outcome
        # field separates the two — and it gets its OWN note (§4DF
        # round 1, MIN-5): the generic "the tick is not reaching the
        # phase (kill switch?)" named two causes the state itself
        # excludes — the tick DID reach the phase; the preflight
        # refused.
        if (now - _last) > bound:
            # ⚠ STALENESS OUTRANKS THE STAND-DOWN LABEL (§4DF round 2,
            # MAJOR-3). The round-1 wording fix checked `stood_down`
            # FIRST, so a job whose LAST act was a stand-down could
            # never raise "SCHEDULE STOPPED" — a probe note asserting
            # "the tick reaches the phase", in the present tense, about
            # a state file it just read as 100 days old. Driven: the
            # 100d-stopped and 5m-fresh stand-down worlds produced
            # byte-identical notes.
            _sd = (", last outcome was a stand-down"
                   if slot.get("last_outcome") == "stood_down" else "")
            stale_jobs.append(
                f"{job} ({(now - _last) / 86400.0:.1f}d ago{_sd})")
        elif slot.get("last_outcome") == "stood_down":
            standing.append(job)
        else:
            fresh.append(job)
    note = ""
    if fresh:
        note = "advancing: " + ", ".join(sorted(fresh))
    if standing:
        note += ("; " if note else "") + (
            "STANDING DOWN: " + ", ".join(sorted(standing))
            + " — the tick reaches the phase but the preflight refuses "
              "(disk/RAM floor); see last_summary in the state file")
    if stale_jobs:
        note += ("; " if note else "") + (
            "SCHEDULE STOPPED for " + ", ".join(sorted(stale_jobs))
            + " — the tick is not reaching the phase (kill switch? "
              "idle window never sampled?)")
    return ProbeResult(ZERO if not fresh else FIRED,
                       count=len(fresh), last_ts=last_ts, note=note)


def _negative_controls_probe(home: Path) -> ProbeResult:
    """E3 — do the cascade's negative controls still demonstrably FIRE?

    §4CS item E. E2's entire value is REFUSING things, and this project's
    own rule is that a guard which never demonstrably fires is presumed
    dead. One control run was performed by hand on 2026-08-23 and passed;
    it wrote its record into a throwaway home, so nothing durable recorded
    that it had ever happened.

    Three states matter and they are NOT the same:

      NO_SOURCE  never run here — an instrumentation gap, and the honest
                 answer on a fresh box. It must not read as a tick.
      alarm      the schedule has STOPPED (older than two intervals), or
                 a control FAILED TO FIRE. Note that the second is not a
                 zero — a run where 2 of 3 controls held has a positive
                 count and is the loudest thing on this view.
      FIRED      every selected control ran, held, and is fresh.
    """
    from ..evolve import negative_controls as NC
    doc = NC.last_run(home)
    if not doc:
        return ProbeResult(
            NO_SOURCE,
            note="never run on this box — E2's value is refusing things, "
                 "and a guard that has never demonstrably fired is "
                 "presumed dead. The scheduled phase writes this file.")
    ts = NC.last_run_ts(home)
    results = doc.get("results") or []
    held = [r for r in results
            if isinstance(r, dict) and r.get("ok") and r.get("verified")]
    failed = [str(r.get("name")) for r in results
              if isinstance(r, dict) and not (r.get("ok") and r.get("verified"))]
    unverified = [str(n) for n in (doc.get("unverified") or [])]
    selected = list(doc.get("selected") or [])
    missing = [n for n in selected
               if n not in {str(r.get("name")) for r in results
                            if isinstance(r, dict)}]
    stale = (ts is not None
             and (time.time() - ts) > NC.STALE_AFTER_S)
    problems = []
    if failed:
        problems.append("DID NOT FIRE: " + ", ".join(sorted(failed)))
    if missing:
        problems.append("never ran: " + ", ".join(sorted(missing)))
    if unverified and not failed:
        problems.append("unverified: " + ", ".join(sorted(unverified)))
    if not doc.get("all_ok") and not problems:
        # partial_ok with a short `selected` — green about a subset.
        problems.append(
            f"PARTIAL run — only {len(selected)} of "
            f"{len(NC.ALL_CONTROLS)} controls were selected, so this is "
            f"not evidence about the others")
    if stale:
        problems.append(
            f"the SCHEDULE HAS STOPPED — last run "
            f"{(time.time() - ts) / 86400.0:.1f} days ago, cadence is "
            f"{NC.INTERVAL_S / 86400.0:.0f} days")
    return ProbeResult(
        ZERO if not held else FIRED,
        count=len(held),
        last_ts=ts,
        alarm=bool(problems),
        note=("; ".join(problems) if problems else
              f"all {len(held)} control(s) fired and held "
              f"({'deep' if doc.get('deep') else 'shallow'} run)"),
    )


def _gepa_applies_probe(home: Path) -> ProbeResult:
    """Loader loads over 7 days — ANNOTATED when the loop is retired.

    ⚠ Two views on ONE screen said opposite things about the same loop.
    After the 2026-08-24 retirement, SUBSYSTEM LIVENESS printed
    `gepa.applies fired n=28 last 12.3h ago` fifty lines above LOOP
    YIELD's `† END prompts.gepa retired`. Both were literally true — the
    window is 168h and the loads are real history — and the older probe
    would keep reading FIRED for another seven days about a loop that is
    settled dead. Neither view knew about the other.

    A count inside a window is not wrong; a count presented without the
    fact that its subject has been withdrawn is. So this probe asks the
    yield side and says so.
    """
    # ⚠ THE SAME THREE LINES. A narrow pattern here read ZERO for a
    # fully-applied artifact whose load line happened to be the WARNING
    # form — and after round 13 there are two warning forms.
    res = _log_probe(_GEPA_LOAD_PATTERN, window_h=168.0)(home)
    try:
        d = home / "system" / "optim"
        live = [f for f in d.glob("*.json")] if d.is_dir() else []
        withdrawn = ([f for f in d.iterdir()
                      if f.is_file() and ".json." in f.name
                      and any(k in f.name
                              for k in ("retired", "rejected", "noop"))]
                     if d.is_dir() else [])
    except Exception:                                       # noqa: BLE001
        return res
    if withdrawn and not live and (res.count or 0):
        res.note = ((res.note + ". ") if res.note else "") + (
            f"⚠ these are HISTORICAL: every artifact has since been "
            f"withdrawn ({len(withdrawn)} on disk, 0 live), so the loads "
            f"in this 168h window are of prompts no longer served. See "
            f"LOOP YIELD `prompts.gepa`, which reads RETIRED")
    return res


PROBES: List[Probe] = [
    # ⚠ DEN_NONE, deliberately: these run off the idle clock, not off
    # traffic, so no amount of quiet excuses their silence.
    Probe("evolve.negative_controls", EXPECT_PERIODIC,
          "system/evolve/negative_controls.json",
          _negative_controls_probe,
          alarm_if_zero=True,
          denominator=DEN_NONE),
    # §4DC: the autonomous GEPA loop. Same class as negative controls —
    # wall-clock jobs whose ledger rows are transition-only, so the
    # LIVENESS signal is the state file's own last_run_epoch, not the
    # ledger count (a quiet week is healthy; a state file that stopped
    # advancing is a dead loop wearing a quiet week's clothes).
    Probe("gepa.autonomy", EXPECT_PERIODIC,
          "system/gepa_autonomy_state.json",
          _gepa_autonomy_probe,
          alarm_if_zero=True,
          denominator=DEN_NONE),
    Probe("metacog.arbitration", EXPECT_GATED,
          "core/agent.py::_METACOG_ARBITER_ENABLED (+ shutdown summary)",
          _arbitration_probe),
    # ⚠ REVIEW ROUND 1: the pattern was `\bverify\b`, which does NOT match
    # "verifier" (verif-Y vs verifi-ER). The real mirror line reads
    # "verifier — LATE CONFIRMED (95%)". The probe reported a FALSE DEAD alarm
    # against 43 genuine verdict lines. A monitor's first output being a false
    # page is how it gets muted on day one.
    # ⚠ `(?i)` IS LOAD-BEARING — review round 1, 2026-08-10. The pattern was
    # case-SENSITIVE and missed **1025 lines**, almost all
    # "Verifier escalation OVERTURNED a cheap-judge refute: … CONFIRMED".
    # Those are proof that verification RAN, so excluding them undercounts the
    # very thing the probe measures. Verified by cross-checking against a
    # deliberately broad net: the router and GEPA patterns skip only genuine
    # non-matches ("Semantic Toolkit Router", an agent *thinking about* a GEPA
    # report), but this one was skipping real evidence on capitalisation alone.
    Probe("verifier.outcomes", EXPECT_ON_OUTPUT,
          "system/ghost-agent.log — verifier verdict + escalation lines",
          _log_probe(r"(?i)verifier\b.*"
                     r"(CONFIRMED|REFUTED|SKIPPED|UNCERTAIN|OVERTURNED|ERROR )",
                     window_h=24.0,
                     # The agent REASONING about the verifier is not the
                     # verifier running. 12/1370 matches were such prose.
                     exclude=r"thinking —"),
          denominator=DEN_REQUESTS),
    Probe("router.decisions", EXPECT_PERIODIC,
          "system/ghost-agent.log — complexity-router lines",
          _log_probe(r"complexity router", window_h=24.0),
          alarm_if_zero=True,
          # Routes on EVERY request (§4AJ: `dispatcher.route()` runs per
          # request), self-play included — so an idle box explains it, a
          # user-quiet one does not.
          denominator=DEN_REQUESTS),
    # ⚠ mtime is the ONLY signal here — checked in review round 2, this file
    # carries no internal timestamp (schema/feature_names/weights/bias/
    # hyperparameters/report/gate_report). So a `touch` WOULD fake a retrain.
    # Stated in the note rather than silently trusted.
    Probe("router.checkpoint", EXPECT_ON_OUTPUT,
          "system/router/checkpoint.json (mtime only — no internal timestamp)",
          _mtime_probe("system/router/checkpoint.json", stale_h=168.0)),
    Probe("calibration.fit", EXPECT_PERIODIC,
          "system/calibration/calibration_params.json :: fitted_at (content)",
          _json_field_ts_probe("system/calibration/calibration_params.json",
                               "fitted_at", stale_h=48.0),
          alarm_if_zero=True),
    # ⚠ BOTH ARE SIMULATION-GATED (§4K: dream/self-play/subagent turns resolve
    # to NOTHING, ledger and index both). Measured 2026-08-11: 24h of hourly
    # self-play produced 28 requests and ZERO rows in either — correct
    # behaviour that reads as a dead loop against any denominator that counts
    # self-play. DEN_USER_TURNS is what makes their zero legible.
    Probe("foresight.predictions", EXPECT_ON_OUTPUT,
          "system/foresight/predictions.jsonl",
          _jsonl_probe("system/foresight/predictions.jsonl", window_h=24.0),
          denominator=DEN_USER_TURNS),
    Probe("rrf.observations", EXPECT_ON_OUTPUT,
          "system/rrf/observations.jsonl",
          _jsonl_probe("system/rrf/observations.jsonl", window_h=24.0),
          denominator=DEN_USER_TURNS),
    # ⚠ REVIEW ROUND 1: this probed `system/experiments.json` and reported
    # NO_SOURCE. That file is an OPTIONAL OVERRIDE — its absence is the NORMAL
    # state (the built-in DEFAULT_SPECS are used), so the probe was reporting a
    # healthy system as an instrumentation gap. Arm stamps actually ride on the
    # TRAJECTORY CORPUS, which is what `introspect action='experiments'` reads.
    Probe("experiments.stamps", EXPECT_ON_DEMAND,
          "system/trajectories/<date>/ — arms are stamped per trajectory",
          _newest_child_probe("system/trajectories", stale_h=72.0),
          denominator=DEN_USER_TURNS),
    # ROUTER SIGNAL COVERAGE (audit 2026-08-10, CORRECTED same day).
    #
    # ⚠ This comment previously asserted "THE SIGNAL NEVER REACHES THE CORPUS …
    # 0 of 1552 records … accuracy is UNMEASURABLE". **That was false and is
    # retracted.** 65 of 1552 records DO carry the decision, nested under
    # `extra` (agent.py merges turn_facts into `_extra` → `Trajectory.extra`).
    # The claim came from a scan that checked top-level keys only — and this
    # probe carried the identical bug, so the instrument agreed with the
    # mistake that produced it.
    #
    # What the probe actually measures: COVERAGE. The decision lands only on
    # turns that produce a trajectory, and trajectory writes are sparse and
    # selective — measured 0–70% per day. So a ZERO here means "not landing
    # right now", never "the plumbing is missing", and any validation built on
    # this subset must bound its bias first.
    Probe("router.signal_durable", EXPECT_ON_OUTPUT,
          "system/trajectories/*/* :: router_label (top level OR under `extra`)",
          _trajectory_router_signal_probe,
          denominator=DEN_USER_TURNS),
    Probe("gepa.applies", EXPECT_ON_DEMAND,
          "system/ghost-agent.log — loader INFO on artifact load",
          _gepa_applies_probe),
]


def probe_all(ghost_home: Optional[Path] = None) -> Dict[str, Any]:
    """Run every probe. Read-only, never raises, ordered worst-news-first."""
    home = _home(ghost_home)
    now = time.time()
    rows = []
    for pr in PROBES:
        try:
            res = pr.fn(home)
        except Exception as e:                               # noqa: BLE001
            res = ProbeResult(NO_SOURCE, note=f"probe raised: {type(e).__name__}")
        rows.append({
            "name": pr.name,
            "expectation": pr.expectation,
            "source": pr.source,
            "status": res.status,
            "count": res.count,
            "age_h": (round((now - res.last_ts) / 3600.0, 1)
                      if res.last_ts else None),
            "note": res.note,
            # ⚠ ALARM ONLY on a genuine zero from a SOURCE THAT EXISTS.
            # NO_SOURCE is a gap to fix, not a dead loop to page about, and
            # GATED is a deliberate choice — conflating either with "dead"
            # is how this view would become furniture.
            "alarm": bool(res.alarm)
                     or bool(pr.alarm_if_zero and res.status == ZERO),
        })
    order = {NO_SOURCE: 0, ZERO: 1, GATED: 2, FIRED: 3}
    rows.sort(key=lambda r: (not r["alarm"], order.get(r["status"], 9),
                             r["name"]))
    turns, requests, turns_note = _count_user_turns(home, 24.0)
    # ⚠ NO-TRAFFIC GUARD. Turn-driven mechanisms are correctly silent when
    # nothing asked them to run; alarming then teaches the operator to ignore
    # this view. Withheld as ONE fact, exactly as activity_liveness withholds
    # per-loop alarms for a stopped agent.
    #
    # ⚠ SCOPE, TIGHTENED 2026-08-11. This cleared EVERY row, and the only two
    # probes that alarm at all (router.decisions, calibration.fit) are
    # PERIODIC — they run off the idle clock, not off user traffic. So the
    # blanket form could only ever silence alarms that a quiet day does not
    # explain: a genuinely dead training loop over a traffic-free weekend
    # would have rendered clean. The bug was invisible because the count
    # included self-play and so never reached 0; fixing the count is what
    # ARMS this branch, which is why both had to move together.
    #
    # UNKNOWN (None) withholds DEN_USER_TURNS alarms too: with no denominator a
    # simulation-gated zero cannot be told from correct silence, and asserting
    # DEAD on it is the false MAJOR this whole guard exists to prevent.
    # DEN_NONE never gets an excuse.
    den = {p.name: p.denominator for p in PROBES}
    for r in rows:
        d = den.get(r["name"], DEN_NONE)
        if d == DEN_USER_TURNS and (turns == 0 or turns is None):
            r["alarm"] = False
        elif d == DEN_REQUESTS and requests == 0:
            r["alarm"] = False
    return {
        "rows": rows,
        "alarms": [r["name"] for r in rows if r["alarm"]],
        "gaps": [r["name"] for r in rows if r["status"] == NO_SOURCE],
        "n_probes": len(rows),
        "user_turns_24h": turns,
        "requests_24h": requests,
        "user_turns_note": turns_note,
    }


def render(ghost_home: Optional[Path] = None) -> str:
    r = probe_all(ghost_home)
    out = ["SUBSYSTEM LIVENESS (probe-per-mechanism — covers what the "
           "activity ledger does NOT):"]
    t = r.get("user_turns_24h")
    note = r.get("user_turns_note") or ""
    if t is None:
        out.append("  ⚠ user-turn count unavailable — turn-driven zeros below "
                   "cannot be interpreted, turn-driven alarms withheld"
                   + (f" [{note}]" if note else ""))
    elif t == 0:
        out.append(f"  ⚠ ZERO real user turns in 24h ({r.get('requests_24h')} "
                   f"requests total, i.e. self-play) — simulation-gated "
                   f"mechanisms are correctly silent and their alarms are "
                   f"withheld (that is ONE fact). Self-play does NOT count "
                   f"here: it is gated out of the very ledgers this "
                   f"denominator interprets")
    else:
        out.append(f"  context: {t} real user turns in 24h of "
                   f"{r.get('requests_24h')} requests (self-play excluded — "
                   f"the denominator for simulation-gated mechanisms)")
    for row in r["rows"]:
        mark = "  ✗ DEAD " if row["alarm"] else "    "
        cnt = "—" if row["count"] is None else str(row["count"])
        age = "never" if row["age_h"] is None else f"{row['age_h']}h ago"
        out.append(f"{mark}{row['name']:<26}{row['status']:<10}"
                   f"n={cnt:<6}last {age:<12}{row['expectation']}")
        if row["note"]:
            out.append(f"        └ {row['note']}")
    if r["gaps"]:
        out.append(f"  ⚠ {len(r['gaps'])} mechanism(s) have NO durable evidence "
                   f"source — cannot be told apart from dead: "
                   + ", ".join(r["gaps"]))
    if r["alarms"]:
        out.append(f"  ✗ {len(r['alarms'])} mechanism(s) SILENT with a live "
                   f"source: " + ", ".join(r["alarms"]))
    elif not r["gaps"]:
        out.append("  all probed mechanisms accounted for")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════
# LOOP YIELD — a SECOND axis, and the one §4CS was found by
# ═══════════════════════════════════════════════════════════════════════
#
# Everything above answers "did this mechanism RUN?". That is not the same
# question as "did it produce anything anyone CONSUMED?", and the gap
# between them is where this project keeps losing work:
#
#   * The macro loop ran for six weeks. Dream fired on schedule, mined
#     sequences, minted 25 composed skills. Every probe above would have
#     read FIRED. Invocations across every auto-mined macro, all time: 0.
#     Nothing anywhere said so (§4CS).
#   * §4CH found dream ledgering a CONSTANT for its busiest subsystem —
#     919 identical rows — which is the same failure wearing a number.
#
# So this axis reports, per PRODUCING loop, the pipeline an artifact must
# survive: minted → activated → invoked, plus when it was last invoked.
#
# THE STATES, and why there are five rather than two:
#
#   YIELDING    invoked > 0 — the loop is feeding something
#   BARREN      the invocation channel IS observable and reads ZERO
#               <- a real finding: the loop produces nothing consumable
#   UNMEASURED  the artifacts exist but NOTHING durably records their use
#               <- an instrumentation gap. NOT a zero, and acting on it as
#                  one would kill a loop that may be working fine.
#   NO_SOURCE   the artifact store itself is absent or unreadable
#   GATED       switched off on purpose
#
# BARREN vs UNMEASURED is the whole point, and it is the same distinction
# `probe_all` draws between ZERO and NO_SOURCE one level up. A view that
# printed "invoked: 0" for both would have said the same thing about the
# macro loop (genuinely barren, and fixable) and the graduated-skill loop
# (injected into the prompt on every matching turn, but with no counter
# anywhere) — opposite problems, opposite remedies.

YIELDING = "yielding"
BARREN = "barren"
UNMEASURED = "unmeasured"
#: ⚠ REVIEW ROUND 1. BARREN used to cover `minted == 0`, so a home where
#: every store existed and was legitimately empty rendered "5 loop(s)
#: produce artifacts NOBODY INVOKES" about five loops that had produced
#: nothing. A fresh install, a post-reset box, or any store before its
#: first write reads that way. "Minted nothing" and "minted things nobody
#: invokes" are different findings with different remedies.
EMPTY = "empty"
#: A SEVENTH state, and the one BARREN could not express (2026-08-24).
#: (Seven: NO_SOURCE, BARREN, UNMEASURED, EMPTY, GATED, RETIRED,
#: YIELDING. An earlier comment said "sixth", having counted the five
#: the §4CS docstring lists and forgotten GATED, which that docstring
#: also omits.)
#:
#: BARREN means "measurable, measured at zero, and the remedy is to find
#: it a consumer". `foresight.gate` read BARREN for weeks while the real
#: answer was that the question is SETTLED: §4CS item G measured the index
#: ANTI-PREDICTIVE over 761 rows — rows it claims will fail fail 7.1%,
#: rows it claims will succeed fail 10.6%. `_evaluate_bucket` rejects on
#: precision BEFORE the interval test, so a bucket under the bar cannot
#: enable at ANY n. More traffic does not fix it.
#:
#: Sorting that at the TOP of a worst-news-first view spends an operator's
#: attention on a decided question every time they read it, and the row
#: says "the remedy is upstream", which reads as owed work. RETIRED says
#: measured dead, decision recorded, nothing owed.
#:
#: ⚠ IT IS DERIVED, NEVER ASSERTED — the same rule as §4CS item C's park.
#: The status is read off the loop's own live verdict, so the moment the
#: index discriminates in the right direction the row un-retires itself
#: with no code change. A hardcoded flag would freeze a measurement into
#: a belief, which is the failure `derived_zero` already exists to avoid.
RETIRED = "retired"


# ── The retrieval-CONCENTRATION axis (2026-08-24) ──────────────────────
#
# `invoked` is a SUM over per-item counters. A sum cannot distinguish a
# store whose 50 items are all being retrieved from a store where one item
# takes every retrieval and 49 never surface — and those are opposite
# findings with opposite remedies (the first is healthy; the second means
# the RETRIEVER is broken, not the store).
#
# This is not hypothetical. arXiv:2604.27003 ("When Continual Learning
# Moves to Memory") measures the failure directly: external memory does
# not remove continual learning's interference problem, it RELOCATES it
# from parameter updates to retrieval, and in their homogeneous-store
# condition **88.5% of queries retrieved the identical top item** despite
# high key-level diversity. Their conclusion — "pool size alone predicts
# nothing about retrieval effectiveness" — is a direct statement that
# `minted` and `invoked`, the two numbers this view already had, cannot
# see the pathology.
#
# NOTHING NEW IS RECORDED. Every store already carries the per-item
# counter that `invoked` sums (`retrievals` on lessons and graduated
# skills, `usage_count` on macros and acquired skills). This is a second
# statistic over the SAME vector, which is why `total` must equal the
# row's `invoked` — pinned, because a spread computed over a different
# population than the row describes would be a plausible lie.

#: Expected count per item under a uniform null, below which no verdict is
#: reported. This is the standard expected-cell-count convention (≥5 per
#: cell) rather than a tuned constant: with fewer draws than that, a top-1
#: share carries no information about concentration — 3 retrievals over 10
#: items puts top-1 at 33% no matter how the retriever behaves.
#:
#: §4CE ("verdict without power") is the reason this gate exists at all:
#: ten of ten arm/metric pairs there reported "no difference detected" for
#: a difference that was arithmetically undetectable. A concentration
#: number printed under this floor would be the same instrument failure.
_SPREAD_MIN_PER_ITEM = 5

#: Both bars are MAJORITY statements, not tuned knobs, and that is
#: deliberate — this project's constants have twice been found calibrated
#: on the wrong statistic (§4BR) or the wrong regime. "One item takes more
#: than all the others combined" and "most of the store never surfaced"
#: are facts about a majority; they do not move when the corpus does.
_SPREAD_TOP1_BAR = 0.5
_SPREAD_COVERAGE_BAR = 0.5

CONCENTRATED = "concentrated"
SPREAD_OK = "distributed"
UNDERPOWERED = "underpowered"
UNDEFINED = "undefined"


@dataclass
class Spread:
    """How a loop's retrievals are distributed across its OWN items.

    `n` is the ELIGIBLE population — what could have been retrieved — not
    the minted one.

    ⚠ THE REASON FIRST GIVEN FOR THIS WAS FALSE, and it was repeated in
    four places before a reviewer executed it. The claim was "scoring
    over all 29 would report 100% top-1 for a store with one runnable
    item". Top-1 SHARE is `top1 / total` and is INVARIANT to adding
    zero-count items — measured on the live macro shape, eligible-only
    and all-29 both give `top1_share = 1.0`. (There are also three
    runnable macros, not one.)

    What the eligible population actually changes is everything ELSE:
    `coverage` (0.333 vs 0.034), the UNDERPOWERED floor (`5*n` = 15 vs
    145, so the whole row would be withheld), and the entropy. The
    filter is right; the illustration was not, and a right thing held
    for a wrong reason is one refactor away from being removed.
    """
    n: int = 0                          # eligible items
    total: int = 0                      # retrievals — MUST equal `invoked`
    top1: int = 0                       # count on the busiest item
    nonzero: int = 0                    # items retrieved at least once
    top1_share: Optional[float] = None
    coverage: Optional[float] = None    # nonzero / n
    entropy_ratio: Optional[float] = None   # H / log(n): 1 uniform, 0 all-one
    verdict: str = UNDEFINED
    why: str = ""


def _eligible_counts(rows, count_of, eligible_now) -> List[int]:
    """The per-item counts to score concentration over — ONE formula, so
    the four probes cannot drift apart on the question that decides what
    the statistic means.

    Eligible = currently eligible, UNION anything with a non-zero count.

    The union half is not generosity, it is an INVARIANT. `invoked` is
    summed over every row the probe owns; if an item was retrieved while
    active and later demoted (a macro un-approved, a skill degraded), a
    "currently eligible" population would drop its retrievals from `total`
    while the row still counts them in `invoked` — and a spread computed
    over a different population than its own row is exactly the kind of
    plausible lie §4CE found three instruments telling. A non-zero count
    is itself proof the item was eligible when it was drawn.

    `total == invoked` is therefore true BY CONSTRUCTION here, and pinned
    as an executed test rather than trusted (`pin identity, not property`).
    """
    out = []
    for r in rows:
        c = int(count_of(r) or 0)
        if c > 0 or eligible_now(r):
            out.append(c)
    return out


def _spread(counts: List[int]) -> Spread:
    """Concentration over one loop's per-item retrieval counts.

    Descriptive, not inferential, and deliberately so. A significance test
    against a UNIFORM null is the wrong instrument here: with 922
    retrievals over 50 lessons every real store rejects uniformity at
    p≈0, because relevance-ranked retrieval is SUPPOSED to be non-uniform.
    The pathology 2604.27003 names is not "non-uniform", it is "one item
    dominates and the tail never surfaces" — so the numbers reported are
    the ones that state exactly that, against a published comparator
    (their 88.5%) rather than against a null nobody believes.
    """
    counts = [int(c or 0) for c in counts]
    n = len(counts)
    total = sum(counts)
    nonzero = sum(1 for c in counts if c > 0)
    top1 = max(counts) if counts else 0
    sp = Spread(n=n, total=total, top1=top1, nonzero=nonzero)

    if n < 2:
        sp.why = (f"only {n} item(s) eligible — with fewer than two there "
                  f"is nothing for retrieval to concentrate ON, so a share "
                  f"here would be 100% by construction")
        return sp
    if total <= 0:
        # NOT underpowered and NOT concentrated: nothing was retrieved at
        # all. The row's own status (BARREN/EMPTY/derived) is the finding;
        # repeating it as a concentration verdict would double-count it.
        sp.why = "nothing retrieved yet — the row's own status is the finding"
        return sp

    sp.top1_share = top1 / total
    sp.coverage = nonzero / n
    ps = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    hmax = math.log(n)
    # Clamped: floating error can push a perfectly uniform ratio a hair
    # over 1.0, and a "1.0000000002 uniform" reads as a bug in the
    # instrument rather than in the store.
    sp.entropy_ratio = max(0.0, min(1.0, h / hmax)) if hmax > 0 else 0.0

    need = _SPREAD_MIN_PER_ITEM * n
    if total < need:
        sp.verdict = UNDERPOWERED
        # ⚠ THE PIGEONHOLE BOUND, not 1/total. With `total` draws over
        # `n` items the busiest holds at least ceil(total/n), so the
        # minimum possible top-1 SHARE is ceil(total/n)/total — equal to
        # 1/total only when n >= total. The first version printed 1/total
        # unconditionally and was wrong on a live row by 5x ("cannot fall
        # below 4%" for 24 retrievals over 5 items, true floor 20.8%), in
        # the direction that makes the observed concentration look more
        # meaningful than it is. At n=100/total=499 it printed "cannot
        # fall below 0%" — a vacuous floor offered as the REASON a verdict
        # is withheld about a 100%-collapsed store.
        floor = math.ceil(total / n) / total
        # ⚠ The formula is exact and the FORMAT was not: at n=201,
        # total=200 the true floor is 0.5% and `{:.0%}` printed "0%" —
        # the vacuous sentence this comment block cites as the defect,
        # reintroduced by rounding.
        # ⚠ STRICT `<` LET THE CITED CASE THROUGH: `f"{0.005:.0%}"` is
        # "0%" (round-half-even), and a brute force over the underpowered
        # region found 816 (n, total) pairs still printing the vacuous
        # floor — including n=201/total=200, literally the example the
        # comment above names. Compare against what will be PRINTED, not
        # against a threshold that approximates it.
        _floor_s = f"{floor:.0%}"
        if floor > 0 and _floor_s == "0%":
            _floor_s = "<1%"
        sp.why = (f"{total} retrieval(s) over {n} eligible item(s) — under "
                  f"the {need} ({_SPREAD_MIN_PER_ITEM}/item) floor, so the "
                  f"share below is REPORTED BUT NOT JUDGED: at this "
                  f"denominator top-1 cannot fall below "
                  f"{_floor_s} however well the retriever behaves")
        return sp

    # ⚠ STRICTLY GREATER. This branch prints "more than every other item
    # combined", and `>=` does not mean that: at n == 2 a uniform [5, 5]
    # store scored exactly 0.5, tripped the bar, and was reported as a
    # broken retriever with a false claim (5 is not more than 5).
    # Exhaustively checked: under `>=`, DISTRIBUTED was UNREACHABLE at
    # n == 2 for every total from 1 to 2000 — a dead verdict on an
    # ordinary store shape (two active skills, two approved macros). The
    # operator now matches the sentence.
    if sp.top1_share > _SPREAD_TOP1_BAR:
        sp.verdict = CONCENTRATED
        sp.why = (f"ONE item takes {sp.top1_share:.0%} of all retrievals — "
                  f"more than every other item combined. The store is not "
                  f"the finding; the RETRIEVER is")
    elif sp.coverage <= _SPREAD_COVERAGE_BAR:
        sp.verdict = CONCENTRATED
        sp.why = (f"{nonzero} of {n} eligible item(s) have EVER been "
                  f"retrieved — most of the store has never surfaced, so "
                  f"minting more of it cannot help")
    else:
        sp.verdict = SPREAD_OK
        sp.why = ("no item takes a majority and most of the store is "
                  "reached — the collapse signature in arXiv:2604.27003 "
                  "is 88.5% top-1")
    return sp


@dataclass
class YieldResult:
    minted: Optional[int] = None
    activated: Optional[int] = None
    invoked: Optional[int] = None
    last_invoked: Optional[float] = None
    note: str = ""
    #: An ASSERTED status, overriding the derivation. Used for GATED and
    #: for RETIRED. ⚠ The comment here said "only to force GATED" for
    #: three rounds after RETIRED started using it — a contract that
    #: describes a subset of its own callers.
    #:
    #: The distinction that matters is unchanged and is pinned: the
    #: retirement DECISION is computed from the gate's data every time
    #: `_yield_foresight_gate` runs. This field carries that decision; it
    #: does not stand in for one.
    status: Optional[str] = None
    #: ⚠ REVIEW ROUND 2. BARREN's contract is "measurable, and MEASURED at
    #: zero", and both live members violated it: the foresight gate's zero
    #: is arithmetic (an allow-list with nothing allowed cannot be
    #: consulted) and the macro loop's is structural (a `proposed` macro is
    #: not advertised and cannot be run). An operator triaging "produce
    #: artifacts NOBODY INVOKES" concludes the artifacts are being ignored;
    #: the truth is they are BLOCKED UPSTREAM, which is a different remedy.
    #: Same BARREN/UNMEASURED distinction, one level finer.
    derived_zero: str = ""
    #: How `invoked` is distributed across the loop's OWN items, when the
    #: loop keeps per-item counters. None means the loop has no per-item
    #: channel to distribute over (gepa serves ONE artifact; the foresight
    #: gate and evolve have no per-item use counter at all) — which is
    #: distinct from a computed spread whose verdict is UNDEFINED.
    spread: Optional[Spread] = None


@dataclass
class YieldProbe:
    name: str
    source: str
    fn: Callable[[Path], YieldResult]
    #: What "activated" and "invoked" MEAN for this loop. Printed, because
    #: they mean different things per loop and an operator reading one
    #: scale across all of them would draw wrong conclusions — the trap
    #: BACKGROUND ACTIVITY already carries a paragraph of warning about.
    activated_means: str
    invoked_means: str


def _yield_status(res: "YieldResult") -> str:
    """Derived, never asserted.

    Order matters: a missing STORE outranks a missing counter, and a
    missing counter outranks a zero — because reading an unmeasured loop
    as barren is the expensive mistake, not the cheap one.
    """
    if res.status:
        return res.status
    if res.minted is None:
        return NO_SOURCE
    # ⚠ REVIEW ROUND 2: this test used to sit BELOW the `invoked is None`
    # one, so EMPTY was unreachable for every probe whose invocation
    # channel is unmeasured. An empty `auto_skills.json` rendered
    # "1 loop(s) have artifacts but NO durable use counter" — asserting
    # artifacts for a store that has none. "No artifacts" outranks "no
    # counter for the artifacts": with nothing minted, the missing counter
    # is not the finding.
    if res.minted == 0:
        return EMPTY
    if res.invoked is None:
        return UNMEASURED
    return YIELDING if res.invoked > 0 else BARREN


def _read_json(home: Path, rel: str):
    """Parsed JSON, or None for absent/unreadable — the two are the same
    answer here (we cannot see the artifacts either way), and both are
    distinct from an EMPTY store, which parses to a real empty container."""
    try:
        p = home / rel
        if not p.is_file():
            return None
        return json.loads(p.read_text() or "null")
    except Exception:                                       # noqa: BLE001
        return None


#: The two auto-mint producers stamp their own `trigger_description`
#: (core/dream.py's miner and the skills_auto graduation mint). Provenance
#: is asked of THAT, not of the name alone: `manage_composed_skills(
#: action="define")` accepts any identifier, `auto_foo` included, so an
#: operator- or model-defined macro would otherwise be booked as loop
#: output — and ITS invocations would make a barren loop read YIELDING.
#: Measured 2026-08-23: prefix and description agree on 26 of 26 live rows.
#: ⚠ REVIEW ROUND 2: these were COPIES of the producers' strings, with
#: nothing linking them. Rewording either producer — a pure refactor —
#: made the loop's own output invisible AND made the row assert a
#: fabricated provenance fact ("2 hand-written macro(s) excluded" about
#: two macros the loop had just minted), with the suite green. That is
#: this project's "token pins vs executed pins" and "lexical proxy for a
#: semantic property" lessons committed inside the fix for them.
#:
#: Both producers now BUILD their `trigger_description` from these, so
#: there is one definition and it cannot drift. `tools/composed_skills.py`
#: imports them; `core/dream.py` and `core/agent.py` interpolate them.
def _macro_marks():
    """The two producer stamps, from the module the PRODUCERS import.

    They live in `tools/composed_skills` because both producers already
    import it to build their description; a copy here is what drifted.
    """
    from ..tools.composed_skills import (
        MACRO_MARK_GRADUATED, MACRO_MARK_MINED,
    )
    return MACRO_MARK_MINED, MACRO_MARK_GRADUATED


def _is_loop_minted_macro(key, row) -> bool:
    """Is this stored macro THIS LOOP's output?

    Asked of the producer's own stamp, not of the name: `action="define"`
    accepts any valid identifier including `auto_*`, so a hand-defined
    macro would otherwise be booked as loop output and ITS invocations
    would make a barren loop read YIELDING.
    """
    if not isinstance(row, dict) or not str(key).startswith("auto_"):
        return False
    desc = str(row.get("trigger_description") or "").lower()
    try:
        mined, graduated = _macro_marks()
    except Exception:                                       # noqa: BLE001
        return False        # cannot tell provenance ⇒ do not claim it
    return mined.lower() in desc or graduated.lower() in desc


def _yield_macros(home: Path) -> YieldResult:
    """Composed-skill macros, AUTO-MINED ONLY.

    The auto-mined subset is the loop's own output. Counting the hand-made
    macros with it is how a barren loop hides: on 2026-08-23 the store read
    26 macros / 3 invocations, and every one of those invocations belonged
    to the single hand-written `youtube_transcribe`. The loop's true figure
    was 25 minted / 0 activated / 0 invoked.
    """
    data = _read_json(home, "system/memory/composed_skills/composed_skills.json")
    if not isinstance(data, dict):
        return YieldResult(note="composed_skills.json absent or unreadable")
    auto = {k: v for k, v in data.items() if _is_loop_minted_macro(k, v)}
    malformed = sum(1 for v in data.values() if not isinstance(v, dict))
    hand = sum(1 for k, v in data.items()
               if isinstance(v, dict) and not _is_loop_minted_macro(k, v))
    invoked = sum(int(v.get("usage_count") or 0) for v in auto.values())
    succeeded = sum(int(v.get("success_count") or 0) for v in auto.values())
    # `_parse_ts`, like every peer probe. A raw `float()` raised
    # ValueError on an ISO stamp — the shape `auto_skills.json` writes for
    # the same concept — and the whole row rendered `no_source`
    # ("probe raised: ValueError"): a populated store reported as missing.
    last = max((_parse_ts(v.get("last_used")) or 0.0 for v in auto.values()),
               default=0.0)
    parked = sum(1 for v in auto.values() if v.get("status") != "active")
    note = (f"{hand} hand-written macro(s) excluded — they are not this "
            f"loop's output")
    if parked:
        # ⚠ REVIEW ROUND 1: with every macro at "proposed" this zero is
        # DERIVED, not observed. A proposed macro is not advertised, not
        # dispatchable, and `action='run'` refuses it — `usage_count`
        # CANNOT be non-zero. Saying so keeps the row honest: the finding
        # is the unapproved backlog, not an ignored tool.
        note += (f"; {parked} still 'proposed' — a proposed macro is not "
                 f"advertised and cannot be run, so their zero is DERIVED, "
                 f"not observed. Approve with "
                 f"manage_composed_skills(action='approve')")
    if invoked:
        note += f"; {succeeded}/{invoked} invocations succeeded"
    if malformed:
        note += f"; ⚠ {malformed} unreadable row(s) skipped"
    active = sum(1 for v in auto.values() if v.get("status") == "active")
    return YieldResult(
        minted=len(auto),
        activated=active,
        invoked=invoked,
        last_invoked=last or None,
        note=note,
        # ELIGIBLE = active (∪ anything already invoked). A `proposed`
        # macro is not advertised and `action='run'` refuses it, so it
        # cannot appear in the denominator of "where did the retrievals
        # go" — scoring over all 29 would report 100% top-1 about a store
        # with one runnable macro, which is the derived-zero error
        # transplanted into the new axis.
        spread=_spread(_eligible_counts(
            auto.values(),
            lambda v: v.get("usage_count"),
            lambda v: v.get("status") == "active")),
        # ⚠ REVIEW ROUND 2: with NOTHING approved this zero is structural,
        # not measured — a proposed macro is not advertised, not
        # dispatchable, and `action='run'` refuses it, so `usage_count`
        # CANNOT be non-zero. "Nobody invokes them" sends an operator
        # looking for a consumer; the remedy is the approval queue.
        derived_zero=("no macro is approved, so none can be invoked"
                      if auto and not active else ""),
    )


def _yield_acquired_skills(home: Path) -> YieldResult:
    """`create_skill` — python tools the agent wrote for itself."""
    data = _read_json(home, "system/memory/acquired_skills/skills_registry.json")
    if not isinstance(data, dict):
        return YieldResult(note="skills_registry.json absent or unreadable")
    rows = [v for v in data.values() if isinstance(v, dict)]
    never = sum(1 for v in rows if not int(v.get("usage_count") or 0))
    degraded = sum(1 for v in rows if v.get("status") != "active")
    return YieldResult(
        minted=len(rows),
        activated=len(rows) - degraded,
        invoked=sum(int(v.get("usage_count") or 0) for v in rows),
        # The registry records a usage COUNT but no timestamp, so "when"
        # is genuinely unavailable rather than never.
        last_invoked=None,
        # ⚠ REVIEW ROUND 2 corrected BOTH halves of the previous note.
        # `status == "active"` really IS an activation gate (it gates
        # advertising, dispatch AND embedding), and `activated == minted`
        # is not "by construction" — a skill sits `degraded` until
        # `retire_degraded_skills` runs. And the row aggregated usage
        # across skills, which is exactly the masking this whole surface
        # exists to expose: 24 invocations, all on ONE of five skills.
        note=(f"{never} of {len(rows)} skill(s) have NEVER been used"
              + (f"; {degraded} degraded (not advertised, not dispatchable)"
                 if degraded else "")
              + ". The registry carries usage_count but no last-used "
                "timestamp, so the age column cannot be filled from it"),
        # ELIGIBLE = active (∪ already used). `degraded` gates advertising,
        # dispatch AND embedding, so a degraded skill cannot be drawn.
        spread=_spread(_eligible_counts(
            rows,
            lambda v: v.get("usage_count"),
            lambda v: v.get("status") == "active")),
    )


def _yield_graduated_skills(home: Path) -> YieldResult:
    """skills_auto graduation — the store behind the `auto_generic_*` macros.

    ⚠ THIS ROW WAS `unmeasured` UNTIL §4CT, AND THAT WAS THE FINDING.
    Graduated skills ARE consumed — `GraduatedSkillStore` injects matching
    ones as a "PROVEN APPROACHES" block on every turn with user content —
    but neither `relevant()` nor `format_for_prompt()` recorded that it
    fired, so there was no durable count of retrievals anywhere. That is an
    instrumentation gap, not a dead loop, and the remedy for the two is
    opposite: wire a counter, do NOT kill the loop. `record_surfaced` is
    that counter.

    ⚠ `verifications` is still NOT usage. It is the re-verification count
    from the graduation gate — a PRODUCER-side number — and reading it as
    consumption would turn a silent loop into a healthy-looking one.

    ⚠ AND `retrievals` IS NOT HELPFULNESS. It counts prompt injections. The
    lessons store carries a second pair of arms fed from the turn's verified
    outcome (`succeeded_retrievals` / `failed_retrievals`); this store does
    not, so nothing here says a surfaced skill was read, used, or useful.
    The note says so, because `invoked` on this view is otherwise read as
    value delivered.
    """
    data = _read_json(home, "system/memory/auto_skills.json")
    if not isinstance(data, dict):
        return YieldResult(note="auto_skills.json absent or unreadable")
    rows = [v for v in data.values() if isinstance(v, dict)]
    if not rows:
        return YieldResult(minted=0, activated=0, invoked=0,
                           note="no skill has graduated yet")
    retr = sum(int(v.get("retrievals") or 0) for v in rows)
    never = sum(1 for v in rows if not int(v.get("retrievals") or 0))
    last = max((_parse_ts(v.get("last_retrieved_at")) or 0.0 for v in rows),
               default=0.0)
    return YieldResult(
        minted=len(rows),
        # Retrieved at least once — the same meaning as the lesson row, and
        # a real statistic. It used to be `len(rows)` ("all of them are
        # eligible"), which is true of every store and evidence of nothing.
        activated=len(rows) - never,
        invoked=retr,
        last_invoked=last or None,
        note=(f"{never} of {len(rows)} graduated skill(s) have NEVER been "
              f"surfaced. ⚠ `invoked` counts PROMPT INJECTIONS, NOT "
              f"helpfulness — unlike lessons there is no outcome arm here, "
              f"so nothing says a surfaced skill was read or used. "
              f"Simulated turns are excluded (`turn_origin == user`), so "
              f"this shares the liveness view's denominator."),
        derived_zero=("nothing has graduated, so nothing can be surfaced"
                      if not rows else ""),
        # ELIGIBLE = every graduated skill. There is no activation gate on
        # this store — `relevant()` considers all of them on every turn
        # with user content — so the minted and eligible populations are
        # genuinely the same here, unlike the other three.
        spread=_spread(_eligible_counts(
            rows, lambda v: v.get("retrievals"), lambda v: True)),
    )


def _yield_lessons(home: Path) -> YieldResult:
    """The lesson playbook — the one loop with a full outcome channel."""
    data = _read_json(home, "system/memory/skills_playbook.json")
    if not isinstance(data, list):
        return YieldResult(note="skills_playbook.json absent or unreadable")
    all_rows = [v for v in data if isinstance(v, dict)]
    # ⚠ REVIEW ROUND 1: quarantine is RETENTION WITHOUT SERVICE. The row
    # stays in the playbook so an operator can see it, and
    # `_filter_quarantined` is applied at BOTH retrieval surfaces, so a
    # quarantined lesson can never be retrieved again. Counting them made
    # 278 of 926 retrievals (30%) dead history reported as live yield.
    served = [v for v in all_rows if not v.get("quarantined")]
    quarantined = len(all_rows) - len(served)
    rows = served
    malformed = len(data) - len(all_rows)
    retr = sum(int(v.get("retrievals") or 0) for v in rows)
    helpful = sum(int(v.get("helpful_retrievals") or 0) for v in rows)
    last = max((_parse_ts(v.get("last_retrieved_at")) or 0.0 for v in rows),
               default=0.0)
    never = sum(1 for v in rows if not int(v.get("retrievals") or 0))
    note = (f"{helpful}/{retr} retrievals marked helpful; {never} lesson(s) "
            f"never retrieved")
    if quarantined:
        note += (f"; {quarantined} quarantined row(s) excluded — retained "
                 f"for an operator to see, filtered out of every retrieval")
    if malformed:
        note += f"; ⚠ {malformed} unreadable row(s) skipped"
    return YieldResult(
        # ⚠ REVIEW ROUND 2: `minted` was the SERVED count, so a store whose
        # rows were ALL quarantined rendered "minted NOTHING yet" — the
        # most alarming state this store can be in shown as its most
        # benign, in a project with a history of mass lesson destruction.
        # It also made `minted` mean "produced AND still served" for this
        # one probe and "produced" everywhere else. Minted is produced.
        minted=len(all_rows),
        activated=len(rows) - never,
        invoked=retr,
        last_invoked=last or None,
        note=note,
        derived_zero=("every lesson is quarantined, so none can be "
                      "retrieved" if all_rows and not rows else ""),
        # ELIGIBLE = SERVED, i.e. not quarantined — `_filter_quarantined`
        # runs at both retrieval surfaces, so a quarantined lesson can
        # never be drawn again. Scoring over `all_rows` would credit the
        # store with a tail that is structurally unreachable, and read as
        # poor coverage when the truth is deliberate withdrawal. This
        # mirrors `invoked`, which is already summed over `rows`.
        spread=_spread(_eligible_counts(
            rows, lambda v: v.get("retrievals"), lambda v: True)),
    )


#: ⚠ REVIEW ROUND 2 rewrote this signal three times over.
#:
#:  * The pattern matched ONE of the loader's two "I served this artifact"
#:    lines. `optim/loader.py` logs "loaded tuned instruction" only when
#:    the artifact carries a `gate_arm`, and logs a WARNING — "predates the
#:    gate schema" — for one that does not, THEN serves it anyway. Twenty
#:    such lines are on the live log. A fully-applied artifact read BARREN.
#:  * `_log_probe` counts inside a window but takes `last_ts` over ALL
#:    matches, so an artifact last served 300h ago rendered
#:    "invoked 0 | last 300.0h ago" — self-contradictory, in the loudest
#:    state on the view. Counting all-time makes the two agree.
#:  * The line fires on a CACHE MISS, so it counts artifact LOADS (roughly,
#:    process boots per signature), not applications. Saying "applies"
#:    would be the same category error as round 1's `invoked = packets`.
#:    The column says LOADS, and the note says what that means.
_GEPA_LOAD_WINDOW_H = 24.0 * 365 * 20          # effectively all-time
# ⚠ EVERY LOAD LINE, AND THERE ARE NOW THREE. §4DA round 13 added a
# third — an artifact promoted with `--no-ab-gate` warns "was promoted
# UNGATED" instead of "predates the gate schema" — and neither probe
# matched it, so the `silent-inoperative-subsystems` instrument went
# blind precisely for the artifacts adopted with NO A/B, the ones it most
# needs to watch. The comment below records the same defect one shape
# earlier ("a fully-applied artifact read BARREN"); the pattern is
# anchored on the shared prefix now so a fourth phrasing cannot repeat
# it.
_GEPA_LOAD_PATTERN = (r"GEPA: (loaded tuned instruction|artifact '[^']+' "
                      r"\(sha [0-9a-f]+\) (?:predates the gate schema"
                      r"|was promoted UNGATED))")


def _yield_gepa(home: Path) -> YieldResult:
    """Tuned prompt artifacts, and how often one was actually loaded.

    ⚠ REVIEW ROUND 1. This reported UNMEASURED, on the reasoning that
    apply counters are per-process and reset at boot. That is true of the
    IN-PROCESS counter and false of the system: the loader logs a line on
    every artifact load, `_log_probe` reads it, and THIS FILE already has
    a `gepa.applies` probe doing exactly that — which printed a live count
    in the same report, 390 lines above, while this row said "never".
    Claiming UNMEASURED prescribes "wire a counter"; the counter existed.

    The log line fires on a cache MISS, so the count is a lower bound —
    durable, timestamped, and strictly better than an unknown.
    """
    d = home / "system" / "optim"
    if not d.is_dir():
        return YieldResult(note="system/optim absent")
    valid = 0
    names = []
    for f in sorted(d.glob("*.json")):
        names.append(f.name)
        data = _read_json(home, str(f.relative_to(home)))
        opt = (data or {}).get("optimized_instruction")
        if isinstance(opt, str) and opt.strip():
            valid += 1
    # ⚠ A RETIRED ARTIFACT IS NOT "MINTED NOTHING". Retiring
    # `planning.decompose` (2026-08-24, measured worse than its own seed)
    # left zero live `.json` files, and the row rendered
    # `empty · minted 0 · invoked 135` — "this loop has minted NOTHING
    # yet" printed beside 135 recorded loads of the thing it minted. The
    # suffix filter is right (a retired artifact must not count as live
    # output); the STATE was wrong, because EMPTY's contract is "produced
    # nothing" and this loop produced something that was then withdrawn.
    #
    # Retired/rejected files are counted separately and reported, so the
    # row can say "the gate worked" rather than "nothing happened" —
    # opposite findings with opposite remedies.
    withdrawn = sorted(
        f.name for f in d.iterdir()
        if f.is_file() and ".json." in f.name
        and any(k in f.name for k in ("retired", "rejected", "noop")))
    loads = _log_probe(_GEPA_LOAD_PATTERN,
                       window_h=_GEPA_LOAD_WINDOW_H)(home)
    if not names and withdrawn:
        # ⚠ RETURNED BEFORE THE NO_SOURCE CHECK BELOW, so an unreadable
        # log became `invoked 0` with the note "the 0 loads are
        # historical" — a FABRICATED zero, which is the missing-vs-empty
        # conflation this module's own docstring calls "the finding".
        # A live artifact with no log correctly claims no count; a
        # retired one must do the same.
        if loads.status == NO_SOURCE:
            return YieldResult(
                minted=0, activated=0, invoked=None, status=RETIRED,
                note=(f"no LIVE artifact, {len(withdrawn)} withdrawn one(s) "
                      f"on disk — the loop produced artifacts and every one "
                      f"has been retired or rejected. "
                      f"{loads.note or 'the agent log is unreadable'}, so no "
                      f"load count is claimed"))
        return YieldResult(
            minted=0, activated=0, invoked=loads.count or 0,
            last_invoked=loads.last_ts, status=RETIRED,
            note=(f"no LIVE artifact, but {len(withdrawn)} withdrawn one(s) "
                  f"on disk ({', '.join(withdrawn[:3])}"
                  f"{'…' if len(withdrawn) > 3 else ''}) — this loop has "
                  f"produced artifacts and every one has since been retired "
                  f"or rejected. That is the GATE WORKING, not a loop that "
                  f"minted nothing; the read sites fall back to the "
                  f"hand-written instruction. The {loads.count or 0} loads "
                  f"are historical, of artifacts no longer served"))
    if loads.status == NO_SOURCE:
        return YieldResult(
            minted=len(names), activated=valid, invoked=None,
            note=f"{loads.note or 'the agent log is unreadable'} — the load "
                 f"signal lives in it, so no count is claimed")
    return YieldResult(
        minted=len(names),
        activated=valid,
        invoked=loads.count or 0,
        last_invoked=loads.last_ts,
        note="counts artifact LOADS from the loader's own log lines (both "
             "the gated and the predates-the-gate-schema shape — the "
             "loader serves both). ⚠ NOT applies: the line fires on a "
             "CACHE MISS, so this is roughly loads-per-process and a LOWER "
             "BOUND on how often the artifact was used; a long-running "
             "daemon can serve it thousands of times and log once. All-time "
             "so the count and the age agree. Retired/rejected/candidate "
             "files are excluded by the .json suffix filter",
        derived_zero=("the agent log has been rotated or truncated"
                      if valid and not (loads.count or 0) else ""),
    )


def _wilson_upper(k: int, n: int, z: float = 1.96) -> Optional[float]:
    """Upper bound of the Wilson interval for k/n.

    Wilson rather than Wald because the live numbers sit at the end of
    the scale (1 of 14), where a Wald interval runs off the end and
    manufactures confidence.
    """
    if n <= 0 or k < 0 or k > n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return min(1.0, centre + half)


def _anti_predictive(disc: Any, *, bar: float = 0.60,
                     min_fail_n: int = 10) -> bool:
    """Can this index's predicted-fail precision NEVER clear the bar?

    ⚠ REWRITTEN AFTER ROUND 1. The first version returned `spread <= 0`
    with no power requirement at all — both denominators merely had to be
    truthy. Two reviewers demonstrated the consequences independently and
    both are disqualifying:

      * `{spread: 0.0, fail_n: 1, ok_n: 1}` RETIRED a loop. On the LIVE
        gate the retirement rested on 14 predicted-fail rows of which ONE
        failed: Fisher exact two-sided p = 1.00, and the Newcombe interval
        for the spread is [-0.100, +0.209] — it contains zero AND contains
        the +0.10 bar the verdict cites as the thing it fails. A terminal
        "nothing owed" verdict on an arithmetically undetectable
        difference is §4CE VERBATIM, in the instrument added to prevent it.
      * The sign of a POOLED spread is a proxy, and it fails both ways. A
        Simpson reversal (every bucket +0.10, pooled -0.47) retired a loop
        whose every bucket discriminates correctly; and precision 0.07 at
        fail_n 500 with a hair-positive spread was NOT retired, i.e.
        reported as owed work forever.

    So the test is now the THING, with power. `_evaluate_bucket` rejects
    on `precision < min_fail_precision` BEFORE the interval test, so a
    bucket whose TRUE precision is under the bar cannot enable at ANY n —
    that is the property that makes a retirement honest, and it is
    established only when the precision interval's UPPER bound is below
    the bar. On the live row: 1/14, Wilson upper 0.315 < 0.60, so the
    retirement stands — on sound grounds rather than on a sign.

    AND the pooled figure alone is not enough. A bucket can still be
    alive under a dead pool, so any bucket whose own precision interval
    reaches the bar keeps the loop out of RETIRED.
    """
    if not isinstance(disc, dict):
        return False
    fail_n = disc.get("fail_n")
    fail_hits = disc.get("fail_hits")
    ok_n = disc.get("ok_n")
    ok_hits = disc.get("ok_hits")
    for v in (fail_n, fail_hits, ok_n, ok_hits):
        if isinstance(v, bool) or not isinstance(v, int):
            return False
    if fail_hits < 0 or fail_hits > fail_n or ok_hits < 0 or ok_hits > ok_n:
        return False

    # ⚠ POWER ON *THIS* LEG TOO. Round 2 measured the asymmetry round 1
    # left: retirement needed 3 predicted-fail rows (0/3 has a Wilson
    # upper of 0.562, under the 0.60 bar) while a bucket needed 10 before
    # it was allowed to OBJECT. Every tie broke toward "settled". Both
    # legs now use the gate's own `min_fail_n`.
    if fail_n < min_fail_n or ok_n <= 0:
        return False

    # ⚠ AND THE SIGN IS BACK, because dropping it was an over-correction.
    # Round 1 tested `spread <= 0` with no power. Round 2 replaced it with
    # a precision test that HAD power and lost the semantics — measured on
    # a synthetic index where predicted-fail rows fail 40% and
    # predicted-ok rows fail 2% (spread +0.38, hugely predictive), the
    # rule RETIRED it, and the row printed the builder's verdict
    # ("discriminates in the right direction… but pooled precision is
    # under the bar") directly above "SETTLED, nothing is owed".
    #
    # Two different findings with two different remedies:
    #   spread <= 0  ..... the INDEX is backwards. Nothing to do but
    #                      build a better one. RETIRED.
    #   spread  > 0  ..... the index works and the BAR is wrong for it.
    #                      That is owed work, so NOT retired.
    #
    # Retirement therefore needs BOTH: the index fails to discriminate,
    # AND its precision is provably unable to reach the bar.
    precision = fail_hits / fail_n
    ok_fail_rate = ok_hits / ok_n
    if precision > ok_fail_rate:
        return False

    # `hi is not None` is reachable ONLY through a malformed count that
    # the bounds check above already rejects — kept because
    # `_wilson_upper` is a public-ish helper whose contract allows None,
    # and a caller that stops honouring it should fail closed rather than
    # TypeError. Pinned by the malformed-input cases.
    hi = _wilson_upper(fail_hits, fail_n)
    return hi is not None and hi < bar


def _gate_bucket_n(data: Any) -> int:
    """`min_bucket_n` — the gate's OTHER power floor.

    `_evaluate_bucket` refuses a bucket with fewer than this many
    RESOLVED rows ("thin bucket: 12 resolved rows < 30") regardless of
    its predicted-fail count, so a bucket under it can never enable and
    must not be allowed to veto retirement. Round 3: a bucket at
    `n=12, fail_n=11` blocked RETIRED forever.
    """
    params = (data or {}).get("params") if isinstance(data, dict) else None
    if isinstance(params, dict):
        m = params.get("min_bucket_n")
        if isinstance(m, int) and not isinstance(m, bool) and m > 0:
            return m
    return 30


def _gate_bars(data: Any) -> tuple:
    """`(min_fail_precision, min_fail_n)` READ FROM THE GATE FILE.

    The builder records its thresholds into the artifact precisely so a
    stale gate says which thresholds produced it. Reading them here —
    rather than copying the defaults — is what stops this view and the
    gate from drifting into disagreeing about the same numbers.
    """
    params = (data or {}).get("params") if isinstance(data, dict) else None
    bar, min_fail_n = 0.60, 10
    if isinstance(params, dict):
        b, m = params.get("min_fail_precision"), params.get("min_fail_n")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and 0 < b <= 1:
            bar = float(b)
        if isinstance(m, int) and not isinstance(m, bool) and m > 0:
            min_fail_n = m
    return bar, min_fail_n


def _assessable_buckets(buckets: Any, min_fail_n: int,
                        min_bucket_n: int = 0) -> int:
    """How many buckets have enough predicted-fail rows to be judged.

    Reported so the retirement note can distinguish "we checked the
    buckets and none qualifies" from "none was eligible to be checked".
    """
    if not isinstance(buckets, dict):
        return 0
    out = 0
    for b in buckets.values():
        if not isinstance(b, dict):
            continue
        fn = b.get("fail_n")
        if isinstance(fn, bool) or not isinstance(fn, int) or fn < min_fail_n:
            continue
        # ⚠ AND `fail_hits` MUST BE READABLE. Round 3: a bucket with
        # `fail_hits=None` was counted assessable and then silently
        # skipped by the qualify check, so the note printed "of the
        # buckets with 10+ rows, none can still reach the bar" about a
        # bucket it never examined — the same unrun-check-presented-as-
        # passed shape that sentence was added to fix.
        fh = b.get("fail_hits")
        if isinstance(fh, bool) or not isinstance(fh, int):
            continue
        rn = b.get("n")
        if (min_bucket_n and not isinstance(rn, bool) and isinstance(rn, int)
                and rn < min_bucket_n):
            continue
        out += 1
    return out


def _live_bucket_can_still_qualify(buckets: Any, *, bar: float = 0.60,
                                   min_fail_n: int = 10,
                                   min_bucket_n: int = 0) -> bool:
    """Does any bucket have ENOUGH DATA to still reach the bar?

    The pooled verdict is an aggregate, and an aggregate can be dead
    while a stratum is alive — this project has paid for pooling across
    strata once already (the negative Platt slope on the calibration
    corpus). So a genuinely live bucket keeps the loop out of RETIRED.

    ⚠ BUT "ALIVE" NEEDS ITS OWN POWER, and this is where round 1's fix
    would have gone wrong a second time. Measured on the live gate: 4 of
    64 buckets have a precision interval reaching the bar — and every one
    of them sits at `fail_n` of 1 to 3, where the Wilson upper bound is
    wide BY CONSTRUCTION (`0/1` reads 0.793). Counting those as alive is
    absence of evidence read as evidence, and it would make RETIRED
    unreachable forever, since some 1-row bucket always exists.

    The threshold is the GATE'S OWN `min_fail_n` — the denominator it
    already refuses to evaluate a bucket below — not a number invented
    here. Under it a bucket is UNMEASURED, and only the pooled figure can
    speak.

    This does not foreclose anything: RETIRED is derived and
    self-retracting, so the moment a bucket accumulates enough
    predicted-fail rows to reach the bar, the row un-retires itself with
    no code change.
    """
    if not isinstance(buckets, dict):
        return False
    for b in buckets.values():
        if not isinstance(b, dict):
            continue
        fn, fh = b.get("fail_n"), b.get("fail_hits")
        if isinstance(fn, bool) or not isinstance(fn, int) or fn < min_fail_n:
            continue
        if isinstance(fh, bool) or not isinstance(fh, int):
            continue
        # `min_bucket_n` too: `_evaluate_bucket` refuses a thin bucket
        # outright, so one that can never enable must not block RETIRED.
        rn = b.get("n")
        if (min_bucket_n and not isinstance(rn, bool) and isinstance(rn, int)
                and rn < min_bucket_n):
            continue
        hi = _wilson_upper(max(0, min(fh, fn)), fn)
        if hi is not None and hi >= bar:
            return True
        # ⚠ AND A BUCKET THAT DISCRIMINATES IS ALIVE EVEN UNDER THE BAR,
        # which round 3 showed this function could not see: it read only
        # `fail_n`/`fail_hits`, never the bucket's own `ok_*`, so it
        # could not detect bucket-level discrimination at all. A
        # Simpson-reversed gate — every bucket +spread, pooled negative —
        # was RETIRED as "nothing owed" while every stratum discriminated
        # correctly. That is the exact scenario `_anti_predictive`'s own
        # docstring calls disqualifying, applied to the pooled figure and
        # not to the strata.
        #
        # Same rule as the pooled one, one level down: spread > 0 means
        # the BAR is wrong for this bucket, which is owed work.
        okn, okh = b.get("ok_n"), b.get("ok_hits")
        if (not isinstance(okn, bool) and isinstance(okn, int) and okn > 0
                and not isinstance(okh, bool) and isinstance(okh, int)
                and 0 <= okh <= okn):
            if (fh / fn) > (okh / okn):
                return True
    return False


def _yield_foresight_gate(home: Path) -> YieldResult:
    """Imagine's calibration gate: buckets measured → buckets enabled →
    steering sites that used one."""
    data = _read_json(home, "system/foresight/gate.json")
    if not isinstance(data, dict):
        return YieldResult(note="gate.json absent or unreadable")
    buckets = data.get("buckets")
    n = len(buckets) if isinstance(buckets, dict) else 0
    # ⚠ REVIEW ROUND 2: this read `enabled_count`, a writer-derived
    # aggregate, while `gate_allows` reads `buckets[*].enabled`. The two
    # can disagree in BOTH directions — buckets flagged enabled with no
    # count (the row then claims "NO bucket is enabled" about a file that
    # says otherwise), or a count with malformed buckets. Read the field
    # the consumer reads.
    enabled = sum(1 for v in (buckets or {}).values()
                  if isinstance(v, dict) and v.get("enabled") is True)
    if enabled == 0:
        # Derived, not assumed: `gate_allows` is an allow-list, so with no
        # bucket enabled no steering site can fire. The zero is arithmetic.
        note = (f"{int(data.get('ledger_rows') or 0)} ledger rows; NO bucket "
                f"is enabled, and `gate_allows` is an allow-list — so zero "
                f"steering sites is derived, not observed")
        # §4CS item G: 63 buckets each saying "needs 17 more" reads as a
        # gate waiting for data, and an operator cannot tell that from a
        # gate that will never open. The POOLED verdict can.
        disc = data.get("discrimination") or {}
        verdict = str(disc.get("verdict") or "")
        if verdict:
            note += f". POOLED: {verdict}"
        # RETIRED — measured dead, decision recorded, nothing owed
        # (2026-08-24). DERIVED from the gate's own pooled COUNTS, never
        # from a flag.
        #
        # ⚠ It does NOT read `disc["spread"]`, despite what this comment
        # said for three rounds — it recomputes precision and
        # ok_fail_rate from `fail_hits/fail_n` and `ok_hits/ok_n` and
        # applies a Wilson interval. Verified: a gate whose `spread` key
        # says +0.99, or is absent entirely, retires identically. An
        # operator debugging a retirement was being pointed at a field
        # with no causal role.
        #
        # Why the distinction is worth a state: BARREN prescribes "find it
        # a consumer" and sorts FIRST in a worst-news-first view. That is
        # the correct handling of a loop that might yet work, and the
        # wrong handling of one whose index has the WRONG SIGN — where
        # `_evaluate_bucket` checks precision before the interval test, so
        # no bucket can enable at any n and more traffic cannot help.
        _bar, _min_fn = _gate_bars(data)
        _min_bn = _gate_bucket_n(data)
        if (_anti_predictive(disc, bar=_bar, min_fail_n=_min_fn)
                and not _live_bucket_can_still_qualify(
                    buckets, bar=_bar, min_fail_n=_min_fn,
                    min_bucket_n=_min_bn)):
            return YieldResult(
                minted=n, activated=0, invoked=0, status=RETIRED,
                note=note.rstrip(". ") + (
                     f". RETIRED 2026-08-24 (§4CU): the predicted-fail "
                     f"precision is {disc.get('fail_hits')}/"
                     f"{disc.get('fail_n')}, whose 95% interval tops out at "
                     f"{(_wilson_upper(int(disc.get('fail_hits') or 0), int(disc.get('fail_n') or 1)) or 0):.2f} "
                     f"— BELOW the {_bar:.2f} bar, and `_evaluate_bucket` "
                     f"rejects on precision BEFORE the interval test, so no "
                     f"bucket can enable at any n. "
                     # ⚠ SAY WHICH IT IS. Round 2: the live gate has ZERO
                     # buckets with >= min_fail_n rows, so "no bucket can
                     # still reach the bar" read to an operator as "we
                     # checked and none qualifies" when in fact NONE WAS
                     # ELIGIBLE TO BE CHECKED — an unrun check presented
                     # as a passed one.
                     + (f"No bucket has {_min_fn}+ predicted-fail rows yet, "
                        f"so no bucket could be assessed individually. "
                        if not _assessable_buckets(buckets, _min_fn,
                                                   _min_bn) else
                        f"Of the buckets with {_min_fn}+ predicted-fail "
                        f"rows, none can still reach the bar. ")
                     + f"SETTLED, not waiting: nothing is owed here. "
                     f"Derived, so it RETRACTS ITSELF the moment a bucket "
                     f"accumulates enough rows to qualify; the builder keeps "
                     f"running, because a closed gate is a measurement"),
                derived_zero="no bucket is enabled, so no steering site exists")
        return YieldResult(
            minted=n, activated=0, invoked=0, note=note,
            derived_zero="no bucket is enabled, so no steering site exists")
    return YieldResult(
        minted=n, activated=enabled, invoked=None,
        note="buckets are enabled but no consumer records a steer")


def _yield_evolve(home: Path) -> YieldResult:
    """E1 mutations → E2/E4 operator packets → an operator acting on one.

    ⚠ REVIEW ROUND 1 found two defects here, and the first was this
    module's own failure mode rebuilt inside it:

      * `invoked` was `packets`, the SAME number as `activated`, from the
        same glob. That is PRODUCTION, not consumption — the row went
        green the moment a packet nobody had opened appeared on disk,
        which is exactly "dream minted 25 and every probe read FIRED".
        Nothing records that an operator read or acted on a packet, so the
        honest answer is UNMEASURED.
      * `minted` counted LEDGER ROWS. Every live row is
        `{"outcome": "disabled", "reason": "GHOST_EVOLVE is off"}` — four
        nightly runs that found the loop switched off. Real mutations
        minted: zero. The mutator's own reader already separates these,
        and the same report said "4 of 4 recorded run(s) found it off"
        twenty lines below "minted 4".
    """
    led = home / "system" / "evolve" / "mutations.jsonl"
    if not led.is_file():
        return YieldResult(note="system/evolve/mutations.jsonl absent")
    try:
        rows = [json.loads(ln) for ln in led.read_text().splitlines()
                if ln.strip()]
    except Exception:                                       # noqa: BLE001
        return YieldResult(note="mutations.jsonl unreadable")
    rows = [r for r in rows if isinstance(r, dict)]
    proposed = sum(1 for r in rows if str(r.get("outcome")) == "proposed")
    disabled = sum(1 for r in rows if str(r.get("outcome")) == "disabled")
    pdir = home / "system" / "evolve" / "proposals"
    packets = sorted(pdir.glob("*.json")) if pdir.is_dir() else []
    # ⚠ REVIEW ROUND 2: this asked "are ALL rows disabled?", over the
    # ALL-TIME ledger, which never rotates. Two failures, both confirmed:
    # an operator enables the loop and every run then TIMES OUT (the phase
    # writes no row on that path), so the frozen ledger keeps reporting
    # "a deliberate state, not a dead loop" forever; and a ledger with old
    # `proposed` rows plus recent `disabled` ones — the loop gated off
    # RIGHT NOW — reported UNMEASURED and never mentioned `disabled` at
    # all. `disabled` is only ever the pre-gate default outcome, overwritten
    # on every other exit path, so the LAST row is what describes the
    # loop's current state.
    if rows and str(rows[-1].get("outcome")) == "disabled":
        return YieldResult(
            minted=proposed, activated=len(packets), invoked=None,
            status=GATED,
            note=f"the most recent of {len(rows)} recorded run(s) found the "
                 f"loop switched off (GHOST_EVOLVE); {disabled} of them did. "
                 f"A deliberate state, not a dead loop — but note the ledger "
                 f"never rotates and a run that TIMES OUT writes no row, so "
                 f"a frozen ledger looks like this too")
    if rows and not proposed:
        return YieldResult(
            minted=0, activated=len(packets), invoked=None,
            note=f"{len(rows)} recorded run(s) and NOT ONE proposed a "
                 f"candidate — the loop runs and mints nothing, which is a "
                 f"different finding from a missing use counter")
    return YieldResult(
        minted=proposed,
        activated=len(packets),
        # Nothing anywhere records that an operator READ or ACTED ON a
        # packet. Counting packets here would make production its own
        # proof of consumption.
        invoked=None,
        note=f"{len(rows)} recorded run(s), {proposed} proposed, "
             f"{len(packets)} operator packet(s) written. Whether an "
             f"operator acted on one is NOT recorded anywhere — E2 refuses "
             f"almost everything by design, so a low count is the cascade "
             f"working, but the consumption end is an instrumentation gap",
    )


def _yield_rubric_shadow(home: Path) -> YieldResult:
    """§4CU — the rubric shadow on declined chat turns.

    Registered on day one, not once it matters. The §4CS finding was a
    loop that ran for six weeks producing nothing consumable while every
    probe read FIRED; the remedy is not to notice faster next time, it is
    to give a new loop its yield row before it has any output to hide.

    `minted` = shadow judgements written. `activated` = those that
    actually GRADED (an ABSTAIN is a correct outcome but not a usable
    one). `invoked` is deliberately the number a HUMAN LABEL has since
    landed on, because that — not the row count — is what could ever
    promote this out of shadow. A shadow verdict nobody can check against
    ground truth is production posing as consumption, which is the exact
    defect round 1 of §4CS item B built into `evolve.candidates`.
    """
    p = home / "system" / "verifier" / "rubric_shadow.jsonl"
    if not p.is_file():
        # NOT a gap: OFF-by-default means "no file" is the expected state
        # of a correctly-configured box, and NO_SOURCE would put a
        # permanent ⚠ on a feature nobody switched on.
        try:
            from .rubric_grader import shadow_enabled
            on = shadow_enabled()
        except Exception:                                   # noqa: BLE001
            on = False
        if not on:
            return YieldResult(
                minted=0, activated=0, invoked=0, status=GATED,
                note="GHOST_RUBRIC_SHADOW is off — the shadow grader is "
                     "built and wired but does not run. Flip it to start "
                     "accruing paired rows; nothing consumes them until "
                     "the agreement report clears its own gate")
        return YieldResult(minted=0, activated=0, invoked=0,
                           note="switched ON but no judgement written yet")
    try:
        from .rubric_grader import (
            GRADED, MIN_PAIRED, RUBRIC_EPOCH, agreement, read_shadow,
        )
        rows = read_shadow(home)
    except Exception as e:                                  # noqa: BLE001
        return YieldResult(note=f"shadow ledger unreadable: {type(e).__name__}")
    cur = [r for r in rows if r.get("epoch") == RUBRIC_EPOCH]
    stale = len(rows) - len(cur)
    if rows and not cur:
        # ⚠ NOT `empty`. A file full of rows from a superseded epoch is
        # not a loop that minted nothing — EMPTY's own docstring says it
        # must mean "produced nothing", and rendering an epoch bump as
        # the most benign state hides that every row was just invalidated.
        return YieldResult(
            minted=0, activated=0, invoked=0, status=GATED,
            note=(f"all {len(rows)} judgement(s) predate the current epoch "
                  f"{RUBRIC_EPOCH} and are not comparable to it — the store "
                  f"is NOT empty, it was invalidated by a prompt or scale "
                  f"change. New rows accrue from the next qualifying turn"))
    graded = [r for r in cur if r.get("status") == GRADED]
    last = max((float(r.get("ts") or 0.0) for r in cur), default=0.0)
    labels = _human_labels(home)
    ag = agreement(cur, labels)
    note = (f"{len(graded)}/{len(cur)} judgement(s) GRADED, the rest "
            f"ABSTAINED (small talk and unparseable grades abstain by "
            f"design — an abstain is a correct outcome, not a failure). "
            f"{ag['verdict']}")
    if stale:
        note += (f". {stale} row(s) from an earlier epoch excluded — a "
                 f"prompt or scale change makes rows incomparable")
    return YieldResult(
        minted=len(cur),
        activated=len(graded),
        invoked=ag["n"],
        last_invoked=last or None,
        note=note,
        # ⚠ BOTH derived zeros, not one. A store of correct ABSTAINs read
        # "✗ produces artifacts NOBODY INVOKES" — but an ABSTAIN can
        # never join a label, so that zero is arithmetic, not neglect.
        # Round 1 covered only the `graded > 0` half.
        derived_zero=(
            (f"{len(graded)} graded judgement(s) exist but none has a human "
             f"label to be checked against yet — the zero is the LABEL "
             f"channel, not the grader")
            if graded and not ag["n"] else
            (f"all {len(cur)} judgement(s) ABSTAINED, and an abstain can "
             f"never join a label — this zero is arithmetic, not neglect")
            if cur and not graded else ""),
    )


def _human_labels(home: Path) -> Dict[str, str]:
    """trajectory_id → outcome, HUMAN rows only.

    Machine verdicts are excluded on purpose: scoring a judge against
    another judge measures their shared blind spot, and §4CE found an
    instrument credited with beating a base rate on a delta whose CI
    straddled zero. The corrections sidecar is the human channel's
    durable record (§4BT).
    """
    out: Dict[str, str] = {}
    p = home / "system" / "trajectories" / "corrections.jsonl"
    if not p.is_file():
        return out
    try:
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                               # noqa: BLE001
                continue
            if not isinstance(r, dict):
                continue
            src = str(r.get("source") or "")
            # `operator_overlay` is documented in core/agent.py as "a
            # manual out-of-process edit" — a HUMAN channel that failed
            # both the prefix test and the allow-list, silently dropping
            # rows from the only ground truth this view has.
            # A PREFIX is a proxy: `startswith("human")` also admits
            # `humanoid_autoverifier`, i.e. a machine, into the only
            # ground truth this view has. The live channels are
            # `human_feedback:*`; require the separator.
            if not (src.startswith("human_feedback:") or src in (
                    "human_feedback", "user_correction", "feedback",
                    "slack_reaction", "operator_overlay", "operator")):
                continue
            tid = str(r.get("trajectory_id") or r.get("id") or "")
            oc = str(r.get("outcome") or "")
            if tid and oc in ("passed", "failed"):
                out[tid] = oc
    except Exception:                                       # noqa: BLE001
        return out
    return out


def _yield_mined_envs(home: Path) -> YieldResult:
    """§4CV — failure environments mined into verifiable-reward tasks.

    Registered on day one, like the rubric shadow, and for the reason a
    round-1 reviewer gave when §4CV shipped WITHOUT one: "the loop that
    can actually go dark is the one without a yield row". §4CS is the
    whole precedent — the macro loop ran for six weeks producing nothing
    consumable while every liveness probe read FIRED.

    `minted` = items that survived BOTH gates and are staged.
    `activated` = those PROMOTED into the live bank directory (an
    explicit operator act; staging deliberately is not arming).
    ⚠ `invoked` IS UNMEASURED, and saying so is the point. The first
    version counted "staged items a GEPA metric COULD train on", which is
    a property of the rows — production wearing a consumption label, the
    exact defect §4CS item B's round 1 built into `evolve.candidates`,
    rebuilt in the probe that cites it. Nothing durable records a GEPA
    run touching this bank: `run_gepa` PRINTS its oracle counts and
    writes nothing. Compare `verifier.rubric_shadow`, whose `invoked` is
    a real external join (paired human labels).
    `activated` (promoted into the live bank) is a real state change and
    stays. The trainable count moves to the note, where it is a fact
    about the store rather than a claim about consumption.
    """
    try:
        from ..optim.env_mining import (
            GRADED_TEXT, MINING_EPOCH, _read_raw, read_staging, staging_path,
        )
    except Exception as e:                                  # noqa: BLE001
        return YieldResult(note=f"env_mining unreadable: {type(e).__name__}")
    try:
        p = staging_path("ghost_failures", str(home))
    except Exception:                                       # noqa: BLE001
        return YieldResult(note="staging path unavailable")
    if not p.is_file():
        # NOT a gap: the miner is operator-triggered, so "never run" is
        # the expected state of a correctly-configured box.
        # ⚠ NOT "never run". This is derived from the STAGING FILE's
        # absence, and `mine_failure_envs.py` returns before writing when
        # a run accepts nothing — so a real run that mined 12 candidates
        # and accepted 0 produces exactly this state. Asserting "never
        # run" about a loop that ran and refused everything is the
        # opposite finding.
        return YieldResult(
            minted=0, activated=0, invoked=None, status=GATED,
            note="no staged items. Either the miner has not been run, or "
                 "a run accepted nothing — the staging file is only "
                 "written on acceptance, so those two are "
                 "INDISTINGUISHABLE from here. Operator-triggered by "
                 "design (scripts/mine_failure_envs.py); nothing "
                 "schedules it")
    cur = read_staging("ghost_failures", str(home))
    stale = len(_read_raw("ghost_failures", str(home))) - len(cur)
    usable = [r for r in cur
              if str(r.get("graded_on") or "") == GRADED_TEXT]
    promoted = 0
    try:
        bank = home / "system" / "bench" / "banks" / "ghost_failures.jsonl"
        if bank.is_file():
            promoted = sum(1 for ln in bank.read_text(
                errors="replace").splitlines() if ln.strip())
    except Exception:                                       # noqa: BLE001
        promoted = 0
    unconfined = sum(1 for r in cur if r.get("verified_confined") is not True)
    note = (f"{len(usable)}/{len(cur)} staged item(s) are text-graded and can "
            f"reach a GEPA metric; the rest are artifact-graded and are bench "
            f"output this consumer cannot train on")
    if stale:
        note += (f". {stale} row(s) from a superseded mining epoch are kept "
                 f"on disk and not served (current: {MINING_EPOCH})")
    if unconfined:
        note += (f". ⚠ {unconfined} item(s) were verified by execution that "
                 f"was NOT kernel-sandboxed — their oracles ran with the "
                 f"agent's own privileges")
    return YieldResult(
        minted=len(cur),
        activated=promoted,
        # None, not len(usable) — see the docstring. UNMEASURED is the
        # honest state, and its remedy (wire a counter) differs from
        # BARREN's (find a consumer).
        invoked=None,
        note=note,
        derived_zero=("nothing is promoted, so no bench run can reach these "
                      "— staging is deliberately not arming"
                      if cur and not promoted else ""),
    )


YIELD_PROBES: List[YieldProbe] = [
    YieldProbe("macros.auto_mined",
               "system/memory/composed_skills/composed_skills.json",
               _yield_macros,
               activated_means="status == active (operator-approved)",
               invoked_means="usage_count summed over auto-mined macros"),
    YieldProbe("skills.acquired",
               "system/memory/acquired_skills/skills_registry.json",
               _yield_acquired_skills,
               activated_means="status == active (a degradation flag — see note)",
               invoked_means="usage_count summed (excludes 'infra' results)"),
    YieldProbe("skills.graduated",
               "system/memory/auto_skills.json",
               _yield_graduated_skills,
               activated_means="surfaced into a prompt at least once",
               invoked_means="prompt injections on REAL turns — NOT helpfulness"),
    YieldProbe("lessons.playbook",
               "system/memory/skills_playbook.json",
               _yield_lessons,
               activated_means="retrieved at least once",
               invoked_means="retrievals summed"),
    YieldProbe("prompts.gepa",
               "system/optim/*.json",
               _yield_gepa,
               activated_means="artifact carries a usable instruction",
               invoked_means=("artifact LOADS from the loader's own log, "
                              "ALL-TIME — a lower bound; see the note")),
    YieldProbe("foresight.gate",
               "system/foresight/gate.json",
               _yield_foresight_gate,
               activated_means="buckets with enabled == true",
               invoked_means="steering sites that consulted a bucket"),
    YieldProbe("evolve.candidates",
               "system/evolve/mutations.jsonl → proposals/",
               _yield_evolve,
               activated_means="operator packets written",
               invoked_means="an operator acting on a packet — NOT RECORDED"),
    YieldProbe("mining.failure_envs",
               "system/optim/mined_envs/ghost_failures.jsonl",
               _yield_mined_envs,
               activated_means="items PROMOTED into the live bank directory",
               invoked_means=("a GEPA run consuming the bank — NOT "
                              "RECORDED; run_gepa prints its oracle "
                              "counts and writes nothing")),
    YieldProbe("verifier.rubric_shadow",
               "system/verifier/rubric_shadow.jsonl",
               _yield_rubric_shadow,
               activated_means="judgements that GRADED (not ABSTAIN)",
               invoked_means="graded rows a HUMAN label can be checked "
                             "against — NOT the row count"),
]


def yield_all(ghost_home: Optional[Path] = None) -> Dict[str, Any]:
    """Run every yield probe. Read-only, never raises, worst-news-first."""
    home = _home(ghost_home)
    now = time.time()
    rows = []
    for pr in YIELD_PROBES:
        try:
            res = pr.fn(home)
        except Exception as e:                              # noqa: BLE001
            res = YieldResult(note=f"probe raised: {type(e).__name__}")
        rows.append({
            "name": pr.name,
            "source": pr.source,
            "status": _yield_status(res),
            "minted": res.minted,
            "activated": res.activated,
            "invoked": res.invoked,
            "age_h": (round((now - res.last_invoked) / 3600.0, 1)
                      if res.last_invoked else None),
            "note": res.note,
            "derived_zero": res.derived_zero,
            "activated_means": pr.activated_means,
            "invoked_means": pr.invoked_means,
            "spread": (None if res.spread is None
                       else {"n": res.spread.n,
                             "total": res.spread.total,
                             "top1": res.spread.top1,
                             "nonzero": res.spread.nonzero,
                             "top1_share": res.spread.top1_share,
                             "coverage": res.spread.coverage,
                             "entropy_ratio": res.spread.entropy_ratio,
                             "verdict": res.spread.verdict,
                             "why": res.spread.why}),
        })
    # RETIRED sorts with the SETTLED states, not the actionable ones. It
    # is deliberately below GATED: a gated loop can be switched back on by
    # an operator decision, a retired one is waiting on a better index.
    order = {NO_SOURCE: 0, BARREN: 1, UNMEASURED: 2, EMPTY: 3, GATED: 4,
             RETIRED: 5, YIELDING: 6}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["name"]))
    return {
        "rows": rows,
        "n_probes": len(rows),
        # BARREN, measured: the artifacts exist, could have been used, and
        # were not. This is the actionable set.
        "barren": [r["name"] for r in rows
                   if r["status"] == BARREN and not r["derived_zero"]],
        # BARREN, DERIVED: the zero follows from an upstream block, so the
        # remedy is upstream and "nobody invokes them" would misdirect.
        "blocked": [r["name"] for r in rows
                    if r["status"] == BARREN and r["derived_zero"]],
        # UNMEASURED is the OTHER actionable set, and the remedy differs —
        # wire a counter, do not kill the loop.
        "unmeasured": [r["name"] for r in rows if r["status"] == UNMEASURED],
        "gaps": [r["name"] for r in rows if r["status"] == NO_SOURCE],
        # Minted nothing at all — a different finding from "minted things
        # nobody invokes", and NOT the same remedy.
        "empty": [r["name"] for r in rows if r["status"] == EMPTY],
        # Measured dead and recorded as such. Listed so it can be ALARMED
        # ON in the opposite direction from everything else here: a name
        # LEAVING this list means a retired loop came back, which is the
        # one event on this view worth waking somebody for.
        "retired": [r["name"] for r in rows if r["status"] == RETIRED],
        # Loops whose retrieval is measurably collapsed onto one item, or
        # whose store is mostly unreachable. Separate from `barren`
        # because the remedy is the RETRIEVER, not the producer — minting
        # more of a store nobody draws from cannot help.
        "concentrated": [r["name"] for r in rows
                         if (r["spread"] or {}).get("verdict") == CONCENTRATED],
    }


def render_yield(ghost_home: Optional[Path] = None) -> str:
    r = yield_all(ghost_home)
    out = ["LOOP YIELD (minted → activated → invoked — did the loop produce "
           "anything anyone CONSUMED?):"]
    # ⚠ SIZED FROM THE DATA, not a literal. A hardcoded 20 silently ate
    # the column separator the moment a probe named `verifier.rubric_
    # shadow` (22 chars) was registered — the row read
    # "verifier.rubric_shadowgated". Same class as the 8-char mark: a
    # fixed width is a constant that has to be re-derived every time the
    # data changes, and nothing fails when it is not.
    _w = max([20] + [len(row["name"]) + 2 for row in r["rows"]])
    for row in r["rows"]:
        def _n(v):
            return "    ?" if v is None else f"{v:>5}"
        # Every mark is 8 chars: a 4-char mark shifted the whole row's
        # columns left, and NO_SOURCE — which sorts FIRST — had no mark
        # at all.
        mark = {BARREN: ("  ⊘ UP  " if row["derived_zero"] else "  ✗ NIL "),
                UNMEASURED: "  ? GAP ",
                EMPTY: "  · NIL ",
                NO_SOURCE: "  ⚠ SRC ",
                GATED: "  - off ",
                RETIRED: "  † END ",
                YIELDING: "        "}.get(row["status"], "        ")
        age = "never" if row["age_h"] is None else f"{row['age_h']}h ago"
        out.append(f"{mark}{row['name']:<{_w}}{row['status']:<11}"
                   f"minted{_n(row['minted'])}  activated{_n(row['activated'])}"
                   f"  invoked{_n(row['invoked'])}  last {age}")
        out.append(f"        └ activated = {row['activated_means']}; "
                   f"invoked = {row['invoked_means']}")
        if row["note"]:
            out.append(f"        └ {row['note']}")
        # ⚠ `derived_zero` reached the mark and the `blocked` summary and
        # NOTHING ELSE, so on any non-BARREN row the one line explaining
        # the zero was computed and thrown away — `mining.failure_envs`
        # knew "staging is deliberately not arming" and never said it.
        # An explanation that only survives in the JSON payload is an
        # explanation the operator does not get.
        if row["derived_zero"] and row["status"] != BARREN:
            out.append(f"        └ why the zero: {row['derived_zero']}")
        sp = row["spread"]
        if sp:
            # A sum cannot see concentration, so the shares are printed
            # even when the verdict is withheld — the numbers are facts,
            # the verdict is a judgement, and only the judgement needs a
            # denominator. Never printing them would hide the very thing
            # the axis was added to expose; printing a VERDICT under the
            # floor would be §4CE's failure rebuilt here.
            share = ("—" if sp["top1_share"] is None
                     else f"{sp['top1_share']:.0%}")
            cov = ("—" if sp["coverage"] is None
                   else f"{sp['nonzero']}/{sp['n']}")
            ent = ("—" if sp["entropy_ratio"] is None
                   else f"{sp['entropy_ratio']:.2f}")
            out.append(f"        └ retrieval spread: {sp['verdict']} — "
                       f"top-1 {share} of {sp['total']}, reached {cov}, "
                       f"evenness {ent} ({sp['why']})")
        out.append(f"        └ source: {row['source']}")
    if r["barren"]:
        out.append(f"  ✗ {len(r['barren'])} loop(s) produce artifacts NOBODY "
                   f"INVOKES: " + ", ".join(r["barren"]))
    if r["blocked"]:
        out.append(f"  ⊘ {len(r['blocked'])} loop(s) read zero because "
                   f"something UPSTREAM blocks them, not because their "
                   f"output is ignored — the remedy is upstream: "
                   + ", ".join(f"{n} ({d})" for n, d in
                               ((row["name"], row["derived_zero"])
                                for row in r["rows"]
                                if row["name"] in r["blocked"])))
    if r["unmeasured"]:
        out.append(f"  ? {len(r['unmeasured'])} loop(s) have artifacts but NO "
                   f"durable use counter — a zero here would be unreadable, "
                   f"so none is claimed: " + ", ".join(r["unmeasured"]))
    if r["gaps"]:
        out.append(f"  ⚠ {len(r['gaps'])} artifact store(s) absent or "
                   f"unreadable: " + ", ".join(r["gaps"]))
    if r["empty"]:
        out.append(f"  · {len(r['empty'])} loop(s) have minted NOTHING yet "
                   f"(not the same as artifacts nobody invokes): "
                   + ", ".join(r["empty"]))
    if r["concentrated"]:
        out.append(f"  ◑ {len(r['concentrated'])} loop(s) are CONSUMED BUT "
                   f"COLLAPSED — the retrievals land on a fraction of the "
                   f"store, so the remedy is the RETRIEVER, not more "
                   f"minting: " + ", ".join(r["concentrated"]))
    if r["retired"]:
        out.append(f"  † {len(r['retired'])} loop(s) RETIRED — measured "
                   f"dead, decision recorded, nothing owed. Derived, so "
                   f"this retracts itself if the measurement flips: "
                   + ", ".join(r["retired"]))
    # ⚠ The closing all-clear must account for EVERY non-yielding state,
    # not just the actionable ones. Before RETIRED existed the one settled
    # state (GATED) never coincided with an otherwise empty finding list,
    # so the omission was unreachable — adding another settled
    # state made "every probed loop has a live consumer" printable on a
    # view whose top two rows were a retired loop and a switched-off one.
    # A summary that contradicts the rows above it is worse than none.
    settled = [row["name"] for row in r["rows"]
               if row["status"] in (GATED, RETIRED)]
    if not (r["barren"] or r["blocked"] or r["unmeasured"] or r["gaps"]
            or r["empty"] or r["concentrated"]):
        if settled:
            out.append(f"  every loop with a live consumer has one; "
                       f"{len(settled)} not in service: " + ", ".join(settled))
        else:
            out.append("  every probed loop has a live consumer")
    return "\n".join(out)
