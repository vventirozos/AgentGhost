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
PROBES: List[Probe] = [
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
          _log_probe(r"GEPA: loaded tuned instruction", window_h=168.0)),
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
            "alarm": bool(pr.alarm_if_zero and res.status == ZERO),
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
        out.append(f"{mark}{row['name']:<24}{row['status']:<10}"
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
