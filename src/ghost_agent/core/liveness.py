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


PROBES: List[Probe] = [
    # ⚠ DEN_NONE, deliberately: these run off the idle clock, not off
    # traffic, so no amount of quiet excuses their silence.
    Probe("evolve.negative_controls", EXPECT_PERIODIC,
          "system/evolve/negative_controls.json",
          _negative_controls_probe,
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


@dataclass
class YieldResult:
    minted: Optional[int] = None
    activated: Optional[int] = None
    invoked: Optional[int] = None
    last_invoked: Optional[float] = None
    note: str = ""
    #: Set only to force GATED. Otherwise derived — see `_yield_status`.
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
    last = max((float(v.get("last_used") or 0.0) for v in auto.values()),
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
_GEPA_LOAD_PATTERN = (r"GEPA: (loaded tuned instruction|artifact '[^']+' "
                      r"\(sha [0-9a-f]+\) predates the gate schema)")


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
    loads = _log_probe(_GEPA_LOAD_PATTERN,
                       window_h=_GEPA_LOAD_WINDOW_H)(home)
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
        verdict = str(((data.get("discrimination") or {}).get("verdict")) or "")
        if verdict:
            note += f". POOLED: {verdict}"
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
               invoked_means="loader log lines over 7 days (a lower bound)"),
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
        })
    order = {NO_SOURCE: 0, BARREN: 1, UNMEASURED: 2, EMPTY: 3, GATED: 4,
             YIELDING: 5}
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
    }


def render_yield(ghost_home: Optional[Path] = None) -> str:
    r = yield_all(ghost_home)
    out = ["LOOP YIELD (minted → activated → invoked — did the loop produce "
           "anything anyone CONSUMED?):"]
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
                YIELDING: "        "}.get(row["status"], "        ")
        age = "never" if row["age_h"] is None else f"{row['age_h']}h ago"
        out.append(f"{mark}{row['name']:<20}{row['status']:<11}"
                   f"minted{_n(row['minted'])}  activated{_n(row['activated'])}"
                   f"  invoked{_n(row['invoked'])}  last {age}")
        out.append(f"        └ activated = {row['activated_means']}; "
                   f"invoked = {row['invoked_means']}")
        if row["note"]:
            out.append(f"        └ {row['note']}")
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
    if not (r["barren"] or r["blocked"] or r["unmeasured"] or r["gaps"]
            or r["empty"]):
        out.append("  every probed loop has a live consumer")
    return "\n".join(out)
