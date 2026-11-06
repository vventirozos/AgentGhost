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
# §4MC MAJOR 1: the log never rotated (82 MB) and this parse held EVERY line —
# 4.6 s and 191 MB resident. Now only the last LOG_SCAN_DAYS (the longest
# windowed probe is 7 days; 14 keeps a recent "last fired") of the current file and its newest rotated
# generation are read: a binary search on the line timestamps finds the start.
_LOG_CACHE: Dict[tuple, List[tuple]] = {}
LOG_SCAN_DAYS = 14.0
_TS_RX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _line_ts(line: str) -> Optional[float]:
    m = _TS_RX.match(line)
    if not m:
        return None
    try:
        return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def _offset_at(fh, size: int, cutoff: float) -> int:
    """Byte offset of the first timestamped line at or after ``cutoff``
    (binary search over the file; lines are appended in time order)."""
    lo, hi = 0, size
    while hi - lo > 4096:
        mid = (lo + hi) // 2
        fh.seek(mid)
        fh.readline()                         # to the next line start
        ts = None
        for _ in range(50):                   # skip untimestamped lines
            line = fh.readline()
            if not line:
                break
            ts = _line_ts(line.decode("utf-8", "replace"))
            if ts is not None:
                break
        if ts is None or ts >= cutoff:
            hi = mid
        else:
            lo = mid
    return lo


def _read_recent(p: Path, cutoff: float, out: List[tuple]) -> bool:
    try:
        with p.open("rb") as fh:
            size = p.stat().st_size
            fh.seek(_offset_at(fh, size, cutoff))
            for raw in fh:
                line = raw.decode("utf-8", "replace")
                ts = _line_ts(line)
                if ts is not None and ts >= cutoff:
                    out.append((ts, line))
        return True
    except OSError:
        return False


def _log_entries(p: Path) -> Optional[List[tuple]]:
    """[(epoch_ts, line)] for timestamped mirror lines of the last
    LOG_SCAN_DAYS, oldest first, across the newest rotated generation and the
    current file. One parse, shared."""
    try:
        st = p.stat()
    except OSError:
        return None
    rotated = p.with_name(p.name + ".1")
    try:
        rst = rotated.stat()
        rkey = (rst.st_size, rst.st_mtime)
    except OSError:
        rkey = None
    key = (str(p), st.st_size, st.st_mtime, rkey)
    hit = _LOG_CACHE.get(key)
    if hit is not None:
        return hit
    _LOG_CACHE.clear()          # only ever hold the current log
    cutoff = time.time() - LOG_SCAN_DAYS * 86400.0
    out: List[tuple] = []
    if rkey is not None:
        _read_recent(rotated, cutoff, out)
    if not _read_recent(p, cutoff, out):
        return None
    _LOG_CACHE[key] = out
    return out


def _stream_count(p: Path, rx, ex, cutoff: float) -> ProbeResult:
    """Count matching timestamped lines at/after ``cutoff`` across the log
    and all its rotated generations, in O(1) memory."""
    n, last = 0, None
    gens = sorted(p.parent.glob(p.name + ".*"),
                  key=lambda q: -int(q.suffix[1:]) if q.suffix[1:].isdigit() else 0)
    try:
        for f in [g for g in gens if g.suffix[1:].isdigit()] + [p]:
            with f.open("rb") as fh:
                for raw in fh:
                    line = raw.decode("utf-8", "replace")
                    if not rx.search(line) or (ex is not None and ex.search(line)):
                        continue
                    ts = _line_ts(line)
                    if ts is None:
                        continue
                    last = max(last or 0.0, ts)
                    if ts >= cutoff:
                        n += 1
    except OSError:
        return ProbeResult(NO_SOURCE, note="log unreadable")
    return ProbeResult(FIRED if n else ZERO, n, last)


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
        if window_h > LOG_SCAN_DAYS * 24.0:
            # an all-time count (GEPA loads): stream every generation, hold
            # only the count and the newest match — never the lines (§4MC)
            return _stream_count(p, rx, ex, time.time() - window_h * 3600.0)
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
    origin_rx = re.compile(r"\borigin=(\w+)")
    n = 0
    total = 0
    unstamped = 0
    # the shared bounded reader: the current file AND its newest rotated
    # generation (§4MC review: a single-file scan under-counted for a day
    # after every rollover)
    entries = _log_entries(p)
    if entries is None:
        return (None, 0, "no-log")
    for ts, line in entries:
        if ts < cutoff or "request started" not in line:
            continue
        total += 1
        o = origin_rx.search(line)
        if o is None:
            unstamped += 1
        elif o.group(1) == "user" and " role=member" not in line:
            n += 1              # a member's turn feeds no owner store (§4KJ R10)
    if unstamped:
        # Mixed or wholly pre-stamp window: the stamped subset is a LOWER
        # bound on user turns, never the count. Say so instead of quoting it.
        return (None, total,
                f"unclassified ({unstamped} request line(s) predate the "
                f"origin stamp; {n} stamped user)")
    return (n, total, "")


# ── THE REGISTRY ─────────────────────────────────────────────────────────────
# Everything here is DELIBERATELY outside autonomous_activity.jsonl. The phases
# inside it stay with `learning_health.activity_liveness`; duplicating them
# would create the two-implementations drift this repo keeps being bitten by.
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
    # §4OX: with the LLM judge off (§4OW) the mechanical arms are the verifier —
    # their "mechanical checks …" and WEB-EXEC / FILE-ARTIFACT lines count as it running
    Probe("verifier.outcomes", EXPECT_ON_OUTPUT,
          "system/ghost-agent.log — verifier verdict, escalation and mechanical-check lines",
          _log_probe(r"(?i)verifier\b.*"
                     r"(CONFIRMED|REFUTED|SKIPPED|UNCERTAIN|OVERTURNED|ERROR |mechanical checks found|"
                     r"WEB-EXEC|FILE-ARTIFACT)",
                     window_h=24.0,
                     # The agent REASONING about the verifier is not the
                     # verifier running. 12/1370 matches were such prose.
                     # §4KV: a reply drafted with thinking off is the same
                     # prose under its own title ("drafting —").
                     exclude=r"(?:thinking|drafting) —"),
          denominator=DEN_REQUESTS),
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
#     Nothing anywhere said so (§4CS). (The miner was removed 2026-10-11
#     and the macro engine itself on 2026-11-05, §4TA — the axis stays.)
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
# (since removed) macro loop (genuinely barren, and fixable) and the graduated-skill loop
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
#: it a consumer". The (since removed) foresight gate read BARREN for weeks
#: while the real answer was that the question was SETTLED: its index had
#: measured ANTI-PREDICTIVE, so more traffic could not fix it.
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
# counter that `invoked` sums (`usage_count` on acquired skills; the lesson
# store's `retrievals` went with it, §4TD). This is a second
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
    zero-count items — measured on the (since removed, §4TA) macro store's
    shape, eligible-only and all-29 both give `top1_share = 1.0`. (There
    were also three runnable macros, not one.)

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
    active and later demoted (a skill degraded), a
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
    # ordinary store shape (two active skills). The
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
    #: retirement DECISION is computed from the loop's own data every time
    #: its yield probe runs. This field carries that decision; it does not
    #: stand in for one.
    status: Optional[str] = None
    #: ⚠ REVIEW ROUND 2. BARREN's contract is "measurable, and MEASURED at
    #: zero", and a live member violated it: the macro loop's zero was
    #: structural (a `proposed` macro was not advertised and could not be
    #: run; that row is gone, §4TA). An operator triaging "produce
    #: artifacts NOBODY INVOKES" concludes the artifacts are being ignored;
    #: the truth is they are BLOCKED UPSTREAM, which is a different remedy.
    #: Same BARREN/UNMEASURED distinction, one level finer.
    derived_zero: str = ""
    #: How `invoked` is distributed across the loop's OWN items, when the
    #: loop keeps per-item counters. None means the loop has no per-item
    #: channel to distribute over (gepa serves ONE artifact) — which is
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
    # channel is unmeasured. An empty store (the since-removed
    # `auto_skills.json`) rendered
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


# (The `macros.auto_mined` row — `_yield_macros`, `_is_loop_minted_macro`,
# `_macro_marks` — went with the macro engine, §4TA 2026-11-05. The old
# `composed_skills/composed_skills.json` is no longer read by anything.)


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


# (The `lessons.playbook` row — `_yield_lessons` — went with the lesson store,
# §4TD 2026-11-06. An old `skills_playbook.json` is no longer read by anything.)


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
    defect round 1 of §4CS item B built into the since-removed evolve probe.
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


YIELD_PROBES: List[YieldProbe] = [
    YieldProbe("skills.acquired",
               "system/memory/acquired_skills/skills_registry.json",
               _yield_acquired_skills,
               activated_means="status == active (a degradation flag — see note)",
               invoked_means="usage_count summed (excludes 'infra' results)"),
    YieldProbe("prompts.gepa",
               "system/optim/*.json",
               _yield_gepa,
               activated_means="artifact carries a usable instruction",
               invoked_means=("artifact LOADS from the loader's own log, "
                              "ALL-TIME — a lower bound; see the note")),
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
