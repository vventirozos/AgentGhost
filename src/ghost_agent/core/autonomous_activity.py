"""Autonomous-activity ledger — the agent's outbound "mouth" (2026-07-11).

The idle battery (dream, reflection, post-mortem, skills graduation, PRM /
router / calibration retrains, self-play) and scheduled turns all do real
work while the operator is away — but until this module, only project
autoadvance ever told the user (via ``core.project_digest``); everything
else surfaced solely as ``pretty_log`` lines on the console. This ledger
records every operator-relevant autonomous outcome as a structured,
append-only JSONL line with three consumers:

1. **The next-turn digest** — ``_finalize_and_return`` renders unseen
   records as a "Background activity" header, watermarked by byte offset
   so each batch shows exactly once (mirrors the project-digest pattern).
2. **The outbound notifier** (``utils.notify``) — records with
   ``severity="notify"`` fire the ``on_notify`` callback for immediate
   push delivery when a transport is configured.
3. **External deliverers** (e.g. the Slack bot) — poll
   ``/api/notifications/pending`` with a durable per-consumer watermark
   and ack what they delivered.

Fail-safe by contract: no public function here may raise into a caller —
a broken activity log must never break a turn or an idle phase.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

SEVERITY_INFO = "info"      # digest-only
SEVERITY_NOTIFY = "notify"  # digest + immediate push + consumer feed
_SEVERITIES = (SEVERITY_INFO, SEVERITY_NOTIFY)

_MAX_SUMMARY_CHARS = 600
_MAX_META_VALUE_CHARS = 200
_MAX_LINE_BYTES = 16384

# Phases the digest must NOT render because another surface already covers
# them: project autoadvance outcomes are rendered by core.project_digest
# (they'd double-report). They still land in the ledger so the notifier /
# consumer feed can push needs-user items immediately.
DIGEST_EXCLUDED_PHASES = ("project",)

# Human labels for digest lines. Unknown phases render as their raw slug.
_PHASE_LABELS = {
    "dream": "dream",
    "reflection": "reflection",
    "postmortem": "post-mortem",
    "skills_auto": "skills",
    "prm_train": "PRM",
    "router_train": "router",
    "calibration": "calibration",
    "self_play": "self-play",
    "selfplay_selftest_skip": "self-play validator-selftest skip",
    "scheduled_task": "scheduled task",
    "agent_message": "message from the agent",
    "open_questions": "open questions",
    "project": "project",
    "service": "service",
    "job": "background job",
    # ⚠ These three were MISSING and rendered as raw slugs (2026-08-10).
    "experiment_verdict": "experiment verdict",
    "native_tool_repair": "native tool-call repair",
    "workspace_tidy": "workspace tidy",
    "bench": "bench bank",
    "imagine_gate": "imagine calibration gate",
    "dream_replay": "counterfactual replay",
    "evolve_mutate": "Evolve mutator",
    "evolve_proposal": "Evolve proposal",
    "negative_controls": "Negative Controls",
    # §4DC Phase 0+1: the autonomous GEPA supply watch + live judge.
    "gepa_autonomy": "GEPA autonomy",
}

# ── LIVENESS REGISTRY ───────────────────────────────────────────────────────
#
# WHAT A ZERO MEANS. Counting ledger records answers "did it fire?" only if you
# already know whether it was SUPPOSED to. Before this registry the background
# report enumerated phases observationally — straight from whatever strings
# happened to be in the ledger — so a loop that stopped writing, or never
# wrote, was INDISTINGUISHABLE FROM ONE THAT DOES NOT EXIST. Measured
# 2026-08-10: 15 phases are instrumented in source, 7 appeared in a 7-day
# window, and the report showed seven green rows and no hint of the other
# eight.
#
# It is not enough to render the zeros either: `reflection` legitimately sits
# at 0 (it records only outcome-producing runs), while `dream` at 0 would be a
# dead loop. A benign zero and a fatal zero MUST NOT look alike — that is the
# whole defect, one level up. So each phase declares what its own zero means:
#
#   PERIODIC   — scheduled; it fires on its own. Zero over the window is an
#                ALARM. These are the five loops measured firing continuously.
#   ON_OUTPUT  — runs on schedule but only RECORDS when it produced something
#                (`if stale:`, `if _tidy_deleted:`). Zero = "no new material",
#                never "not scheduled". Alarming on these is how a monitor
#                teaches its operator to ignore it.
#   ON_DEMAND  — fires only when a turn, an operator or a verdict triggers it.
#                Zero is the normal state.
#   GATED      — needs a flag or consumer that may be off (PRM retrain skips
#                when neither .score() nor .uncertainty() is live). Zero is
#                expected while the gate is closed, so it reports the gate
#                rather than crying about the count.
EXPECT_PERIODIC = "periodic"
EXPECT_ON_OUTPUT = "on_output"
EXPECT_ON_DEMAND = "on_demand"
EXPECT_GATED = "gated"

PHASE_EXPECTATION = {
    # measured firing continuously (7d/24h): 159/21, 155/22, 89/15, 99/14, 43/5
    "self_play": EXPECT_PERIODIC,
    "calibration": EXPECT_PERIODIC,
    "dream": EXPECT_PERIODIC,
    "skills_auto": EXPECT_PERIODIC,
    "router_train": EXPECT_PERIODIC,
    # ⚠ NOT periodic-on-a-24h-window. `negative_controls` runs on a SEVEN
    # DAY interval (evolve/negative_controls.py: INTERVAL_S = 7*24*3600),
    # so EXPECT_PERIODIC against a 24h alarm window brands a healthy weekly
    # loop DEAD six days in seven.
    #
    # ⚠ CORRECTION (an earlier version of this comment said it "writes no
    # ledger row at all"). The agent's idle phase DOES record on all three
    # paths — cannot-run, success and exception (agent.py ~7645/7683/7695).
    # What is measured is narrower and still decisive: the live ledger holds
    # 0 negative_controls rows across 4,405 rows and ~45 days, while the
    # phase's own state file shows a successful run 6.4 days ago. So the
    # runs that happen are not reaching those row-writing paths, and the
    # LEDGER is not a usable liveness signal for this phase either way.
    # Its real signal
    # is its own state file's timestamp, which `core/liveness.py`'s
    # `_negative_controls_probe` already reads; the sibling `gepa_autonomy`
    # was reclassified for exactly this reason and this one was missed.
    # Live check 2026-08-30: last run 6.4 days ago, all 3 controls held —
    # i.e. not yet due, and reported as dead.
    # records only when there was an outcome to record
    "reflection": EXPECT_ON_OUTPUT,          # skips unchanged-corpus ticks
    "selfplay_selftest_skip": EXPECT_ON_OUTPUT,   # a SKIP is the event
    "open_questions": EXPECT_ON_OUTPUT,      # `if stale:`
    "workspace_tidy": EXPECT_ON_OUTPUT,      # `if _tidy_deleted:`
    "native_tool_repair": EXPECT_ON_OUTPUT,  # ~0 by design; each one is news
    # externally triggered
    "scheduled_task": EXPECT_ON_DEMAND,
    "project": EXPECT_ON_DEMAND,
    "experiment_verdict": EXPECT_ON_DEMAND,  # needs an arm to reach n>=30
    "agent_message": EXPECT_ON_DEMAND,
    "service": EXPECT_ON_DEMAND,
    "job": EXPECT_ON_DEMAND,
    # behind a flag / consumer
    "prm_train": EXPECT_GATED,               # skips unless .score()/.uncertainty() live
    "postmortem": EXPECT_GATED,              # --postmortem
    # §4BF Track 1b: fires per deep-idle tick once banks are imported;
    # GATED (not periodic) because it is inert on a box with no banks on
    # disk and killable via --no-bench — a zero must report the gate, not
    # alarm.
    "bench": EXPECT_GATED,
    # §4CL I0: the Imagine calibration gate rebuild. PERIODIC because it
    # is what keeps the gate from going stale, and a stale allow-list is
    # exactly the failure this registry exists to make visible — a zero
    # here means the question "is the precedent index good enough to
    # steer with yet?" has stopped being asked.
    "imagine_gate": EXPECT_PERIODIC,
    # §4DC Phase 0+1: the GEPA supply watch / live judge. ON_OUTPUT —
    # the jobs run on their own wall clock, but a ledger row is written
    # only on a TRANSITION (supply parked->ready, a retirement, an
    # instrument failure); the steady state (parked / KEEP / nothing
    # live) is console-log-only by design (`chat-noise-preference`), so
    # a zero here means "nothing changed", not "the loop is dead" — the
    # loop's own liveness is the pretty_log lines and the state file's
    # last_run_epoch.
    "gepa_autonomy": EXPECT_ON_OUTPUT,
    # §4CM D3: the counterfactual replay batch. GATED, not periodic —
    # `GHOST_DREAM_REPLAY` defaults OFF and the whole engine is inert
    # until an operator turns it on, so a zero must report the gate
    # rather than manufacture an alarm (the `bench` precedent).
    "dream_replay": EXPECT_GATED,
    # §4CN E1: the Evolve mutator. GATED for the same reason —
    # `GHOST_EVOLVE` defaults OFF and the loop is inert until an operator
    # turns it on, so a zero must report the gate rather than manufacture
    # an alarm. When it IS on it writes a row every firing, including the
    # firings that proposed nothing, so a zero here means the phase never
    # ran — not that it had nothing to say.
    "evolve_mutate": EXPECT_GATED,
    # §4CS item E: the scheduled E3 negative controls. PERIODIC, and the
    # zero here is the alarm — E2's entire value is REFUSING things, and
    # this project's rule is that a guard which never demonstrably fires is
    # presumed dead. It fires WEEKLY (a run costs minutes: the guard
    # control runs stage 1 for real over ~45 pin files), and it writes a
    # row on EVERY path including the ones where a control could not be
    # built, so a zero means the phase stopped running rather than that it
    # had nothing to say. `GHOST_NEGCTRL=0` disables it, and that is the
    # only state in which a zero is benign.
    "negative_controls": EXPECT_ON_OUTPUT,
    # §4CN E2 stage 4: a proposal packet reaching an operator. GATED, and
    # a zero here is the EXPECTED reading twice over — `GHOST_EVOLVE` is
    # off by default, and even with it on a candidate must clear static,
    # pins, bench smoke AND a significant paired win on held-out items
    # before one is written. Most generations end in an honest refusal,
    # so a zero means "nothing earned an operator's attention", never
    # "the phase is broken".
    "evolve_proposal": EXPECT_GATED,
}


def phase_expectation(phase: str) -> str:
    """What a ZERO count for ``phase`` means.

    Unknown phases are ON_DEMAND: a slug nobody registered must never manu-
    facture an alarm. The `tests/test_activity_liveness.py` coverage test is
    what keeps "unknown" from becoming a quiet dumping ground — it fails when
    a phase literal exists in source but not in this registry.
    """
    return PHASE_EXPECTATION.get(str(phase or "").strip(), EXPECT_ON_DEMAND)

# Internal request-id prefixes: turns the agent fires at ITSELF (cron jobs,
# delegated sub-agents). The finalize digest must skip these — an internal
# turn consuming the watermark would silently eat the operator's next
# "while you were away" report.
INTERNAL_REQUEST_PREFIXES = ("sched-", "job-", "sub-")


def is_internal_request(req_id) -> bool:
    return str(req_id or "").startswith(INTERNAL_REQUEST_PREFIXES)


@dataclass
class ActivityRecord:
    ts: float
    phase: str
    summary: str
    severity: str = SEVERITY_INFO
    meta: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "phase": self.phase,
            "summary": self.summary,
            "severity": self.severity,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActivityRecord":
        meta = d.get("meta")
        return cls(
            ts=float(d.get("ts") or 0.0),
            phase=str(d.get("phase") or "unknown"),
            summary=str(d.get("summary") or ""),
            severity=(
                str(d.get("severity"))
                if d.get("severity") in _SEVERITIES else SEVERITY_INFO
            ),
            meta=dict(meta) if isinstance(meta, dict) else {},
        )


# ── idle-phase ATTEMPT heartbeat ──────────────────────────────────────────
#
# The activity ledger records OUTCOMES: a phase writes a row when it did
# something. That is the right contract for a digest, and it makes one
# question unanswerable — "did this loop RUN?" — because a loop that ran and
# correctly declined looks identical to one that never ran at all.
#
# Measured 2026-08-30: the liveness view reported
#     ✗ DEAD router_train 0/24h 0/7d
# while the loop had run 30 minutes earlier and logged
#     "router train: the labelled corpus is 89% the same as the last look …
#      not re-running the same test on the same evidence"
# Its last ledger row (2026-08-16) is its last real RETRAIN, which is
# exactly what the ledger promises. The alarm was reading "produced nothing"
# as "is dead", and the operator was being pointed at a healthy loop.
#
# This heartbeat answers the other half. It is deliberately NOT more ledger
# rows: a skip-per-idle-cycle would add ~30 rows/day to a file that never
# rotates, and would blur the ledger's outcome contract. One tiny JSON map,
# phase -> last attempt timestamp, bounded by the number of phases.

from ..tools.file_system import (  # noqa: E402
    write_text_nofollow as _write_text_nofollow,
)

_ATTEMPTS_FILENAME = "idle_attempts.json"

# An attempt's OUTCOME CLASS, not just its timestamp.
#
# ⚠ THE FIRST VERSION STORED A BARE TIMESTAMP AND WAS A FALSE GREEN.
# The stamp is written where a phase ENTERS its work, before the try block
# that does it. With only a timestamp, a loop that crashes on every single
# cycle is byte-identical to one that ran and correctly declined — and the
# renderer then positively asserts "RAN 0.0h ago … a healthy outcome-free
# run" about a permanently broken loop. That is the false GREEN this
# module's own comments call worse than the false alarm it replaced.
#
# So only an explicit DECLINED suppresses the alarm. ENTERED (started,
# outcome unknown — the state a crash leaves behind) and FAILED do not.
ATTEMPT_ENTERED = "entered"
ATTEMPT_DECLINED = "declined"
ATTEMPT_FAILED = "failed"

#: Slack for clock granularity, NOT for clock faults.
#:
#: The consumer captures `now_ts` and then reads the file, so a stamp
#: written microseconds later is legitimately "in the future" by a hair. A
#: strict `0 <= age` rejected exactly that and refused to suppress a genuine
#: DECLINED run. One minute absorbs scheduling jitter and NTP slew while
#: still rejecting the failure this bound exists for — a stamp hours or days
#: ahead, which would otherwise suppress the alarm forever.
_FUTURE_TOLERANCE_S = 60.0

#: Only this outcome means "ran, and producing nothing was correct".
_SUPPRESSING_RESULTS = frozenset({ATTEMPT_DECLINED})

# Guards the read-modify-write below. `ActivityLog` next door has always had
# one; the first version of this store had none, and 8 threads x 30 phases
# recovered 0 of 240 keys AND left the file unparseable.
_ATTEMPTS_LOCK = threading.Lock()


def attempts_path(ledger_path) -> Path:
    """The heartbeat lives beside the ledger it complements."""
    return Path(ledger_path).parent / _ATTEMPTS_FILENAME


def record_attempt(ledger_path, phase: str,
                   result: str = ATTEMPT_ENTERED) -> bool:
    """Stamp `phase` as having run now, with the outcome class it reached.

    Never raises — a heartbeat that can break an idle phase is worse than no
    heartbeat.
    """
    if result not in (ATTEMPT_ENTERED, ATTEMPT_DECLINED, ATTEMPT_FAILED):
        result = ATTEMPT_ENTERED
    tmp = None
    try:
        p = attempts_path(ledger_path)
        with _ATTEMPTS_LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            raw = None
            try:
                raw = p.read_text()
            except FileNotFoundError:
                raw = ""
            except OSError:
                # ⚠ DO NOT RESET. The first version fell back to `data = {}`
                # on ANY read failure and then COMMITTED that reset —
                # destroying every other phase's heartbeat, permanently, with
                # a True return and no log line. One transient EIO or an
                # unreadable file wiped the instrument. Refuse instead.
                logger.warning(
                    "idle-attempt heartbeat: %s unreadable — refusing to "
                    "overwrite it (other phases' stamps would be lost)", p)
                return False
            try:
                data = json.loads(raw or "{}")
            except Exception:  # noqa: BLE001 — corrupt file, safe to replace
                data = {}
            if not isinstance(data, dict):
                data = {}
            data[str(phase)] = {"ts": time.time(), "result": result}
            # pid + uuid, like `sandbox/services.py::_save` and
            # `sandbox/jobs.py`. A pid-only suffix collides between THREADS
            # of one process: B truncates the tmp A is mid-write on, A
            # renames it into place, and B's remaining bytes land in the live
            # file. That is §4BW, which both siblings already fixed.
            tmp = p.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            _write_text_nofollow(tmp, json.dumps(data, indent=2))
            os.replace(tmp, p)
            tmp = None
            return True
    except Exception as e:  # noqa: BLE001
        logger.debug("idle-attempt heartbeat write failed: %s", e)
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def read_attempts(ledger_path) -> Dict[str, dict]:
    """`{phase: {"ts": float, "result": str}}`.

    Empty on any failure — an unreadable heartbeat must degrade to
    "unknown", which the consumer treats as the OLD alarm, never to a green.
    A single bad row is dropped on its own; it must not discard the others
    (one un-floatable value used to empty the whole map).
    """
    out: Dict[str, dict] = {}
    try:
        data = json.loads(attempts_path(ledger_path).read_text() or "{}")
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    for k, v in data.items():
        try:
            if isinstance(v, dict):
                ts, result = v.get("ts"), v.get("result")
            else:
                # A bare timestamp is the pre-outcome-class format. It cannot
                # prove the run DECLINED, so it is read as ENTERED and does
                # not suppress.
                ts, result = v, ATTEMPT_ENTERED
            if isinstance(ts, bool) or not isinstance(ts, (int, float)):
                continue
            ts = float(ts)
            if ts != ts or ts in (float("inf"), float("-inf")):
                continue          # NaN / +-inf are instrument faults
            out[str(k)] = {"ts": ts,
                           "result": (str(result) if result else
                                      ATTEMPT_ENTERED)}
        except Exception:  # noqa: BLE001 — one bad row, not the whole map
            continue
    return out


def attempt_suppresses_alarm(entry, now_ts: float,
                             window_s: float) -> bool:
    """Does this heartbeat license withholding a DEAD alarm?

    Three conditions, all required:
      * the run DECLINED — entered-but-unfinished is what a crash leaves;
      * it was recent;
      * it is not in the FUTURE. The first version tested only
        `now - ts <= window`, so a future-dated stamp (an NTP step back, a
        hand-edited file) suppressed the alarm forever. This file already
        documents that exact hazard for ledger timestamps 100 lines above —
        "a single garbage-ts row would keep a dead PERIODIC phase looking
        alive in the liveness table forever" — and the first version did not
        inherit it.
    """
    try:
        if not isinstance(entry, dict):
            return False
        if str(entry.get("result")) not in _SUPPRESSING_RESULTS:
            return False
        age = float(now_ts) - float(entry.get("ts"))
        return -_FUTURE_TOLERANCE_S <= age <= float(window_s)
    except Exception:  # noqa: BLE001
        return False


class ActivityLog:
    """Append-only JSONL ledger. Thread-safe, never raises."""

    def __init__(self, path,
                 on_notify: Optional[Callable[[ActivityRecord], None]] = None):
        self.path = Path(path)
        self.on_notify = on_notify
        self._lock = threading.Lock()

    def record(self, phase: str, summary: str,
               severity: str = SEVERITY_INFO, **meta) -> bool:
        """Append one record. Returns False (never raises) on any failure.
        ``severity="notify"`` additionally fires ``on_notify`` (errors in
        the callback are swallowed — delivery is best-effort by design)."""
        try:
            if severity not in _SEVERITIES:
                severity = SEVERITY_INFO
            rec = ActivityRecord(
                ts=time.time(),
                phase=str(phase or "unknown")[:64],
                summary=" ".join(str(summary or "").split())[:_MAX_SUMMARY_CHARS],
                severity=severity,
                meta={
                    str(k)[:64]: str(v)[:_MAX_META_VALUE_CHARS]
                    for k, v in (meta or {}).items()
                },
            )
            line = json.dumps(rec.to_dict(), ensure_ascii=False)
            if len(line.encode("utf-8", "ignore")) > _MAX_LINE_BYTES:
                rec.meta = {}
                rec.summary = rec.summary[:200]
                line = json.dumps(rec.to_dict(), ensure_ascii=False)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # §4CB R1 B-M6: heal a crash-truncated tail. If the process
                # died mid-write, the file ends without "\n"; appending
                # directly would MERGE this record into the partial line,
                # making both unparseable — read_since skips the merged line
                # and advances, so the first post-restart record silently
                # never reached the digest/poller consumers. A leading
                # newline isolates the damage to the half-written record.
                _prefix = ""
                try:
                    _sz = os.path.getsize(self.path)
                    if _sz > 0:
                        with open(self.path, "rb") as _rf:
                            _rf.seek(_sz - 1)
                            if _rf.read(1) != b"\n":
                                _prefix = "\n"
                except OSError:
                    pass
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(_prefix + line + "\n")
        except Exception as e:  # noqa: BLE001 — fail-safe contract
            logger.debug("activity record failed: %s", e)
            return False
        if rec.severity == SEVERITY_NOTIFY and self.on_notify is not None:
            try:
                self.on_notify(rec)
            except Exception as e:  # noqa: BLE001
                logger.debug("activity on_notify failed: %s", e)
        return True

    def current_offset(self) -> int:
        """Current end-of-file byte offset (the baseline watermark)."""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def read_since(self, offset: int, *, limit: int = 200,
                   severity: Optional[str] = None,
                   ) -> Tuple[List[ActivityRecord], int]:
        """Read records appended after byte ``offset``.

        Returns ``(records, new_offset)``. The offset only advances past
        COMPLETE lines (a partially-written tail line is left for the next
        read). A stale offset (> file size — the file was removed or
        truncated) silently re-baselines to EOF instead of re-dumping
        history. Malformed lines are skipped but still advance the offset.
        ``severity`` filters the returned records without affecting the
        offset. Never raises.
        """
        records: List[ActivityRecord] = []
        try:
            offset = max(0, int(offset or 0))
        except (TypeError, ValueError):
            offset = 0
        try:
            size = self.current_offset()
            if offset >= size:
                return [], size
            new_offset = offset
            parsed = 0
            with open(self.path, "rb") as f:
                f.seek(offset)
                while parsed < limit:
                    line = f.readline()
                    if not line or not line.endswith(b"\n"):
                        break  # EOF or partial tail write — don't consume
                    new_offset += len(line)
                    try:
                        rec = ActivityRecord.from_dict(
                            json.loads(line.decode("utf-8", "replace")))
                    except Exception:  # noqa: BLE001 — skip corrupt line
                        continue
                    parsed += 1
                    if severity is None or rec.severity == severity:
                        records.append(rec)
            return records, new_offset
        except Exception as e:  # noqa: BLE001
            logger.debug("activity read_since failed: %s", e)
            return [], offset


# --------------------------------------------------------------------------
# Watermarks — same shape as core.project_digest's, but keyed on byte offset.
# --------------------------------------------------------------------------

def load_offset(path) -> Optional[int]:
    """Saved digest offset, or ``None`` on first run (caller baselines)."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        return int(json.loads(p.read_text()).get("offset", 0))
    except Exception:  # noqa: BLE001
        return None


def save_offset(path, offset: int) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offset": int(offset)}))
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.debug("activity watermark save failed: %s", e)


# Per-consumer watermarks for /api/notifications (e.g. the Slack bot).
_consumers_lock = threading.Lock()


def load_consumer_offset(path, consumer: str) -> Optional[int]:
    """Saved offset for ``consumer``, or ``None`` when this consumer has
    never acked (caller baselines to EOF instead of replaying history)."""
    try:
        data = json.loads(Path(path).read_text())
        val = data.get(str(consumer))
        return None if val is None else int(val)
    except Exception:  # noqa: BLE001
        return None


def save_consumer_offset(path, consumer: str, offset: int) -> None:
    try:
        with _consumers_lock:
            p = Path(path)
            try:
                data = json.loads(p.read_text())
                if not isinstance(data, dict):
                    data = {}
            except Exception:  # noqa: BLE001
                data = {}
            # No-op acks skip the write (2026-08-01). Pollers (slack,
            # web-ui) used to re-ack an unchanged watermark every cycle,
            # rewriting this file every ~30s around the clock — pointless
            # fsync churn, and it kept the file's mtime permanently fresh,
            # which destroyed its documented diagnostic value ("stale
            # notify_consumers.json mtime = wedged consumer", 2026-07-13
            # postmortem). Identical value → nothing to persist.
            if data.get(str(consumer)) == int(offset):
                return
            data[str(consumer)] = int(offset)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.debug("consumer watermark save failed: %s", e)


# --------------------------------------------------------------------------
# Digest rendering
# --------------------------------------------------------------------------

def render_activity_digest(records: List[ActivityRecord], *,
                           max_items: int = 6,
                           exclude_phases=DIGEST_EXCLUDED_PHASES,
                           current_req_id: str = "",
                           severities=None) -> str:
    """Render unseen records as a short markdown header block. Empty string
    when nothing digest-worthy. Notify-severity items lead (stable order
    otherwise). ``current_req_id`` filters out records THIS turn authored
    (a notify_operator call mid-turn must not be echoed back as "while you
    were away" in the same reply). ``severities`` restricts the render to
    those severities (None = all) — the finalize banner passes
    ``(SEVERITY_NOTIFY,)`` so routine maintenance stays out of the chat
    (operator decision 2026-07-17: the info-severity refit lines read as
    noise; they remain reachable via ``introspect action='activity'``).
    Identical (phase, summary) repeats collapse into one "×N" line."""
    items = [r for r in (records or [])
             if r.summary and r.phase not in (exclude_phases or ())
             and (severities is None or r.severity in severities)
             and not (current_req_id
                      and r.meta.get("req_id") == current_req_id)]
    if not items:
        return ""
    items.sort(key=lambda r: 0 if r.severity == SEVERITY_NOTIFY else 1)
    # Collapse exact repeats (two REM cycles between turns rendered as two
    # identical bullets) while preserving first-seen order.
    counts: Dict[Tuple[str, str], int] = {}
    ordered: List[ActivityRecord] = []
    for r in items:
        key = (r.phase, r.summary)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            ordered.append(r)
    lines = ["**Background activity while you were away:**"]
    for r in ordered[:max_items]:
        label = _PHASE_LABELS.get(r.phase, r.phase)
        # Per-item clamp keeps the whole block comfortably under the
        # 1500-char leading-banner bound `_strip_leading_banners` peels —
        # a longer block would defeat the correction-fingerprint peel and
        # resurrect the stash/lookup mismatch fixed 2026-07-07.
        s = r.summary if len(r.summary) <= 140 else r.summary[:139] + "…"
        n = counts[(r.phase, r.summary)]
        lines.append(f"  - [{label}] {s}" + (f" (×{n})" if n > 1 else ""))
    extra = len(ordered) - max_items
    if extra > 0:
        lines.append(f"  - …and {extra} more")
    return "\n".join(lines)


def _age_str(ts: float, now: Optional[float] = None) -> str:
    delta = max(0.0, (now if now is not None else time.time()) - float(ts or 0))
    if delta < 90:
        return "just now"
    if delta < 5400:
        return f"{int(delta // 60)}m ago"
    if delta < 129600:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def render_activity_report(records: List[ActivityRecord], *,
                           hours: float = 24.0,
                           limit: int = 30,
                           now: Optional[float] = None) -> str:
    """On-demand full view of the ledger — ALL severities including the
    routine maintenance the finalize banner no longer auto-surfaces. The
    answer to "what did you do while I was away?" (via ``introspect
    action='activity'``). Newest first, timestamped, ×N-collapsed."""
    now_ts = now if now is not None else time.time()
    cutoff = now_ts - float(hours) * 3600.0
    items = [r for r in (records or []) if r.summary and r.ts >= cutoff]
    if not items:
        return (f"No background activity recorded in the last "
                f"{hours:g} hours.")
    items.sort(key=lambda r: r.ts, reverse=True)
    # Collapse exact repeats, keeping the NEWEST timestamp for the line.
    counts: Dict[Tuple[str, str], int] = {}
    ordered: List[ActivityRecord] = []
    for r in items:
        key = (r.phase, r.summary)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
            ordered.append(r)
    shown = ordered[:max(1, int(limit))]
    lines = [f"Background activity (last {hours:g}h, newest first):"]
    for r in shown:
        label = _PHASE_LABELS.get(r.phase, r.phase)
        mark = "❗" if r.severity == SEVERITY_NOTIFY else "·"
        s = r.summary if len(r.summary) <= 300 else r.summary[:299] + "…"
        n = counts[(r.phase, r.summary)]
        lines.append(f"  {mark} [{label}] {_age_str(r.ts, now_ts)} — {s}"
                     + (f" (×{n})" if n > 1 else ""))
    extra = len(ordered) - len(shown)
    if extra > 0:
        lines.append(f"  … and {extra} more distinct item(s) in the window")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Scheduled-turn capture helpers (called from main._run_proactive_task)
# --------------------------------------------------------------------------

_BANNER_HEADS = ("**While you were away**",
                 "**Background activity while you were away:**")
_BANNER_SEP = "\n\n---\n\n"


def summarize_turn_content(content, *, limit: int = 300) -> str:
    """Collapse a turn's final content into one digest-sized line.
    Strips any leading digest banners (a scheduled turn's reply could
    itself carry one) so a digest never quotes a digest."""
    text = str(content or "")
    # Peel stacked leading banner blocks — same separator contract as
    # agent._strip_leading_banners.
    for _ in range(4):
        if text.lstrip().startswith(_BANNER_HEADS) and _BANNER_SEP in text:
            text = text.split(_BANNER_SEP, 1)[1]
        else:
            break
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def record_scheduled_result(log: Optional[ActivityLog], *, job_id: str,
                            task_name: str = "", content=None,
                            ok: bool = True,
                            duration_s: Optional[float] = None) -> None:
    """Sink a scheduled turn's CONCLUSION into the ledger (severity=notify —
    a cron job's whole point is "tell me what you found"). Previously the
    final content was discarded and only pass/fail reached the workspace
    task ledger. Never raises."""
    if log is None:
        return
    try:
        name = (task_name or job_id or "task").strip()
        body = summarize_turn_content(content)
        if ok:
            summary = f"'{name}': {body or '(completed, no text output)'}"
        else:
            summary = f"'{name}' FAILED: {body or '(no detail)'}"
        meta = {"job_id": job_id, "ok": str(bool(ok))}
        if duration_s is not None:
            meta["duration_s"] = f"{float(duration_s):.1f}"
        log.record("scheduled_task", summary,
                   severity=SEVERITY_NOTIFY, **meta)
    except Exception as e:  # noqa: BLE001
        logger.debug("record_scheduled_result failed: %s", e)


def get_activity_log(context) -> Optional[ActivityLog]:
    """The context-attached ledger, or None. Accessor so call sites don't
    need to know the attribute name (and tests can monkeypatch one spot)."""
    return getattr(context, "activity_log", None)


__all__ = [
    "SEVERITY_INFO", "SEVERITY_NOTIFY",
    "DIGEST_EXCLUDED_PHASES", "INTERNAL_REQUEST_PREFIXES",
    "ActivityRecord", "ActivityLog",
    "is_internal_request",
    "load_offset", "save_offset",
    "load_consumer_offset", "save_consumer_offset",
    "render_activity_digest", "render_activity_report",
    "summarize_turn_content", "record_scheduled_result",
    "get_activity_log",
]
