"""Learning-health telemetry — read-only aggregation of the on-disk stores
the agent's background loops write to.

The 2026-07 loop-closing work shipped a large stack, each part carrying an
explicit "watch/keep-or-kill in ~2 weeks" criterion — but nothing surfaced
the data to make that call. This turns those pending watches into one
screen. Pure reads; never mutates; never raises (every store access is
defensive so a missing/corrupt file degrades to "n/a", not a crash).

What it reports today: the episode store, which registered background jobs
fired (mechanism and subsystem liveness), what the loops produced that anyone
used (loop yield), prompt-optimisation artifacts, verifier escalations and the
tool-call framing-leak watch. The lesson playbook's block went with the lesson
store (§4TD, 2026-11-06; §4PJ had removed the competence profile and the
confidence calibration): an old ``skills_playbook.json`` is not read.

Exposed as ``introspect action='learning'`` and ``scripts/learning_health.py``.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# The "a report must not die" handlers log here.
logger = logging.getLogger("GhostAgent")


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    # With a limit, a bounded deque keeps memory O(limit) instead of
    # materialising the whole file first (the activity ledger never
    # rotates). The limit also holds on a mid-read failure — the old
    # early return skipped the tail slice.
    buf = deque(maxlen=limit) if limit else []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    buf.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass  # partial read — return what we have (still bounded)
    return list(buf)


def collect_learning_health(memory_dir, args: Any = None) -> Dict[str, Any]:
    """Aggregate the learning stores under ``memory_dir`` (the
    system/memory directory). Returns a nested dict of sections; each is
    self-contained so a failed read leaves the others intact."""
    md = Path(memory_dir)
    report: Dict[str, Any] = {}

    # (§4TD: no lessons block — the lesson store was removed, and an old
    # `skills_playbook.json` under the memory dir is not read)

    # -- Adaptive threshold ------------------------------------------------
    at = _load_json(md / "adaptive_threshold.json", {}) or {}
    if isinstance(at, dict) and "threshold" in at:
        win = at.get("window") or []
        report["adaptive_threshold"] = {
            "learned_threshold": round(float(at.get("threshold", 0.0)), 4),
            "window_samples": len(win),
        }

    # -- Episodes ----------------------------------------------------------
    epi = _episode_stats(md / "episodic_memory.db")
    if epi is not None:
        report["episodes"] = epi

    # -- Background firing (dream / etc.) ---------------------------------
    report["activity"] = _activity_counts(md.parent / "autonomous_activity.jsonl")
    # Registry-driven, so a mechanism that never wrote is a ROW, not a silence.
    report["liveness"] = activity_liveness(
        md.parent / "autonomous_activity.jsonl")

    # -- Cognitive-subsystem wiring ---------------------------------------
    # Observability for the "wire-or-retire" question (improvement #5): which
    # producers run and which consumers are live vs gated OFF. Retiring a
    # producer is a product decision for the operator — this makes it an
    # informed one instead of guessing.
    report["cognitive_wiring"] = _cognitive_wiring(args, memory_dir)

    # -- GEPA prompt-optimization activation (§4F Phase 0) ----------------
    report["optim"] = _optim_activation(md.parent / "optim")

    # -- verifier escalation rate (§4F false-positive watch metric) -------
    report["verifier_escalation"] = _escalation_health(
        md.parent / "verifier" / "escalations.jsonl")

    # -- tool-call framing leak: RECURRENCE watch ------------------------
    report["framing_leak"] = _framing_leak_health(md.parent / "trajectories")

    # ── LOOP YIELD, machine-readable (§4CS item B, review round 2) ──
    # The axis existed only in the human RENDER, so nothing could alarm on
    # it programmatically and `--json` carried no yield key at all: the
    # §4CS failure mode ("the loop produced nothing and no instrument said
    # so") was fixed only for a human who reads the report. Same read-only
    # probes, same GHOST_HOME resolution as the rendered block.
    try:
        from .liveness import yield_all as _yield_all
        report["loop_yield"] = _yield_all(md.parent.parent)
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        report["loop_yield"] = {"error": f"{type(e).__name__}: {e}"}


    return report


def _optim_activation(optim_dir: Path) -> Dict[str, Any]:
    """Tuned-prompt artifacts on disk + in-process application counters.

    The pairing is the point: an artifact WITH zero in-process applies means
    the read-site is not firing (the write-only defect this loop already had
    once). Counters live in optim.loader and reset per process — "tuned file
    present, applied 0 since boot" is the actionable line."""
    out: Dict[str, Any] = {"artifacts": {}, "activation": {}}
    try:
        if optim_dir.is_dir():
            for p in sorted(optim_dir.glob("*.json")):
                if p.name.endswith(".candidate"):
                    continue  # staging files are not live artifacts
                data = _load_json(p, {}) or {}
                opt = data.get("optimized_instruction")
                out["artifacts"][p.stem] = {
                    "chars": len(opt) if isinstance(opt, str) else 0,
                    "valid": bool(isinstance(opt, str) and opt.strip()),
                }
    except Exception:
        pass
    try:
        from ..optim.loader import activation_stats
        out["activation"] = activation_stats()
    except Exception:
        pass
    return out


def _escalation_health(ledger_path: Path) -> Dict[str, Any]:
    """Verifier escalation outcomes from the durable ledger.

    This is the §4F false-positive-rate watch metric. It spent weeks being
    "recorded" into `VerifyResult.to_dict()`, which has no production caller,
    so the only way to read it was hand-run log archaeology — and a naive grep
    over-counts by 9 points because the OVERTURNED line is a WARNING mirrored
    to a second logger while "verdict stands" is INFO. The ledger writes both
    outcomes at the point escalation RESOLVES, which is what makes a rate
    computable at all.

    Reported per (route, kind) — claim-refute and code-refute are different
    populations and averaging them hides both (measured 2026-08-04 from the
    LOG, pre-ledger: claim refutes overturned 84%, all 7 code refutes upheld
    on replay — note a ``code/*`` arm appears HERE only once a code refute
    escalates after the ledger shipped 2026-08-04; its absence from the file
    is not evidence it never fires). ``unavailable`` stays in the
    denominator: the call was spent.
    """
    out: Dict[str, Any] = {"present": False, "arms": {}, "total": 0}
    rows = _load_jsonl(ledger_path, limit=5000)
    if not rows:
        # Distinguish "no ledger yet" from "ledger with nothing in it" — the
        # file is created on the FIRST escalation after a restart, so absence
        # right after a deploy is expected, not a broken instrument.
        out["reason"] = ("no ledger file" if not Path(ledger_path).exists()
                         else "ledger empty")
        return out
    out["present"] = True
    out["total"] = len(rows)
    # §4M (Lens D F6b): the counters were undated — the ledger begins
    # 2026-08-04 and pre-ledger history (log-only) is invisible, so an
    # undated "100% overturned" invited trending against a population the
    # file does not contain.
    out["first_ts"] = min((str(r.get("ts") or "") for r in rows), default="")
    out["last_ts"] = max((str(r.get("ts") or "") for r in rows), default="")
    for rec in rows:
        arm = f"{rec.get('route') or '?'}/{rec.get('kind') or '?'}"
        slot = out["arms"].setdefault(
            arm, {"overturned": 0, "upheld": 0, "withheld": 0,
                  "unavailable": 0, "downgraded": 0,
                  "replaced_uncertain": 0, "truncation_guard": 0,
                  "mechanically_upheld": 0, "mechanically_dismissed": 0,
                  "claim_binding": 0,          # §4IN/§4IP: the binder's validated verdict shipped over the incumbent's
                  "n": 0})
        outcome = str(rec.get("outcome") or "").strip()
        if outcome in slot:
            slot[outcome] += 1
        else:
            # ⚠ Vocabulary drift must be VISIBLE (§4L Lens-C MINOR-3):
            # the outcome vocabulary grew 5 new strings in one month,
            # and an unknown string used to bump only `n` — buckets
            # quietly stopped summing to n.
            slot["other"] = slot.get("other", 0) + 1
        slot["n"] += 1
    for arm, slot in out["arms"].items():
        decided = slot["overturned"] + slot["upheld"]
        slot["overturn_rate"] = (round(slot["overturned"] / decided, 3)
                                 if decided else None)
    return out


# The native-dialect fix (`QWEN_TOOL_PROMPT_NATIVE` split) deployed
# 2026-07-31 ~18:54 local. Every one of the 17 known framing leaks predates
# it; 161 trajectories recorded after it carry zero. So the watch is not
# "is the count above zero" — the corpus is append-only, so it never will be —
# it is "is the LATEST occurrence newer than the fix".
# §4M (Lens D F5): this constant is string-compared against trajectory
# `last_seen` values that are UTC ("...Z"). The fix landed ~18:54 LOCAL
# (UTC+3) = ~15:54Z — the old value 19:00 (local, unconverted) left a
# 3h06m blind window where a real recurrence rendered "all historical".
_FRAMING_FIX_TS = "2026-07-31T15:54"
# A post-fix leak is a REGRESSION only while it is this recent; older ones
# are reported as historical post-fix (§4UI, 2026-11-12).
_FRAMING_RECENT_DAYS = 7


def _framing_leak_health(traj_root: Path,
                         now: Optional[float] = None) -> Dict[str, Any]:
    """Recurrence watch for tool-call framing leaking into argument values.

    The journal recorded "repair fires ≈0 now, each one is news" — and then
    nothing WATCHED, so 17 historical cases were re-discovered by hand four
    days later and briefly mistaken for a live defect. This is the listener
    that note assumed existed.

    Verdict is keyed on the newest occurrence, not the count — and, since
    2026-11-12, on its AGE: one post-fix leak (a 10-12 probe write, the
    write itself landed) held "REGRESSION" on every overview for a month
    and the agent invented a "quirk" to explain it to the owner. A
    post-fix leak older than ``_FRAMING_RECENT_DAYS`` is ``post_fix`` but
    not ``regression``.
    """
    out: Dict[str, Any] = {"available": False}
    try:
        from ..utils.leaked_framing import scan_trajectories
        from ..distill.collector import TrajectoryCollector
        if not Path(traj_root).is_dir():
            out["reason"] = "no trajectory corpus"
            return out
        stats = scan_trajectories(
            # A corruption DETECTOR wants the whole corpus, probe turns included —
            # a dialect regression that appears first in a probe must still register
            # (§4FS review). include_probes=True is the documented exception.
            TrajectoryCollector(root=Path(traj_root),
                                session_id="reader").iter_trajectories(include_probes=True))
    except Exception as e:  # noqa: BLE001 — telemetry never breaks a turn
        out["reason"] = f"scan failed ({e})"
        return out
    import datetime as _dt
    _now = _dt.datetime.fromtimestamp(time.time() if now is None else now,
                                      tz=_dt.timezone.utc)
    # String-compared like _FRAMING_FIX_TS: both sides UTC ISO.
    _cutoff = (_now - _dt.timedelta(days=_FRAMING_RECENT_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%S")
    _post_fix = bool(stats["last_seen"]
                     and stats["last_seen"] > _FRAMING_FIX_TS)
    out.update({
        "available": True,
        "scanned": stats["scanned"],
        "calls": stats["calls"],
        "corrupt_calls": stats["corrupt_calls"],
        "by_tool": stats["by_tool"],
        "last_seen": stats["last_seen"],
        # The load-bearing field. A post-fix timestamp means the dual-dialect
        # corruption is BACK, which historically wasted whole turns.
        "post_fix": _post_fix,
        "regression": bool(_post_fix and stats["last_seen"] > _cutoff),
        "recent_days": _FRAMING_RECENT_DAYS,
    })
    return out


def _consumer_state(value: Optional[bool], unknown: str = "unknown") -> str:
    """ON / OFF / unknown for a wiring row. A row must never print a
    state it did not check (§4BM R2 MIN-I: the sibling rows printed OFF
    for a None that meant 'core.agent was not importable')."""
    if value is None:
        return unknown
    return "ON" if value else "OFF"


def _cognitive_wiring(args: Any = None, memory_dir=None) -> Dict[str, Any]:
    """Report each cognitive subsystem's producer/consumer wiring. Reads
    the gate flags from core.agent when importable; degrades to 'unknown'
    otherwise (e.g. a headless run where importing agent is undesirable).

    ``args`` (§4BM R1 MIN-2): the runtime args namespace, for consumers
    gated by CLI flags rather than module constants. None ⇒ a row reports
    "unknown", never a state it did not check."""
    flags: Dict[str, Any] = {}
    try:
        from . import agent as _ag
        flags = {
            "selfhood_prefix_consumer": getattr(_ag, "_SELFHOOD_PREFIX_ENABLED", None),
        }
    except Exception:
        flags = {}
    return {
        # producer runs every turn; consumer (wake-up prefix) gated by flag,
        # but ALSO read by `introspect` (summary/narrative/recent) — so not
        # write-only. Store is compaction-bounded (2MB/512KB).
        "selfhood": {
            "producer": "per-turn (capture_turn/record_outcome)",
            "prefix_consumer_enabled": flags.get("selfhood_prefix_consumer"),
            "also_read_by": "introspect",
            "store_bounded": True,
        },
        # RETIRED 2026-07-27 (operator-approved): module + export removed
        # after the INERT flag held — no production or offline caller
        # anywhere in the tree. Entry kept so the telemetry documents the
        # decision instead of the subsystem silently vanishing.
        "self_consistency": {
            "status": "RETIRED 2026-07-27 — module removed (was INERT, "
                      "no caller)",
        },
    }


def _episode_stats(db_path: Path) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            def _c(sql):
                return conn.execute(sql).fetchone()[0]
            total = _c("SELECT COUNT(*) FROM episodes")
            with_ctx = _c("SELECT COUNT(*) FROM episodes WHERE context != ''")
            with_cluster = _c("SELECT COUNT(*) FROM episodes WHERE cluster_id != ''")
            success = _c("SELECT COUNT(*) FROM episodes WHERE outcome_success = 1")
            consolidated = _c("SELECT COUNT(*) FROM episodes WHERE consolidated = 1")
            return {
                "total": total,
                "with_context": with_ctx,
                "with_cluster": with_cluster,
                "context_coverage_pct": round(100 * with_ctx / total, 1) if total else 0.0,
                "success": success,
                "consolidated": consolidated,
                "pending_consolidation": total - consolidated,
            }
        finally:
            conn.close()
    except Exception:
        return None


def _activity_counts(ledger_path: Path, *, window_hours: float = 168.0) -> Dict[str, int]:
    """Count ledger records by phase over a recent window.

    ``ActivityRecord.to_dict`` serializes the kind as ``phase`` — keying on
    kind/type/category matched NOTHING, so this always returned {} and the
    "BACKGROUND ACTIVITY" section silently never rendered against a ledger
    with 1500+ real records (2026-07-27: indistinguishable from "nothing
    ran", in the very report meant to answer whether the loops fire).
    """
    counts: Dict[str, int] = {}
    cutoff = time.time() - max(0.0, float(window_hours)) * 3600.0
    for rec in _load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT):
        try:
            ts = float(rec.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if not ts:
            # Unparseable/zero timestamp: SKIP, don't count. `if ts and
            # ts < cutoff` kept these rows in EVERY window — a single
            # garbage-ts row would keep a dead PERIODIC phase looking
            # alive in the liveness table forever, and that table's whole
            # job is telling benign zeros from fatal ones.
            continue
        if ts < cutoff:
            continue
        kind = str(rec.get("phase") or rec.get("kind") or rec.get("type")
                   or rec.get("category") or "").strip()
        try:                                  # §4NM: dream → maintenance
            from .autonomous_activity import canonical_phase as _cp
            kind = _cp(kind)
        except Exception:  # noqa: BLE001
            pass
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


# The ledger is append-only and already past 2600 records. A tail-read cap of
# 2000 does not bite at today's ~565 records/week, but it is a SILENT cap: the
# day it bites, phases quietly lose counts and the report keeps its confident
# tone. Raised, and `activity_liveness` reports when the cap was actually hit.
_ACTIVITY_READ_LIMIT = 20000


def activity_liveness(ledger_path: Path,
                      *, alarm_window_hours: float = 24.0,
                      context_window_hours: float = 168.0) -> Dict[str, Any]:
    """Did every registered background mechanism actually fire?

    ⚠ WHY THIS IS NOT `_activity_counts`. That function enumerates phases
    OBSERVATIONALLY — it can only report slugs that are already in the ledger.
    A mechanism that stopped writing, or never wrote, produces no key and is
    therefore invisible, which is indistinguishable from a mechanism that does
    not exist. Measured 2026-08-10: 15 phases instrumented in source, 7 in the
    ledger, and the report rendered seven green rows with no hint of the
    other eight. This function iterates the REGISTRY instead, so absence is a
    row rather than a silence.

    Two windows on purpose. The alarm fires on 24h, but a phase that last ran
    three days ago reads very differently from one dark for a week, and
    collapsing that to a single boolean throws away the only evidence that
    distinguishes "slow" from "dead".
    """
    from .autonomous_activity import (
        EXPECT_PERIODIC, PHASE_EXPECTATION, phase_expectation, RETIRED_PHASES,
    )

    # ⚠ MISSING LEDGER != IDLE AGENT (caught in fresh-eye review, 2026-08-10).
    #
    # Without this the two render IDENTICALLY: every registered phase at zero,
    # `agent_silent` true, no alarms — the same screen for "the agent was
    # quiet" and "this function is pointed at a path that does not exist".
    #
    # It is the SAME defect this file's sibling fix removed from the response
    # cache hours earlier (an empty cache dir reading exactly like a stale
    # one), reproduced in the code written to detect defects of that shape.
    # Absence of the instrument is a louder fact than absence of activity, and
    # it must never be inferable only by noticing that everything is zero.
    ledger_missing = not Path(ledger_path).exists()

    recent = _activity_counts(ledger_path, window_hours=alarm_window_hours)
    context = _activity_counts(ledger_path, window_hours=context_window_hours)
    n_read = len(_load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT))

    # ⚠ AGENT-DOWN GUARD. If NOTHING fired, the agent was off or idle — that
    # is ONE fact, not five independent dead loops. Emitting five alarms for
    # a stopped agent is how an operator learns to scroll past this section,
    # which would rebuild by hand exactly the blindness it exists to remove.
    silent_everywhere = sum(recent.values()) == 0

    # The attempt heartbeat, read from beside the ledger. Failure here is
    # not an alarm condition: `read_attempts` returns {} and every phase
    # falls back to the outcome-only rule.
    try:
        from .autonomous_activity import (
            attempt_suppresses_alarm as _suppresses,
            read_attempts as _read_attempts)
        attempts = _read_attempts(ledger_path) if ledger_path else {}
    except Exception:  # noqa: BLE001
        attempts = {}
        def _suppresses(_e, _n, _w):  # noqa: E306
            return False
    now_ts = time.time()
    alarm_window_s = float(alarm_window_hours) * 3600.0

    rows = []
    # ⚠ `set(attempts)` is in the union. Without it a phase that has ONLY
    # ever stamped a heartbeat — a new loop wired to the heartbeat before it
    # ever produces an outcome — appears nowhere at all, which is the exact
    # invisibility this registry exists to remove, re-entered through the new
    # door.
    for phase in sorted(set(PHASE_EXPECTATION) | set(context) | set(recent)
                        | set(attempts)):
        if phase in RETIRED_PHASES:
            continue                       # §4PJ: its producer is gone; old rows are history
        exp = phase_expectation(phase)
        n24, n168 = recent.get(phase, 0), context.get(phase, 0)
        rows.append({
            "phase": phase,
            "n_alarm_window": n24,
            "n_context_window": n168,
            "expectation": exp,
            # Only PERIODIC can alarm: the others are at zero by design, and
            # a monitor that cries on a benign zero gets ignored on a real one.
            # ⚠ A RECENT ATTEMPT SUPPRESSES THE ALARM, and is reported.
            #
            # The ledger records OUTCOMES, so "produced nothing in 24h" and
            # "did not run in 24h" were the same row. Measured 2026-08-30:
            # this printed `✗ DEAD router_train 0/24h 0/7d` while the loop
            # had run 30 minutes earlier and correctly declined an 89%
            # unchanged corpus. An alarm that fires on a healthy loop is how
            # the column gets learned-ignored — and then misses a real one.
            #
            # The heartbeat is a SEPARATE signal (`idle_attempts.json`), so
            # this is not the ledger vouching for itself. Absent heartbeat →
            # unchanged behaviour: an unreadable or never-written heartbeat
            # must degrade to the old alarm, never to a false GREEN.
            # ⚠ ASK THE PREDICATE, do not re-derive the comparison here.
            # The first version inlined `(now - ts) <= window`, which is
            # one-sided: any FUTURE timestamp — an NTP step back, a
            # hand-edited file, a JSON `Infinity` (json.loads accepts the
            # literal) — satisfied it forever and turned a real DEAD alarm
            # permanently green. This file already hardens the LEDGER
            # timestamps against exactly that ~100 lines above; the new
            # signal, the only one that can SILENCE an alarm, did not
            # inherit it. One predicate, one place.
            "attempted_at": (attempts.get(phase) or {}).get("ts"),
            "attempted_result": (attempts.get(phase) or {}).get("result"),
            "attempted_age_s": (
                None if not attempts.get(phase)
                else now_ts - float(attempts[phase]["ts"])),
            "attempted_recently": _suppresses(
                attempts.get(phase), now_ts, alarm_window_s),
            "alarm": bool(exp == EXPECT_PERIODIC and n24 == 0
                          and not silent_everywhere
                          and not _suppresses(attempts.get(phase), now_ts,
                                              alarm_window_s)),
            "registered": phase in PHASE_EXPECTATION,
        })
    # Zeros FIRST. The old renderer sorted by count descending and cut to the
    # top 8, so the dead loop — the one thing worth looking at — was
    # structurally the first row dropped.
    rows.sort(key=lambda r: (not r["alarm"], r["n_alarm_window"], r["phase"]))
    return {
        "rows": rows,
        "alarms": [r["phase"] for r in rows if r["alarm"]],
        "agent_silent": silent_everywhere,
        # Distinct from `agent_silent`: no instrument, vs an instrument
        # reporting nothing. The renderer must never merge these.
        "ledger_missing": ledger_missing,
        "ledger_path": str(ledger_path),
        "alarm_window_hours": alarm_window_hours,
        "context_window_hours": context_window_hours,
        # Loud rather than silent if the tail-read ever truncates the window.
        "read_truncated": n_read >= _ACTIVITY_READ_LIMIT,
    }


def render_framing_leak(fl: Dict[str, Any]) -> List[str]:
    """The FRAMING LEAK lines of the report (each starts with a newline).
    §4UI: a regression is ONE ⚠ line carrying its own detail."""
    lines: List[str] = []
    if fl.get("available"):
        n, tot = fl["corrupt_calls"], fl["calls"]
        if fl.get("regression"):
            # ONE line (§4UI): the overview keeps only ⚠ lines, and a
            # heading whose detail sat on the next line reached the model
            # as a bare "REGRESSION:" it then explained by invention.
            lines.append(
                f"\n⚠ TOOL-CALL FRAMING LEAK — REGRESSION: {n}/{tot} calls "
                f"carry leaked framing; newest {fl['last_seen'][:16]} is "
                f"AFTER the 2026-07-31 fix and within "
                f"{fl.get('recent_days', _FRAMING_RECENT_DAYS)} d — the "
                f"dual-dialect corruption is back {fl.get('by_tool')}")
        elif fl.get("post_fix"):
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: {n}/{tot} calls; newest "
                f"{fl['last_seen'][:16]} is after the 2026-07-31 fix but "
                f"older than {fl.get('recent_days', _FRAMING_RECENT_DAYS)} d "
                f"{fl.get('by_tool')} — historical, no action unless it recurs")
        elif n and not fl.get("last_seen"):
            # Undated leaks cannot be placed against the fix — say so
            # instead of the "pre-fix" all-clear (§4UI review MINOR-4).
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: {n}/{tot} calls, none dated — "
                f"cannot tell pre- from post-fix {fl.get('by_tool')}")
        elif n:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: {n}/{tot} calls, all historical "
                f"(newest {fl['last_seen'][:16]}, pre-fix) — no action")
        else:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: none in {tot} calls "
                f"across {fl['scanned']} trajectories")
    elif fl:
        lines.append(f"\nTOOL-CALL FRAMING LEAK: {fl.get('reason', 'unknown')}")
    return lines


def render_learning_health(memory_dir, args: Any = None) -> str:
    """Human-readable one-screen learning-health report.

    ``args`` threads the runtime namespace to the wiring section so
    flag-gated consumers report their real state (§4BM R1 MIN-2).
    Passed POSITIONALLY only when present (§4BM R2 CRIT-A): nine existing
    tests monkeypatch ``collect_learning_health`` with one-arg lambdas,
    and an unconditional 2-arg call broke every one of them — including
    the MISSING-INSTRUMENT and CONSUMER-DEAD guards, i.e. the other
    broken-instrument tests."""
    r = (collect_learning_health(memory_dir) if args is None
         else collect_learning_health(memory_dir, args))
    lines: List[str] = ["### LEARNING HEALTH"]

    epi = r.get("episodes")
    if epi:
        lines.append(
            f"\nEPISODES: {epi['total']} "
            f"({epi['pending_consolidation']} pending consolidation, "
            f"{epi['success']} success)")
        # §4M (Lens D F6c): the old single line said "context/cluster"
        # with a context-only numerator — the two counts differ live.
        lines.append(
            f"  context coverage: {epi['with_context']}/{epi['total']} "
            f"({epi['context_coverage_pct']}%), cluster coverage: "
            f"{epi.get('with_cluster', '?')}/{epi['total']} — populated only "
            f"on episodes recorded AFTER the 2026-07-26 field-population fix")

    lv = r.get("liveness")
    if lv and lv.get("rows"):
        aw = int(lv["alarm_window_hours"])
        cw = int(lv["context_window_hours"] / 24)
        lines.append(f"\nMECHANISM LIVENESS (registry-driven — every "
                     f"registered loop appears, firing or not):")
        if lv.get("ledger_missing"):
            # ⚠ Checked BEFORE agent_silent: a missing ledger is also silent,
            # so reporting idleness first would describe the symptom and hide
            # the cause. This whole table is derived from that file — if it is
            # not there, every zero below means nothing at all.
            lines.append(
                f"  ⚠⚠ LEDGER NOT FOUND at {lv.get('ledger_path')} — the "
                f"counts below are not evidence of anything. This is a "
                f"MISSING INSTRUMENT, not an idle agent.")
        elif lv.get("agent_silent"):
            lines.append(
                f"  ⚠ NO activity of ANY kind in {aw}h — the agent was off or "
                f"fully idle. That is ONE fact, so per-loop alarms are "
                f"withheld; they would all be this.")
        if lv.get("read_truncated"):
            lines.append("  ⚠ ledger tail-read hit its cap — counts below "
                         "UNDERSTATE the window.")
        for row in lv["rows"]:
            mark = "  ✗ DEAD " if row["alarm"] else "    "
            note = "" if row["registered"] else "  ← UNREGISTERED phase"
            lines.append(
                f"{mark}{row['phase']:<24} {row['n_alarm_window']:>4} /{aw}h "
                f"{row['n_context_window']:>5} /{cw}d   "
                f"{row['expectation']}{note}")
            # ⚠ SAY IT, don't just withhold the alarm. A zero with the alarm
            # quietly suppressed is indistinguishable from a zero nobody
            # looked at — the operator has no way to tell "ran and declined"
            # from "we stopped checking". The heartbeat is positive evidence,
            # so it gets a line.
            from .autonomous_activity import (
                EXPECT_PERIODIC as _EXPECT_PERIODIC_NAME)
            # Only a row that COULD have alarmed needs the explanation, and
            # the age comes from the row — a second `time.time()` here can
            # print an age that contradicts the `attempted_recently` it is
            # explaining.
            if (row["expectation"] == _EXPECT_PERIODIC_NAME
                    and row["n_alarm_window"] == 0
                    and row.get("attempted_recently")):
                _age_h = float(row["attempted_age_s"] or 0.0) / 3600.0
                lines.append(
                    f"        └ RAN {_age_h:.1f}h ago and produced nothing — "
                    f"this loop records an OUTCOME, and declining (e.g. an "
                    f"unchanged corpus) is a healthy outcome-free run. Not "
                    f"dead; the heartbeat is a separate signal from the "
                    f"ledger.")
                if int(row.get("n_context_window") or 0) == 0:
                    # §4MI: a decline every 3 h for 17 days silenced the
                    # alarm while the router checkpoint aged to 409 h —
                    # the heartbeat vouched for a loop whose gate never
                    # opened. Not an alarm (the gate may be right), but
                    # said.
                    lines.append(
                        f"        └ ⚠ and no outcome in 7 d: a loop that only ever "
                        f"declines is one whose input never changes — check its gate")
        if lv.get("alarms"):
            lines.append(
                f"  ✗ {len(lv['alarms'])} PERIODIC loop(s) silent for {aw}h: "
                + ", ".join(lv["alarms"]))
        else:
            lines.append("  all periodic loops fired within the alarm window")
        # ⚠ Only PERIODIC alarms. on_output/on_demand/gated phases sit at zero
        # by design, and a monitor that cries on a benign zero is one the
        # operator learns to scroll past — rebuilding by hand the blindness
        # this section exists to remove.

    # ── SUBSYSTEM LIVENESS (probe-per-mechanism) ─────────────────────────
    # The section above covers phases that write to autonomous_activity.jsonl.
    # This one covers everything that does NOT — the gap that let metacog
    # arbitration sit dead and invisible (0 firings against 118 opportunities)
    # until it was found by grepping the log by hand.
    try:
        from .liveness import render as _liveness_render
        # ⚠ GHOST_HOME, not its `system/` child. `memory_dir` is
        # $GHOST_HOME/system/memory, so the home is TWO levels up — the probes
        # resolve "system/<store>/..." relative to it. Passing `.parent`
        # made all 8 probes report NO_SOURCE. That the view said NO_SOURCE
        # rather than quietly reporting zeros is the third state earning its
        # keep on its first real outing.
        lines.append("\n" + _liveness_render(Path(memory_dir).parent.parent))
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        lines.append(f"\nSUBSYSTEM LIVENESS: unavailable ({type(e).__name__})")

    # ── LOOP YIELD (§4CS) ────────────────────────────────────────────────
    # The SECOND axis, and the one the macro loop had to be found by hand
    # to expose. Liveness above asks "did this mechanism RUN?"; this asks
    # "did it produce anything anyone CONSUMED?". Dream fired on schedule
    # for six weeks and minted 25 macros that were invoked ZERO times, and
    # every liveness probe would have read FIRED throughout. (The macro
    # engine and its row are removed, §4TA 2026-11-05; the axis stays for
    # the loops that are left.)
    #
    # Extended here rather than given its own introspect action on purpose:
    # a parallel surface is one more thing an operator has to remember to
    # look at, and this report already carries the probe-per-mechanism
    # section it belongs beside.
    try:
        from .liveness import render_yield as _yield_render
        # Same two-levels-up resolution as the liveness block above:
        # `memory_dir` is $GHOST_HOME/system/memory.
        lines.append("\n" + _yield_render(Path(memory_dir).parent.parent))
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        lines.append(f"\nLOOP YIELD: unavailable ({type(e).__name__})")

    act = r.get("activity")
    if act:
        # NO top-N CUT. It used to sort by count descending and slice [:8], so
        # a phase at zero — the only kind worth acting on — sorted last and was
        # structurally the first row dropped.
        top = sorted(act.items(), key=lambda kv: -kv[1])
        lines.append("\nBACKGROUND ACTIVITY (recent ledger): "
                     + ", ".join(f"{k}={v}" for k, v in top))
        # These counts are NOT comparable across phases, and reading them as
        # a workload budget is a live trap: reflection appeared to run ~10x
        # less than dream, which looks like a starved loop but is a recording
        # artifact. Reflection only writes a ledger event when it produced
        # OUTCOMES, and it deliberately skips ticks whose trajectory corpus
        # is unchanged since an all-duplicate pass. §4CB R1/R2: dream too now
        # ledgers ONLY cycles that did consolidation work — skip AND error
        # cycles mint nothing (so the zero-rows liveness alarm can fire on a
        # permanently failing dream). A low count here means "little new
        # material", not "under-scheduled" — check the cooldowns instead.
        lines.append(
            "  (per-phase recording policies differ — reflection logs only "
            "outcome-producing runs and skips unchanged-corpus ticks; "
            "maintenance logs only passes that wrote something (digests, "
            "manifests, store care), so a zero can mean quiet OR erroring — "
            "do NOT read these as a workload budget)")

    cw = r.get("cognitive_wiring")
    if cw:
        lines.append("\nCOGNITIVE WIRING (producer → consumer status):")
        sf = cw.get("selfhood", {})
        lines.append(
            f"  selfhood: writes every turn; wake-up prefix "
            + _consumer_state(sf.get("prefix_consumer_enabled"),
                              "unknown (gate not read)")
            + ", also read by introspect (bounded store)")
        sc = cw.get("self_consistency", {})
        if sc.get("status"):
            lines.append(f"  self_consistency: {sc['status']}")

    op = r.get("optim")
    if op and (op.get("artifacts") or op.get("activation")):
        lines.append("\nPROMPT OPTIMIZATION (GEPA artifacts → read-site activation):")
        arts = op.get("artifacts") or {}
        acts = op.get("activation") or {}
        for name in sorted(set(arts) | set(acts)):
            a = arts.get(name)
            c = acts.get(name) or {}
            applied = c.get("applied", 0)
            fallback = c.get("fallback", 0)
            if a and a.get("valid"):
                state = f"tuned ({a['chars']} chars)"
                # ⚠ The applies counter is PER-PROCESS (§4L Lens-C
                # MINOR-2): a headless `scripts/learning_health.py` run
                # has trivially 0 for every artifact and used to render
                # the alarm regardless of live-agent truth. Only the
                # in-process view (introspect action='learning') can
                # distinguish "read-site never ran" from "wrong
                # process"; the script now says which one it is.
                import sys as _sys
                _headless = not _sys.argv[0].endswith(("main.py", "-m"))
                if applied == 0 and _headless:
                    flag = ("  (0 applies IN THIS PROCESS — per-process "
                            "counter; run introspect action='learning' "
                            "in the agent for live truth)")
                elif applied == 0:
                    flag = "  ⚠ tuned but 0 applies since boot"
                else:
                    flag = ""
            elif a:
                state = "artifact INVALID"
                flag = ""
            else:
                state = "no artifact (baseline)"
                flag = ""
            lines.append(
                f"  {name}: {state}; applied {applied} / fallback {fallback}{flag}")

    esc = r.get("verifier_escalation") or {}
    if esc.get("present"):
        _span = ""
        if esc.get("first_ts"):
            _span = (f" — ledger span {str(esc['first_ts'])[:10]} → "
                     f"{str(esc['last_ts'])[:10]}; pre-ledger history is "
                     f"log-only and NOT counted here")
        lines.append(f"\nVERIFIER ESCALATION (§4F false-positive watch){_span}:")
        for arm in sorted(esc.get("arms") or {}):
            s = esc["arms"][arm]
            rate = s.get("overturn_rate")
            rate_s = "n/a" if rate is None else f"{rate:.0%} overturned"
            extra = ""
            if s.get("withheld"):
                extra += f", {s['withheld']} withheld"
            if s.get("unavailable"):
                extra += f", {s['unavailable']} unavailable (call spent)"
            if s.get("downgraded"):
                extra += (f", {s['downgraded']} downgraded "
                          f"(gloss-tier, no call)")
            # Every category renders or the arithmetic doesn't add up to
            # `n` and the mechanical layer reads as "never fired"
            # (round-2 review). ⚠ overturn_rate is conditioned on
            # SURVIVING the mechanical layer since 2026-08-07, and
            # strong-UNCERTAIN moved from `overturned` to
            # `replaced_uncertain` — do not trend the rate across that
            # date without re-baselining.
            if s.get("mechanically_dismissed"):
                extra += (f", {s['mechanically_dismissed']} mech-dismissed "
                          f"(no call)")
            if s.get("mechanically_upheld"):
                extra += (f", {s['mechanically_upheld']} mech-upheld "
                          f"(protected, no call)")
            if s.get("truncation_guard"):
                extra += f", {s['truncation_guard']} truncation-guarded"
            if s.get("replaced_uncertain"):
                extra += (f", {s['replaced_uncertain']} replaced-uncertain "
                          f"(strong judge punted)")
            if s.get("claim_binding"):
                extra += (f", {s['claim_binding']} claim-binding "
                          f"(validated quotes decided, no appeal)")
            if s.get("other"):
                # §4L R2 NEW-6: the counter existed but the renderer
                # omitted it — vocabulary drift invisible again, one
                # layer up.
                extra += (f", {s['other']} OTHER (unknown outcome "
                          f"strings — vocabulary drift, investigate)")
            lines.append(f"  {arm}: {s['n']} escalations — {rate_s} "
                         f"({s['overturned']}/{s['overturned'] + s['upheld']} "
                         f"decided){extra}")
    elif esc:
        # An absent ledger is only news once the agent has been up a while;
        # say WHICH silence it is rather than printing nothing.
        lines.append(f"\nVERIFIER ESCALATION: {esc.get('reason', 'unknown')} "
                     "— the file appears on the first escalation after a boot")

    lines.extend(render_framing_leak(r.get("framing_leak") or {}))

    return "\n".join(lines)
