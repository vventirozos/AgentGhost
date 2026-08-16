"""introspect tool — read-only introspection over the agent's selfhood.

The selfhood wake-up prefix is already spliced into every system prompt, so
a casual "tell me about yourself" question is grounded in the diary by
default. This tool is the explicit handle the agent reaches for when it
wants a richer, deterministic snapshot rather than relying on whatever
the prefix happened to surface — useful for introspective questions
("what have you been working on?", "what do you remember about X?") and
for evaluation / probing scripts.

All actions are read-only. The writable counterpart is ``self_state``
(``tools/self_state.py``); the two are deliberately split so a single
tool description does not conflate "introspect myself" with "author my
forward-looking continuity slot".

Routes through ``context.self_model`` so the SelfModel facade owns the
read API and a disabled selfhood ("--no-self-model" / "--no-memory")
degrades to a clear message instead of crashing.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"summary", "stats", "narrative", "recent",
                            "recall", "activity", "learning", "experiments"})

_DEFAULT_RECENT = 5
_DEFAULT_RECALL = 5
_MAX_LIMIT = 25
_SUMMARY_RECENT_N = 5

# action='activity' window/size defaults. Separate caps from the selfhood
# actions: the ledger view is line-per-event and 30 lines is already a
# screenful.
_DEFAULT_ACTIVITY_HOURS = 24.0
_MAX_ACTIVITY_HOURS = 24.0 * 14
_DEFAULT_ACTIVITY_LIMIT = 30
_MAX_ACTIVITY_LIMIT = 100


def _exp_age(ts_iso) -> str:
    """Compact age ("3.2h ago") from an Experience's ISO-Z timestamp.
    Empty string when unparseable — a malformed stamp must not break a
    render. Rides ``autonomous_activity._age_str`` so the selfhood views
    and the activity report speak the same time language."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts_iso or "").rstrip("Z"))
        epoch = dt.replace(tzinfo=timezone.utc).timestamp()
        from ..core.autonomous_activity import _age_str
        return _age_str(epoch)
    except Exception:  # noqa: BLE001
        return ""


def _format_experience(exp) -> str:
    line = f"  - {exp.summary}"
    outcome = getattr(exp, "outcome", "") or ""
    if outcome and outcome != "unknown":
        line += f" [{outcome}]"
    age = _exp_age(getattr(exp, "timestamp", ""))
    if age:
        line += f" ({age})"
    return line


def _clamp_limit(value, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, _MAX_LIMIT)


def _render_stats(stats: dict) -> str:
    if not stats:
        return "No self-state on file."
    if stats.get("enabled") is False:
        return "Selfhood is disabled."
    lines: List[str] = []
    lines.append(f"Experiences on file: {stats.get('experience_count', 0)}")
    lines.append(f"Open questions: {stats.get('open_questions', 0)}")
    lines.append(f"Unfinished threads: {stats.get('unfinished_threads', 0)}")
    mood = stats.get("last_mood") or ""
    if mood:
        lines.append(f"Last noted mood: {mood}")
    last = stats.get("last_session_at") or ""
    if last:
        lines.append(f"Last active: {last}")
    lines.append(
        f"Running narrative: {'present' if stats.get('narrative_present') else 'none yet'}"
    )
    clusters = stats.get("clusters") or {}
    if clusters:
        ranked = sorted(clusters.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k}={v}" for k, v in ranked[:5])
        lines.append(f"Topic clusters: {top}")
    # The values layer is "behaviour-shaping" by design — it must not be
    # invisible to introspection (it was, until 2026-07-27).
    pc = stats.get("principle_count")
    if pc is not None:
        lines.append(f"Operating principles: {pc}")
    return "\n".join(lines)


def _render_summary(self_model) -> str:
    stats = self_model.stats()
    parts: List[str] = []
    parts.append("Who I am — a snapshot of my self-state:")
    parts.append(_render_stats(stats))

    narrative = ""
    if self_model.narrative is not None:
        narrative = (self_model.narrative.latest() or "").strip()
    if narrative:
        parts.append("\nMy running first-person diary:")
        parts.append(narrative)

    # Operating principles — the normative substrate. "Tell me about
    # yourself" is exactly where these belong.
    principles = ""
    try:
        principles = (self_model.principles_text() or "").strip()
    except Exception as e:  # noqa: BLE001 — summary must render without them
        logger.debug("introspect principles render skipped: %s", e)
    if principles:
        parts.append("\nMy operating principles:")
        parts.append(principles)

    recent = []
    if self_model.autobio is not None:
        try:
            recent = self_model.autobio.recent(limit=_SUMMARY_RECENT_N)
        except Exception as e:  # noqa: BLE001 — read path is secondary
            logger.debug("introspect recent() failed: %s", e)
            recent = []
    if recent:
        parts.append("\nRecent things I remember doing:")
        for exp in recent:
            parts.append(_format_experience(exp))

    return "\n".join(parts).rstrip()


def _render_recent(self_model, limit: int) -> str:
    if self_model.autobio is None:
        return "No autobiographical log on file."
    try:
        recent = self_model.autobio.recent(limit=limit)
    except Exception as e:  # noqa: BLE001
        logger.warning("introspect recent failed: %s", e)
        return f"Could not read the autobiographical log: {type(e).__name__}: {e}"
    if not recent:
        return "I have no experiences on file yet."
    lines = [f"My {len(recent)} most recent experiences:"]
    for exp in recent:
        lines.append(_format_experience(exp))
    return "\n".join(lines)


def _render_recall(self_model, query: str, limit: int) -> str:
    matches = self_model.recall_relevant(query, limit=limit) or []
    if not matches:
        return f"Nothing in my past matches '{query}'."
    lines = [f"What I remember about '{query}':"]
    for exp in matches:
        lines.append(_format_experience(exp))
    return "\n".join(lines)


# Bytes of ledger TAIL scanned for the activity report. The ledger never
# rotates, so a from-byte-0 replay grows without bound; the report only ever
# renders a recent window, so scanning the tail is both sufficient and O(1).
_ACTIVITY_TAIL_BYTES = 512 * 1024
# Records kept while scanning (the render caps output well below this).
_ACTIVITY_SCAN_KEEP = 1000


def _read_activity_tail(log):
    """Return ``(records, truncated, failed)`` — the most RECENT records
    in the ledger, oldest-first.

    ``read_since`` caps each call at ``limit=200`` records, so the previous
    single ``read_since(0)`` returned the 200 OLDEST lines and the report
    went permanently blind once the never-rotated ledger passed 200 entries
    — live, that meant every 'what did you do while I was away' answer was
    built from records that stopped on 2026-07-13 (verified 2026-07-27).
    Seek near the end and drain forward instead.

    ``truncated`` is True when the scan did NOT start at byte 0 (or the
    record cap trimmed) — the caller uses it to annotate a window the tail
    doesn't fully cover instead of silently under-reporting. ``failed`` is
    True when the read died mid-scan; a failed read must render as a read
    error, never as "no background activity" (a broken instrument reading
    as a calm 'nothing ran' is the exact defect class this subsystem keeps
    growing).
    """
    records = []
    truncated = False
    try:
        size = log.current_offset()
        off = max(0, size - _ACTIVITY_TAIL_BYTES)
        truncated = off > 0
        # A tail seek can land mid-line; that partial line fails to parse and
        # is skipped by read_since (one boundary record lost, by design).
        for _ in range(1000):  # hard stop — never spin on a pathological file
            chunk, new_off = log.read_since(off)
            if chunk:
                records.extend(chunk)
                if len(records) > _ACTIVITY_SCAN_KEEP:
                    records = records[-_ACTIVITY_SCAN_KEEP:]
                    truncated = True
            if new_off <= off:
                break
            off = new_off
    except Exception as e:  # noqa: BLE001 — a report must never break the tool
        logger.debug("activity tail read failed: %s: %s", type(e).__name__, e)
        return records, truncated, True
    return records, truncated, False


def _render_activity(context, *, hours=None, limit=None) -> str:
    """Full-ledger view (ALL severities) over a bounded recent window.
    Reads the ledger TAIL (see ``_read_activity_tail``); the finalize
    banner's watermark is deliberately NOT consumed (this is a read, not
    an ack)."""
    from ..core.autonomous_activity import (
        get_activity_log, render_activity_report,
    )
    log = get_activity_log(context)
    if log is None:
        return ("Background-activity ledger is not attached in this "
                "session — nothing to report.")
    try:
        h = float(hours) if hours else _DEFAULT_ACTIVITY_HOURS
    except (TypeError, ValueError):
        h = _DEFAULT_ACTIVITY_HOURS
    h = max(0.25, min(h, _MAX_ACTIVITY_HOURS))
    try:
        n = int(limit) if limit else _DEFAULT_ACTIVITY_LIMIT
    except (TypeError, ValueError):
        n = _DEFAULT_ACTIVITY_LIMIT
    n = max(1, min(n, _MAX_ACTIVITY_LIMIT))
    records, truncated, failed = _read_activity_tail(log)
    if failed and not records:
        # A dead read must NOT render as a calm "no background activity".
        return ("Could not read the background-activity ledger — the tail "
                "scan failed (see the debug log). This is a read error, "
                "not \"nothing ran\".")
    pretty_log("Introspect", f"activity report requested ({h:g}h window)",
               icon=Icons.BRAIN_SUM)
    report = render_activity_report(records, hours=h, limit=n)
    notes = []
    if failed:
        notes.append("the ledger read was interrupted mid-scan — records "
                     "may be missing (see the debug log)")
    if truncated and records:
        # records are oldest-first: if even the oldest scanned record is
        # newer than the window start, the capped tail scan did not reach
        # the whole requested window — say so instead of silently
        # under-reporting ("no silent caps").
        cutoff = time.time() - h * 3600.0
        oldest = records[0].ts
        if oldest > cutoff:
            covered_h = max(0.0, (time.time() - oldest) / 3600.0)
            notes.append(
                f"scan capped at the ledger tail "
                f"(~{_ACTIVITY_TAIL_BYTES // 1024}KB/{_ACTIVITY_SCAN_KEEP} "
                f"records) — it reaches back only ~{covered_h:.1f}h of the "
                f"requested {h:g}h window")
    if notes:
        report += "\n  (note: " + "; ".join(notes) + ")"
    return report


async def tool_introspect(
    action: str = None,
    query: str = None,
    limit: int = None,
    hours: float = None,
    self_model=None,
    context=None,
    **kwargs,
) -> str:
    """Read-only introspection over the agent's selfhood.

    Never raises — introspection is secondary to the user turn.
    """
    raw_action = (action or "summary").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    # One log line per introspection, whatever the action — the operator
    # monitors the live stream ('activity' logs itself, with the clamped
    # window it actually used).
    if raw_action != "activity":
        pretty_log("Introspect", f"{raw_action} requested",
                   icon=Icons.BRAIN_SUM)

    # 'activity' reads the autonomous-activity ledger, not the SelfModel —
    # it must keep working when selfhood is disabled, so it branches before
    # the self_model gate. This is the on-demand home of the maintenance
    # records the finalize banner no longer auto-surfaces (2026-07-17):
    # "what did you do while I was away?" lands here.
    if raw_action == "activity":
        # Guarded here (it returns before the selfhood try below): this
        # branch previously escaped the tool's never-raises contract — an
        # exception in the ledger render surfaced as a raw invocation error
        # instead of a graceful message.
        try:
            return _render_activity(context, hours=hours, limit=limit)
        except Exception as e:  # noqa: BLE001 — never break the turn
            logger.warning("introspect activity failed: %s: %s",
                           type(e).__name__, e)
            return f"Activity report failed: {type(e).__name__}: {e}"

    # 'learning' reads the learning-loop stores (lessons, competence,
    # episodes, calibration), not the SelfModel — it branches before the
    # self_model gate so it works with selfhood disabled. This is the
    # instrument for the "watch/keep-or-kill in ~2 weeks" criteria the
    # 2026-07 loop-closing work left pending.
    if raw_action == "learning":
        try:
            from ..core.learning_health import render_learning_health
            _md = getattr(context, "memory_dir", None)
            if _md is None:
                return "Learning health: memory_dir unavailable."
            # Off the event loop: this now includes a full trajectory-corpus
            # walk (the experiment stamp-coverage block), exactly like the
            # sibling 'experiments' action below.
            import asyncio as _asyncio
            # Pass args so flag-gated consumer rows report their REAL
            # state (§4BM R1 MIN-2 — the PRM's .uncertainty() row used to
            # print a hardcoded string that read as a live wiring claim).
            return await _asyncio.to_thread(
                render_learning_health, _md,
                getattr(context, "args", None))
        except Exception as e:
            return f"Learning health unavailable: {type(e).__name__}: {e}"

    # 'experiments' reads the trajectory corpus (arms are stamped on each
    # turn's record), not the SelfModel — so it branches before the selfhood
    # gate too. This is the read-side of the live randomized-arm framework:
    # "is the change I shipped actually better, on real traffic?"
    if raw_action == "experiments":
        try:
            from pathlib import Path as _Path
            from ..core.experiments import report_from_trajectories
            _md = getattr(context, "memory_dir", None)
            if _md is None:
                return "Experiments: memory_dir unavailable."
            # Off the event loop: the walk touches every day partition, and
            # this tool is called from an async handler that also serves SSE.
            import asyncio as _asyncio
            # DENY the other scope's names (R6 — consistent across all
            # four live-view surfaces): a spec re-scoped to bench must
            # not render its stale live stamps here, while disabled or
            # retired specs keep their own-population history.
            try:
                from ..core.experiments import (
                    SCOPE_BENCH as _SB, load_registry as _lr,
                    registry_path_for_context as _rpfc)
                _deny_live = set(_lr(_rpfc(context)).names_in_scope(_SB))
            except Exception:  # noqa: BLE001
                _deny_live = None
            _live = await _asyncio.to_thread(
                lambda: report_from_trajectories(
                    _Path(str(_md)).parent / "trajectories",
                    deny_names=_deny_live))
            # §4BF 1c: the BENCH population, rendered as its own clearly
            # labeled section — never folded into the live numbers. Only
            # when bench-scoped specs exist AND the bench corpus does.
            # Routed through the admissibility chokepoint with the
            # registry's bench names + the bench population label (R5
            # review: the direct-root read bypassed --no-bench, rendered
            # live-scoped names under the bench banner, and titled itself
            # "live randomized arms" with a user-turn denominator).
            try:
                from ..core.experiments import (
                    SCOPE_BENCH, load_registry, registry_path_for_context,
                    render_report, summarize_streaming)
                from ..core.admissibility import iter_bench_trajectories
                _reg = load_registry(registry_path_for_context(context))
                _bnames = set(_reg.names_for_scope(SCOPE_BENCH))
                if _bnames:
                    from ..core.experiments import SCOPE_LIVE as _SL
                    _deny_bench = set(_reg.names_in_scope(_SL))

                    def _bench_report():
                        b_all, b_trig, b_cov = summarize_streaming(
                            iter_bench_trajectories(
                                "experiments_bench",
                                getattr(context, "args", None)),
                            admit_task_kinds=("bench",),
                            deny_names=_deny_bench)
                        if not b_all and not b_cov.get("admitted_turns"):
                            return ""
                        return render_report(
                            b_all, triggered=b_trig, coverage=b_cov,
                            population=SCOPE_BENCH)
                    _bench = await _asyncio.to_thread(_bench_report)
                    if _bench:
                        _live += ("\n\n══ BENCH-SCOPED EXPERIMENTS ══\n"
                                  + _bench)
            except Exception:  # noqa: BLE001 — bench section is additive
                pass
            return _live
        except Exception as e:
            return f"Experiment report unavailable: {type(e).__name__}: {e}"

    if self_model is None or not getattr(self_model, "enabled", False):
        return (
            "Introspection is unavailable — the selfhood module is "
            "disabled (--no-self-model / --no-memory)."
        )

    try:
        if raw_action == "stats":
            return _render_stats(self_model.stats())

        if raw_action == "narrative":
            if self_model.narrative is None:
                return "No narrative on file."
            text = (self_model.narrative.latest() or "").strip()
            return text or "No narrative on file yet."

        if raw_action == "recent":
            return _render_recent(
                self_model, _clamp_limit(limit, _DEFAULT_RECENT),
            )

        if raw_action == "recall":
            q = (query or "").strip()
            if not q:
                return "SYSTEM ERROR: 'query' is required for recall."
            return _render_recall(
                self_model, q, _clamp_limit(limit, _DEFAULT_RECALL),
            )

        # Default: summary.
        return _render_summary(self_model)
    except Exception as e:  # noqa: BLE001 — never break the turn
        logger.warning("introspect tool failed: %s: %s", type(e).__name__, e)
        return f"Introspection failed: {type(e).__name__}: {e}"
