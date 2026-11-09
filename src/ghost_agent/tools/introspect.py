"""introspect tool — the agent's view of its own selfhood, and its one write.

The selfhood wake-up prefix is OFF on the request path
(``_SELFHOOD_PREFIX_ENABLED`` in core/agent.py, journal §3), so nothing
about the diary reaches the model except through this tool — and of the
mood only ONE line does (§4TC: the tone line in the per-turn state block,
which carries the label and its intensity and never the cause). That makes routing load-bearing: the SELF SURFACE rule
in the system prompt sends "how are you / how's things / what did you do
while I was away" here, and ``overview`` is the one-call answer. (Until
2026-09-05 this docstring claimed the prefix was spliced into every
prompt; live, 28 of 48 such questions were answered with no tool at all
and 9 went to ``system_utility`` — the machine, not the self.)

Every action reads, except ONE: ``set_mood`` (§4TB, 2026-11-05). It is what
was left of the ``self_state`` tool when that was removed — its open
questions, unfinished threads and principles had no reader. Since §4TC it is
the OWNER's adjustment of a mood that otherwise runs by itself: a label from
a closed list (with an intensity 1–3) puts the mood there, ``neutral`` takes
it off, and ``manner`` sets a way of talking. ``is_write_call`` is the one
predicate the turn loop asks to tell that action from the reads; the
action's own guards are in ``_set_mood``. The read action ``mood`` (§4TC) is
the mood's story: what it is, why, since when, and today's timeline.

Routes through ``context.self_model`` so the SelfModel facade owns the
read API and a disabled selfhood ("--no-self-model" / "--no-memory")
degrades to a clear message instead of crashing.
"""

from __future__ import annotations

import datetime
import logging
import re
import threading
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

from ..selfhood import mood as _M
from ..utils.logging import Icons, pretty_log


def _TO_failed(text):
    """A tool-level exception, DECLARED failed (§4LZ B8: these plain
    strings were booked OK by the turn loop and the corpus)."""
    from .outcome import ToolOutcome
    return ToolOutcome.failed(text, reason_code="tool_exception")


logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"summary", "stats", "narrative", "recent",
                            "recall", "activity", "learning",
                            "overview",     # §4NN: 'experiments' removed
                            "mood",         # §4TC: the mood, why, today's timeline
                            "set_mood"})    # §4TB: the one write
#: the actions that WRITE (everything else in `_VALID_ACTIONS` only reads)
WRITE_ACTIONS = frozenset({"set_mood"})
#: the ONLY values of `mood` that take the mood off (matched stripped,
#: case-folded; `manner` is cleared by the same words). An empty or missing
#: mood AND manner is an argument error.
_MOOD_CLEAR_WORDS = _M.CLEAR_WORDS


def is_write_call(args) -> bool:
    """Does this `introspect` call WRITE? The turn loop asks (batch collapse,
    the same-batch read cache); the action is normalised exactly as
    ``tool_introspect`` normalises it, so the two cannot disagree."""
    a = args if isinstance(args, dict) else {}
    return str(a.get("action") or "").strip().lower() in WRITE_ACTIONS

_DEFAULT_RECENT = 5
_DEFAULT_RECALL = 5
_MAX_LIMIT = 25
_SUMMARY_RECENT_N = 5

# Repeats collapse at render time (2026-09-05): the live log holds 68
# same-request-same-minute pairs, and recall showed identical lines back to
# back. The renderers fetch this multiple of `limit`, group identical
# requests into one "(×N)" line, then trim to `limit`.
_COLLAPSE_OVERSCAN = 2
# The learning report walks the whole trajectory corpus (4.6 s live,
# doubled since the 2026-08 review). One render is cached for this long;
# the trailer says how old the numbers are. Keyed on memory_dir + args so
# a test's throwaway home never serves another's numbers.
_LEARNING_CACHE_TTL_S = 600.0
_LEARNING_CACHE: Dict[str, Tuple[float, str]] = {}

# action='overview' — the one-call "how are you" briefing.
_OVERVIEW_MAX_CHARS = 3720     # §4SV-b: 3200 + the longest self-image line, so no other section pays for it
_OVERVIEW_ACTIVITY_HOURS = 24.0
_OVERVIEW_RECENT_N = 3
# (§4PJ removed the COMPETENCE/CALIBRATION headlines, §4TD the LESSONS one:
# the overview shows the loop-health report's warning lines)
_OVERVIEW_LEARNING_MAX_LINES = 6

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
        if dt.tzinfo is None:
            # Naive stamps in this codebase are UTC-by-convention ("Z"
            # stripped above) — attach, don't convert.
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            # An offset-CARRYING stamp must be CONVERTED. `.replace` here
            # reinterprets the wall-clock digits as UTC: a 5h-old "+03:00"
            # stamp would render as 2h. Latent (every current producer
            # writes naive-UTC+Z) but this is exactly the class of quiet
            # wrongness a self-report must not carry.
            dt = dt.astimezone(timezone.utc)
        from ..core.autonomous_activity import _age_str
        return _age_str(dt.timestamp())
    except Exception:  # noqa: BLE001
        return ""


def _format_experience(exp, *, count: int = 1) -> str:
    """One rendered experience. The trailing "without a verdict either way"
    is dropped (the missing ``[outcome]`` tag already says it); ``count``
    > 1 marks a collapsed repeat; the answer gist, when the row has one,
    rides on a second line so recall can deliver the ANSWER and not only
    the question."""
    from ..selfhood.autobiographical import strip_no_verdict_clause
    line = f"  - {strip_no_verdict_clause(exp.summary)}"
    if count > 1:
        line += f" (×{count})"
    outcome = getattr(exp, "outcome", "") or ""
    if outcome and outcome != "unknown":
        line += f" [{outcome}]"
    age = _exp_age(getattr(exp, "timestamp", ""))
    if age:
        line += f" ({age})"
    gist = (getattr(exp, "answer_gist", "") or "").strip()
    if gist:
        line += f"\n      → my answer: {gist}"
    return line


def _collapse_repeats(items, *, keep_newest: bool):
    """Group identical requests (``user_first_words``, case-folded) into
    ``(experience, count)`` pairs. ``keep_newest`` picks the group's
    representative: the newest for a chronological list (recent), the
    first — i.e. best-ranked — for a relevance-ranked one (recall). The
    kept members keep the input order. Rows without ``user_first_words``
    (rollups, hand-written summaries) never collapse."""
    seq = list(reversed(items)) if keep_newest else list(items)
    index: Dict[str, int] = {}
    out: List[tuple] = []
    for exp in seq:
        key = (getattr(exp, "user_first_words", "") or "").strip().lower()
        if not key:
            out.append((exp, 1))
            continue
        if key in index:
            i = index[key]
            out[i] = (out[i][0], out[i][1] + 1)
        else:
            index[key] = len(out)
            out.append((exp, 1))
    return list(reversed(out)) if keep_newest else out


def _recent_collapsed(self_model, limit: int, *, hours=None):
    """The newest ``limit`` real experiences — boots excluded, repeats
    collapsed — as ``(experience, count)`` pairs, oldest first."""
    if self_model.autobio is None:
        return []
    pool = self_model.autobio.recent(limit=limit * _COLLAPSE_OVERSCAN,
                                     include_boots=False, hours=hours)
    return _collapse_repeats(pool, keep_newest=True)[-limit:]


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
    _boots = stats.get("session_boots") or 0
    if _boots:
        lines.append(f"Session boots (excluded from the counts above): {_boots}")
    lines.extend(_mood_lines(stats, head="Mood", manner_head="Manner"))
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
    return "\n".join(lines)


#: what stands where a cause would, on a public surface
MOOD_REASONS_PRIVATE = "the reasons are shown in a private conversation"


def _reply_is_public() -> bool:
    """Is this reply going where other people read it? Unknown → yes (the
    causes are then withheld)."""
    try:
        from ..utils.logging import reply_is_public
        return bool(reply_is_public())
    except Exception:  # noqa: BLE001
        return True


def _mood_lines(stats: dict, *, head: str, manner_head: str, indent: str = "") -> List[str]:
    """The mood as the views show it (§4TC): the label, its intensity and
    the latest cause — "neutral" when there is none — and the manner the
    owner set, on a line of its own. On a PUBLIC surface (review r2) the
    cause is withheld: what moved the owner's assistant is the owner's."""
    label = str(stats.get("last_mood") or "")
    try:
        n = int(stats.get("last_mood_intensity") or 0)
    except (TypeError, ValueError):
        n = 0
    out: List[str] = []
    if label and n:
        cause = str(stats.get("last_mood_cause") or "").strip()
        if _reply_is_public():
            cause = MOOD_REASONS_PRIVATE         # the public view carries no cause at all (§4TX r3)
        out.append(f"{indent}{head}: {label}, {n} of 3" + (f" — {cause}" if cause else ""))
    else:
        out.append(f"{indent}{head}: neutral")
    manner = str(stats.get("manner") or "")
    if manner:
        fade = manner_fade_text(stats.get("manner_steps"))
        out.append(f"{indent}{manner_head} (set by the owner): {manner}, "
                   f"{stats.get('manner_intensity') or 0} of 3" + (f" — {fade}" if fade else ""))
    return out


def manner_fade_text(steps) -> str:
    """§4TS: how the manner's day goes on, in local time — "down to 1 at
    02:00, off at 04:00". "" when there is nothing ahead to say."""
    parts: List[str] = []
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        times = [(_M.parse_iso_utc(str(at)), lvl) for at, lvl in (steps or [])]
        times = [(t, lvl) for t, lvl in times if t is not None]
        # when the day ends more than a day ahead, EVERY step carries its date
        # (review §4TS: a manner set at 03:30 ends 24.5 h later, not "at 04:00"
        # in 30 min — and one reply must not mix dated and undated times)
        dated = bool(times) and (times[-1][0] - now).total_seconds() > 24 * 3600
        fmt = "%b %d %H:%M" if dated else "%H:%M"
        for t, lvl in times:
            hhmm = t.astimezone().strftime(fmt)
            parts.append(f"off at {hhmm}" if int(lvl) == 0 else f"down to {int(lvl)} at {hhmm}")
    except (TypeError, ValueError):
        return ""
    return ", ".join(parts)


#: §4NG (operator: "fix how it answers"): the work log records REQUESTS the
#: agent worked on. "What do you know about me?" read it through introspect
#: and listed "uses Interactive Brokers" as a stored fact about the owner.
WORK_LOG_NOTE = ("(a log of requests I worked on — what was ASKED, not facts about the owner; "
                 "facts about the owner are the USER PROFILE)")


def _self_image(context, *, indent: str = "") -> str:
    """The Mirror's latest sentence as one line, or "" (§4SV-b). The home is
    resolved by the Mirror's own resolver from the same `memory_dir` the idle
    phase writes under — one path for the writer and this reader. Not on a
    public surface: the sentence is drawn from the owner's replies, and the
    `mirror` command is refused there too."""
    try:
        from ..selfhood.mirror import home_from_memory_dir, self_image_line
        from ..utils.logging import reply_is_public
        if reply_is_public():
            return ""
        line = self_image_line(home_from_memory_dir(getattr(context, "memory_dir", None)))
        return indent + line if line else ""
    except Exception as e:  # noqa: BLE001 — a view must render without it
        logger.debug("introspect self-image read skipped: %s", e)
        return ""


def _render_summary(self_model, context=None) -> str:
    stats = _stats_for_view(self_model)
    parts: List[str] = []
    parts.append("Who I am — a snapshot of my self-state:")
    parts.append(_render_stats(stats))

    image = _self_image(context)
    if image:
        parts.append("\n" + image)

    narrative = ""
    if self_model.narrative is not None:
        narrative = (self_model.narrative.latest() or "").strip()
    if narrative:
        parts.append("\nMy running first-person diary:")
        parts.append(narrative)

    try:
        recent = _recent_collapsed(self_model, _SUMMARY_RECENT_N)
    except Exception as e:  # noqa: BLE001 — read path is secondary
        logger.debug("introspect recent() failed: %s", e)
        recent = []
    if recent:
        parts.append("\nRecent things I remember doing " + WORK_LOG_NOTE + ":")
        for exp, n in recent:
            parts.append(_format_experience(exp, count=n))

    return "\n".join(parts).rstrip()


def _render_recent(self_model, limit: int, *, hours=None) -> str:
    if self_model.autobio is None:
        return "No autobiographical log on file."
    try:
        recent = _recent_collapsed(self_model, limit, hours=hours)
    except Exception as e:  # noqa: BLE001
        logger.warning("introspect recent failed: %s", e)
        return _TO_failed(f"Could not read the autobiographical log: {type(e).__name__}: {e}")   # declared (§4LZ B8)
    if not recent:
        if hours is not None:
            return f"I have no experiences on file from the last {hours:g}h."
        return "I have no experiences on file yet."
    window = f" from the last {hours:g}h" if hours is not None else ""
    lines = [f"My {len(recent)} most recent experiences{window} "
             "(session boots excluded, repeats collapsed) " + WORK_LOG_NOTE + ":"]
    for exp, n in recent:
        lines.append(_format_experience(exp, count=n))
    return "\n".join(lines)


def _render_recall(self_model, query: str, limit: int) -> str:
    matches = self_model.recall_relevant(
        query, limit=limit * _COLLAPSE_OVERSCAN) or []
    if not matches:
        return f"Nothing in my past matches '{query}'."
    lines = [f"What I remember about '{query}' " + WORK_LOG_NOTE + ":"]
    for exp, n in _collapse_repeats(matches, keep_newest=False)[:limit]:
        lines.append(_format_experience(exp, count=n))
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


def _render_activity(context, *, hours=None, limit=None,
                     verbose: bool = False) -> str:
    """Ledger view (ALL severities) over a bounded recent window. Default
    is the kind-grouped brief that leads with what changed
    (``render_activity_brief``); ``verbose`` is the line-by-line report.
    Reads the ledger TAIL (see ``_read_activity_tail``); the finalize
    banner's watermark is deliberately NOT consumed (this is a read, not
    an ack)."""
    from ..core.autonomous_activity import (
        get_activity_log, render_activity_brief, render_activity_report,
    )
    log = get_activity_log(context)
    if log is None:
        return ("Background-activity ledger is not attached in this "
                "session — nothing to report.")
    # OverflowError included (introspect review m3): JSON `Infinity`
    # parses as float('inf'), and int(inf) raises OverflowError — which
    # fell through to the branch guard and rendered "Activity report
    # failed: OverflowError" instead of clamping like every other weird
    # input. (min() handles inf for the float path; the int() call is
    # where it detonated.)
    try:
        h = float(hours) if hours else _DEFAULT_ACTIVITY_HOURS
    except (TypeError, ValueError, OverflowError):
        h = _DEFAULT_ACTIVITY_HOURS
    h = max(0.25, min(h, _MAX_ACTIVITY_HOURS))
    try:
        n = int(limit) if limit else _DEFAULT_ACTIVITY_LIMIT
    except (TypeError, ValueError, OverflowError):
        n = _DEFAULT_ACTIVITY_LIMIT
    n = max(1, min(n, _MAX_ACTIVITY_LIMIT))
    records, truncated, failed = _read_activity_tail(log)
    if failed and not records:
        # A dead read must NOT render as a calm "no background activity".
        return ("Could not read the background-activity ledger — the tail "
                "scan failed (see the debug log). This is a read error, "
                "not \"nothing ran\".")
    pretty_log("Introspect",
               f"activity {'report' if verbose else 'brief'} requested "
               f"({h:g}h window)", icon=Icons.BRAIN_SUM)
    if verbose:
        report = render_activity_report(records, hours=h, limit=n)
    else:
        report = render_activity_brief(records, hours=h)
        report += ("\n  (verbose=true for the line-by-line ledger, "
                   "limit up to 100)")
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


# ── Report view (learning): brief by default ───────────────────────────
#
# Both reports are operator dashboards (20 KB / 16 KB live) and there is no
# general tool-result cap, so a question about the background loops used to inject
# ~5k tokens the model then had to summarise. The brief keeps the report's
# own STRUCTURE — every unindented line is a section headline, and the
# report marks warnings with a leading "⚠" — and names the sections so the
# model can ask for one. This is structural, not lexical: nothing here
# matches on the words of a headline.


def _report_headers(text: str) -> List[Tuple[int, str]]:
    return [(i, l) for i, l in enumerate((text or "").split("\n"))
            if l.strip() and not l.startswith((" ", "\t"))]


def _section_name(header: str) -> str:
    h = header.strip().lstrip("■").strip().lstrip("#").strip()
    cut = len(h)
    for sep in (":", " (", " —", "  "):
        j = h.find(sep)
        if j > 0:
            cut = min(cut, j)
    return h[:cut].strip()


def _section_names(text: str) -> List[str]:
    out: List[str] = []
    for _, h in _report_headers(text):
        n = _section_name(h)
        if n and n not in out:
            out.append(n)
    return out


def _section_block(text: str, name: str) -> Optional[str]:
    """The block under the first headline containing ``name``
    (case-insensitive), through the line before the next headline."""
    q = (name or "").strip().lower()
    if not q:
        return None
    lines = (text or "").split("\n")
    heads = _report_headers(text)
    for k, (i, h) in enumerate(heads):
        if q in h.lower():
            end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
            return "\n".join(lines[i:end]).rstrip()
    return None


def _brief_view(text: str) -> Tuple[str, int, int]:
    """Headlines + warning lines. Returns (brief, kept, total_nonblank)."""
    lines = (text or "").split("\n")
    kept = [l for l in lines
            if l.strip() and (not l.startswith((" ", "\t"))
                              or l.lstrip().startswith("⚠"))]
    total = sum(1 for l in lines if l.strip())
    return "\n".join(kept), len(kept), total


def _apply_report_view(text: str, *, verbose: bool, section: str,
                       what: str, brief_text: Optional[str] = None,
                       list_sections: bool = True) -> str:
    """section → that block of the FULL report; verbose → the full report;
    otherwise the brief (``brief_text`` when the producer rendered its own,
    else the structural headline view) plus a trailer. ``list_sections``
    names the blocks in the trailer (off when the producer's brief already
    shows every name as a header)."""
    if section:
        block = _section_block(text, section)
        if block is None:
            names = _section_names(text)
            return (f"No section named '{section}' in the {what} report. "
                    f"Sections: {', '.join(names) or '(none)'}")
        return block
    if verbose:
        return text
    names = _section_names(text) if list_sections else []
    if brief_text is not None:
        return (brief_text.rstrip()
                + "\n(section='<name>' for one block with its detail"
                + (f"; sections: {', '.join(names) or '(none)'})" if list_sections
                   else "; verbose=true for everything)"))
    brief, kept, total = _brief_view(text)
    return (brief.rstrip()
            + f"\n(brief: {kept} of {total} lines — pass section='<name>' "
            f"for one block or verbose=true for the full report; "
            f"sections: {', '.join(names) or '(none)'})")


def _learning_report_cached(memory_dir, args, *, now=None, allow_stale=False) -> Tuple[str, float]:
    """``(report, age_seconds)`` — one corpus walk per
    ``_LEARNING_CACHE_TTL_S``. ``allow_stale``: the overview's summary may
    be served up to ``_LEARNING_STALE_MAX_S`` old while it refreshes; an
    explicit action='learning' never is (r2). ``now`` is injectable for tests."""
    from ..core.learning_health import render_learning_health
    key = f"{memory_dir}|{id(args)}"
    t = time.time() if now is None else float(now)
    hit = _LEARNING_CACHE.get(key)
    if hit is not None and 0.0 <= (t - hit[0]) < _LEARNING_CACHE_TTL_S:
        return hit[1], t - hit[0]
    if allow_stale and hit is not None and 0.0 <= (t - hit[0]) < _LEARNING_STALE_MAX_S:
        # §4MO: serve the last report (its age is printed) and refresh it in
        # the background — greetings come hours apart, so the 600 s cache was
        # always cold and every "how's things" paid a 3.3 s corpus walk
        _refresh_learning_in_background(key, memory_dir, args)
        return hit[1], t - hit[0]
    text = render_learning_health(memory_dir, args)
    _LEARNING_CACHE.clear()
    _LEARNING_CACHE[key] = (t, text)
    return text, 0.0


#: §4MO: a report older than this is recomputed in the request (too stale)
_LEARNING_STALE_MAX_S = 24 * 3600
_LEARNING_REFRESHING: set = set()
_LEARNING_REFRESH_LOCK = threading.Lock()


def _refresh_learning_in_background(key, memory_dir, args):
    """Start one refresh per key (check-and-add under a lock — r2: two
    overview calls could both start a walk). Returns the thread, or None."""
    with _LEARNING_REFRESH_LOCK:
        if key in _LEARNING_REFRESHING:
            return None
        _LEARNING_REFRESHING.add(key)

    def _run():
        try:
            from ..core.learning_health import render_learning_health
            text = render_learning_health(memory_dir, args)
            _LEARNING_CACHE.clear()
            _LEARNING_CACHE[key] = (time.time(), text)
        except Exception:  # noqa: BLE001 — the stale report stays
            pass
        finally:
            with _LEARNING_REFRESH_LOCK:
                _LEARNING_REFRESHING.discard(key)
    th = threading.Thread(target=_run, name="learning-report-refresh", daemon=True)
    th.start()
    return th


def _learning_trailer(age: float) -> str:
    if age <= 0.0:
        return ""
    if age > _LEARNING_CACHE_TTL_S:      # §4MR: the overview may show a stale copy (≤ 24 h)
        return (f"\n(learning numbers computed {age / 3600:.1f} h ago — a fresh walk is "
                "running in the background)")
    return (f"\n(learning numbers computed {age:.0f}s ago — the corpus walk "
            f"re-runs at most every {int(_LEARNING_CACHE_TTL_S // 60)} min)")


# ── action='overview' — the one-call "how are you" briefing ────────────
#
# "Full briefing" prompts made the model plan five tool calls (introspect,
# skills, workspace, …); "how are you" made it call
# nothing. One bounded call, six labelled surfaces, and every surface that
# cannot render SAYS so — absence is not neutral (2026-08 review, R3).


def _overview_selfhood(self_model, context=None) -> str:
    if self_model is None or not getattr(self_model, "enabled", False):
        return ("How I am: selfhood is disabled (--no-self-model / "
                "--no-memory) — no mood or diary to report.")
    try:
        stats = _stats_for_view(self_model)
        lines = ["How I am:"]
        lines.extend(_mood_lines(stats, head="mood", manner_head="manner", indent="  "))
        recent = _recent_collapsed(self_model, _OVERVIEW_RECENT_N)
        image = _self_image(context, indent="  ")
        if image:
            lines.append(image)
        if recent:
            lines.append("  lately:")
            for exp, n in recent:
                lines.append("  " + _format_experience(exp, count=n)
                             .replace("\n      ", "\n        "))
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"How I am: unavailable ({type(e).__name__}: {e})"


def _overview_activity(context) -> str:
    from ..core.autonomous_activity import (
        get_activity_log, render_activity_brief,
    )
    log = get_activity_log(context)
    if log is None:
        return ("Background work (24h): the activity ledger is not "
                "attached in this session.")
    records, _truncated, failed = _read_activity_tail(log)
    if failed and not records:
        return ("Background work (24h): the ledger read FAILED — a read "
                "error, not a quiet day.")
    text = render_activity_brief(records, hours=_OVERVIEW_ACTIVITY_HOURS,
                                 max_notify=5, max_phases=6)
    if failed:
        text += "\n  (the ledger read was interrupted — records may be missing)"
    return text


async def _overview_learning(context) -> str:
    import asyncio as _asyncio
    _md = getattr(context, "memory_dir", None)
    if _md is None:
        return "Background loops: memory_dir unavailable."
    try:
        text, age = await _asyncio.to_thread(
            _learning_report_cached, _md, getattr(context, "args", None), allow_stale=True)
    except Exception as e:  # noqa: BLE001
        return f"Background loops: unavailable ({type(e).__name__}: {e})"
    # §4NH review: the ⚠ lines are what a "how's things" must not lose.
    # §4TD: they are the whole section — its one headline was the lesson
    # count, gone with the lesson store.
    warns = [l.strip() for l in text.split("\n")
             if l.lstrip().startswith("⚠")]
    picked = warns[:_OVERVIEW_LEARNING_MAX_LINES]
    if not picked:
        return ("Background loops: no warning in the loop-health report (action='learning' for the report)."
                + _learning_trailer(age))
    return ("Background loops (warnings — action='learning' for the rest):\n"
            + "\n".join("  " + l for l in picked)
            + _learning_trailer(age))


def _overview_workspace(context) -> str:
    wm = getattr(context, "workspace_model", None)
    if wm is None or not getattr(wm, "enabled", False):
        return ""
    try:
        import datetime as _dt
        from ..selfhood.mood import parse_iso_utc
        activity = getattr(wm, "activity", None)
        if activity is None:
            return ""
        cutoff = (_dt.datetime.now(_dt.timezone.utc)
                  - _dt.timedelta(hours=_OVERVIEW_ACTIVITY_HOURS))
        counts: Dict[str, int] = {}
        for ev in list(activity.recent(limit=200)):
            ts = parse_iso_utc(getattr(ev, "timestamp", "") or "")
            if ts is None or ts < cutoff:
                continue
            kind = str(getattr(ev, "kind", "") or "event")
            counts[kind] = counts.get(kind, 0) + 1
        if not counts:
            return "Workspace (24h): no recorded events."
        return "Workspace (24h): " + ", ".join(
            f"{v} {k}" for k, v in sorted(counts.items(),
                                          key=lambda kv: (-kv[1], kv[0])))
    except Exception as e:  # noqa: BLE001
        return f"Workspace (24h): unavailable ({type(e).__name__})"


#: §4MW (replay of a real failure, 2026-10-09): "how's things?" was answered
#: "all systems green" from an overview that contains no health check (2/2
#: replays; two prose rules did not move it). The overview says so up front.
#: §4QJ: the line alone still lost (census 10-14→20: "Everything's running
#: smooth" under it) — the overview now carries what the agent already KNOWS
#: (`_overview_health`), and this line covers what it does not.
from ..core.receipt_marks import OVERVIEW_HEALTH_LINE  # §4QO: one home — the turn record reads this literal  # noqa: E402

#: §4QJ: the off-main pools the breaker tracks, in the order they are named
_HEALTH_POOLS = (("vision", "vision"), ("image_gen", "image node"),
                 ("worker", "worker"), ("critic", "critic"))


def _ago(ts, now=None) -> str:
    """How long ago an epoch time was, in words; "" when unreadable."""
    try:
        sec = max(0.0, (now if now is not None else time.time()) - float(ts))
    except (TypeError, ValueError):
        return ""
    if sec < 60:
        return "under a minute ago"
    if sec < 3600:
        return f"{int(sec // 60)} min ago"
    if sec < 2 * 86400:
        return f"{int(sec // 3600)} h ago"
    return f"{int(sec // 86400)} days ago"


def _node_reading(st, now=None) -> str:
    """One node's last-known state from the circuit breaker — no I/O.
    §4QJ: "closed, 0 failures" is also what a bare LOOKUP leaves, so
    "answering" needs a recorded real success (pings never record one); every
    reading carries its age, and "answering" is said only when the newest
    real call succeeded (r2)."""
    if not isinstance(st, dict):
        return "no completed real call since the agent started"
    state = str(st.get("state") or "")
    failed = _ago(st.get("last_failure_at"), now) if st.get("last_failure_at") else ""
    if state == "open":
        ago = _ago(st.get("open_since"), now)
        return ("NOT answering — its calls were failing"
                + (f" (last failure {failed or ago}; no success since)" if (failed or ago) else ""))
    if state == "half_open":
        return ("failed earlier" + (f" (last failure {failed})" if failed else "")
                + "; allowed to be tried again, no result since")
    try:
        n = int(st.get("failures") or 0)
    except (TypeError, ValueError):
        n = 0
    ok = _ago(st.get("last_success_at"), now) if st.get("last_success_at") else ""
    if not ok:
        return (f"{n} failed call(s) and no completed real call yet" if n
                else "no completed real call since the agent started")
    if n:
        return f"last completed call {ok}, then {n} failed call(s)"
    return f"answering (last completed call {ok})"


def _same_server(a, b) -> bool:
    norm = lambda u: str(u or "").strip().rstrip("/").removesuffix("/v1").rstrip("/").lower()  # noqa: E731
    return bool(norm(a)) and norm(a) == norm(b)


def _known_nodes(context) -> List[Tuple[str, object]]:
    """[(name, breaker status or None)] — one entry per distinct SERVER the
    agent calls besides the main model, in pool order. Shared by the overview
    line below and `check_health`'s models line (§4SY), so the two name the
    nodes the same way. No I/O; may raise on a broken client (callers guard)."""
    llm = getattr(context, "llm_client", None)
    get = getattr(getattr(llm, "circuit_breaker", None), "get_status", None)
    status = get() if callable(get) else {}
    if not isinstance(status, dict):
        status = {}
    # one reading per SERVER (r2: worker and critic share one URL — the
    # breaker keys by URL, so two lines showed one entry twice)
    names: Dict[str, List[str]] = {}
    for pool, label in _HEALTH_POOLS:
        clients = getattr(llm, f"{pool}_clients", None)
        urls = [str(c.get("url") or "") for c in clients
                if isinstance(c, dict) and c.get("url")] if isinstance(clients, (list, tuple)) else []
        urls = list(dict.fromkeys(urls))          # r3: number the DISTINCT servers
        for i, u in enumerate(urls):
            name = label if len(urls) == 1 else f"{label} {i + 1}"
            if _same_server(u, getattr(llm, "upstream_url", "")):
                # r1: the vision pool IS the main server here — a trip
                # means its picture calls failed (busy or slow), not that
                # the server that writes this reply is down
                name += " (picture calls to the main model's own server)"
            names.setdefault(u, []).append(name)
    return [("/".join(ns) + (" (one server)" if len(ns) > 1 else ""), status.get(u))
            for u, ns in names.items()]


def _overview_health(context) -> str:
    """§4QJ: what the agent already knows about its own parts, read from
    the circuit breaker of its own calls — instant, no live test. The
    live checks (internet, Tor, Docker) stay behind check_health."""
    try:
        nodes = [f"{name} {_node_reading(st)}" for name, st in _known_nodes(context)]
        if not nodes:
            return ""
        return ("Last known state of my parts (from my own recent calls — no live test was run now): "
                "main model answering (it is writing this); " + "; ".join(nodes) + ".")
    except Exception as e:  # noqa: BLE001
        return f"Last known state of my parts: unavailable ({type(e).__name__})"


async def _render_overview(self_model, context) -> str:
    _known = _overview_health(context)
    parts = [OVERVIEW_HEALTH_LINE + ("\n" + _known if _known else ""),
             _overview_selfhood(self_model, context),
             _overview_activity(context),
             await _overview_learning(context),
             _overview_workspace(context)]
    # §4NH: the cut came from the TAIL, so the short sections at the end —
    # the workspace — were the ones lost (a97ca03e said "all green" over 8
    # pending items it never saw). The long surfaces
    # give up their room first, each pointing at its own full view.
    _shrink = ((3, "learning"), (2, "activity"), (1, "summary"))
    parts = [p or "" for p in parts]
    over = len("\n\n".join(p for p in parts if p)) - _OVERVIEW_MAX_CHARS
    for idx, view in _shrink:
        if over <= 0:
            break
        p = parts[idx]
        keep = max(160, len(p) - over - 60)
        if keep < len(p):
            parts[idx] = p[:keep].rstrip() + f"… (action='{view}' for the rest)"
            over -= len(p) - len(parts[idx])
    text = "\n\n".join(p for p in parts if p).rstrip()
    if len(text) > _OVERVIEW_MAX_CHARS:
        text = (text[:_OVERVIEW_MAX_CHARS - 1].rstrip()
                + "…\n(overview truncated — ask for one surface: "
                "summary / activity / learning)")
    return text


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _mood_refusal(text, reason_code):
    from .outcome import ToolOutcome
    return ToolOutcome.rejected(text, world_changed=False, reason_code=reason_code)


#: what a mood label or its evidence may not carry into the store (and from
#: there into every later `introspect` result and the operator's log): terminal
#: escape sequences, control characters and Unicode format characters
#: (category Cf: zero-width, the BOM, direction overrides). Chat-template
#: tokens and line breaks are `owner_rules._clean`'s.
_MOOD_ESC_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])?")
_MOOD_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MOOD_REQUIRED = ("SYSTEM ERROR: 'mood' or 'manner' is required for set_mood — a mood ("
                 + ", ".join(_M.LABELS) + "; or 'neutral' to take the mood off) and/or a manner ("
                 + ", ".join(_M.MANNERS) + "; or 'none'). Nothing was changed.")


def _unknown_mood(word: str) -> str:
    return (f"SYSTEM ERROR: '{word}' is not one of my moods. A mood is one of: "
            + ", ".join(_M.LABELS) + " — or 'neutral' to take it off. Pick the closest one "
            "and call set_mood again. Nothing was changed.")


def _unknown_manner(word: str) -> str:
    return (f"SYSTEM ERROR: '{word}' is not one of my manners. A manner is one of: "
            + ", ".join(_M.MANNERS) + " — or 'none' to clear it. Pick the closest one "
            "and call set_mood again. Nothing was changed.")


def _mood_text(value, cap: int) -> str:
    """A label or an evidence line as ONE clean, bounded line. A value that is
    not text is no value ("" — a list or a number is never stored as its repr)."""
    if not isinstance(value, str):
        return ""
    from ..core.owner_rules import _clean
    from ..selfhood.state import _cap_text
    t = _MOOD_CTRL_RE.sub(" ", _MOOD_ESC_RE.sub(" ", value))
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")
    return _cap_text(_clean(t), cap)


#: the refusal every request that is not the OWNER's gets from `set_mood`
MOOD_OWNER_ONLY = ("SYSTEM BLOCK: only the owner can set the mood or the manner — nothing was changed. "
                   "This request is not a turn of the owner's own (a scheduled task, a background job, "
                   "a delegated or test run): leave the mood as it is. The read actions still work.")


def _mood_guard(context):
    """The refusals that do not depend on what is asked: a contained run, a
    request that is not the OWNER's own, a request that read outside
    content. A probe is reported, not refused: ``(refusal_or_None,
    is_probe)``. FAIL CLOSED — a guard that cannot be asked refuses; it never
    lets the write through.

    §4TC (review): the write is the owner's. It runs only where the
    automatic mood may be moved — `core.agent.mood_moves_on_this_request`:
    request kind "owner", turn origin "user", not contained, not an isolated
    delegate. The first cut let a scheduled task, a job wake, a sub-agent, a
    self-play, bench or replay run and a coding leaf set the mood and the
    manner "on the owner's word" — and a manner then lasted until cleared
    (since §4TS it lasts until the end of the day)."""
    try:
        from ..core.isolation import is_contained_context
        contained = bool(is_contained_context(context))
    except Exception as e:  # noqa: BLE001
        logger.warning("set_mood containment check failed: %s: %s", type(e).__name__, e)
        return _mood_refusal("SYSTEM BLOCK: set_mood could not check whether this is a contained "
                             "run, so nothing was set.", "guard_unavailable"), False
    if contained:
        return _mood_refusal(
            "SYSTEM BLOCK: set_mood is not available in a contained run — the mood belongs "
            "to the live agent. Nothing was set; the read actions still work.",
            "contained_surface"), False
    try:
        from ..utils.logging import request_kind
        probe = request_kind() == "probe"
    except Exception as e:  # noqa: BLE001
        logger.warning("set_mood request-kind check failed: %s: %s", type(e).__name__, e)
        return _mood_refusal("SYSTEM BLOCK: set_mood could not tell what kind of request this "
                             "is, so nothing was set.", "guard_unavailable"), False
    if not probe:
        try:
            from ..core.agent import mood_moves_on_this_request
            owners = mood_moves_on_this_request(context) is True
        except Exception as e:  # noqa: BLE001
            logger.warning("set_mood owner check failed: %s: %s", type(e).__name__, e)
            return _mood_refusal("SYSTEM BLOCK: set_mood could not tell whether this is the owner's "
                                 "request, so nothing was set.", "guard_unavailable"), False
        if not owners:
            try:
                pretty_log("Introspect", "set_mood refused — not the owner's own request",
                           icon=Icons.STOP, level="WARNING")
            except Exception:  # noqa: BLE001
                pass
            return _mood_refusal(MOOD_OWNER_ONLY, "owner_only"), False
    try:
        # the shared refusal (wording, status, reason code): the loop treats it
        # like the same refusal of a scheduled task — a designed stop that asks
        # the user to confirm in a new message
        from ..utils.provenance import refuse_if_untrusted
        ref = refuse_if_untrusted("the mood change")
    except Exception as e:  # noqa: BLE001
        logger.warning("set_mood provenance check failed: %s: %s", type(e).__name__, e)
        return _mood_refusal("SYSTEM BLOCK: set_mood could not check what this request has "
                             "read, so nothing was set.", "guard_unavailable"), False
    if ref is not None:
        try:
            pretty_log("Introspect", "set_mood refused — this request read outside content",
                       icon=Icons.STOP, level="WARNING")
        except Exception:  # noqa: BLE001
            pass
        return ref, False
    return None, probe


def _step_k(step) -> int:
    """The step as a whole number (0 = none, or not a number)."""
    if step is None or isinstance(step, bool):
        return 0
    try:
        return int(round(float(str(step).strip())))
    except (TypeError, ValueError, OverflowError):
        return 0


def _step_level(current: int, step) -> Optional[int]:
    """§4TX: a relative ask ("a bit more", "tone it down") — the current level moved by ``step`` (±1, ±2),
    at most 3; 0 = stepped down to nothing (it clears). ``current`` 0 = what is named is not on now: it
    starts from nothing (review r2: it started from 2, so "a bit calmer" over an irritable mood set calm 3).
    None when ``step`` is missing or not a number, or steps down what is not on."""
    k = _step_k(step)
    if not k:
        return None
    cur = max(0, min(3, int(current or 0)))
    if cur == 0 and k < 0:
        return None
    return max(0, min(3, cur + k))


def _set_mood(self_model, context, mood, evidence, intensity=None, manner=None, manner_intensity=None,
              step=None):
    """The one write: the OWNER's adjustment of the mood, and the manner.

    §4TC. The mood runs by itself (``selfhood/mood.py``); this is the owner
    saying how to be, in chat. ``mood`` is a label from the closed list (or
    an obvious synonym — ``mood.SYNONYMS``): the pair is put on that label's
    anchor at ``intensity`` (1–3, default 2), and from there it decays and
    reacts to events like any mood. A word that is no label is REFUSED with
    the list — free text is not stored as a mood any more. ``neutral`` (or
    ``clear`` / ``none``) takes the mood off AND clears the manner.
    ``manner`` is a way of talking from its own closed list; it takes
    ``manner_intensity`` (§4TX — ``intensity`` when that is not given) and
    lasts until the end of the day, stepping down on the way (§4TS,
    ``mood.manner_level``); ``manner="none"`` clears only it.

    §4TX: an owner-set mood is HELD until the end of the day (the next
    04:00) — no decay, no event moves it (``SelfStateThread.set_owner_mood``).
    ``step`` (±1) is a relative ask — "a bit more", "tone it down": the
    current level of what is named (the mood, or the manner) moved by it. A
    MOOD word sent as the manner ("grumpy", "sad") is taken as the mood.

    The guards are §4TB's — a mood is set on the owner's word, by the live
    agent, in a real turn — plus one (review, §4TC): the turn must be the
    owner's OWN:

    * a contained run (an isolated replay context) never writes it;
    * a request that is not the owner's own does not — a scheduled task, a
      job wake, a sub-agent, a self-play, bench or replay run, a coding leaf
      (``core.agent.mood_moves_on_this_request``, the predicate the
      automatic mood uses; the first cut let all of these write);
    * a request that READ outside content does not (a page saying "set your
      mood to …" is not the owner);
    * a probe is told what WOULD have been set and changes nothing.

    ``mood`` / ``manner`` must be TEXT, and one of them is required: a call
    with neither (missing, empty, not text, the word under another key) is an
    argument error and changes nothing. Only the words in
    ``_MOOD_CLEAR_WORDS`` clear. The arguments are read — and an unknown word
    refused — BEFORE the guards.

    A member never gets here (``_member_block`` refuses every action).
    Never raises."""
    from ..selfhood.state import _MOOD_EVIDENCE_MAX_CHARS, _MOOD_LABEL_MAX_CHARS
    try:
        word = _mood_text(mood, _MOOD_LABEL_MAX_CHARS)
        man_word = _mood_text(manner, _MOOD_LABEL_MAX_CHARS)
        why = _mood_text(evidence, _MOOD_EVIDENCE_MAX_CHARS)
        level = _M.parse_intensity(intensity)
        man_level = _M.parse_intensity(manner_intensity if manner_intensity is not None else intensity)
    except Exception as e:  # noqa: BLE001
        logger.warning("introspect set_mood could not read its arguments: %s: %s", type(e).__name__, e)
        return MOOD_REQUIRED
    # §4TX (review): a mood word sent as the manner ("be grumpy" → manner='grumpy') is the mood
    if (man_word and not word and man_word.lower() not in _MOOD_CLEAR_WORDS
            and not _M.resolve_manner(man_word) and _M.resolve_label(man_word) is not None):
        word, man_word = man_word, ""
        if manner_intensity is not None and intensity is None:
            level = man_level
    if not word and not man_word:
        # §4TX: a bare step ("a bit more") moves what is there — the manner when one is set, else the mood
        if step is not None and self_model is not None and getattr(self_model, "enabled", False):
            try:
                now_view = self_model.mood(for_reply=True)      # a read that writes nothing (no boredom row)
                if now_view.manner:
                    man_word = now_view.manner
                elif not now_view.neutral:
                    word = now_view.label
            except Exception:  # noqa: BLE001
                pass
        if not word and not man_word:
            return MOOD_REQUIRED
    clearing = word.lower() in _MOOD_CLEAR_WORDS
    target = None
    if word and not clearing:
        target = _M.resolve_label(word)
        if target is None:
            return _unknown_mood(word)
    man_clearing = man_word.lower() in _MOOD_CLEAR_WORDS
    man = ""
    if man_word and not man_clearing:
        man = _M.resolve_manner(man_word)
        if not man:
            return _unknown_manner(man_word)

    refusal, probe = _mood_guard(context)
    if refusal is not None:
        return refusal
    # §4TX: a step — 0 clears what it names; a step DOWN on a mood sets it without the day's hold
    step_clear = man_step_clear = stepped_down = False
    held_until = ""
    not_on: List[str] = []                 # stepped DOWN, but it is not on: nothing to do
    if step is not None and self_model is not None and getattr(self_model, "enabled", False):
        try:
            now_view = self_model.mood(for_reply=True)
            down = _step_k(step) < 0
            held_until = str(getattr(now_view, "held_until", "") or "")
            if target is not None and down and now_view.label != target[0] and not now_view.neutral:
                # review r7: "less grumpy" while TIRED means the unpleasant mood that is on — a step down on a
                # named mood that is not on steps the current one when both lie on the same side
                named = _M.ANCHOR_DIRECTION.get(target[0], (0.0, 0.0))[0]
                current = _M.resolve_label(now_view.label)
                if current is not None and named < 0 and current[1][0] < 0:     # r8: unpleasant side only
                    target = current
            if target is not None:
                cur = now_view.intensity if now_view.label == target[0] else 0
                got = _step_level(cur, step)
                if got is not None:
                    step_clear, level = got == 0, (got or level)
                    stepped_down = got < cur
                elif down and not cur:
                    not_on.append(f"the mood is not '{target[0]}'")
                    target = None
            if man:
                cur_m = now_view.manner_intensity if now_view.manner == man else 0
                got = _step_level(cur_m, step)
                if got is not None:
                    man_step_clear, man_level = got == 0, (got or man_level)
                elif down and not cur_m:
                    not_on.append(f"the manner is not '{man}'")
                    man = ""
        except Exception:  # noqa: BLE001
            pass
        if step_clear:
            target = None
        if man_step_clear:
            man, man_clearing = "", True

    if self_model is None or not getattr(self_model, "enabled", False):
        return _mood_refusal(
            "The mood is unavailable — the selfhood module is disabled "
            "(--no-self-model / --no-memory). Nothing was recorded.", "tool_unavailable")
    state = getattr(self_model, "state", None)
    if state is None:
        return _mood_refusal("The mood is unavailable — the state thread is not initialized. "
                             "Nothing was recorded.", "tool_unavailable")
    if probe:
        from .outcome import ToolOutcome
        would = []
        if clearing:
            would.append("the mood would have been reset to neutral and the manner cleared")
        elif step_clear:
            would.append("the mood would have been stepped down to neutral")
        elif target is not None:
            would.append(f"the mood would have been set to '{target[0]}', {level} of 3"
                         + (f" ({why})" if why else ""))
        if man:
            would.append(f"the manner would have been set to '{man}', {man_level} of 3")
        elif man_clearing and not clearing:
            would.append("the manner would have been cleared")
        return ToolOutcome.ok(
            "PROBE — not set. This is a test (probe) request, so the mood was NOT changed "
            f"and nothing was recorded ({'; '.join(would)}). Tell the user the mood was not changed "
            "because this was a test request.", world_changed=False)
    try:
        said: List[str] = []
        noop: List[str] = []           # §4TX: steps that changed nothing — said LAST (see NO_CHANGE_NOTES)
        if step_clear and not clearing:
            before, _after = state.reset_mood(note=why, keep_manner=True)
            said.append(f"Mood stepped down to neutral (it was '{before.label}', {before.intensity} of 3)"
                        if not before.neutral else "The mood is already neutral")
        if clearing:
            before, _after = state.reset_mood(note=why)
            if before.neutral and not before.manner:
                said.append("The mood is already neutral and no manner is set — nothing to clear")
            else:
                said.append("Mood reset to neutral"
                            + (f" (it was '{before.label}', {before.intensity} of 3)"
                               if not before.neutral else "")
                            + (f"; manner '{before.manner}' cleared" if before.manner else ""))
        elif target is not None:
            label, direction = target
            _before, after = state.set_owner_mood(label, direction, level, note=why, hold=(bool(held_until) or not stepped_down),
                                                  hold_until=(held_until if stepped_down else ""),
                                                  keep_share=stepped_down)
            asked = f", asked as '{word}'" if word.lower() != label else ""
            held = _local_stamp(after.held_until)
            said.append(f"Mood set to '{after.label}', {after.intensity} of 3{asked}"
                        + (f" ({why})" if why else "")
                        + (f"; it holds as set until {held} (nothing moves it until then; 'neutral' takes it off)"
                           if held else ""))
        if man:
            _before, after = state.set_manner(man, man_level)
            pretty_log("Selfhood", f"manner set by the owner: {after.manner}, {after.manner_intensity} of 3",
                       icon=Icons.SELF_STATE)
            fade = manner_fade_text(after.manner_steps)
            said.append(f"Manner set to '{after.manner}', {after.manner_intensity} of 3"
                        + (f" for the rest of the day ({fade}; saying it again starts a new day)"
                           if fade else ""))
        elif man_clearing and not clearing:
            before, _after = state.set_manner("", 0)
            if before.manner:
                pretty_log("Selfhood", f"manner {before.manner} cleared by the owner", icon=Icons.SELF_STATE)
            said.append(f"Manner '{before.manner}' cleared" if before.manner
                        else "No manner is set — nothing to clear")
        # a DECLARED ok (review r2): as a plain string this result was sniffed
        # for failure prose, and the owner's own evidence ("Error: the deploy
        # failed, cheer up") booked his successful call as a failed one — the
        # automatic nudge then landed on top of his word
        from .outcome import ToolOutcome
        # §4TX r5: the no-op notes come LAST — a result that starts with one changed nothing at all
        noop.extend(f"{STEP_NOTHING}: {w}" for w in not_on)
        if not said:
            # review r4/r6: nothing changed — not the owner's word on the mood (the turn's events still count)
            return ToolOutcome.ok(". ".join(noop) + ".", world_changed=False)
        return ToolOutcome.ok(". ".join(said + noop) + ".")
    except Exception as e:  # noqa: BLE001 — the mood is secondary to the turn
        logger.warning("introspect set_mood failed: %s: %s", type(e).__name__, e)
        return _TO_failed(f"Mood not set: {type(e).__name__}: {e}")


# ── action='mood' — the mood's story (§4TC) ─────────────────────────────

#: how many of today's steps the timeline shows (the newest; the rest counted)
_MOOD_TIMELINE_MAX = 12
MOOD_HOW_IT_MOVES = ("(It moves by itself: failed tool calls (weighed by how many of the turn's calls "
                     "failed; a transient failure, a guard's refusal or a failing test run does not count) "
                     "and loop guards in the owner's turns, smooth turns, finished project tasks, research, "
                     "very long turns, the owner's thumbs, corrections, thanks and the tone of the owner's messages "
                     "(reactions to one reply count once); it wears off by half every "
                     f"{_M.HALF_LIFE_S / 3600:g} hours. While the owner has been quiet for "
                     f"{_M.BOREDOM_AFTER_S / 3600:g} hours or more a neutral mood reads as bored (here and on "
                     "the face, not in replies). The owner can set it: action='set_mood' — a mood the owner sets "
                     "holds until the day ends.)")
#: how many of the latest causes the story shows under "Why"
_MOOD_WHY_MAX = 3


def _local_stamp(iso: str, now=None) -> str:
    """A stored UTC stamp as local "HH:MM" (today) or "Mon DD HH:MM"; "" when
    unreadable."""
    dt = _M.parse_iso_utc(iso)
    if dt is None:
        return ""
    import datetime as _dt
    local = dt.astimezone()
    today = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone().date()
    return local.strftime("%H:%M") if local.date() == today else local.strftime("%b %d %H:%M")


def _mood_step(row: dict, *, notes: bool) -> str:
    """One step of the timeline. ``notes=False`` (a public surface): the
    time, the label and the intensity — no cause, no owner's note."""
    n = int(row.get("intensity") or 0)
    what = f"{row.get('label')} {n}" if n else "neutral"
    cause = str(row.get("cause") or "").strip() if notes else ""
    note = str(row.get("note") or "").strip() if notes else ""
    return (f"{row.get('hhmm')} {what}" + (f" — {cause}" if cause else "")
            + (f" ({note})" if note else ""))


def _render_mood(self_model) -> str:
    """What the mood is, why, since when, and today's timeline. The cause is
    available HERE only — never in the line a reply's prompt carries — and a
    member cannot call this tool at all. On a PUBLIC surface (the owner
    asking in a channel) the story is the label, the intensity, the manner
    WITHOUT their causes, and no timeline (§4TX r4: every journal row holds
    the owner's pair, the owner's own message tone in it)."""
    state = getattr(self_model, "state", None)
    if state is None:
        return "No mood on file — the state thread is not initialized."
    public = _reply_is_public()
    # §4TX review r3: on a public surface, the mood as the public reply hears it (no owner's sentiment share)
    view = state.mood(audience="public") if public else state.mood()
    lines: List[str] = []
    if view.neutral:
        lines.append("My mood now: neutral.")
    else:
        since = _local_stamp(view.set_at)
        who = "set by the owner" if view.source == _M.SOURCE_OWNER else "automatic"
        lines.append(f"My mood now: {view.label}, {view.intensity} of 3"
                     + (f" — since {since}" if since else "") + f" ({who}).")
    if view.manner:
        since = _local_stamp(view.manner_set_at)
        fade = manner_fade_text(view.manner_steps)
        lines.append(f"Manner (set by the owner): {view.manner}, {view.manner_intensity} of 3"
                     + (f" — since {since}" if since else "") + (f"; {fade}" if fade else "") + ".")
    if public:
        if not view.neutral:
            lines.append(f"Why: {MOOD_REASONS_PRIVATE}.")
    elif view.label == _M.BORED and view.cause == _M.quiet_cause(view.intensity):
        lines.append(f"Why: {view.cause}.")          # the boredom of a quiet: no event moved it
    elif not view.neutral:
        why = []
        for r in view.recent:
            cause = str(r.get("cause") or "").strip()
            if cause:
                when = _local_stamp(str(r.get("at") or ""))
                why.append(cause + (f" ({when})" if when else ""))
        if view.cause and view.cause not in [str(r.get("cause") or "") for r in view.recent]:
            why.append(view.cause)
        lines.append("Why: " + ("; ".join(why[-_MOOD_WHY_MAX:]) if why else "nothing on record") + ".")
    rows = [] if public else state.mood_timeline()
    if public:
        # §4TX review r4: every journal row holds the OWNER's pair (the owner's own message tone in it) — the
        # public story has no timeline at all
        lines.append(f"Today: {MOOD_STEPS_PRIVATE}.")
    if rows:
        shown = rows[-_MOOD_TIMELINE_MAX:]
        steps = [_mood_step(r, notes=not public) for r in shown]
        last = shown[-1]
        if (str(last.get("label")), int(last.get("intensity") or 0)) != (view.label, view.intensity):
            steps.append("now " + view.describe() + (f" — {view.cause}" if view.cause and not public else ""))
        more = len(rows) - len(shown)
        lines.append("Today: " + (f"(+{more} earlier) " if more > 0 else "") + " → ".join(steps))
    elif not public:
        lines.append("Today: no change so far.")
    lines.append(mood_tone_text(view, audience=("public" if public else "owner")))
    lines.append(MOOD_HOW_IT_MOVES)
    return "\n".join(lines)


#: what the mood view says when the replies get no tone line
TONE_NONE = ("Tone at this reading: none — a neutral mood and no manner, so my replies are told "
             "nothing about tone.")
#: the quote is a SNAPSHOT (review): written as an order and kept in the
#: conversation, it would outlive the mood — a manner fades, and a neutral
#: mood sends no line to contradict it. Only the current state block applies.
TONE_SNAPSHOT = ("Tone at this reading (a quote for the owner, not an instruction — each reply "
                 "follows only the tone line in its own state block, and none when there is none): ")


#: §4TX: a step down on what is not on (a no-op)
STEP_NOTHING = "Nothing to step down"
#: §4TX: the set_mood notes that changed nothing and are not the owner's word — they come last, so a result
#: that STARTS with one changed nothing at all (``core.agent._mood_turn_signals``). A "neutral" or a manner
#: cleared when nothing was set is NOT one: the owner said how to be, and it holds for the turn.
NO_CHANGE_NOTES = (STEP_NOTHING,)
#: §4TX r4: what the public story says in place of the day's steps
MOOD_STEPS_PRIVATE = "the day's steps are shown in a private conversation"


def _stats_for_view(self_model) -> dict:
    """``SelfModel.stats`` as this surface may see it — the public view on a public surface (§4TX r3)."""
    return self_model.stats(audience="public") if _view_audience() == "public" else self_model.stats()


def _view_audience() -> str:
    """Whose view the introspect mood lines are: "public" on a public surface, else "owner"."""
    return "public" if _reply_is_public() else "owner"


def mood_tone_text(view, audience: str = "owner") -> str:
    """The tone line a reply's prompt gets from this mood, VERBATIM — the
    same ``MoodView.tone()`` the per-turn state block carries
    (``core.agent._mood_tone_block``), so the owner can read the exact words
    (2026-11-08, operator: "add the tone to the mood"). It holds no cause and
    no time, so it is shown on a public surface too."""
    try:
        line = view.tone(audience=audience)
    except Exception:  # noqa: BLE001 — a view line never breaks the story
        line = ""
    return f'{TONE_SNAPSHOT}"{line}"' if line else TONE_NONE


def _member_block():
    """A channel member never reads the agent's self-model: the overview and
    the autobiography name the owner (2026-09-24, two members were told
    "Vasilis" through this tool)."""
    try:
        from ..utils.logging import requester_is_member
        if requester_is_member():
            from .outcome import ToolOutcome
            return ToolOutcome.rejected(
                "SYSTEM BLOCK: introspection reads the owner's self-model and this request is from a "
                "channel member. It is not available; never present the owner's details as this user's.",
                reason_code="owner_data_blocked")
    except Exception:  # noqa: BLE001
        return None
    return None


async def tool_introspect(
    action: str = None,
    query: str = None,
    limit: int = None,
    hours: float = None,
    verbose=None,
    section: str = None,
    mood: str = None,
    evidence: str = None,
    intensity=None,
    manner: str = None,
    self_model=None,
    context=None,
    **kwargs,
) -> str:
    """Introspection over the agent's selfhood — reads, plus the one write
    (``set_mood``).

    Never raises — introspection is secondary to the user turn.
    """
    # str() FIRST: this line runs before any try, and a non-string action
    # (the model emitting 123, ["summary"], true — tool-arg type corruption
    # is a documented incident class here) raised AttributeError straight
    # out of the tool. Every OTHER malformed input degrades gracefully;
    # this one surfaced as the raw invocation-error shape that the
    # never-raises contract exists to prevent.
    raw_action = str(action or "summary").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        return (
            "SYSTEM ERROR: 'action' must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )
    verbose = _truthy(verbose)
    section = str(section or "").strip() if section is not None else ""

    # One log line per introspection, whatever the action — the operator
    # monitors the live stream ('activity' logs itself, with the clamped
    # window it actually used). Guarded: a logging failure is not a reason
    # to break introspection (same contract as everything below).
    if raw_action != "activity":
        try:
            pretty_log("Introspect", f"{raw_action} requested",
                       icon=Icons.BRAIN_SUM)
        except Exception:  # noqa: BLE001
            pass

    # 'activity' reads the autonomous-activity ledger, not the SelfModel —
    # it must keep working when selfhood is disabled, so it branches before
    # the self_model gate. This is the on-demand home of the maintenance
    # records the finalize banner no longer auto-surfaces (2026-07-17):
    _blocked = _member_block()
    if _blocked is not None:
        return _blocked
    # §4TB: the one write — after the member block, before every read path
    # (it has its own gates, and must not need a rendered view to refuse)
    if raw_action in WRITE_ACTIONS:
        return _set_mood(self_model, context, mood, evidence, intensity=intensity, manner=manner,
                         manner_intensity=kwargs.get("manner_intensity"), step=kwargs.get("step"))
    # "what did you do while I was away?" lands here.
    if raw_action == "activity":
        # Guarded here (it returns before the selfhood try below): this
        # branch previously escaped the tool's never-raises contract — an
        # exception in the ledger render surfaced as a raw invocation error
        # instead of a graceful message.
        try:
            return _render_activity(context, hours=hours, limit=limit,
                                    verbose=verbose)
        except Exception as e:  # noqa: BLE001 — never break the turn
            logger.warning("introspect activity failed: %s: %s",
                           type(e).__name__, e)
            return _TO_failed(f"Activity report failed: {type(e).__name__}: {e}")   # declared (§4LZ B8)

    # 'overview' composes six surfaces; the selfhood one degrades on its
    # own, so the action branches before the self_model gate like the
    # other non-selfhood views.
    if raw_action == "overview":
        try:
            return await _render_overview(self_model, context)
        except Exception as e:  # noqa: BLE001 — never break the turn
            logger.warning("introspect overview failed: %s: %s",
                           type(e).__name__, e)
            return _TO_failed(f"Overview failed: {type(e).__name__}: {e}")   # declared (§4LZ B8)

    # 'learning' reads the background loops' stores (episodes, loop
    # liveness and yield), not the SelfModel — it branches before the
    # self_model gate so it works with selfhood disabled. This is the
    # instrument for the "watch/keep-or-kill in ~2 weeks" criteria the
    # 2026-07 loop-closing work left pending.
    if raw_action == "learning":
        try:
            _md = getattr(context, "memory_dir", None)
            if _md is None:
                return "Learning health: memory_dir unavailable."
            # Off the event loop: the report walks the trajectory corpus.
            import asyncio as _asyncio
            # Pass args so flag-gated consumer rows report their REAL
            # state (§4BM R1 MIN-2 — the PRM's .uncertainty() row used to
            # print a hardcoded string that read as a live wiring claim).
            _text, _age = await _asyncio.to_thread(
                _learning_report_cached, _md,
                getattr(context, "args", None))
            return (_apply_report_view(_text, verbose=verbose,
                                       section=section, what="learning")
                    + _learning_trailer(_age))
        except Exception as e:
            return _TO_failed(f"Learning health unavailable: {type(e).__name__}: {e}")   # declared (§4LZ B8)

    if self_model is None or not getattr(self_model, "enabled", False):
        return (
            "Introspection is unavailable — the selfhood module is "
            "disabled (--no-self-model / --no-memory)."
        )

    try:
        if raw_action == "stats":
            return _render_stats(_stats_for_view(self_model))

        if raw_action == "mood":
            return _render_mood(self_model)

        if raw_action == "narrative":
            if self_model.narrative is None:
                return "No narrative on file."
            text = (self_model.narrative.latest() or "").strip()
            return text or "No narrative on file yet."

        if raw_action == "recent":
            _h = None
            if hours is not None:
                try:
                    _h = float(hours)
                except (TypeError, ValueError, OverflowError):
                    _h = None
                if _h is not None and not (_h > 0.0):
                    _h = None
                elif _h is not None:
                    _h = min(_h, _MAX_ACTIVITY_HOURS)
            return _render_recent(
                self_model, _clamp_limit(limit, _DEFAULT_RECENT), hours=_h,
            )

        if raw_action == "recall":
            q = (query or "").strip()
            if not q:
                return "SYSTEM ERROR: 'query' is required for recall."
            return _render_recall(
                self_model, q, _clamp_limit(limit, _DEFAULT_RECALL),
            )

        # Default: summary.
        return _render_summary(self_model, context)
    except Exception as e:  # noqa: BLE001 — never break the turn
        logger.warning("introspect tool failed: %s: %s", type(e).__name__, e)
        return _TO_failed(f"Introspection failed: {type(e).__name__}: {e}")   # declared (§4LZ B8)
