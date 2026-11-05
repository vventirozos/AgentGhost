"""introspect tool — read-only introspection over the agent's selfhood.

The selfhood wake-up prefix is OFF on the request path
(``_SELFHOOD_PREFIX_ENABLED`` in core/agent.py, journal §3), so NOTHING
about mood, diary, principles or open questions reaches the model except
through this tool. That makes routing load-bearing: the SELF SURFACE rule
in the system prompt sends "how are you / how's things / what did you do
while I was away" here, and ``overview`` is the one-call answer. (Until
2026-09-05 this docstring claimed the prefix was spliced into every
prompt; live, 28 of 48 such questions were answered with no tool at all
and 9 went to ``system_utility`` — the machine, not the self.)

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
import threading
import time
from typing import Dict, List, Optional, Tuple

from ..utils.logging import Icons, pretty_log


def _TO_failed(text):
    """A tool-level exception, DECLARED failed (§4LZ B8: these plain
    strings were booked OK by the turn loop and the corpus)."""
    from .outcome import ToolOutcome
    return ToolOutcome.failed(text, reason_code="tool_exception")


logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"summary", "stats", "narrative", "recent",
                            "recall", "activity", "learning",
                            "overview"})    # §4NN: 'experiments' removed

_DEFAULT_RECENT = 5
_DEFAULT_RECALL = 5
_MAX_LIMIT = 25
_SUMMARY_RECENT_N = 5

# Repeats collapse at render time (2026-09-05): the live log holds 68
# same-request-same-minute pairs, and recall showed identical lines back to
# back. The renderers fetch this multiple of `limit`, group identical
# requests into one "(×N)" line, then trim to `limit`.
_COLLAPSE_OVERSCAN = 2
# Open questions / unfinished threads rendered in summary + overview.
_CARRIED_MAX = 5

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
_OVERVIEW_LEARNING_HEADS = ("LESSONS:",)   # §4PJ: COMPETENCE/CALIBRATION removed
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
    lines.append(f"Open questions: {stats.get('open_questions', 0)}")
    lines.append(f"Unfinished threads: {stats.get('unfinished_threads', 0)}")
    mood = stats.get("last_mood") or ""
    if mood:
        # Age + provenance so an old label reads as history, not as a
        # current state (the 23-day "curious" problem).
        from ..selfhood.mood import (
            age_seconds, describe_mood_provenance, mood_is_stale,
        )
        set_at = stats.get("last_mood_set_at") or ""
        prov = describe_mood_provenance(
            stats.get("last_mood_source") or "self", age_seconds(set_at),
        )
        stale = " — STALE" if mood_is_stale(set_at) else ""
        # The evidence is the falsifiable half ("my last 5 verdict-bearing
        # turns all passed"); a bare label is the vibe derive_mood replaced.
        ev = (stats.get("last_mood_evidence") or "").strip()
        lines.append(f"Last noted mood: {mood}"
                     + (f" — {ev}" if ev else "")
                     + f" ({prov}{stale})")
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


def _render_carried_state(self_model, *, indent: str = "") -> str:
    """Open questions and unfinished threads AS TEXT. Until 2026-09-05
    every surface rendered only their counts — with the wake-up prefix
    off, the agent could not read its own open questions at all."""
    state = getattr(self_model, "state", None)
    if state is None:
        return ""
    lines: List[str] = []
    try:
        qs = list(state.open_questions() or [])
        th = list(state.unfinished_threads() or [])
    except Exception as e:  # noqa: BLE001 — a summary must render without them
        logger.debug("introspect carried-state read skipped: %s", e)
        return ""
    if qs:
        lines.append(f"{indent}Open questions I'm still carrying:")
        for q in qs[-_CARRIED_MAX:]:
            age = _exp_age(getattr(q, "opened_at", ""))
            lines.append(f"{indent}  - {q.text}"
                         + (f" (opened {age})" if age else ""))
    if th:
        lines.append(f"{indent}Threads I left unfinished:")
        for t in th[-_CARRIED_MAX:]:
            age = _exp_age(getattr(t, "opened_at", ""))
            lines.append(f"{indent}  - {t.descriptor}"
                         + (f" (since {age})" if age else ""))
    return "\n".join(lines)


def _render_principles(self_model) -> str:
    """Principles with their age — "noted 59d ago" lets a reader weigh a
    principle the way the profile's as_of lets it weigh a fact."""
    try:
        principles = list(self_model.principles() or [])
    except Exception as e:  # noqa: BLE001
        logger.debug("introspect principles read skipped: %s", e)
        try:
            return (self_model.principles_text() or "").strip()
        except Exception:  # noqa: BLE001
            return ""
    lines: List[str] = []
    for p in principles:
        text = (getattr(p, "text", "") or "").strip()
        if not text:
            continue
        age = _exp_age(getattr(p, "added_at", ""))
        lines.append(f"- {text}" + (f" (noted {age})" if age else ""))
    return "\n".join(lines)


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
    stats = self_model.stats()
    parts: List[str] = []
    parts.append("Who I am — a snapshot of my self-state:")
    parts.append(_render_stats(stats))

    image = _self_image(context)
    if image:
        parts.append("\n" + image)

    carried = _render_carried_state(self_model)
    if carried:
        parts.append("\n" + carried)

    narrative = ""
    if self_model.narrative is not None:
        narrative = (self_model.narrative.latest() or "").strip()
    if narrative:
        parts.append("\nMy running first-person diary:")
        parts.append(narrative)

    # Operating principles — the normative substrate. "Tell me about
    # yourself" is exactly where these belong.
    principles = _render_principles(self_model)
    if principles:
        parts.append("\nMy operating principles:")
        parts.append(principles)

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
# general tool-result cap, so "how are your lessons doing?" used to inject
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
# lessons, skills, workspace); "how are you" made it call
# nothing. One bounded call, six labelled surfaces, and every surface that
# cannot render SAYS so — absence is not neutral (2026-08 review, R3).


def _failed_listed(recent) -> int:
    """§4QJ: how many listed turns (a collapsed repeat counts once) are
    [failed]. r2: compared with the mood's ``set_at`` it never fired — the
    mood is re-derived (and re-stamped) after every turn, and a late verdict
    back-fills "failed" without moving the turn's timestamp."""
    return sum(1 for exp, _count in (recent or ())
               if str(getattr(exp, "outcome", "") or "") == "failed")


def _overview_selfhood(self_model, context=None) -> str:
    if self_model is None or not getattr(self_model, "enabled", False):
        return ("How I am: selfhood is disabled (--no-self-model / "
                "--no-memory) — no mood, open questions or diary to report.")
    try:
        from ..selfhood.mood import (
            MOOD_STALE_AFTER_HOURS, age_seconds, describe_mood_provenance,
            mood_is_stale,
        )
        stats = self_model.stats()
        lines = ["How I am:"]
        mood = stats.get("last_mood") or ""
        set_at = stats.get("last_mood_set_at") or ""
        recent = _recent_collapsed(self_model, _OVERVIEW_RECENT_N)
        if mood and not mood_is_stale(set_at):
            prov = describe_mood_provenance(
                stats.get("last_mood_source") or "self", age_seconds(set_at))
            ev = (stats.get("last_mood_evidence") or "").strip()
            # §4QJ: the mood is derived when a turn is CAPTURED, and a verdict
            # that arrives later (a late verifier, a back-filled outcome)
            # never re-derives it — "satisfied — my last 5 turns all passed"
            # sat above a [failed] turn 47 min old and the reply said "all my
            # recent turns smooth" (r3: the earlier "counts only turns with a
            # verdict" was false — a failed outcome IS a verdict)
            _bad = _failed_listed(recent) if mood == "satisfied" else 0
            lines.append(f"  mood: {mood}" + (f" — {ev}" if ev else "")
                         + f" ({prov})"
                         + (f" — BUT {_bad} of the turns listed below FAILED; the mood was set before "
                            "that verdict arrived, so do not say all recent turns went well"
                            if _bad else ""))
        elif mood:
            lines.append(f"  mood: no fresh reading (last noted '{mood}', "
                         f"now STALE — older than "
                         f"{MOOD_STALE_AFTER_HOURS:.0f}h)")
        else:
            lines.append("  mood: no reading on file")
        image = _self_image(context, indent="  ")
        if image:
            lines.append(image)
        carried = _render_carried_state(self_model, indent="  ")
        lines.append(carried if carried else
                     "  nothing carried over: no open questions or "
                     "unfinished threads")
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
        return "Learning: memory_dir unavailable."
    try:
        text, age = await _asyncio.to_thread(
            _learning_report_cached, _md, getattr(context, "args", None), allow_stale=True)
    except Exception as e:  # noqa: BLE001
        return f"Learning: unavailable ({type(e).__name__}: {e})"
    heads = [l.strip() for l in text.split("\n")
             if l.strip().startswith(_OVERVIEW_LEARNING_HEADS)]
    warns = [l.strip() for l in text.split("\n")
             if l.lstrip().startswith("⚠")]
    # §4NH review: warnings first — the overview's shrink cuts this section
    # from its END, and the ⚠ lines are what a "how's things" must not lose
    picked = (warns + heads)[:_OVERVIEW_LEARNING_MAX_LINES]
    if not picked:
        return "Learning: the health report rendered no headline lines."
    return ("Learning (headline — action='learning' for the rest):\n"
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


def _overview_health(context) -> str:
    """§4QJ: what the agent already knows about its own parts, read from
    the circuit breaker of its own calls — instant, no live test. The
    live checks (internet, Tor, Docker) stay behind check_health."""
    try:
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
        nodes = [("/".join(ns) + (" (one server)" if len(ns) > 1 else "")
                  + f" {_node_reading(status.get(u))}") for u, ns in names.items()]
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
    self_model=None,
    context=None,
    **kwargs,
) -> str:
    """Read-only introspection over the agent's selfhood.

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

    # 'learning' reads the learning-loop stores (lessons, episodes, loop
    # liveness), not the SelfModel — it branches before the
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
            return _render_stats(self_model.stats())

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
