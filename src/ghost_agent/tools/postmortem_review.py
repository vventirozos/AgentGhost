"""postmortem tool — read-only view over the post-mortem defect queue.

The post-mortem engine (biological phase 2.5c, ``reflection/postmortem.py``)
files durable, classified defect reports when the agent fails badly: a
behavioural lesson, a configuration gap, or a code defect — the last
carrying an LLM-proposed reproducing test + diff. This tool is how the
operator (or the agent itself, when asked "what have you found broken in
yourself?") reads that queue. All actions are read-only; nothing here
applies a patch or mutates a report.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("GhostAgent")

_VALID_ACTIONS = frozenset({"pending", "list", "show", "stats"})
_MAX_LIMIT = 25


def _clamp(value, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, _MAX_LIMIT)


def _fmt_short(rep) -> str:
    sev = f"{rep.severity:.2f}"
    title = rep.title or "(untitled)"
    return f"  [{rep.status}] {rep.category} · sev {sev} · {rep.id[:8]} — {title}"


def _fmt_full(rep) -> str:
    lines = [
        f"Defect {rep.id[:12]} [{rep.status}]",
        f"  category : {rep.category}",
        f"  severity : {rep.severity:.2f}",
        f"  title    : {rep.title or '(untitled)'}",
        f"  evidence : {rep.evidence}",
        f"  root cause: {rep.root_cause}",
    ]
    if rep.category == "configuration" and rep.config_change:
        lines.append(f"  config change: {rep.config_change}")
    if rep.category == "code_defect":
        if rep.code_fix:
            lines.append(f"  code fix: {rep.code_fix}")
        if rep.proposed_test:
            lines.append("  --- proposed reproducing test (NOT applied) ---")
            lines.append(rep.proposed_test)
        if rep.proposed_patch:
            lines.append("  --- proposed patch (NOT applied) ---")
            lines.append(rep.proposed_patch)
    if rep.category == "behavioural" and rep.lesson:
        lines.append(f"  lesson: {rep.lesson.get('solution', '')}")
    if rep.source_trajectory_ids:
        lines.append(f"  from trajectory: {', '.join(t[:8] for t in rep.source_trajectory_ids)}")
    return "\n".join(lines)


async def tool_postmortem(
    action: str = "pending",
    *,
    defect_queue=None,
    defect_id: Optional[str] = None,
    limit: Optional[int] = None,
    **_ignored,
) -> str:
    """Read the post-mortem defect queue.

    Actions:
      * ``pending`` (default) — open defects, most severe first.
      * ``list`` — all defects (any status).
      * ``show`` — full detail of one defect (needs ``defect_id``).
      * ``stats`` — counts by category and status.
    """
    if defect_queue is None:
        return (
            "Post-mortem engine is not enabled. Start the agent with "
            "--postmortem to file defect reports on bad runs."
        )

    act = (action or "pending").strip().lower()
    if act not in _VALID_ACTIONS:
        return f"Unknown action '{action}'. Valid: {', '.join(sorted(_VALID_ACTIONS))}."

    try:
        if act == "stats":
            reports = defect_queue.all()
            if not reports:
                return "Post-mortem queue is empty — no bad runs have been triaged yet."
            by_cat: dict = {}
            by_status: dict = {}
            for r in reports:
                by_cat[r.category] = by_cat.get(r.category, 0) + 1
                by_status[r.status] = by_status.get(r.status, 0) + 1
            cat = ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
            sts = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
            return f"Defect queue: {len(reports)} total\n  by category: {cat}\n  by status:   {sts}"

        if act == "show":
            if not defect_id:
                return "action='show' needs a defect_id (use action='pending' to find one)."
            for r in defect_queue.all():
                if r.id.startswith(defect_id) or r.id == defect_id:
                    return _fmt_full(r)
            return f"No defect found with id starting '{defect_id}'."

        reports = defect_queue.pending() if act == "pending" else defect_queue.all()
        if not reports:
            label = "open" if act == "pending" else "filed"
            return f"No {label} post-mortem defects."
        n = _clamp(limit, 10)
        shown = reports[:n]
        header = (
            f"{len(reports)} {'open' if act == 'pending' else 'total'} defect(s)"
            f"{f', showing {n}' if len(reports) > n else ''}:"
        )
        body = "\n".join(_fmt_short(r) for r in shown)
        hint = "\nUse action='show' with a defect_id for full detail + any proposed patch."
        return f"{header}\n{body}{hint}"
    except Exception as e:
        logger.warning("postmortem tool failed: %s", e)
        return f"Could not read the defect queue: {e}"
