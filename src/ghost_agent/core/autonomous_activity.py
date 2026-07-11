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
    "scheduled_task": "scheduled task",
    "open_questions": "open questions",
    "project": "project",
    "service": "service",
    "job": "background job",
}

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
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
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
                           exclude_phases=DIGEST_EXCLUDED_PHASES) -> str:
    """Render unseen records as a short markdown header block. Empty string
    when nothing digest-worthy. Notify-severity items lead (stable order
    otherwise)."""
    items = [r for r in (records or [])
             if r.summary and r.phase not in (exclude_phases or ())]
    if not items:
        return ""
    items.sort(key=lambda r: 0 if r.severity == SEVERITY_NOTIFY else 1)
    lines = ["**Background activity while you were away:**"]
    for r in items[:max_items]:
        label = _PHASE_LABELS.get(r.phase, r.phase)
        # Per-item clamp keeps the whole block comfortably under the
        # 1500-char leading-banner bound `_strip_leading_banners` peels —
        # a longer block would defeat the correction-fingerprint peel and
        # resurrect the stash/lookup mismatch fixed 2026-07-07.
        s = r.summary if len(r.summary) <= 140 else r.summary[:139] + "…"
        lines.append(f"  - [{label}] {s}")
    extra = len(items) - max_items
    if extra > 0:
        lines.append(f"  - …and {extra} more")
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
    "render_activity_digest",
    "summarize_turn_content", "record_scheduled_result",
    "get_activity_log",
]
