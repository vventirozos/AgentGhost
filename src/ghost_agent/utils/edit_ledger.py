"""Append-only ledger for file-EDIT outcomes — what the tolerance ladder costs.

Why this exists (§4KB, 2026-09-23). `file_system operation='replace'` resolves
a SEARCH block through a four-rung ladder — exact → whitespace-flexible →
fuzzy → anchor — and `_locate_block` already computes WHICH rung matched. Until
now the caller threw that away: the rung reached the model as an advisory
sentence ("rescued by tolerant matching") and reached the operator not at all.

So the one question that decides whether the ladder is an asset or a liability
could not be asked: **how often does an applied edit rest on a rung below
`exact`, and how often is that edit corrected immediately afterwards?** The
07-14 and 07-19 corruption incidents were both tolerant matching landing in the
wrong region and reporting SUCCESS. Every guard added since checks the RESULT
(marker leak, syntax regression); none of them can see a fuzzy match that lands
somewhere syntactically valid and semantically wrong. That class is only
visible in aggregate, which means it needs a ledger.

Design constraints, all deliberate:

  * **Never raises.** A telemetry write that can fail a turn is worse than no
    telemetry. Every public function traps and returns a falsy value.
  * **Append-only JSONL**, one row per `tool_replace_text` call, at
    ``<GHOST_HOME>/system/edits/ladder.jsonl`` — the `rubric_shadow.jsonl`
    idiom, and the same join keys (`req_id`), because a measurement that cannot
    be joined to the turn it came from can never be evaluated.
  * **One row per CALL, not per rung.** A block-form call applying three
    envelopes is one row carrying three strategies; that keeps "how many edits
    did this turn apply" answerable by counting rows.
  * **Records rejections too.** The rejected calls are half the signal: a
    strictness change that trades applied-but-wrong for rejected-and-retried
    is only legible if both are counted. `applied=False` rows carry the
    `reason` code.

Read it with `scripts/edit_ladder_report.py`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Relative to GHOST_HOME. Mirrors `core.rubric_grader.SHADOW_REL`.
LEDGER_REL = "system/edits/ladder.jsonl"

#: The ladder's rung vocabulary, in descending confidence. `fuzzy` and
#: `anchor` carry a suffix at runtime (`fuzzy:87%`, `anchor:12-40`), so
#: consumers compare with `rung_family`, never with `==`.
RUNG_EXACT = "exact"
RUNG_FLEXIBLE = "flexible"
RUNG_FUZZY = "fuzzy"
RUNG_ANCHOR = "anchor"

#: Every rung the ladder can report. A rung absent from this tuple is a bug in
#: the ledger, not in the ladder — `tests/test_edit_ladder_ledger.py` walks
#: `_locate_block`'s own vocabulary against it.
RUNG_FAMILIES = (RUNG_EXACT, RUNG_FLEXIBLE, RUNG_FUZZY, RUNG_ANCHOR)

#: Rungs that did NOT byte-match the file. These are the ones whose cost this
#: ledger exists to price.
TOLERANT_FAMILIES = (RUNG_FLEXIBLE, RUNG_FUZZY, RUNG_ANCHOR)


def rung_family(strategy: str) -> str:
    """The family of a rung string: ``"fuzzy:87%"`` → ``"fuzzy"``.

    Returns ``""`` for an empty/None strategy (a rejected call that never
    reached the ladder) and passes an unrecognised string through unchanged
    rather than guessing — an unknown rung must show up in the report as
    itself, not be silently folded into a known family.
    """
    s = str(strategy or "")
    if not s:
        return ""
    return s.split(":", 1)[0]


def ledger_path(home: Optional[Path] = None) -> Path:
    base = Path(home) if home else Path(
        os.environ.get("GHOST_HOME") or Path.home() / "Data" / "AI" / "Data")
    return base / LEDGER_REL


def record_edit(*, path: str, op: str = "replace", applied: bool = False,
                strategies: Optional[List[str]] = None, reason: str = "",
                search_len: int = 0, file_len: int = 0,
                blocks_total: int = 0, blocks_applied: int = 0,
                req_id: str = "", home: Optional[Path] = None,
                ts: Optional[float] = None) -> bool:
    """Append one edit row. Never raises; returns whether it landed.

    ``strategies`` is the list of rungs that APPLIED within this call (one
    entry per envelope for the block form, one for the two-argument form,
    empty for a rejection). It is stored as a list even in the single-edit
    case so consumers never branch on shape.
    """
    try:
        p = ledger_path(home)
        p.parent.mkdir(parents=True, exist_ok=True)
        strat = [str(s) for s in (strategies or [])]
        row: Dict[str, Any] = {
            "ts": float(ts if ts is not None else time.time()),
            "req_id": str(req_id or ""),
            "path": str(path or ""),
            "op": str(op or "replace"),
            "applied": bool(applied),
            "strategies": strat,
            "families": [rung_family(s) for s in strat],
            "reason": str(reason or ""),
            "search_len": int(search_len or 0),
            "file_len": int(file_len or 0),
            "blocks_total": int(blocks_total or 0),
            "blocks_applied": int(blocks_applied or 0),
        }
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:                                # noqa: BLE001
        logger.debug("edit ledger write failed: %s", exc)
        return False


def read_ledger(home: Optional[Path] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Rows oldest-first, malformed lines skipped. Never raises.

    ``limit`` keeps the LAST ``limit`` rows (the recent window is what every
    consumer wants), matching `core.learning_health._load_jsonl`.
    """
    try:
        p = ledger_path(home)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:                           # noqa: BLE001
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        if limit is not None and limit >= 0:
            return out[-limit:]
        return out
    except Exception as exc:                                # noqa: BLE001
        logger.debug("edit ledger read failed: %s", exc)
        return []
