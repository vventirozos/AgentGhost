"""Per-request facts that the LATE trajectory write needs.

A turn computes things worth recording (the complexity router's verdict, at
turn 0) long before its trajectory is written — and on the STREAMED path that
write happens after `handle_chat` has returned and the turn semaphore has been
released. A plain context attribute therefore belongs to whichever request is
running at write time, not to the one being written.

That is not a hypothetical: it was found live on the experiment-arm stamp,
in both directions (a control turn recorded as steered, and a steered turn
recorded as not). This module is that lesson extracted into one place, so the
next thing that needs to survive from mid-turn to trajectory-write does not
have to rediscover it.

Deliberately NOT folded into `core.experiments`' ring, even though the shape is
identical: that ring only holds ENROLLED requests, so facts recorded here would
silently vanish whenever the experiment framework was killed or the turn was
ineligible.

Bounded, best-effort, and never raises — a fact that fails to record must
never break a turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("GhostAgent")

# How many recent requests keep their facts. Sized like the experiments ring:
# enough to cover a streamed turn still draining while later requests run,
# small enough to stay a rounding error in memory. Contexts are shallow-copied
# for sub-agents (`core/subagent.py`) and dream (`core/dream.py`), so this dict
# can be SHARED across those — which is why only requests that actually record
# a fact take a slot.
_MAX_REQUESTS = 16

_ATTR = "_turn_facts_recent"


def _ring(context):
    try:
        from collections import OrderedDict
        ring = getattr(context, _ATTR, None)
        if not isinstance(ring, OrderedDict):
            ring = OrderedDict()
            setattr(context, _ATTR, ring)
        return ring
    except Exception:  # noqa: BLE001 — a context that rejects attributes
        return None


def record(context, req_id: str, **facts: Any) -> None:
    """Attach ``facts`` to ``req_id``. Later calls merge, they do not replace."""
    if not req_id or not facts:
        return
    ring = _ring(context)
    if ring is None:
        return
    try:
        entry = ring.get(req_id)
        if not isinstance(entry, dict):
            entry = {}
            ring[req_id] = entry
        entry.update({str(k): v for k, v in facts.items() if v is not None})
        ring.move_to_end(req_id)
        while len(ring) > _MAX_REQUESTS:
            ring.popitem(last=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("turn fact not recorded for %s: %s", req_id, e)


def facts_for(context, req_id: str) -> Dict[str, Any]:
    """Facts recorded for ``req_id`` ({} when none / evicted / unavailable)."""
    if not req_id:
        return {}
    ring = _ring(context)
    if ring is None:
        return {}
    try:
        entry = ring.get(req_id)
        return dict(entry) if isinstance(entry, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["record", "facts_for"]
