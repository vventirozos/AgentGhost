"""Learned RRF intent→source weights — load-if-present capability.

The memory bus fuses the retrieval tiers with a per-intent source-weight
matrix (`bus.MemoryBus._INTENT_WEIGHTS`) that is hand-tuned magic. This
module lets an OFFLINE-fitted matrix supersede those defaults at runtime,
using the same posture as the PRM / router / GEPA checkpoints: when a
fitted `weights.json` exists it is loaded and used; when absent, the bus
keeps its hand-tuned defaults — **zero behaviour change**. That fail-safe
default matters here more than anywhere else, because the fusion weights
sit on the hot retrieval path that feeds every turn; an ungrounded auto-
fit must never silently degrade recall.

`fit_intent_weights` derives a matrix from ``(intent, source, success)``
observations: a source that correlates with successful turns under a
given intent is up-weighted. Cells below a sample floor keep the base
weight, and every weight is clamped to a sane band so a thin or biased
sample can neither zero out nor explode a source.

Pure stdlib; JSON; fail-safe load (any problem → ``None`` → defaults).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger("GhostAgent")

SCHEMA_VERSION = "ghost.rrf.weights.v1"

# Mirror of bus.MemoryBus._INTENT_WEIGHTS — the safe fallback AND the
# anchor a fit starts from (cells with too little data keep these).
DEFAULT_INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "factual":    {"graph": 2.0, "vector": 1.0, "skill": 0.5, "episodic": 0.3, "session": 0.8},
    "procedural": {"graph": 0.5, "vector": 1.0, "skill": 2.0, "episodic": 1.5, "session": 0.5},
    "contextual": {"graph": 1.0, "vector": 1.5, "skill": 1.0, "episodic": 1.0, "session": 1.2},
}

WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.0


def _clamp(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return WEIGHT_MIN if v < WEIGHT_MIN else WEIGHT_MAX if v > WEIGHT_MAX else v


def load_intent_weights(path) -> Optional[Dict[str, Dict[str, float]]]:
    """Read a fitted weight matrix, or ``None`` when absent / corrupt /
    wrong-schema (so the caller falls back to the hand-tuned defaults)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("rrf weights load failed: %s", e)
        return None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA_VERSION:
        return None
    raw = d.get("weights")
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Dict[str, float]] = {}
    for intent, sw in raw.items():
        if not isinstance(sw, dict):
            continue
        cell = {}
        for src, val in sw.items():
            try:
                cell[str(src)] = _clamp(float(val))
            except (TypeError, ValueError):
                continue
        if cell:
            out[str(intent)] = cell
    return out or None


def fit_intent_weights(
    observations: Iterable[Tuple[str, str, bool]],
    *,
    base: Optional[Dict[str, Dict[str, float]]] = None,
    min_obs_per_cell: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Fit an intent→source weight matrix from ``(intent, source,
    success)`` observations.

    Per (intent, source) cell with ≥ ``min_obs_per_cell`` samples, the
    weight is set from the success rate mapped onto ``[WEIGHT_MIN,
    WEIGHT_MAX]`` (rate 0 → MIN, 0.5 → ~1.0, 1 → MAX). Cells below the
    floor keep the ``base`` weight, so a thin sample never overrides a
    sensible default. Always returns a full matrix (base merged)."""
    base = base or DEFAULT_INTENT_WEIGHTS
    agg: Dict[Tuple[str, str], list] = {}
    for intent, source, success in observations:
        key = (str(intent), str(source))
        cell = agg.setdefault(key, [0, 0])
        cell[0] += 1
        cell[1] += 1 if success else 0
    out: Dict[str, Dict[str, float]] = {i: dict(sw) for i, sw in base.items()}
    for (intent, source), (n, wins) in agg.items():
        if n < max(1, int(min_obs_per_cell)):
            continue
        rate = wins / n if n else 0.5
        w = _clamp(WEIGHT_MIN + rate * (WEIGHT_MAX - WEIGHT_MIN))
        out.setdefault(intent, {})[source] = round(w, 3)
    return out


def save_intent_weights(path, weights: Dict[str, Dict[str, float]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"schema": SCHEMA_VERSION, "weights": weights}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


__all__ = [
    "DEFAULT_INTENT_WEIGHTS",
    "load_intent_weights",
    "fit_intent_weights",
    "save_intent_weights",
    "SCHEMA_VERSION",
]
