"""Loader for GEPA-optimized prompts.

`scripts/run_gepa.py` (via `optim/run_gepa.py`) writes tuned instructions to
`$GHOST_HOME/system/optim/<signature_name>.json` with the field
``optimized_instruction``. Nothing read those back, so the optimization was
write-only and never reached inference. This module closes that loop: it reads
the tuned instruction for a signature at prompt-build time, falling back to the
hand-written baseline when no tuned file exists.

Results are cached per process (the offline GEPA run produces files between
sessions, not mid-turn). Call ``clear_cache()`` to force a reload after a
retrain.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("GhostAgent")

# signature_name -> tuned instruction str (or None when absent/invalid)
_CACHE: Dict[str, Optional[str]] = {}

# Activation telemetry: how often each signature's prompt-build actually
# APPLIED a tuned instruction vs fell back to the hand-written baseline.
# The dominant failure mode of harness additions is the component silently
# never firing (this exact loop was write-only once already) — so activation
# is counted at the only chokepoint every read-site goes through, and
# surfaced via learning-health. In-process counters: they reset on restart,
# which is fine — "zero applies since boot despite a tuned file on disk"
# is precisely the signal we are after.
_APPLIED_COUNTS: Dict[str, int] = {}
_FALLBACK_COUNTS: Dict[str, int] = {}
# Loaded-then-refused by the read-site's own validator. See `note_rejected`.
_REJECTED_COUNTS: Dict[str, int] = {}


def _optim_dir() -> Path:
    """`$GHOST_HOME/system/optim` — the SAME path scripts/run_gepa.py writes to
    (default GHOST_HOME `~/ghost_llamacpp`)."""
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    return base / "system" / "optim"


def tuned_instruction(signature_name: str, default: str = "") -> str:
    """Return the GEPA-`optimized_instruction` for ``signature_name``, or
    ``default`` (the hand-written baseline) when no valid tuned file exists.
    Never raises — a missing/corrupt file silently yields the baseline."""
    if not signature_name:
        return default
    if signature_name in _CACHE:
        cached = _CACHE[signature_name]
        if cached:
            _APPLIED_COUNTS[signature_name] = _APPLIED_COUNTS.get(signature_name, 0) + 1
        else:
            _FALLBACK_COUNTS[signature_name] = _FALLBACK_COUNTS.get(signature_name, 0) + 1
        return cached if cached else default

    value: Optional[str] = None
    try:
        path = _optim_dir() / f"{signature_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            opt = data.get("optimized_instruction")
            if isinstance(opt, str) and opt.strip():
                value = opt.strip()
                logger.info(
                    "GEPA: loaded tuned instruction for '%s' (%d chars)",
                    signature_name, len(value),
                )
    except Exception as e:
        logger.debug("GEPA tuned_instruction('%s') load failed: %s", signature_name, e)

    _CACHE[signature_name] = value
    if value:
        _APPLIED_COUNTS[signature_name] = _APPLIED_COUNTS.get(signature_name, 0) + 1
    else:
        _FALLBACK_COUNTS[signature_name] = _FALLBACK_COUNTS.get(signature_name, 0) + 1
    return value if value else default


def note_rejected(signature_name: str, reason: str = "") -> None:
    """A read-site LOADED a tuned instruction and then REFUSED it.

    `tuned_instruction` counts what it hands out; it cannot see what the
    caller does next. Every read-site has its own validator (per-tool caps,
    an aggregate-inflation ceiling, placeholder checks), so a tuned artifact
    could be loaded, counted as "applied", and then dropped — leaving the
    ONE instrument built to catch silent inoperativeness reporting
    `applied: 1, fallback: 0` while nothing whatsoever reached the model.
    Measured: 6 over-inflated tool artifacts + 1 broken verifier template
    read as fully healthy.

    Read-sites call this when they reject; the count is moved out of
    `applied` so the telemetry describes reality.
    """
    if not signature_name:
        return
    _REJECTED_COUNTS[signature_name] = _REJECTED_COUNTS.get(signature_name, 0) + 1
    if _APPLIED_COUNTS.get(signature_name):
        _APPLIED_COUNTS[signature_name] -= 1
    if reason:
        logger.debug("GEPA: '%s' loaded but rejected by its read-site (%s)",
                     signature_name, reason)


def activation_stats() -> Dict[str, Dict[str, int]]:
    """Per-signature tuned-vs-baseline application counts since process
    start: ``{signature: {"applied": n, "fallback": m, "rejected": r}}``.

    ``applied`` counts artifacts that a read-site actually USED (loads minus
    read-site rejections); ``rejected`` counts the ones it loaded and then
    refused. A signature with a tuned file on disk and zero ``applied``
    means the read-site is not firing — the defect class this counter exists
    to catch — and a non-zero ``rejected`` says the artifact is reaching the
    read-site but failing its validator, which is a different problem with a
    different fix."""
    names = set(_APPLIED_COUNTS) | set(_FALLBACK_COUNTS) | set(_REJECTED_COUNTS)
    return {
        n: {
            "applied": max(0, _APPLIED_COUNTS.get(n, 0)),
            "fallback": _FALLBACK_COUNTS.get(n, 0),
            "rejected": _REJECTED_COUNTS.get(n, 0),
        }
        for n in sorted(names)
    }


def clear_cache() -> None:
    """Drop the in-process cache so the next lookup re-reads disk (e.g. after
    an offline GEPA retrain produced new tuned files).

    ⚠ NEVER call on a live agent process: a mid-session reload changes the
    prompt bytes under the KV stable-prefix pin and forces a full re-prime
    every turn thereafter. Retrain offline, then deploy via restart.
    Activation counters are deliberately NOT cleared — they describe the
    process, not the cache."""
    _CACHE.clear()
