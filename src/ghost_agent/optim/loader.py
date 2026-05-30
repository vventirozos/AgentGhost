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
    return value if value else default


def clear_cache() -> None:
    """Drop the in-process cache so the next lookup re-reads disk (e.g. after
    an offline GEPA retrain produced new tuned files)."""
    _CACHE.clear()
