"""Baseline snapshot & comparison.

Freeze a SuiteResult to disk; later, load it and diff against a fresh
run. The file is plain JSON on purpose — human-readable, git-diffable,
and doesn't need a schema migration path when we add metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .metrics import SuiteResult, diff_summaries


def freeze_baseline(result: SuiteResult, path: Union[str, Path]) -> Path:
    """Atomically write `result` to `path` as JSON. Returns the resolved path.

    Writes to a sibling tempfile then renames so a crashed write can't
    leave a half-baseline on disk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(result.to_json(indent=2))
    tmp.replace(p)
    return p


def load_baseline(path: Union[str, Path]) -> SuiteResult:
    p = Path(path)
    raw = json.loads(p.read_text())
    return SuiteResult.from_dict(raw)


def compare_to_baseline(
    baseline: Union[SuiteResult, str, Path, Dict[str, Any]],
    current: SuiteResult,
    *,
    pass_rate_tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Compare `current` against a previously frozen baseline.

    Accepts the baseline as a SuiteResult, a file path, or a pre-loaded
    dict — whichever the caller has on hand.
    """
    if isinstance(baseline, SuiteResult):
        b_summary = baseline.summary
    elif isinstance(baseline, dict):
        b_summary = baseline.get("summary") or {}
    else:
        b_summary = load_baseline(baseline).summary

    return diff_summaries(b_summary, current.summary, pass_rate_tolerance=pass_rate_tolerance)
