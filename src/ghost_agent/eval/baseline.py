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


def freeze_baseline(
    result: SuiteResult,
    path: Union[str, Path],
    *,
    provenance: Dict[str, Any] = None,
) -> Path:
    """Atomically write `result` to `path` as JSON. Returns the resolved path.

    Writes to a sibling tempfile then renames so a crashed write can't
    leave a half-baseline on disk.

    ``provenance`` — HOW this baseline was produced (runner kind, model,
    suite, whether it is a real capability baseline). Stored under
    ``_provenance`` so a later ``compare`` can refuse to certify a run
    whose baseline was frozen with the stub, or whose model/suite differ.
    A baseline with no provenance (older file) is treated as untrusted.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["_provenance"] = dict(provenance or {})
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(p)
    return p


def load_baseline(path: Union[str, Path]) -> SuiteResult:
    p = Path(path)
    raw = json.loads(p.read_text())
    return SuiteResult.from_dict(raw)


def load_baseline_provenance(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the ``_provenance`` block a baseline was frozen with (empty
    dict for an older, provenance-less file)."""
    try:
        raw = json.loads(Path(path).read_text())
    except Exception:
        return {}
    prov = raw.get("_provenance")
    return prov if isinstance(prov, dict) else {}


def baseline_trust_warnings(
    baseline_provenance: Dict[str, Any],
    current_provenance: Dict[str, Any],
) -> list:
    """Return a list of human-readable trust warnings for a compare. An
    EMPTY list means the comparison is a trustworthy capability verdict.

    Flags: a stub runner on either side (the metric is meaningless — the
    exact 'stub compare always passes' footgun), a missing/old baseline
    provenance, and a model/suite mismatch (comparing incomparable runs).
    """
    warnings: list = []
    bp = baseline_provenance or {}
    cp = current_provenance or {}
    if not bp:
        warnings.append(
            "baseline has NO provenance (frozen by an older tool) — cannot "
            "confirm it is a real capability baseline.")
    if bp.get("runner") == "stub":
        warnings.append(
            "baseline was frozen with the STUB runner — it is a pipeline "
            "check, NOT a capability baseline; re-freeze with --runner http.")
    if cp.get("runner") == "stub":
        warnings.append(
            "this run used the STUB runner — the comparison is meaningless "
            "(a stub-vs-stub compare trivially passes).")
    if bp.get("model") and cp.get("model") and bp["model"] != cp["model"]:
        warnings.append(
            f"model mismatch: baseline={bp['model']!r} vs current={cp['model']!r} "
            "— comparing incomparable runs.")
    if bp.get("suite") and cp.get("suite") and bp["suite"] != cp["suite"]:
        warnings.append(
            f"suite mismatch: baseline={bp['suite']!r} vs current={cp['suite']!r}.")
    return warnings


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
