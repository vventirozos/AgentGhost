"""Eval metrics data types.

Pure dataclasses; no I/O, no async. Used by suite.py to shape results
and by baseline.py to snapshot/diff them.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TaskResult:
    """Outcome of running a single EvalTask end-to-end."""

    task_id: str
    category: str           # "template" | "regression" | "curated"
    cluster: Optional[str]  # challenge cluster, None for non-template
    tier: Optional[str]     # difficulty tier, None for non-template
    passed: bool
    duration_s: float
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    tokens_used: int = 0
    final_output: str = ""
    failure_reason: str = ""
    # Free-form per-task metadata; stays out of the main schema so
    # adding a metric doesn't break baseline comparison.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskResult":
        known = {k: d.get(k) for k in (
            "task_id", "category", "cluster", "tier", "passed",
            "duration_s", "steps", "tool_calls", "tool_errors",
            "tokens_used", "final_output", "failure_reason",
        )}
        extra = dict(d.get("extra") or {})
        return cls(**known, extra=extra)


@dataclass
class SuiteResult:
    """Aggregate of all TaskResults from one EvalSuite run."""

    suite_name: str
    timestamp: str                 # ISO8601
    ghost_version: str
    results: List[TaskResult] = field(default_factory=list)
    # `summary` is recomputed on construction / via aggregate(); stored
    # for serialization so a baseline file reads without reloading raw
    # results.
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Honour the "recomputed on construction" contract above: if a caller
        # builds a SuiteResult from results without an explicit summary, derive
        # it now. Otherwise `compare_to_baseline` reads an empty `.summary`,
        # `diff_summaries` treats the missing pass_rate as 0.0, and a whole run
        # is spuriously flagged as a regression/improvement.
        if not self.summary and self.results:
            self.summary = aggregate(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "ghost_version": self.ghost_version,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuiteResult":
        return cls(
            suite_name=d["suite_name"],
            timestamp=d["timestamp"],
            ghost_version=d["ghost_version"],
            results=[TaskResult.from_dict(r) for r in d.get("results", [])],
            summary=dict(d.get("summary") or {}),
        )


def _safe_mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def aggregate(results: List[TaskResult]) -> Dict[str, Any]:
    """Compute summary stats from a list of TaskResults.

    Kept deliberately flat (no nested objects) so baseline diffing can
    walk the keys without recursion.
    """
    if not results:
        return {
            "n": 0,
            "pass_rate": 0.0,
            "mean_duration_s": 0.0,
            "mean_steps": 0.0,
            "mean_tool_calls": 0.0,
            "total_tokens": 0,
            "by_category": {},
            "by_cluster": {},
            "by_tier": {},
        }

    by_cat: Dict[str, List[TaskResult]] = {}
    by_cluster: Dict[str, List[TaskResult]] = {}
    by_tier: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
        if r.cluster:
            by_cluster.setdefault(r.cluster, []).append(r)
        if r.tier:
            by_tier.setdefault(r.tier, []).append(r)

    def _bucket(bs: List[TaskResult]) -> Dict[str, float]:
        return {
            "n": len(bs),
            "pass_rate": _safe_mean([1.0 if r.passed else 0.0 for r in bs]),
            "mean_duration_s": _safe_mean([r.duration_s for r in bs]),
            "mean_steps": _safe_mean([float(r.steps) for r in bs]),
        }

    return {
        "n": len(results),
        "pass_rate": _safe_mean([1.0 if r.passed else 0.0 for r in results]),
        "mean_duration_s": _safe_mean([r.duration_s for r in results]),
        "mean_steps": _safe_mean([float(r.steps) for r in results]),
        "mean_tool_calls": _safe_mean([float(r.tool_calls) for r in results]),
        "total_tokens": int(sum(r.tokens_used for r in results)),
        "mean_tool_errors": _safe_mean([float(r.tool_errors) for r in results]),
        "by_category": {k: _bucket(v) for k, v in by_cat.items()},
        "by_cluster": {k: _bucket(v) for k, v in by_cluster.items()},
        "by_tier": {k: _bucket(v) for k, v in by_tier.items()},
    }


def diff_summaries(baseline: Dict[str, Any], current: Dict[str, Any],
                   pass_rate_tolerance: float = 0.02) -> Dict[str, Any]:
    """Return {regressions, improvements, unchanged_keys} between two
    summaries produced by `aggregate`. Used by baseline.compare.

    A regression is pass_rate dropping by more than `pass_rate_tolerance`
    on the top-level or any bucket. Duration/step changes are reported
    but do NOT count as regressions — they're often environmental.
    """
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []

    def _cmp(path: str, b: Dict[str, Any], c: Dict[str, Any]) -> None:
        b_pr = float(b.get("pass_rate", 0.0))
        c_pr = float(c.get("pass_rate", 0.0))
        delta = c_pr - b_pr
        if delta < -pass_rate_tolerance:
            regressions.append({
                "path": path, "metric": "pass_rate",
                "baseline": b_pr, "current": c_pr, "delta": delta,
            })
        elif delta > pass_rate_tolerance:
            improvements.append({
                "path": path, "metric": "pass_rate",
                "baseline": b_pr, "current": c_pr, "delta": delta,
            })

    _cmp("suite", baseline, current)
    for bucket_name in ("by_category", "by_cluster", "by_tier"):
        b_map = baseline.get(bucket_name) or {}
        c_map = current.get(bucket_name) or {}
        for key in sorted(set(b_map) | set(c_map)):
            _cmp(f"{bucket_name}.{key}", b_map.get(key, {}), c_map.get(key, {}))

    return {
        "regressions": regressions,
        "improvements": improvements,
        "pass_rate_delta": float(current.get("pass_rate", 0.0)) - float(baseline.get("pass_rate", 0.0)),
    }
