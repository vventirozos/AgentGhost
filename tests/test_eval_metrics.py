"""Tests for ghost_agent.eval.metrics — pure dataclass + aggregator."""

from ghost_agent.eval.metrics import (
    TaskResult, SuiteResult, aggregate, diff_summaries,
)


def _tr(task_id, category="template", cluster="sql", tier="basic",
        passed=True, duration_s=1.0, steps=3, tool_calls=2,
        tool_errors=0, tokens_used=100):
    return TaskResult(
        task_id=task_id, category=category, cluster=cluster, tier=tier,
        passed=passed, duration_s=duration_s, steps=steps,
        tool_calls=tool_calls, tool_errors=tool_errors, tokens_used=tokens_used,
    )


def test_task_result_roundtrip():
    r = _tr("t1")
    d = r.to_dict()
    r2 = TaskResult.from_dict(d)
    assert r2 == r


def test_task_result_extra_field_preserved():
    r = TaskResult(
        task_id="x", category="curated", cluster=None, tier=None,
        passed=True, duration_s=0.5, extra={"router_confidence": 0.9},
    )
    d = r.to_dict()
    assert d["extra"] == {"router_confidence": 0.9}
    r2 = TaskResult.from_dict(d)
    assert r2.extra == {"router_confidence": 0.9}


def test_aggregate_empty():
    s = aggregate([])
    assert s["n"] == 0
    assert s["pass_rate"] == 0.0
    assert s["by_cluster"] == {}


def test_aggregate_basic():
    results = [
        _tr("a", passed=True, duration_s=1.0, steps=4, tool_calls=3, tool_errors=0, tokens_used=100),
        _tr("b", passed=False, duration_s=2.0, steps=6, tool_calls=5, tool_errors=2, tokens_used=200),
        _tr("c", passed=True, duration_s=0.5, steps=2, tool_calls=1, tool_errors=0, tokens_used=50),
    ]
    s = aggregate(results)
    assert s["n"] == 3
    assert abs(s["pass_rate"] - (2 / 3)) < 1e-9
    assert s["total_tokens"] == 350
    assert abs(s["mean_tool_errors"] - (2 / 3)) < 1e-9


def test_aggregate_buckets_by_cluster_and_tier():
    results = [
        _tr("a", cluster="sql", tier="basic", passed=True),
        _tr("b", cluster="sql", tier="basic", passed=False),
        _tr("c", cluster="bash", tier="advanced", passed=True),
    ]
    s = aggregate(results)
    assert s["by_cluster"]["sql"]["n"] == 2
    assert s["by_cluster"]["bash"]["n"] == 1
    assert s["by_tier"]["basic"]["n"] == 2
    assert s["by_tier"]["advanced"]["n"] == 1
    assert s["by_cluster"]["sql"]["pass_rate"] == 0.5


def test_aggregate_skips_missing_cluster_and_tier():
    results = [
        _tr("a", cluster=None, tier=None, passed=True),
        _tr("b", cluster="sql", tier="basic", passed=True),
    ]
    s = aggregate(results)
    assert "sql" in s["by_cluster"]
    assert None not in s["by_cluster"]
    assert "basic" in s["by_tier"]


def test_suite_result_json_roundtrip():
    results = [_tr("a"), _tr("b", passed=False)]
    sr = SuiteResult(
        suite_name="test", timestamp="2026-01-01T00:00:00Z",
        ghost_version="0.1.0", results=results, summary=aggregate(results),
    )
    import json
    restored = SuiteResult.from_dict(json.loads(sr.to_json()))
    assert restored.suite_name == "test"
    assert len(restored.results) == 2
    assert restored.summary["n"] == 2


def test_diff_summaries_detects_regression():
    baseline_results = [_tr(f"b{i}", passed=True) for i in range(10)]
    current_results = [_tr(f"c{i}", passed=(i < 5)) for i in range(10)]
    b = aggregate(baseline_results)
    c = aggregate(current_results)
    d = diff_summaries(b, c, pass_rate_tolerance=0.02)
    assert d["pass_rate_delta"] < -0.4
    suite_regressions = [r for r in d["regressions"] if r["path"] == "suite"]
    assert suite_regressions, "top-level pass_rate regression should be reported"


def test_diff_summaries_detects_improvement():
    baseline_results = [_tr(f"b{i}", passed=(i < 3)) for i in range(10)]
    current_results = [_tr(f"c{i}", passed=True) for i in range(10)]
    b = aggregate(baseline_results)
    c = aggregate(current_results)
    d = diff_summaries(b, c)
    assert d["pass_rate_delta"] > 0.4
    assert any(i["path"] == "suite" for i in d["improvements"])


def test_diff_summaries_tolerance_masks_noise():
    baseline_results = [_tr(f"b{i}", passed=(i < 50)) for i in range(100)]
    current_results = [_tr(f"c{i}", passed=(i < 49)) for i in range(100)]
    b = aggregate(baseline_results)
    c = aggregate(current_results)
    d = diff_summaries(b, c, pass_rate_tolerance=0.05)
    assert not any(r["path"] == "suite" for r in d["regressions"]), (
        "1% pass_rate drop should not count as regression with 5% tolerance"
    )
