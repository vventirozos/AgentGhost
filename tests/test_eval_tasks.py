"""Tests for ghost_agent.eval.tasks."""

import pytest

from ghost_agent.eval.tasks import (
    EvalTask,
    ChallengeTemplateTask,
    CuratedRequestTask,
    RegressionProbeTask,
    load_default_suite,
)


def test_base_task_default_validator():
    t = EvalTask(task_id="x", category="other", prompt="hi")
    ok, _ = t.validate("something")
    assert ok
    bad, reason = t.validate("")
    assert not bad
    assert reason


def test_challenge_template_task_dict_output():
    t = ChallengeTemplateTask(
        task_id="t1", category="", prompt="solve", cluster="sql", tier="basic"
    )
    assert t.category == "template"
    ok, _ = t.validate({"passed": True})
    assert ok
    bad, reason = t.validate({"passed": False, "reason": "validator exited 1"})
    assert not bad
    assert "validator" in reason


def test_challenge_template_task_string_fallback():
    t = ChallengeTemplateTask(task_id="t1", category="", prompt="solve")
    ok, _ = t.validate("some non-empty output")
    assert ok
    bad, _ = t.validate("")
    assert not bad


def test_curated_keyword_validator():
    t = CuratedRequestTask(
        task_id="c1", category="", prompt="what is 7*6?", validator=["42"]
    )
    ok, _ = t.validate("the answer is 42, obviously")
    assert ok
    bad, reason = t.validate("forty two")
    assert not bad
    assert "no keyword" in reason


def test_curated_callable_validator():
    t = CuratedRequestTask(
        task_id="c1", category="", prompt="say hi",
        validator=lambda out, _ctx: ("hi" in out.lower(), ""),
    )
    ok, _ = t.validate("hi there")
    assert ok
    bad, _ = t.validate("goodbye")
    assert not bad


def test_curated_callable_that_raises_is_caught():
    def bad_validator(_o, _c):
        raise RuntimeError("boom")
    t = CuratedRequestTask(
        task_id="c1", category="", prompt="", validator=bad_validator,
    )
    ok, reason = t.validate("anything")
    assert not ok
    assert "boom" in reason


def test_curated_none_validator_passes_nonempty():
    t = CuratedRequestTask(task_id="c", category="", prompt="")
    ok, _ = t.validate("anything")
    assert ok
    bad, _ = t.validate("")
    assert not bad


def test_regression_probe_callable_required():
    t = RegressionProbeTask(task_id="p", category="", prompt="probe")
    ok, reason = t.validate(None, None)
    assert not ok
    assert "missing callable" in reason


def test_regression_probe_happy_path():
    t = RegressionProbeTask(
        task_id="p", category="", prompt="probe",
        validator=lambda _ctx: (True, ""),
    )
    ok, _ = t.validate(None, None)
    assert ok


def test_regression_probe_tuple_failure_surfaces_reason():
    t = RegressionProbeTask(
        task_id="p", category="", prompt="probe",
        validator=lambda _ctx: (False, "invariant broken"),
    )
    ok, reason = t.validate(None, None)
    assert not ok
    assert reason == "invariant broken"


def test_regression_probe_exception_is_captured():
    def raising(_ctx):
        raise RuntimeError("kaboom")
    t = RegressionProbeTask(
        task_id="p", category="", prompt="", validator=raising,
    )
    ok, reason = t.validate(None, None)
    assert not ok
    assert "kaboom" in reason


def test_load_default_suite_contains_each_category():
    tasks = load_default_suite()
    cats = {t.category for t in tasks}
    assert "regression" in cats
    assert "curated" in cats
    # Templates are optional — only assert if the bank loaded.
    if any(t.category == "template" for t in tasks):
        # Expect at least one template per cluster of the bank.
        clusters = {t.cluster for t in tasks if t.category == "template"}
        assert len(clusters) >= 1


def test_load_default_suite_toggle_flags():
    tasks = load_default_suite(
        include_templates=False, include_curated=False, include_capability=False)
    cats = {t.category for t in tasks}
    assert cats == {"regression"}


def test_regression_probes_all_have_callable_validators():
    tasks = load_default_suite(include_templates=False, include_curated=False)
    for t in tasks:
        assert callable(t.validator), f"{t.task_id} should carry a callable validator"


def test_regression_probe_cooldown_invariant_passes():
    from ghost_agent.eval.tasks import _load_regression_probes
    probes = _load_regression_probes()
    probe = next(p for p in probes if "cooldown" in p.task_id)
    ok, reason = probe.validate(None, None)
    assert ok, f"cooldown probe failed: {reason}"
