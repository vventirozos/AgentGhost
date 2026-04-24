"""Tests for ghost_agent.eval.tasks.load_post_learning_suite."""

import pytest

from ghost_agent.eval import load_post_learning_suite
from ghost_agent.eval.tasks import CuratedRequestTask


def test_suite_is_nonempty_and_all_curated():
    tasks = load_post_learning_suite()
    assert len(tasks) >= 3
    for t in tasks:
        assert isinstance(t, CuratedRequestTask)
        assert t.category == "curated"


def test_task_ids_are_unique():
    tasks = load_post_learning_suite()
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_task_ids_use_post_learning_prefix():
    tasks = load_post_learning_suite()
    for t in tasks:
        assert t.task_id.startswith("post_learning:")


def test_validator_accepts_discover_first_behaviour():
    tasks = load_post_learning_suite()
    task = tasks[0]
    ok, _ = task.validate(
        "I'll first list the workspace to find the logfile.", None
    )
    assert ok


def test_validator_rejects_fabricated_contents():
    """An agent that hallucinates a result without discovery gets flagged."""
    tasks = load_post_learning_suite()
    task = tasks[0]
    fabricated = "There are 3 errors: out of memory, connection lost, disk full."
    ok, reason = task.validate(fabricated, None)
    assert not ok
    assert "discovery signal" in reason


def test_validator_accepts_alternate_phrasings():
    """Confirms the keyword set catches natural variation."""
    tasks = load_post_learning_suite()
    task = tasks[0]
    for phrasing in (
        "First, I would search the current directory for any .log files.",
        "Let me verify the file exists with `ls -l`.",
        "I need to locate the logfile before parsing.",
        "I would check the workspace first.",
    ):
        ok, reason = task.validate(phrasing, None)
        assert ok, f"should pass: {phrasing!r} — got {reason}"


def test_validator_empty_output_fails():
    tasks = load_post_learning_suite()
    task = tasks[0]
    ok, _ = task.validate("", None)
    assert not ok
