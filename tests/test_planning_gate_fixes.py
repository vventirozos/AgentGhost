"""Regression tests for the 2026-07-20 planning.py cohort fixes.

The plan-postcondition gate (agent.py) went live 2026-07-20, making
previously masked planner-output malformations reachable:

  1. ``load_from_json`` coercion — string ``postconditions`` graded
     per-char, ``"status": null`` crashed ``.upper()``.
  2. HUMAN_GATE on the interactive close path — textually graded to
     FAILED instead of parking at NEEDS_USER
     (contract: project_safety.enforce_human_gate).
  3. Dangling alternative id consumed and parent BLOCKED even with
     valid alternatives still queued.
  4. Dead ReplanBridge — ``ProjectPlan`` had no ``request_revision``,
     so revisions landed on a throwaway in-memory tree.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.planning import (
    ProjectPlan, TaskStatus, TaskTree, DependencyType,
)
from ghost_agent.memory.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


# ----------------------------------------------------------- 1. load_from_json coercion

def test_string_postconditions_become_single_postcondition():
    tree = TaskTree()
    tree.load_from_json({
        "id": "root", "description": "goal",
        "postconditions": "file exists on disk",
    })
    assert tree.nodes["root"].postconditions == ["file exists on disk"]


def test_string_postconditions_not_graded_per_char():
    tree = TaskTree()
    tree.load_from_json({
        "id": "root", "description": "goal",
        "postconditions": "zqxj wvkp mfgh",  # tokens absent from any reply
    })
    root = tree.nodes["root"]
    root.result_summary = "something unrelated"
    unsat = tree._check_postconditions(root)
    assert unsat == ["zqxj wvkp mfgh"]  # ONE unsatisfied item, not one per char


def test_null_status_defaults_to_pending_without_crash():
    tree = TaskTree()
    tree.load_from_json({"id": "r", "description": "goal", "status": None})
    assert tree.nodes["r"].status == TaskStatus.PENDING


def test_malformed_scalar_fields_all_coerced():
    tree = TaskTree()
    tree.load_from_json({
        "id": "r", "description": None,
        "status": 3, "dependency_type": None,
        "postconditions": None, "alternatives": "alt-1",
        "depends_on": None,
    })
    node = tree.nodes["r"]
    assert node.description == "Unknown Task"
    assert node.status == TaskStatus.PENDING
    assert node.dependency_type == DependencyType.ALL
    assert node.postconditions == []
    assert node.alternatives == ["alt-1"]
    assert node.depends_on == []


def test_list_postconditions_drop_null_and_blank_entries():
    tree = TaskTree()
    tree.load_from_json({
        "id": "r", "description": "goal",
        "postconditions": [None, "  ", "real one"],
    })
    assert tree.nodes["r"].postconditions == ["real one"]


def test_checker_coerces_direct_string_postconditions():
    """Belt to the load_from_json braces: a node whose field was set to a
    bare string outside load_from_json must still grade as ONE
    postcondition, not per-char garbage in failure_reason."""
    tree = TaskTree()
    tid = tree.add_task("t")
    tree.nodes[tid].postconditions = "zqxj wvkp"
    tree.update_status(tid, TaskStatus.DONE, result="unrelated output")
    node = tree.nodes[tid]
    assert node.status == TaskStatus.FAILED
    assert "zqxj wvkp" in node.failure_reason
    assert ";" not in node.failure_reason  # single unsat, not per-char list


# ----------------------------------------------------------- 2. HUMAN_GATE close path

def test_human_gate_close_parks_needs_user_not_failed():
    tree = TaskTree()
    root = tree.add_task("root")
    tid = tree.add_task("deploy", parent_id=root,
                        postconditions=["HUMAN_GATE: cto approval"])
    tree.update_status(tid, TaskStatus.DONE, result="deployed to staging")
    assert tree.nodes[tid].status == TaskStatus.NEEDS_USER
    # No spurious FAILED cascade up the tree
    assert tree.nodes[root].status == TaskStatus.PENDING


def test_human_gate_echoed_text_does_not_satisfy_gate():
    """The gate must not be satisfiable by result-text similarity."""
    tree = TaskTree()
    tid = tree.add_task("deploy",
                        postconditions=["HUMAN_GATE: cto approval"])
    tree.update_status(tid, TaskStatus.DONE, result="HUMAN_GATE: cto approval")
    assert tree.nodes[tid].status == TaskStatus.NEEDS_USER


def test_human_gate_second_close_is_the_approval():
    tree = TaskTree()
    tid = tree.add_task("deploy",
                        postconditions=["HUMAN_GATE: cto approval"])
    tree.update_status(tid, TaskStatus.DONE, result="deployed")
    assert tree.nodes[tid].status == TaskStatus.NEEDS_USER
    # The user was asked; the close path being taken again is the approval.
    tree.update_status(tid, TaskStatus.DONE, result="deployed")
    assert tree.nodes[tid].status == TaskStatus.DONE


def test_human_gate_explicit_approval_flag_closes_directly():
    tree = TaskTree()
    tid = tree.add_task("deploy", postconditions=["HUMAN_GATE: ship it"])
    tree.update_status(tid, TaskStatus.DONE, result="shipped",
                       human_approved=True)
    assert tree.nodes[tid].status == TaskStatus.DONE


def test_human_gate_park_records_reason_when_no_result():
    tree = TaskTree()
    tid = tree.add_task("deploy", postconditions=["HUMAN_GATE: cto approval"])
    tree.update_status(tid, TaskStatus.DONE)
    assert tree.nodes[tid].status == TaskStatus.NEEDS_USER
    assert "cto approval" in tree.nodes[tid].result_summary


def test_human_gate_other_postconditions_still_enforced_after_approval():
    tree = TaskTree()
    tid = tree.add_task("deploy", postconditions=[
        "HUMAN_GATE: cto approval", "zqxj wvkp responds",
    ])
    tree.update_status(tid, TaskStatus.DONE, result="unrelated",
                       human_approved=True)
    node = tree.nodes[tid]
    assert node.status == TaskStatus.FAILED
    assert "zqxj wvkp" in node.failure_reason
    # The gate itself is never textually graded into the unsat list
    assert "HUMAN_GATE" not in node.failure_reason


def test_postcondition_check_skips_human_gate_postconditions():
    """A HUMAN_GATE postcondition is never textually graded into the
    unsatisfied list (§4RS: was pinned through the removed root gate)."""
    tree = TaskTree()
    tid = tree.add_task("goal", postconditions=["HUMAN_GATE: sign-off"])
    node = tree.nodes[tid]
    node.result_summary = "any reply"
    assert tree._check_postconditions(node) == []


def test_project_plan_forwards_human_approved(store):
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("deploy",
                        postconditions=["HUMAN_GATE: operator approval"])
    plan.update_status(tid, TaskStatus.DONE, result="done")
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    plan.update_status(tid, TaskStatus.DONE, result="done",
                       human_approved=True)
    assert store.get_task(tid)["status"] == "DONE"


# ----------------------------------------------------------- 3. dangling alternatives

def test_dangling_then_valid_alternative_resolves_the_valid_one():
    tree = TaskTree()
    root = tree.add_task("root")
    c1 = tree.add_task("primary", parent_id=root)
    alt = tree.add_task("fallback")
    tree.nodes[root].alternatives = ["ghost-id", alt]
    tree.update_status(c1, TaskStatus.FAILED, failure_reason="boom")
    assert tree.nodes[root].status != TaskStatus.BLOCKED
    assert tree.nodes[alt].status == TaskStatus.READY
    assert alt in tree.nodes[root].children
    assert tree.nodes[alt].parent_id == root
    assert tree.nodes[root].alternatives == []  # dangling id consumed too


def test_all_dangling_alternatives_still_block_parent():
    tree = TaskTree()
    root = tree.add_task("root")
    c1 = tree.add_task("primary", parent_id=root)
    tree.nodes[root].alternatives = ["ghost-1", "ghost-2"]
    tree.update_status(c1, TaskStatus.FAILED, failure_reason="boom")
    assert tree.nodes[root].status == TaskStatus.BLOCKED
    assert tree.nodes[root].alternatives == []


# 4. persisted request_revision: removed with the ReplanBridge, its only caller (§4PT)
