"""Tests for the Hierarchical Goal Decomposition upgrades to planning.py."""

import pytest

from ghost_agent.core.planning import TaskTree, TaskNode, TaskStatus, DependencyType


class TestDependencyType:
    def test_enum_values(self):
        assert DependencyType.ALL.value == "ALL"
        assert DependencyType.ANY.value == "ANY"
        assert DependencyType.BEST.value == "BEST"


class TestTaskNodeExtended:
    def test_defaults(self):
        node = TaskNode(id="t1", description="test")
        assert node.dependency_type == DependencyType.ALL
        assert node.alternatives == []
        assert node.postconditions == []
        assert node.estimated_cost == 0.0

    def test_to_dict_includes_new_fields(self):
        node = TaskNode(
            id="t1", description="test",
            dependency_type=DependencyType.ANY,
            alternatives=["t2"],
            postconditions=["output exists"],
        )
        d = node.to_dict()
        assert d["dependency_type"] == "ANY"
        assert d["alternatives"] == ["t2"]
        assert d["postconditions"] == ["output exists"]


class TestAnyDependency:
    """Test the ANY dependency type — first successful child satisfies the parent."""

    def test_any_parent_completes_on_first_child_done(self):
        tree = TaskTree()
        parent_id = tree.add_task("Get data", dependency_type=DependencyType.ANY)
        child1_id = tree.add_task("Try API", parent_id=parent_id)
        child2_id = tree.add_task("Try scraping", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.DONE, result="API data loaded")
        assert tree.nodes[parent_id].status == TaskStatus.DONE
        assert tree.nodes[parent_id].result_summary == "API data loaded"

    def test_any_parent_not_blocked_if_one_child_fails(self):
        tree = TaskTree()
        parent_id = tree.add_task("Get data", dependency_type=DependencyType.ANY)
        child1_id = tree.add_task("Try API", parent_id=parent_id)
        child2_id = tree.add_task("Try scraping", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.FAILED, failure_reason="API down")
        # Parent should NOT be blocked yet — child2 is still pending
        assert tree.nodes[parent_id].status != TaskStatus.BLOCKED

    def test_any_parent_blocked_when_all_children_fail(self):
        tree = TaskTree()
        parent_id = tree.add_task("Get data", dependency_type=DependencyType.ANY)
        child1_id = tree.add_task("Try API", parent_id=parent_id)
        child2_id = tree.add_task("Try scraping", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.FAILED, failure_reason="API down")
        tree.update_status(child2_id, TaskStatus.FAILED, failure_reason="Scraping blocked")
        assert tree.nodes[parent_id].status == TaskStatus.BLOCKED


class TestBestDependency:
    """Test the BEST dependency type — all children run, pick the best result."""

    def test_best_waits_for_all_children(self):
        tree = TaskTree()
        parent_id = tree.add_task("Analyze", dependency_type=DependencyType.BEST)
        child1_id = tree.add_task("Method A", parent_id=parent_id)
        child2_id = tree.add_task("Method B", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.DONE, result="Short answer")
        # Parent should NOT be done yet — child2 hasn't finished
        assert tree.nodes[parent_id].status != TaskStatus.DONE

    def test_best_picks_longest_result(self):
        tree = TaskTree()
        parent_id = tree.add_task("Analyze", dependency_type=DependencyType.BEST)
        child1_id = tree.add_task("Method A", parent_id=parent_id)
        child2_id = tree.add_task("Method B", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.DONE, result="Short")
        tree.update_status(child2_id, TaskStatus.DONE, result="Much longer and more detailed analysis")
        assert tree.nodes[parent_id].status == TaskStatus.DONE
        assert "longer" in tree.nodes[parent_id].result_summary

    def test_best_handles_mixed_results(self):
        tree = TaskTree()
        parent_id = tree.add_task("Analyze", dependency_type=DependencyType.BEST)
        child1_id = tree.add_task("Method A", parent_id=parent_id)
        child2_id = tree.add_task("Method B", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.FAILED, failure_reason="Error")
        tree.update_status(child2_id, TaskStatus.DONE, result="Good result")
        assert tree.nodes[parent_id].status == TaskStatus.DONE
        assert tree.nodes[parent_id].result_summary == "Good result"

    def test_best_fails_when_all_fail(self):
        tree = TaskTree()
        parent_id = tree.add_task("Analyze", dependency_type=DependencyType.BEST)
        child1_id = tree.add_task("Method A", parent_id=parent_id)
        child2_id = tree.add_task("Method B", parent_id=parent_id)

        tree.update_status(child1_id, TaskStatus.FAILED)
        tree.update_status(child2_id, TaskStatus.FAILED)
        assert tree.nodes[parent_id].status == TaskStatus.FAILED


class TestAlternatives:
    """Test the alternatives mechanism for fallback task activation."""

    def test_alternative_activated_on_child_failure(self):
        tree = TaskTree()
        parent_id = tree.add_task("Process data")
        primary_id = tree.add_task("Parse CSV", parent_id=parent_id)
        fallback_id = tree.add_task("Parse as JSON", parent_id=parent_id, status=TaskStatus.PENDING)

        # Set the fallback as an alternative
        tree.nodes[parent_id].alternatives = [fallback_id]

        tree.update_status(primary_id, TaskStatus.FAILED, failure_reason="Not a CSV")
        # Parent should NOT be blocked — alternative should be activated
        assert tree.nodes[parent_id].status != TaskStatus.BLOCKED
        assert tree.nodes[fallback_id].status == TaskStatus.READY


class TestLoadFromJsonExtended:
    def test_loads_dependency_type(self):
        tree = TaskTree()
        tree.load_from_json({
            "id": "root",
            "description": "Main goal",
            "status": "IN_PROGRESS",
            "dependency_type": "ANY",
            "children": [
                {"id": "c1", "description": "Option 1", "status": "READY"},
                {"id": "c2", "description": "Option 2", "status": "READY"},
            ],
        })
        assert tree.nodes["root"].dependency_type == DependencyType.ANY

    def test_loads_alternatives(self):
        tree = TaskTree()
        tree.load_from_json({
            "id": "root",
            "description": "Main",
            "status": "IN_PROGRESS",
            "alternatives": ["alt1"],
            "children": [],
        })
        assert tree.nodes["root"].alternatives == ["alt1"]

    def test_loads_postconditions(self):
        tree = TaskTree()
        tree.load_from_json({
            "id": "root",
            "description": "Main",
            "status": "IN_PROGRESS",
            "postconditions": ["output file exists"],
            "children": [],
        })
        assert tree.nodes["root"].postconditions == ["output file exists"]

    def test_unknown_dependency_type_defaults_to_all(self):
        tree = TaskTree()
        tree.load_from_json({
            "id": "root",
            "description": "Main",
            "status": "IN_PROGRESS",
            "dependency_type": "INVALID",
            "children": [],
        })
        assert tree.nodes["root"].dependency_type == DependencyType.ALL


class TestToJsonExtended:
    def test_serializes_new_fields(self):
        tree = TaskTree()
        tree.add_task("Main", dependency_type=DependencyType.ANY,
                       alternatives=["alt1"], postconditions=["check"])
        data = tree.to_json()
        assert data["dependency_type"] == "ANY"
        assert data["alternatives"] == ["alt1"]
        assert data["postconditions"] == ["check"]

    def test_omits_defaults(self):
        tree = TaskTree()
        tree.add_task("Simple task")
        data = tree.to_json()
        # DependencyType.ALL is the default — should be omitted
        assert "dependency_type" not in data
        assert "alternatives" not in data


class TestAddTaskExtended:
    def test_add_task_with_dependency_type(self):
        tree = TaskTree()
        task_id = tree.add_task("parent", dependency_type=DependencyType.BEST)
        assert tree.nodes[task_id].dependency_type == DependencyType.BEST

    def test_add_task_with_postconditions(self):
        tree = TaskTree()
        task_id = tree.add_task("task", postconditions=["file exists", "no errors"])
        assert tree.nodes[task_id].postconditions == ["file exists", "no errors"]


class TestPostconditionEnforcement:
    """Postconditions: a task marked DONE whose postconditions are not met
    flips back to FAILED, and any declared alternatives get a shot."""

    def test_done_flips_to_failed_when_postcondition_unmet(self):
        tree = TaskTree()
        task_id = tree.add_task(
            "write the report",
            postconditions=["report.md file exists and contains summary"],
        )
        # Result summary has nothing to do with the postcondition tokens.
        tree.update_status(task_id, TaskStatus.DONE, result="greetings")
        assert tree.nodes[task_id].status == TaskStatus.FAILED
        assert "Postcondition" in tree.nodes[task_id].failure_reason

    def test_done_stays_done_when_postcondition_matches(self):
        tree = TaskTree()
        task_id = tree.add_task(
            "write the report",
            postconditions=["report contains summary"],
        )
        tree.update_status(
            task_id,
            TaskStatus.DONE,
            result="wrote report.md with a summary of findings",
        )
        assert tree.nodes[task_id].status == TaskStatus.DONE

    def test_done_stays_done_with_no_postconditions(self):
        tree = TaskTree()
        task_id = tree.add_task("task with no postcondition")
        tree.update_status(task_id, TaskStatus.DONE, result="anything")
        assert tree.nodes[task_id].status == TaskStatus.DONE

    def test_custom_checker_callable_overrides_textual_match(self):
        tree = TaskTree()
        tree.postcondition_checker = lambda pc, node: True  # always satisfied
        task_id = tree.add_task(
            "task", postconditions=["something obviously not in result"],
        )
        tree.update_status(task_id, TaskStatus.DONE, result="unrelated")
        assert tree.nodes[task_id].status == TaskStatus.DONE

    def test_failed_postcondition_triggers_alternative(self):
        """The enforcement should propagate to the parent's failure path so
        alternatives get promoted."""
        tree = TaskTree()
        parent = tree.add_task("fetch data")
        failing = tree.add_task(
            "via API",
            parent_id=parent,
            postconditions=["200 OK status returned"],
        )
        # Pre-register an alternative that the failure cascade can activate.
        alt = tree.add_task("via scraping")
        tree.nodes[parent].alternatives = [alt]

        tree.update_status(failing, TaskStatus.DONE, result="500 Internal")
        assert tree.nodes[failing].status == TaskStatus.FAILED
        assert tree.nodes[alt].status == TaskStatus.READY
