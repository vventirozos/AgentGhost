"""Tests for plan revision loop (#1).

Verifies that:
- TaskNode has failure_reason, revision_count, actual_tool_used fields
- request_revision resets task to PENDING and increments revision_count
- Max revisions (3) are enforced
- generate_retrospective produces structured output
- to_json includes new fields
"""

import pytest
from ghost_agent.core.planning import TaskTree, TaskNode, TaskStatus


@pytest.fixture
def tree():
    t = TaskTree()
    root = t.add_task("Root task")
    child1 = t.add_task("Step 1", parent_id=root)
    child2 = t.add_task("Step 2", parent_id=root)
    return t, root, child1, child2


class TestTaskNodeExtensions:
    def test_failure_reason_default(self):
        node = TaskNode(id="1", description="test")
        assert node.failure_reason == ""
        assert node.revision_count == 0
        assert node.actual_tool_used is None

    def test_update_status_with_failure_reason(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="timeout error")
        assert t.nodes[child1].failure_reason == "timeout error"

    def test_update_status_with_actual_tool(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE, actual_tool="execute")
        assert t.nodes[child1].actual_tool_used == "execute"


class TestPlanRevision:
    def test_request_revision_resets_to_pending(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="first failure")

        result = t.request_revision(child1, "need different approach")
        assert result is True
        assert t.nodes[child1].status == TaskStatus.PENDING
        assert t.nodes[child1].revision_count == 1
        assert t.nodes[child1].failure_reason == "need different approach"

    def test_max_revisions_enforced(self, tree):
        t, root, child1, child2 = tree

        for i in range(3):
            t.update_status(child1, TaskStatus.FAILED)
            assert t.request_revision(child1, f"attempt {i+1}") is True

        # 4th revision should fail
        t.update_status(child1, TaskStatus.FAILED)
        assert t.request_revision(child1, "too many") is False
        assert t.nodes[child1].revision_count == 3

    def test_revision_unblocks_parent(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED)
        # Parent should be BLOCKED
        assert t.nodes[root].status == TaskStatus.BLOCKED

        t.request_revision(child1, "retry")
        # Parent should be unblocked
        assert t.nodes[root].status == TaskStatus.IN_PROGRESS

    def test_revision_nonexistent_task(self, tree):
        t, root, child1, child2 = tree
        assert t.request_revision("nonexistent", "reason") is False


class TestGetFailedTasks:
    def test_returns_failed_tasks(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="error A")
        t.update_status(child2, TaskStatus.DONE)

        failed = t.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].id == child1

    def test_returns_empty_when_all_done(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.DONE)

        assert len(t.get_failed_tasks()) == 0


class TestRetrospective:
    def test_successful_retrospective(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE, actual_tool="execute")
        t.update_status(child2, TaskStatus.DONE, actual_tool="file_system")

        retro = t.generate_retrospective()
        assert retro["completed"] == 3  # root auto-completes
        assert retro["failed"] == 0
        assert retro["success_rate"] == 1.0
        assert "successfully" in retro["summary"]

    def test_partial_failure_retrospective(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.FAILED, failure_reason="tool not found")

        retro = t.generate_retrospective()
        assert retro["failed"] >= 1
        assert len(retro["what_failed"]) >= 1
        assert retro["what_failed"][0]["reason"] == "tool not found"

    def test_revision_history_in_retrospective(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED)
        t.request_revision(child1, "retry 1")
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.DONE)

        retro = t.generate_retrospective()
        assert retro["revised"] >= 1
        assert retro["revision_history"][0]["attempts"] == 1

    def test_empty_tree_retrospective(self):
        t = TaskTree()
        retro = t.generate_retrospective()
        assert retro["summary"] == "No plan executed."


class TestToJsonExtensions:
    def test_includes_failure_reason(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="connection error")

        data = t.to_json()
        child_data = data["children"][0]
        assert child_data.get("failure_reason") == "connection error"

    def test_includes_revision_count(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED)
        t.request_revision(child1, "retry")

        data = t.to_json()
        child_data = data["children"][0]
        assert child_data.get("revision_count") == 1

    def test_omits_empty_fields(self, tree):
        t, root, child1, child2 = tree
        data = t.to_json()
        # Default values should not appear
        assert "failure_reason" not in data
        assert "revision_count" not in data
