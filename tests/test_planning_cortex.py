import pytest
from ghost_agent.core.planning import TaskTree, TaskStatus

def test_tree_generation():
    """Test loading a JSON payload into TaskTree with sub-tasks."""
    tree = TaskTree()
    payload = {
        "id": "root",
        "description": "Root Goal",
        "status": "IN_PROGRESS",
        "children": [
            {"id": "c1", "description": "Child 1", "status": "PENDING"},
            {"id": "c2", "description": "Child 2", "status": "PENDING"},
            {"id": "c3", "description": "Child 3", "status": "PENDING"}
        ]
    }
    
    tree.load_from_json(payload)
    
    assert tree.root_id == "root"
    assert len(tree.nodes) == 4
    assert len(tree.nodes["root"].children) == 3
    assert tree.nodes["c1"].parent_id == "root"

def test_auto_completion():
    """
    Test that completing all child tasks automatically marks 
    the parent task as DONE.
    """
    tree = TaskTree()
    # Setup manually to control IDs
    root_id = tree.add_task("Root", status=TaskStatus.IN_PROGRESS)
    c1 = tree.add_task("C1", parent_id=root_id, status=TaskStatus.PENDING)
    c2 = tree.add_task("C2", parent_id=root_id, status=TaskStatus.PENDING)
    c3 = tree.add_task("C3", parent_id=root_id, status=TaskStatus.PENDING)
    
    # Complete them one by one
    tree.update_status(c1, TaskStatus.DONE)
    assert tree.nodes[root_id].status == TaskStatus.IN_PROGRESS
    
    tree.update_status(c2, TaskStatus.DONE)
    assert tree.nodes[root_id].status == TaskStatus.IN_PROGRESS
    
    tree.update_status(c3, TaskStatus.DONE)
    
    # Assert Root is now DONE
    assert tree.nodes[root_id].status == TaskStatus.DONE

def test_recovery_prioritization():
    """
    Test that FAILED nodes are NO LONGER prioritized.
    The planner should move on to other pending tasks.
    """
    tree = TaskTree()
    root_id = tree.add_task("Root", status=TaskStatus.IN_PROGRESS)
    
    # Children with mixed states
    c1 = tree.add_task("C1", parent_id=root_id, status=TaskStatus.DONE)
    c2 = tree.add_task("C2", parent_id=root_id, status=TaskStatus.PENDING)
    c3 = tree.add_task("C3", parent_id=root_id, status=TaskStatus.FAILED)
    
    active_node = tree.get_active_node()
    
    # Should IGNORE the failed node (c3) and pick the pending one (c2)
    assert active_node.id == c2
    assert active_node.status == TaskStatus.PENDING
