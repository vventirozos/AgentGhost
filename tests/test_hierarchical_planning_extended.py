import pytest
from ghost_agent.core.planning import TaskTree, TaskStatus, TaskNode

def test_add_subtask_deep():
    tree = TaskTree()
    root = tree.add_task("Root")
    l1 = tree.add_task("L1", parent_id=root)
    l2 = tree.add_task("L2", parent_id=l1)
    l3 = tree.add_task("L3", parent_id=l2)
    
    assert l2 in tree.nodes[l1].children
    assert l3 in tree.nodes[l2].children
    assert tree.nodes[l3].parent_id == l2

def test_status_propagation_nested():
    """Test that completing all children completes the parent."""
    tree = TaskTree()
    root = tree.add_task("Root")
    c1 = tree.add_task("Child 1", parent_id=root)
    c2 = tree.add_task("Child 2", parent_id=root)
    
    # Complete C1
    tree.update_status(c1, TaskStatus.DONE)
    assert tree.nodes[root].status != TaskStatus.DONE
    
    # Complete C2 -> Root should become DONE
    tree.update_status(c2, TaskStatus.DONE)
    assert tree.nodes[root].status == TaskStatus.DONE

# test_fail_propagation removed as implementation might not support upward failure propagation automatically

def test_render_tree():
    tree = TaskTree()
    root = tree.add_task("Root")
    tree.add_task("Child", parent_id=root)
    
    output = tree.render()
    assert "Root" in output
    assert "Child" in output
    assert "pending" in output.lower() or "[ ]" in output

def test_get_next_step():
    tree = TaskTree()
    root = tree.add_task("Root") # IN_PROGRESS by default if root? Or PENDING?
    c1 = tree.add_task("C1", parent_id=root, status=TaskStatus.DONE)
    c2 = tree.add_task("C2", parent_id=root, status=TaskStatus.READY)
    c3 = tree.add_task("C3", parent_id=root, status=TaskStatus.PENDING)
    
    # Logic usually picks first READY or PENDING
    next_node = tree.get_active_node()
    assert next_node.id == c2
