
import pytest
from ghost_agent.core.planning import TaskTree, TaskStatus

def test_failure_propagation():
    tree = TaskTree()
    
    # 1. Create a hierarchy: Root -> Child -> Grandchild
    root_id = tree.add_task("Root Task", status=TaskStatus.IN_PROGRESS)
    child_id = tree.add_task("Child Task", parent_id=root_id, status=TaskStatus.IN_PROGRESS)
    grandchild_id = tree.add_task("Grandchild Task", parent_id=child_id, status=TaskStatus.IN_PROGRESS)
    
    # 2. Fail the grandchild
    tree.update_status(grandchild_id, TaskStatus.FAILED)
    
    # 3. Verify propagation
    # Grandchild should be FAILED
    assert tree.nodes[grandchild_id].status == TaskStatus.FAILED
    
    # Child should be BLOCKED (because a dependency failed)
    assert tree.nodes[child_id].status == TaskStatus.BLOCKED
    
    # Root should be BLOCKED (recursive propagation)
    assert tree.nodes[root_id].status == TaskStatus.BLOCKED

def test_blocked_propagation():
    tree = TaskTree()
    
    # 1. Parent -> Child
    root_id = tree.add_task("Root", status=TaskStatus.IN_PROGRESS)
    child_id = tree.add_task("Child", parent_id=root_id, status=TaskStatus.IN_PROGRESS)
    
    # 2. Block the child directly
    tree.update_status(child_id, TaskStatus.BLOCKED)
    
    # 3. Verify parent Blocked
    assert tree.nodes[root_id].status == TaskStatus.BLOCKED
