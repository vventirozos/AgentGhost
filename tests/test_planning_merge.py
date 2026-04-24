
import pytest
import json
from ghost_agent.core.planning import TaskTree, TaskStatus, TaskNode

def test_task_tree_merge_stateful():
    tree = TaskTree()
    
    # 1. Create initial tree
    root_id = tree.add_task("Root Task", status=TaskStatus.PENDING)
    child_id = tree.add_task("Child Task", parent_id=root_id, status=TaskStatus.PENDING)
    
    # Set a custom field that is NOT in the JSON update (to prove object identity preservation)
    # result_summary is a standard field
    tree.nodes[child_id].result_summary = "Original Result"
    
    original_child_node = tree.nodes[child_id]
    
    # 2. Prepare JSON update
    # Updates status of child, description of root
    update_data = {
        "id": root_id,
        "description": "Root Task Updated",
        "status": "IN_PROGRESS",
        "children": [
            {
                "id": child_id,
                "description": "Child Task",
                "status": "DONE"
            },
            {
                "id": "new-child",
                "description": "New Child Task",
                "status": "PENDING"
            }
        ]
    }
    
    # 3. Load JSON (Merge)
    tree.load_from_json(update_data)
    
    # 4. Verify Identity Preservation
    assert tree.nodes[child_id] is original_child_node, "Node object should be the same instance"
    assert tree.nodes[child_id].result_summary == "Original Result", "Field not in JSON should be preserved"
    
    # 5. Verify Updates
    assert tree.nodes[root_id].description == "Root Task Updated"
    assert tree.nodes[root_id].status == TaskStatus.IN_PROGRESS
    assert tree.nodes[child_id].status == TaskStatus.DONE
    
    # 6. Verify New Node creation
    assert "new-child" in tree.nodes
    assert tree.nodes["new-child"].description == "New Child Task"
    
    # 7. Verify Structure
    assert "new-child" in tree.nodes[root_id].children
    assert child_id in tree.nodes[root_id].children
    # Ensure no duplicates
    assert tree.nodes[root_id].children.count(child_id) == 1
