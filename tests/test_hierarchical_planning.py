import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ghost_agent.core.planning import TaskTree, TaskStatus

def test_task_tree_creation():
    tree = TaskTree()
    root_id = tree.add_task("Root Goal")
    child1 = tree.add_task("Child 1", parent_id=root_id)
    child2 = tree.add_task("Child 2", parent_id=root_id)
    
    # Check structure
    assert tree.root_id == root_id, "Root ID mismatch"
    assert len(tree.nodes) == 3, "Node count mismatch"
    assert root_id in tree.nodes, "Root node missing"
    assert child1 in tree.nodes, "Child 1 missing"
    
    # Check relationships
    root_node = tree.nodes[root_id]
    assert child1 in root_node.children, "Child 1 not in root children"
    assert child2 in root_node.children, "Child 2 not in root children"
    assert tree.nodes[child1].parent_id == root_id, "Parent ID mismatch"
    print("test_task_tree_creation: PASS")

def test_status_updates():
    tree = TaskTree()
    root_id = tree.add_task("Root")
    c1 = tree.add_task("C1", parent_id=root_id)
    
    tree.update_status(c1, TaskStatus.DONE)
    assert tree.nodes[c1].status == TaskStatus.DONE, "Status update failed"
    
    # Check parent auto-complete logic
    # If parent has only one child and it's done, parent should be done
    assert tree.nodes[root_id].status == TaskStatus.DONE, "Parent auto-complete failed"
    print("test_status_updates: PASS")

def test_active_node_selection():
    tree = TaskTree()
    root_id = tree.add_task("Root", status=TaskStatus.IN_PROGRESS)
    c1 = tree.add_task("C1", parent_id=root_id, status=TaskStatus.DONE)
    c2 = tree.add_task("C2", parent_id=root_id, status=TaskStatus.READY)
    c3 = tree.add_task("C3", parent_id=root_id, status=TaskStatus.PENDING)
    
    active = tree.get_active_node()
    assert active is not None, "No active node found"
    assert active.id == c2, f"Wrong active node: {active.id} (expected {c2})"
    print("test_active_node_selection: PASS")

def test_json_load():
    tree = TaskTree()
    json_data = {
        "id": "root",
        "description": "Main",
        "status": "IN_PROGRESS",
        "children": [
            {"id": "c1", "description": "Sub 1", "status": "DONE"},
            {"id": "c2", "description": "Sub 2", "status": "READY", "children": []}
        ]
    }
    
    tree.load_from_json(json_data)
    
    assert tree.root_id == "root", "Root ID load failed"
    assert tree.nodes["root"].status == TaskStatus.IN_PROGRESS, "Root status mismatch"
    assert len(tree.nodes["root"].children) == 2, "Root children count mismatch"
    assert tree.nodes["c1"].status == TaskStatus.DONE, "Child 1 status mismatch"
    assert tree.nodes["c2"].status == TaskStatus.READY, "Child 2 status mismatch"
    print("test_json_load: PASS")

if __name__ == "__main__":
    test_task_tree_creation()
    test_status_updates()
    test_active_node_selection()
    test_json_load()
