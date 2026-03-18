# src/ghost_agent/core/planning.py

import json
import uuid
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"

@dataclass
class TaskNode:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    result_summary: str = ""

    def to_dict(self):
        return asdict(self)

class TaskTree:
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.root_id: Optional[str] = None

    def add_task(self, description: str, parent_id: Optional[str] = None, status: TaskStatus = TaskStatus.PENDING) -> str:
        node_id = str(uuid.uuid4())[:4]
        node = TaskNode(id=node_id, description=description, status=status, parent_id=parent_id)
        self.nodes[node_id] = node
        
        if parent_id:
            if parent_id in self.nodes:
                self.nodes[parent_id].children.append(node_id)
        elif self.root_id is None:
            self.root_id = node_id
            
        return node_id

    def update_status(self, task_id: str, status: TaskStatus, result: str = ""):
        if task_id in self.nodes:
            self.nodes[task_id].status = status
            if result:
                self.nodes[task_id].result_summary = result
            if status == TaskStatus.DONE:
                self._check_parent_completion(self.nodes[task_id].parent_id)
            elif status in [TaskStatus.FAILED, TaskStatus.BLOCKED]:
                self._check_parent_failure(self.nodes[task_id].parent_id)
                
    def _check_parent_completion(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)
        
        parent = self.nodes[parent_id]
        if not parent.children: return
        
        all_done = all(self.nodes.get(child_id) and self.nodes[child_id].status == TaskStatus.DONE for child_id in parent.children)
        if all_done:
            parent.status = TaskStatus.DONE
            self._check_parent_completion(parent.parent_id, visited)

    def _check_parent_failure(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)
        
        parent = self.nodes[parent_id]
        # If any child is failed or blocked, the parent is blocked
        # (It can't complete if a subtask failed)
        
        # We implicitly know one child failed/blocked to trigger this, 
        # so we can just set status to BLOCKED.
        # Unless we want to support partial success? For now, strict dependency.
        
        if parent.status not in [TaskStatus.DONE, TaskStatus.FAILED]:
            parent.status = TaskStatus.BLOCKED
            self._check_parent_failure(parent.parent_id, visited)

    def get_active_node(self) -> Optional[TaskNode]:
        if not self.root_id: return None
        
        def find_status(node_id: str, target_statuses: List[TaskStatus], visited: set) -> Optional[TaskNode]:
            if node_id in visited: return None
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if not node: return None
            for child_id in node.children:
                found = find_status(child_id, target_statuses, visited)
                if found: return found
                
            if node.status in target_statuses and not node.children: 
                return node
            return None

        in_prog = find_status(self.root_id, [TaskStatus.IN_PROGRESS], set())
        if in_prog: return in_prog
        
        ready = find_status(self.root_id, [TaskStatus.READY, TaskStatus.PENDING], set()) 
        if ready: return ready
        
        return None

    def render(self) -> str:
        if not self.root_id: return "No Plan."
        lines = []
        self._render_node(self.root_id, 0, lines, set())
        return "\n".join(lines)

    def _render_node(self, node_id: str, depth: int, lines: List[str], visited: set):
        if node_id in visited or depth > 20: return
        visited.add(node_id)
        
        node = self.nodes.get(node_id)
        if not node: return
        indent = "  " * depth
        icon = {
            "PENDING": "⏳", "READY": "🟢", "IN_PROGRESS": "🔄",
            "DONE": "✅", "FAILED": "❌", "BLOCKED": "🛑"
        }.get(node.status.value, "➖")
        
        lines.append(f"{indent}{icon} [{node.id}] {node.description} ({node.status.value})")
        for child_id in node.children:
            self._render_node(child_id, depth + 1, lines, visited)

    def load_from_json(self, json_data: Any):
        if not json_data: return
        
        # Stateful merge: do not clear self.nodes or self.root_id
        
        def traverse(node_data: Any, parent_id: Optional[str] = None, visited: set = None):
            if visited is None: visited = set()
            if not isinstance(node_data, dict): return
            
            node_id = node_data.get("id", str(uuid.uuid4())[:4])
            if node_id in visited: return
            visited.add(node_id)
            desc = node_data.get("description", "Unknown Task")
            status_str = node_data.get("status", "PENDING").upper()
            try:
                status = TaskStatus[status_str]
            except KeyError:
                status = TaskStatus.PENDING
                
            if node_id in self.nodes:
                # Update existing node
                node = self.nodes[node_id]
                node.description = desc
                # ANTI-REGRESSION GUARD: Don't allow DONE tasks to revert
                if node.status == TaskStatus.DONE and status != TaskStatus.DONE:
                    pass
                else:
                    node.status = status
                # Parent ID might theoretically change in a re-org, though rare. 
                # If parent_id is provided here (from recursion), we update it.
                if parent_id: node.parent_id = parent_id
            else:
                # Create new node
                node = TaskNode(id=node_id, description=desc, status=status, parent_id=parent_id, children=[])
                self.nodes[node_id] = node
            
            if not parent_id and not self.root_id:
                self.root_id = node_id
            elif parent_id:
                if parent_id in self.nodes:
                    # Prevent duplicates
                    if node_id not in self.nodes[parent_id].children:
                        self.nodes[parent_id].children.append(node_id)
                    
            children_data = node_data.get("children", [])
            if isinstance(children_data, list):
                for child in children_data:
                    traverse(child, node_id, visited)
                    
        traverse(json_data)

    def to_json(self) -> Dict[str, Any]:
        if not self.root_id: return {}
        
        def serialize(node_id: str) -> Dict[str, Any]:
            node = self.nodes[node_id]
            return {
                "id": node.id,
                "description": node.description,
                "status": node.status.value,
                "children": [serialize(cid) for cid in node.children]
            }
            
        return serialize(self.root_id)
        