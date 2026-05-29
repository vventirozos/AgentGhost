# src/ghost_agent/core/planning.py

import json
import uuid
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    # Added for long-term projects: tasks can suspend between sessions
    # (PAUSED) or explicitly wait for human input (NEEDS_USER). Neither
    # counts as a failure, so postcondition/parent-failure propagation
    # must treat them as in-flight, not terminal.
    PAUSED = "PAUSED"
    NEEDS_USER = "NEEDS_USER"


class DependencyType(str, Enum):
    """How child task results combine to satisfy the parent."""
    ALL = "ALL"    # All children must succeed (default — strict dependency)
    ANY = "ANY"    # First successful child satisfies the parent (OR-branch)
    BEST = "BEST"  # All children run; parent picks the best result


@dataclass
class TaskNode:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    result_summary: str = ""
    failure_reason: str = ""
    revision_count: int = 0
    actual_tool_used: Optional[str] = None
    # ── Hierarchical goal decomposition extensions ──
    dependency_type: DependencyType = DependencyType.ALL
    alternatives: List[str] = field(default_factory=list)  # fallback task IDs
    postconditions: List[str] = field(default_factory=list)  # validation descriptions
    estimated_cost: float = 0.0
    actual_cost: float = 0.0

    def to_dict(self):
        d = asdict(self)
        # Serialize enums as strings for JSON compat
        d["dependency_type"] = self.dependency_type.value
        return d

class TaskTree:
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.root_id: Optional[str] = None

    def add_task(self, description: str, parent_id: Optional[str] = None,
                 status: TaskStatus = TaskStatus.PENDING,
                 dependency_type: DependencyType = DependencyType.ALL,
                 alternatives: Optional[List[str]] = None,
                 postconditions: Optional[List[str]] = None) -> str:
        node_id = str(uuid.uuid4())[:8]
        # Guard against (astronomically unlikely) collisions
        while node_id in self.nodes:
            node_id = str(uuid.uuid4())[:8]
        node = TaskNode(
            id=node_id, description=description, status=status,
            parent_id=parent_id, dependency_type=dependency_type,
            alternatives=alternatives or [], postconditions=postconditions or [],
        )
        self.nodes[node_id] = node

        if parent_id:
            if parent_id in self.nodes:
                self.nodes[parent_id].children.append(node_id)
            else:
                import logging as _logging
                _logging.getLogger("GhostAgent").warning(
                    "TaskTree: parent_id %r not found when adding node %r", parent_id, node_id
                )
        elif self.root_id is None:
            self.root_id = node_id
            
        return node_id

    def update_status(self, task_id: str, status: TaskStatus, result: str = "",
                      failure_reason: str = "", actual_tool: str = ""):
        if task_id in self.nodes:
            node = self.nodes[task_id]
            node.status = status
            if result:
                node.result_summary = result
            if failure_reason:
                node.failure_reason = failure_reason
            if actual_tool:
                node.actual_tool_used = actual_tool
            if status == TaskStatus.DONE:
                # Postcondition gate: if the task declared postconditions,
                # evaluate them against the result summary. Any unsatisfied
                # postcondition flips the task back to FAILED so the
                # alternatives path gets a chance to rerun with a different
                # approach. Keeping this inline keeps the contract local:
                # `update_status(DONE)` is the one place completion is
                # declared, so it's also the one place we gate.
                unsat = self._check_postconditions(node)
                if unsat:
                    node.status = TaskStatus.FAILED
                    node.failure_reason = (
                        f"Postcondition(s) not satisfied: {'; '.join(unsat)}"
                    )
                    self._check_parent_failure(node.parent_id)
                    return
                self._check_parent_completion(node.parent_id)
            elif status in [TaskStatus.FAILED, TaskStatus.BLOCKED]:
                self._check_parent_failure(node.parent_id)

    def _check_postconditions(self, node: TaskNode) -> List[str]:
        """Return the list of unsatisfied postconditions for `node`.

        Empty postconditions list → always satisfied. Each postcondition is
        a free-form string (e.g. "file exists", "exit_code == 0"). Cheap
        textual verification: satisfied when the postcondition's key
        tokens appear in the result summary, or the postcondition is
        literally echoed back. For structured gating (postcondition DSL
        with actual file/exit-code checks) callers can pass
        ``self.postcondition_checker``, an optional callable on the tree.
        """
        if not node.postconditions:
            return []
        checker = getattr(self, "postcondition_checker", None)
        result_summary = (node.result_summary or "").lower()
        unsat: List[str] = []
        for pc in node.postconditions:
            if not pc or not str(pc).strip():
                continue
            pc_str = str(pc).strip()
            if checker is not None:
                try:
                    if checker(pc_str, node):
                        continue
                except Exception:
                    pass
            pc_lower = pc_str.lower()
            if pc_lower in result_summary:
                continue
            # Token-overlap heuristic: a postcondition is considered
            # satisfied when ≥60 % of its non-trivial tokens appear in
            # the result summary. This tolerates paraphrasing ("file
            # exists" matches "created file at path …") without needing
            # a separate LLM call.
            tokens = [
                t for t in pc_lower.replace("_", " ").split()
                if len(t) > 2 and t not in {"the", "and", "for", "not", "are", "has"}
            ]
            if tokens:
                hits = sum(1 for t in tokens if t in result_summary)
                if hits / len(tokens) >= 0.6:
                    continue
            unsat.append(pc_str)
        return unsat
                
    def root_postconditions_unsatisfied(self, response_text: str) -> List[str]:
        """Check a FINAL agent response against the root task's postconditions.

        ``_check_postconditions`` gates *internal* task completion; this
        method gates the agent's actual user-facing answer against the
        top-level plan's declared success criteria. That is what makes
        the plan load-bearing on the response, not just on internal
        bookkeeping. Returns the list of postconditions the response
        does not appear to satisfy (empty when the root declares none,
        or when all are satisfied)."""
        if not self.root_id:
            return []
        root = self.nodes.get(self.root_id)
        if root is None or not root.postconditions:
            return []
        saved = root.result_summary
        root.result_summary = response_text or ""
        try:
            return self._check_postconditions(root)
        finally:
            root.result_summary = saved

    def _check_parent_completion(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)

        parent = self.nodes[parent_id]
        if not parent.children: return

        child_statuses = [
            self.nodes[cid].status for cid in parent.children
            if cid in self.nodes
        ]

        if parent.dependency_type == DependencyType.ALL:
            # All children must be DONE. `child_statuses` can be EMPTY
            # even when parent.children is non-empty (child IDs dangling /
            # not yet hydrated); guard that case so `all([])`'s vacuous
            # True can't mark a parent DONE with zero real children
            # completed (which then cascades up to the root).
            if child_statuses and all(s == TaskStatus.DONE for s in child_statuses):
                parent.status = TaskStatus.DONE
                self._check_parent_completion(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.ANY:
            # First successful child satisfies the parent (OR-branch)
            if any(s == TaskStatus.DONE for s in child_statuses):
                parent.status = TaskStatus.DONE
                # Collect the winning child's result
                for cid in parent.children:
                    child = self.nodes.get(cid)
                    if child and child.status == TaskStatus.DONE:
                        parent.result_summary = child.result_summary
                        parent.actual_tool_used = child.actual_tool_used
                        break
                self._check_parent_completion(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.BEST:
            # All children must be terminal (DONE or FAILED), then pick best
            terminal = {TaskStatus.DONE, TaskStatus.FAILED}
            if child_statuses and all(s in terminal for s in child_statuses):
                done_children = [
                    self.nodes[cid] for cid in parent.children
                    if cid in self.nodes and self.nodes[cid].status == TaskStatus.DONE
                ]
                if done_children:
                    # Pick the child with the longest result (heuristic for quality)
                    best = max(done_children, key=lambda c: len(c.result_summary))
                    parent.result_summary = best.result_summary
                    parent.actual_tool_used = best.actual_tool_used
                    parent.status = TaskStatus.DONE
                else:
                    parent.status = TaskStatus.FAILED
                    parent.failure_reason = "All BEST-dependency children failed"
                self._check_parent_completion(parent.parent_id, visited)

    def _check_parent_failure(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)

        parent = self.nodes[parent_id]

        if parent.status in [TaskStatus.DONE, TaskStatus.FAILED]:
            return

        child_statuses = [
            self.nodes[cid].status for cid in parent.children
            if cid in self.nodes
        ]
        failed_statuses = {TaskStatus.FAILED, TaskStatus.BLOCKED}

        if parent.dependency_type == DependencyType.ALL:
            # Any child failure blocks the parent (strict)
            if any(s in failed_statuses for s in child_statuses):
                # Try alternatives before blocking
                if parent.alternatives:
                    alt_id = parent.alternatives.pop(0)
                    if alt_id in self.nodes:
                        self.nodes[alt_id].status = TaskStatus.READY
                        # Integrate the alternative into the parent's children
                        # so completion/failure cascading sees it.
                        if alt_id not in parent.children:
                            parent.children.append(alt_id)
                            self.nodes[alt_id].parent_id = parent_id
                        return  # Don't block — alternative is being tried
                parent.status = TaskStatus.BLOCKED
                self._check_parent_failure(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.ANY:
            # Only block if ALL children have failed. Guard empty: all([])
            # is vacuously True and would BLOCK a parent whose child IDs
            # are all dangling.
            if child_statuses and all(s in failed_statuses for s in child_statuses):
                parent.status = TaskStatus.BLOCKED
                parent.failure_reason = "All ANY-dependency children failed"
                self._check_parent_failure(parent.parent_id, visited)
            # Otherwise, some children are still running — don't block yet

        elif parent.dependency_type == DependencyType.BEST:
            # Don't block until all children are terminal
            terminal = {TaskStatus.DONE, TaskStatus.FAILED}
            if child_statuses and all(s in terminal for s in child_statuses):
                # If all failed, parent fails
                if all(s == TaskStatus.FAILED for s in child_statuses):
                    parent.status = TaskStatus.FAILED
                    parent.failure_reason = "All BEST-dependency children failed"
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
            "DONE": "✅", "FAILED": "❌", "BLOCKED": "🛑",
            "PAUSED": "⏸", "NEEDS_USER": "🙋",
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
                
            # Parse extended fields
            dep_type_str = node_data.get("dependency_type", "ALL").upper()
            try:
                dep_type = DependencyType[dep_type_str]
            except KeyError:
                dep_type = DependencyType.ALL
            alternatives = node_data.get("alternatives", [])
            postconditions = node_data.get("postconditions", [])

            if node_id in self.nodes:
                # Update existing node
                node = self.nodes[node_id]
                node.description = desc
                # ANTI-REGRESSION GUARD: Don't allow DONE tasks to revert.
                # Log a warning when we reject a regression so downstream
                # operators can spot planner confusion (the previous silent
                # `pass` made this invisible).
                if node.status == TaskStatus.DONE and status != TaskStatus.DONE:
                    import logging as _logging
                    _logging.getLogger("GhostAgent").warning(
                        "TaskTree: rejected status regression for node %r (%s → %s)",
                        node_id, node.status.name if hasattr(node.status, 'name') else node.status,
                        status.name if hasattr(status, 'name') else status,
                    )
                else:
                    node.status = status
                # Update extended fields
                node.dependency_type = dep_type
                if alternatives:
                    node.alternatives = alternatives
                if postconditions:
                    node.postconditions = postconditions
                # Parent ID might theoretically change in a re-org, though rare.
                # If parent_id is provided here (from recursion), we update it.
                if parent_id: node.parent_id = parent_id
            else:
                # Create new node
                node = TaskNode(
                    id=node_id, description=desc, status=status,
                    parent_id=parent_id, children=[],
                    dependency_type=dep_type, alternatives=alternatives,
                    postconditions=postconditions,
                )
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

    def request_revision(self, task_id: str, failure_reason: str) -> bool:
        """Request a plan revision for a failed task.

        Increments the revision counter and resets the task to PENDING so
        the planner can re-route. Returns False if max revisions (3) exceeded.
        """
        MAX_REVISIONS = 3
        if task_id not in self.nodes:
            return False
        node = self.nodes[task_id]
        if node.revision_count >= MAX_REVISIONS:
            return False
        node.revision_count += 1
        node.failure_reason = failure_reason
        node.status = TaskStatus.PENDING
        # Also unblock the parent if it was BLOCKED by this task
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if parent.status == TaskStatus.BLOCKED:
                parent.status = TaskStatus.IN_PROGRESS
        return True

    def get_failed_tasks(self) -> List[TaskNode]:
        """Return all tasks with FAILED status and their failure reasons."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]

    def generate_retrospective(self) -> Dict[str, Any]:
        """Generate a structured retrospective of the plan execution.

        Returns a dict suitable for feeding to learn_lesson with:
        - what worked (DONE tasks)
        - what failed (FAILED tasks with reasons)
        - revision attempts
        - tool usage patterns
        """
        if not self.root_id:
            return {"summary": "No plan executed."}

        done_tasks = [n for n in self.nodes.values() if n.status == TaskStatus.DONE]
        # FAILED comes first so that tasks with concrete failure reasons
        # appear at the head of `what_failed`. BLOCKED tasks (propagated
        # failures on ancestors that never ran) are kept in the count
        # but sorted after, since their failure_reason is typically empty.
        failed_tasks = [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]
        failed_tasks += [n for n in self.nodes.values() if n.status == TaskStatus.BLOCKED]
        revised_tasks = [n for n in self.nodes.values() if n.revision_count > 0]
        total = len(self.nodes)

        retro = {
            "total_tasks": total,
            "completed": len(done_tasks),
            "failed": len(failed_tasks),
            "revised": len(revised_tasks),
            "success_rate": len(done_tasks) / total if total > 0 else 0.0,
            "what_worked": [
                {"id": n.id, "description": n.description, "tool": n.actual_tool_used}
                for n in done_tasks
            ],
            "what_failed": [
                {
                    "id": n.id,
                    "description": n.description,
                    "reason": n.failure_reason,
                    "revisions": n.revision_count,
                }
                for n in failed_tasks
            ],
            "revision_history": [
                {
                    "id": n.id,
                    "description": n.description,
                    "attempts": n.revision_count,
                    "final_status": n.status.value,
                }
                for n in revised_tasks
            ],
        }

        # Generate a summary string
        if failed_tasks:
            failure_summary = "; ".join(
                f"{n.description[:50]}: {n.failure_reason[:80]}" for n in failed_tasks[:3]
            )
            retro["summary"] = f"Plan completed {len(done_tasks)}/{total} tasks. Failures: {failure_summary}"
        else:
            retro["summary"] = f"Plan completed successfully: {len(done_tasks)}/{total} tasks done."

        return retro

    def to_json(self) -> Dict[str, Any]:
        if not self.root_id: return {}

        def serialize(node_id: str, visited: set = None) -> Dict[str, Any]:
            if visited is None:
                visited = set()
            if node_id in visited or node_id not in self.nodes:
                return {"id": node_id, "description": "[cycle or missing]", "status": "BLOCKED", "children": []}
            visited.add(node_id)
            node = self.nodes[node_id]
            result = {
                "id": node.id,
                "description": node.description,
                "status": node.status.value,
                "children": [serialize(cid, visited) for cid in node.children]
            }
            if node.failure_reason:
                result["failure_reason"] = node.failure_reason
            if node.revision_count > 0:
                result["revision_count"] = node.revision_count
            if node.actual_tool_used:
                result["actual_tool_used"] = node.actual_tool_used
            if node.dependency_type != DependencyType.ALL:
                result["dependency_type"] = node.dependency_type.value
            if node.alternatives:
                result["alternatives"] = node.alternatives
            if node.postconditions:
                result["postconditions"] = node.postconditions
            return result

        return serialize(self.root_id)


class ProjectPlan:
    """Persistent TaskTree bound to a ProjectStore.

    Hydrates a ``TaskTree`` from the rows in ``memory/projects.py`` for
    a given project_id. Mutations (``add_task``, ``update_status``,
    ``decompose``) write back through the store so every change is
    durable and appears in ``project_events`` for the resumption
    briefing. The in-memory ``tree`` is the authoritative cascade
    engine — postcondition gating and parent completion/failure logic
    lives there — and we mirror its state to SQLite after each op.

    Note: the stored ``id`` on a TaskNode is the same as the
    ``tasks.id`` column, so hydration is reversible without a mapping
    layer.
    """

    def __init__(self, store, project_id: str):
        self.store = store
        self.project_id = project_id
        self.tree = TaskTree()
        self._hydrate()

    # ------------------------------------------------------------------ hydrate

    def _hydrate(self):
        rows = self.store.list_tasks(self.project_id)
        if not rows:
            return
        for row in rows:
            try:
                status = TaskStatus[row["status"]]
            except KeyError:
                status = TaskStatus.PENDING
            try:
                dep = DependencyType[row["dependency_type"]]
            except KeyError:
                dep = DependencyType.ALL
            node = TaskNode(
                id=row["id"],
                description=row["description"],
                status=status,
                parent_id=row["parent_id"],
                children=[],
                result_summary=row["result_summary"] or "",
                failure_reason=row["failure_reason"] or "",
                revision_count=int(row["revision_count"] or 0),
                actual_tool_used=row["actual_tool_used"],
                dependency_type=dep,
                alternatives=list(row.get("alternatives") or []),
                postconditions=list(row.get("postconditions") or []),
                estimated_cost=float(row["estimated_cost"] or 0.0),
                actual_cost=float(row["actual_cost"] or 0.0),
            )
            self.tree.nodes[node.id] = node
            if node.parent_id is None and self.tree.root_id is None:
                self.tree.root_id = node.id
        # Link children (rows are depth-ordered so parents land first)
        for node in self.tree.nodes.values():
            if node.parent_id and node.parent_id in self.tree.nodes:
                parent = self.tree.nodes[node.parent_id]
                if node.id not in parent.children:
                    parent.children.append(node.id)

    # ------------------------------------------------------------------ mutate

    def add_task(self, description: str, parent_id: Optional[str] = None,
                 status: TaskStatus = TaskStatus.PENDING,
                 dependency_type: DependencyType = DependencyType.ALL,
                 alternatives: Optional[List[str]] = None,
                 postconditions: Optional[List[str]] = None,
                 estimated_cost: float = 0.0) -> str:
        task_id = self.store.add_task(
            self.project_id, description, parent_id=parent_id,
            status=status.value, dependency_type=dependency_type.value,
            alternatives=alternatives, postconditions=postconditions,
            estimated_cost=estimated_cost,
        )
        node = TaskNode(
            id=task_id, description=description, status=status,
            parent_id=parent_id,
            dependency_type=dependency_type,
            alternatives=list(alternatives or []),
            postconditions=list(postconditions or []),
            estimated_cost=estimated_cost,
        )
        self.tree.nodes[task_id] = node
        if parent_id and parent_id in self.tree.nodes:
            self.tree.nodes[parent_id].children.append(task_id)
        elif parent_id is None and self.tree.root_id is None:
            self.tree.root_id = task_id
        return task_id

    def update_status(self, task_id: str, status: TaskStatus,
                      result: str = "", failure_reason: str = "",
                      actual_tool: str = ""):
        """Cascade through the in-memory tree, then persist every node
        whose status changed as a side effect."""
        before = {tid: (n.status, n.result_summary, n.failure_reason,
                        n.actual_tool_used, n.revision_count)
                  for tid, n in self.tree.nodes.items()}
        self.tree.update_status(task_id, status, result=result,
                                failure_reason=failure_reason,
                                actual_tool=actual_tool)
        for tid, node in self.tree.nodes.items():
            prev = before.get(tid)
            current = (node.status, node.result_summary, node.failure_reason,
                       node.actual_tool_used, node.revision_count)
            if prev != current:
                self.store.update_task(
                    tid,
                    status=node.status.value,
                    result_summary=node.result_summary,
                    failure_reason=node.failure_reason,
                    actual_tool_used=node.actual_tool_used,
                    revision_count=node.revision_count,
                )

    def decompose(self, task_id: str,
                  subtask_descriptions: List[str]) -> List[str]:
        """Expand a task into ordered subtasks. Returns new task ids."""
        if task_id not in self.tree.nodes:
            raise ValueError(f"unknown task: {task_id}")
        ids: List[str] = []
        for desc in subtask_descriptions:
            if not desc or not desc.strip():
                continue
            ids.append(self.add_task(desc.strip(), parent_id=task_id))
        return ids

    # ------------------------------------------------------------------ query

    def next_ready_leaf(self) -> Optional[TaskNode]:
        """Return the next leaf that is READY or PENDING and not
        blocked by an upstream PAUSED/NEEDS_USER/BLOCKED ancestor.

        Scans every leaf in the tree, not just descendants of
        ``root_id`` — projects routinely have multiple sibling root
        tasks that share no common parent. ``root_id`` is just the
        first root ever seen, so a walk anchored there would silently
        skip every subsequent top-level task.
        """
        if not self.tree.nodes:
            return None
        blocking = {TaskStatus.PAUSED, TaskStatus.NEEDS_USER,
                    TaskStatus.BLOCKED}
        eligible = {TaskStatus.PENDING, TaskStatus.READY}

        def ancestor_blocked(node: TaskNode) -> bool:
            cur = node
            while cur.parent_id:
                parent = self.tree.nodes.get(cur.parent_id)
                if not parent:
                    return False
                if parent.status in blocking:
                    return True
                cur = parent
            return False

        for node in self.tree.nodes.values():
            if node.children:
                continue
            if node.status not in eligible:
                continue
            if ancestor_blocked(node):
                continue
            return node
        return None

    def render(self) -> str:
        return self.tree.render()

    def to_json(self) -> Dict[str, Any]:
        return self.tree.to_json()

    def generate_retrospective(self) -> Dict[str, Any]:
        return self.tree.generate_retrospective()
