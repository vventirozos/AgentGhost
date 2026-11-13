# src/ghost_agent/core/planning.py

import json
import re
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


def _coerce_str_list(val: Any) -> List[str]:
    """Defensively coerce a planner-supplied field into a list of strings.

    Planner output is LLM JSON: fields documented as lists routinely
    arrive as null or as a bare string, and iterating a raw string walks
    it char-by-char — with the plan gate live that emitted one
    "unsatisfied postcondition" PER CHARACTER into the user reply.
    None → []; str → single-element list; list/tuple → items
    stringified with blanks dropped.
    """
    if val is None:
        return []
    if isinstance(val, str):
        val = val.strip()
        return [val] if val else []
    if isinstance(val, (list, tuple)):
        out: List[str] = []
        for item in val:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(val).strip()
    return [s] if s else []


#: §4HZ — how alike two descriptions must be for a reused task id to count
#: as the SAME task. Token-set Jaccard: a planner rewording ("check GRIB
#: tooling" → "check GRIB tooling is installed") stays the same task and
#: keeps the DONE guard; a replacement ("check GRIB tooling" → "install
#: ecmwf-api") is a new task under an old id and takes its own status.
SAME_TASK_JACCARD = 0.6
_TASK_WORD_RE = re.compile(r"[a-z0-9]+")


def same_task(old_desc: Any, new_desc: Any) -> bool:
    """True when two task descriptions name the same work (see
    ``SAME_TASK_JACCARD``). An empty side compares equal only to an empty
    side — a node whose description was never set is not "the same task"
    as whatever is written into it next."""
    a = set(_TASK_WORD_RE.findall(str(old_desc or "").lower()))
    b = set(_TASK_WORD_RE.findall(str(new_desc or "").lower()))
    if not a or not b:
        return not a and not b
    return len(a & b) / len(a | b) >= SAME_TASK_JACCARD


@dataclass
class TaskNode:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    result_summary: str = ""
    failure_reason: str = ""
    actual_tool_used: Optional[str] = None
    # ── Hierarchical goal decomposition extensions ──
    dependency_type: DependencyType = DependencyType.ALL
    alternatives: List[str] = field(default_factory=list)  # fallback task IDs
    postconditions: List[str] = field(default_factory=list)  # validation descriptions
    # Explicit user constraints this task must honor (verbatim clauses
    # from the request, e.g. "don't come up with some random AI"). Unlike
    # postconditions (which describe the SUCCESS state), constraints are
    # standing prohibitions/requirements re-shown to the model each turn
    # and replayed to the verifier at DONE time.
    constraints: List[str] = field(default_factory=list)
    # Sibling-DAG prerequisites: task IDs that must be DONE before this
    # task becomes eligible for execution. Distinct from parent/child
    # rollup (dependency_type) — this orders *peers*. An empty list means
    # "no prerequisites" (the historical behaviour, every leaf eligible).
    depends_on: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    # §4SU Phase 2: the DECLARED kind (core/task_kind.py; "" = undeclared,
    # routed by the wording classifiers as before)
    kind: str = ""
    # §4SU Phase 3: why a waiting (NEEDS_USER / BLOCKED) task waits — fields,
    # not text marks (core/task_wait.py); empty for any other status
    wait_on: str = ""
    wait_code: str = ""
    wait_why: str = ""

    def set_wait(self, on: str, code: str, why: str = "") -> None:
        self.wait_on, self.wait_code, self.wait_why = on, code, str(why or "")[:300]

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
                 postconditions: Optional[List[str]] = None,
                 constraints: Optional[List[str]] = None,
                 depends_on: Optional[List[str]] = None) -> str:
        node_id = str(uuid.uuid4())[:8]
        # Guard against (astronomically unlikely) collisions
        while node_id in self.nodes:
            node_id = str(uuid.uuid4())[:8]
        node = TaskNode(
            id=node_id, description=description, status=status,
            parent_id=parent_id, dependency_type=dependency_type,
            alternatives=alternatives or [], postconditions=postconditions or [],
            constraints=constraints or [],
            depends_on=depends_on or [],
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
                      failure_reason: str = "", actual_tool: str = "",
                      human_approved: bool = False, postconditions_waived: bool = False,
                      wait=None):
        """``postconditions_waived`` (§4SU Phase 1): the close's checker
        (core.task_close) already judged the postconditions and the waiver
        was recorded — the textual demotion below is not re-applied.
        ``wait`` (§4SU Phase 3): (wait_on, wait_code, wait_why) for a
        waiting status; see core/task_wait.py."""
        if task_id in self.nodes:
            node = self.nodes[task_id]
            prior_status = node.status
            prior_code = node.wait_code
            node.status = status
            # §4SU Phase 3 r1 MAJOR: every status write starts from NO wait —
            # only this writer's own ``wait=`` sets one (else settle_waits'
            # safe default). A kept wait turned the model's park "the owner
            # must approve the wording" into a HOLD the owner's run ran.
            node.set_wait(*(wait if wait and status in (TaskStatus.NEEDS_USER, TaskStatus.BLOCKED)
                            else ("", "", "")))
            if result:
                node.result_summary = result
            elif status == TaskStatus.DONE and (prior_code.startswith("rollup")
                                                or str(node.result_summary or "").startswith("not closed:")):
                # §4SU Phase 1 r2 M2: a roll-up refusal's text is not a DONE
                # record (a grandparent's roll-up read it as evidence)
                node.result_summary = ""
            if failure_reason:
                node.failure_reason = failure_reason
            if actual_tool:
                node.actual_tool_used = actual_tool
            if status == TaskStatus.DONE:
                # §4UK: a HUMAN_GATE postcondition no longer parks the close (no approvals); it is
                # still never graded as text (below).
                # Postcondition gate: if the task declared postconditions,
                # evaluate them against the result summary. Any unsatisfied
                # postcondition flips the task back to FAILED so the
                # alternatives path gets a chance to rerun with a different
                # approach. Keeping this inline keeps the contract local:
                # `update_status(DONE)` is the one place completion is
                # declared, so it's also the one place we gate.
                unsat = [] if postconditions_waived else self._check_postconditions(node)
                if unsat:
                    node.status = TaskStatus.FAILED
                    node.failure_reason = (
                        f"Postcondition(s) not satisfied: {'; '.join(unsat)}"
                    )
                    self._check_parent_failure(node.parent_id)
                    return
                # A task that actually reaches DONE has no CURRENT failure.
                # Keeping the old failure_reason made every later briefing/
                # ledger read show a resolved defect as live ("core.py does
                # NOT parse" survived on a DONE row — 2026-08-01 Mini AI
                # incident), and downstream failure-dimension tooling counted
                # it as an open failure. Cleared only AFTER the gates above
                # pass, so a demoted DONE keeps its real reason.
                node.failure_reason = ""
                # r4 (older gap): a FAILED child closed DONE directly also
                # lifts the block it caused — the parent stayed BLOCKED and
                # its other children were never handed out
                self._check_parent_unblock(node.parent_id)
                self._check_parent_completion(node.parent_id)
            elif status in [TaskStatus.FAILED, TaskStatus.BLOCKED]:
                self._check_parent_failure(node.parent_id)
            elif status in (TaskStatus.PENDING, TaskStatus.READY):
                # §4SU Phase 0 (audit C): a revived child unblocks the parent
                # it had blocked — nothing ever did, so the project sat ACTIVE
                # with open work next_ready_leaf never handed out
                self._check_parent_unblock(node.parent_id)
                # §4SU Phase 3 r3: a parent the owner released (a paused or
                # asked phase) whose children closed or failed meanwhile is
                # judged now — the waits skipped it, and nothing else would
                # (it is not a leaf: it sat PENDING forever)
                # r4: only a RELEASED wait — a DONE parent reopened by the
                # model or the owner must stay open (it re-closed at once and
                # its dependants ran), and a cascade block lifted here must not
                # spend an alternative
                if node.children and prior_status in (TaskStatus.NEEDS_USER, TaskStatus.PAUSED):
                    self._check_parent_completion(node.id)
                    if node.status == status:
                        self._check_parent_failure(node.id)

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
        # Coerce here too (belt to load_from_json's braces): a node built
        # outside load_from_json can still carry a bare-string
        # postconditions field, and iterating it raw grades per-char.
        for pc_str in _coerce_str_list(node.postconditions):
            from .project_safety import is_human_gate_text as _ihg
            if _ihg(pc_str):
                # Never textually scored (§4UK: a HUMAN_GATE line is a note,
                # not a park): grading it would fail every task that carries
                # one — a result never echoes the gate.
                continue
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
                
    def depends_back(self, start: str, target: str) -> bool:
        """True when ``start`` really waits, directly or not, on ``target``
        (``start`` itself included) — a dependency that waits on its waiter.
        Only edges that hold something are followed (r23–r25, checked by a
        30,000-tree fuzz against the old rule — tests/test_4qa_dependency_fuzz.py):
          * a LEAF waits on its ``depends_on`` — not on unknown ids nor on
            its own ancestors (open_dependencies does not hold those; r25:
            following them made a sibling's wait look like a cycle);
          * an ALL/BEST parent waits on any open child (it closes after
            them); never on its own ``depends_on``, which holds nothing (r24);
          * an ANY parent waits only when EVERY open child does, and not at
            all once one child is DONE (r25: skipping ANY deadlocked real
            cycles through it);
          * a DONE node waits on nothing."""
        nodes, memo = self.nodes, {}

        def _anc(nid: str) -> set:
            a, c = set(), nodes[nid]
            while c.parent_id and c.parent_id in nodes and c.parent_id not in a:
                a.add(c.parent_id)
                c = nodes[c.parent_id]
            return a

        cuts = [0]  # r26: a False reached by hitting the path depends on the path

        def _waits(nid: str, path: frozenset) -> bool:
            if nid == target:
                return True
            if nid not in nodes:
                return False
            if nid in path:
                cuts[0] += 1
                return False
            if nid in memo:
                return memo[nid]
            before = cuts[0]
            n = nodes[nid]
            if n.status == TaskStatus.DONE:
                res = False
            else:
                path = path | {nid}
                if n.children:
                    kids = [c for c in n.children if c in nodes]
                    if n.dependency_type == DependencyType.ANY:
                        res = (bool(kids)
                               and not any(nodes[c].status == TaskStatus.DONE for c in kids)
                               and all(_waits(c, path) for c in kids))
                    else:
                        res = any(_waits(c, path) for c in kids)
                else:
                    a = _anc(nid)
                    res = any(_waits(d, path) for d in (n.depends_on or [])
                              if d in nodes and d not in a)
            if cuts[0] == before:  # cache only path-independent answers
                memo[nid] = res
            return res

        return _waits(start, frozenset())

    def open_dependencies(self, node_id: str) -> List[str]:
        """§4QA-d: the known ``depends_on`` of ``node_id`` that are not DONE.
        Not held (structure, or "a bad reference can't deadlock the plan"):
        unknown ids, a dependency inside the node's own subtree (a phase that
        waits on one of its steps), an ancestor (a step that "depends on" its
        own phase) and a dependency that waits back on the node (a cycle)."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        anc, cur = set(), node
        while cur.parent_id and cur.parent_id in self.nodes and cur.parent_id not in anc:
            anc.add(cur.parent_id)
            cur = self.nodes[cur.parent_id]

        def _under(dep_id: str) -> bool:
            d, hops = self.nodes.get(dep_id), 0
            while d is not None and d.parent_id and hops < 64:
                if d.parent_id == node_id:
                    return True
                d, hops = self.nodes.get(d.parent_id), hops + 1
            return False

        return [d for d in (node.depends_on or [])
                if d in self.nodes and d != node_id and d not in anc
                and self.nodes[d].status != TaskStatus.DONE
                and not _under(d) and not self.depends_back(d, node_id)]

    def settle_waits(self) -> None:
        """§4SU Phase 3: a node that is not waiting carries no wait; a
        waiting node set by a path that named none gets the safe default
        (``owner`` / ``dependency`` — never a hold)."""
        from .task_wait import default_wait
        for n in self.nodes.values():
            if n.status in (TaskStatus.NEEDS_USER, TaskStatus.BLOCKED):
                if not n.wait_on:
                    n.set_wait(*default_wait(n.status.value), n.result_summary)
            elif n.wait_on or n.wait_code or n.wait_why:
                n.set_wait("", "", "")

    def _rollup_held(self, parent, result_text: str = "") -> bool:
        """§4SU Phase 1: the roll-up closes a parent through the same close
        check as every path. A refusal of
        the check ProjectPlan installs (``close_checker`` — the declared
        file, a model run with no transcript (§4ST: the roll-up bypass), the
        postconditions, judged against what the children reported) makes it
        wait for the owner (NEEDS_USER, "not closed: …"), recorded like a park."""
        if parent.status == TaskStatus.DONE:
            # r1 M2: a parent already DONE (the owner's click over a gate, a
            # waiver, an approval) is never re-judged by a later child close
            return False
        chk = getattr(self, "close_checker", None)
        if chk is None:
            return False
        try:
            why = chk(parent, result_text)
        except Exception:  # noqa: BLE001
            import logging as _lg
            _lg.getLogger("GhostAgent").warning("roll-up close check failed (the roll-up stands)", exc_info=True)
            return False
        if not why:
            return False
        new_summary = f"not closed: {why}"[:2000]
        _gate = str(why).split(":", 1)[0].strip().lower().replace(" ", "_") if ":" in str(why) else ""
        # r3 D2: a parent refused again for a DIFFERENT reason is re-stated
        # and re-logged (the owner was still told the file was missing);
        # a NEEDS_USER park of another kind is left alone
        if (parent.status != TaskStatus.NEEDS_USER
                or (parent.wait_code.startswith("rollup")
                    and parent.result_summary != new_summary)):
            parent.status = TaskStatus.NEEDS_USER
            parent.set_wait("owner", f"rollup_{_gate}" if _gate else "rollup", why)
            parent.result_summary = new_summary
            if not hasattr(self, "rollup_parked"):
                self.rollup_parked = []
            self.rollup_parked.append(parent.id)
        return True

    def _done_children_text(self, parent) -> str:
        """The done children's results — only text the system did not write
        (r2 M2: a hold or refusal quotes the postcondition it refused)."""
        try:
            from ..tools.projects import _stored_evidence
            return " ".join(_stored_evidence(self.nodes[c].result_summary) for c in parent.children
                            if c in self.nodes and self.nodes[c].status == TaskStatus.DONE)[:4000]
        except Exception:  # noqa: BLE001 — r3 L3: never escape after a status was set
            import logging as _lg
            _lg.getLogger("GhostAgent").warning("roll-up evidence text failed (judged on none)", exc_info=True)
            return ""

    @staticmethod
    def _evidence_of(text) -> str:
        try:
            from ..tools.projects import _stored_evidence
            return _stored_evidence(text)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _owner_own_wait(parent) -> bool:
        """§4SU r2–r4: a parent waiting for the owner for ITS OWN reason
        (paused, asked, review…) — not a park the system makes and re-judges
        (a roll-up refusal, a human-gate park, a parent that carries a gate)."""
        return (parent.status == TaskStatus.NEEDS_USER and parent.wait_code != "human_gate"
                and not parent.wait_code.startswith("rollup"))

    def _check_parent_completion(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)

        parent = self.nodes[parent_id]
        if not parent.children: return
        # §4SU Phase 3 r2: a parent that waits for the owner for its own
        # reason (paused, asked, review…) is not closed by its children — only
        # a roll-up refusal or a human-gate park is re-judged here
        # (a gated one goes on: its roll-up parks it as the gate, r1)
        if self._owner_own_wait(parent):
            return

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
                if self._rollup_held(parent, self._done_children_text(parent)):
                    return
                if (parent.wait_code.startswith("rollup")
                        or str(parent.result_summary or "").startswith("not closed:")):
                    # r1 D2: an earlier refusal's text is not the DONE record
                    # (r5: also after the park was replaced by a cascade block)
                    parent.result_summary = self._done_children_text(parent)[:2000]
                parent.status = TaskStatus.DONE
                # Same rule as the direct DONE path in update_status: a node
                # that reaches DONE has no CURRENT failure. A revived child
                # closing DONE promotes a once-FAILED parent here, and the
                # stale reason would otherwise survive on the DONE row.
                parent.failure_reason = ""
                self._check_parent_unblock(parent.parent_id)   # r7: a roll-up close lifts the block above too
                self._check_parent_completion(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.ANY:
            # First successful child satisfies the parent (OR-branch)
            if any(s == TaskStatus.DONE for s in child_statuses):
                if self._rollup_held(parent, self._done_children_text(parent)):
                    return
                parent.status = TaskStatus.DONE
                parent.failure_reason = ""
                # Collect the winning child's result
                for cid in parent.children:
                    child = self.nodes.get(cid)
                    if child and child.status == TaskStatus.DONE:
                        parent.result_summary = child.result_summary
                        parent.actual_tool_used = child.actual_tool_used
                        break
                self._check_parent_unblock(parent.parent_id)   # r7: a roll-up close lifts the block above too
                self._check_parent_completion(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.BEST:
            # All children must be terminal, then pick best. BLOCKED counts as
            # terminal — a BEST child whose own subtree failed (no alternatives)
            # is BLOCKED, and excluding it left a BEST parent unresolvable
            # forever even when a SIBLING child finished DONE.
            terminal = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}
            if child_statuses and all(s in terminal for s in child_statuses):
                done_children = [
                    self.nodes[cid] for cid in parent.children
                    if cid in self.nodes and self.nodes[cid].status == TaskStatus.DONE
                ]
                if done_children:
                    # Pick the child with the longest result (heuristic for quality)
                    # r4 M1: by the length of what the system did not write —
                    # a longer hold / gate text hid a real sibling's result
                    best = max(done_children, key=lambda c: (len(self._evidence_of(c.result_summary)),
                                                             len(c.result_summary or "")))
                    # r3 M1: judged on the best child's text only when the
                    # system did not write it (as the ALL / ANY branches)
                    if self._rollup_held(parent, self._evidence_of(best.result_summary)):
                        # r3: park BEFORE copying — on a re-roll of a parked
                        # parent the child's prose replaced the gate text and
                        # read as close evidence
                        return
                    parent.result_summary = best.result_summary
                    parent.actual_tool_used = best.actual_tool_used
                    parent.status = TaskStatus.DONE
                    parent.failure_reason = ""
                else:
                    parent.status = TaskStatus.FAILED
                    parent.failure_reason = "All BEST-dependency children failed"
                self._check_parent_unblock(parent.parent_id)   # r7: a roll-up close lifts the block above too
                self._check_parent_completion(parent.parent_id, visited)

    def _check_parent_unblock(self, parent_id: str, visited: set = None):
        """The reverse of the failure cascade: a BLOCKED parent whose children
        no longer block it (by its dependency type) is open again, and so on
        up the tree."""
        if visited is None:
            visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited:
            return
        visited.add(parent_id)
        parent = self.nodes[parent_id]
        # r5: only the cascade's own block (code child_failed) — a parent the
        # model blocked was reopened when any child closed DONE
        if parent.status != TaskStatus.BLOCKED or parent.wait_code != "child_failed":
            return
        # (r6: the code is the proof — the r1 failure_reason test also kept a
        # cascade block whose parent had an old, unrelated reason, forever)
        failed = {TaskStatus.FAILED, TaskStatus.BLOCKED}
        child_statuses = [self.nodes[c].status for c in parent.children if c in self.nodes]
        if parent.dependency_type == DependencyType.ANY:
            still = bool(child_statuses) and all(st in failed for st in child_statuses)
        else:
            still = any(st in failed for st in child_statuses)
        if still:
            return
        parent.status = TaskStatus.PENDING
        parent.failure_reason = ""
        self._check_parent_unblock(parent.parent_id, visited)

    def _check_parent_failure(self, parent_id: str, visited: set = None):
        if visited is None: visited = set()
        if not parent_id or parent_id not in self.nodes or parent_id in visited: return
        visited.add(parent_id)

        parent = self.nodes[parent_id]

        if parent.status in [TaskStatus.DONE, TaskStatus.FAILED]:
            return
        # §4SU Phase 3 r2: a parent waiting for the owner keeps that wait — the
        # cascade's BLOCKED (dependency/child_failed) was later unblocked to
        # PENDING, and a paused or asked phase then ran with no release
        # r4: only its OWN wait — a gate or roll-up park over a failed child
        # must block (it was approvable with a FAILED child under it)
        if self._owner_own_wait(parent):
            return
        # r6: a parent blocked for its OWN reason keeps it — re-coded by the
        # cascade, the block was later lifted with the reason lost
        if parent.status == TaskStatus.BLOCKED and parent.wait_code != "child_failed":
            return

        child_statuses = [
            self.nodes[cid].status for cid in parent.children
            if cid in self.nodes
        ]
        failed_statuses = {TaskStatus.FAILED, TaskStatus.BLOCKED}

        if parent.dependency_type == DependencyType.ALL:
            # Any child failure blocks the parent (strict)
            if any(s in failed_statuses for s in child_statuses):
                # Try alternatives before blocking. The queue can hold
                # dangling ids (planner referenced a node that was never
                # created); consume those until one resolves — blocking on
                # the FIRST dangling id stranded valid alternatives still
                # queued behind it.
                alt_id = None
                while parent.alternatives:
                    _cand = parent.alternatives.pop(0)
                    if _cand in self.nodes:
                        alt_id = _cand
                        break
                    import logging as _logging
                    _logging.getLogger("GhostAgent").warning(
                        "TaskTree: dropping dangling alternative %r on parent %r",
                        _cand, parent.id,
                    )
                if alt_id is not None:
                    self.nodes[alt_id].status = TaskStatus.PENDING   # §4SU Phase 3: READY folded
                    # Unlink the failed child(ren) the alternative
                    # supersedes. Leaving them in the rollup makes an
                    # ALL parent unsatisfiable: `all(s == DONE)` can
                    # never hold with a FAILED child still counted, so
                    # the parent would hang forever even after the
                    # alternative succeeds. The nodes stay in
                    # self.nodes for history; they just stop counting.
                    _unlinked = [
                        cid for cid in parent.children
                        if cid in self.nodes
                        and self.nodes[cid].status in failed_statuses
                    ]
                    parent.children = [
                        cid for cid in parent.children if cid not in _unlinked
                    ]
                    # Also clear the unlinked child's parent_id. The store
                    # persists ONLY parent_id and rebuilds `children` from it
                    # on reload, so a failed child left pointing at the parent
                    # re-links after any restart → the ALL parent becomes
                    # unsatisfiable again (FAILED child back in the rollup),
                    # defeating alternative recovery across sessions.
                    for _cid in _unlinked:
                        self.nodes[_cid].parent_id = None
                    # Integrate the alternative into the parent's children
                    # so completion/failure cascading sees it.
                    if alt_id not in parent.children:
                        parent.children.append(alt_id)
                        self.nodes[alt_id].parent_id = parent_id
                    return  # Don't block — alternative is being tried
                parent.status = TaskStatus.BLOCKED
                parent.set_wait("dependency", "child_failed", "a subtask failed")   # §4SU Phase 3 r1
                self._check_parent_failure(parent.parent_id, visited)

        elif parent.dependency_type == DependencyType.ANY:
            # Only block if ALL children have failed. Guard empty: all([])
            # is vacuously True and would BLOCK a parent whose child IDs
            # are all dangling.
            if child_statuses and all(s in failed_statuses for s in child_statuses):
                parent.status = TaskStatus.BLOCKED
                parent.set_wait("dependency", "child_failed", "a subtask failed")   # §4SU Phase 3 r1
                parent.failure_reason = "All ANY-dependency children failed"
                self._check_parent_failure(parent.parent_id, visited)
            # Otherwise, some children are still running — don't block yet

        elif parent.dependency_type == DependencyType.BEST:
            # Don't block until all children are terminal. BLOCKED counts as
            # terminal (a child whose subtree failed) — otherwise a BEST parent
            # with a BLOCKED child never resolves even when a sibling is DONE.
            terminal = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}
            if child_statuses and all(s in terminal for s in child_statuses):
                # If all failed/blocked, parent fails
                if all(s in {TaskStatus.FAILED, TaskStatus.BLOCKED} for s in child_statuses):
                    parent.status = TaskStatus.FAILED
                    parent.failure_reason = "All BEST-dependency children failed"
                    self._check_parent_failure(parent.parent_id, visited)
                else:
                    # Mixed outcomes with at least one DONE: when the LAST
                    # child to finish FAILED, only this failure path runs —
                    # _check_parent_completion never re-fires, so without
                    # this delegation the parent stays unresolved forever
                    # and the winning result is never rolled up.
                    self._check_parent_completion(parent_id)

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
        
        _k = f"<{node.kind}> " if getattr(node, "kind", "") else ""     # §4SU Phase 2: the declared kind
        lines.append(f"{indent}{icon} [{node.id}] {_k}{node.description} ({node.status.value})")
        for child_id in node.children:
            self._render_node(child_id, depth + 1, lines, visited)

    def load_from_json(self, json_data: Any):
        if not json_data: return
        
        # Stateful merge: do not clear self.nodes or self.root_id
        
        def traverse(node_data: Any, parent_id: Optional[str] = None, visited: set = None):
            if visited is None: visited = set()
            if not isinstance(node_data, dict): return
            
            # Everything below is planner (LLM) JSON — coerce each field
            # defensively before the now-live plan gate reads it: a null
            # status crashed `.upper()` (aborting the whole planning
            # step) and a bare-string postconditions field was iterated
            # per-char by _check_postconditions.
            node_id = node_data.get("id")
            if node_id is None or (isinstance(node_id, str) and not node_id.strip()):
                node_id = str(uuid.uuid4())[:4]
            elif not isinstance(node_id, str):
                node_id = str(node_id)
            if node_id in visited: return
            visited.add(node_id)
            desc = node_data.get("description") or "Unknown Task"
            if not isinstance(desc, str):
                desc = str(desc)
            status_raw = node_data.get("status", "PENDING")
            status_str = (status_raw.strip().upper()
                          if isinstance(status_raw, str) else "PENDING")
            try:
                status = TaskStatus[status_str]
            except KeyError:
                status = TaskStatus.PENDING

            # Parse extended fields
            dep_raw = node_data.get("dependency_type", "ALL")
            dep_type_str = (dep_raw.strip().upper()
                            if isinstance(dep_raw, str) else "ALL")
            try:
                dep_type = DependencyType[dep_type_str]
            except KeyError:
                dep_type = DependencyType.ALL
            alternatives = _coerce_str_list(node_data.get("alternatives"))
            postconditions = _coerce_str_list(node_data.get("postconditions"))
            depends_on = _coerce_str_list(node_data.get("depends_on"))

            if node_id in self.nodes:
                # Update existing node
                node = self.nodes[node_id]
                # ANTI-REGRESSION GUARD: Don't allow DONE tasks to revert.
                # Log a warning when we reject a regression so downstream
                # operators can spot planner confusion (the previous silent
                # `pass` made this invisible).
                #
                # ⚠ §4HZ: the guard protects a TASK, not an id. A re-plan
                # (System 3's crisis pivot) emits a fresh tree that reuses
                # `task_1..task_N` for DIFFERENT work; keying the guard on
                # the id kept the OLD status while the description was
                # overwritten, so "install ecmwf-api" was born DONE and the
                # next planner monologue read "task_1 (install ecmwf-api) is
                # DONE" for a step that never ran (req d594668e, twice —
                # both crisis interventions neutralised). The guard now holds
                # only while the node is recognisably the same task
                # (`same_task`); a re-described id adopts the incoming status.
                _same = same_task(node.description, desc)
                node.description = desc
                if (node.status == TaskStatus.DONE and status != TaskStatus.DONE
                        and _same):
                    import logging as _logging
                    _logging.getLogger("GhostAgent").warning(
                        "TaskTree: rejected status regression for node %r (%s → %s)",
                        node_id, node.status.name if hasattr(node.status, 'name') else node.status,
                        status.name if hasattr(status, 'name') else status,
                    )
                else:
                    if node.status == TaskStatus.DONE and status != TaskStatus.DONE:
                        import logging as _logging
                        _logging.getLogger("GhostAgent").info(
                            "TaskTree: node %r was re-described (%r → %r) — a new "
                            "task under a reused id; adopting its status %s",
                            node_id, node.description[:60], desc[:60],
                            status.name if hasattr(status, 'name') else status,
                        )
                    node.status = status
                # Update extended fields
                node.dependency_type = dep_type
                if alternatives:
                    node.alternatives = alternatives
                if postconditions:
                    node.postconditions = postconditions
                if depends_on:
                    node.depends_on = depends_on
                # Parent ID might theoretically change in a re-org, though rare.
                # If parent_id is provided here (from recursion), we update it.
                if parent_id: node.parent_id = parent_id
            else:
                # Create new node
                node = TaskNode(
                    id=node_id, description=desc, status=status,
                    parent_id=parent_id, children=[],
                    dependency_type=dep_type, alternatives=alternatives,
                    postconditions=postconditions, depends_on=depends_on,
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

    def get_failed_tasks(self) -> List[TaskNode]:
        """Return all tasks with FAILED status and their failure reasons."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]

    def generate_retrospective(self) -> Dict[str, Any]:
        """Generate a structured retrospective of the plan execution.

        Returns a dict with:
        - what worked (DONE tasks)
        - what failed (FAILED tasks with reasons)
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
        total = len(self.nodes)

        # Wall-clock actually spent, from tasks.actual_cost (seconds).
        # The column existed since the schema's birth but nothing wrote
        # it until 2026-07-18 (the advancer now stamps each tick's
        # elapsed time), so treat 0 as "not measured", not "free".
        total_cost_s = sum(
            float(getattr(n, "actual_cost", 0.0) or 0.0)
            for n in self.nodes.values()
        )

        retro = {
            "total_tasks": total,
            "completed": len(done_tasks),
            "failed": len(failed_tasks),
            "success_rate": len(done_tasks) / total if total > 0 else 0.0,
            "total_actual_cost_s": round(total_cost_s, 1),
            "what_worked": [
                {"id": n.id, "description": n.description, "tool": n.actual_tool_used}
                for n in done_tasks
            ],
            "what_failed": [
                {
                    "id": n.id,
                    "description": n.description,
                    "reason": n.failure_reason,
                }
                for n in failed_tasks
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
            if node.actual_tool_used:
                result["actual_tool_used"] = node.actual_tool_used
            if node.dependency_type != DependencyType.ALL:
                result["dependency_type"] = node.dependency_type.value
            if node.alternatives:
                result["alternatives"] = node.alternatives
            if node.postconditions:
                result["postconditions"] = node.postconditions
            if node.depends_on:
                result["depends_on"] = node.depends_on
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
        # §4SU Phase 1: the roll-up runs the one close check (TaskTree._rollup_held)
        self.tree.close_checker = self._rollup_close_check

    def _rollup_close_check(self, parent, result_text: str) -> str:
        """The refusal reason for closing ``parent`` by roll-up, or ""."""
        from .task_close import check_close
        v = check_close(self.store, self.project_id, self.tree, parent.id, actor="rollup",
                        result=result_text, skip={"dependencies", "evidence"})
        if not v.ok:
            return f"{v.first.gate}: {v.first.reason}"
        # operator 11-02 ("wait for the owner"): the constraint AUDIT is an
        # LLM judgement the roll-up does not run. A parent with constraints
        # and files of its own that the audit would read waits instead of
        # closing unaudited — the model's own close (which audits) or the
        # owner's click closes it. A fileless parent has nothing to audit.
        try:
            cons = [str(c) for c in (parent.constraints or [])]
            meta = (self.store.get_project(self.project_id) or {}).get("metadata") or {}
            cons += [str(c) for c in (meta.get("constraints") or [])]
            if cons:
                from .task_deliverable import declared_deliverable
                from ..tools.projects import _files_for_task
                decl = declared_deliverable(parent.description or "")
                if _files_for_task(self.store, self.project_id, parent.id, [decl] if decl else []):
                    return ("constraints: its files have not been audited against "
                            + "; ".join(cons[:3])[:200]
                            + " — close it with task_update (that runs the audit) or by the owner's click")
        except Exception:  # noqa: BLE001 — the mechanical check above stands
            import logging as _lg
            _lg.getLogger("GhostAgent").warning("roll-up constraint hold skipped", exc_info=True)
        return ""

    # ------------------------------------------------------------------ hydrate

    def _hydrate(self):
        rows = self.store.list_tasks(self.project_id)
        if not rows:
            return
        for row in rows:
            try:
                from .task_wait import fold_status
                status = TaskStatus[fold_status(row["status"])]   # §4SU Phase 3: READY/PAUSED → PENDING
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
                actual_tool_used=row["actual_tool_used"],
                dependency_type=dep,
                # _coerce_str_list, not list(): a string smuggled into a
                # row's JSON column would explode into per-char entries.
                alternatives=_coerce_str_list(row.get("alternatives")),
                postconditions=_coerce_str_list(row.get("postconditions")),
                constraints=_coerce_str_list(row.get("constraints")),
                depends_on=_coerce_str_list(row.get("depends_on")),
                estimated_cost=float(row["estimated_cost"] or 0.0),
                actual_cost=float(row["actual_cost"] or 0.0),
                kind=str(row.get("kind") or ""),
                wait_on=str(row.get("wait_on") or ""),
                wait_code=str(row.get("wait_code") or ""),
                wait_why=str(row.get("wait_why") or ""),
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
                 constraints: Optional[List[str]] = None,
                 depends_on: Optional[List[str]] = None,
                 estimated_cost: float = 0.0, kind: str = "") -> str:
        from .task_kind import normalize_kind
        kind = normalize_kind(kind)
        task_id = self.store.add_task(
            self.project_id, description, parent_id=parent_id,
            status=status.value, dependency_type=dependency_type.value,
            alternatives=alternatives, postconditions=postconditions,
            constraints=constraints,
            depends_on=depends_on,
            estimated_cost=estimated_cost,
            kind=kind,
        )
        node = TaskNode(
            id=task_id, description=description, status=status,
            parent_id=parent_id,
            dependency_type=dependency_type,
            alternatives=list(alternatives or []),
            postconditions=list(postconditions or []),
            constraints=list(constraints or []),
            depends_on=list(depends_on or []),
            estimated_cost=estimated_cost,
            kind=kind,
        )
        self.tree.nodes[task_id] = node
        if parent_id and parent_id in self.tree.nodes:
            self.tree.nodes[parent_id].children.append(task_id)
        elif parent_id is None and self.tree.root_id is None:
            self.tree.root_id = task_id
        return task_id

    def update_status(self, task_id: str, status: TaskStatus,
                      result: str = "", failure_reason: str = "",
                      actual_tool: str = "", human_approved: bool = False,
                      postconditions_waived: bool = False, wait=None):
        """Cascade through the in-memory tree, then persist every node
        whose state changed as a side effect.

        The diff tracks ``parent_id`` and ``alternatives`` in addition to
        the status fields because the failure-cascade path
        (``TaskTree._check_parent_failure``) *consumes* an alternative by
        popping it off the parent and re-parenting it onto the failed
        parent. Those mutations are silent on status, so a status-only diff
        never persisted them — on the next hydrate the alternative was no
        longer linked and the fallback edge was lost. Tracking both keys
        makes the alternatives/fallback feature durable across sessions."""
        def _snapshot(n: "TaskNode"):
            return (n.status, n.result_summary, n.failure_reason,
                    n.actual_tool_used, n.parent_id,
                    tuple(n.alternatives), tuple(n.depends_on),
                    n.wait_on, n.wait_code, n.wait_why)

        before = {tid: _snapshot(n) for tid, n in self.tree.nodes.items()}
        self.tree.rollup_parked = []
        self.tree.update_status(task_id, status, result=result,
                                failure_reason=failure_reason,
                                actual_tool=actual_tool,
                                human_approved=human_approved,
                                postconditions_waived=postconditions_waived,
                                wait=wait)
        self._persist_changes(before, _snapshot)

    @staticmethod
    def _snap(n: "TaskNode"):
        return (n.status, n.result_summary, n.failure_reason,
                n.actual_tool_used, n.parent_id,
                tuple(n.alternatives), tuple(n.depends_on),
                n.wait_on, n.wait_code, n.wait_why)

    def recheck_parents(self, parent_ids) -> None:
        """§4SU Phase 4 r1 (replan): re-run the roll-up of parents whose open
        children were removed — a parent left with only DONE children was
        never closed (no child close fires it) and stayed PENDING forever."""
        before = {tid: self._snap(n) for tid, n in self.tree.nodes.items()}
        self.tree.rollup_parked = []
        for pid in parent_ids or []:
            if pid in self.tree.nodes:
                self.tree._check_parent_completion(pid)
        self._persist_changes(before, self._snap)

    def _persist_changes(self, before, _snapshot) -> None:
        self.tree.settle_waits()
        for tid, node in self.tree.nodes.items():
            prev = before.get(tid)
            current = _snapshot(node)
            if prev != current:
                self.store.update_task(
                    tid,
                    status=node.status.value,
                    result_summary=node.result_summary,
                    failure_reason=node.failure_reason,
                    actual_tool_used=node.actual_tool_used,
                    parent_id=node.parent_id,
                    alternatives=node.alternatives,
                    depends_on=node.depends_on,
                    wait_on=node.wait_on, wait_code=node.wait_code, wait_why=node.wait_why,
                )
        # §4SU Phase 1: a parent the close check refused at roll-up is recorded (the digest reports it)
        for _pid_parked in list(getattr(self.tree, "rollup_parked", []) or []):
            try:
                from ..utils.logging import request_id_context
                _n = self.tree.nodes.get(_pid_parked)
                self.store.log_event(self.project_id, _pid_parked, "rollup_refused",
                                     {"reason": str(_n.result_summary or "")[:300] if _n else "",
                                      "req": str(request_id_context.get() or "")})
            except Exception:  # noqa: BLE001
                import logging as _lg
                _lg.getLogger("GhostAgent").warning("roll-up park not recorded", exc_info=True)

    def decompose(self, task_id: str,
                  subtask_descriptions: List[str],
                  sequential: bool = False,
                  kinds: Optional[List[str]] = None) -> List[str]:
        """Expand a task into ordered subtasks. Returns new task ids.

        When ``sequential`` is True each subtask declares a ``depends_on``
        edge to the previous one, so the autoadvancer runs them strictly in
        order (subtask N+1 only becomes eligible once subtask N is DONE).
        The default (False) preserves the historical behaviour where all
        subtasks are independently eligible.
        """
        if task_id not in self.tree.nodes:
            raise ValueError(f"unknown task: {task_id}")
        ids: List[str] = []
        prev_id: Optional[str] = None
        kinds = list(kinds or [])
        for i, desc in enumerate(subtask_descriptions):
            if not desc or not desc.strip():
                continue
            deps = [prev_id] if (sequential and prev_id) else None
            new_id = self.add_task(desc.strip(), parent_id=task_id,
                                   depends_on=deps,
                                   kind=kinds[i] if i < len(kinds) else "")   # §4SU Phase 2
            ids.append(new_id)
            prev_id = new_id
        return ids

    # ------------------------------------------------------------------ query

    def next_ready_leaf(self, skip=None) -> Optional[TaskNode]:
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
            # visited-set guard: a parent_id cycle (A→B→A — the store
            # explicitly treats these as reachable) would otherwise spin this
            # loop forever and hang next_ready_leaf. Every other traversal in
            # this file already guards against cycles; this one didn't.
            seen = {cur.id}
            while cur.parent_id:
                parent = self.tree.nodes.get(cur.parent_id)
                if not parent:
                    return False
                if parent.status in blocking:
                    return True
                if parent.id in seen:
                    return False  # cycle — stop rather than loop
                seen.add(parent.id)
                cur = parent
            return False

        def deps_satisfied(node: TaskNode) -> bool:
            """Every declared prerequisite must be DONE before this task is
            eligible. Unknown dep ids (stale/typo'd) are treated as
            satisfied so a bad reference can't deadlock the whole plan;
            a known dep in any non-DONE state holds this task back.

            A depends_on CYCLE (A→B→A — unlike set_dependency, add_task /
            decompose don't reject these) would leave every task in the
            cycle permanently ineligible, so next_ready_leaf returns None
            and advance_many reports "project_done" on a plan that still
            holds open PENDING work. A dependency that waits back on the
            node (through depends_on or a parent's children) is therefore
            not held, so the cycle can drain."""
            # §4QA-d r22: the one rule task_update's DONE gate also uses
            # (TaskTree.open_dependencies) — unknown ids, a step's own phase
            # and a dependency that waits back on the node (a cycle) are
            # not held; every other open dependency is
            return not self.tree.open_dependencies(node.id)

        for node in self.tree.nodes.values():
            if node.children:
                continue
            if node.status not in eligible:
                continue
            if ancestor_blocked(node):
                continue
            if not deps_satisfied(node):
                continue
            if skip is not None and skip(node):
                continue
            return node
        return None

    def render(self) -> str:
        return self.tree.render()

    def to_json(self) -> Dict[str, Any]:
        return self.tree.to_json()

    def generate_retrospective(self) -> Dict[str, Any]:
        return self.tree.generate_retrospective()


# §4UL (review: req c6b9f65d wrote findings.md over two experiments the batch had skipped): plan
# order is a SOFT dependency. A task whose own depends_on is empty does not start while a task
# BEFORE it among its siblings (or inside such a sibling) waits (held for the owner, blocked,
# paused), is still running, or was skipped by the run (``excluded``: lab-held tasks, model
# experiments the batch stopped at). A FAILED task does NOT hold later work (§4UL review M1: one
# failure silently stopped the whole plan); it is reported, and the owner re-runs it. A task that declares its
# own depends_on is governed by that list only; a task the plan declared independent
# (``depends_on: []`` in decompose/task_update → project metadata ``independent_tasks``) is never
# held by order. In the owner's own run the held tasks are counted as PENDING for the pick, so the
# owner's run takes them first instead of stopping behind them.
SOFT_ORDER_STOPS = frozenset({TaskStatus.NEEDS_USER, TaskStatus.BLOCKED,
                              TaskStatus.PAUSED, TaskStatus.IN_PROGRESS})


def independent_task_ids(project_meta) -> set:
    """The ids declared independent of plan order (project metadata ``independent_tasks``)."""
    try:
        return {str(x) for x in ((project_meta or {}).get("independent_tasks") or [])}
    except Exception:  # noqa: BLE001
        return set()


def soft_order_blocked(plan_or_tree, excluded=(), independent=()) -> set:
    """Ids held back by plan order (see SOFT_ORDER_STOPS above): every task (and its subtree)
    after a sibling that waits, runs or is ``excluded`` (never one that merely failed), unless it
    declares its own depends_on or is ``independent``."""
    return set(soft_order_stoppers(plan_or_tree, excluded, independent))


def soft_order_stoppers(plan_or_tree, excluded=(), independent=()) -> Dict[str, set]:
    """§4UL review M1: the same plan-order rule as :func:`soft_order_blocked`, keyed by the held
    id, with the EARLIER sibling(s) that hold it (at its own level or an ancestor's) as the value,
    so a caller can say WHAT a held task waits behind (the stuck-project check)."""
    tree = getattr(plan_or_tree, "tree", plan_or_tree)
    nodes = getattr(tree, "nodes", None) or {}
    excluded, independent = set(excluded or ()), set(independent or ())
    memo: Dict[str, bool] = {}

    def stops(nid: str, seen=None) -> bool:
        if nid in memo:
            return memo[nid]
        seen = set(seen or ()) | {nid}
        n = nodes.get(nid)
        r = bool(n is not None and (nid in excluded or n.status in SOFT_ORDER_STOPS or any(
            stops(c, seen) for c in n.children if c in nodes and c not in seen)))
        memo[nid] = r
        return r

    def subtree(nid: str) -> List[str]:
        out, stack = [], [nid]
        seen: set = set()
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in nodes:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(nodes[cur].children)
        return out

    groups: Dict[Optional[str], List[str]] = {}
    for nid, n in nodes.items():                       # insertion = plan order (depth, position)
        pid = n.parent_id if n.parent_id in nodes else None
        groups.setdefault(pid, []).append(nid)
    held: Dict[str, set] = {}
    for sibs in groups.values():
        stoppers: set = set()
        for nid in sibs:
            n = nodes[nid]
            if stoppers and not n.depends_on and nid not in independent:
                for d in subtree(nid):
                    held.setdefault(d, set()).update(stoppers)
                continue
            if n.status != TaskStatus.DONE and stops(nid):
                stoppers = stoppers | {nid}
    return held


def later_done_dependents(plan_or_tree, task_id: str, independent=()) -> List[str]:
    """§4UL: the DONE tasks that were built on ``task_id`` before it closed — the ones that
    declare it in depends_on, and the later siblings that run in plan order (no depends_on of
    their own, not independent). Their work may predate this close: re-check them."""
    tree = getattr(plan_or_tree, "tree", plan_or_tree)
    nodes = getattr(tree, "nodes", None) or {}
    me = nodes.get(task_id)
    if me is None:
        return []
    independent = set(independent or ())
    out: List[str] = [nid for nid, n in nodes.items()
                      if nid != task_id and n.status == TaskStatus.DONE and task_id in (n.depends_on or [])]
    pid = me.parent_id if me.parent_id in nodes else None
    sibs = [nid for nid, n in nodes.items() if (n.parent_id if n.parent_id in nodes else None) == pid]
    after = sibs[sibs.index(task_id) + 1:] if task_id in sibs else []
    for nid in after:
        n = nodes[nid]
        if (n.status == TaskStatus.DONE and not n.depends_on and nid not in independent
                and nid not in out):
            out.append(nid)
    return out
