"""§4SU Phase 1 — one close check for a project task.

Before this module a task reached DONE four ways — the model's
``task_update``, the web console's PATCH, the autonomous advancer (build and
research) and the parent roll-up — each with its own set of checks, in its
own order, with its own outcome for the same defect (the audit: a missing
declared file was refused on one path, FAILED on another, held on a third and
not checked on the fourth).

``check_close`` is the one verdict. It runs the MECHANICAL gates in a fixed
order and has NO side effects — nothing is registered, logged or written;
each caller acts on the verdict for its actor:

  1. ``dependencies``   — a ``depends_on`` task still open (TaskTree.open_dependencies)
  2. ``deliverable``    — the declared file (``→ path``) missing / hollow (task_deliverable)
  3. ``model_run``      — a model experiment with no kept probe_model transcript (§4ST)
  4. ``evidence``       — a visual task, or a task under constraints, closed with no result
  5. ``postconditions`` — a postcondition the result does not show (the tree's own check)

Two more steps stay with their owners, after these: the constraint AUDIT (an
LLM judgement of the task's files — the model's tool only; its refusal is
reported in the same ``refused`` shape) and the HUMAN GATE (a status
transition in TaskTree.update_status — a park, never a refusal).

Overrides (``override={gate: reason}``): ``model_run`` (a reason of at least
MIN_WAIVER_CHARS, one task per call — §4ST) and ``postconditions`` may be
waived by the model with a reason; the caller records each waiver. (The
model's tool also reads ``override["constraints"]`` — for the constraint
AUDIT, which is outside this check.) The
owner's own click (``actor="owner"``) closes over every gate — each one it
went over is returned in ``overridden`` for the caller to record.

The actor decides what a refusal does (the caller applies it):
  model            → the close is refused, the task unchanged (``refused`` in the reply)
  owner            → closed; the gates it went over are recorded as the owner's overrides
  advancer_owner   → FAILED with the reason (the owner's own advance)
  advancer_unattended → held for the owner (NEEDS_USER, hold kind = the gate)
  rollup           → the parent waits for the owner (NEEDS_USER)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("GhostAgent")

#: the fixed order (also the order the reply lists them in)
GATE_ORDER = ("dependencies", "deliverable", "model_run", "evidence", "postconditions")
#: gates the MODEL may waive with a reason (recorded); the owner's click goes over any
#: §4UI: "deliverable" too — a plan that changed may drop its declared file,
#: with a recorded reason (the dense model is trusted to say why; the owner reads it)
MODEL_WAIVABLE = frozenset({"deliverable", "model_run", "postconditions"})
ACTORS = frozenset({"model", "owner", "advancer_owner", "advancer_unattended", "rollup"})


@dataclass
class Refusal:
    gate: str
    reason: str
    fix: str = ""
    sub: str = ""           # e.g. evidence: "visual" | "constraints"

    def as_dict(self, task_id: str = "") -> Dict[str, str]:
        d = {"gate": self.gate, "reason": self.reason[:400], "fix": self.fix[:300]}
        if task_id:
            d = {"task_id": task_id, **d}
        return d


@dataclass
class Verdict:
    ok: bool
    refused: List[Refusal] = field(default_factory=list)
    waived: Dict[str, str] = field(default_factory=dict)        # gate → the model's reason
    overridden: List[Refusal] = field(default_factory=list)     # gates the owner's click went over
    notes: List[Refusal] = field(default_factory=list)          # advisories: the close stands (§4UY P7 r2)

    @property
    def first(self) -> Optional[Refusal]:
        return self.refused[0] if self.refused else None


def _norm_override(override) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(override, dict):
        for k, v in override.items():
            g = str(k or "").strip().lower()
            r = " ".join(str(v or "").split())
            if g and r:
                out[g] = r[:300]
    return out


# ── the gates (each: Optional[Refusal]; pure) ──────────────────────────────

def gate_dependencies(tree, task_id: str) -> Optional[Refusal]:
    try:
        unmet = tree.open_dependencies(task_id)
    except AttributeError:
        unmet = []
    if not unmet:
        return None
    names = ", ".join(f"{d} ({tree.nodes[d].status.value})" for d in unmet[:8] if d in tree.nodes)
    # §4SU r3/r4: one line per open dependency, by what it is — "close it
    # first" pushed the model to force-close an owner wait, a running task,
    # or a failed parent (status=PENDING strands a parent: it is not a leaf)
    lines = []
    for d in unmet[:6]:
        n = tree.nodes.get(d)
        if n is None:
            continue
        st = n.status.value
        code = getattr(n, "wait_code", "") or ""
        on = getattr(n, "wait_on", "") or ""
        failed_like = st == "FAILED" or (st == "BLOCKED" and code == "child_failed")
        if failed_like:
            # r5: name the FAILED LEAVES — reviving a failed parent (or a
            # nested blocked one) to PENDING strands it: it is not a leaf
            leaves = _failed_leaves(tree, d)
            _f = [x for x in leaves if tree.nodes[x].status.value == "FAILED"]
            _b = [x for x in leaves if tree.nodes[x].status.value != "FAILED"]
            if leaves == [d]:
                lines.append(f"{d} failed — redo it (task_update status=PENDING) or replan; "
                             "do not close it unchanged")
            elif leaves:
                lines.append(f"{d} is held up by "
                             + (f"failed task(s) {', '.join(_f[:5])} — redo them (task_update status=PENDING)"
                                if _f else "")
                             + ("; " if _f and _b else "")
                             + (f"blocked task(s) {', '.join(_b[:5])} — tell the owner what they wait on"
                                if _b else "")
                             + " — or replan")
            else:
                # r6–r8: no failed or blocked leaf to name — a parent revived
                # to PENDING is not a leaf and strands; replan is always safe
                lines.append(f"{d} failed — replan it (its subtasks give no single task to redo)")
        elif st == "NEEDS_USER" and on == "owner_turn":
            # §4UL: held only from UNATTENDED runs — in the owner's own conversation, do it
            lines.append(f"{d} was held from an unattended run — do it now (or task_update it back to PENDING)")
        elif st == "BLOCKED":
            # §4UL: fix it yourself when you can; only what the owner alone has (a file, a login) waits
            lines.append(f"{d} is blocked ({(getattr(n, 'wait_why', '') or n.failure_reason or 'no reason')[:80]}) "
                         "— remove the cause yourself if you can, else say what only the owner can give")
        elif st == "NEEDS_USER":
            lines.append(f"{d} waits for something only the owner has — say what, in one line")
        elif st == "IN_PROGRESS":
            lines.append(f"{d} is running now — wait for it")
        else:
            lines.append(f"{d} is open — do it first")
    return Refusal("dependencies", f"{task_id} waits on {names}",
                   "; ".join(lines) + ". Do not retry this one before they are DONE (if the plan changed, "
                   "replan).")


def _failed_leaves(tree, tid: str, _seen=None) -> List[str]:
    """The FAILED or BLOCKED leaves under ``tid`` (``tid`` itself when it is a leaf)."""
    _seen = _seen if _seen is not None else set()
    n = tree.nodes.get(tid)
    if n is None or tid in _seen:
        return []
    _seen.add(tid)
    kids = [c for c in (n.children or []) if c in tree.nodes]
    if kids and n.status.value == "DONE":
        return []                       # r7: a DONE subtree needs nothing redone
    if not kids:
        return [tid] if n.status.value in ("FAILED", "BLOCKED") else []
    out: List[str] = []
    for c in kids:
        out.extend(_failed_leaves(tree, c, _seen))
    return out


_DELIVERABLE_FIX = ("write the declared file at that path — or, if the plan changed, pass a "
                    "`description` that ends with → the real path — or, if the task no longer "
                    "makes a file, override={\"deliverable\": \"<why, in a sentence>\"} for that one task")


def gate_deliverable(workspace, description: str, *, since: Optional[float] = None,
                     touched=()) -> Optional[Refusal]:
    """The declared file must be there with content; with ``since`` also the §4UY P7 stale check
    (`gate_stale_deliverable`)."""
    from .task_deliverable import deliverable_problem
    problem = deliverable_problem(workspace, description)
    if problem:
        return Refusal("deliverable", problem, _DELIVERABLE_FIX)
    return gate_stale_deliverable(workspace, description, since=since, touched=touched)


def gate_stale_deliverable(workspace, description: str, *, since: Optional[float] = None,
                           touched=()) -> Optional[Refusal]:
    """§4UY P7: a declared file that already existed before the task was created (``since``) and that this
    request's own successful tool calls did not write, read or run (``touched``) — existence alone closed a task
    that named a pre-existing file with no work at all; the result's prose is no evidence (review). Where it
    refuses and where it only notes is `_stale_mode` (review 2: a fuzzy judge must not gate the owner)."""
    from .task_deliverable import stale_deliverable_problem
    problem = stale_deliverable_problem(workspace, description, since=since, touched=touched)
    return Refusal("deliverable", problem, _DELIVERABLE_FIX, sub="stale") if problem else None


#: §4UY P7 review 2: the advisory the owner's own close carries instead of a refusal
STALE_NOTE = ("the declared file predates this task and nothing in this request touched it — say in the result "
              "what changed")


def _stale_mode(store, task_id: str, node, actor: str) -> str:
    """§4UY P7 review 2 — "off": an owner-kind task (the owner did it themselves) · "note": a close in the
    OWNER's own turn, in a task turn of the owner's own batch (`_owner_task_turn`), or by the advancer for the
    owner's batch (``advancer_owner``) — advisory, the close stands (review 3) · "refuse": an unattended /
    scheduled / idle close (the case the finding is about)."""
    try:
        kind = str(getattr(node, "kind", "") or (store.get_task(task_id) or {}).get("kind") or "")
    except Exception:  # noqa: BLE001
        kind = ""
    if kind.strip().lower() == "owner":
        return "off"
    if actor == "advancer_owner":
        return "note"
    if actor in ("model", "rollup"):
        try:
            from ..tools.projects import _owner_task_turn, _owner_turn_now
            if _owner_turn_now() or _owner_task_turn():
                return "note"
        except Exception:  # noqa: BLE001
            pass
    return "refuse"


def _touched_now() -> set:
    """The files this request's own successful tool calls wrote, read or ran (`task_deliverable.touched_in`)."""
    try:
        from ..utils.logging import request_id_context
        from .task_deliverable import touched_in
        return touched_in(str(request_id_context.get() or ""))
    except Exception:  # noqa: BLE001
        return set()


def _task_since(store, task_id: str) -> Optional[float]:
    """When the task was created (its declared file must be newer, or touched by this request's own tool
    calls — `_touched_now`) — None unknown."""
    try:
        v = float((store.get_task(task_id) or {}).get("created_at") or 0.0)
        return v or None
    except Exception:  # noqa: BLE001
        return None


def _trusted_kind(store, project_id, task_id, node) -> str:
    from .experiment_gate import planned_kind
    return planned_kind(store, project_id, task_id, str(getattr(node, "kind", "") or ""))


def gate_model_run(store, project_id: str, description: str, *, files: Iterable[str],
                   result_text: str, kind: str = "") -> Optional[Refusal]:
    from .experiment_gate import unrun_experiment_problem
    problem = unrun_experiment_problem(store, project_id, description,
                                       files=list(files or []), result_text=str(result_text or ""),
                                       kind=str(kind or ""))
    if not problem:
        return None
    return Refusal("model_run", problem,
                   "run it with probe_model and cite the kept transcript — or, if its data are "
                   "not a model's answers, override={\"model_run\": \"<why, in a sentence>\"} for that one task")


def gate_evidence(node, *, result: str, live_evidence: str, constraints: List[str]) -> Optional[Refusal]:
    from ..tools.projects import _stored_evidence, _is_visual_artifact_task
    if str(result or "").strip():
        return None
    stored = _stored_evidence(getattr(node, "result_summary", ""))
    if not stored and _is_visual_artifact_task(getattr(node, "description", ""),
                                               str(getattr(node, "kind", "") or "")):
        return Refusal("evidence", "a visual / runnable artifact closed with no verification evidence",
                       "verify it (open it, run it, screenshot it — a browser RENDER_CHECK line counts) "
                       "and pass what you observed as `result`",
                       sub="visual")
    if constraints and not live_evidence and not stored:
        return Refusal("evidence", "the task is under constraints and the close carries no result",
                       "pass a `result` that shows how the work honours: " + "; ".join(constraints[:5]),
                       sub="constraints")
    return None


def gate_postconditions(tree, node, result_text: str) -> Optional[Refusal]:
    """The tree's own textual check, run on the PROPOSED result — so the
    close is refused (or overridden) instead of demoted to FAILED after."""
    if node is None or not getattr(node, "postconditions", None):
        return None
    probe = copy.copy(node)
    if str(result_text or "").strip():
        probe.result_summary = str(result_text)
    else:
        # r1 M1: with no result, only a stored result the SYSTEM did not write
        # counts — a hold or a roll-up refusal quotes the postcondition itself
        from ..tools.projects import _stored_evidence
        probe.result_summary = _stored_evidence(getattr(node, "result_summary", ""))
    try:
        unsat = tree._check_postconditions(probe)
    except Exception:  # noqa: BLE001
        logger.warning("postcondition pre-check failed (the tree still checks)", exc_info=True)
        return None
    if not unsat:
        return None
    return Refusal("postconditions", "not shown by the result: " + "; ".join(str(u) for u in unsat[:5]),
                   "pass a `result` that shows each one was met — or, if it no longer applies, "
                   "override={\"postconditions\": \"<why>\"}")


# ── the verdict ─────────────────────────────────────────────────────────────

def check_close(store, project_id: str, tree, task_id: str, *, actor: str,
                result: str = "", description: Optional[str] = None,
                files: Iterable[str] = (), live_evidence: str = "",
                constraints: Iterable[str] = (), override=None,
                skip: Iterable[str] = (), single_task: bool = True) -> Verdict:
    """The mechanical gates for closing ``task_id`` DONE, in GATE_ORDER.

    ``description``: the text judged for the declared file (a rewrite in the
    same call names the file it really made); the STORED description is what
    the model-run gate reads (a rewrite could drop its words). ``files``: the
    files this close delivers. ``skip``: gates the caller has already
    decided (e.g. ``model_run`` on a re-close of a DONE task). Never raises:
    a gate that crashes is logged and passes (the tree's own checks remain).
    """
    if actor not in ACTORS:
        raise ValueError(f"unknown close actor {actor!r}")
    node = tree.nodes.get(task_id) if tree is not None else None
    stored_desc = str(getattr(node, "description", "") or "")
    judged_desc = description if description else stored_desc
    over = _norm_override(override)
    skip = set(skip or ())
    try:
        ws = store.ensure_workspace(project_id)
    except Exception:  # noqa: BLE001
        ws = None
    files = [f for f in (files or []) if f]
    v = Verdict(ok=True)

    def _deliverable():
        r = gate_deliverable(ws, judged_desc)
        if r is not None:
            return r
        mode = _stale_mode(store, task_id, node, actor)
        if mode == "off":
            return None
        r = gate_stale_deliverable(ws, judged_desc, since=_task_since(store, task_id), touched=_touched_now())
        if r is not None and mode == "note":
            from .task_deliverable import declared_deliverable
            v.notes.append(Refusal("deliverable", f"{declared_deliverable(judged_desc)}: {STALE_NOTE}", "",
                                   sub="stale"))
            return None
        return r

    checks = {
        "dependencies": lambda: gate_dependencies(tree, task_id),
        "deliverable": _deliverable,
        "model_run": lambda: gate_model_run(store, project_id, stored_desc,
                                            files=files, result_text=result,
                                            kind=_trusted_kind(store, project_id, task_id, node)),
        "evidence": lambda: gate_evidence(node, result=result, live_evidence=live_evidence,
                                          constraints=list(constraints or [])),
        "postconditions": lambda: gate_postconditions(tree, node, result or live_evidence),
    }
    for gate in GATE_ORDER:
        if gate in skip:
            continue
        try:
            r = checks[gate]()
        except Exception:  # noqa: BLE001
            logger.warning("close gate %s crashed (passes)", gate, exc_info=True)
            r = None
        if r is None:
            continue
        if actor == "owner":
            v.overridden.append(r)              # the owner's click goes over it — recorded
            continue
        if actor == "model" and gate in MODEL_WAIVABLE and gate in over:
            from .experiment_gate import MIN_WAIVER_CHARS
            if gate in ("model_run", "deliverable") and (len(over[gate]) < MIN_WAIVER_CHARS
                                                          or not single_task):
                v.refused.append(Refusal(gate, r.reason,
                                         f"a waiver needs a reason of at least {MIN_WAIVER_CHARS} "
                                         "characters, for ONE task per call (task_id, not task_ids)"))
                v.ok = False
                return v
            v.waived[gate] = over[gate]
            continue
        v.refused.append(r)
        v.ok = False
        return v                                 # the first refusal decides (fixed order)
    return v


def hold_kind(gate: str) -> str:
    """The advancer's hold kind for a refusal (Phase 5's re-check reads it)."""
    return {"deliverable": "undelivered", "model_run": "unrun_experiment"}.get(gate, f"refused_{gate}")


def instruction_for(refused: List[Dict[str, Any]]) -> str:
    """One short instruction for a ``refused`` list (≤ ~300 chars + the fixes)."""
    if not refused:
        return ""
    _stay = sum(1 for r in refused if "the task stays DONE" in str(r.get("reason") or ""))
    if _stay == len(refused):
        # r8: only refused RE-closes of tasks already DONE — nothing to repair
        head = (f"{len(refused)} re-close(s) changed nothing: the task(s) stay DONE with their earlier "
                "result. Leave them, or re-close with what each fix names. ")
    else:
        # §4SU Phase 4 (the legacy per-gate texts carried it): a result-only
        # call never closes, so the retry must re-pass status=done — the
        # observed spin was gate → result-only → gate, forever
        head = (f"{len(refused)} task(s) were NOT closed — each entry in `refused` names the gate and the fix. "
                "When a fix is yours to do, do it, then retry with status=done AND the result in the same call; "
                "when it says stop, stop. ")
    fixes = " | ".join(f"{r.get('task_id', '')} [{r['gate']}]: {r.get('fix', '')}" for r in refused[:4])
    return (head + fixes)[:900]
