# src/ghost_agent/tools/composed_skills.py
"""Tool Composition and Macro Learning.

Compiled multi-step tool sequences that the agent has discovered through
repeated use. Unlike single acquired skills, composed skills are
*sequences* of tool calls with conditional branching — reusable
procedures the agent can execute as a single macro.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

# A composed-skill name is advertised to the LLM as a top-level tool name
# (see `to_tool_definitions`), so it must be a bare identifier — no spaces,
# dots, slashes, or punctuation that would break the function catalogue or
# let a macro masquerade as a path. Same shape-guard rationale as the
# acquired-skill name check in acquired_skills.py.
_SAFE_COMPOSED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_composed_name(name: str) -> str:
    """Return `name` if it is a safe identifier; raise ValueError otherwise."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"composed-skill name must be a non-empty string, got {name!r}")
    if not _SAFE_COMPOSED_NAME_RE.match(name):
        raise ValueError(
            f"composed-skill name {name!r} rejected: must match "
            f"[A-Za-z_][A-Za-z0-9_]{{0,63}} (it becomes an LLM tool name, so no "
            f"spaces, dots, slashes, or punctuation)."
        )
    return name


# Per-step result cap for composed-skill execution. Each step's body is
# bounded so one chatty step can't blow the context budget when the macro's
# combined output is handed back to the LLM. The original cap was 1000 chars,
# which SILENTLY truncated a list-bearing step — e.g. the morning briefing's
# "latest 10 headlines" step (well over 1000 chars for 10 items) was cut down
# to ~2 headlines, which is exactly the "briefing only shows 2, not 10" bug.
# 4000 chars comfortably carries ~10 headlines while still bounding a runaway
# step; across a handful of steps the macro's total stays context-safe.
MAX_STEP_RESULT_CHARS = 4000


def _cap_step_result(result_str: str, limit: int = MAX_STEP_RESULT_CHARS) -> str:
    """Bound a single step's result body, marking any truncation EXPLICITLY.

    Returns ``result_str`` unchanged when it fits within ``limit``. When it
    does not, truncates to ``limit`` chars and appends a visible marker noting
    how many chars were dropped — so the truncation is never silent. The model
    (and the verifier gate) can then SEE that content was cut and re-fetch the
    step standalone, instead of believing it received the whole list and
    delivering a short answer. That silent-drop-then-believe-it-was-complete
    path is what made the briefing ship 2 (and later 8) of 10 headlines.
    """
    if result_str is None:
        return ""
    if len(result_str) <= limit:
        return result_str
    dropped = len(result_str) - limit
    return (
        f"{result_str[:limit]}\n"
        f"…[truncated {dropped} chars — this step's full output exceeded the "
        f"{limit}-char per-step cap; re-run this step's tool standalone to get "
        f"the complete result]"
    )


def _step_result_ok(result_str: str) -> bool:
    """Classify a step's RESULT as success/failure.

    Tools in this codebase RETURN error strings (``"[error] …"``,
    ``"Error: …"``, ``"[SYSTEM ERROR] …"``, ``"SYSTEM BLOCK …"``) rather than
    raising, so a macro that only checks "did the executor raise" records an
    all-failed run as a success (inflating success_rate and telling the LLM the
    macro worked). Mirror the acquired-skill result gate: inspect the string.
    """
    s = str(result_str or "").lstrip()
    if not s:
        return True  # empty output is not an error
    if "[SYSTEM ERROR]" in s or "SYSTEM BLOCK" in s or "Critical Tool Error" in s:
        return False
    m = re.search(r"EXIT CODE:\s*(\d+)", s)
    if m:
        return m.group(1) == "0"
    return not s.startswith(
        ("[error]", "Error", "ERROR", "SYSTEM ERROR", "Traceback")
    )


@dataclass
class SkillStep:
    """A single step in a composed skill."""
    tool_name: str
    description: str
    param_template: Dict[str, str] = field(default_factory=dict)
    # If set, this step branches on a condition in the result
    branch_condition: str = ""  # e.g., "error" means branch if result contains error
    branch_target: str = ""     # Name of the alternative step sequence to follow
    optional: bool = False      # If True, failure doesn't abort the macro

    def to_dict(self) -> dict:
        d = {
            "tool_name": self.tool_name,
            "description": self.description,
            "param_template": self.param_template,
        }
        if self.branch_condition:
            d["branch_condition"] = self.branch_condition
            d["branch_target"] = self.branch_target
        if self.optional:
            d["optional"] = True
        return d


@dataclass
class ComposedSkill:
    """A reusable sequence of tool calls with conditional branching."""
    name: str
    trigger_description: str  # For semantic matching
    steps: List[SkillStep] = field(default_factory=list)
    branches: Dict[str, List[SkillStep]] = field(default_factory=dict)
    # "sequential" (default) runs steps in order with branching support;
    # "parallel" fans every step out concurrently and returns all results
    # — the right mode for an independent read-only bundle like a briefing.
    execution_mode: str = "sequential"
    # "active" macros are advertised to the LLM and dispatchable; "proposed"
    # macros are auto-discovered drafts (mined from the trajectory log by the
    # dream cycle) awaiting user approval — they are stored and listable but
    # deliberately NOT advertised or executable until approved via
    # manage_composed_skills(action="approve").
    status: str = "active"
    usage_count: int = 0
    success_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trigger_description": self.trigger_description,
            "steps": [s.to_dict() for s in self.steps],
            "branches": {k: [s.to_dict() for s in v] for k, v in self.branches.items()},
            "execution_mode": self.execution_mode,
            "status": self.status,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }


class ComposedSkillRegistry:
    """Manages composed skills — discovery, storage, retrieval, and execution."""

    MAX_SKILLS = 50

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir
        self.skills: Dict[str, ComposedSkill] = {}
        self._save_lock = threading.Lock()
        if storage_dir:
            self._load()

    def _registry_path(self) -> Path:
        return self.storage_dir / "composed_skills.json" if self.storage_dir else Path("/dev/null")

    def _load(self):
        """Load composed skills from disk."""
        path = self._registry_path()
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load composed skills: %s", exc)
            return
        # Per-entry try so ONE malformed macro doesn't drop every macro
        # defined after it (single outer try aborted the whole load).
        for name, skill_data in (data or {}).items():
            try:
                steps = [
                    SkillStep(**{k: v for k, v in s.items() if k in SkillStep.__dataclass_fields__})
                    for s in skill_data.get("steps", [])
                ]
                branches = {}
                for branch_name, branch_steps in skill_data.get("branches", {}).items():
                    branches[branch_name] = [
                        SkillStep(**{k: v for k, v in s.items() if k in SkillStep.__dataclass_fields__})
                        for s in branch_steps
                    ]
                self.skills[name] = ComposedSkill(
                    name=name,
                    trigger_description=skill_data.get("trigger_description", ""),
                    steps=steps,
                    branches=branches,
                    execution_mode=skill_data.get("execution_mode", "sequential"),
                    status=skill_data.get("status", "active"),
                    usage_count=skill_data.get("usage_count", 0),
                    success_count=skill_data.get("success_count", 0),
                    last_used=skill_data.get("last_used", 0),
                    created_at=skill_data.get("created_at", time.time()),
                )
            except Exception as exc:
                logger.warning("Skipping malformed composed skill %r: %s", name, exc)

    def save(self):
        """Persist composed skills to disk. Atomic (temp + os.replace) under a
        lock so a concurrent dream-cycle register / a macro's record_usage save
        can't interleave and truncate/corrupt the registry file."""
        if not self.storage_dir:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._registry_path()
        try:
            data = {name: skill.to_dict() for name, skill in self.skills.items()}
            with self._save_lock:
                tmp = path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to save composed skills: %s", exc)

    def register(self, skill: ComposedSkill) -> bool:
        """Register a new composed skill."""
        # Only evict when ADDING a genuinely new name — re-registering an
        # existing macro doesn't grow the count, so it must not evict a
        # bystander.
        if skill.name not in self.skills and len(self.skills) >= self.MAX_SKILLS:
            # Evict proposed (unapproved) drafts before any active macro,
            # then by lowest usage — so a flood of auto-proposals can never
            # push out a macro the user actually approved or uses.
            worst = min(
                self.skills.values(),
                key=lambda s: (s.status == "active", s.usage_count),
            )
            del self.skills[worst.name]
            logger.info(
                "Evicted composed skill '%s' (status=%s, usage=%d)",
                worst.name, worst.status, worst.usage_count,
            )

        self.skills[skill.name] = skill
        self.save()
        logger.info("Registered composed skill: %s (%d steps)", skill.name, len(skill.steps))
        return True

    def find_matching(self, query: str, limit: int = 3) -> List[ComposedSkill]:
        """Find composed skills matching a query by keyword overlap."""
        if not query or not self.skills:
            return []
        query_words = set(query.lower().split())
        scored = []
        for skill in self.skills.values():
            trigger_words = set(skill.trigger_description.lower().split())
            overlap = len(query_words & trigger_words)
            if overlap > 0:
                scored.append((skill, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:limit]]

    def record_usage(self, skill_name: str, success: bool):
        """Record that a composed skill was used."""
        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill.usage_count += 1
            if success:
                skill.success_count += 1
            skill.last_used = time.time()
            self.save()

    def compile_from_pattern(self, pattern_name: str,
                             tool_sequence: List[Dict[str, Any]],
                             description: str,
                             *,
                             status: str = "proposed",
                             execution_mode: str = "sequential") -> ComposedSkill:
        """Compile a detected tool-call pattern into a ComposedSkill.

        Called by the dream cycle when it mines a recurring tool-call
        sequence from the trajectory log. Defaults to ``status="proposed"``
        — an auto-discovered draft that is stored and listable but NOT
        advertised to the LLM or dispatchable until the user approves it via
        ``manage_composed_skills(action="approve")``.
        """
        steps = []
        for i, entry in enumerate(tool_sequence):
            steps.append(SkillStep(
                tool_name=entry.get("tool", "unknown"),
                description=entry.get("description", f"Step {i+1}"),
                param_template=entry.get("params", {}) or {},
            ))
        skill = ComposedSkill(
            name=pattern_name,
            trigger_description=description,
            steps=steps,
            execution_mode=execution_mode,
            status=status,
        )
        self.register(skill)
        return skill

    def to_tool_definitions(self) -> List[dict]:
        """Render each registered composed skill as an LLM-facing tool definition.

        Mirrors the shape used by the static TOOL_DEFINITIONS list in
        registry.py so callers can simply `.extend()` the agent's active
        tool list.
        """
        defs: List[dict] = []
        for name, skill in self.skills.items():
            # Proposed (auto-discovered, unapproved) macros are NOT shown to
            # the LLM — they await user approval first.
            if skill.status != "active":
                continue
            param_keys: set = set()
            for step in skill.steps:
                if isinstance(step.param_template, dict):
                    for v in step.param_template.values():
                        if isinstance(v, str) and v.startswith("$"):
                            param_keys.add(v[1:])
            properties = {
                k: {"type": "string", "description": f"Runtime value for ${k}."}
                for k in sorted(param_keys)
            }
            schema = {
                "type": "object",
                "properties": properties,
                "required": [],  # composed skills don't enforce required runtime params
            }
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"[COMPOSED SKILL] {skill.trigger_description} "
                        f"({len(skill.steps)} steps; "
                        f"used {skill.usage_count}x with {skill.success_rate:.0%} success)"
                    ),
                    "parameters": schema,
                }
            })
        return defs

    @staticmethod
    def _resolve_args(step: "SkillStep", params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a step's `$variable` param templates against runtime params."""
        resolved_args = {}
        for k, v in step.param_template.items():
            if isinstance(v, str) and v.startswith("$"):
                # Resolve $var against runtime params. An UNresolved template
                # must become "" (a missing value), not the literal "$var" —
                # otherwise the step tool receives e.g. location="$city".
                resolved_args[k] = params.get(v[1:], "")
            else:
                resolved_args[k] = v
        return resolved_args

    async def execute(self, skill_name: str,
                      executor: Callable,
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a composed skill using the provided tool executor.

        Dispatches on the skill's ``execution_mode``: ``"parallel"`` fans
        all steps out concurrently (ideal for an independent read-only
        bundle like a morning briefing), ``"sequential"`` (default) runs
        them in order with conditional branching.

        Parameters
        ----------
        skill_name : name of the composed skill
        executor : async callable(tool_name, tool_args) -> result_str
        params : runtime parameters to fill templates

        Returns
        -------
        Dict with 'success', 'results', 'steps_completed', 'total_steps'
        and 'mode' keys.
        """
        if skill_name not in self.skills:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        skill = self.skills[skill_name]
        params = params or {}
        if skill.execution_mode == "parallel":
            return await self._execute_parallel(skill, executor, params)
        return await self._execute_sequential(skill, executor, params)

    async def _execute_sequential(self, skill: "ComposedSkill",
                                  executor: Callable,
                                  params: Dict[str, Any]) -> Dict[str, Any]:
        """Run steps in order, honouring conditional branches and optional steps."""
        results = []
        success = True

        active_steps = list(skill.steps)
        step_idx = 0
        # Bound total step executions so a self-referential branch (hand-
        # authored/loaded branches JSON) can't loop forever issuing tool calls.
        _MAX_STEP_EXECUTIONS = 64
        _executions = 0

        while step_idx < len(active_steps):
            if _executions >= _MAX_STEP_EXECUTIONS:
                results.append({
                    "step": "(aborted)", "tool": "-",
                    "error": f"step-execution cap ({_MAX_STEP_EXECUTIONS}) hit — "
                             "possible branch loop.",
                    "success": False,
                })
                success = False
                break
            _executions += 1
            step = active_steps[step_idx]
            resolved_args = self._resolve_args(step, params)

            try:
                result = await executor(step.tool_name, resolved_args)
                result_str = str(result)
                # Classify from the RESULT (tools return error strings, not raises).
                step_ok = _step_result_ok(result_str)
                results.append({
                    "step": step.description,
                    "tool": step.tool_name,
                    "result": _cap_step_result(result_str),
                    "success": step_ok,
                })
                if not step_ok:
                    if not step.optional:
                        success = False
                        break
                    step_idx += 1
                    continue

                # Check branch condition (only on a genuinely successful step)
                if step.branch_condition and step.branch_condition.lower() in result_str.lower():
                    branch_steps = skill.branches.get(step.branch_target, [])
                    if branch_steps:
                        active_steps = branch_steps
                        step_idx = 0
                        continue
            except Exception as exc:
                results.append({
                    "step": step.description,
                    "tool": step.tool_name,
                    "error": str(exc),
                    "success": False,
                })
                if not step.optional:
                    success = False
                    break

            step_idx += 1

        self.record_usage(skill.name, success)
        return {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "total_steps": len(skill.steps),
            "mode": "sequential",
        }

    async def _execute_parallel(self, skill: "ComposedSkill",
                                executor: Callable,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Fan every step out concurrently, then collect all results.

        Branching does NOT apply in parallel mode — there is no ordered
        result to test a `branch_condition` against, so branches are
        ignored. Every step runs even if a sibling fails; a non-optional
        step failing marks the whole macro failed but never aborts the
        fan-out (we want the briefing's other panels regardless).
        """
        async def _run_step(step: "SkillStep") -> Dict[str, Any]:
            resolved_args = self._resolve_args(step, params)
            try:
                result = await executor(step.tool_name, resolved_args)
                result_str = str(result)
                # Classify from the RESULT — tools return error strings.
                return {
                    "step": step.description,
                    "tool": step.tool_name,
                    "result": _cap_step_result(result_str),
                    "success": _step_result_ok(result_str),
                    "optional": step.optional,
                }
            except Exception as exc:
                return {
                    "step": step.description,
                    "tool": step.tool_name,
                    "error": str(exc),
                    "success": False,
                    "optional": step.optional,
                }

        results = list(await asyncio.gather(*[_run_step(s) for s in skill.steps]))
        # A step is "tolerated" if it succeeded or was declared optional.
        success = all(r["success"] or r.get("optional") for r in results)

        self.record_usage(skill.name, success)
        return {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "total_steps": len(skill.steps),
            "mode": "parallel",
        }


def _registry_from_context(context) -> Optional[ComposedSkillRegistry]:
    """Build (or fetch a cached) ComposedSkillRegistry from a context object.

    The registry caches the loaded skills in-memory so we don't re-read
    the JSON file on every tool-list build. We attach the cached instance
    on the context under `_composed_skill_registry` (best-effort — if the
    context is a frozen MagicMock or otherwise rejects assignment we
    silently fall back to building a fresh registry).
    """
    if context is None:
        return None
    cached = getattr(context, "_composed_skill_registry", None)
    if isinstance(cached, ComposedSkillRegistry):
        return cached
    # Prefer memory_dir so macros persist across sandbox wipes (same
    # rationale as acquired skills); fall back to sandbox_dir for
    # early-init contexts that haven't wired memory_dir yet.
    base = getattr(context, "memory_dir", None) or getattr(context, "sandbox_dir", None)
    if base is None:
        return None
    try:
        storage_dir = Path(base) / "composed_skills"
    except Exception:
        return None
    reg = ComposedSkillRegistry(storage_dir=storage_dir)
    try:
        setattr(context, "_composed_skill_registry", reg)
    except Exception:
        # If the context refuses attribute assignment we just rebuild
        # next time — correctness is preserved, perf takes a small hit.
        pass
    return reg


def register_composed_skills(tool_definitions: list, context) -> int:
    """Mutate ``tool_definitions`` in-place to include composed-skill entries.

    Mirrors how acquired-skill definitions are appended in
    `registry.get_active_tool_definitions`. Returns the number of
    composed-skill entries added.

    Composed-skill names are skipped if they would shadow an existing
    tool entry — same shadowing policy as acquired skills.
    """
    if not isinstance(tool_definitions, list):
        return 0
    reg = _registry_from_context(context)
    if reg is None:
        return 0
    existing_names = {
        t.get("function", {}).get("name")
        for t in tool_definitions
        if isinstance(t, dict)
    }
    added = 0
    for entry in reg.to_tool_definitions():
        name = entry.get("function", {}).get("name")
        if not name or name in existing_names:
            if name:
                logger.warning(
                    "Composed skill '%s' shadows an existing tool — skipping.",
                    name,
                )
            continue
        tool_definitions.append(entry)
        existing_names.add(name)
        added += 1
    if added:
        logger.info("Registered %d composed skill(s) into tool definitions.", added)
    return added


def _format_execution_result(skill_name: str, result: Dict[str, Any]) -> str:
    """Render an `execute()` result dict as a compact, LLM-readable string.

    This is what the agent's dispatch loop hands back as the tool result;
    the model then synthesises the briefing/answer from the per-step
    blocks. Each step's body is already bounded to ``MAX_STEP_RESULT_CHARS``
    by `execute()` (via `_cap_step_result`), so a chatty step can't blow the
    context budget — and any step that DID hit the cap carries an explicit
    truncation marker, so a list-bearing step is never silently shortened.
    """
    # Guard-style failures (unknown skill) carry an 'error' and no 'results'.
    if not result.get("success") and "error" in result and "results" not in result:
        return f"[composed skill '{skill_name}' error] {result.get('error')}"

    mode = result.get("mode", "sequential")
    header = (
        f"COMPOSED SKILL '{skill_name}' — "
        f"{result.get('steps_completed', 0)}/{result.get('total_steps', 0)} steps "
        f"({mode}), overall {'OK' if result.get('success') else 'PARTIAL/FAIL'}."
    )
    blocks = [header]
    for i, r in enumerate(result.get("results", []), 1):
        head = f"[{i}] {r.get('tool')} — {r.get('step')}"
        if r.get("success"):
            blocks.append(f"{head}:\n{r.get('result', '')}")
        else:
            opt = " (optional)" if r.get("optional") else ""
            blocks.append(f"{head}: FAILED{opt} — {r.get('error', 'unknown error')}")
    return "\n\n".join(blocks)


def build_step_executor(tools_ref: Dict[str, Callable], composed_names) -> Callable:
    """Return an async ``(tool_name, args) -> result_str`` dispatcher that
    runs a single composed-skill STEP against the agent's live tool dict.

    ``tools_ref`` is the same dict ``get_available_tools`` builds — captured
    by reference, so it is fully populated by the time a step actually runs.
    ``composed_names`` is the set of composed-skill names; steps are
    FORBIDDEN from invoking another composed skill, which is what stops a
    macro from recursing into itself (or a cycle of macros) and blowing the
    stack.
    """
    composed = set(composed_names or ())

    async def _exec_step(tool_name: str, tool_args: Dict[str, Any]):
        if tool_name in composed:
            return (
                f"[blocked] '{tool_name}' is itself a composed skill; "
                f"composed skills cannot be nested as steps."
            )
        fn = tools_ref.get(tool_name)
        if fn is None:
            return f"[error] step tool '{tool_name}' is not available."
        return await fn(**(tool_args or {}))

    return _exec_step


def make_composed_skill_runner(skill_name: str, registry: "ComposedSkillRegistry",
                               tools_ref: Dict[str, Callable], composed_names) -> Callable:
    """Build the top-level tool runner for one composed skill.

    The returned coroutine is what ``get_available_tools`` registers under
    the macro's name. Calling it fans the macro's steps out through
    ``build_step_executor`` and returns a formatted, LLM-readable summary.
    """
    async def _run(**kwargs):
        skill = registry.skills.get(skill_name)
        nsteps = len(skill.steps) if skill else "?"
        mode = skill.execution_mode if skill else "?"
        pretty_log(
            "Composed Skill",
            f"Running macro '{skill_name}' ({nsteps} steps, {mode}).",
            icon=Icons.BRAIN_PLAN,
        )
        executor = build_step_executor(tools_ref, composed_names)
        result = await registry.execute(skill_name, executor, params=kwargs)
        return _format_execution_result(skill_name, result)

    return _run


def register_composed_skill_runners(tools: Dict[str, Callable], context) -> int:
    """Mutate the ``tools`` executor dict in-place to add a runner for each
    registered composed skill. Counterpart to ``register_composed_skills``
    (which adds the LLM-facing DEFINITIONS) — together they make a macro
    both visible to the model AND dispatchable.

    A macro name is skipped if it would shadow a built-in / acquired-skill
    runner already in ``tools`` (same shadow policy as the definition side),
    which keeps a macro from hijacking a real tool name.
    """
    if not isinstance(tools, dict):
        return 0
    reg = _registry_from_context(context)
    if reg is None or not reg.skills:
        return 0
    composed_names = set(reg.skills.keys())
    added = 0
    for name, skill in reg.skills.items():
        # Proposed (auto-discovered, unapproved) drafts are not dispatchable
        # until the user approves them.
        if skill.status != "active":
            continue
        if name in tools:
            logger.warning(
                "Composed skill '%s' shadows an existing tool runner — skipping.",
                name,
            )
            continue
        tools[name] = make_composed_skill_runner(name, reg, tools, composed_names)
        added += 1
    if added:
        logger.info("Wired %d composed-skill runner(s) into the tool dispatch.", added)
    return added


async def tool_manage_composed_skills(context=None, action: str = None,
                                      name: str = None, description: str = None,
                                      steps=None, mode: str = "parallel",
                                      known_tools=None, **_extra):
    """Define / list / delete composed skills — named macros that bundle
    several tool calls into ONE invocation.

    Actions
    -------
    define : register a new macro. ``steps`` is a list of
        ``{tool, description, params, optional}`` objects. ``mode`` is
        ``"parallel"`` (default — fan out independent steps) or
        ``"sequential"`` (ordered). The macro becomes a top-level tool the
        agent invokes by ``name``.
    list   : show all registered macros.
    delete : remove one by ``name``.
    """
    if not action:
        return "SYSTEM ERROR: 'action' is MANDATORY (define | list | delete)."
    reg = _registry_from_context(context)
    if reg is None:
        return ("SYSTEM ERROR: composed-skill storage is unavailable "
                "(no sandbox/memory dir on the active context).")
    action = str(action).strip().lower()  # str() so a non-string can't raise

    if action == "list":
        if not reg.skills:
            return "No composed skills defined yet."
        active = [(n, sk) for n, sk in reg.skills.items() if sk.status == "active"]
        proposed = [(n, sk) for n, sk in reg.skills.items() if sk.status != "active"]
        out = ["Composed skills (macros):"]
        for n, sk in active:
            out.append(
                f"- {n} [{sk.execution_mode}] — {sk.trigger_description} "
                f"({len(sk.steps)} steps; used {sk.usage_count}x, "
                f"{sk.success_rate:.0%} ok)"
            )
        if not active:
            out.append("(none active)")
        if proposed:
            out.append("")
            out.append("Proposed (auto-discovered from your tool-use history — approve to activate):")
            for n, sk in proposed:
                seq = " → ".join(s.tool_name for s in sk.steps)
                out.append(f"- {n} [proposed, {sk.execution_mode}] — {sk.trigger_description} (steps: {seq})")
            out.append("")
            out.append(
                "Approve with manage_composed_skills(action='approve', name='<name>'); "
                "reject with action='delete'."
            )
        return "\n".join(out)

    if action == "approve":
        if not name:
            return "Error: 'name' is required for approve."
        if name not in reg.skills:
            return f"Error: composed skill '{name}' not found."
        sk = reg.skills[name]
        if sk.status == "active":
            return f"Composed skill '{name}' is already active."
        sk.status = "active"
        reg.save()
        pretty_log("Macro Approved", f"Activated proposed macro: {name}", icon=Icons.OK)
        return (
            f"Success: composed skill '{name}' approved and activated. It is now "
            f"a top-level tool — invoke it by name. (Its step parameters were "
            f"mined from past calls; delete + redefine if you want to adjust them.)"
        )

    if action == "delete":
        if not name:
            return "Error: 'name' is required for delete."
        if name not in reg.skills:
            return f"Error: composed skill '{name}' not found."
        del reg.skills[name]
        reg.save()
        pretty_log("Macro Forgotten", f"Deleted composed skill: {name}", icon=Icons.MEM_WIPE)
        return f"Success: composed skill '{name}' deleted."

    if action == "define":
        if not name or not description or not steps:
            return ("SYSTEM ERROR: 'name', 'description', and 'steps' are "
                    "MANDATORY for define.")
        try:
            name = _validate_composed_name(name)
        except ValueError as ve:
            return f"Error: {ve}"
        # Reject a name that shadows a built-in / acquired tool: the runner
        # wiring skips such a macro (the built-in wins), so persisting it and
        # telling the model "it's now a TOP-LEVEL TOOL" is a lie.
        if known_tools and name in known_tools:
            return (f"Error: '{name}' is already a built-in/acquired tool; a "
                    "composed skill can't shadow it. Choose a different name.")
        if not isinstance(steps, list) or not steps:
            return "Error: 'steps' must be a non-empty list of step objects."
        mode = str(mode or "parallel").strip().lower()
        if mode not in ("parallel", "sequential"):
            return "Error: 'mode' must be 'parallel' or 'sequential'."

        skill_steps: List[SkillStep] = []
        unknown_tools: List[str] = []
        for i, raw in enumerate(steps):
            if not isinstance(raw, dict):
                return f"Error: step {i + 1} must be an object, got {type(raw).__name__}."
            tool = (raw.get("tool") or raw.get("tool_name") or "").strip()
            if not tool:
                return f"Error: step {i + 1} is missing 'tool'."
            if tool == name:
                return (f"Error: step {i + 1} references the macro itself "
                        f"('{name}') — composed skills cannot recurse.")
            if known_tools and tool not in known_tools:
                unknown_tools.append(tool)
            params = raw.get("params") or raw.get("param_template") or {}
            if not isinstance(params, dict):
                return f"Error: step {i + 1} 'params' must be an object."
            skill_steps.append(SkillStep(
                tool_name=tool,
                description=(raw.get("description") or f"Step {i + 1}"),
                param_template=params,
                optional=bool(raw.get("optional", False)),
            ))

        skill = ComposedSkill(
            name=name,
            trigger_description=description,
            steps=skill_steps,
            execution_mode=mode,
        )
        reg.register(skill)
        pretty_log(
            "Macro Defined",
            f"Composed skill '{name}' ({len(skill_steps)} steps, {mode}).",
            icon=Icons.MEM_SAVE,
        )
        msg = (
            f"Success: composed skill '{name}' defined with {len(skill_steps)} "
            f"steps ({mode} mode). It is now a TOP-LEVEL TOOL — invoke it by "
            f"name like any built-in; its steps run and the combined results "
            f"come back for you to synthesise."
        )
        if unknown_tools:
            msg += (
                f"\nWARNING: these step tools aren't recognised built-ins and "
                f"will error at run time unless they are acquired skills: "
                f"{', '.join(sorted(set(unknown_tools)))}."
            )
        return msg

    return f"Error: unknown action '{action}' (use define | list | approve | delete)."
