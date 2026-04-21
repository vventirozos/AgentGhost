# src/ghost_agent/tools/composed_skills.py
"""Tool Composition and Macro Learning.

Compiled multi-step tool sequences that the agent has discovered through
repeated use. Unlike single acquired skills, composed skills are
*sequences* of tool calls with conditional branching — reusable
procedures the agent can execute as a single macro.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


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
            for name, skill_data in data.items():
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
                    usage_count=skill_data.get("usage_count", 0),
                    success_count=skill_data.get("success_count", 0),
                    last_used=skill_data.get("last_used", 0),
                    created_at=skill_data.get("created_at", time.time()),
                )
        except Exception as exc:
            logger.warning("Failed to load composed skills: %s", exc)

    def save(self):
        """Persist composed skills to disk."""
        if not self.storage_dir:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._registry_path()
        try:
            data = {name: skill.to_dict() for name, skill in self.skills.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save composed skills: %s", exc)

    def register(self, skill: ComposedSkill) -> bool:
        """Register a new composed skill."""
        if len(self.skills) >= self.MAX_SKILLS:
            # Evict the least-used skill
            worst = min(self.skills.values(), key=lambda s: s.usage_count)
            del self.skills[worst.name]
            logger.info("Evicted composed skill '%s' (usage=%d)", worst.name, worst.usage_count)

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
                             tool_sequence: List[Dict[str, str]],
                             description: str) -> ComposedSkill:
        """Compile a detected tool-call pattern into a ComposedSkill.

        Called by the dream cycle when it detects a recurring sequence.
        """
        steps = []
        for i, entry in enumerate(tool_sequence):
            steps.append(SkillStep(
                tool_name=entry.get("tool", "unknown"),
                description=entry.get("description", f"Step {i+1}"),
                param_template=entry.get("params", {}),
            ))
        skill = ComposedSkill(
            name=pattern_name,
            trigger_description=description,
            steps=steps,
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

    async def execute(self, skill_name: str,
                      executor: Callable,
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a composed skill using the provided tool executor.

        Parameters
        ----------
        skill_name : name of the composed skill
        executor : async callable(tool_name, tool_args) -> result_str
        params : runtime parameters to fill templates

        Returns
        -------
        Dict with 'success', 'results', and 'steps_completed' keys.
        """
        if skill_name not in self.skills:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        skill = self.skills[skill_name]
        params = params or {}
        results = []
        success = True

        active_steps = list(skill.steps)
        step_idx = 0

        while step_idx < len(active_steps):
            step = active_steps[step_idx]
            # Resolve parameter templates
            resolved_args = {}
            for k, v in step.param_template.items():
                if isinstance(v, str) and v.startswith("$"):
                    resolved_args[k] = params.get(v[1:], v)
                else:
                    resolved_args[k] = v

            try:
                result = await executor(step.tool_name, resolved_args)
                result_str = str(result)
                results.append({
                    "step": step.description,
                    "tool": step.tool_name,
                    "result": result_str[:1000],
                    "success": True,
                })

                # Check branch condition
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

        self.record_usage(skill_name, success)
        return {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "total_steps": len(skill.steps),
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
    sandbox_dir = getattr(context, "sandbox_dir", None)
    if sandbox_dir is None:
        return None
    try:
        storage_dir = Path(sandbox_dir) / "composed_skills"
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
