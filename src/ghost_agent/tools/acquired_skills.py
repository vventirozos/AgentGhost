import json
import logging
import re
import threading
import os
from pathlib import Path

from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")


# Skill names are used as bare filenames (`<skills_dir>/<name>.py`) and
# as registry keys visible in the tool catalogue shown to the LLM. They
# must be pure identifiers — no separators, no traversal.
#
# Prior behaviour wrote `<skills_dir>/../../escaped.py` when the LLM
# hallucinated a name containing `..`, which landed the file OUTSIDE
# the sandbox. Confirmed exploitable: a `learn_skill(name="../../pwn",
# python_code="...")` call would write to the sandbox's parent dir on
# disk. The validation below makes any such name raise before it
# touches the filesystem.
_SAFE_SKILL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class SkillNameError(ValueError):
    """Raised when a skill name fails the identifier-shape check."""


def _validate_skill_name(name: str) -> str:
    """Return `name` if it is a safe identifier; raise otherwise.

    Rules:
      * Non-empty string.
      * Starts with ASCII letter or underscore.
      * Body is ASCII alphanumerics or underscores only (no separators,
        no dots, no `..`).
      * At most 64 chars (generous but caps registry-key bloat).
    """
    if not isinstance(name, str) or not name:
        raise SkillNameError(f"skill name must be a non-empty string, got {name!r}")
    if not _SAFE_SKILL_NAME_RE.match(name):
        raise SkillNameError(
            f"skill name {name!r} rejected: must match "
            f"[A-Za-z_][A-Za-z0-9_]{{0,63}} (no slashes, no '..', no dots, "
            f"no punctuation). This guards the `<skills_dir>/<name>.py` "
            f"write path against traversal."
        )
    return name

class AcquiredSkillManager:
    def __init__(self, sandbox_dir: Path, memory_system):
        self.sandbox_dir = Path(sandbox_dir)
        self.memory_system = memory_system
        self.skills_dir = self.sandbox_dir / "acquired_skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.skills_dir / "skills_registry.json"
        self._lock = threading.RLock()
        
        if not self.registry_path.exists():
            self._save_registry({})

    def _save_registry(self, registry: dict):
        with self._lock:
            temp_path = self.registry_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(registry, indent=2))
            os.replace(temp_path, self.registry_path)

    def _load_registry(self) -> dict:
        with self._lock:
            try:
                content = self.registry_path.read_text()
                return json.loads(content) if content else {}
            except Exception:
                return {}

    def save_skill(self, name: str, description: str, parameters_schema: dict, python_code: str):
        try:
            # Hard-fail on traversal-shaped or separator-bearing names
            # BEFORE any write. See `_validate_skill_name` for the
            # exploit this guards against.
            try:
                name = _validate_skill_name(name)
            except SkillNameError as ve:
                logger.error(f"Rejected unsafe skill name: {ve}")
                pretty_log(
                    "Skill Rejected",
                    f"Unsafe skill name: {name!r}. Must be an identifier.",
                    level="WARNING", icon=Icons.SHIELD,
                )
                return

            import hashlib as _hashlib
            import json as _json
            new_hash = _hashlib.md5(
                (python_code + _json.dumps(parameters_schema, sort_keys=True, default=str) + (description or "")).encode("utf-8")
            ).hexdigest()

            # 1. Write the Python code. Belt-and-braces: even with the
            # name validated above, re-confirm the resolved path sits
            # under `skills_dir` so any future widening of the name
            # regex can't accidentally re-open the traversal.
            skill_path = (self.skills_dir / f"{name}.py").resolve()
            skills_dir_resolved = self.skills_dir.resolve()
            try:
                skill_path.relative_to(skills_dir_resolved)
            except ValueError:
                logger.error(
                    f"Skill path {skill_path} escapes skills_dir {skills_dir_resolved}; refusing write"
                )
                return
            skill_path.write_text(python_code, encoding="utf-8")

            # 2. Add/Update the JSON registry
            with self._lock:
                registry = self._load_registry()

                existing = registry.get(name, {})
                content_unchanged = existing.get("content_hash") == new_hash
                registry[name] = {
                    "name": name,
                    "description": description,
                    "parameters_schema": parameters_schema,
                    "usage_count": existing.get("usage_count", 0),
                    "failure_count": existing.get("failure_count", 0),
                    "status": existing.get("status", "active"),
                    "content_hash": new_hash,
                }

                self._save_registry(registry)

            # 3. Embed the tool description into the memory system — but ONLY
            # if the content actually changed. Re-embedding identical skill
            # text was bloating the vector store on every replan.
            if self.memory_system and not content_unchanged:
                self.memory_system.add(
                    description,
                    {"type": "acquired_skill", "name": name}
                )
                
            logger.info(f"Successfully saved acquired skill: {name}")
            pretty_log("SKILL ACQUIRED", f"Permanently learned new tool: {name}", icon=Icons.MEM_SAVE)
                
        except Exception as e:
            logger.error(f"Failed to save acquired skill {name}: {e}")

    def get_all_skills(self) -> dict:
        """Reads and returns the active registry."""
        return self._load_registry()

    def log_telemetry(self, name: str, success: bool):
        try:
            with self._lock:
                registry = self._load_registry()
                if name not in registry:
                    logger.warning(f"Skill '{name}' not found in registry for telemetry.")
                    return
                
                skill = registry[name]
                skill["usage_count"] += 1
                
                if success:
                    skill["failure_count"] = 0
                    logger.debug(f"Telemetry success logged for skill '{name}'")
                else:
                    skill["failure_count"] += 1
                    logger.warning(f"Telemetry failure logged for skill '{name}'. Threshold: {skill['failure_count']}/3")
                    if skill["failure_count"] >= 3:
                        skill["status"] = "degraded"
                        logger.error(f"Skill '{name}' has degraded due to >= 3 repeated failures.")
                        pretty_log("Skill Degraded", f"Acquired tool '{name}' has been failing and is now flagged.", level="WARNING", icon=Icons.WARN)
                
                self._save_registry(registry)
        except Exception as e:
            logger.error(f"Failed to log telemetry for skill {name}: {e}")

    def retire_degraded_skills(self) -> list:
        """Auto-archive skills that are degraded and underused.

        A skill qualifies for retirement when EITHER condition holds:
        * ``failure_count >= 3`` AND ``failure_count`` represents the
          CURRENT consecutive-failure streak (we use the same counter that
          ``log_telemetry`` resets to 0 on success, so any nonzero value
          IS a consecutive streak — the old "5 + <10 uses" rule was too
          permissive and let chronically-broken skills linger).
        * OR ``failure_count >= 5`` AND ``usage_count < 10`` (legacy rule:
          tool has been failing repeatedly AND isn't valuable enough to
          warrant manual fixing).

        Retired skills are moved to a ``retired/`` subdirectory and removed
        from the active registry and vector store, but their code is
        preserved so they can be manually restored if needed.

        Returns the list of retired skill names.
        """
        retired_names = []
        with self._lock:
            registry = self._load_registry()
            to_retire = []
            for name, info in registry.items():
                failure_count = info.get("failure_count", 0)
                usage_count = info.get("usage_count", 0)
                # `failure_count` resets to 0 on every successful call (see
                # `log_telemetry`), so a nonzero value here is the current
                # consecutive-failure streak.
                consecutive_streak = failure_count >= 3
                low_value_chronic = failure_count >= 5 and usage_count < 10
                if consecutive_streak or low_value_chronic:
                    to_retire.append(name)

            if not to_retire:
                return []

            retired_dir = self.skills_dir / "retired"
            retired_dir.mkdir(parents=True, exist_ok=True)

            for name in to_retire:
                # Move the code file to retired/
                skill_path = self.skills_dir / f"{name}.py"
                if skill_path.exists():
                    try:
                        import shutil
                        shutil.move(str(skill_path), str(retired_dir / f"{name}.py"))
                    except Exception as e:
                        logger.warning(f"Failed to move skill {name} to retired: {e}")

                # Remove from vector store
                if self.memory_system:
                    try:
                        self.memory_system.collection.delete(
                            where={"name": name, "type": "acquired_skill"}
                        )
                    except Exception:
                        pass

                # Remove from active registry
                del registry[name]
                retired_names.append(name)
                pretty_log(
                    "Skill Retired",
                    f"Auto-retired degraded skill '{name}' (failures={to_retire})",
                    level="WARNING", icon="🗄️"
                )
                logger.info(f"Auto-retired skill '{name}'")

            self._save_registry(registry)
        return retired_names

    def delete_skill(self, name: str) -> bool:
        with self._lock:
            registry = self._load_registry()
            if name in registry:
                del registry[name]
                self._save_registry(registry)
                
                # Delete the Python file
                skill_path = self.skills_dir / f"{name}.py"
                if skill_path.exists():
                    try:
                        skill_path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete file {skill_path}: {e}")
                
                # Delete from semantic vector memory
                if self.memory_system:
                    try:
                        self.memory_system.collection.delete(where={"name": name, "type": "acquired_skill"})
                    except Exception as e:
                        logger.warning(f"Failed to remove skill '{name}' from vector memory: {e}")
                        
                return True
            return False

async def tool_create_skill(sandbox_dir: Path = None, memory_system=None, sandbox_manager=None, name: str = None, description: str = None, parameters_schema: str = None, python_code: str = None, test_payload: str = None, **_extra):
    # Tolerate stray kwargs the LLM sometimes invents (observed: `filename`
    # when the model confuses this tool with `execute`). Without this
    # catch-all the registry's `**kwargs` pass-through would raise a
    # TypeError deep inside the tool pathway and the agent would spin
    # retrying the same wrong shape — the real error ("name is mandatory")
    # never gets surfaced. If unknown kwargs arrive, try to rescue the
    # most common confusion (filename → name) so the agent has a chance
    # to recover, and log the rest for post-mortem.
    if _extra:
        if not name and isinstance(_extra.get("filename"), str):
            candidate = _extra["filename"].strip()
            if candidate.endswith(".py"):
                candidate = candidate[:-3]
            name = candidate or None
        logger.warning(
            f"tool_create_skill received unknown kwargs {list(_extra.keys())}; "
            f"ignoring (rescued name={name!r} from filename if present)."
        )
    if not name or not description or not parameters_schema or not python_code or not test_payload:
        return "SYSTEM ERROR: 'name', 'description', 'parameters_schema', 'python_code', and 'test_payload' are MANDATORY."
    import json
    from .execute import tool_execute
    
    try:
        schema_dict = json.loads(parameters_schema)
    except json.JSONDecodeError as e:
        return f"Skill creation failed: invalid parameters_schema JSON -> {e}. Fix the schema and try again."
        
    try:
        # Just to validate it's proper JSON
        json.loads(test_payload)
    except json.JSONDecodeError as e:
        return f"Skill creation failed: invalid test_payload JSON -> {e}. Fix the test payload and try again."

    test_file = sandbox_dir / "test_skill.py"
    
    try:
        test_file.write_text(python_code, encoding="utf-8")
    except Exception as e:
        return f"Skill creation failed: Could not write test file -> {e}"
        
    logger.info(f"Starting TDD test for new skill '{name}'")
    pretty_log("TESTING SKILL", f"Running sandbox test for new skill: {name}", icon=Icons.TOOL_CODE)
        
    execution_result = await tool_execute(
        filename="test_skill.py",
        sandbox_dir=sandbox_dir,
        sandbox_manager=sandbox_manager,
        args=[test_payload]
    )
    
    if "EXIT CODE: 0" not in execution_result or "(Process executed successfully, but no output was printed to stdout" in execution_result:
        try:
            test_file.unlink()
        except Exception:
            pass
        logger.warning(f"TDD failure for '{name}':\n{execution_result}")
        pretty_log("TEST FAILED", f"Skill '{name}' failed its TDD test.", level="WARNING", icon=Icons.FAIL)
        
        if "(Process executed successfully, but no output was printed to stdout" in execution_result:
            return f"Skill creation failed: The script executed successfully but printed absolutely NOTHING to stdout. You MUST print the final result so the system can read it, and ensure you actually parse sys.argv[1] and call your function inside an 'if __name__ == \"__main__\":' block."
        return f"Skill creation failed: {execution_result}. Fix the code and try again."
        
    try:
        test_file.unlink()
    except Exception:
        pass
        
    logger.info(f"TDD passed for '{name}'")
    pretty_log("TEST PASSED", f"Skill '{name}' successfully completed TDD verification.", icon=Icons.OK)
        
    mgr = AcquiredSkillManager(sandbox_dir, memory_system)
    mgr.save_skill(name, description, schema_dict, python_code)
    
    return f"Success: Skill '{name}' acquired and tested successfully."

async def tool_manage_skills(sandbox_dir: Path = None, memory_system=None, action: str = None, skill_name: str = None):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    mgr = AcquiredSkillManager(sandbox_dir, memory_system)
    if action == "list":
        skills = mgr.get_all_skills()
        if not skills:
            return "No custom skills have been acquired yet."
        
        result = "Custom Skills:\n"
        for name, info in skills.items():
            result += f"- {name}: {info.get('description', '')}\n"
        return result
        
    elif action == "delete":
        if not skill_name:
            return "Error: skill_name is required for 'delete' action."
        success = mgr.delete_skill(skill_name)
        if success:
            logger.info(f"Successfully deleted acquired skill: {skill_name}")
            pretty_log("SKILL FORGOTTEN", f"Permanently deleted tool: {skill_name}", icon=Icons.MEM_WIPE)
            return f"Success: Skill '{skill_name}' has been deleted."
        else:
            return f"Error: Skill '{skill_name}' not found."
            
    return f"Error: Unknown action '{action}'."
