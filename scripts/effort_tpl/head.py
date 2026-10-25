"""Build a representative 27B head (system prompt + tools) for the §4RB template measurement.
Placeholder profile; GHOST_HOME is a temp dir (never the live data)."""
import json, os, sys, types
out = sys.argv[1]
from ghost_agent.core import agent as A
from ghost_agent.tools.registry import get_active_tool_definitions
sys_prompt = (A.SYSTEM_PROMPT.replace("{{PROFILE}}", "(owner profile omitted for this measurement)")
              + A._STABLE_SKILL_INSTRUCTION).replace("\r", "")
class _Any:
    def __init__(self, **kw): self.__dict__.update(kw)
    def __getattr__(self, n): return None
ctx = _Any(args=_Any(native_tools=True))
try:
    tools = get_active_tool_definitions(ctx, query="", disabled=None) or []
except Exception as e:
    print("tools failed:", type(e).__name__, e); raise
json.dump({"system": sys_prompt, "tools": tools}, open(out, "w"))
print(len(sys_prompt), "chars system;", len(tools), "tools;", len(json.dumps(tools)), "chars tools")
