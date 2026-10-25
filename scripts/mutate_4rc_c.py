"""§4RC battery (c) — Default turns record the server's level (2026-10-24).
Reuses the §4QA harness.   python3 scripts/mutate_4rc_c.py [all|check|rerun R1 R2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"
EF = "src/ghost_agent/core/effort.py"
m.PINS = ["tests/test_4rc_effort_default.py", "tests/test_4rb_reasoning_effort.py"]


def _tests():
    return list(m.PINS)


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", EF, "def level_in_render(prompt: str) -> str:", "# ctl\ndef level_in_render(prompt: str) -> str:"),
    ("CONTROL known-bad (must DIE): never recorded", AG,
     "                    if default:\n                        from . import turn_facts as _tf",
     "                    if False:\n                        from . import turn_facts as _tf"),
    ("C1 no sentence reads as nothing", EF, '    return m.group(1) if m else "medium"', '    return m.group(1) if m else ""'),
    ("C2 the lookup sends a level", EF, "        prompt = _render(base_url, timeout)\n",
     '        prompt = _render(base_url, timeout, {"reasoning_effort": "medium"})\n'),
    ("C3 an empty render reads as medium", EF,
     '        return level_in_render(prompt) if prompt else ""',
     '        return level_in_render(prompt or "")'),
    ("C3b a template without levels reads as medium", EF,
     '        if "Reasoning effort is set to low" not in _render(base_url, timeout, {"reasoning_effort": "low"}):\n            return ""\n', ""),
    ("C4 a failed lookup clears the last value", EF,
     "    if level:\n        with _server_default_lock:\n            _server_default.update(level=level, at=time.monotonic())",
     "    with _server_default_lock:\n        _server_default.update(level=level, at=time.monotonic())"),
    ("C5 the busy flag is never cleared", EF, '            _server_default["busy"] = False\n', '            pass\n'),
    ("C6 never stale", EF, "        stale = not level or time.monotonic() - at > SERVER_DEFAULT_TTL_S",
     "        stale = not level"),
    ("C7 two lookups at once", EF, "        start = stale and bool(base_url) and not _server_default[\"busy\"]",
     "        start = stale and bool(base_url)"),
    ("C8 the lookup blocks the turn", EF,
     "        threading.Thread(target=_refresh_in_thread, args=(base_url,), daemon=True,\n                         name=\"effort-server-default\").start()",
     "        _refresh_in_thread(base_url)"),
    ("C15 no lookup at request start", AG,
     "                await self._effort_default_refresh(req_id)   # §4RC: the server's level, for the record\n", ""),
    ("C16 the request lookup runs for a picked level", AG,
     "                    or _effort.parse_override(effort_override_context.get())):\n                return\n            from .llm import _node_key\n            url",
     "                    ):\n                return\n            from .llm import _node_key\n            url"),
    ("C17 the request lookup runs for members", AG,
     '            if (not _effort.enabled() or request_kind(str(req_id or "") or None) != "owner"',
     '            if (not _effort.enabled()'),
    ("C18 the request lookup waits 3 s", AG,
     "                await asyncio.to_thread(_effort.refresh_server_default, url, 1.0)",
     "                await asyncio.to_thread(_effort.refresh_server_default, url)"),
    ("C9 members record too", AG, "                if owner and not _effort.thinking_off(payload):\n                    # §4RC",
     "                if not _effort.thinking_off(payload):\n                    # §4RC"),
    ("C10 thinking-off turns record", AG, "                if owner and not _effort.thinking_off(payload):\n                    # §4RC",
     "                if owner:\n                    # §4RC"),
    ("C11 no default flag", AG, "reasoning_effort=default,\n                                   reasoning_effort_default=True)",
     "reasoning_effort=default)"),
    ("C12 the /v1 url is used as is", AG,
     '                    default = _effort.server_default(\n                        _node_key(getattr(getattr(self.context, "llm_client", None), "upstream_url", "")))',
     '                    default = _effort.server_default(\n                        getattr(getattr(self.context, "llm_client", None), "upstream_url", ""))'),
    ("C13 every warm-up primes", AG, "            if not quiet and _effort.enabled():\n                # §4RC",
     "            if _effort.enabled():\n                # §4RC"),
    ("C14 the boot warm-up does not prime", AG, "            if not quiet and _effort.enabled():\n                # §4RC",
     "            if False:\n                # §4RC"),
]

if __name__ == "__main__":
    m.main()
