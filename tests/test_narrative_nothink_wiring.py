"""Wiring pins: the narrative critique closures must disable thinking.

With thinking ON, the reasoning upstream burns the whole max_tokens
budget inside <think> and returns empty content (verified live — see
core/project_research._llm_complete's docstring). For the two narrative
closures in main.py that failure was SILENT: the summarisers fell back
to template voice every cycle, so the agent spent a full night
(2026-07-13 log) writing 'Lately, I worked on "reply with just: pong"'
diaries. These pins keep the /no_think + enable_thinking=False +
system-nudge pattern from regressing; they inspect the source because
the closures are defined inside main.lifespan and are not importable.
"""

import inspect
import re
from pathlib import Path

import ghost_agent.main as ghost_main

MAIN_SRC = Path(inspect.getsourcefile(ghost_main)).read_text(encoding="utf-8")


def _closure_body(name: str) -> str:
    m = re.search(
        rf"async def {name}\(.*?\n(.*?)\n\s*context\.", MAIN_SRC, re.DOTALL
    )
    assert m, f"{name} not found in main.py"
    return m.group(1)


def test_selfhood_critique_disables_thinking():
    body = _closure_body("_selfhood_critique_fn")
    assert "/no_think" in body
    assert '"enable_thinking": False' in body
    assert "_strip_think" in body


def test_workspace_critique_disables_thinking():
    body = _closure_body("_workspace_critique_fn")
    assert "/no_think" in body
    assert '"enable_thinking": False' in body
    assert "_strip_think" in body
