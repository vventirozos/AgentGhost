"""Regression tests for two nudges that steer the LLM toward invoking
an acquired skill directly as a tool, instead of wrapping it.

Incident context (2026-04-24, greece_top_news session)
------------------------------------------------------
After the skill-storage relocation (memory_dir canonical path), a user
asked to RUN a previously-created skill. The LLM took 8 turns to
figure out it should invoke the skill by name — it kept trying:

  * ``python -c "from greece_top_news import greece_top_news; ..."``
  * ``ls /workspace/acquired_skills/`` (skill no longer lives there)
  * ``python3 acquired_skills/greece_top_news.py`` (ditto — ENOENT)
  * writing a wrapper ``run_news.py`` that tried ``import greece_top_news``

All of the above missed the point: acquired skills are registered as
top-level callable tools. The fix is two-fold:

  1. Tool description hardening — the description shown to the LLM now
     explicitly says "CALL BY NAME", gives a concrete invocation
     example using the skill's own name, and forbids the wrong
     patterns.
  2. Inline-c block heuristic — when the blocked body looks like a
     skill-wrapping attempt (``from X import X`` or
     ``acquired_skills/X.py``), the SYSTEM BLOCK error message
     appends a targeted hint pointing at the right invocation.

Both nudges together should kill the 8-turn loop.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# 1. Acquired-skill tool description is explicit
# ---------------------------------------------------------------------------


def test_acquired_skill_description_names_the_tool(tmp_path, monkeypatch):
    """The rendered tool description visible to the LLM must include
    the skill's own name in an invocation example + forbid the wrong
    patterns. Without this, the LLM gravitates to `python -c` /
    import / file-read heuristics."""
    from ghost_agent.tools.registry import get_active_tool_definitions
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager

    mem_dir = tmp_path / "memory"
    sandbox = tmp_path / "sandbox"
    mem_dir.mkdir()
    sandbox.mkdir()

    # Pre-populate a skill at the canonical location.
    mgr = AcquiredSkillManager(mem_dir, memory_system=None)
    mgr.save_skill(
        "greece_top_news",
        "Fetches top Greek news stories.",
        {"type": "object", "properties": {"count": {"type": "integer"}}},
        "print('ok')\n",
    )

    # Build a minimal context that get_active_tool_definitions
    # expects.
    ctx = SimpleNamespace(
        sandbox_dir=sandbox,
        memory_dir=mem_dir,
        memory_system=MagicMock(),
        args=SimpleNamespace(model="qwen", anonymous=True, max_context=4000),
        llm_client=MagicMock(image_gen_clients=None),
        tor_proxy=None,
    )
    # Bypass semantic routing — no query, so all active skills are shown.
    defs = get_active_tool_definitions(ctx, query=None)
    skill_def = next(
        (t for t in defs if t.get("function", {}).get("name") == "greece_top_news"),
        None,
    )
    assert skill_def is not None, "skill must be advertised as a tool"
    desc = skill_def["function"]["description"]

    # Invocation example uses the actual skill name.
    assert "greece_top_news(" in desc
    # Explicit "call by name" marker.
    assert "CALL BY NAME" in desc.upper()
    # Forbids the wrong patterns observed in the incident.
    assert "python -c" in desc.lower()
    assert "import" in desc.lower()
    # User's description is preserved (the LLM needs it for deciding
    # WHEN to call).
    assert "Greek news" in desc


def test_acquired_skill_description_mentions_file_location(tmp_path):
    """The description should tell the LLM where the file actually
    lives so it doesn't guess wrong. `$GHOST_HOME/system/memory/…` is
    the canonical path."""
    from ghost_agent.tools.registry import get_active_tool_definitions
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager

    mem_dir = tmp_path / "memory"
    sandbox = tmp_path / "sandbox"
    mem_dir.mkdir()
    sandbox.mkdir()
    AcquiredSkillManager(mem_dir, memory_system=None).save_skill(
        "my_skill", "a skill", {"type": "object", "properties": {}},
        "print('x')\n",
    )
    ctx = SimpleNamespace(
        sandbox_dir=sandbox, memory_dir=mem_dir,
        memory_system=MagicMock(),
        args=SimpleNamespace(model="qwen", anonymous=True, max_context=4000),
        llm_client=MagicMock(image_gen_clients=None),
        tor_proxy=None,
    )
    defs = get_active_tool_definitions(ctx, query=None)
    skill_def = next(t for t in defs if t.get("function", {}).get("name") == "my_skill")
    desc = skill_def["function"]["description"]
    assert "acquired_skills" in desc  # path hint
    # Explains WHY import won't work.
    assert "outside the sandbox" in desc.lower() or "ModuleNotFoundError" in desc


# ---------------------------------------------------------------------------
# 2. Inline -c block error includes skill-invocation nudge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inline_c_block_nudges_toward_direct_invocation_on_from_import(tmp_path):
    """The exact incident shape: LLM runs `python -c "from X import X;
    print(X(...))"`. The block error should tell it to call `X(...)`
    directly."""
    from ghost_agent.tools.execute import tool_execute

    mgr_mock = MagicMock()

    # Any inline `python -c` body that imports a name equal to the
    # module name looks like a skill-invocation wrap.
    body = (
        "from greece_top_news import greece_top_news; import json; "
        "print(greece_top_news({'count': 5}))"
    )
    cmd = f'python3 -c "{body}"'

    result = await tool_execute(
        command=cmd,
        sandbox_dir=tmp_path,
        sandbox_manager=mgr_mock,
    )
    assert "SYSTEM BLOCK" in result
    assert "HINT:" in result
    assert "greece_top_news" in result
    # The hint should point at direct invocation with a concrete call.
    assert "greece_top_news(" in result


@pytest.mark.asyncio
async def test_inline_c_block_nudges_on_acquired_skills_file_path(tmp_path):
    """Alternative wrap: `python3 acquired_skills/foo.py` invoked via
    inline -c (unusual but observed). Hint should fire."""
    from ghost_agent.tools.execute import tool_execute

    # Use a bash-c with the subprocess.run-style path.
    body = (
        "import subprocess, json; subprocess.run(['python3', "
        "'acquired_skills/my_skill.py', json.dumps({'x':1})])"
    )
    cmd = f"bash -c \"{body}\""

    result = await tool_execute(
        command=cmd,
        sandbox_dir=tmp_path,
        sandbox_manager=MagicMock(),
    )
    assert "SYSTEM BLOCK" in result
    assert "HINT:" in result
    assert "my_skill" in result


@pytest.mark.asyncio
async def test_inline_c_block_hint_absent_for_unrelated_bodies(tmp_path):
    """Negative: when the blocked body is NOT a skill-wrap (just a
    long one-liner doing something unrelated), the SYSTEM BLOCK error
    should stay generic. No phantom skill nudges."""
    from ghost_agent.tools.execute import tool_execute

    # Blocked because it's substantive (>120 chars), NOT because of the import —
    # and it's not a skill wrap, so the generic message must carry no skill hint.
    body = (
        "import os, json; "
        "print(json.dumps({k: os.environ.get(k) for k in "
        "['HOME', 'PATH', 'USER', 'SHELL', 'TERM', 'LANG', 'PWD', 'TZ', 'EDITOR']}))"
    )
    assert len(body) >= 120
    cmd = f'python3 -c "{body}"'
    result = await tool_execute(
        command=cmd,
        sandbox_dir=tmp_path,
        sandbox_manager=MagicMock(),
    )
    assert "SYSTEM BLOCK" in result
    assert "HINT:" not in result, (
        "unrelated body must not produce a phantom skill-invocation hint"
    )


@pytest.mark.asyncio
async def test_inline_c_block_hint_not_on_asymmetric_from_import(tmp_path):
    """Negative: `from X import Y` where X != Y is NOT a skill-wrap
    (it's a normal stdlib/third-party import). Hint must NOT fire."""
    from ghost_agent.tools.execute import tool_execute

    body = (
        "from urllib.request import urlopen; "
        "from html.parser import HTMLParser; "
        "print('started'); print(urlopen('http://example.com').read()[:100])"
    )
    cmd = f'python3 -c "{body}"'
    result = await tool_execute(
        command=cmd,
        sandbox_dir=tmp_path,
        sandbox_manager=MagicMock(),
    )
    assert "SYSTEM BLOCK" in result
    assert "HINT:" not in result
