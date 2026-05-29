"""Tier 3: logging-readability invariants.

Source-level guards (the codebase already uses inspect.getsource for log
invariants) that the readability fixes stay fixed: literal-emoji icons
replaced with Icons constants, node-failover messages no longer crammed
into the 18-char title, browser uses its own glyph, and the overloaded
"Sandbox" boot title is de-collided.
"""

import inspect

from ghost_agent.core import llm as llm_mod
from ghost_agent.tools import browser, search, image_gen, report_pdf
from ghost_agent.sandbox import docker as docker_mod


def test_no_literal_emoji_icons_in_key_modules():
    for mod, banned in [
        (llm_mod, ['icon="🎨"', 'icon="⚙️"', 'icon="⚡"']),
        (docker_mod, ['icon="⚙️"', 'icon="📥"', 'icon="📦"', 'icon="✅"', 'icon="⚠️"']),
        (image_gen, ['icon="🎨"']),
        (report_pdf, ['icon="📄"']),
    ]:
        src = inspect.getsource(mod)
        for lit in banned:
            assert lit not in src, f"{mod.__name__} still uses literal {lit}"


def test_llm_node_failover_uses_title_plus_content():
    src = inspect.getsource(llm_mod)
    # The whole message must NOT be the title (it'd truncate to 18 chars,
    # losing the error). Each failover gets a short title + content.
    for kind in ("Vision", "Worker", "Coding", "Swarm"):
        assert f'pretty_log(f"{kind} node' not in src
        assert f'"{kind} Node Failed"' in src


def test_browser_uses_dedicated_icon():
    src = inspect.getsource(browser)
    assert "icon=Icons.TOOL_BROWSER" in src


def test_docker_sandbox_titles_decollided():
    src = inspect.getsource(docker_mod)
    for t in ('"Sandbox Provision"', '"Sandbox Chromium"', '"Sandbox Ready"',
              '"Sandbox Image"', '"Sandbox Tor"'):
        assert t in src, f"missing distinct boot title {t}"


def test_fact_check_not_stop_icon():
    src = inspect.getsource(search)
    assert 'pretty_log("Fact Check", query_text[:50] + "..", icon=Icons.STOP)' not in src
    assert "icon=Icons.TOOL_DEEP" in src
