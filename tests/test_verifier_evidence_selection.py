"""Tests for _find_substantive_tool_for_verifier.

Regression target (2026-04-19 trace E8): verifier REFUTED at 10%
because it used `manage_projects action=exit` as evidence ({"exited":
"..."}) instead of the actual `execute` / `file_system` tool outputs
from earlier in the same turn. With bookkeeping tools skipped, the
verifier either finds real evidence or correctly skips.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import (
    _find_substantive_tool_for_verifier,
    _BOOKKEEPING_TOOL_NAMES,
)


def test_returns_none_for_empty_list():
    assert _find_substantive_tool_for_verifier([]) is None
    assert _find_substantive_tool_for_verifier(None) is None


def test_returns_none_when_all_tools_are_bookkeeping():
    tools = [
        {"name": "manage_projects", "content": '{"created": "x"}'},
        {"name": "manage_projects", "content": '{"task_id": "y"}'},
        {"name": "manage_projects", "content": '{"exited": "x"}'},
    ]
    assert _find_substantive_tool_for_verifier(tools) is None


def test_returns_last_substantive_tool_when_mixed():
    """The trace shape: several substantive tools interleaved with
    bookkeeping. The last substantive one wins."""
    tools = [
        {"name": "file_system", "content": "SUCCESS: Wrote design.md"},
        {"name": "manage_projects", "content": '{"updated": [...]}'},
        {"name": "execute", "content": "exit code 0\nEndpoint stats..."},
        {"name": "manage_projects", "content": '{"exited": "x"}'},
    ]
    res = _find_substantive_tool_for_verifier(tools)
    assert res is not None
    assert res["name"] == "execute"


def test_single_substantive_tool_is_returned():
    tools = [{"name": "execute", "content": "hello"}]
    assert _find_substantive_tool_for_verifier(tools)["name"] == "execute"


def test_handles_empty_or_none_entries():
    tools = [
        None,
        {},
        {"name": "", "content": ""},
        {"name": "file_system", "content": "ok"},
    ]
    res = _find_substantive_tool_for_verifier(tools)
    assert res and res["name"] == "file_system"


def test_normalizes_aliases_and_case():
    """The agent normalizes names in several places (manage-tasks,
    Manage_Projects). The bookkeeping filter must match those too."""
    tools = [
        {"name": "Manage-Projects", "content": "x"},
        {"name": "MANAGE_TASKS", "content": "y"},
    ]
    assert _find_substantive_tool_for_verifier(tools) is None


def test_known_bookkeeping_set_includes_expected_names():
    for name in {"manage_projects", "scratchpad", "learn_skill",
                 "update_profile", "replan"}:
        assert name in _BOOKKEEPING_TOOL_NAMES


def test_execute_not_classified_as_bookkeeping():
    """Sanity: substantive tools aren't accidentally filtered."""
    for name in {"execute", "file_system", "web_search", "deep_research",
                 "recall", "postgres_admin", "vision_analysis"}:
        assert name not in _BOOKKEEPING_TOOL_NAMES


def test_trace_e8_scenario_reproduces():
    """Exact trace shape from the E8 run: write, task_update,
    execute x2, task_update, exit. The verifier must land on the
    last `execute` — the stats output — not on `exit`."""
    tools_run_this_turn = [
        {"name": "file_system", "content": "SUCCESS: Wrote design.md"},
        {"name": "manage_projects", "content": '{"updated": [...]}'},
        {"name": "file_system", "content": "SUCCESS: Wrote nginx_log_parser.py"},
        {"name": "file_system", "content": "SUCCESS: Wrote generate_logs.py"},
        {"name": "execute", "content": "exit code 0\n(generator ran)"},
        {"name": "execute", "content": "exit code 0\nEndpoint /api/foo: p50=0.3 p99=4.5"},
        {"name": "manage_projects", "content": '{"updated": [...]}'},
        {"name": "manage_projects", "content": '{"exited": "35a0df2b5138"}'},
    ]
    res = _find_substantive_tool_for_verifier(tools_run_this_turn)
    assert res["name"] == "execute"
    assert "p50" in res["content"]
