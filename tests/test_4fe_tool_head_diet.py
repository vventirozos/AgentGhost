"""§4FE pins: the tool head diet (flag-gated, default OFF).

Measured 2026-09-07 on the live tokenizer: 39 static schemas = 18,625 tokens (≈18,705 after the §4FJ workspace text, +80)
(the rendered head ~21.4k), a 16-tool core = ~11.3k. The diet advertises the
core plus `tool_catalog`; every other built-in stays dispatchable by name and
describable on demand. The set is static for the request (never mutated).
"""
import asyncio
import json

import pytest

from ghost_agent.tools import registry as R


def _names(defs):
    return [t.get("function", {}).get("name") for t in defs]


def test_flag_off_leaves_the_advertised_set_byte_identical(mock_context, monkeypatch):
    """World where it fails: the diet applies without the flag — every
    prefix-cache pin and the r5 drift guard would move under the operator."""
    monkeypatch.delenv("GHOST_TOOL_HEAD_DIET", raising=False)
    # §4FH: compare against the builder with the diet forced OFF — "unset"
    # vs "0" were the same world and could not fail under an always-on diet
    undieted = json.dumps(R.get_active_tool_definitions(mock_context, None, serve_tuned=False,
                                                        apply_diet=False), sort_keys=True)
    default = json.dumps(R.get_active_tool_definitions(mock_context, None, serve_tuned=False), sort_keys=True)
    assert default == undieted
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "0")
    after = json.dumps(R.get_active_tool_definitions(mock_context, None, serve_tuned=False), sort_keys=True)
    assert after == undieted
    assert "tool_catalog" not in _names(R.get_active_tool_definitions(mock_context, None, serve_tuned=False))


def test_flag_on_advertises_core_plus_catalog_only(mock_context, monkeypatch):
    """World where it fails: a hidden built-in leaks into the advertised
    set, a core tool is dropped, or the catalog is missing."""
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "1")
    defs = R.get_active_tool_definitions(mock_context, None, serve_tuned=False)
    names = set(_names(defs))
    assert "tool_catalog" in names
    static = {t["function"]["name"] for t in R.TOOL_DEFINITIONS}
    leaked = (names & static) - R.TOOL_HEAD_CORE
    assert leaked == set(), leaked
    for core in ("file_system", "execute", "browser", "web_search", "manage_projects"):
        assert core in names
    assert "self_play_loop" not in names and "postgres_admin" not in names
    # order of the kept core is the original order (pinned prefix stays stable)
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "0")
    full = _names(R.get_active_tool_definitions(mock_context, None, serve_tuned=False))
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "1")
    kept = [n for n in _names(defs) if n != "tool_catalog"]
    static_all = static | R.TOOL_HEAD_STATIC_EXTRA
    assert kept == [n for n in full if n in R.TOOL_HEAD_CORE or n not in static_all]


def test_hidden_tools_remain_dispatchable_and_the_catalog_is_registered(mock_context, monkeypatch):
    """The class-level guarantee: hiding a schema must never remove a
    handler. World where it fails: the diet filters the handler map too, so
    a described tool cannot be called by name."""
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "1")
    handlers = R.get_available_tools(mock_context)
    hidden = R.hidden_tool_definitions(mock_context)
    assert hidden, "the diet hides something"
    missing = [t["function"]["name"] for t in hidden if t["function"]["name"] not in handlers]
    assert missing == [], missing
    assert "tool_catalog" in handlers


def test_catalog_lists_every_hidden_tool_and_describes_one(mock_context, monkeypatch):
    """World where it fails: the catalog's inventory is built from a
    different set than the prompt hides (it could name an undispatchable
    tool or omit a hidden one), or `describe` returns prose instead of the
    schema."""
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "1")
    hidden = R.hidden_tool_definitions(mock_context)
    listing = asyncio.run(R.tool_catalog(action="list", context=mock_context))
    assert listing.startswith(f"{len(hidden)} additional tools")
    for t in hidden:
        assert f"- {t['function']['name']}:" in listing
    one = hidden[0]["function"]["name"]
    desc = asyncio.run(R.tool_catalog(action="describe", name=one, context=mock_context))
    assert desc.startswith("TOOL SCHEMA")
    schema = json.loads(desc.split("\n", 1)[1])
    assert schema["function"]["name"] == one and "parameters" in schema["function"]
    core = asyncio.run(R.tool_catalog(action="describe", name="file_system", context=mock_context))
    assert "already listed" in core
    nope = asyncio.run(R.tool_catalog(action="describe", name="no_such_tool", context=mock_context))
    assert nope.startswith("Error:") and one in nope


def test_catalog_with_the_diet_off_reports_nothing_hidden(mock_context, monkeypatch):
    """World where it fails: with the flag off the catalog still claims
    tools are hidden (it would tell the model to go looking for schemas
    that are already in its prompt)."""
    monkeypatch.setenv("GHOST_TOOL_HEAD_DIET", "0")
    assert R.hidden_tool_definitions(mock_context) == []
    assert asyncio.run(R.tool_catalog(action="list", context=mock_context)).startswith("0 additional tools")


def test_core_set_covers_the_high_traffic_tools():
    """World where it fails: someone trims the core below the census — the
    eight tools that make 95% of real calls must stay advertised."""
    for n in ("web_search", "file_system", "browser", "manage_projects", "execute",
              "manage_services", "system_utility", "deep_research"):
        assert n in R.TOOL_HEAD_CORE
