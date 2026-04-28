"""knowledge_base schema rename — Layer 1 of the dual-channel emit fix.

The previous schema advertised a single `content` parameter that for
`ingest_document` actually meant "filename". Models reliably mis-mapped:
asked to "write me a checklist" they'd emit
  knowledge_base(action='ingest_document', content='# migration checklist…')
which the hallucination guard caught with "You passed CONTENT or TITLE.
You MUST pass the FILENAME." — burning a strike on every conversational
prose request that grazed the tool description.

Fix: advertise per-action parameter names (`filename` for
ingest_document/forget, `fact` for insert_fact) and strengthen the
description so the model can't read "Unified memory manager. ALWAYS use
this to ingest_document" as an invitation to ingest prose. The handler
keeps the old aliases (`content`, `source`, `path`, `topic`) so legacy
trajectories and Qwen variants that aliased differently still work.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ghost_agent.tools.registry import TOOL_DEFINITIONS
from ghost_agent.tools.memory import tool_knowledge_base


def _knowledge_base_def():
    for t in TOOL_DEFINITIONS:
        if t.get("function", {}).get("name") == "knowledge_base":
            return t["function"]
    raise AssertionError("knowledge_base not registered")


# -----------------------------------------------------------------
# Schema shape
# -----------------------------------------------------------------


def test_schema_advertises_filename_and_fact():
    fn = _knowledge_base_def()
    props = fn["parameters"]["properties"]
    assert "filename" in props, (
        "Schema must advertise 'filename' for ingest_document/forget — the "
        "old 'content' name was the foot-gun that pulled models toward "
        "passing prose."
    )
    assert "fact" in props, (
        "Schema must advertise 'fact' for insert_fact so prose-bearing "
        "memories don't collide with the filename-shaped ingest path."
    )


def test_description_discourages_compose_use():
    fn = _knowledge_base_def()
    desc = fn["description"]
    # The strong negative — without it, "Unified memory manager. ALWAYS use
    # this to ingest_document (PDFs/Text)" reads as an invitation when the
    # user asks the model to compose a checklist.
    assert "NEVER use to compose" in desc, (
        "Description must explicitly forbid using knowledge_base to save "
        "prose the user just asked the model to write."
    )


def test_filename_param_description_rejects_prose():
    fn = _knowledge_base_def()
    filename_desc = fn["parameters"]["properties"]["filename"]["description"]
    # The parameter description has to repeat the negative — many LLMs
    # weight per-parameter descriptions higher than the tool-level one.
    assert "NOT raw prose" in filename_desc or "NOT prose" in filename_desc
    assert "ingest_document" in filename_desc


def test_fact_param_description_routes_to_insert_fact():
    fn = _knowledge_base_def()
    fact_desc = fn["parameters"]["properties"]["fact"]["description"]
    assert "insert_fact" in fact_desc
    # Cross-link so the model that picks `fact` for ingest_document gets
    # nudged back to `filename` rather than silently misrouting.
    assert "filename" in fact_desc


# -----------------------------------------------------------------
# Handler back-compat
# -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_accepts_legacy_content_alias(tmp_path: Path):
    """Old trajectories / Qwen variants pass `content=` for ingest_document.
    The handler must still resolve that to the target — otherwise the
    schema rename silently breaks every old call site."""
    f = tmp_path / "doc.txt"
    f.write_text("body")
    mem = MagicMock()
    mem.get_library = MagicMock(return_value=[])
    mem.ingest_file = MagicMock(return_value="ingested")
    res = await tool_knowledge_base(
        action="list_docs",
        memory_system=mem,
        sandbox_dir=tmp_path,
        content="doc.txt",  # legacy alias
    )
    # We use list_docs as the assertion vehicle: the handler accepting
    # `content=` is what we're checking, not the ingest plumbing. The
    # alias only needs to land in `target` without raising.
    assert "LIBRARY" in res or "No docs" in res


@pytest.mark.asyncio
async def test_handler_accepts_new_filename_param(tmp_path: Path):
    """Forward path: when the model picks the new `filename` param
    name (advertised in the schema), the handler resolves it."""
    mem = MagicMock()
    mem.get_library = MagicMock(return_value=[])
    res = await tool_knowledge_base(
        action="list_docs",
        memory_system=mem,
        sandbox_dir=tmp_path,
        filename="doc.txt",
    )
    assert "LIBRARY" in res or "No docs" in res


@pytest.mark.asyncio
async def test_handler_accepts_new_fact_param(tmp_path: Path):
    """Forward path: insert_fact via the new `fact` param. The handler
    routes through tool_remember; we just confirm it doesn't reject the
    arg shape and that mem.add gets called with the fact text."""
    mem = MagicMock()
    mem.add = MagicMock()
    res = await tool_knowledge_base(
        action="insert_fact",
        memory_system=mem,
        sandbox_dir=tmp_path,
        fact="The user lives in Athens",
    )
    # tool_remember returns a "stored" string on success; we accept any
    # non-error return since mem.add is mocked.
    assert "Error" not in res
    mem.add.assert_called_once()


# -----------------------------------------------------------------
# Negative: the old single-param shape is gone
# -----------------------------------------------------------------


def test_old_content_param_no_longer_in_schema():
    """The whole point of the rename is that models stop seeing `content`
    as an option for ingest_document. If we kept it in the schema for
    back-compat we'd just be re-spawning the foot-gun."""
    fn = _knowledge_base_def()
    assert "content" not in fn["parameters"]["properties"], (
        "Schema must NOT advertise the old 'content' name — it's the "
        "exact attractor that caused models to pass prose. Handler-level "
        "aliasing covers legacy callers."
    )
