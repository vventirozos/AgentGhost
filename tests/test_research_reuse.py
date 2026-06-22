"""Research the agent writes must actually be USED, not write-only.

Two gaps fixed (observed live on the PetAI project: a careful research brief
was written once to PetAI/research/transformer-from-scratch.md and then never
read again):

  1. A research brief written with a bare `file_system` write (not
     `action=research`) never reached the research index, so the briefing
     didn't surface it → `reconcile_research_dir` now indexes it.
  2. The coding executor skipped the whole `research/` tree, so build tasks
     never saw the design decisions → `_gather_research_briefs` now feeds the
     briefs to the build as read-only reference (while `_gather_project_files`
     keeps them OUT of the editable file set).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json

import pytest

from types import SimpleNamespace

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core.project_research import (
    reconcile_research_dir, get_research_index,
)
from ghost_agent.core.project_advancer import (
    _gather_project_files, _gather_research_briefs,
)
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        contradiction_log=None,
        current_project_id=None,
    )


def _write(store, pid, rel, text):
    p = Path(store.sandbox_root) / "projects" / pid / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# --------------------------------------------------- reconcile_research_dir

def test_indexes_directly_written_research(store):
    pid = store.create_project("PetAI", goal="train a model")
    _write(store, pid, "research/transformer.md",
           "# Transformer Architecture\n\nDecoder-only, RoPE, RMSNorm.\n")
    n = reconcile_research_dir(store, pid)
    assert n == 1
    idx = get_research_index(store, pid)
    assert len(idx) == 1
    assert idx[0]["path"] == "research/transformer.md"
    assert idx[0]["topic"] == "Transformer Architecture"      # from heading
    assert idx[0]["origin"] == "direct_write"


def test_indexes_research_nested_under_subdir(store):
    # The agent often nests research under a self-named subdir (PetAI/research/…)
    # instead of the workspace root — must still be found.
    pid = store.create_project("PetAI")
    _write(store, pid, "PetAI/research/bpe.md", "# BPE\n\nByte-level merges.\n")
    n = reconcile_research_dir(store, pid)
    assert n == 1
    assert get_research_index(store, pid)[0]["path"] == "PetAI/research/bpe.md"


def test_reconcile_is_idempotent(store):
    pid = store.create_project("P")
    _write(store, pid, "research/a.md", "# A\n\nstuff\n")
    assert reconcile_research_dir(store, pid) == 1
    assert reconcile_research_dir(store, pid) == 0            # already indexed
    assert len(get_research_index(store, pid)) == 1


def test_reconcile_skips_index_md_and_non_md(store):
    pid = store.create_project("P")
    _write(store, pid, "research/INDEX.md", "# Index\n")
    _write(store, pid, "research/notes.txt", "not markdown")
    assert reconcile_research_dir(store, pid) == 0
    assert get_research_index(store, pid) == []


def test_reconcile_ignores_non_research_dirs(store):
    pid = store.create_project("P")
    _write(store, pid, "src/model.py", "# not research\nprint(1)\n")
    assert reconcile_research_dir(store, pid) == 0


# --------------------------------------------------- executor visibility

def test_research_excluded_from_editable_files(store):
    pid = store.create_project("PetAI")
    _write(store, pid, "src/model.py", "print('model')\n")
    _write(store, pid, "research/transformer.md", "# Transformer\nnotes\n")
    _write(store, pid, "PetAI/research/bpe.md", "# BPE\nnotes\n")
    files = _gather_project_files(store, pid)
    assert "src/model.py" in files
    # research briefs are NOT build targets — wherever they live
    assert "research/transformer.md" not in files
    assert "PetAI/research/bpe.md" not in files


def test_research_briefs_gathered_as_reference(store):
    pid = store.create_project("PetAI")
    _write(store, pid, "research/transformer.md",
           "# Transformer\n\nd_model=128, n_heads=4, RoPE.\n")
    _write(store, pid, "PetAI/research/bpe.md", "# BPE\n\nbyte-level.\n")
    briefs = _gather_research_briefs(store, pid)
    assert set(briefs) == {"research/transformer.md", "PetAI/research/bpe.md"}
    assert "d_model=128" in briefs["research/transformer.md"]


def test_research_brief_excerpt_is_truncated(store):
    pid = store.create_project("P")
    _write(store, pid, "research/big.md", "# Big\n\n" + ("x" * 5000))
    briefs = _gather_research_briefs(store, pid, per_brief_chars=200)
    assert len(briefs["research/big.md"]) < 1000
    assert "truncated" in briefs["research/big.md"]


# --------------------------------------------------- end-to-end surfacing

@pytest.mark.asyncio
async def test_closing_a_task_indexes_research_and_briefing_surfaces_it(context, store):
    # Full loop: agent writes a research brief directly, closes a task → the
    # brief is indexed → the next briefing advertises it.
    pid = store.create_project("PetAI", goal="train a model")
    tid = store.add_task(pid, "Research transformer architecture")
    _write(store, pid, "research/transformer.md",
           "# Transformer Architecture\n\nDecoder-only, RoPE, RMSNorm, d_model=128.\n")

    # Before closing: not yet in the index, briefing doesn't mention it.
    assert get_research_index(store, pid) == []
    assert "transformer.md" not in build_project_briefing(store, pid)

    await tool_manage_projects(context, action="task_update",
                               project_id=pid, task_id=tid, status="DONE",
                               result="researched the architecture")

    idx = get_research_index(store, pid)
    assert any(e["path"] == "research/transformer.md" for e in idx)
    briefing = build_project_briefing(store, pid)
    assert "RESEARCH NOTES" in briefing
    assert "research/transformer.md" in briefing
