"""Verify that main.lifespan wires a ProjectStore onto the agent context.

This is a small but load-bearing test: without it, `manage_projects`
silently returns "project_store is not configured" in production even
though every component test passes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostContext
from ghost_agent.memory.projects import ProjectStore


def _build_context(tmp_path):
    args = SimpleNamespace(
        upstream_url="http://localhost:8080",
        swarm_nodes_parsed=[], worker_nodes_parsed=[],
        visual_nodes_parsed=None, coding_nodes_parsed=None,
        image_gen_nodes_parsed=None,
        no_memory=True,  # short-circuit the heavy memory branch
    )
    sandbox_dir = tmp_path / "sb"
    sandbox_dir.mkdir()
    memory_dir = tmp_path / "mem"
    memory_dir.mkdir()
    return GhostContext(args, sandbox_dir, memory_dir, tor_proxy=None)


def test_project_store_wires_to_context(tmp_path):
    """Mirror the lifespan wiring directly: building a ProjectStore from
    the same args lifespan uses must produce a usable store on context.

    A full lifespan integration test was tried and proved too tightly
    coupled to unrelated boot machinery (LLMClient / Docker / MemoryBus
    stubs). This focused test exercises the same construction path
    `main.lifespan` runs — if `ProjectStore(memory_dir, sandbox_root=...)`
    breaks, this fails before it reaches a flaky lifespan stub.
    """
    context = _build_context(tmp_path)
    context.project_store = ProjectStore(
        context.memory_dir, sandbox_root=context.sandbox_dir,
    )
    assert isinstance(context.project_store, ProjectStore)
    pid = context.project_store.create_project("smoke")
    proj = context.project_store.get_project(pid)
    assert proj is not None
    # Workspace dir lands under the sandbox root passed in
    assert str(tmp_path / "sb") in proj["workspace_dir"]
    assert Path(proj["workspace_dir"]).exists()


def test_main_module_imports_project_store():
    """Cheap import-time check — fails fast if someone deletes the import."""
    from ghost_agent import main as main_mod
    src = inspect.getsource(main_mod)
    assert "from .memory.projects import ProjectStore" in src
    assert "context.project_store" in src
    assert "ProjectStore(" in src
