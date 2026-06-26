#!/usr/bin/env python3
"""Backfill cross-project concept edges into the knowledge graph (feature 3A).

The live agent links a project's libraries/techniques to shared canonical
graph nodes (``library:<name>`` / ``technique:<name>``) on project create and
whenever its ledger/config changes. Projects created BEFORE that wiring have
no such edges, so the cross-project map (feature 3B) can't see them. This
one-shot script walks every project in the store and runs the same extractor
over its durable text + workspace requirements, emitting the missing edges.

Idempotent: ``GraphMemory.add_triplets`` upserts, so re-running only bumps
edge weights — it never duplicates.

Usage:
  GHOST_HOME=/path/to/ghost python scripts/backfill_project_concepts.py
  python scripts/backfill_project_concepts.py --memory-dir /path/to/system/memory
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.core.project_concepts import (
    link_project_concepts,
    extract_project_concepts,
)


def _default_memory_dir() -> Path:
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    return base / "system" / "memory"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memory-dir", default=None,
                    help="Path to $GHOST_HOME/system/memory (defaults from GHOST_HOME).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be linked without writing edges.")
    args = ap.parse_args()

    mem_dir = Path(args.memory_dir) if args.memory_dir else _default_memory_dir()
    if not (mem_dir / "projects.db").exists():
        print(f"No projects.db under {mem_dir} — nothing to backfill.")
        return 0

    store = ProjectStore(mem_dir)
    graph = None if args.dry_run else GraphMemory(mem_dir)

    projects = store.list_projects() if hasattr(store, "list_projects") else []
    if not projects:
        print("No projects found.")
        return 0

    total_edges = 0
    for p in projects:
        proj = store.get_project(p.get("id", "")) or p
        meta = proj.get("metadata") or {}
        config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        from ghost_agent.core.project_concepts import _read_requirements
        libs, techs = extract_project_concepts(
            title=proj.get("title", ""),
            goal=proj.get("goal", ""),
            ledger=meta.get("design_ledger", ""),
            config=config,
            requirements_text=_read_requirements(proj.get("workspace_dir")),
        )
        n = len(libs) + len(techs)
        label = f"{proj.get('title', '?')} ({proj.get('id', '?')})"
        if n == 0:
            print(f"  - {label}: no recognisable concepts")
            continue
        print(f"  - {label}: libraries={sorted(libs)} techniques={sorted(techs)}")
        if not args.dry_run:
            total_edges += link_project_concepts(graph, proj)

    if args.dry_run:
        print("\nDry run — no edges written.")
    else:
        print(f"\nBackfill complete: {total_edges} concept edges written across "
              f"{len(projects)} projects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
