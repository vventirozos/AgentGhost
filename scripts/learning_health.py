#!/usr/bin/env python3
"""Headless learning-health report — the same data as
`introspect action='learning'`, runnable without the agent process.

    PYTHONPATH=src python3 scripts/learning_health.py [--json] [--memory-dir DIR]

Defaults GHOST_HOME/system/memory (env GHOST_HOME, else the operator path).
Read-only; never mutates a store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.core.learning_health import (  # noqa: E402
    collect_learning_health, render_learning_health,
)


def _default_memory_dir() -> Path:
    home = os.environ.get("GHOST_HOME")
    if home:
        return Path(home) / "system" / "memory"
    return Path.home() / "Data" / "AI" / "Data" / "system" / "memory"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--json", action="store_true", help="raw JSON instead of the report")
    args = ap.parse_args()

    md = Path(args.memory_dir) if args.memory_dir else _default_memory_dir()
    if not md.exists():
        print(f"memory dir not found: {md}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(collect_learning_health(md), indent=2, default=str))
    else:
        print(render_learning_health(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
