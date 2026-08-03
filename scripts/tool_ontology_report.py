#!/usr/bin/env python3
"""§4F Phase 2b+ — tool-ontology report: is the toolbox carved right?

Two read-only analyses over data already on disk (see
`ghost_agent/optim/tool_ontology.py` for the reasoning):

  * MACRO CANDIDATES — recurring consecutive tool sequences in the trajectory
    corpus, ranked by the loop steps a single macro call would remove and by
    whether the calls actually operate on one shared target. Runs on the
    trajectory corpus alone: no LLM, no replay, no prod involvement.
  * TOOL CONFUSION — where replayed tool choices disagreed with production,
    classified into "the boundary is wrong" vs "the description is wrong".
    Needs a replay dump from
    `scripts/optimize_tool_descriptions.py --confusion-out <file>`.

Nothing here modifies the tool registry. Proposals are for the operator.

Usage:
    PYTHONPATH=src python scripts/tool_ontology_report.py
    PYTHONPATH=src python scripts/tool_ontology_report.py \
        --replays /tmp/confusion.jsonl --min-support 5 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402
from ghost_agent.optim.tool_ontology import (  # noqa: E402
    DEFAULT_MIN_PAIR, DEFAULT_MIN_SUPPORT, analyze_confusion, load_replay_rows,
    mine_sequences, render_confusion, render_sequences, report_to_dict,
)


def _default_root() -> Path:
    """Same resolution the miner uses — a forgotten GHOST_HOME must not send
    the two tools at different directories."""
    base = os.getenv("GHOST_HOME")
    if base:
        return Path(base) / "system" / "trajectories"
    return Path.home() / "ghost_llamacpp" / "system" / "trajectories"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectories", default="",
                    help="corpus root (default $GHOST_HOME/system/trajectories)")
    ap.add_argument("--replays", default="",
                    help="JSONL from optimize_tool_descriptions.py --confusion-out")
    ap.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT,
                    help="distinct turns an n-gram must appear in")
    ap.add_argument("--min-pair", type=int, default=DEFAULT_MIN_PAIR,
                    help="misses before a confusion pair is called a pattern")
    ap.add_argument("--all-kinds", action="store_true",
                    help="include self-play/reflection trajectories too "
                         "(default: real user turns only)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.trajectories) if args.trajectories else _default_root()
    if not root.exists():
        # Exit 2, not a rendered conclusion. "No recurring tool sequences
        # above the support threshold" printed against a MISSING corpus is
        # this project's own "measured the corpus, not the signal" failure
        # mode — and in --json there was no signal at all.
        msg = (f"no trajectory corpus at {root} — set GHOST_HOME or pass "
               "--trajectories")
        if args.json:
            print(json.dumps({"error": "corpus_missing", "root": str(root)},
                             indent=2))
        print(msg, file=sys.stderr)
        return 2
    collector = TrajectoryCollector(root=root, session_id="reader")
    macros = mine_sequences(
        collector.iter_trajectories(),
        min_support=args.min_support,
        task_kinds=None if args.all_kinds else ("user_request",),
    )

    confusion = None
    if args.replays:
        rows = load_replay_rows(args.replays)
        if rows:
            confusion = analyze_confusion(rows, min_pair=args.min_pair)
        else:
            print(f"no usable replay rows in {args.replays}", file=sys.stderr)

    if args.json:
        print(json.dumps(report_to_dict(confusion, macros), indent=2))
        return 0

    if confusion is not None:
        print(render_confusion(confusion, top=args.top))
        print()
    print(render_sequences(macros, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
