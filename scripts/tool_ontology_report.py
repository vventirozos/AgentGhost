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
    DEFAULT_MIN_PAIR, DEFAULT_MIN_SUPPORT, analyze_confusion,
    fs_batch_arm_uptake, load_replay_rows, mine_sequences, render_confusion,
    render_fs_batch_arms, render_sequences, report_to_dict, simulate_fs_batch,
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
    ap.add_argument("--simulate-fs-batch", action="store_true",
                    help="rewrite the corpus as if the `fs_batch` macro had "
                         "been available and always used, then mine that. "
                         "Diff against the plain run to see which n-grams "
                         "the macro collapses (an UPPER BOUND: it assumes "
                         "full model uptake, which the live arm measures).")
    ap.add_argument("--fs-batch-arms", action="store_true",
                    help="split the corpus by the live `fs_batch` arm and ask "
                         "the MECHANICAL question §4F names: did the macro "
                         "land? Reports the ELIGIBLE denominator first and "
                         "declines to read uptake when it is thin — zero "
                         "`paths` calls on zero opportunities is not a "
                         "finding. Needs no statistical power, so it is "
                         "readable long before the outcome comparison is.")
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
    # Its own walk: the arm read needs `extra` (the arm stamp), which
    # `simulate_fs_batch` strips, and a single stream cannot be consumed twice.
    arm_report = None
    if args.fs_batch_arms:
        arm_report = fs_batch_arm_uptake(
            TrajectoryCollector(root=root, session_id="reader")
            .iter_trajectories())
    _stream = collector.iter_trajectories()
    if args.simulate_fs_batch:
        _stream = simulate_fs_batch(_stream)
    macros = mine_sequences(
        _stream,
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
        _d = report_to_dict(confusion, macros)
        if arm_report is not None:
            _d["fs_batch_arms"] = {
                **{k: v for k, v in arm_report.items() if k != "arms"},
                "arms": {k: vars(v) for k, v in arm_report["arms"].items()},
            }
        print(json.dumps(_d, indent=2))
        return 0

    if arm_report is not None:
        print(render_fs_batch_arms(arm_report))
        print()

    if confusion is not None:
        print(render_confusion(confusion, top=args.top))
        print()
    if args.simulate_fs_batch:
        print("*** SIMULATED corpus: `fs_batch` macro applied to every turn "
              "(upper bound — assumes full uptake) ***\n")
    # Corpus purity, stated before the numbers rather than assumed. `scanned`
    # is printed even at zero so "no corruption" and "the scan never ran" can
    # never read the same — the distinction this project keeps relearning.
    try:
        from ghost_agent.utils.leaked_framing import scan_trajectories
        _pur = scan_trajectories(
            TrajectoryCollector(root=root, session_id="reader")
            .iter_trajectories())
        _n, _c = _pur["calls"], _pur["corrupt_calls"]
        _line = (f"corpus purity: {_c} of {_n} tool calls carried leaked "
                 f"tool-call framing and were EXCLUDED "
                 f"({(_c / _n if _n else 0):.2%})")
        if _c:
            _line += (f"; last seen {_pur['last_seen'][:16]} — all known cases"
                      f" predate the 2026-07-31 native-dialect fix, so a LATER"
                      f" timestamp here is a REGRESSION")
        print(_line + "\n")
    except Exception as _e:  # noqa: BLE001 — a purity line must not kill the report
        print(f"corpus purity: unavailable ({_e})\n")
    print(render_sequences(macros, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
