#!/usr/bin/env python3
"""§4F Phase 2b — mine tool-choice fixtures from the GHOST_LLM_RECORD
day-files, joined to trajectory ground truth (corrections overlay applied).

Usage:
  python -m scripts.mine_tool_fixtures \\
      --recordings $GHOST_HOME/system/llm_recordings \\
      --trajectories $GHOST_HOME/system/trajectories \\
      --out $GHOST_HOME/system/optim/tool_choice_fixtures.jsonl

The pinned 2026-08-01 contract lives in ghost_agent/optim/tool_fixtures.py:
era filter (>= 2026-07-31T19:15 LOCAL), structured tool_calls only,
trajectory outcome labels via iter_trajectories() joined on request_id,
honest-failure turns excluded, split by request_id hash.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ghost_agent.optim.tool_fixtures import (  # noqa: E402
    DEFAULT_ERA_CUTOFF_LOCAL,
    mine_fixtures,
    write_fixtures_jsonl,
)


def _ghost_home() -> Path:
    return Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=None,
                        help="Day-file dir (default $GHOST_HOME/system/llm_recordings).")
    parser.add_argument("--trajectories", type=Path, default=None,
                        help="Trajectory root (default $GHOST_HOME/system/trajectories).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output fixtures JSONL (default "
                             "$GHOST_HOME/system/optim/tool_choice_fixtures.jsonl).")
    parser.add_argument("--era-cutoff", default=DEFAULT_ERA_CUTOFF_LOCAL,
                        help="LOCAL-time ISO cutoff; records before it are "
                             f"dropped (default {DEFAULT_ERA_CUTOFF_LOCAL} — "
                             "the native-prompt split + honest-failure rule).")
    parser.add_argument("--private-pct", type=int, default=30,
                        help="Private holdout percentage (request_id-hash split).")
    parser.add_argument("--max-result-chars", type=int, default=1500)
    parser.add_argument("--min-fixtures", type=int, default=200,
                        help="Warn (exit 1) below this supply — the GEPA run "
                             "should wait for more days of recording instead.")
    parser.add_argument("--force-write", action="store_true",
                        help="write the fixtures file even when the supply "
                             "gates fail (overwrites the live GEPA input).")
    parser.add_argument("--include-experiment-context", action="store_true",
                        help="Keep turns whose prompt context was mutated by a "
                             "live A/B treatment (core.experiments). Excluded "
                             "by default: the optimizer replays recorded "
                             "payloads verbatim, so those fixtures would tune "
                             "descriptions against a context only one arm "
                             "sees. See the drop accounting line "
                             "'experiment_context_excluded'.")
    args = parser.parse_args()

    recordings_dir = args.recordings or _ghost_home() / "system" / "llm_recordings"
    trajectory_root = args.trajectories or _ghost_home() / "system" / "trajectories"
    out_path = args.out or _ghost_home() / "system" / "optim" / "tool_choice_fixtures.jsonl"

    day_files = sorted(recordings_dir.glob("*.jsonl"))
    if not day_files:
        # Exit 2 = hard error (recorder off / wrong dir), distinct from
        # exit 1 = "mined fine, supply not ready yet".
        print(f"No day-files under {recordings_dir} — is GHOST_LLM_RECORD capturing?")
        return 2

    fixtures, stats = mine_fixtures(
        day_files, trajectory_root,
        era_cutoff_local=args.era_cutoff,
        private_pct=args.private_pct,
        max_result_chars=args.max_result_chars,
        exclude_mutated_context=not args.include_experiment_context,
    )

    print("Drop accounting (no silent caps):")
    for key, val in stats.items():
        print(f"  {key:26s} {val}")
    tiers = Counter(f.tier for f in fixtures)
    labels = Counter("positive" if f.label >= 0.5 else "negative" for f in fixtures)
    tools = Counter(c["name"] for f in fixtures for c in f.chosen_tools)
    print(f"Tiers: {dict(tiers)}")
    print(f"Labels: {dict(labels)}")
    print("Per-tool distribution (chosen):")
    for name, count in tools.most_common():
        print(f"  {name:26s} {count}")

    # Supply gates (exit 1 = wait for more recording days): a GEPA run on a
    # one-class corpus cannot discriminate descriptions in EITHER direction,
    # so both zero-negative and zero-positive block, not just low volume.
    #
    # EVALUATED BEFORE THE WRITE. The write is an atomic replace of the LIVE
    # GEPA input, and it used to happen first: a mistyped --trajectories, an
    # unmounted corpus, or an era-cutoff typo replaced a good fixture file
    # with an EMPTY one and then exited 1. Verified live — 116 bytes -> 0.
    n = len(fixtures)
    ready = True
    for cls in ("positive", "negative"):
        if labels.get(cls, 0) == 0:
            print(f"⚠ ZERO {cls} fixtures — a one-class corpus cannot "
                  "discriminate descriptions; wait for more supply.")
            ready = False
    if n < args.min_fixtures:
        print(f"⚠ Supply {n} < --min-fixtures {args.min_fixtures} — "
              "recommend waiting for more recording days before the GEPA run.")
        ready = False

    if ready or args.force_write:
        written = write_fixtures_jsonl(fixtures, out_path)
        print(f"Wrote {written} fixtures -> {out_path}")
    else:
        # Leave the live artifact alone; park the mine next to it so the
        # result is still inspectable.
        parked = out_path.with_suffix(out_path.suffix + ".notready")
        write_fixtures_jsonl(fixtures, parked)
        print(f"Gates NOT met — live artifact left untouched at {out_path}; "
              f"this mine parked at {parked} (--force-write to overwrite).")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
