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
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ghost_agent.optim import gate_contract  # noqa: E402
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
                        help="Warn (exit 1) below this TOTAL supply — the GEPA "
                             "run should wait for more recording days instead. "
                             "This is a volume floor, NOT the consumer's gate; "
                             "see --min-positives.")
    parser.add_argument("--min-positives", type=int, default=200,
                        help="Warn (exit 1) below this many POSITIVE fixtures. "
                             "This is the gate that actually binds: "
                             "optimize_tool_descriptions.py scores "
                             "positives-only, and its own --min-fixtures counts "
                             "positives under the same flag name. Measured "
                             "2026-08-04: 183 fixtures / 65 positives — gating "
                             "on the total alone would have declared 'ready' "
                             "and overwritten the live pool at ~71 positives "
                             "while the runner still refused to start.")
    parser.add_argument("--min-delta", type=float, default=0.02,
                        help="Advisory only: the ship threshold the runner will "
                             "use, so this mine can report whether the PRIVATE "
                             "positive tier is fine enough to resolve it. The "
                             "runner refuses to start when it is not; reporting "
                             "it here is what makes 'is it time yet?' "
                             "answerable without starting a run.")
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
    # ⚠ BEFORE ANY I/O — the caller's proof the SCRIPT ran (argparse and
    # the interpreter both exit 2, the same code as "no corpus").
    print(f"{gate_contract.MINER_RUN_BANNER} (recordings "
          f"{args.recordings or 'default'})")

    recordings_dir = args.recordings or _ghost_home() / "system" / "llm_recordings"
    trajectory_root = args.trajectories or _ghost_home() / "system" / "trajectories"
    out_path = args.out or (_ghost_home() / "system" / "optim"
                            / gate_contract.TOOL_FIXTURES_BASENAME)

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
    # The marker an autonomous caller requires before trusting this
    # run's exit code as a SUPPLY verdict (a crash also exits 1).
    print(f"{gate_contract.MINER_RAN_MARKER} {dict(labels)}")
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
    n_pos = labels.get("positive", 0)
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
    # The gate the CONSUMER applies. `optimize_tool_descriptions.py` scores
    # positives-only and calls that count `--min-fixtures` too, so a total-only
    # gate here reads "ready" while the runner refuses — and this gate is the
    # one guarding an atomic overwrite of the live pool.
    #
    # ⚠ REAL POSITIVES, because that is what the runner counts. §4DA made the
    # runner's supply gate real-only (bench may TEACH on the public side,
    # never GRADE, and never be the reason a run starts) and did not touch
    # this one — re-opening the exact divergence the paragraph above exists
    # to close, one abstraction over. Measured 2026-08-25 on the real corpus:
    # the miner wrote the live pool and exited 0 on 403 positives while the
    # runner refused at "121 REAL positive fixtures < 200".
    n_pos_real = sum(1 for f in fixtures
                     if f.label >= 0.5 and getattr(f, "origin", "") != "bench")
    if n_pos_real < args.min_positives:
        print(f"⚠ REAL positives {n_pos_real} < --min-positives "
              f"{args.min_positives} ({n_pos} counting bench, which may "
              f"teach but never grade) — the runner scores REAL positives "
              f"only and will refuse to start.")
        ready = False

    # Advisory: the runner's resolution refusal, reported BEFORE a run is
    # started. The private share is hashed per REQUEST and a request emits
    # 1-40 fixtures, so the realised private share is not --private-pct and
    # has to be measured on the positives themselves rather than assumed.
    #
    # §4BF 1c (R2 review): REAL positives only — the runner evicts bench
    # fixtures from the gate tier before ITS resolution check, so counting
    # bench here said "OK" on mines the runner then refuses (a false
    # reassurance in exactly the bench-heavy scenario 1c created).
    priv_pos = sum(1 for f in fixtures
                   if f.tier == "private" and f.label >= 0.5
                   and getattr(f, "origin", "") != "bench")
    priv_bench = sum(1 for f in fixtures
                     if f.tier == "private" and f.label >= 0.5
                     and getattr(f, "origin", "") == "bench")
    if priv_bench and not priv_pos:
        # All-bench private tier: the runner will refuse outright — say so
        # instead of silently skipping the advisory.
        print(f"Private positives are ALL bench ({priv_bench}) — the "
              f"runner's gate is real-only and will refuse; collect real "
              f"traffic before running the optimizer.")
    if priv_pos and args.min_delta > 0:
        resolution = 1.0 / priv_pos
        # Real-over-real (R3 review): dividing real-private by ALL
        # positives (bench included) stopped measuring the hash split the
        # number is compared against.
        n_pos_real = sum(1 for f in fixtures
                         if f.label >= 0.5
                         and getattr(f, "origin", "") != "bench")
        share = priv_pos / n_pos_real if n_pos_real else 0.0
        # ⚠ THE RUNNER'S REFUSAL IS max(RESOLUTION, SIGNIFICANCE FLOOR) and
        # this reported only the RESOLUTION half, so at a coarse
        # --min-delta this instrument printed `OK` for a tier the runner
        # refuses — measured, priv_pos=4 at --min-delta 0.25: verdict OK
        # here, REFUSING TO RUN there. This is the operator's "is it time
        # yet?" instrument and its own comment claims to report the
        # runner's refusal; reporting the weaker half is how an operator
        # learns the real number only after paying for a run. The floor
        # comes from `ab_eval` rather than a fourth private copy.
        from ghost_agent.optim.ab_eval import significance_floor
        need_priv = max(significance_floor(), math.ceil(1.0 / args.min_delta))
        verdict = "OK" if priv_pos >= need_priv else "TOO COARSE"
        print(f"Private REAL positives: {priv_pos}/{n_pos_real} real "
              f"(+{priv_bench} bench private, excluded — the runner's gate "
              f"is real-only) "
              f"(realised share {share:.0%}, requested {args.private_pct}%); "
              f"smallest step {resolution:.3f} vs --min-delta "
              f"{args.min_delta}, floor {significance_floor()} discordant "
              f"— {verdict}")
        if priv_pos < need_priv:
            need_pos = math.ceil(need_priv / share) if share else 0
            print(f"  → needs ~{need_priv} private positives "
                  f"(~{need_pos} REAL positives at today's realised share) "
                  "or a larger --min-delta; the runner refuses below this.")

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
    # ⚠ AFTER the writes — this is what an autonomous caller trusts
    # before filing exit 0/1 as a SUPPLY verdict. `Labels:` prints before
    # the gates and the pool write, so a crash at the write carried it
    # anyway (round 2, F1).
    print(f"{gate_contract.MINER_DONE_MARKER} "
          f"{'ready' if ready else 'parked'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
