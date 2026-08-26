#!/usr/bin/env python3
"""§4CV — mine the agent's own FAILED turns into verifiable-reward tasks.

OPERATOR-TRIGGERED, like every other optimizer entry point here. Nothing
in `src/` calls this; GEPA's producer half has always been a script and
this matches it.

    # see what would be mined, no LLM calls, no writes
    python3 scripts/mine_failure_envs.py --dry-run

    # mine 20 seeds into STAGING (never the live bank)
    python3 scripts/mine_failure_envs.py --limit 20

    # arm the live bench flywheel with the staged items (SEPARATE act)
    python3 scripts/mine_failure_envs.py --promote

Why staging and promotion are separate: `eval.banks.pick_next_item` walks
EVERY bank in `system/bench/banks/`, and the biological watchdog calls it
in production. Writing there is not "saving a file", it is arming a live
loop with unvetted synthetic items, so it needs its own decision.

Expect a LOW acceptance rate. Envs-FORGE needed 291 attempts for 100
accepted environments (~34%); if this run accepts nearly everything, the
oracle self-test is not biting and the items are worthless.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.optim.env_mining import (          # noqa: E402
    MINING_EPOCH, mine, mine_seeds, promote_to_bank, read_staging,
    staging_path, trainset_from_items, write_staging,
)


def _load_trajectories(home: str):
    from ghost_agent.distill.collector import TrajectoryCollector
    c = TrajectoryCollector(Path(home) / "system" / "trajectories")
    return list(c.iter_trajectories())


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME") or
                    str(Path.home() / "Data" / "AI" / "Data"))
    ap.add_argument("--name", default="ghost_failures")
    ap.add_argument("--limit", type=int, default=20,
                    help="max seeds to synthesize (each costs one LLM call)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the seed funnel only — no LLM calls, no writes")
    ap.add_argument("--promote", action="store_true",
                    help="copy STAGED items into the live bank directory. "
                         "This ARMS the bench flywheel; it is not a save.")
    from ghost_agent.core.llm import DEFAULT_UPSTREAM_URL
    ap.add_argument("--upstream", default=os.environ.get(
        "GHOST_UPSTREAM_URL", DEFAULT_UPSTREAM_URL))
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the DETERMINACY gate (saves 4 LLM calls per "
                         "surviving candidate). An item accepted without it "
                         "may encode a preference as a fact — the live first "
                         "run produced exactly one.")
    ap.add_argument("--probe-k", type=int, default=4,
                    help="independent attempts per candidate in the "
                         "determinacy probe")
    args = ap.parse_args()

    if args.promote:
        p = promote_to_bank(args.name, args.home)
        if p is None:
            print(f"nothing staged at {staging_path(args.name, args.home)} "
                  f"for epoch {MINING_EPOCH} — mine first", file=sys.stderr)
            return 1
        print(f"PROMOTED {len(read_staging(args.name, args.home))} item(s) "
              f"→ {p}\n⚠ the bench flywheel will now walk this bank")
        return 0

    trajs = _load_trajectories(args.home)
    seeds = mine_seeds(trajs)
    print(f"corpus {len(trajs)} trajectories → {len(seeds)} mineable seeds "
          f"(failed, tool-using, containment-clean)")
    if not seeds:
        print("no seeds — nothing to mine", file=sys.stderr)
        return 1
    if args.dry_run:
        for s in seeds[:args.limit]:
            print(f"  {s.trajectory_id[:12]}  tools={','.join(s.tool_names[:3])}"
                  f"  {s.user_request[:70]!r}")
        print(f"\n(dry run — {min(len(seeds), args.limit)} of {len(seeds)} "
              f"shown; no LLM calls made, nothing written)")
        return 0

    from ghost_agent.core.llm import LLMClient
    # LLMClient takes the upstream URL positionally; the model is
    # carried on the PAYLOAD, not the constructor (matches
    # scripts/run_gepa.py:224).
    llm = LLMClient(args.upstream)

    def _tick(item, ok, why):
        print(f"  {'✓' if ok else '✗'} {item.item_id}  "
              f"{'accepted' if ok else why[:100]}")

    rep = await mine(seeds[:args.limit], llm, on_item=_tick,
                     probe=not args.no_probe, probe_k=args.probe_k)
    print("\n" + rep.summary())
    if not rep.accepted:
        print("nothing accepted — no staging file written", file=sys.stderr)
        return 1
    p = write_staging(rep.accepted, args.name, args.home)
    print(f"STAGED {len(rep.accepted)} item(s) → {p}")
    ex = trainset_from_items([i.to_bank_row() for i in rep.accepted],
                             "planning.decompose")
    print(f"GEPA-usable (text-graded) examples: {len(ex)} of "
          f"{len(rep.accepted)}")
    print("NOT armed: run with --promote to add these to the live bank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
