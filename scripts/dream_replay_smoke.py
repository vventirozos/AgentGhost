#!/usr/bin/env python3
"""§4CM — the first live look at the replay engine, and the driver D4 lacked.

THE ONE NUMBER THIS EXISTS FOR: **the control-leg agreement rate.** How
often does an UNPERTURBED replay of a recorded episode reproduce that
episode's recorded outcome? Every verdict Dream will ever produce sits on
top of it, and nothing has ever measured it. If control legs mostly
disagree, the two-sided self-test discards the corpus and the engine
produces nothing — which would make D4's seeded positives, D5's taps and
D5.4's foresight thickening all work on top of an engine with no input.

The review that preceded this run named the reasons to doubt it: a replay
sees the recorded `user_request` and nothing else — no conversation
history, no project scope, no profile memory, `smart_memory` forced to
0.0, a `### REPLAY` framing prefix, and a 20-turn cap. The workspace is a
fork of the LIVE sandbox rather than the one that existed at recording
time. Any of those can flip an outcome.

WHAT IT ALSO REPORTS, because a bare agreement rate cannot be acted on:

  * validator admissibility — how many episodes get an executable check
    at all;
  * negative-control discrimination — how many of those checks FAIL on an
    empty workspace, i.e. actually check something (`sys.exit(0)` passes
    the static screen and would agree with every `passed` episode);
  * per-leg wall clock, which is what the night's budget is spent on.

SAFETY. It runs against a COPY of `$GHOST_HOME` (memory, sandbox and the
trajectory corpus), never the live one — the live agent holds the Chroma
database, and a second writer is a corruption risk the isolation recipe
does not cover because it is upstream of it. Nothing is written to the
real home. `--home` overrides the copy if you have already made one.

    GHOST_HOME=/Users/vasilis/Data/AI/Data/ PYTHONPATH=src \\
        python scripts/dream_replay_smoke.py --episodes 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Copied into the scratch home. `trajectories` is the corpus the triage
#: reads; `memory` carries the playbook and the vector store the replay
#: hydrates from; `sandbox` is what each fork starts from.
_COPY_TREES = ("system/memory", "system/trajectories", "system/bench",
               "sandbox")


def _stage_home(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for tree in _COPY_TREES:
        s, d = src / tree, dest / tree
        if not s.exists():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            subprocess.run([rsync, "-a", "--delete",
                            str(s) + os.sep, str(d) + os.sep],
                           check=False, capture_output=True, timeout=600)
        else:
            shutil.copytree(s, d, dirs_exist_ok=True)


def _build_context(home: Path, upstream: str, *, with_vector: bool):
    """A REAL context, built the way main.py builds one — same
    `parse_args`, same constructors — but pointed at the scratch home."""
    from ghost_agent.core.agent import GhostContext
    from ghost_agent.main import parse_args
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.memory.skills import SkillMemory

    argv = ["ghost", "--upstream-url", upstream, "--no-verifier"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        args = parse_args()
    finally:
        sys.argv = old_argv
    # The replay forces these itself; set them to the live values so the
    # copy is faithful up to the point the isolation changes them.
    args.smart_memory = 0.9

    memory_dir = home / "system" / "memory"
    sandbox_dir = home / "sandbox"
    memory_dir.mkdir(parents=True, exist_ok=True)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    ctx = GhostContext(args, sandbox_dir, memory_dir, None)
    from ghost_agent.core.llm import LLMClient
    ctx.llm_client = LLMClient(upstream, tor_proxy=None)
    ctx.skill_memory = SkillMemory(memory_dir)
    ctx.scratchpad = Scratchpad()
    if with_vector:
        from ghost_agent.memory.vector import VectorMemory
        ctx.memory_system = VectorMemory(memory_dir, upstream, None)
        try:
            from ghost_agent.memory.graph import GraphMemory
            ctx.graph_memory = GraphMemory(memory_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  (graph memory unavailable: {exc})", file=sys.stderr)
    return ctx


async def smoke(ctx, *, episodes: int, leg_timeout: float) -> dict:
    from ghost_agent.core import replay_engine as RE

    out = {"episodes": 0, "no_validator": 0, "vacuous_validator": 0,
           "not_checkable": 0, "control_ungradable": 0,
           "agreed": 0, "disagreed": 0, "rows": []}
    src = RE.EpisodeSource(args=getattr(ctx, "args", None))
    picked = list(src.iter_episodes(limit=episodes))
    print(f"triage: {src.seen} records seen, {len(picked)} picked "
          f"(rejections: {json.dumps(src.rejected)})", flush=True)

    for traj, tri in picked:
        tid = str(getattr(traj, "id", ""))[:12]
        out["episodes"] += 1
        row = {"id": tid, "recorded": tri.outcome, "n_steps": tri.n_steps,
               "tools": sorted(set(tri.tools))}
        t0 = time.monotonic()

        validator = await RE.synthesize_validator(
            traj, getattr(ctx, "llm_client", None))
        row["validator_chars"] = len(validator or "")
        if not validator:
            out["no_validator"] += 1
            row["result"] = "no admissible validator"
            out["rows"].append(row)
            print(f"  {tid}  {row['result']}", flush=True)
            continue

        neg = await RE.run_validator_only(ctx, validator,
                                          leg_timeout_s=leg_timeout)
        row["neg_exit"] = neg.validator_exit
        if neg.validator_exit == 2:
            # Exit 2 is the validator's reserved "I cannot check this from
            # the filesystem" — a statement about the EPISODE (a
            # conversational turn with no artifact), not a defective
            # check. Counting it as vacuous hides how much of the corpus
            # is simply not replayable.
            out["not_checkable"] += 1
            row["result"] = "episode is not filesystem-checkable (exit 2)"
        elif neg.passed is not False:
            out["vacuous_validator"] += 1
            row["result"] = (f"validator does not discriminate "
                             f"(empty workspace exit {neg.validator_exit}, "
                             f"{neg.reason})")
        if row.get("result"):
            out["rows"].append(row)
            print(f"  {tid}  {row['result']}", flush=True)
            continue

        spec = {"spec_id": f"smoke-{tid}", "trajectory_id": tid,
                "perturbation": RE.PERTURB_VERIFY_TOGGLE,
                "target": "verifier", "fork_step": 0,
                "user_request": str(getattr(traj, "user_request", "")),
                "recorded_outcome": tri.outcome, "n_steps": tri.n_steps}
        leg = await RE.run_leg(ctx, spec, arm="control", validator=validator,
                               leg_timeout_s=leg_timeout)
        row["leg_s"] = round(leg.duration_s, 1)
        row["leg_steps"] = leg.steps
        row["leg_exit"] = leg.validator_exit
        if leg.passed is None:
            out["control_ungradable"] += 1
            row["result"] = f"control ungradable: {leg.reason}"
        elif leg.passed == (tri.outcome == "passed"):
            out["agreed"] += 1
            row["result"] = f"AGREED ({tri.outcome})"
        else:
            out["disagreed"] += 1
            row["result"] = (f"DISAGREED (recorded {tri.outcome}, replay "
                             f"{'passed' if leg.passed else 'failed'})")
        row["total_s"] = round(time.monotonic() - t0, 1)
        out["rows"].append(row)
        print(f"  {tid}  {row['result']}  [{row['total_s']}s]", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--leg-timeout", type=float, default=300.0)
    ap.add_argument("--upstream", default="http://127.0.0.1:8088")
    ap.add_argument("--home", default="",
                    help="scratch GHOST_HOME (default: a fresh copy)")
    ap.add_argument("--no-vector", action="store_true",
                    help="skip the vector store (faster boot, LESS faithful "
                         "hydration — say so if you report the number)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    live = Path(os.getenv("GHOST_HOME", "").strip() or "")
    if not live.is_dir():
        print("GHOST_HOME is not set to a real directory", file=sys.stderr)
        return 2

    if args.home:
        scratch = Path(args.home)
    else:
        scratch = Path(os.getenv("TMPDIR", "/tmp")) / "ghost-replay-smoke"
        print(f"staging a COPY of {live} → {scratch} "
              f"(the live agent holds the Chroma DB; a second writer is a "
              f"corruption risk upstream of the isolation recipe)",
              flush=True)
        _stage_home(live, scratch)

    os.environ["GHOST_HOME"] = str(scratch)
    from ghost_agent.core.replay_engine import preflight
    ok, why = preflight()
    print(f"preflight: {'CLEAR' if ok else 'BLOCKED'} — {why}", flush=True)
    if not ok:
        return 2

    ctx = _build_context(scratch, args.upstream,
                         with_vector=not args.no_vector)
    started = time.monotonic()
    out = asyncio.run(smoke(ctx, episodes=args.episodes,
                            leg_timeout=args.leg_timeout))
    out["wall_s"] = round(time.monotonic() - started, 1)
    out["vector"] = not args.no_vector

    gradable = out["agreed"] + out["disagreed"]
    out["agreement_rate"] = (round(out["agreed"] / gradable, 3)
                             if gradable else None)
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        print("\n── §4CM smoke ─────────────────────────────────────────")
        print(f"  episodes attempted        {out['episodes']}")
        print(f"  no admissible validator   {out['no_validator']}")
        print(f"  validator did not check   {out['vacuous_validator']}")
        print(f"  episode not checkable     {out['not_checkable']}")
        print(f"  control ungradable        {out['control_ungradable']}")
        print(f"  control AGREED            {out['agreed']}")
        print(f"  control DISAGREED         {out['disagreed']}")
        print(f"  ── agreement rate         "
              f"{'—' if out['agreement_rate'] is None else out['agreement_rate']}"
              f"   (over {gradable} gradable)")
        print(f"  wall clock                {out['wall_s']}s"
              f"   vector store: {'on' if out['vector'] else 'OFF'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
