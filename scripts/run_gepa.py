#!/usr/bin/env python3
"""Run GEPA (or MIPROv2 fallback) on one of Ghost's optimizable prompt
signatures.

Usage:
  python -m scripts.run_gepa \\
      --signature planning.decompose \\
      --trajectories $GHOST_HOME/trajectories \\
      --upstream-url http://127.0.0.1:8080 \\
      --model qwen-3.6-35b-a3 \\
      --max-iterations 8 \\
      --output $GHOST_HOME/system/optim/planning.decompose.json

Defaults are conservative (8 iterations, low-T sampling) so the run
terminates in minutes on a local upstream. The script uses Ghost's
LLMClient as the optimizer LM — no external teacher, no outbound API.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Telemetry kill-switch must run before any lib imports that respect it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402
from ghost_agent.optim.signatures import SIGNATURES  # noqa: E402
from ghost_agent.optim.trainset import build_trainset, split_train_eval  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", required=True,
                        choices=sorted(SIGNATURES.keys()),
                        help="Which optimizable signature to tune.")
    parser.add_argument("--trajectories", type=Path, default=None,
                        help="Path to the trajectory store root. Defaults to $GHOST_HOME/trajectories.")
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--optimizer", default="GEPA")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the tuned instruction JSON. Defaults to $GHOST_HOME/system/optim/<signature>.json")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    # The A/B ship-gate is now ON BY DEFAULT: the tuned instruction is written
    # to a STAGING path and only PROMOTED to the live path if it beats the
    # baseline on the eval split. `--ab-gate` is kept as a deprecated no-op
    # (the gate always runs unless explicitly opted out) so old invocations
    # still parse; `--no-ab-gate` opts out to adopt an unverified candidate.
    parser.add_argument("--ab-gate", action="store_true", default=False,
                        help="(deprecated no-op — the A/B gate is on by default)")
    parser.add_argument("--no-ab-gate", action="store_true", default=False,
                        help="Skip the A/B gate and adopt the tuned prompt UNVERIFIED. Use only when there is no eval split to compare on.")
    parser.add_argument("--ab-min-delta", type=float, default=0.02,
                        help="Minimum eval pass-rate improvement for --ab-gate to ship the candidate. Default 0.02.")
    args = parser.parse_args()

    # Resolve default paths
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    traj_root = args.trajectories or (base / "trajectories")
    output_path = args.output or (base / "system" / "optim" / f"{args.signature}.json")

    sig = SIGNATURES[args.signature]

    if not traj_root.exists():
        print(f"trajectory root {traj_root} does not exist — log some turns first", file=sys.stderr)
        return 2

    collector = TrajectoryCollector(root=traj_root, session_id="reader")
    trajectories = list(collector.iter_trajectories())
    if not trajectories:
        print(f"no trajectories under {traj_root} — run some user turns first", file=sys.stderr)
        return 2

    examples = build_trainset(
        trajectories,
        signature_name=sig.name,
        max_examples=args.max_examples,
    )
    if not examples:
        print(
            f"trainset empty — 0 validator-passing trajectories suitable for "
            f"{sig.name}. Run more turns or loosen require_passed.",
            file=sys.stderr,
        )
        return 2

    train_set, eval_set = split_train_eval(examples, eval_fraction=args.eval_fraction)
    print(f"{len(train_set)} train / {len(eval_set)} eval examples for {sig.name}")

    # Build LLM client + metric
    from ghost_agent.core.llm import LLMClient
    llm_client = LLMClient(args.upstream_url)

    # Simple metric: substring match of the expected output's
    # final_response in the model's output. Callers with a richer
    # verifier should copy + adapt this script; GEPA/MIPROv2 both accept
    # any callable as the metric.
    def _metric(example, prediction) -> float:
        want = str(getattr(example, "expected_output", {}).get("final_response", "")).strip().lower()
        got = str(getattr(prediction, "final_response", prediction) or "").strip().lower()
        if not want or not got:
            return 0.0
        return 1.0 if want[:120] in got else 0.0

    # Write the tuned instruction to a STAGING path, NOT the live path the
    # agent reads. It is only promoted after passing the A/B gate below, so a
    # crash / rejected candidate can never leave an unproven prompt live.
    staging_path = output_path.with_name(output_path.name + ".candidate")
    from ghost_agent.optim.run_gepa import run_gepa
    result = run_gepa(
        sig,
        trainset=train_set,
        llm_client=llm_client,
        model=args.model,
        metric=_metric,
        max_iterations=args.max_iterations,
        optimizer=args.optimizer,
        output_path=staging_path,
    )

    print(f"optimized instruction written to staging {staging_path}")
    print(f"baseline: {result.baseline_instruction[:120]}...")
    print(f"optimized: {result.optimized_instruction[:120]}...")

    def _discard_staging():
        try:
            staging_path.unlink()
        except FileNotFoundError:
            pass

    def _promote_staging():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_path, output_path)

    # A/B ship-gate — ON BY DEFAULT. Only let the tuned prompt supersede the
    # baseline at inference if it actually wins on the held-out eval split.
    # Previously the gate was opt-in (`--ab-gate`) AND the tuned file was
    # written straight to the live path, so the documented invocation adopted
    # an UNPROVEN prompt globally (planning / tool-selection / reflection).
    if args.no_ab_gate:
        _promote_staging()
        print(f"A/B gate DISABLED (--no-ab-gate) — adopted UNVERIFIED at {output_path}")
        return 0

    if not eval_set:
        _discard_staging()
        print("A/B gate is ON but the eval split is empty — cannot verify the "
              "candidate; NOT promoting. Re-run with --no-ab-gate to adopt it "
              "unverified, or log more passing trajectories.", file=sys.stderr)
        return 1

    from ghost_agent.optim.ab_eval import compare_prompts
    import json as _json

    async def _ab_runner(payload):
        instruction = payload.get("prompt", "")
        inputs = payload.get("inputs") or {}
        user_req = (
            inputs.get("user_request")
            or next((str(v) for v in inputs.values() if v), "")
            or _json.dumps(inputs, default=str)
        )
        res = await llm_client.chat_completion({
            "model": args.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": str(user_req)},
            ],
            "temperature": 0.0, "max_tokens": 1024, "stream": False,
        })
        got = ((res or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "")
        want = str((payload.get("expected_output") or {})
                   .get("final_response", "")).strip().lower()
        passed = bool(want and want[:120] in got.strip().lower())
        return {"passed": passed, "output": got}

    cmp = await compare_prompts(
        result.baseline_instruction, result.optimized_instruction,
        eval_set, _ab_runner, min_delta=args.ab_min_delta,
    )
    print(f"A/B: baseline={cmp.baseline_pass_rate:.2f} "
          f"candidate={cmp.candidate_pass_rate:.2f} "
          f"delta={cmp.delta:+.2f} ships={cmp.candidate_ships}")
    if not cmp.candidate_ships:
        _discard_staging()
        print(f"A/B gate REJECTED the candidate (delta {cmp.delta:+.2f} "
              f"≤ {args.ab_min_delta}); discarded staging — baseline stands.")
        return 1
    _promote_staging()
    print(f"A/B gate PASSED — candidate promoted to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
