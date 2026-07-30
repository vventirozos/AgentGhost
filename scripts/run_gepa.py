#!/usr/bin/env python3
"""Run GEPA (or MIPROv2 fallback) on one of Ghost's optimizable prompt
signatures.

Usage:
  python -m scripts.run_gepa \\
      --signature planning.decompose \\
      --trajectories $GHOST_HOME/system/trajectories \\
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
import re
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
from ghost_agent.optim.trainset import (  # noqa: E402
    build_trainset,
    split_public_private,
    split_train_eval,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", required=True,
                        choices=sorted(SIGNATURES.keys()),
                        help="Which optimizable signature to tune.")
    parser.add_argument("--trajectories", type=Path, default=None,
                        help="Path to the trajectory store root. Defaults to $GHOST_HOME/system/trajectories (where the live agent writes).")
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
    parser.add_argument("--private-pct", type=int, default=30,
                        help="Percent of examples reserved (by per-item hash) for the PRIVATE "
                             "holdout the A/B ship-gate judges on. The optimizer never sees "
                             "them. Membership is stable per trajectory across runs. Default 30.")
    args = parser.parse_args()

    # Resolve default paths
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    # "system/trajectories", NOT "trajectories": prod writes via
    # memory_dir.parent (= $GHOST_HOME/system) — the old default pointed one
    # level up at a directory that never exists on a live deployment.
    traj_root = args.trajectories or (base / "system" / "trajectories")
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

    # No cap yet — the signature-target filter below must see the whole
    # corpus first (plan-bearing trajectories are ~1 in 4 of PASSED; capping
    # first would truncate away most of the usable examples).
    examples = build_trainset(
        trajectories,
        signature_name=sig.name,
        max_examples=None,
    )
    if not examples:
        print(
            f"trainset empty — 0 validator-passing trajectories suitable for "
            f"{sig.name}. Run more turns or loosen require_passed.",
            file=sys.stderr,
        )
        return 2

    # Prefer examples that carry a target for one of the signature's OWN
    # output fields (e.g. planning_output → "plan"): the metric grades
    # against those fields, and an example without them can only score via
    # the weaker final_response fallback.
    keyed = [e for e in examples
             if any((e.expected_output or {}).get(f) for f in sig.outputs)]
    if len(keyed) >= 20:
        if len(keyed) < len(examples):
            print(f"filtered to {len(keyed)}/{len(examples)} examples with a "
                  f"{sorted(sig.outputs)} target")
        examples = keyed
    elif keyed:
        print(f"only {len(keyed)} examples carry a signature-output target "
              f"(<20) — keeping all {len(examples)}; metric falls back to "
              f"final_response overlap")
    examples = examples[:args.max_examples]

    # PUBLIC/PRIVATE first: the private tier is hash-assigned per trajectory
    # and is the ONLY thing the A/B ship-gate judges on. The optimizer —
    # including its internal train/val split below — works exclusively on
    # the public tier. Judging the ship decision on data the optimizer
    # could see is how proxy-gamed prompts get promoted.
    public_set, private_set = split_public_private(examples, private_pct=args.private_pct)
    train_set, eval_set = split_train_eval(public_set, eval_fraction=args.eval_fraction)
    print(f"{len(train_set)} train / {len(eval_set)} val (public) / "
          f"{len(private_set)} PRIVATE holdout examples for {sig.name}")

    # Build LLM client + metric
    from ghost_agent.core.llm import LLMClient
    llm_client = LLMClient(args.upstream_url)

    # Ignition metric: graded token recall of the expected target inside the
    # prediction's declared output fields. Deterministic, zero extra LLM
    # calls, and GRADED — GEPA needs a gradient, and the old binary
    # substring check scored ~everything 0. Replaced by real benches
    # (verify_bench / replay fixtures) in §4F Phase 2.
    def _overlap(want: str, got: str) -> float:
        w = set(re.findall(r"[a-z0-9_]+", want.lower()))
        g = set(re.findall(r"[a-z0-9_]+", got.lower()))
        if not w or not g:
            return 0.0
        return len(w & g) / len(w)

    def _expected_target(fields_obj) -> str:
        """First non-empty signature-output field on the gold (falling back
        to final_response). `fields_obj` is a dspy.Example after
        `_to_dspy_examples` — attribute access — or a raw dict."""
        for f in list(sig.outputs) + ["final_response"]:
            if isinstance(fields_obj, dict):
                v = fields_obj.get(f, "")
            else:
                v = getattr(fields_obj, f, "")
            v = str(v or "").strip()
            if v:
                return v
        return ""

    # dspy 3.x GEPA validates this exact 5-positional signature at
    # construction: (gold, pred, trace, pred_name, pred_trace).
    def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        want = _expected_target(gold)
        got = " ".join(
            str(getattr(pred, f, "") or "") for f in sig.outputs
        ).strip() or str(pred or "")
        if not want:
            return 0.0
        return _overlap(want, got)

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
        # Public-tier val split for GEPA's own candidate selection — the
        # PRIVATE holdout stays exclusive to the A/B ship-gate below.
        valset=eval_set,
    )

    print(f"optimized instruction written to staging {staging_path}")
    print(f"baseline: {result.baseline_instruction[:120]}...")
    print(f"optimized: {result.optimized_instruction[:120]}...")

    def _discard_staging():
        # Keep the rejected candidate for post-mortem instead of deleting
        # the only copy of what GEPA produced. The ".rejected" suffix is
        # invisible to the loader and to learning-health (both match
        # exactly "*.json").
        try:
            os.replace(staging_path,
                       Path(str(staging_path) + ".rejected"))
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

    if not private_set:
        _discard_staging()
        print("A/B gate is ON but the PRIVATE holdout is empty — cannot verify "
              "the candidate; NOT promoting. Log more passing trajectories "
              "(or raise --private-pct); --no-ab-gate adopts it unverified.",
              file=sys.stderr)
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
            # Mirror the optimizer-rollout regime (no-think + full budget).
            # At 1024 tokens with thinking on, the reasoning phase consumed
            # the entire budget, content came back empty, and BOTH arms
            # scored at the noise floor (baseline 0.05 vs candidate 0.00) —
            # a gate that can only ever reject.
            "temperature": 0.0, "max_tokens": 8192, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        got = ((res or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "")
        # Same graded target/overlap as the optimizer metric so baseline and
        # candidate are judged on identical semantics; 0.3 recall = "the
        # prediction substantially covers the validator-approved target".
        want = _expected_target(payload.get("expected_output") or {})
        passed = bool(want) and _overlap(want, got) >= 0.3
        return {"passed": passed, "output": got}

    cmp = await compare_prompts(
        result.baseline_instruction, result.optimized_instruction,
        private_set, _ab_runner, min_delta=args.ab_min_delta,
    )
    print(f"A/B (PRIVATE holdout, n={len(private_set)}): "
          f"baseline={cmp.baseline_pass_rate:.2f} "
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
