#!/usr/bin/env python3
"""Does the LIVE GEPA artifact still beat the hand-written baseline?

`does-the-incumbent-pass-its-own-gate` — run the real tree through its own
checker. A promotion is a MEASUREMENT taken on one day against one holdout;
it is not a property the artifact carries forever. `planning.decompose` was
promoted 2026-08-07 on a 28-example private tier (F1 0.071 -> 0.393, delta
+0.321). The tier is hash-stable but the corpus grows, so the question
"does it still win" is answerable and unanswered.

This is the A/B GATE ONLY — no optimization, no promotion, no writes to the
live artifact. Two arms x N private examples, and it prints the verdict.

    PYTHONPATH=src python3 scripts/recheck_gepa_incumbent.py
    PYTHONPATH=src python3 scripts/recheck_gepa_incumbent.py --signature planning.decompose

⚠ Everything here — the runner payload, the F1 metric, the 0.3 pass bar,
the private-tier split — is IMPORTED from `scripts/run_gepa.py` rather than
re-implemented, because a second private notion of "did this prompt win" is
how two answers to the same question come to disagree.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghost_agent.optim.ab_eval import compare_prompts          # noqa: E402
from ghost_agent.optim.trainset import (                       # noqa: E402
    build_trainset, real_only_gate, split_public_private,
)


def _load_signature(name: str):
    from ghost_agent.optim import signatures as S
    for attr in dir(S):
        obj = getattr(S, attr)
        if getattr(obj, "name", None) == name:
            return obj
    raise SystemExit(f"unknown signature {name!r}")


def _artifact_path(name: str, home: str) -> Path:
    return Path(home) / "system" / "optim" / f"{name}.json"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signature", default="planning.decompose")
    ap.add_argument("--artifact", default=None, metavar="PATH",
                    help="score a specific artifact file, including a "
                         "WITHDRAWN one (*.retired-*). Re-measuring a "
                         "retirement under a corrected metric is exactly "
                         "what this script is for.")
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME") or
                    str(Path.home() / "Data" / "AI" / "Data"))
    ap.add_argument("--upstream", default=os.environ.get(
        "GHOST_UPSTREAM_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--private-pct", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=0.05,
                    help="the bar the ORIGINAL promotion used")
    # ⚠ 360, matching the gate. At 120 this sat exactly on the
    # `TestTheGateIsNotALatencyRace` boundary — and a timeout is scored
    # as a FAILED example, so the LONGER-OUTPUT arm (the artifact) would
    # lose races it did not lose on quality, with no timeout count
    # printed to tell them apart.
    ap.add_argument("--timeout", type=float, default=360.0)
    ap.add_argument("--full", action="store_true",
                    help="score EVERY keyed example, not just the private "
                         "tier. The public tier is biased TOWARD the "
                         "artifact (the optimizer saw it), so an artifact "
                         "that loses there too is losing on its home turf.")
    args = ap.parse_args()

    sig = _load_signature(args.signature)
    ap_path = (Path(args.artifact) if args.artifact
               else _artifact_path(sig.name, args.home))
    if not ap_path.is_file():
        print(f"no live artifact at {ap_path} — nothing to re-check",
              file=sys.stderr)
        return 1
    art = json.loads(ap_path.read_text())
    incumbent = str(art.get("optimized_instruction") or "").strip()
    baseline = str(art.get("baseline_instruction") or "").strip() or sig.instruction
    if not incumbent:
        print("the artifact carries no instruction", file=sys.stderr)
        return 1

    prev = art.get("gate") or {}
    print(f"live artifact: {ap_path}")
    print(f"  promoted    : {prev.get('promoted_utc', '(no stamp)')}")
    print(f"  gate arm    : {art.get('gate_arm', '(none)')}")
    if prev:
        print(f"  ORIGINAL    : n={prev.get('n_private')} "
              f"incumbent={prev.get('incumbent_pass_rate')} "
              f"candidate={prev.get('candidate_pass_rate')} "
              f"delta={prev.get('delta')} (bar {prev.get('min_delta')})")
    print()

    from ghost_agent.distill.collector import TrajectoryCollector
    trajs = list(TrajectoryCollector(
        Path(args.home) / "system" / "trajectories").iter_trajectories())
    examples = build_trainset(trajs, signature_name=sig.name)
    keyed = [e for e in examples
             if any((e.expected_output or {}).get(f) for f in sig.outputs)]
    public, private = split_public_private(keyed, private_pct=args.private_pct)
    public, private, _moved = real_only_gate(public, private)
    print(f"corpus {len(trajs)} trajectories -> {len(examples)} examples -> "
          f"{len(keyed)} with a {sorted(sig.outputs)} target")
    print(f"  PRIVATE ship-gate tier: {len(private)} "
          f"(was {prev.get('n_private', '?')} at promotion)")
    if not private:
        print("empty private tier — cannot re-check", file=sys.stderr)
        return 1
    scored = keyed if args.full else private
    if args.full:
        print(f"  --full: scoring ALL {len(scored)} keyed examples. The "
              f"public tier is biased TOWARD the artifact (the optimizer "
              f"saw it), so a loss here is a loss on its home turf.")

    # ── the runner and metric, imported from run_gepa ────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rg", Path(__file__).resolve().parent / "run_gepa.py")
    rg = importlib.util.module_from_spec(spec)
    sys.modules["_rg"] = rg
    spec.loader.exec_module(rg)

    from ghost_agent.core.llm import LLMClient
    llm = LLMClient(args.upstream)

    async def _runner(payload):
        instruction = payload.get("prompt", "")
        inputs = payload.get("inputs") or {}
        user_req = (inputs.get("user_request")
                    or next((str(v) for v in inputs.values() if v), "")
                    or json.dumps(inputs, default=str))
        res = await llm.chat_completion({
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": str(user_req)}],
            # the optimizer-rollout regime, verbatim
            "temperature": 0.0, "max_tokens": 8192, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        got = ((res or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "")
        want = rg._expected_target(payload.get("expected_output") or {}, sig)
        # ⚠ `rg._PASS_BAR`, not a second literal. The docstring above
        # promises everything is imported; the bar was copy-pasted, which
        # is exactly the "two definitions of did-this-prompt-win" the
        # promise exists to prevent.
        _gf = rg._gold_field(payload.get("expected_output") or {}, sig)
        if _gf:
            got = rg._section_of(got, _gf) or got
        return {"passed": bool(want) and rg._overlap(want, got) >= rg._PASS_BAR,
                "output": got}

    print(f"\nrunning {len(scored)} x 2 arms (baseline vs live incumbent), "
          f"temp 0 / no-think ...")
    cmp = await compare_prompts(
        baseline, incumbent, scored, _runner,
        min_delta=args.min_delta, per_example_timeout_s=args.timeout)

    print(f"\n  baseline (hand-written) pass rate : "
          f"{cmp.baseline_pass_rate:.4f}")
    print(f"  live incumbent pass rate          : "
          f"{cmp.candidate_pass_rate:.4f}")
    print(f"  delta                             : {cmp.delta:+.4f} "
          f"(bar {args.min_delta})")
    print(f"  per-example: incumbent wins {cmp.candidate_wins}, "
          f"baseline wins {cmp.baseline_wins}, ties {cmp.ties}")

    # McNemar on the DISCORDANT pairs — the only ones that carry
    # information, and the same statistic §4CN's promotion arithmetic
    # uses. A pass-rate delta on 31 paired examples is not a verdict.
    b, c = cmp.baseline_wins, cmp.candidate_wins
    if b + c:
        from math import comb
        n_d = b + c
        k = min(b, c)
        p = sum(comb(n_d, i) for i in range(k + 1)) / (2 ** n_d) * 2
        p = min(1.0, p)
        print(f"  McNemar exact (discordant n={n_d}): p = {p:.4f}")
    else:
        p = 1.0
        print("  McNemar: no discordant pairs — the arms are identical here")

    # ⚠ BOTH ARMS AT ZERO IS AN INSTRUMENT FAILURE, NOT A VERDICT — and
    # this script printed a confident "the incumbent no longer clears its
    # own bar" off exactly that on its first run. The cause was mine:
    # `_expected_target` was NESTED inside run_gepa's `main()`, so the
    # runner raised AttributeError on every example and
    # `ab_eval._run_one`'s broad `except` turned it into `passed=False`
    # for both arms. §4F documents the same shape from the other
    # direction ("both arms scored at the noise floor — a gate that can
    # only ever reject").
    #
    # A comparison in which NOTHING passed has not compared anything.
    if cmp.baseline_pass_rate == 0.0 and cmp.candidate_pass_rate == 0.0:
        print()
        print("‼ NO VERDICT — BOTH ARMS SCORED ZERO ON EVERY EXAMPLE.")
        print("  A comparison in which nothing passed has not compared "
              "anything: this is an instrument failure until proven "
              "otherwise, not evidence about the artifact.")
        fails = [r for r in cmp.per_example
                 if r.get("baseline_meta", {}).get("failure_reason")
                 or r.get("candidate_meta", {}).get("failure_reason")]
        if fails:
            r = fails[0]
            print(f"  first failure_reason: "
                  f"{r.get('baseline_meta', {}).get('failure_reason') or r.get('candidate_meta', {}).get('failure_reason')}")
        print("  Check: does the runner raise? is `want` empty? is the "
              "model returning content?")
        return 2

    print()
    if cmp.candidate_ships:
        print(f"✅ THE INCUMBENT STILL EARNS ITS PLACE: it beats the "
              f"hand-written baseline by {cmp.delta:+.4f} on {len(scored)} "
              f"held-out examples.")
    elif cmp.delta < 0:
        print(f"⚠ THE INCUMBENT IS NOW WORSE THAN THE BASELINE "
              f"({cmp.delta:+.4f}). It is serving every planner turn.")
        # ⚠ NOT "restores the hand-written instruction", which this line
        # claimed and which is FALSE for `planning.decompose`: the read
        # site PREPENDS the artifact to a separate production prompt, so
        # retirement removes a prefix and nothing replaces it. The seed
        # scored here is not a prompt production has ever issued.
        print("  ⚠ Retiring removes the PREFIX. Check the read site: if "
              "the artifact is prepended to a production prompt, nothing "
              "replaces it — the seed scored above is a GATE arm, not a "
              "deployed prompt.")
    else:
        print(f"⚠ THE INCUMBENT NO LONGER CLEARS ITS OWN BAR "
              f"({cmp.delta:+.4f} < {args.min_delta}). It is not measurably "
              f"worse than the baseline, but it is no longer a measured WIN "
              f"— which is what promoted it.")
    if b + c and p > 0.05:
        print(f"   ⚠ AND THE DIFFERENCE IS NOT SIGNIFICANT (McNemar "
              f"p={p:.3f} on {b + c} discordant pairs). At this tier size "
              f"the pass-rate delta above cannot settle the question on "
              f"its own — read it as a direction, not a verdict.")
    print("\n(nothing was written; this is a measurement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
