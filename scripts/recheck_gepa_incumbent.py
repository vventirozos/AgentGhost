#!/usr/bin/env python3
"""Does the LIVE GEPA artifact still beat the hand-written baseline?

`does-the-incumbent-pass-its-own-gate` — run the real tree through its own
checker. A promotion is a MEASUREMENT taken on one day against one holdout;
it is not a property the artifact carries forever. `planning.decompose` was
promoted 2026-08-07 on a 28-example private tier (F1 0.071 -> 0.393, delta
+0.321). The tier is hash-stable but the corpus grows, so the question
"does it still win" is answerable whenever an artifact is live.

⚠ AS OF 2026-08-26 THERE IS NO LIVE ARTIFACT for any signature: the optim
dir holds only `planning.decompose.json.retired-4cw` (retired 2026-08-24),
`.prev`, `.candidate.rejected` and `.notready` files. The default
invocation therefore exits 2 ("no live artifact — nothing to re-check"),
and the branch text below that says an artifact "is serving every planner
turn" describes the state this script was written for, not today's.

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
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghost_agent.optim import ab_eval
from ghost_agent.optim import gate_contract                        # noqa: E402
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
    # ⚠ NOT SystemExit. Its only caller catches it as an "artifact-only
    # signature" signal, so this never reached a shell — but an internal
    # signal dressed as an exit is one refactor away from exiting 1
    # ("no longer wins") for a typo, and the conformance scan cannot
    # tell control flow from a real exit. LookupError is what it means.
    raise LookupError(f"unknown signature {name!r}")


def _artifact_path(name: str, home: str) -> Path:
    return Path(home) / "system" / "optim" / f"{name}.json"


def _fmt_p(p) -> str:
    """`None` must never reach a `:.4f` — reachable via a negative
    `--min-delta`, validated against the same bound the gates use."""
    return "n/a" if p is None else f"{p:.4f}"


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
    ap.add_argument("--min-delta", type=float, default=None,
                    help="the MARGIN half of the gate (the significance "
                         "half is ab_eval.SHIP_ALPHA). DEFAULT: the bar "
                         "recorded in the artifact's own gate block, so "
                         "the re-check holds it to the bar it was "
                         "promoted under; 0.05 when it records none.")
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

    # ⚠ THIS RAN UNCONDITIONALLY AND `SystemExit`ED ON ANY ARTIFACT-ONLY
    # SIGNATURE. `optim/signatures.py` holds three names; tool
    # descriptions are DELIBERATELY artifact-only (registry.py's artifact-only note), so
    # `--signature tool_description.<tool>` died with "unknown signature"
    # before `--artifact` was even consulted. §4DA reshaped the artifact
    # specifically so THIS instrument could print the override warning,
    # and the gate that rejects the file was left in place — the reshape
    # was verified by a test that RE-TYPED this reader's expressions
    # instead of running it.
    #
    # The RE-SCORE genuinely needs a signature (it replays a trainset the
    # signature defines); the AUDIT REPORT does not. So an artifact-only
    # signature now reports and then says plainly why it cannot re-score,
    # instead of refusing to open the file at all.
    try:
        sig = _load_signature(args.signature)
    except LookupError:
        sig = None
    ap_path = (Path(args.artifact) if args.artifact
               else _artifact_path(args.signature, args.home))
    if not ap_path.is_file():
        print(f"no live artifact at {ap_path} — nothing to re-check",
              file=sys.stderr)
        # ⚠ 2, NOT 1. This file declares "0 = still wins, 1 = no longer
        # wins, 2 = could not measure" and then returned 1 from three
        # branches that could not measure anything — so a caller acting
        # on the codes would read an instrument failure as a verdict to
        # retire. The collision round 5 carved out a code for, in the
        # other three branches.
        return 2
    art = json.loads(ap_path.read_text())
    incumbent = str(art.get("optimized_instruction") or "").strip()
    # ⚠ `sig` IS None FOR AN ARTIFACT-ONLY SIGNATURE, and an artifact
    # need not carry `baseline_instruction` — round 4's own fixture writes
    # one that does not. Without this guard that combination is an
    # `AttributeError: 'NoneType' object has no attribute 'instruction'`.
    baseline = str(art.get("baseline_instruction") or "").strip()
    if not baseline and sig is not None:
        baseline = sig.instruction
    if not incumbent:
        print("the artifact carries no instruction", file=sys.stderr)
        return 2

    prev = art.get("gate") or {}
    print(f"live artifact: {ap_path}")
    print(f"  promoted    : {prev.get('promoted_utc', '(no stamp)')}")
    print(f"  gate arm    : {art.get('gate_arm', '(none)')}")
    if prev:
        # ⚠ n MUST BE THE TIER THE GATE DECIDED ON. Printing `n_private`
        # beside a PAIRED delta reported n=60 for a comparison made over
        # 48 pairs — the tier size and the number over it disagreeing by
        # 20% with nothing saying so, while `n_usable_pairs` sat unread
        # in the same file.
        _n_used = prev.get("n_usable_pairs", prev.get("n_private"))
        _n_note = ("" if _n_used == prev.get("n_private")
                   else f" (of {prev.get('n_private')} in the tier)")
        print(f"  ORIGINAL    : n={_n_used}{_n_note} "
              f"incumbent={prev.get('incumbent_pass_rate')} "
              f"candidate={prev.get('candidate_pass_rate')} "
              f"delta={prev.get('delta')} (bar {prev.get('min_delta')})")
        # ⚠ THE §4CY AUDIT FIELDS HAD NO READER. `run_gepa` stamps
        # p_value / discordant_pairs / candidate_wins / incumbent_wins /
        # ship_alpha / significance_overridden precisely so a promotion can
        # be re-examined later — and the only instrument that opens the
        # gate block printed none of them, so an OVERRIDDEN promotion was
        # byte-identical on screen to an honest one. The change's own
        # justification ("a promotion whose record cannot answer HOW MANY
        # EXAMPLES ACTUALLY MOVED is unauditable") was recorded and not
        # delivered.
        if prev.get("discordant_pairs") is not None:
            print(f"              : McNemar p={prev.get('p_value')} "
                  f"(bar {prev.get('ship_alpha')}) over "
                  f"{prev.get('discordant_pairs')} discordant pairs, "
                  f"{prev.get('candidate_wins')} candidate / "
                  f"{prev.get('incumbent_wins')} incumbent")
        # ⚠ THE SET-LEVEL CAVEAT MUST REACH THE READER. §4DA round 13
        # added `co_promoted`/`gate_scope` precisely so the record would
        # not imply a per-component measurement nobody made — and the one
        # reader of the gate block printed the numbers and not the
        # caveat, so an operator read a set's `delta`/`candidate_wins` as
        # this component's. Same shape as round 5's "the §4CY audit
        # fields had no reader".
        _co = [c for c in (prev.get("co_promoted") or [])
               if c != args.signature]
        if _co:
            print(f"  ⚠ SET-LEVEL EVIDENCE: the numbers above come from "
                  f"ONE A/B over {len(_co) + 1} components promoted "
                  f"together ({', '.join(sorted(_co))} and this one). No "
                  f"per-component contribution was measured, so they are "
                  f"the SET's win, not this signature's.")
        elif prev.get("gate_scope"):
            print(f"  gate scope  : {prev['gate_scope']}")
        if prev.get("significance_overridden"):
            print("  ⚠ THAT PROMOTION USED --allow-insignificant-ship: it "
                  "cleared the margin but NOT the significance bar. Its "
                  "'win' was an operator judgement call, not a measured "
                  "result — weigh the re-check below accordingly.")
        # ⚠ KEYED ON THE TOTAL, NOT ON ONE CAUSE. Keyed on
        # `outage_excluded` alone this stayed silent for a promotion whose
        # tier shrank entirely through a CORPUS GAP — round 4 drew that
        # distinction and then hid half of it from the only reader.
        _excl = prev.get("transport_excluded") or 0
        if _excl:
            # ⚠ AND SAY WHEN THE SPLIT IS NOT MEASURED. `run_gepa.py`
            # replays LIVE and marks every runner exception `UNREACHED`,
            # so a metric bug, a malformed example and a per-example
            # timeout all land in `outage_excluded` — the one cause this
            # line tells the operator is re-runnable. Artifacts from that
            # gate carry `exclusion_cause_distinguished: False`.
            _o = prev.get("outage_excluded") or 0
            _g = prev.get("corpus_gap_excluded") or 0
            _why = (f"{_o} transport outage, {_g} no recorded payload")
            if prev.get("exclusion_cause_distinguished") is False:
                _why += ("; the gate that wrote this replays live and "
                         "does not distinguish an outage from a metric "
                         "error, so read that split as 'never reached a "
                         "verdict'")
            print(f"  ⚠ {_excl} pairs never reached a verdict in both arms "
                  f"({_why}) and "
                  f"were EXCLUDED; the gate decided on "
                  f"{prev.get('n_usable_pairs')} pairs (raw delta over all "
                  f"rows {prev.get('raw_delta')}).")
        # ⚠ THROUGH THE SHARED READER. `_sa.get("overridden")` opened
        # `run_gepa.py`'s shape and not the tool-description gate's
        # (`seed_loss_overridden`), so this warning — the one that says a
        # live artifact was promoted despite LOSING to the hand-written
        # text — was structurally unreachable for every artifact that
        # gate writes. `gate_contract.read_seed_arm` understands both,
        # including the pre-contract files already on disk.
        _sa = gate_contract.read_seed_arm(prev) or {}
        if _sa.get("overridden"):
            print("  ⚠ THAT PROMOTION USED --allow-seed-loss: it lost to "
                  "the hand-written seed and was promoted anyway.")
        elif _sa.get("undecidable"):
            print("  ⚠ THAT RUN'S SEED ARM WAS UNDECIDABLE (an outage ate "
                  "the pairs) and the run was refused — the veto neither "
                  "fired nor cleared.")
        elif _sa.get("vetoed"):
            print("  ⚠ THAT PROMOTION'S SEED ARM FIRED THE VETO and the "
                  "run was refused — this artifact is from a LATER run.")
    if sig is None and art.get("signature_name") \
            and str(art["signature_name"]) != args.signature:
        # ⚠ A TYPO IS NOT AN ARTIFACT-ONLY FAMILY. `--signature
        # planning.decompos` (one letter short) resolved no signature and
        # was then reported as "ARTIFACT-ONLY — no signature and no
        # trainset", with two dead remedies, while the script was holding
        # the disproof: the artifact's own `signature_name` says
        # `planning.decompose`. Round 4 widened the reader to artifact-only
        # families and widened it past the error case too.
        print(f"\n'{args.signature}' matches no signature, and the "
              f"artifact at {ap_path} says it belongs to "
              f"'{art['signature_name']}'. Did you mean that? Nothing was "
              f"re-scored.", file=sys.stderr)
        return 2

    if sig is None:
        # ⚠ NAME THE RIGHT OPTIMIZER. The first version named
        # `optimize_tool_descriptions.py` for EVERY artifact-only
        # signature, and the only artifact-only family with files on disk
        # today is `verifier.*`, whose optimizer is `optimize_verifier.py`
        # — the pins drove only `tool_description.*`, the region where the
        # fix and the string agree.
        _owner = {"tool_description.":
                  "scripts/optimize_tool_descriptions.py",
                  "verifier.": "scripts/optimize_verifier.py"}
        _script = next((v for k, v in _owner.items()
                        if args.signature.startswith(k)),
                       "its own optimizer")
        print()
        print(f"'{args.signature}' is ARTIFACT-ONLY — no signature and no "
              f"trainset, so there is nothing to re-score against. The "
              f"gate record above is the whole report. To re-measure it, "
              f"re-run {_script}, or judge it on production turns with "
              f"scripts/gepa_live_check.py --signature {args.signature}.")
        # ⚠ NOT 0. Zero here is the same code as "re-scored and it still
        # wins", so for the entire family this reader was extended to
        # cover, a re-check that measured NOTHING was indistinguishable
        # from one that measured a win.
        return 3
    print()

    # ⚠ ONE MARGIN, AND BY DEFAULT IT IS THE ARTIFACT'S OWN. This script
    # defaulted to 0.05 while `run_gepa` promotes at `--ab-min-delta`
    # (default 0.02), so "THE INCUMBENT NO LONGER CLEARS ITS OWN BAR" was
    # measured against a bar the artifact was never held to — and the run
    # PRINTS the artifact's real bar three lines above. A +0.041 delta
    # clears 0.02 twice over and was reported as a failure.
    #
    # The branch conditions, the message and `compare_prompts`'s
    # `min_delta` must all be the SAME number, or fixing the message alone
    # just moves the false inequality to the other branch.
    # ⚠ ONLY CLAIM "ITS OWN BAR" WHEN IT IS. The fallback is the same
    # hardcoded 0.05 this block condemns; announcing that as the
    # artifact's own bar just relocates the false claim. Five of the six
    # artifacts in the live store record NO margin at all, so the fallback
    # is the common path, not the exception.
    # ⚠ THE SAME VALIDATION BOTH GATES DO, ON THE INSTRUMENT THAT ACTS
    # ON THEIR OUTPUT. This script's own docstring said "`--min-delta`,
    # which this script does not validate." A NEGATIVE margin makes
    # `cmp.delta > _margin` trivially true, so an artifact losing by
    # -0.40 exits 0 — "IT STILL EARNS ITS PLACE" — and `1/_margin` in
    # the retire power floor is the sibling's ZeroDivisionError/
    # OverflowError pair, one script over.
    if args.min_delta is not None and not 1e-6 <= args.min_delta < 1:
        print(f"REFUSING TO RUN: --min-delta {args.min_delta} is not a "
              f"usable margin. It must be >=1e-6 (a bar of 0 admits any "
              f"non-zero swing, and anything smaller cannot be resolved "
              f"by a holdout of any size) and <1 (no pass-rate delta can "
              f"exceed 1.0, so nothing could ever clear it). This is the "
              f"same bound both ship gates enforce.", file=sys.stderr)
        return 2
    _margin = args.min_delta
    _from_artifact = False
    if _margin is None:
        _recorded = prev.get("min_delta")
        if _recorded is None:
            _margin = 0.05
            print(f"  the artifact records no bar of its own; re-checking "
                  f"against the default {_margin} (set --min-delta to "
                  f"choose)")
        else:
            _margin = float(_recorded)
            _from_artifact = True
            print(f"  re-checking against the artifact's own bar: "
                  f"{_margin} (override with --min-delta)")
    else:
        print(f"  re-checking against --min-delta {_margin}")

    # ⚠ AND THE ARTIFACT'S OWN BAR IS A MARGIN TOO. Round 15 validated
    # the FLAG and left the DEFAULT path — which is the one this block's
    # own comment calls the default ("ONE MARGIN, AND BY DEFAULT IT IS
    # THE ARTIFACT'S OWN"). Driven with `gate.min_delta: -0.4` recorded
    # and no flag, against an incumbent losing by -0.30:
    #
    #   ⚠ THE INCUMBENT IS NOW WORSE THAN THE BASELINE (-0.3000).
    #   rc = 0
    #
    # Exit 0 is "it still earns its place": the prose says retire and the
    # code a script branches on says keep, because `delta > -0.4` is
    # trivially true. A recorded 0 slips past the `_margin > 0` guard the
    # same way and makes the bar "any non-zero swing".
    if not 1e-6 <= _margin < 1:
        print(f"REFUSING TO RUN: the bar this re-check would use "
              f"({_margin}"
              + (", recorded in the artifact's own gate block"
                 if _from_artifact else "")
              + f") is not a usable margin. It must be >=1e-6 and <1 — "
                f"the same bound both ship gates enforce. Pass "
                f"--min-delta to choose one, or repair the artifact.",
              file=sys.stderr)
        return 2

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
        return 2
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
        min_delta=_margin, per_example_timeout_s=args.timeout)

    # ⚠ THE EXCLUSION ACCOUNTING FOR THE RUN THIS SCRIPT JUST DID.
    # `:171` reports it for the RECORDED gate block and the live `cmp`'s
    # `transport_excluded` / `raw_delta` / `raw_*_pass_rate` were read
    # nowhere. Driven: a 45-example tier that lost 40 to a one-arm outage
    # printed a delta of -1.0000 and the verdict "THE INCUMBENT IS NOW
    # WORSE THAN THE BASELINE" **byte-identical** to the healthy run,
    # with the only trace a buried `discordant n=5`. This is the script
    # whose own docstring calls it the one an operator uses to decide
    # whether to RETIRE a live artifact.
    _n_paired = len(scored) - cmp.transport_excluded
    if cmp.transport_excluded:
        print(f"\n  ⚠ {cmp.transport_excluded} of {len(scored)} examples "
              f"never reached a verdict in BOTH arms (transport, not a "
              f"wrong answer) and were EXCLUDED. Everything below is over "
              f"{_n_paired} pairs; over all examples the rates were "
              f"{cmp.raw_baseline_pass_rate:.4f}/"
              f"{cmp.raw_candidate_pass_rate:.4f} "
              f"({cmp.raw_delta:+.4f}).")
    _unmeasurable = False
    print(f"\n  baseline (hand-written) pass rate : "
          f"{cmp.baseline_pass_rate:.4f}")
    print(f"  live incumbent pass rate          : "
          f"{cmp.candidate_pass_rate:.4f}")
    print(f"  delta                             : {cmp.delta:+.4f} "
          f"(bar {_margin}, over {_n_paired} pairs)")
    print(f"  per-example: incumbent wins {cmp.candidate_wins}, "
          f"baseline wins {cmp.baseline_wins}, ties {cmp.ties}")

    # McNemar on the DISCORDANT pairs — the only ones that carry
    # information, and the same statistic §4CN's promotion arithmetic
    # uses. A pass-rate delta on 31 paired examples is not a verdict.
    #
    # ONE implementation, in `ab_eval` — this file and `run_gepa.py` each
    # carried an inline copy (TWO, not three: the library had none until
    # `p_value` was added). Same function here, asked a DIFFERENT question:
    #
    # ⚠ NOT `cmp.p_value`, and NOT a two-sided p either. Two rounds of
    # review, two different wrong answers here, so both are recorded:
    #
    # ROUND 2 read `cmp.p_value`. That is ONE-SIDED toward the CANDIDATE,
    # and this script calls compare_prompts(baseline=seed,
    # candidate=incumbent) — so for an incumbent LOSING 8-1 the tail goes
    # to 0.998 by construction and the script announced "not significant"
    # on the strongest evidence for retirement.
    #
    # ROUND 3 switched to `alternative="two-sided"`. That fixed the loss
    # branch and broke the win branch: a 5-0 sweep is the SMALLEST
    # evidence `run_gepa._significance_floor()` calls shippable
    # (one-sided p=0.03125), yet its two-sided p is 0.0625 — so the gate
    # ships it while this instrument printed "NOT SIGNIFICANT (p=0.062 >
    # 0.05)". A two-sided statistic judged against a ONE-SIDED bar is a
    # 0.025 bar wearing a 0.05 label.
    #
    # The two branches ask genuinely DIFFERENT, each pre-specified,
    # questions — "does the incumbent still earn its place?" and "is the
    # incumbent worse?" — so each gets its own directional tail against
    # the same SHIP_ALPHA. That is not choosing the direction after
    # seeing the data; it is choosing it by which verdict is being
    # reported.
    b, c = cmp.baseline_wins, cmp.candidate_wins
    #: The gate's own number — what `candidate_ships` was decided on.
    p_win = ab_eval.mcnemar_p(b, c, alternative="candidate")
    #: The mirror, for "is the incumbent measurably WORSE?".
    p_loss = ab_eval.mcnemar_p(b, c, alternative="baseline")
    _losing = cmp.delta < 0
    p = p_loss if _losing else p_win
    if p is not None:
        _dir = ("incumbent better" if c > b else
                "seed better" if b > c else "even")
        _q = ("is the seed better?" if _losing
              else "does the incumbent still earn its place?")
        print(f"  McNemar exact, one-sided ({_q}) — discordant n={b + c}, "
              f"{c} incumbent / {b} seed, {_dir}: p = {p:.4f}")
    else:
        # ⚠ NOT 1.0. No discordant pairs is an absence of evidence, not
        # evidence of no difference (`verdict-without-power`); folding it
        # to 1.0 lets "we learned nothing" read as "they are the same".
        print("  McNemar: no discordant pairs — nothing here can "
              "distinguish the arms, which is not the same as their "
              "being equal")

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
    # The number of pairs needed to RESOLVE `_margin` — a pre-flight
    # quantity (computed before any delta is known) used by two branches
    # below: the transport guard on the loss side, and the power gate on
    # the no-longer-clears side.
    _retire_need = max(ab_eval.significance_floor(),
                       math.ceil(1.0 / _margin) if _margin > 0 else 0)
    if cmp.candidate_ships:
        print(f"✅ THE INCUMBENT STILL EARNS ITS PLACE: it beats the "
              f"hand-written baseline by {cmp.delta:+.4f} on {len(scored)} "
              + ("examples — NOTE: with --full most of these are the "
                 "PUBLIC tier the optimizer trained on, which is the "
                 "artifact's home turf, not held-out."
                 if args.full else "held-out examples."))
    elif cmp.delta < 0:
        print(f"⚠ THE INCUMBENT IS NOW WORSE THAN THE BASELINE "
              f"({cmp.delta:+.4f}). It is serving every planner turn.")
        # ⚠ AND SAY WHEN THAT VERDICT IS UNDERPOWERED. A retire
        # recommendation over 5 surviving pairs read identically to one
        # over 45. The bar is the same one the gates use.
        # ⚠ THE SAME BAR THE GATES USE, NOT HALF OF IT.
        # `significance_floor()` alone is 5, while both gates refuse below
        # `max(floor, ceil(1/min_delta))` — 20 to 50 in practice. At 5 the
        # firing window is 1..4 surviving pairs, and there the pre-existing
        # "not significant" caveat already fires, so the guard added ZERO
        # coverage: driven at 8 of 45 pairs, McNemar p=0.0039, neither
        # fired, and "THE INCUMBENT IS NOW WORSE THAN THE BASELINE
        # (-1.0000). It is serving every planner turn." rendered
        # uncaveated over 8 pairs. Round 2 recorded this exact
        # half-of-`_need` shape.
        # ⚠ THIS GUARD IS ARMED ON THE CAUSE ON PURPOSE, and the general
        # "is there enough evidence?" question is answered by the
        # significance test below instead. `_retire_need` is the number of
        # pairs needed to RESOLVE `_margin` — a pre-flight quantity, computed
        # before any delta is known — so applying it to an OBSERVED loss of
        # -0.30 would refuse a 31-pair tier that resolves that loss easily.
        # What transport adds is that the shortfall is RE-RUNNABLE: five
        # survivors of forty-five can be significant (5-0 is p=0.031) and
        # still be an outage wearing a measured loss's clothes.
        if cmp.transport_excluded and _n_paired < _retire_need:
            _unmeasurable = True
            print(f"  ⚠ BUT ONLY {_n_paired} OF {len(scored)} EXAMPLES "
                  f"SURVIVED — under the {_retire_need} this comparison "
                  f"needs to resolve a {_margin} margin at "
                  f"p<={ab_eval.SHIP_ALPHA}. This is a TRANSPORT failure "
                  f"wearing a measured loss's clothes. Re-run when the "
                  f"upstream is stable BEFORE retiring anything.")
        # ⚠ AND THE RETIREMENT NEEDS THE SAME EVIDENCE BAR THE KEEP DOES.
        # `return 0 if cmp.delta > _margin else 1` made "still wins" require
        # significance and "retire it" require none: driven with identical
        # evidence strength in both directions (2 discordant pairs, p=0.25,
        # |delta|=0.30 on a 45-pair tier, no transport loss),
        # `delta -0.30` exited 1 — "it no longer earns its place" — while
        # `delta +0.30` exited 2. The instrument printed "read it as a
        # direction, not a verdict" and then returned the verdict. Every
        # sibling is symmetric in its own direction: `_ship_decision` needs
        # `cleared_margin and significant`, `run_gepa`'s `candidate_ships`
        # needs `p <= SHIP_ALPHA`, the §4CW seed veto needs
        # `_seed_p <= SHIP_ALPHA`, and `live_check.verdict` REVERTs only on
        # `p_worse <= alpha`.
        if p is None or p > ab_eval.SHIP_ALPHA:
            _unmeasurable = True
            print(f"  ⚠ AND THE LOSS IS NOT SIGNIFICANT "
                  f"(McNemar one-sided p={_fmt_p(p)} > "
                  f"{ab_eval.SHIP_ALPHA} on {b + c} discordant pairs). "
                  f"A retirement needs the same bar a promotion does, so "
                  f"this exits 2 (could not measure), NOT 1 (retire it).")
        # ⚠ NOT "restores the hand-written instruction", which this line
        # claimed and which is FALSE for `planning.decompose`: the read
        # site PREPENDS the artifact to a separate production prompt, so
        # retirement removes a prefix and nothing replaces it. The seed
        # scored here is not a prompt production has ever issued.
        print("  ⚠ Retiring removes the PREFIX. Check the read site: if "
              "the artifact is prepended to a production prompt, nothing "
              "replaces it — the seed scored above is a GATE arm, not a "
              "deployed prompt.")
    elif cmp.delta > _margin:
        # ⚠ THIS BRANCH EXISTS BECAUSE §4CY CHANGED WHAT `candidate_ships`
        # MEANS AND THIS INSTRUMENT WAS NOT UPDATED WITH IT. The rule used
        # to be `delta > min_delta`, so `not candidate_ships` implied the
        # margin was missed and the `else` below could say so. Once
        # significance was added, a large-margin-but-underpowered incumbent
        # fell into that `else` and the script printed a literal falsehood:
        #
        #   delta=+0.1290  p=0.219  ->  "NO LONGER CLEARS ITS OWN BAR
        #                                (+0.1290 < 0.02)"
        #
        # Three independent reviewers caught it. An instrument that states
        # an arithmetic impossibility is worse than one that says nothing,
        # because this is the script the operator uses to decide whether to
        # RETIRE a live artifact.
        _unmeasurable = True
        print(f"⚠ THE INCUMBENT'S WIN IS NO LONGER MEASURABLE: it clears "
              f"the margin (delta {cmp.delta:+.4f}, bar {_margin}) but the "
              f"discordant pairs do not support it "
              f"(McNemar p={_fmt_p(p)} > {ab_eval.SHIP_ALPHA} on {b + c} "
              f"pairs, {cmp.candidate_wins} incumbent / "
              f"{cmp.baseline_wins} seed). This is an UNDERPOWERED "
              f"verdict on n={len(scored)}, NOT a measured loss — it is "
              f"not evidence for retiring the artifact, it is evidence "
              f"that this holdout cannot settle the question.")
    else:
        # ⚠ "ITS OWN BAR" MUST BE THE ARTIFACT'S BAR. This compared against
        # `--min-delta` (this script's own default 0.05) while the artifact
        # records the margin it was actually promoted under
        # (`--ab-min-delta`, default 0.02) — and the script PRINTS that 0.02
        # a few lines earlier and then ignored it. A delta of +0.041 clears
        # 0.02 twice over and was reported as failing "its own bar".
        _whose = ("THE BAR IT WAS PROMOTED UNDER" if _from_artifact
                  else f"THE {_margin} BAR")
        print(f"⚠ THE INCUMBENT NO LONGER CLEARS {_whose} "
              f"(delta {cmp.delta:+.4f}, bar {_margin}). It is not measurably "
              f"worse than the baseline, but it is no longer a measured WIN "
              f"— which is what promoted it.")
        # ⚠ AND THIS BRANCH NEEDS POWER TOO. Round 16 gave the LOSS
        # branch an evidence bar and left this one returning 1 — "no
        # longer earns its place" — unconditionally, so with 2 discordant
        # pairs (p=0.25 either way) delta -0.044 exited 2 while +0.044
        # exited 1: the point-estimate-BETTER direction got the harsher
        # code, and which one a wrapper saw was the sign of noise. The
        # claim here is "the win is gone", and that needs a comparison
        # able to RESOLVE the margin — the same `_retire_need` bar the
        # transport guard uses, applied to the pairs that survived.
        if _n_paired < _retire_need:
            _unmeasurable = True
            print(f"  ⚠ BUT ONLY {_n_paired} PAIRS SURVIVED, under the "
                  f"{_retire_need} needed to resolve the {_margin} margin "
                  f"at p<={ab_eval.SHIP_ALPHA} — a delta this small over "
                  f"a tier this thin is the sign of noise, not a "
                  f"vanished win. Exit 2: could not measure.")
    # `p is None` when nothing disagreed — short-circuiting on `b + c`
    # worked, but only by coincidence of evaluation order. Say it.
    # Suppressed in the branch above, which already reports p in context;
    # an unconditional "AND ..." there asserted the margin was missed too.
    if (p is not None and p > ab_eval.SHIP_ALPHA
            and not cmp.delta > _margin):
        print(f"   ⚠ AND THE DIFFERENCE IS NOT SIGNIFICANT (McNemar "
              f"p={p:.3f} > {ab_eval.SHIP_ALPHA} on {b + c} discordant "
              f"pairs). At this tier size "
              f"the pass-rate delta above cannot settle the question on "
              f"its own — read it as a direction, not a verdict.")
    print("\n(nothing was written; this is a measurement)")
    # ⚠ THE VERDICT MUST REACH A SCRIPT, NOT ONLY A READER. All four
    # branches returned 0, so "the incumbent still earns its place" and
    # "the incumbent is now WORSE than the baseline" were the same exit
    # code — the collision §4DA round 5 carved out exit 3 for
    # ("zero here is the same code as re-scored and it still wins") and
    # left in the win/loss pair. 0 = still wins, 1 = no longer wins,
    # 2 = the instrument could not measure, 3 = reported, not re-scored.
    # ⚠ AND 2 WHEN IT SAYS SO ITSELF. The "no longer measurable" branch
    # returned `0 if delta > margin`, i.e. "still wins", about a state it
    # had just called evidence that the holdout cannot settle the
    # question — the collision round 5 carved out a code for, one branch
    # over. 0 = still wins, 1 = no longer wins, 2 = could not measure,
    # 3 = reported, not re-scored.
    if _unmeasurable:
        return 2
    return 0 if cmp.delta > _margin else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
